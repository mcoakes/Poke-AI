"""Finite paired command sequences, with inspector evidence for every switch."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from firered_rl.backends.mgba_bridge import MGBABridgeBackend
from firered_rl.battle_rules import restrictions
from firered_rl.link_test import connection_errors
from firered_rl.link_inspection import pair_inspections


def positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_commands(path, environments):
    plan = json.loads(Path(path).read_text())
    if not isinstance(plan, dict) or set(plan)-{"name", "repeats", "turns_per_repeat", "players"}:
        raise ValueError("invalid command plan fields")
    repeats = positive_int(plan.get("repeats"), "repeats")
    width = positive_int(plan.get("turns_per_repeat"), "turns_per_repeat")
    total = repeats*width
    if total > 100000:
        raise ValueError("command plan exceeds 100000 turns")
    players = plan.get("players")
    if not isinstance(players, list) or len(players) not in (2, environments):
        raise ValueError("commands need two alternating player plans, or one per environment")
    for player in players:
        if not isinstance(player, dict) or set(player)-{"name", "cycle"}:
            raise ValueError("invalid player command fields")
        if not isinstance(player.get("cycle"), list) or not player["cycle"]:
            raise ValueError("each player needs a nonempty command cycle")
        for command in player["cycle"]:
            if not isinstance(command, dict) or set(command) != {"switch"}:
                raise ValueError('only {"switch": slot} commands are currently supported')
            if type(command["switch"]) is not int or not 1 <= command["switch"] <= 6:
                raise ValueError("switch slots must be integers from 1 to 6")
    return dict(name=plan.get("name", Path(path).stem), repeats=repeats,
                turns_per_repeat=width, total_turns=total,
                players=[players[i % len(players)] for i in range(environments)])


def switch_slot(player_plan, index):
    cycle = player_plan["cycle"]
    return cycle[index % len(cycle)]["switch"]


def active_mon(report, side="player"):
    mons = [m for m in report["battle"].get("battlers", [])
            if m.get("side") == side and m.get("valid") and not m.get("absent")]
    if len(mons) != 1:
        raise ValueError(f"requires exactly one valid {side} battler")
    return mons[0]


def check_command_menu(report):
    battle = report["battle"]
    if not battle.get("active") or battle.get("type_flags", 0) & 3 != 2 or battle.get("outcome_raw") != 0:
        raise ValueError("command sequence requires an ongoing singles link battle")
    menus = battle.get("menus", [])
    if len(menus) != 1 or menus[0].get("state") != "command":
        raise ValueError("expected the local battle command menu")
    if not report.get("counts_valid"):
        raise ValueError("party inspection is invalid")
    return active_mon(report)


def party_by_slot(report):
    return {m["slot"]: m for m in report["party"] if m.get("valid") and not m.get("is_egg")}


def preflight(report, player_plan, total):
    current = check_command_menu(report)["party_slot"]
    if report["battle"]["turn"]+total > 255:
        raise ValueError("sequence exceeds FireRed's saturating 255-turn counter; use a shorter run")
    party = party_by_slot(report)
    for turn in range(total):
        target = switch_slot(player_plan, turn)
        if target not in party or party[target]["current_hp"] <= 0:
            raise ValueError(f"command {turn+1}: target slot {target} is unavailable")
        if target == current:
            raise ValueError(f"command {turn+1}: slot {target} would already be active")
        current = target
    return current


def verify_switches(initial, reports, targets, turns_elapsed):
    """Compare fresh inspector data to the plan and the original party identity."""
    errors, checks = [], []
    for i, (before, report, target) in enumerate(zip(initial, reports, targets)):
        mon = check_command_menu(report)
        party = party_by_slot(before)
        expected = party[target]
        expected_turn = before["battle"]["turn"]+turns_elapsed
        if report["battle"]["turn"] != expected_turn:
            errors.append(f"player {i+1}: expected turn {expected_turn}, got {report['battle']['turn']}")
        if mon["party_slot"] != target or mon["species_id"] != expected["species_id"]:
            errors.append(f"player {i+1}: expected slot {target}/species {expected['species_id']}, "
                          f"got {mon['party_slot']}/{mon['species_id']}")
        other = active_mon(report, "opponent")
        peer_expected = party_by_slot(initial[1-i])[targets[1-i]]
        if other["party_slot"] != targets[1-i] or other["species_id"] != peer_expected["species_id"]:
            errors.append(f"player {i+1}: opponent active Pokemon disagrees with partner's command")
        # Switch-only stress test: no attacks should consume HP or PP.
        after_party = party_by_slot(report)
        if set(party) != set(after_party):
            errors.append(f"player {i+1}: party membership changed")
        for slot, old in party.items():
            new = after_party.get(slot, {})
            for field in ("species_id", "current_hp", "max_hp"):
                if new.get(field) != old[field]:
                    errors.append(f"player {i+1} slot {slot}: {field} changed")
            old_pp = [(m["move_id"], m["current_pp"]) for m in old["moves"]]
            new_pp = [(m["move_id"], m["current_pp"]) for m in new.get("moves", [])]
            if old_pp != new_pp:
                errors.append(f"player {i+1} slot {slot}: moves or PP changed")
        checks.append(dict(expected_slot=target, actual_slot=mon["party_slot"],
                           expected_species_id=expected["species_id"], species_id=mon["species_id"],
                           turn=report["battle"]["turn"], frame=report["frame"]))
    if errors:
        raise ValueError("; ".join(errors))
    return checks


def run_pair_commands(ports, session_id, players, plan, directory, timeout=120,
                      cancel=None, announce=print, decorate=None):
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    bridges = [MGBABridgeBackend("127.0.0.1", port) for port in ports]
    last = [None, None]
    completed = 0
    def save_reports(name, reports):
        for i, report in enumerate(reports, 1):
            value = decorate(report) if decorate else report
            (directory / f"{name}-player{i}.json").write_text(json.dumps(value, indent=2)+"\n")
    with ThreadPoolExecutor(max_workers=2) as executor, (directory / "commands.jsonl").open("w") as log:
        def record(**event):
            log.write(json.dumps(event)+"\n")
            log.flush()
        def request(method, payloads=None):
            payloads = payloads or [{}, {}]
            futures = [executor.submit(getattr(b, method), **p) for b, p in zip(bridges, payloads)]
            results, errors = [], []
            for port, future in zip(ports, futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append(None)
                    errors.append(f"port {port} {method}: {error}")
            if errors:
                raise RuntimeError("; ".join(errors))
            return results
        try:
            statuses = request("status")
            if any(not s.get("link_turn_supported") or not s.get("link_peer_simulation_supported")
                   or not s.get("native_link_session")
                   or s.get("link_session_id") != session_id for s in statuses):
                raise RuntimeError("command runner needs adapter 0.11.2 in the assigned native link session")
            if [s.get("link_player") for s in statuses] != [1, 2]:
                raise RuntimeError("adapter players do not match command assignment")
            errors = connection_errors(request("link_status"), require_battle=True)
            if errors:
                raise RuntimeError("; ".join(errors))
            raw = request("inspect")
            save_reports("raw-initial-inspection", raw)
            initial = pair_inspections(raw, ports)
            save_reports("initial-inspection", initial)
            expected_final = [preflight(r, p, plan["total_turns"]) for r, p in zip(initial, players)]
            reports = initial
            record(event="start", ports=list(ports), players=players, total_turns=plan["total_turns"],
                   expected_final=expected_final)
            for index in range(plan["total_turns"]):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("pool command run cancelled")
                targets = [switch_slot(p, index) for p in players]
                for report in reports:
                    mon = check_command_menu(report)
                    if not restrictions(report["battle"], mon)["can_switch"]:
                        raise ValueError("active Pokemon cannot switch at this turn")
                request_id = index+1
                last = request("link_turn_prepare", [dict(request_id=request_id, slot=slot,
                                expected_turn=r["battle"]["turn"]) for slot, r in zip(targets, reports)])
                if any(s.get("phase") != "prepared" or s.get("request_id") != request_id for s in last):
                    raise RuntimeError("both commands must be prepared before committing either")
                request("link_turn_commit", [dict(request_id=request_id)]*2)
                deadline = time.monotonic()+timeout
                phases = None
                while True:
                    if cancel is not None and cancel.is_set():
                        raise RuntimeError("pool command run cancelled")
                    last = request("link_turn_status")
                    now = [s.get("phase") for s in last]
                    if now != phases:
                        record(event="progress", request_id=request_id, reports=last)
                        phases = now
                    if any(s.get("error") or s.get("request_id") != request_id
                           or (not s.get("active") and s.get("phase") != "turn_complete") for s in last):
                        raise RuntimeError(f"paired switch failed: {last}")
                    if all(s.get("phase") == "turn_complete" for s in last):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"paired switch timed out after {timeout:g}s: {last}")
                    if cancel is None:
                        time.sleep(0.02)
                    else:
                        cancel.wait(0.02)
                reports = pair_inspections(request("inspect"), ports)
                completed += 1
                checks = verify_switches(initial, reports, targets, completed)
                record(event="turn_complete", request_id=request_id, targets=targets, checks=checks)
                if completed % plan["turns_per_repeat"] == 0:
                    announce(f"ports {ports[0]}/{ports[1]}: cycle {completed//plan['turns_per_repeat']}/"
                             f"{plan['repeats']} complete; active slots {targets[0]}/{targets[1]}")
            raw = request("inspect")
            save_reports("raw-final-inspection", raw)
            reports = pair_inspections(raw, ports)
            checks = verify_switches(initial, reports, expected_final, completed)
            errors = connection_errors(request("link_status"), require_battle=True)
            if errors:
                raise RuntimeError("; ".join(errors))
            save_reports("final-inspection", reports)
            for check, port, report in zip(checks, ports, reports):
                check.update(port=port, species=active_mon(report).get("species", str(check["species_id"])))
            result = dict(status="commands_passed", turns=completed, repeats=plan["repeats"],
                          ports=list(ports), checks=checks)
            (directory / "commands-result.json").write_text(json.dumps(result, indent=2)+"\n")
            record(event="passed", **result)
            return result
        except BaseException as error:
            record(event="failed", completed_turns=completed, error=str(error), last_reports=last)
            try:
                save_reports("failure-inspection", request("inspect"))
            except Exception:
                pass
            raise
        finally:
            for bridge in bridges:
                try:
                    bridge.link_turn_cancel()
                except (OSError, RuntimeError, ValueError):
                    pass
                bridge.close()
