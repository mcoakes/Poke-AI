#!/usr/bin/env python3
"""First-pass, read-only party and bag inspection for a running mGBA adapter."""
import argparse
import json
from pathlib import Path

from firered_rl.backends.mgba_bridge import MGBABridgeBackend
from firered_rl.link_inspection import pair_inspections
from firered_rl.link_test import connection_errors


def decorate(report):
    if report.get("profile") != "firered_us_v1_0":
        raise ValueError("no name table for this profile")
    names = json.loads((Path(__file__).parent / "profiles/firered_us_v1_0_names.json").read_text())

    def name(group, value):
        return names[group].get(str(value), f"Unknown #{value}")

    for mon in report["party"] + report.get("battle", {}).get("opponent_party", []):
        if not mon.get("valid"):
            continue
        mon["species"] = name("species", mon["species_id"])
        mon["held_item"] = name("items", mon["held_item_id"])
        mon["base_types"] = [name("types", t) for t in mon["base_type_ids"]]
        raw = mon["status_raw"]
        mon["status"] = (["Sleep"] if raw & 7 else []) + [
            text for flag, text in [(8, "Poison"), (16, "Burn"), (32, "Freeze"),
                                    (64, "Paralysis"), (128, "Bad poison")] if raw & flag]
        for move in mon["moves"]:
            move["name"] = name("moves", move["move_id"])
            if "type_id" in move:
                move["type"] = name("types", move["type_id"])
    for entries in report["bag"].values():
        for item in entries:
            item["name"] = name("items", item["item_id"])
    battle = report.get("battle", {})
    for mon in battle.get("battlers", []):
        mon["species"] = name("species", mon["species_id"])
        mon["types"] = [name("types", value) for value in mon["type_ids"]]
        mon["held_item"] = name("items", mon["held_item_id"])
        for move in mon["moves"]:
            move["name"] = name("moves", move["move_id"])
            if "base_type_id" in move:
                move["base_type"] = name("types", move["base_type_id"])
    for menu in battle.get("menus", []):
        cursor = menu.get("cursor_raw")
        if menu["state"] == "command" and cursor in range(4):
            menu["selected"] = ["Fight", "Bag", "Pokemon", "Run"][cursor]
        elif menu["state"] == "move" and cursor in range(4):
            menu["selected"] = f"Move slot {cursor + 1}"
        elif menu["state"] == "party":
            cursor = menu.get("party_slot", 0)-1 if cursor in range(6) else cursor
            if cursor in range(6):
                mon = report["party"][cursor]
                menu["selected"] = f"Party slot {cursor + 1}: {mon.get('species', 'Empty/invalid')}"
            elif cursor == 6:
                menu["selected"] = "Cancel"
            else:
                menu["selected"] = f"Unknown party selection ({cursor})"
            actions = {0: "Summary", 1: "Switch", 2: "Cancel", 3: "Item", 4: "Give",
                       5: "Take", 6: "Mail", 7: "Take mail", 8: "Read", 9: "Cancel",
                       10: "Shift", 11: "Send out", 12: "Enter", 13: "No entry",
                       14: "Store", 15: "Register", 16: "Trade", 17: "Trade"}
            if "action_id" in menu:
                menu["selected_action"] = actions.get(menu["action_id"], f"Action #{menu['action_id']}")
        elif menu["state"] == "bag":
            pockets = {0: "items", 1: "key_items", 2: "poke_balls"}
            pocket = pockets.get(menu.get("pocket_raw"))
            index = menu.get("item_index_raw")
            if pocket is not None and index is not None:
                menu["pocket"] = pocket
                entries = report["bag"][pocket]
                if 0 <= index < len(entries):
                    menu["highlighted_item"] = entries[index].copy()
                    item = entries[index]
                    menu["selected"] = f"{item['name']} x{item['quantity']}"
                elif index == len(entries):
                    menu["selected"] = "Cancel/Close bag"
    return report


def display(report):
    print(f"Frame {report['frame']} | {report['profile']} | party/bag inspection")
    print("Party slots are 1-based; types are base types, not active-battle overrides.")
    battle = report.get("battle", {})
    print(f"Battle active: {battle.get('active', False)}")
    if "simulation_source_port" in battle:
        print(f"Paired simulation data: port {battle['simulation_source_port']} (local menus/party retained)")
    elif battle.get("simulation_authoritative") is False:
        print("Link follower: active simulation records are unavailable locally; add --peer-port for paired inspection.")
    print(f"Outcome: {battle.get('outcome', 'unknown')} | raw={battle.get('outcome_raw', '?')} "
          f"| scope={battle.get('outcome_scope', 'unknown')} | terminal={battle.get('terminal', False)}")
    for side in ("player", "opponent"):
        counts = battle.get(f"{side}_counts")
        if counts:
            remaining = counts['remaining'] if counts['valid'] else 'unknown'
            fainted = counts['fainted'] if counts['valid'] else 'unknown'
            print(f"  {side}: remaining={remaining} | fainted={fainted} "
                  f"| counts_valid={counts['valid']}")
            for warning in counts.get('warnings', []):
                print(f"    Warning: {warning}")
    if battle.get("active"):
        print("Active battlers (debug/evaluator data, including unrevealed opponent details):")
        for mon in battle["battlers"]:
            validity = "" if mon["valid"] else " [UNINITIALIZED/INVALID]"
            if mon.get("data_source") == "unavailable_on_link_follower":
                validity = " [SIMULATION DATA ON PEER]"
            absent = " [ABSENT]" if mon["absent"] else ""
            print(f"  {mon['side']} battler {mon['battler_id']}: {mon['species']} "
                  f"party slot {mon['party_slot']} Lv{mon['level']} "
                  f"HP {mon['current_hp']}/{mon['max_hp']} "
                  f"[{'/'.join(dict.fromkeys(mon['types']))}]{absent}{validity}")
            print(f"    Held: {mon['held_item']} | status=0x{mon['status_raw']:08x} "
                  f"volatile=0x{mon['volatile_status_raw']:08x}")
            for move in mon["moves"]:
                if move["move_id"]:
                    print(f"    {move['slot']}: {move['name']} "
                          f"PP {move['current_pp']}/{move.get('max_pp', '?')} "
                          f"(base type: {move.get('base_type', '?')})")
        for menu in battle["menus"]:
            print(f"  Menu for battler {menu['battler_id']}: {menu['state']} "
                  f"| selected={menu.get('selected', '?')} | cursor_raw={menu.get('cursor_raw', '?')} "
                  f"| controller=0x{menu['controller']:08x}")
            if "submenu" in menu:
                print(f"    Screen detail: {menu['submenu']} "
                      f"| action={menu.get('selected_action', 'not decoded/applicable')} "
                      f"| pocket={menu.get('pocket', 'not applicable')}")
    print("Player party:")
    display_party(report["party"])
    if report["counts_valid"]:
        print(f"Remaining: {report['remaining']} | Fainted: {report['fainted']} (excludes Eggs)")
    if "opponent_party" in battle:
        print("\nOpponent party (evaluator/debug data, including unrevealed Pokémon):")
        display_party(battle["opponent_party"])
        print("Party values may lag active-battler values during battle animations.")
    for pocket, entries in report["bag"].items():
        print(f"\n{pocket.replace('_', ' ').title()}:")
        for item in entries:
            print(f"  {item['name']} x{item['quantity']} (ID {item['item_id']}, slot {item['slot']})")
        if not entries:
            print("  Empty")
    for warning in report["warnings"]:
        print("WARNING: " + warning)


def display_party(party):
    for mon in party:
        prefix = f"  {mon['slot']}. "
        if not mon["occupied"]:
            print(prefix + "Empty")
        elif not mon.get("valid"):
            print(prefix + "INVALID RECORD — see warnings")
        else:
            types = "/".join(dict.fromkeys(mon["base_types"]))
            status = ", ".join(mon["status"]) or "Healthy"
            egg = " [Egg]" if mon["is_egg"] else ""
            print(prefix + f"{mon['species']}{egg} Lv{mon['level']} [{types}] "
                  f"HP {mon['current_hp']}/{mon['max_hp']} | {status} | Held: {mon['held_item']}")
            for move in mon["moves"]:
                if move["move_id"]:
                    print(f"     {move['slot']}: {move['name']} [{move.get('type', '?')}] "
                          f"PP {move['current_pp']}/{move.get('max_pp', '?')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55355)
    parser.add_argument("--peer-port", type=int,
                        help="Partner's adapter port for a complete paired link-battle inspection")
    parser.add_argument("--json", action="store_true", help="Print complete named JSON diagnostics")
    args = parser.parse_args()
    bridge = MGBABridgeBackend(args.host, args.port)
    peer = MGBABridgeBackend(args.host, args.peer_port) if args.peer_port is not None else None
    try:
        if peer is not None:
            if args.peer_port == args.port:
                raise ValueError("--peer-port must identify the other emulator")
            statuses = [b.status() for b in (bridge, peer)]
            if (any(not s.get("native_link_session") or not s.get("link_session_id") for s in statuses)
                    or statuses[0]["link_session_id"] != statuses[1]["link_session_id"]
                    or {s.get("link_player") for s in statuses} != {1, 2}):
                raise ValueError("ports must belong to the same native two-player link session")
            errors = connection_errors([b.link_status() for b in (bridge, peer)], require_battle=True)
            if errors:
                raise ValueError("; ".join(errors))
            report = pair_inspections([b.inspect() for b in (bridge, peer)],
                                      [args.port, args.peer_port])[0]
        else:
            report = bridge.inspect()
        report = decorate(report)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            display(report)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"inspect_state: {error}\n")
    finally:
        bridge.close()
        if peer is not None:
            peer.close()


if __name__ == "__main__":
    main()
