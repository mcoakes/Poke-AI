"""Pair local controller observations with FireRed's one authoritative simulation."""
from copy import deepcopy


def pair_inspections(reports, ports=None, allow_absent=False):
    if len(reports) != 2:
        raise ValueError("paired inspection requires two emulator reports")
    battles = [r["battle"] for r in reports]
    if any(not b.get("active") or b.get("type_flags", 0) & 3 != 2 for b in battles):
        raise ValueError("paired inspection requires active singles link battles")
    masters = [i for i, b in enumerate(battles) if b["type_flags"] & 4]
    if len(masters) != 1:
        raise ValueError("paired inspection requires exactly one simulation master")
    master = masters[0]
    authority = battles[master]
    if authority.get("outcome_raw") != 0:
        raise ValueError("paired switch inspection requires an ongoing battle")
    source = {m["battler_id"]: m for m in authority["battlers"]}
    if len(source) != 2 or any(not m.get("valid") or (m.get("absent") and not allow_absent) for m in source.values()):
        raise ValueError("simulation master's active battlers are unavailable")
    local_ids = []
    output = deepcopy(reports)
    for i, report in enumerate(output):
        battle = report["battle"]
        local = [m for m in battle["battlers"] if m["side"] == "player" and (allow_absent or not m.get("absent"))]
        if len(local) != 1:
            raise ValueError(f"player {i+1}: local battler identity is unavailable")
        local_ids.append(local[0]["battler_id"])
        if {m["battler_id"] for m in battle["battlers"]} != set(source):
            raise ValueError("players disagree on battler identities")
        battle["raw_battlers"] = deepcopy(battle["battlers"])
        battle["turn_raw"] = battle["turn"]
        battle["turn"] = authority["turn"]
        battle["simulation_source_player"] = master+1
        if ports is not None:
            battle["simulation_source_port"] = ports[master]
        merged = []
        for mon in battle["raw_battlers"]:
            canonical = deepcopy(source[mon["battler_id"]])
            if mon["party_slot"] != canonical["party_slot"]:
                raise ValueError(f"player {i+1}: local party index disagrees with simulation master")
            # Network battler IDs agree; visual player/opponent positions differ.
            canonical.update(side=mon["side"], position=mon.get("position"),
                             data_source="local_simulation" if i == master else "peer_simulation")
            merged.append(canonical)
        battle["battlers"] = merged
    if len(set(local_ids)) != 2:
        raise ValueError("paired reports claim the same local battler")
    return output


def terminal_inspections(snapshot, previous, master, ports):
    """Use the master's latched game outcome; follower outcomes may arrive later."""
    battle = snapshot["battle"]
    raw = battle.get("outcome_raw")
    if raw not in (1, 2, 3) or not battle.get("type_flags", 0) & 4:
        raise ValueError("expected a win, loss or draw latched by the simulation master")
    if not battle.get("battlers"):
        raise ValueError("terminal simulation snapshot is missing battlers")
    for side in ("player", "opponent"):
        counts = battle.get(f"{side}_counts", {})
        if not counts.get("valid"):
            warnings = snapshot.get("warnings") if side == "player" else counts.get("warnings")
            detail = "; ".join(warnings or []) or "counts unavailable"
            raise ValueError(f"terminal party counts are invalid: {side}, "
                             f"frame={snapshot.get('frame')}: {detail}")
    counts = [battle["player_counts"]["remaining"], battle["opponent_counts"]["remaining"]]
    if (raw == 1 and counts[1] != 0) or (raw == 2 and counts[0] != 0) or (raw == 3 and any(counts)):
        raise ValueError("game outcome disagrees with remaining Pokemon")
    result = deepcopy(previous)
    for i, report in enumerate(result):
        own = i == master
        report["party"] = deepcopy(snapshot["party"] if own else battle["opponent_party"])
        report["battle"] = deepcopy(battle)
        b = report["battle"]
        b["opponent_party"] = deepcopy(battle["opponent_party"] if own else snapshot["party"])
        b["menus"] = []
        b["type_flags"] = previous[i]["battle"]["type_flags"]
        b["simulation_authoritative"] = own
        b["simulation_source_port"] = ports[master]
        b["terminal_source_frame"] = snapshot["frame"]
        b["outcome_raw"] = raw if own or raw == 3 else 3-raw
        b["outcome"] = {1:"won", 2:"lost", 3:"draw"}[b["outcome_raw"]]
        b["terminal"] = True
        if not own:
            b["player_counts"], b["opponent_counts"] = b["opponent_counts"], b["player_counts"]
            for mon in b["battlers"]:
                mon["side"] = "opponent" if mon["side"] == "player" else "player"
                mon["position"] ^= 1
        report["remaining"], report["fainted"] = b["player_counts"]["remaining"], b["player_counts"]["fainted"]
        for mon in b["battlers"]:
            # Terminal HP can precede the copy to the stored party.
            party = report["party"] if mon["side"] == "player" else b["opponent_party"]
            if mon.get("valid"):
                party[mon["party_slot"]-1]["current_hp"] = mon["current_hp"]
    return result
