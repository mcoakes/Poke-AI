"""FireRed singles restrictions; separate from ROM addresses and item-use actions."""


def required_switch(report):
    """The game needs a recipient after fainting or a move such as Baton Pass."""
    battle = report["battle"]
    menu = next(iter(battle.get("menus", [])), {})
    if menu.get("state") != "party" or menu.get("submenu") != "choose_pokemon":
        return False
    return bool(menu.get("switch_required") or any(
        mon.get("valid") and mon["side"] == "player" and mon["current_hp"] == 0
        for mon in battle.get("battlers", [])))


def restrictions(battle, player):
    r = player.get("restrictions", {})
    status = player.get("volatile_status_raw", 0)
    enemies = [m for m in battle.get("battlers", []) if m["side"] != player["side"]
               and not m.get("absent") and m.get("current_hp", 0) > 0]
    trapped = bool(status & (0xE000 | 0x4000000) or r.get("status3", 0) & 0x400)
    for enemy in enemies:
        ability = enemy.get("ability_id", 0)
        trapped |= ability == 23 or (ability == 71 and 2 not in player["type_ids"]
                    and player.get("ability_id") != 26) or (ability == 42 and 8 in player["type_ids"])
    forced = bool(status & (0x1000 | 0x400000))
    moves = [False]*4
    for move in player["moves"]:
        mid = move["move_id"]
        blocked = not mid or not move["current_pp"] or mid == r.get("disabled_move", 0)
        blocked |= bool(status & 0x80000000 and mid == r.get("last_move", 0))
        blocked |= bool(r.get("taunt_timer", 0) and move.get("power", 1) == 0)
        blocked |= bool(r.get("encore_timer", 0) and mid != r.get("encored_move", 0))
        choice = r.get("choice_move", 0)
        blocked |= bool(r.get("hold_effect") == 29 and choice not in (0, 65535, mid))
        blocked |= any(e.get("restrictions", {}).get("status3", 0) & 0x2000
                       and any(m["move_id"] == mid for m in e["moves"]) for e in enemies)
        moves[move["slot"]-1] = not blocked
    struggle = not any(moves) and not forced
    if struggle:
        moves[0] = True  # stable action 0 submits Fight; engine supplies Struggle
    if forced:
        moves = [False]*4  # no new decision during recharge/locked turns
    run_override = player.get("ability_id") == 50 or r.get("hold_effect") == 37
    return dict(moves=moves, can_switch=not trapped and not forced,
                can_run=not forced and not battle.get("type_flags", 0) & (8 | 16)
                    and (not trapped or run_override),
                automatic=forced, struggle=struggle,
                skip_move_menu=struggle or bool(r.get("encored_move", 0)))
