"""Read-only evidence checks for a two-player cable session (not an RL scheduler)."""


def connection_errors(reports, require_battle=False):
    errors = []
    if len(reports) != 2:
        return ["expected two reports"]
    for i, r in enumerate(reports, 1):
        if r.get("wireless"):
            errors.append(f"player {i}: wireless mode, expected cable")
        if r.get("error_occurred") or r.get("error_bits"):
            errors.append(f"player {i}: link error")
        if not r.get("connection_established") or r.get("player_count") != 2:
            errors.append(f"player {i}: two-player connection not established")
        if not r.get("received_remote_players") or len(r.get("players", [])) != 2:
            errors.append(f"player {i}: player-data exchange incomplete")
        if require_battle and not r.get("link_battle"):
            errors.append(f"player {i}: not inside a link battle")
    if {r.get("local_id") for r in reports} != {0, 1}:
        errors.append("expected distinct cable player IDs 0 and 1")
    if reports[0].get("players") != reports[1].get("players"):
        errors.append("the games disagree on exchanged player records")
    if reports[0].get("link_type") != reports[1].get("link_type"):
        errors.append("the games selected different link activities")
    return errors
