from copy import deepcopy

from firered_rl.link_test import connection_errors


def pair():
    base = dict(wireless=False, error_occurred=False, error_bits=0,
                connection_established=True, player_count=2, received_remote_players=True,
                link_type=0x2233, link_battle=False,
                players=[dict(id=0, trainer_id=1), dict(id=1, trainer_id=2)])
    return [dict(deepcopy(base), local_id=i) for i in range(2)]


def test_two_reachable_ports_are_not_enough():
    p = pair()
    p[1]["connection_established"] = False
    assert connection_errors(p)


def test_handshake_and_battle_are_separate_gates():
    p = pair()
    assert not connection_errors(p)
    assert connection_errors(p, require_battle=True)
    for report in p:
        report["link_battle"] = True
    assert not connection_errors(p, require_battle=True)


def test_mismatch_error_and_duplicate_id_fail():
    for change in [dict(local_id=0), dict(error_bits=0x1000), dict(wireless=True),
                   dict(players=[]), dict(link_type=0x1122), dict(player_count=1)]:
        p = pair()
        p[1].update(change)
        assert connection_errors(p)
