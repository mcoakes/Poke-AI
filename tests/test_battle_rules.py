from copy import deepcopy

import pytest

from firered_rl.battle_rules import restrictions
from firered_rl.turn_env import encode
from test_turn_env import report


def setup():
    r = report()
    p = r["battle"]["battlers"][0]
    p["moves"] = [dict(slot=i+1, move_id=i+10, current_pp=10, power=50 if i else 0)
                  for i in range(4)]
    p["restrictions"] = {}
    return r, p


@pytest.mark.parametrize("fields,status,expected", [
    ({"disabled_move": 11}, 0, [True, False, True, True]),
    ({"encored_move": 12, "encore_timer": 2}, 0, [False, False, True, False]),
    ({"taunt_timer": 2}, 0, [False, True, True, True]),
    ({"last_move": 13}, 0x80000000, [True, True, True, False]),
    ({"choice_move": 11, "hold_effect": 29}, 0, [False, True, False, False]),
    ({"locked_move": 12}, 0x1000, [False]*4),
    ({}, 0x400000, [False]*4),
])
def test_move_restrictions(fields, status, expected):
    r, p = setup()
    p["restrictions"].update(fields)
    p["volatile_status_raw"] = status
    assert restrictions(r["battle"], p)["moves"] == expected


@pytest.mark.parametrize("status,status3,ability,types,player_ability,trapped", [
    (0x2000, 0, 0, [0, 0], 0, True),
    (0x4000000, 0, 0, [0, 0], 0, True),
    (0, 0x400, 0, [0, 0], 0, True),
    (0, 0, 23, [0, 0], 0, True),
    (0, 0, 71, [0, 0], 0, True),
    (0, 0, 71, [2, 0], 0, False),
    (0, 0, 71, [0, 0], 26, False),
    (0, 0, 42, [8, 0], 0, True),
    (0, 0, 42, [0, 0], 0, False),
])
def test_trapping(status, status3, ability, types, player_ability, trapped):
    r, p = setup()
    p.update(volatile_status_raw=status, type_ids=types, ability_id=player_ability)
    p["restrictions"]["status3"] = status3
    r["battle"]["battlers"][1]["ability_id"] = ability
    assert restrictions(r["battle"], p)["can_switch"] != trapped


def test_struggle_encore_and_forced_replacement():
    r, p = setup()
    for move in p["moves"]:
        move["current_pp"] = 0
    rules = restrictions(r["battle"], p)
    assert rules["struggle"] and rules["skip_move_menu"] and rules["moves"] == [True, False, False, False]
    p["restrictions"].update(encored_move=12, encore_timer=2)
    assert restrictions(r["battle"], p)["skip_move_menu"]
    p["current_hp"] = 0
    p["volatile_status_raw"] = 0x4000000
    r["party"].append(dict(deepcopy(r["party"][0]), slot=2))
    r["battle"]["menus"][0].update(state="party", submenu="choose_pokemon")
    assert encode(r)["action_mask"][5] == 1


def test_baton_pass_requires_only_living_recipients_even_when_trapped():
    r, p = setup()
    p["volatile_status_raw"] = 0x2000  # partial trapping, e.g. Sand Tomb
    for slot, fields in [(2, {}), (3, {"current_hp": 0}),
                         (4, {"is_egg": True}), (5, {"valid": False})]:
        r["party"].append(dict(deepcopy(r["party"][0]), slot=slot, **fields))
    assert not restrictions(r["battle"], p)["can_switch"]
    r["battle"]["menus"][0].update(state="party", submenu="choose_pokemon", switch_required=True)
    obs = encode(r)
    assert p["current_hp"] > 0
    assert obs["context"][1] == 1
    assert obs["action_mask"].tolist() == [0]*5 + [1] + [0]*5
    # An ordinary party screen cannot bypass the trapping restrictions.
    r["battle"]["menus"][0]["switch_required"] = False
    assert encode(r)["context"][1] == 0 and not encode(r)["action_mask"][4:10].any()


def test_full_opponent_moves_available_before_any_reveal():
    r = report()
    r["battle"]["revealed_opponents"] = [dict(species_id=13, moves=[dict(move_id=33, type_id=0)])]
    r["battle"]["revealed_active_index"] = 1
    obs = encode(r)
    assert obs["opponent_moves"][0, 0].tolist() == [57, 11, 15, 15]
    assert not obs["opponent_moves"][0, 1:].any()
    assert "opponent_revealed_moves" not in obs
    bench = deepcopy(r["battle"]["opponent_party"][0])
    bench["slot"] = 2
    bench["moves"][0]["current_pp"] = 7
    r["battle"]["opponent_party"].append(bench)
    r["battle"]["battlers"][1]["moves"] = [dict(bench["moves"][0], current_pp=3)]
    obs = encode(r)
    assert obs["opponent_moves"][0, 0, 2] == 3
    assert obs["opponent_moves"][1, 0, 2] == 7


def test_imprison_and_escape_override():
    r, p = setup()
    enemy = r["battle"]["battlers"][1]
    enemy.update(restrictions={"status3": 0x2000}, moves=[{"move_id": 11}])
    assert restrictions(r["battle"], p)["moves"] == [True, False, True, True]
    r["battle"]["type_flags"] = 0
    p["volatile_status_raw"] = 0x4000000
    p["ability_id"] = 50
    rules = restrictions(r["battle"], p)
    assert rules["can_run"] and not rules["can_switch"]


def test_forced_actions_allow_neither_switch_nor_run():
    r, p = setup()
    r["battle"]["type_flags"] = 0
    p["volatile_status_raw"] = 0x400000
    rules = restrictions(r["battle"], p)
    assert rules["automatic"] and not rules["can_switch"] and not rules["can_run"]
