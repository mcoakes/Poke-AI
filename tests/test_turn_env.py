from copy import deepcopy
import numpy as np
import pytest

from firered_rl.backends.mgba_battle import MGBABattleBackend
from firered_rl.turn_env import FireRedTurnEnv, encode, observation_space


def report():
    move = dict(slot=1, move_id=57, type_id=11, base_type_id=11, current_pp=15, max_pp=15)
    player = dict(slot=1, occupied=True, valid=True, species_id=9, level=50,
        current_hp=100, max_hp=100, base_type_ids=[11, 11], status_raw=0,
        held_item_id=0, is_egg=False, moves=[move])
    enemy = dict(player, species_id=13, current_hp=20, max_hp=20)
    def active(mon, side):
        return dict(mon, side=side, party_slot=1, type_ids=mon["base_type_ids"],
                    volatile_status_raw=0, absent=False)
    return dict(frame=1, party=[player], bag={}, battle=dict(active=True, turn=0,
        outcome="ongoing", outcome_raw=0, type_flags=8, opponent_party=[enemy],
        player_counts=dict(valid=True, fainted=0), opponent_counts=dict(valid=True, fainted=0),
        battlers=[active(player, "player"), active(enemy, "opponent")],
        menus=[dict(state="command", cursor_raw=0, move_count=4)]))


def test_observation_shapes_mask_and_full_opponent_data():
    r = report()
    obs = encode(r)
    assert observation_space().contains(obs)
    assert list(obs["action_mask"]) == [1]+[0]*10
    other = deepcopy(r)
    other["battle"]["opponent_party"][0]["species_id"] = 150
    other["battle"]["battlers"][1]["moves"][0]["move_id"] = 354
    # The report fixture shares move dicts; restore player's moves after the change.
    other["party"] = deepcopy(r["party"])
    other["battle"]["battlers"][0]["moves"] = deepcopy(r["party"][0]["moves"])
    assert np.array_equal(obs["moves"], encode(other)["moves"])
    assert encode(other)["opponent_party"][0, 1] == 150
    assert encode(other)["opponent_moves"][0, 0, 0] == 354


@pytest.mark.parametrize("side", ["player", "opponent"])
@pytest.mark.parametrize("slots", [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 3), ()])
def test_active_moves_replace_entire_row_and_preserve_bench(side, slots):
    r = report()
    party = []
    for slot in range(1, 7):
        mon = deepcopy(r["party"][0])
        mon["slot"] = slot
        mon["moves"] = [dict(slot=i+1, move_id=move_id, type_id=11,
                             base_type_id=11, current_pp=10, max_pp=15)
                        for i, move_id in enumerate((57, 58, 44, 182))]
        party.append(mon)
    r["party"] = deepcopy(party)
    r["battle"]["opponent_party"] = deepcopy(party)
    active = r["battle"]["battlers"][0 if side == "player" else 1]
    active["party_slot"] = 6
    active["moves"] = [dict(slot=slot, move_id=150+slot, base_type_id=0,
                             current_pp=5, max_pp=5) for slot in slots]
    original = deepcopy(r)
    obs = encode(r)
    key = "moves" if side == "player" else "opponent_moves"
    expected = np.zeros((4, 4), dtype=np.int64)
    for slot in slots:
        expected[slot-1] = [150+slot, 0, 5, 5]
    np.testing.assert_array_equal(obs[key][5], expected)
    expected_bench = [[move_id, 11, 10, 15] for move_id in (57, 58, 44, 182)]
    for index in range(5):
        np.testing.assert_array_equal(obs[key][index], expected_bench)
    assert observation_space().contains(obs)
    assert r == original


def test_three_move_grid_routes_around_empty_fourth_slot():
    env = FireRedTurnEnv.__new__(FireRedTurnEnv)
    env.report = report()
    m = env.report["battle"]["menus"][0]
    m.update(state="move", cursor_raw=2, move_count=3)
    keys = []
    def tap(key):
        keys.append(key)
        if key == 3:
            m["cursor_raw"] ^= 2
        elif key == 6:
            m["cursor_raw"] ^= 1
        else:
            pytest.fail(f"unexpected routing input: {key}")
        assert m["cursor_raw"] < 3
    env._tap = tap
    env._grid(1)
    assert keys == [3, 6] and m["cursor_raw"] == 1


def test_step_waits_for_next_command_and_returns_one_reward():
    env = FireRedTurnEnv.__new__(FireRedTurnEnv)
    import gymnasium as gym
    env.action_space = gym.spaces.Discrete(11)
    env.backend = MGBABattleBackend()
    env.report = report()
    env.state = env.backend._convert({"inspection": env.report}, resetting=True)
    env.backend.previous = env.state
    env.done, env.steps, env.max_steps, env.timeout = False, 0, 20, 10
    pauses = []
    env._pause = lambda: pauses.append(True)
    env._resume = lambda: None
    def tap(key):
        env.report = deepcopy(env.report)
        m = env.report["battle"]["menus"][0]
        if m["state"] == "command":
            m["state"] = "move"
        elif m["state"] == "move":
            m["state"] = "busy_or_unknown"
        else:
            m["state"] = "command"
            env.report["battle"]["turn"] = 1
            env.report["battle"]["battlers"][1]["current_hp"] = 10
        env.report["frame"] += 5
    env._tap = tap
    env._advance_text = lambda: tap(1)
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 10 and not terminated and not truncated
    assert info["turn_start"] == 0 and info["turn_end"] == 1
    assert pauses == [True] and obs["context"][0] == 1


def test_text_release_reaching_command_does_not_press_a():
    env = FireRedTurnEnv.__new__(FireRedTurnEnv)
    env.report = report()
    env.report["battle"]["menus"][0]["state"] = "busy_or_unknown"
    calls = []
    def poll(key=0, frames=1, guard_menu=False):
        calls.append((key, frames, guard_menu))
        env.report["battle"]["menus"][0]["state"] = "command"
    env._poll = poll
    env._advance_text()
    assert calls == [(0, 2, False)]


def test_switch_back_waits_for_cursor_release_and_confirms_slot_zero():
    env = FireRedTurnEnv.__new__(FireRedTurnEnv)
    env.report = report()
    m = env.report["battle"]["menus"][0]
    m.update(state="party", submenu="choose_pokemon", cursor_raw=0, party_slot=2)
    pending = []
    confirmations = []
    def tap(key):
        assert key == 4
        pending.append(0)  # cursor update is visible only after key release
    def poll(key=0, frames=1):
        if pending:
            m["party_slot"] = pending.pop()+1
            m["cursor_raw"] = 1
    def confirm(target, submenu):
        assert m["party_slot"] == target+1 == 1
        confirmations.append((target, submenu))
        m.update(submenu="pokemon_actions", action_id=10)
    env._tap, env._poll, env._party_confirm = tap, poll, confirm
    env._select_party(0)
    assert confirmations == [(0, "choose_pokemon"), (0, "pokemon_actions")]


def test_move_cursor_fade_waits_without_confirming_or_repeating_direction():
    env = FireRedTurnEnv.__new__(FireRedTurnEnv)
    env.report = report()
    m = env.report["battle"]["menus"][0]
    m.update(state="move", cursor_raw=0)
    taps, polls = [], []
    def tap(key):
        taps.append(key)
        m.update(state="busy_or_unknown")
        m.pop("cursor_raw")
    def poll(key=0, frames=1):
        polls.append(key)
        if len(polls) == 3:
            m.update(state="move", cursor_raw=2)
    env._tap, env._poll = tap, poll
    env._grid(2)
    assert taps == [4] and polls == [0, 0, 0]
