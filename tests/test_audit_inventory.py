"""Audit checks verify the production pipeline, not just inventory labels."""
from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from firered_rl.audit_inventory import inventory_checks, observation_inventory
from firered_rl.showdown_env import showdown_observation


def training_modules():
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from firered_rl.ppo_features import BattleFeatures, policy_observation, policy_space
    return torch, BattleFeatures, policy_observation, policy_space


def known_report():
    """Independent fixture: known fractions, IDs and high/low status bits."""
    own = dict(slot=1, valid=True, species_id=6, level=40, current_hp=50, max_hp=200,
               base_type_ids=[10, 2], status_raw=(1 << 30) | 3, held_item_id=200,
               is_egg=False, moves=[dict(slot=1, move_id=53, type_id=10,
                                        base_type_id=10, current_pp=7, max_pp=15)])
    enemy = dict(deepcopy(own), species_id=9, current_hp=99, max_hp=150,
                 status_raw=64, held_item_id=0)
    player_active = dict(deepcopy(own), side="player", party_slot=1, type_ids=[10, 2],
                         current_hp=75, volatile_status_raw=(1 << 31) | 1, absent=False)
    opponent_active = dict(deepcopy(enemy), side="opponent", party_slot=1,
                           type_ids=[11, 11], volatile_status_raw=(1 << 24), absent=False)
    return dict(party=[own], bag={}, action_mask=[1]+[0]*10,
                battle=dict(turn=5, type_flags=10, outcome_raw=0,
                            opponent_party=[enemy], battlers=[player_active, opponent_active],
                            menus=[dict(state="command")]))


def test_inventory_is_serializable_and_does_not_import_training_dependencies():
    rows = observation_inventory()
    assert len({row["name"] for row in rows}) == len(rows)
    assert all(set(row) == {"name", "status", "source", "policy_representation", "limitation"}
               for row in rows)
    assert all(row["status"] in {"present", "partial", "missing"} for row in rows)
    assert json.loads(json.dumps(rows)) == rows
    code = ("import sys; from firered_rl.audit_inventory import observation_inventory; "
            "assert observation_inventory(); assert 'torch' not in sys.modules; "
            "assert 'stable_baselines3' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_known_values_reach_flat_features_and_bits_survive_float32():
    torch, BattleFeatures, policy_observation, policy_space = training_modules()
    raw = showdown_observation(known_report())
    assert raw["party"][0].tolist() == [1, 6, 40, 50, 200, 10, 2, (1 << 30) | 3, 200, 0]
    assert raw["active"][1, 3:5].tolist() == [66, 100]
    assert raw["moves"][0, 0].tolist() == [53, 10, 7, 15]
    assert not raw["moves"][0, 1:].any()
    policy = policy_observation(raw)
    assert policy["status_bits"][0, [0, 1, 30]].tolist() == [1, 1, 1]
    assert policy["status_bits"][7, 6] == 1
    assert policy["volatile_bits"][0, [0, 31]].tolist() == [1, 1]
    assert not policy["volatile_bits"][1].any()
    assert not policy["party"][:, 7].any()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(47)
        extractor = BattleFeatures(policy_space(), features_dim=4)
    with torch.no_grad():
        flat = extractor._flatten({key: torch.tensor(value)[None] for key, value in policy.items()})
    # A party record is species(12), numeric(4), types(8), item(4), egg(1).
    np.testing.assert_allclose(flat[0, 12:16].numpy(), [1, .4, .25, 200 / 1024])
    # Six party records precede the first active record.
    np.testing.assert_allclose(flat[0, 6 * 29 + 12:6 * 29 + 16].numpy(),
                               [1, .4, .375, 200 / 1024])
    np.testing.assert_array_equal(flat[0, -320:-64].numpy(), policy["status_bits"].flatten())
    np.testing.assert_array_equal(flat[0, -64:].numpy(), policy["volatile_bits"].flatten())


def test_checks_distinguish_missing_information_from_stale_moves():
    training_modules()
    checks = inventory_checks()
    rows = {row["name"]: row for row in checks}
    assert json.loads(json.dumps(checks)) == checks
    for name in ("explicit_move_power", "explicit_move_accuracy", "explicit_move_category",
                 "stats", "stat_stages", "boosts", "weather", "ability_id",
                 "opponent_bench_hp", "opponent_bench_major_status", "opponent_held_items",
                 "opponent_volatiles"):
        assert rows[name]["status"] == "limitation"
        assert rows[name]["details"]["changed_feature_values"] == 0
    for name in ("own_bench_hp", "own_held_items", "own_active_hp", "own_volatiles",
                 "own_party_major_status", "opponent_active_major_status",
                 "major_status_low_bit_reaches_features", "volatile_status_low_bit_reaches_features",
                 "all_moves_opponent_party_bench", "opponent_active_removed_moves_zeroed"):
        assert rows[name]["status"] == "pass"
        assert rows[name]["details"]["changed_feature_values"] > 0
    assert rows["own_active_removed_moves_zeroed"]["status"] == "pass"
    assert rows["own_active_removed_moves_zeroed"]["details"]["observation_changed_keys"] == ["action_mask", "moves"]
    assert rows["own_active_removed_moves_zeroed"]["details"]["value_assertion_passed"] is True
    assert rows["showdown_authoritative_action_mask"]["details"]["encoder_changed_keys"] == []


def test_checks_are_deterministic_and_restore_caller_rng():
    torch, *_ = training_modules()
    before = torch.get_rng_state().clone()
    first = inventory_checks()
    assert torch.equal(torch.get_rng_state(), before)
    assert inventory_checks() == first
    assert torch.equal(torch.get_rng_state(), before)


def test_checks_detect_information_dropped_by_real_preprocessor(monkeypatch):
    _, _, original, _ = training_modules()
    from firered_rl import ppo_features

    def lose_hp(raw):
        policy = original(raw)
        policy["party"][:, 3] = 0
        policy["active"][:, 3] = 0
        return policy

    monkeypatch.setattr(ppo_features, "policy_observation", lose_hp)
    rows = {row["name"]: row for row in inventory_checks()}
    assert rows["own_bench_hp"]["status"] == "fail"
    assert rows["own_active_hp"]["status"] == "fail"
    assert rows["own_held_items"]["status"] == "pass"
    assert rows["own_bench_hp"]["details"]["observation_changed_keys"] == ["party"]
    assert rows["own_bench_hp"]["details"]["policy_changed_keys"] == []


def test_checks_detect_status_bits_dropped_by_feature_extractor(monkeypatch):
    _, BattleFeatures, *_ = training_modules()
    original = BattleFeatures._flatten

    def lose_status(self, observations):
        changed = dict(observations)
        changed["status_bits"] = observations["status_bits"] * 0
        return original(self, changed)

    monkeypatch.setattr(BattleFeatures, "_flatten", lose_status)
    rows = {row["name"]: row for row in inventory_checks()}
    assert rows["own_party_major_status"]["status"] == "fail"
    assert rows["opponent_active_major_status"]["status"] == "fail"
    assert rows["major_status_low_bit_reaches_features"]["status"] == "fail"
    assert rows["own_volatiles"]["status"] == "pass"
