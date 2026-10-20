from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

from firered_rl.observation_v2 import encode_v2, versioned_raw_space, v2_spaces
from firered_rl.turn_env import observation_space


def native_observation():
    base = {key: np.zeros(space.shape, dtype=space.dtype)
            for key, space in observation_space().spaces.items()}
    base["party"][0] = [1, 25, 50, 1, 3, 13, 13, 0, 1, 0]
    base["active"][:] = [[1, 25, 50, 1, 3, 13, 13, 0, 1, 0],
                         [1, 25, 50, 33, 100, 13, 13, 0, 0, 0]]
    base["opponent_party"][0] = [1, 25]
    base["opponent_active_index"][:] = 1
    for name in ("moves", "opponent_moves"):
        base[name][0, 0] = [85, 13, 10, 15]
    base["action_mask"][0] = 1
    native = {key: np.zeros(space.shape, dtype=space.dtype)
              for key, space in v2_spaces().items()}
    native["v2_party"][:, 0] = [1, 50, 1, 3, 13, 13, 1, 9, 0, 0]
    native["v2_stats"][:, 0] = [81, 63, 75, 92, 123]
    native["v2_active"][:] = [1, 3, 1, 9]
    native["v2_active_stats"][:] = [90, 75, 92, 80, 121]
    native["v2_move_details"][:, 0, 0] = [95, 100, 0, 1, 0, 13]
    native["v2_weather"][:] = [2, 4, 0]
    return base, {"backend": "showdown", "observation_v2": native}


def policy_tools():
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from firered_rl import ppo_features
    return ppo_features


def test_raw_v1_is_frozen_v2_is_native_and_nonmutating():
    base, report = native_observation()
    before = deepcopy(report)
    raw = encode_v2(report, base)
    assert versioned_raw_space() == observation_space()
    assert versioned_raw_space("v2").contains(raw)
    assert raw["v2_volatiles"].dtype == np.uint32
    assert raw["active"][1, 3:5].tolist() == [33, 100]
    assert raw["v2_active"][1, :2].tolist() == [1, 3]
    for key in base:
        np.testing.assert_array_equal(raw[key], base[key])
        assert not np.shares_memory(raw[key], base[key])
    for key in before["observation_v2"]:
        np.testing.assert_array_equal(report["observation_v2"][key], before["observation_v2"][key])
        assert not np.shares_memory(raw[key], report["observation_v2"][key])


def test_per_turn_validation_reuses_private_specs_and_public_spaces_stay_fresh(monkeypatch):
    policy = policy_tools()
    from firered_rl import observation_v2
    base, report = native_observation()
    raw = encode_v2(report, base)
    expected = policy.policy_observation(raw)
    public = versioned_raw_space("v2")
    public["v2_weather"].high[:] = 0
    public["party"].high[:] = 0
    assert versioned_raw_space("v2").contains(raw)

    def rebuilt_space(*args, **kwargs):
        raise AssertionError("schema factory called on the per-turn validation path")

    monkeypatch.setattr(observation_v2, "v2_spaces", rebuilt_space)
    monkeypatch.setattr(observation_v2, "observation_space", rebuilt_space)
    actual = policy.policy_observation(encode_v2(report, base))
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("key,index,value", [
    ("v2_party", (0, 0, 0), 2), ("v2_party", (0, 0, 1), 101),
    ("v2_party", (0, 0, 2), -1), ("v2_party", (0, 0, 4), 32),
    ("v2_party", (0, 0, 6), 512), ("v2_party", (0, 0, 7), 256),
    ("v2_party", (0, 0, 8), 2), ("v2_party", (0, 0, 9), 2**32),
    ("v2_stats", (0, 0, 0), -1), ("v2_stats", (0, 0, 0), 65536),
    ("v2_active_stats", (0, 0), np.nan), ("v2_active", (0, 0), np.inf),
    ("v2_boosts", (0, 0), -7), ("v2_boosts", (0, 0), 7),
    ("v2_boosts", (0, 0), 1.5), ("v2_move_details", (0, 0, 0, 1), 101),
    ("v2_move_details", (0, 0, 0, 2), 2), ("v2_move_details", (0, 0, 0, 3), 3),
    ("v2_move_details", (0, 0, 0, 4), -8), ("v2_move_details", (0, 0, 0, 4), 8),
    ("v2_move_details", (0, 0, 0, 5), 32),
    ("v2_weather", (0,), 5), ("v2_weather", (1,), -1),
    ("v2_weather", (2,), 2), ("v2_volatiles", (0,), -1),
    ("v2_volatiles", (1,), 2**32),
])
def test_native_ranges_checked_before_casting(key, index, value):
    base, report = native_observation()
    field = report["observation_v2"][key].astype(np.float64)
    field[index] = value
    report["observation_v2"][key] = field
    with pytest.raises(ValueError, match=key):
        encode_v2(report, base)


def test_native_report_requires_backend_all_fields_and_exact_shapes():
    base, report = native_observation()
    for backend in (None, "mgba"):
        with pytest.raises(ValueError):
            encode_v2({**report, "backend": backend}, base)
    with pytest.raises(ValueError, match="missing native"):
        encode_v2({"backend": "showdown"}, base)
    for key in report["observation_v2"]:
        missing = deepcopy(report)
        del missing["observation_v2"][key]
        with pytest.raises(ValueError, match="missing"):
            encode_v2(missing, base)
        wrong = deepcopy(report)
        wrong["observation_v2"][key] = wrong["observation_v2"][key].reshape(-1, 1)
        with pytest.raises(ValueError, match="shape"):
            encode_v2(wrong, base)
    report["observation_v2"]["v2_guess"] = np.zeros(1)
    with pytest.raises(ValueError, match="extra"):
        encode_v2(report, base)


def test_signed_bounds_and_all_bits_survive_preprocessing():
    policy = policy_tools()
    base, report = native_observation()
    base["party"][0, 7] = base["active"][0, 9] = (1 << 31) | 1
    native = report["observation_v2"]
    native["v2_party"][:, 0, 9] = (1 << 31) | 3
    native["v2_volatiles"][:] = [(1 << 31) | 1, (1 << 30) | 5]
    native["v2_boosts"][:, :2] = [-6, 6]
    native["v2_move_details"][:, 0, 0, 4] = [-7, 7]
    raw = encode_v2(report, base)
    before = deepcopy(raw)
    obs = policy.policy_observation(raw)
    assert policy.policy_space("v2").contains(obs)
    assert obs["v2_status_bits"][:, 0, [0, 1, 31]].tolist() == [[1, 1, 1], [1, 1, 1]]
    assert obs["v2_volatile_bits"][0, [0, 31]].tolist() == [1, 1]
    assert obs["v2_volatile_bits"][1, [0, 2, 30]].tolist() == [1, 1, 1]
    assert not obs["v2_party"][..., 9].any()
    assert not obs["v2_volatiles"].any()
    for key in raw:
        np.testing.assert_array_equal(raw[key], before[key])
    projected = policy.policy_for_version(obs, "v1")
    assert policy.policy_space().contains(projected)
    direct = policy.policy_observation(base)
    explicit = policy.policy_observation(raw, "v1")
    for key in direct:
        np.testing.assert_array_equal(projected[key], direct[key])
        np.testing.assert_array_equal(explicit[key], direct[key])
        assert projected[key] is obs[key]
    batched = {key: np.stack([value, value]) for key, value in obs.items()}
    batched_v1 = policy.policy_for_version(batched, "v1")
    assert batched_v1["status_bits"] is batched["status_bits"]


def test_checkpoint_version_detection_and_rejected_upgrades():
    policy = policy_tools()
    base, report = native_observation()
    v1 = policy.policy_observation(base)
    for version in ("v1", "v2"):
        assert policy.observation_version(policy.policy_space(version)) == version
    with pytest.raises(ValueError, match="cannot upgrade"):
        policy.policy_for_version(v1, "v2")
    with pytest.raises(ValueError, match="missing"):
        policy.policy_observation(base, "v2")
    malformed = deepcopy(policy.policy_space("v2"))
    del malformed.spaces["v2_stats"]
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        policy.observation_version(malformed)
    malformed = deepcopy(policy.policy_space())
    malformed.spaces["context"] = gym.spaces.Box(0, 2, (4,), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        policy.observation_version(malformed)
    malformed = deepcopy(policy.policy_space("v2"))
    malformed["v2_weather"].high[0] = np.nextafter(np.float32(4), np.float32(5))
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        policy.observation_version(malformed)
    for invalid in ("v3", "", 2):
        with pytest.raises(ValueError, match="unsupported observation version"):
            policy.policy_observation(base, invalid)
        with pytest.raises(ValueError, match="unsupported observation version"):
            versioned_raw_space(invalid)
    raw = encode_v2(report, base)
    del raw["v2_weather"]
    with pytest.raises(ValueError, match="missing"):
        policy.policy_observation(raw)


def test_v1_network_parameter_names_and_shapes_remain_frozen():
    policy = policy_tools()
    model = policy.BattleFeatures(policy.policy_space())
    assert {name: tuple(value.shape) for name, value in model.state_dict().items()} == {
        "species.weight": (512, 12), "move.weight": (512, 16),
        "item.weight": (512, 4), "types.weight": (32, 4),
        "network.0.weight": (256, 3144), "network.0.bias": (256,),
        "network.2.weight": (128, 256), "network.2.bias": (128,),
    }


def flatten(model, raw):
    import torch
    from firered_rl.ppo_features import policy_observation
    obs = {key: torch.from_numpy(value[None]) for key, value in policy_observation(raw).items()}
    with torch.no_grad():
        return model._flatten(obs)


@pytest.mark.parametrize("key,indices", [
    ("v2_party", [(1, 0, column) for column in range(10)]),
    ("v2_stats", [(1, 0, column) for column in range(5)]),
    ("v2_active", [(1, column) for column in range(4)]),
    ("v2_active_stats", [(1, column) for column in range(5)]),
    ("v2_boosts", [(1, column) for column in range(7)]),
    ("v2_move_details", [(1, 0, 0, column) for column in range(6)]),
    ("v2_weather", [(column,) for column in range(3)]),
    ("v2_volatiles", [(0,), (1,)]),
])
def test_every_v2_field_reaches_policy_features(key, indices):
    policy = policy_tools()
    import torch
    torch.manual_seed(7)
    model = policy.BattleFeatures(policy.policy_space("v2"))
    base, report = native_observation()
    raw = encode_v2(report, base)
    original = flatten(model, raw)
    for index in indices:
        changed = deepcopy(raw)
        value = changed[key][index]
        changed[key][index] = 0 if value else 1
        assert not torch.equal(flatten(model, changed), original), (key, index)


def test_v2_masks_empty_party_move_and_active_rows():
    policy = policy_tools()
    import torch
    model = policy.BattleFeatures(policy.policy_space("v2"))
    base, report = native_observation()
    base["active"][1] = 0
    raw = encode_v2(report, base)
    original = flatten(model, raw)
    changed = deepcopy(raw)
    changed["v2_party"][1, 5, 1:] = [100, 500, 500, 5, 5, 100, 22, 1, 2**31+1]
    changed["v2_stats"][1, 5] = 999
    changed["v2_move_details"][1, 5, 0] = [120, 75, 0, 2, -3, 3]
    changed["v2_move_details"][1, 0, 3] = [120, 75, 0, 2, -3, 3]
    changed["v2_active"][1] = [200, 500, 22, 5]
    changed["v2_active_stats"][1] = 500
    changed["v2_boosts"][1] = -6
    changed["v2_volatiles"][1] = 2**31+1
    assert torch.equal(flatten(model, changed), original)
