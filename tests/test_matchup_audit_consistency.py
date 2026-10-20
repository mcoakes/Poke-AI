"""Independent audit/feature agreement on complete synthetic v2 observations.

These tests compare two separate implementations. They do not claim that type
and ability relationships predict damage after switching, accuracy or Protect.
"""
import copy
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
from firered_rl.matchup_move_policy import MatchupRelations
from firered_rl.ppo_features import policy_space
from firered_rl.staged_team_lab import StageDecisionAudit


ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "config/generation/firered_us_v1_0.json").read_text())
CHART = json.loads((ROOT / "config/generation/gen3_type_chart.json").read_text())
TYPES = {name: int(number) for number, name in CHART["types"].items()}
SPACE = policy_space("v2")


def observed(move="Thunderbolt", *, own_types=(13, 13), target_types=(4, 5),
             ability=0, slot=0, foresight=False, frozen=False):
    """Full policy-shape fixture; active types may differ after Transform."""
    obs = {name: np.zeros(space.shape, dtype=np.float32)
           for name, space in SPACE.spaces.items()}
    obs["active"][:, 0] = 1
    obs["active"][:, 1] = [26, 76]
    obs["active"][:, 2] = 50
    obs["active"][:, 3:5] = 100
    obs["active"][0, 5:7] = own_types
    obs["active"][1, 5:7] = target_types
    obs["active"][:, 8] = [slot + 1, 1]
    obs["opponent_active_index"][0] = 1
    obs["v2_active"][:, :2] = 100
    obs["v2_active"][1, 3] = ability
    obs["v2_party"][:, 0, :4] = [1, 50, 100, 100]
    obs["v2_party"][0, slot, :4] = [1, 50, 100, 100]
    # Deliberately stale base typing: matchup features must use the active types.
    obs["v2_party"][:, :, 4:6] = [0, 0]
    obs["v2_party"][1, 0, 7] = ability
    obs["party"][slot, :2] = [1, 26]
    obs["action_mask"][:2] = 1
    obs["action_mask"][5 if slot != 1 else 4] = 1
    obs["v2_volatile_bits"][1, 29] = foresight
    obs["status_bits"][7, 5] = frozen
    obs["v2_status_bits"][1, 0, 5] = frozen
    for index, name in enumerate((move, "Strength")):
        entry = CATALOG["moves"][name]
        kind = TYPES[entry["type"]]
        obs["moves"][slot, index] = [entry["id"], kind, 10, 10]
        obs["v2_move_details"][0, slot, index] = [entry["power"], 100, 0,
            {"physical": 0, "special": 1, "status": 2}[entry["category"]], 0, kind]
    assert SPACE.contains(obs)
    return obs


def tensor_batch(observations, device="cpu"):
    return {name: torch.as_tensor(np.stack([o[name] for o in observations]), device=device)
            for name in observations[0]}


def assert_audit_agreement(audit, obs, values, action=0):
    expected = audit.current_target_matchup(obs, action)
    assert values[0] == int(expected["assessed"])
    if not expected["assessed"]:
        np.testing.assert_array_equal(values, np.zeros(5))
        return
    assert values[2] * 4 == pytest.approx(expected["known_immunity_multiplier"])
    assert values[3] == int(expected["currently_immune"])
    slot = int(obs["active"][0, 8]) - 1
    kind = int(obs["v2_move_details"][0, slot, action, 5])
    pure = np.prod([CHART["multipliers"][kind][t]
                    for t in set(map(int, obs["active"][1, 5:7]))])
    assert values[1] * 4 == pytest.approx(pure)
    expected_stab = 1.5 if kind in obs["active"][0, 5:7] else 1.0
    assert values[4] * 1.5 == pytest.approx(expected_stab)


def test_batched_chart_and_ability_features_agree_with_independent_audit():
    audit = StageDecisionAudit(ROOT, None)
    moves = ("Thunderbolt", "Earthquake", "Surf", "Flamethrower", "Leaf Blade",
             "Strength", "Brick Break", "Shadow Ball", "Pursuit", "Hyper Voice")
    defender_types = [(t, t) for t in range(18)] + [(0, 2), (4, 5), (3, 12), (6, 7), (7, 3)]
    observations = [observed(move, target_types=types, ability=ability, slot=index % 6)
        for index, (move, types, ability) in enumerate(product(moves, defender_types,
                                                               (0, 10, 11, 18, 25, 26, 43)))]
    result = MatchupRelations()(tensor_batch(observations)).detach().numpy()
    assert result.shape == (1610, 4, 5)
    for obs, row in zip(observations, result):
        assert_audit_agreement(audit, obs, row[0])
        assert_audit_agreement(audit, obs, row[1], action=1)
        np.testing.assert_array_equal(row[2:], 0)


@pytest.mark.parametrize("types,move,ability,frozen,foresight", [
    ((7, 7), "Strength", 0, False, True),
    ((7, 3), "Brick Break", 0, False, True),
    ((7, 3), "Brick Break", 25, False, True),
    ((10, 10), "Flamethrower", 18, False, False),
    ((10, 10), "Flamethrower", 18, True, False),
])
def test_foresight_and_frozen_flash_fire_adjust_current_not_pure_chart(
        types, move, ability, frozen, foresight):
    obs = observed(move, target_types=types, ability=ability,
                   frozen=frozen, foresight=foresight)
    values = MatchupRelations()(tensor_batch([obs]))[0, 0].numpy()
    assert_audit_agreement(StageDecisionAudit(ROOT, None), obs, values)
    if foresight:
        assert values[1] == 0  # The separate pure chart column remains unchanged.


def test_dynamic_move_and_attacker_types_override_base_descriptor_for_stab():
    obs = observed("Hidden Power", own_types=(11, 11), target_types=(4, 5), slot=4)
    # Native Hidden Power descriptor supplies the actual type and power; the
    # legacy move table still labels this move Normal with variable power.
    obs["v2_move_details"][0, 4, 0] = [70, 100, 0, 1, 0, 11]
    values = MatchupRelations()(tensor_batch([obs]))[0, 0].numpy()
    assert_audit_agreement(StageDecisionAudit(ROOT, None), obs, values)
    assert values[1] == 1  # Water vs Rock/Ground is 4x, scaled by four.
    assert values[4] == 1  # Current Water typing grants STAB exactly once.


def test_excluded_moves_and_dead_targets_are_unassessed_in_both_implementations():
    observations = [observed(move) for move in ("Struggle", "Future Sight", "Beat Up",
        "Nature Power", "Weather Ball", "Doom Desire", "Counter", "Growl")]
    zero_pp = observed()
    zero_pp["moves"][0, 0, 2] = 0
    unknown = observed()
    unknown["moves"][0, 0, 0] = 499
    dead = observed()
    dead["v2_active"][1, 0] = 0
    observations.extend([zero_pp, unknown, dead])
    result = MatchupRelations()(tensor_batch(observations)).numpy()
    audit = StageDecisionAudit(ROOT, None)
    for obs, row in zip(observations, result):
        assert_audit_agreement(audit, obs, row[0])
        assert row[0, 0] == 0


def test_relations_preserve_observations_action_masks_and_random_state():
    obs = tensor_batch([observed(), observed("Surf", ability=11)])
    snapshot = {k: v.clone() for k, v in obs.items()}
    module = MatchupRelations()
    rng = torch.get_rng_state().clone()
    first = module(obs)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for name, value in obs.items():
        torch.testing.assert_close(value, snapshot[name])
    assert not list(module.parameters())
    assert first.device == obs["active"].device
    assert not first.requires_grad
    # A disabled move can have descriptive features, but remains disabled; the
    # feature helper neither changes nor predicts the separate legal mask.
    changed_mask = copy.copy(obs)
    changed_mask["action_mask"] = obs["action_mask"].clone()
    changed_mask["action_mask"][:, 0] = 0
    torch.testing.assert_close(module(changed_mask), first)
    assert torch.count_nonzero(changed_mask["action_mask"][:, 0]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable in test environment")
def test_relation_buffers_follow_cuda_device():
    observations = [observed("Earthquake", target_types=(0, 2)), observed("Surf", ability=11)]
    expected = MatchupRelations()(tensor_batch(observations))
    result = MatchupRelations().to("cuda")(tensor_batch(observations, "cuda"))
    assert result.device.type == "cuda"
    torch.testing.assert_close(result.cpu(), expected)
