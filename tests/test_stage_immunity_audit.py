"""Immunity diagnostics count decisions without altering the selected actions."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
from firered_rl.staged_team_lab import StageDecisionAudit


ROOT = Path(__file__).resolve().parents[1]


def observation(move=85, kind=13, power=95, target_types=(4, 5), ability=0,
                alternative=70, alternative_kind=0, alternative_power=80):
    obs = dict(action_mask=np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
        active=np.zeros((2, 10), dtype=np.float32),
        party=np.zeros((6, 10), dtype=np.float32),
        moves=np.zeros((6, 4, 4), dtype=np.float32),
        context=np.array([1, 0, 0, 0], dtype=np.float32),
        v2_active=np.zeros((2, 4), dtype=np.float32),
        v2_move_details=np.zeros((2, 6, 4, 6), dtype=np.float32),
        v2_volatile_bits=np.zeros((2, 32), dtype=np.float32),
        status_bits=np.zeros((8, 32), dtype=np.float32))
    obs["active"][:, 0] = 1
    obs["active"][:, 1] = [26, 76]
    obs["active"][0, 8] = 1
    obs["active"][1, 5:7] = target_types
    obs["v2_active"][:, :2] = 100
    obs["v2_active"][1, 3] = ability
    obs["party"][1, 0:2] = [1, 9]
    obs["moves"][0, 0] = [move, kind, 10, 10]
    obs["moves"][0, 1] = [alternative, alternative_kind, 10, 10]
    obs["v2_move_details"][0, 0, 0] = [power, 100, 0, 1, 0, kind]
    obs["v2_move_details"][0, 0, 1] = [alternative_power, 100, 0, 0, 0, alternative_kind]
    return obs


def observer(**kwargs):
    audit = StageDecisionAudit(ROOT, None, **kwargs)
    audit.rules.choose = lambda obs: (1, [])
    return audit


def case(episode=1):
    return dict(episode=episode, matchup=episode, seed=17, candidate_player=1)


def test_type_immunity_counts_all_battles_and_legal_alternatives_without_mutation(tmp_path):
    audit = observer(trace_cases=1, trace_decisions=1)
    obs = observation()
    original = copy.deepcopy(obs)
    audit(case(), 1, obs, 0)
    audit(case(2), 1, obs, 0)  # Outside trace bound, still included in counters.
    obs["action_mask"][1] = 0
    audit(case(3), 1, obs, 0)  # A damaging alternative exists but is not legal.
    obs["action_mask"][1] = 1
    for key in original:
        np.testing.assert_array_equal(obs[key], original[key])
    data = audit.finish(tmp_path)
    assert data["totals"]["matchup_classified_move_actions"] == 3
    assert data["totals"]["current_target_immune_move_actions"] == 3
    assert data["totals"]["current_target_type_immune_move_actions"] == 3
    assert data["totals"]["current_target_ability_immune_move_actions"] == 0
    assert data["totals"]["current_target_immune_with_nonimmune_alternative"] == 2
    assert data["by_episode"]["2"]["current_target_immune_move_actions"] == 1
    assert data["traced_decisions"] == 1
    trace = json.loads((tmp_path / "decision-traces.jsonl").read_text())
    assert trace["chosen"]["move_id"] == 85  # Audit never substitutes Strength.
    assert trace["current_target_matchup"]["type_multiplier"] == 0
    assert trace["legal_nonimmune_damaging_alternatives"][0]["move_id"] == 70
    assert data["current_target_immunity_scope"]["version"] == 1


@pytest.mark.parametrize("move,kind,power,ability", [
    (89, 4, 100, 26), (85, 13, 95, 10), (57, 11, 95, 11),
    (53, 10, 95, 18), (304, 0, 90, 43),
])
def test_known_ability_immunity_with_nonimmune_type(move, kind, power, ability):
    audit = observer()
    result = audit.current_target_matchup(observation(move, kind, power, (0, 0), ability), 0)
    assert result["assessed"] and result["currently_immune"]
    assert not result["type_immunity"] and result["known_ability_immunity"]


def test_wonder_guard_uses_combined_type_multiplier_and_allows_super_effective():
    audit = observer()
    obs = observation(move=53, kind=10, target_types=(6, 7), ability=25)
    assert not audit.current_target_matchup(obs, 0)["currently_immune"]
    obs["v2_move_details"][0, 0, 0, 5] = 11
    result = audit.current_target_matchup(obs, 0)
    assert result["type_multiplier"] == 1 and result["known_ability_immunity"]


def test_flash_fire_while_frozen_and_foresight_are_not_false_immunities():
    audit = observer()
    obs = observation(move=53, kind=10, target_types=(10, 10), ability=18)
    obs["status_bits"][7, 5] = 1
    result = audit.current_target_matchup(obs, 0)
    assert result["type_multiplier"] == .5
    assert not result["currently_immune"]
    ghost = observation(move=70, kind=0, target_types=(7, 3))
    assert audit.current_target_matchup(ghost, 0)["type_immunity"]
    ghost["v2_volatile_bits"][1, 29] = 1
    assert not audit.current_target_matchup(ghost, 0)["currently_immune"]


@pytest.mark.parametrize("move,power,category,pp", [
    (165, 50, 0, 10), (248, 80, 1, 10), (251, 10, 0, 10),
    (311, 50, 0, 10), (353, 120, 1, 10), (68, 1, 0, 10),
    (85, 95, 1, 0), (45, 0, 2, 10), (999, 95, 1, 10),
])
def test_special_unknown_status_or_zero_pp_moves_are_unclassified(move, power, category, pp):
    audit = observer()
    obs = observation(move=move, power=power)
    obs["moves"][0, 0, 2] = pp
    obs["v2_move_details"][0, 0, 0, 3] = category
    audit(case(), 1, obs, 0)
    assert audit.counts["matchup_unclassified_move_actions"] == 1
    assert audit.counts["current_target_immune_move_actions"] == 0


def test_excluded_move_is_not_counted_as_nonimmune_alternative():
    audit = observer()
    obs = observation(alternative=165, alternative_power=50)
    audit(case(), 1, obs, 0)
    assert audit.counts["current_target_immune_move_actions"] == 1
    assert audit.counts["current_target_immune_with_nonimmune_alternative"] == 0


def test_resisted_move_switch_and_wait_do_not_add_immunity_counts():
    audit = observer()
    obs = observation(target_types=(13, 13))
    assert audit.current_target_matchup(obs, 0)["type_multiplier"] == .5
    audit(case(), 1, obs, 0)
    audit(case(), 2, obs, 5)
    obs["action_mask"][:] = 0
    obs["action_mask"][10] = 1
    audit(case(), 3, obs, 10)
    assert audit.counts["move_actions"] == 1
    assert audit.counts["matchup_classified_move_actions"] == 1
    assert audit.counts["current_target_immune_move_actions"] == 0
