"""Frozen replacement interventions preserve all unrelated decisions and inputs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('sb3_contrib')
from firered_rl.evaluation import digest, validate_suite
from firered_rl.fixed_team_lab import pokemon
from firered_rl.replacement_diagnostics import (
    FrozenForcedReplacementPolicy, ReplacementAudit, make_candidate, transform_suite,
)
from firered_rl.staged_team_lab import StageDecisionAudit


ROOT = Path(__file__).resolve().parents[1]


def suite(size=3):
    mons = [('Blastoise', ['Surf', 'Strength']), ('Raichu', ['Thunderbolt', 'Body Slam']),
            ('Golem', ['Earthquake', 'Double Edge']), ('Venusaur', ['Razor Leaf']),
            ('Arcanine', ['Flamethrower']), ('Lapras', ['Ice Beam'])]
    a = dict(team=dict(name='candidate', party=[pokemon(*m) for m in mons[:size]]))
    b = dict(team=dict(name='opponent', party=[pokemon(*m) for m in reversed(mons[:size])]))
    keys = [digest(a), digest(b)]
    return validate_suite(dict(version=1, backend='showdown', seed=18, max_steps=100,
        swap_sides=True, assets=dict(zip(keys, [a, b])), cases=[
            dict(episode=i + 1, matchup=1, assets=keys.copy(), seed=123, candidate_player=i + 1)
            for i in range(2)]))


@pytest.mark.parametrize('size', [1, 2, 3, 6])
def test_candidate_bench_reversal_preserves_lead_opponent_moves_and_rng(size):
    original = suite(size)
    preserved = copy.deepcopy(original)
    reversed_suite = transform_suite(original)
    assert original == preserved
    assert reversed_suite == transform_suite(original)
    assert reversed_suite['swap_sides'] is False
    validate_suite(reversed_suite)
    for old, new in zip(original['cases'], reversed_suite['cases']):
        assert {k: v for k, v in old.items() if k != 'assets'} == {
            k: v for k, v in new.items() if k != 'assets'}
        side = old['candidate_player'] - 1
        assert old['assets'][1-side] == new['assets'][1-side]
        party = original['assets'][old['assets'][side]]['team']['party']
        actual = reversed_suite['assets'][new['assets'][side]]['team']['party']
        assert actual == party[:1] + party[:0:-1]
        assert actual[0] == party[0]
        assert sorted(map(digest, actual)) == sorted(map(digest, party))
    restored = transform_suite(reversed_suite)
    assert restored['cases'] == original['cases']


def test_transform_rejects_tampered_assets_without_touching_input():
    original = suite()
    next(iter(original['assets'].values()))['team']['name'] = 'tampered'
    with pytest.raises(ValueError, match='hash mismatch'):
        transform_suite(original)


def test_shared_case_asset_lists_do_not_transform_both_sides():
    original = suite()
    original['cases'][1]['assets'] = original['cases'][0]['assets']
    assert original['cases'][0]['assets'] is original['cases'][1]['assets']
    changed = transform_suite(original)
    for before, after in zip(original['cases'], changed['cases']):
        side = before['candidate_player'] - 1
        assert after['assets'][1-side] == before['assets'][1-side]
        assert after['assets'][side] != before['assets'][side]


def observation(action=5, *, forced=True, legal=(5, 6)):
    obs = dict(action_mask=np.zeros(11, dtype=np.float32),
        active=np.zeros((2, 10), dtype=np.float32),
        party=np.zeros((6, 10), dtype=np.float32),
        moves=np.zeros((6, 4, 4), dtype=np.float32),
        context=np.array([1, int(forced), 0, 0], dtype=np.float32),
        v2_active=np.zeros((2, 4), dtype=np.float32),
        v2_move_details=np.zeros((2, 6, 4, 6), dtype=np.float32),
        v2_volatile_bits=np.zeros((2, 32), dtype=np.float32),
        status_bits=np.zeros((8, 32), dtype=np.float32),
        base_choice=np.asarray(action, dtype=np.int64))
    obs['action_mask'][list(legal)] = 1
    obs['active'][:, 0] = 1
    obs['active'][:, 1] = [26, 76]
    obs['active'][0, 8] = 1
    obs['v2_active'][:, :2] = 100
    obs['party'][:3, 0] = 1
    obs['party'][:3, 1] = [26, 9, 76]
    return obs


class FakeBase:
    def __init__(self):
        self.calls = []
        self.training = True
        self.state = {'unchanged': True}
        self.policy = SimpleNamespace(set_training_mode=self.set_training_mode)

    def set_training_mode(self, mode):
        self.training = mode

    def predict(self, obs, *, action_masks, deterministic=True, **kwargs):
        assert deterministic is True
        np.testing.assert_array_equal(action_masks, obs['action_mask'].astype(bool))
        self.calls.append(copy.deepcopy(obs))
        return obs['base_choice'].copy(), self.state


def fake_rule(obs, mask=None):
    legal = np.flatnonzero(obs['action_mask'])
    return np.int64(legal[-1]), []


def hybrid(base=None):
    value = FrozenForcedReplacementPolicy(base or FakeBase(), ROOT)
    value.rules.choose = fake_rule
    return value


def test_only_multi_option_forced_replacements_change_in_mixed_batch():
    rows = [observation(), observation(0, forced=False, legal=(0, 1, 5, 6)),
            observation(5, forced=False, legal=(0, 1, 5, 6)),
            observation(6, legal=(6,)), observation(10, forced=False, legal=(10,))]
    batch = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    original = copy.deepcopy(batch)
    candidate = hybrid()
    actions, state = candidate.predict(batch, action_masks=batch['action_mask'].astype(bool))
    assert actions.tolist() == [6, 0, 5, 6, 10]
    assert state is candidate.base_model.state
    assert len(candidate.base_model.calls) == 1
    for key in batch:
        np.testing.assert_array_equal(batch[key], original[key])
    assert not candidate.base_model.training
    assert not hasattr(candidate.policy, 'get_distribution')
    with pytest.raises(ValueError, match='frozen'):
        candidate.policy.set_training_mode(True)


def test_authoritative_masks_and_single_prediction_shape():
    candidate = hybrid()
    obs = observation()
    action, _ = candidate.predict(obs, action_masks=obs['action_mask'].astype(bool))
    assert isinstance(action, np.integer) and action == 6
    bad = obs['action_mask'].astype(bool).copy()
    bad[6] = False
    with pytest.raises(ValueError, match='authoritative'):
        candidate.predict(obs, action_masks=bad)
    with pytest.raises(ValueError, match='deterministic'):
        candidate.predict(obs, action_masks=obs['action_mask'], deterministic=False)
    # Empty party padding never becomes eligible merely because it has a slot.
    assert np.flatnonzero(obs['action_mask']).tolist() == [5, 6]


def test_illegal_base_and_rules_actions_fail_closed():
    candidate = hybrid()
    obs = observation(9)
    with pytest.raises(ValueError, match='base policy selected an illegal'):
        candidate.predict(obs, action_masks=obs['action_mask'])
    obs = observation()
    candidate.rules.choose = lambda obs, mask: (0, [])
    with pytest.raises(ValueError, match='legal forced replacement'):
        candidate.predict(obs, action_masks=obs['action_mask'])


def test_make_candidate_preserves_unassisted_identity():
    base = FakeBase()
    assert make_candidate(base, False, ROOT) is base
    assert make_candidate(base, True, ROOT).base_model is base
    with pytest.raises(ValueError, match='boolean'):
        make_candidate(base, 1, ROOT)


def case(episode):
    return dict(episode=episode, matchup=episode, seed=42, candidate_player=1)


@pytest.mark.parametrize('assisted', [False, True])
def test_audit_counts_all_battles_independently_of_trace_bounds(tmp_path, assisted):
    candidate = hybrid() if assisted else FakeBase()
    audit = ReplacementAudit(ROOT, candidate, trace_cases=1, trace_decisions=1)
    audit.rules.choose = fake_rule
    obs = observation()
    original = copy.deepcopy(obs)
    audit(case(1), 1, obs, 6 if assisted else 5)
    audit(case(2), 5, obs, 6 if assisted else 5)
    same = observation(6)
    audit(case(3), 1, same, 6)
    automatic = observation(6, legal=(6,))
    audit(case(4), 1, automatic, 6)
    for k in obs:
        np.testing.assert_array_equal(obs[k], original[k])
    result = audit.finish(tmp_path)
    counts = result['totals']
    assert counts['replacement_eligible'] == 3
    assert counts['replacement_base_rules_disagreements'] == 2
    assert counts['replacement_base_lowest_legal'] == 2
    assert counts['replacement_actual_lowest_legal'] == (0 if assisted else 2)
    assert counts['replacement_override_eligible'] == (3 if assisted else 0)
    assert counts['replacement_overrides_changed'] == (2 if assisted else 0)
    assert result['by_episode']['2']['replacement_eligible'] == 1
    assert result['traced_decisions'] == 1
    for key, total in counts.items():
        assert sum(row.get(key, 0) for row in result['by_episode'].values()) == total
    metadata = result['replacement_diagnostic']
    assert metadata['assisted'] is assisted
    assert metadata['all_battle_counts'] is True
    assert sum(metadata['actual_species_ids'].values()) == 3
    row = json.loads((tmp_path / 'decision-traces.jsonl').read_text())
    assert row['replacement_diagnostic']['base_proposal']['action'] == 5
    assert row['replacement_diagnostic']['executed']['action'] == (6 if assisted else 5)
    assert row['replacement_diagnostic']['actual_override'] is assisted
    assert json.loads((tmp_path / 'decision-audit.json').read_text()) == result
    # No NumPy scalar escapes nested custom trace metadata.
    json.dumps(result, allow_nan=False)
    json.dumps(row, allow_nan=False)


def test_hybrid_trace_probabilities_are_explicitly_base_policy(monkeypatch):
    audit = ReplacementAudit(ROOT, hybrid())
    monkeypatch.setattr(StageDecisionAudit, 'probabilities',
        lambda self, obs: dict(action_probabilities=np.array([.25, .75]), entropy_nats=np.float32(.5)))
    values = audit.probabilities(observation())
    assert 'action_probabilities' not in values
    assert 'entropy_nats' not in values
    assert 'base_policy_action_probabilities' in values
    assert 'not the executed hybrid' in values['probability_scope']


def test_audit_rejects_mismatched_executed_intervention():
    audit = ReplacementAudit(ROOT, hybrid())
    audit.rules.choose = fake_rule
    with pytest.raises(ValueError, match='observed action differs'):
        audit(case(1), 1, observation(), 5)


def test_audit_supports_baseline_policy_without_ppo_distribution(tmp_path):
    # Actual heuristic classes provide predict, but no PPO policy distribution.
    from firered_rl.rule_opponent import RuleBasedOpponent
    baseline = RuleBasedOpponent(ROOT)
    baseline.choose = fake_rule
    audit = ReplacementAudit(ROOT, baseline, trace_cases=1)
    audit.rules.choose = fake_rule
    audit(case(1), 1, observation(), 6)
    result = audit.finish(tmp_path)
    assert result['totals']['replacement_base_rules_disagreements'] == 0
    assert 'action_probabilities' not in audit.rows[0]
