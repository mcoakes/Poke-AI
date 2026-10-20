import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from firered_rl.decision_audit import DecisionAudit
from firered_rl.loss_review import prepare_review, report_review
from test_rule_opponent import observation, move
from firered_rl.rule_opponent import RuleBasedOpponent

ROOT = Path(__file__).resolve().parents[1]


def test_audit_does_not_modify_observation_and_bounds_trace(tmp_path):
    rules = RuleBasedOpponent(ROOT)
    obs = observation(rules)
    move(rules, obs, 0, 'Tackle')
    move(rules, obs, 1, 'Surf')
    before = {k:v.copy() for k,v in obs.items()}
    audit = DecisionAudit(ROOT, tmp_path, SimpleNamespace(log_max_mb=.003, log_backups=1))
    for i in range(30):
        audit(dict(episode=1, candidate_player=1), i+1, obs, 0)
    assert all(np.array_equal(v, obs[k]) for k,v in before.items())
    audit.finish()
    counts = json.loads((tmp_path/'decision-audit.json').read_text())['by_episode']['1']
    assert counts['choices'] == 30 and counts['heuristic_disagreement'] == 30
    traces = list(tmp_path.glob('candidate-decisions.jsonl*'))
    assert len(traces) == 2 and all(p.stat().st_size <= int(.003*1024*1024) for p in traces)
    obs['action_mask'][:] = 0
    obs['action_mask'][0] = 1
    audit(dict(episode=2, candidate_player=1), 1, obs, 0)
    assert audit.counts['2']['automatic'] == 1
    assert audit.counts['2']['heuristic_disagreement'] == 0


def test_real_loss_review_replays_same_weights_cases_and_results(tmp_path, monkeypatch):
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    import torch
    from firered_rl.ppo_features import policy_space, BattleFeatures
    from test_observation_v2_training import SHOWDOWN
    if not SHOWDOWN:
        pytest.skip('Showdown unavailable')
    torch.set_num_threads(1)
    class Shape(gym.Env):
        observation_space = policy_space('v2')
        action_space = gym.spaces.Discrete(11)
    model = MaskablePPO('MultiInputPolicy', Shape(), n_steps=2, batch_size=2, seed=17,
                       policy_kwargs=dict(features_extractor_class=BattleFeatures))
    checkpoint = tmp_path/'model.zip'
    model.save(checkpoint)
    original = checkpoint.read_bytes()
    teams = []
    for i, (species, level, moves) in enumerate((('Magikarp', 1, ['Splash', 'Tackle']), ('Mewtwo', 100, ['Psychic', 'Confusion']))):
        path = tmp_path/f'team-{i}.json'
        path.write_text(json.dumps(dict(party=[dict(species=species, level=level, moves=moves)])))
        teams.append(path)
    monkeypatch.syspath_prepend(str(ROOT/'mgba'))
    evaluate = importlib.import_module('evaluate_ppo')
    review = importlib.import_module('review_rule_losses')
    def invoke(cli, args):
        monkeypatch.setattr(sys, 'argv', ['cli', *map(str, args)])
        return cli.main()
    source = tmp_path/'source'
    assert invoke(evaluate, ['--checkpoint', checkpoint, '--rule-based-opponent', '--teams', *teams,
                            '--episodes', 4, '--max-steps', 5, '--environments', 2, '--output-dir', source]) == 0
    first = prepare_review(source, tmp_path/'prepared', matchups=1, seed=2)
    again = prepare_review(source, tmp_path/'prepared2', matchups=1, seed=2)
    assert first == again and len(first['cases']) == 2
    # Also exercise Rules inside a combined/pinned opponent pool.
    from firered_rl.evaluation import digest
    suite = json.loads((source/'suite.json').read_text())
    suite['opponents'] = {'rules': dict(kind='rule_based', group='rules', identity=suite.pop('rule_opponent'))}
    for case in suite['cases']:
        case.update(opponent_id='rules', opponent_group='rules', opponent_seed=case['seed'] ^ 0x47B9)
    (source/'suite.json').write_text(json.dumps(suite))
    config = json.loads((source/'config.json').read_text())
    config['suite_sha256'] = digest(suite)
    (source/'config.json').write_text(json.dumps(config))
    rows_path = source/'episodes.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    for row in rows:
        row.update(suite['cases'][row['episode']-1])
    rows_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    run = tmp_path/'review'
    assert invoke(review, ['--evaluation-dir', source, '--output-dir', run, '--matchups', 1]) == 0
    data = json.loads((run/'review.json').read_text())
    assert data['replay_verified'] and data['selected_loss_battles'] >= 1
    assert data['loss_decision_counts']['decisions'] >= 1
    assert checkpoint.read_bytes() == original
    move_review = importlib.import_module('audit_move_order')
    move_run = tmp_path/'move-review'
    assert invoke(move_review, ['--evaluation-dir', source, '--output-dir', move_run, '--matchups', 2]) == 0
    move_data = json.loads((move_run/'replay/move-order-audit.json').read_text())
    assert move_data['counts']['permutations'] > 0
    assert move_data['counts'].get('execution_matched', 0) > 0
    assert not move_data['counts'].get('execution_review_mismatch', 0)
    assert json.loads((move_run/'verification.json').read_text())['verified']
    assert checkpoint.read_bytes() == original
    # Outcome divergence must be clearly flagged, never silently treated as reproduction.
    rows_path = run/'replay/episodes.jsonl'
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]['decisions'] += 1
    rows_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert not report_review(run)['replay_verified']
    # Tampered snapshots must be rejected before output creation.
    (source/'candidate.zip').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint hash'):
        prepare_review(source, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_audit_serializes_numpy_damage_in_trace_and_report(tmp_path, monkeypatch):
    rules = RuleBasedOpponent(ROOT)
    obs = observation(rules)
    move(rules, obs, 0, 'Sonic Boom', accuracy=90)
    move(rules, obs, 1, 'Surf')
    audit = DecisionAudit(ROOT, tmp_path, SimpleNamespace(log_max_mb=2, log_backups=1))
    # Reproduce the scalar returned by fixed-damage arithmetic independently
    # of the NumPy version's promotion rules.
    monkeypatch.setattr(audit.rules, 'damage', lambda *args: (np.float32(18), np.float32(.9)))
    monkeypatch.setattr(audit.rules, 'choose', lambda obs: (1, [
        dict(action=0, score=0., reason='fixed damage'),
        dict(action=1, score=1., reason='estimated damage')]))
    audit(dict(episode=1, candidate_player=1), 1, obs, 0)
    audit.finish()
    row = json.loads((tmp_path/'candidate-decisions.jsonl').read_text())
    report = json.loads((tmp_path/'decision-audit.json').read_text())
    assert row['estimated_direct_damage']['0'] == 18.
    assert report['examples'][0]['estimated_direct_damage']['0'] == 18.
