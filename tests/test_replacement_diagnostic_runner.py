"""Frozen diagnostic provenance, saved evidence, and no-training orchestration."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('replacement_runner_tests', ROOT / 'run_replacement_diagnostic.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
from test_transfer_lab import checkpoint, write
from test_replacement_diagnostics import suite
from firered_rl.evaluation import digest


@pytest.fixture
def source(tmp_path, monkeypatch):
    import run_matchup_ablation
    import firered_rl.showdown_env
    monkeypatch.setattr(run_matchup_ablation, 'verify_saved_checks', lambda *args: {'verified': True})
    monkeypatch.setattr(firered_rl.showdown_env, 'validate_showdown', lambda *args: None)
    root = tmp_path / 'source'
    frozen = suite()
    write(root / 'inputs/reward.json', {'win': 5000, 'loss': -5000})
    teams = {}
    for group in ('train', 'mirror'):
        teams[group] = []
        for i, asset in enumerate(frozen['assets'].values()):
            path = write(root / 'inputs/teams' / group / f'team-{i}.json', asset['team'])
            teams[group].append(str(path.relative_to(root)))
    suites = {}
    for name in ('test', 'heldout', 'validation', 'mirror'):
        path = write(root / 'inputs/suites' / f'{name}.json', frozen)
        suites[name] = str(path.relative_to(root))
    models = {}
    for kind, steps in (('best', 128), ('final', 256)):
        path = root / 'runs/matchup/checkpoints' / (kind + '.zip')
        checkpoint(path, steps)
        models[kind] = dict(checkpoint=str(path.relative_to(root)), checkpoint_sha256=runner.checksum(path))
    write(root / 'runs/matchup/checkpoints/best.json', dict(timesteps=128, checkpoint_sha256=models['best']['checkpoint_sha256']))
    selected = dict(name='matchup', checkpoint_sha256=models['best']['checkpoint_sha256'])
    write(root / 'result.json', dict(status='completed', policies=dict(matchup=models),
        training=dict(matchup=dict(ending_timesteps=256)), selected_by_validation=selected))
    write(root / 'plan.json', dict(version=1, experiment='matchup-feature-ablation', smoke=False,
        teams=teams, suites=suites, pins={str(path.relative_to(root)): runner.checksum(path)
            for path in (root / 'inputs').rglob('*') if path.is_file()}))
    return root


@pytest.fixture
def prepared(source, tmp_path):
    root = tmp_path / 'diagnostic'
    plan = runner.freeze(root, SimpleNamespace(source_run=source, smoke=False))
    return root, plan


def saved_check(root, check, suite_data, *, wins=None):
    """Write internally consistent artifacts, as the simulator would."""
    root.mkdir(parents=True, exist_ok=True)
    write(root / 'suite.json', suite_data)
    cases = suite_data['cases']
    if wins is None: wins = {cases[0]['episode']}
    episodes = [dict(case, candidate_outcome='won' if case['episode'] in wins else 'lost',
        terminated=True, truncated=False, candidate_reward=1, opponent_reward=-1, decisions=1) for case in cases]
    (root / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in reversed(episodes)))
    totals = dict.fromkeys(runner.REPLACEMENT_COUNTERS, 0)
    totals['decisions'] = len(cases)
    audit = dict(totals=totals, by_episode={str(case['episode']): {'decisions': 1} for case in cases},
        replacement_diagnostic=dict(version=1, assisted=check['assisted'], all_battle_counts=True),
        trace_limits=dict(cases=16, decisions_per_case=24), traced_decisions=0)
    write(root / 'decision-audit.json', audit)
    (root / 'decision-traces.jsonl').write_text('')
    result = dict(status='completed', requested_battles=len(cases), attempted_battles=len(cases),
        completed_battles=len(cases), truncated=0, candidate_wins=len(wins),
        opponent_wins=len(cases)-len(wins), draws=0, suite_sha256=digest(suite_data),
        audit_totals=totals, wins_per_requested=len(wins)/len(cases))
    return result


def cached_fixture(root, plan, check, *, wins=None):
    suite_data, identity = runner.check_context(root, plan, check)
    directory = root / 'checks' / check['label']
    result = saved_check(directory, check, suite_data, wins=wins)
    result.update(identity, result_directory=str(directory.relative_to(root)))
    write(directory / 'summary.json', result)
    return directory, result


def test_source_read_is_immutable_and_preserves_validation_choice(source):
    before = {str(path): runner.checksum(path) for path in source.rglob('*') if path.is_file()}
    evidence = runner.source_inputs(source)
    assert evidence['models']['best']['timesteps'] == 128
    assert evidence['models']['final']['timesteps'] == 256
    assert evidence['source_selected']['name'] == 'matchup'
    assert before == {str(path): runner.checksum(path) for path in source.rglob('*') if path.is_file()}


def test_diagnostic_can_target_matchup_even_when_source_selected_control(source):
    result = runner.read_json(source / 'result.json')
    result['selected_by_validation'] = dict(name='control', checkpoint_sha256='different-valid-source-arm')
    write(source / 'result.json', result)
    assert runner.source_inputs(source)['source_selected']['name'] == 'control'


@pytest.mark.parametrize('change', ['weights', 'best_metadata', 'selected_hash', 'steps', 'reward', 'escape'])
def test_source_rejects_broken_provenance(source, change):
    result = runner.read_json(source / 'result.json')
    if change == 'weights':
        (source / result['policies']['matchup']['best']['checkpoint']).write_bytes(b'changed')
    elif change == 'best_metadata':
        write(source / 'runs/matchup/checkpoints/best.json', dict(timesteps=128, checkpoint_sha256='changed'))
    elif change == 'selected_hash':
        result['selected_by_validation']['checkpoint_sha256'] = 'changed'
    elif change == 'steps':
        result['training']['matchup']['ending_timesteps'] = 999
    elif change == 'reward':
        write(source / 'inputs/reward.json', {'win': 100})
    elif change == 'escape':
        result['policies']['matchup']['best']['checkpoint'] = '../outside.zip'
    write(source / 'result.json', result)
    with pytest.raises(ValueError): runner.source_inputs(source)


def test_freeze_makes_exact_twenty_checks_and_never_targets_source(prepared, source):
    root, plan = prepared
    assert len(plan['checks']) == 20
    assert sum(check['checkpoint'] is not None for check in plan['checks']) == 16
    assert sum(check['assisted'] for check in plan['checks']) == 8
    assert all(model['checkpoint'].startswith('inputs/models/') for model in plan['models'].values())
    assert all(runner.checksum(root / model['checkpoint']) == model['sha256'] for model in plan['models'].values())
    assert plan['no_training'] and plan['no_automatic_promotion']
    runner.verify(root, plan)
    with pytest.raises(ValueError, match='outside'):
        runner.freeze(source / 'nested', SimpleNamespace(source_run=source, smoke=False))


@pytest.mark.parametrize('change', ['input', 'membership', 'model_mapping', 'conditions', 'implementation'])
def test_verify_detects_changed_inputs_or_conditions(prepared, change, monkeypatch):
    root, plan = prepared
    if change == 'input': (root / 'inputs/reward.json').write_text('{}')
    elif change == 'membership': write(root / 'inputs/teams/train/extra.json', {})
    elif change == 'model_mapping': plan['models']['best'] = copy.deepcopy(plan['models']['final'])
    elif change == 'conditions': plan['checks'][0]['assisted'] = True
    elif change == 'implementation': monkeypatch.setattr(runner, 'implementation_identity', lambda: {})
    with pytest.raises(ValueError): runner.verify(root, plan)


def test_cache_requires_complete_episode_schedule_and_replacement_audit(prepared):
    root, plan = prepared; check = plan['checks'][0]
    directory, result = cached_fixture(root, plan, check)
    assert runner.cached_check(root, plan, check) is not None
    (directory / 'episodes.jsonl').write_text('')
    assert runner.cached_check(root, plan, check) is None
    directory, result = cached_fixture(root, plan, check)
    rows = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines()]
    rows[0]['seed'] += 1
    (directory / 'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert runner.cached_check(root, plan, check) is None


@pytest.mark.parametrize('change', ['metadata', 'assisted', 'version', 'totals', 'suite', 'override', 'traces_missing', 'traces_partial'])
def test_cache_rejects_missing_or_inconsistent_diagnostic(prepared, change):
    root, plan = prepared; check = plan['checks'][0]
    directory, result = cached_fixture(root, plan, check)
    audit = runner.read_json(directory / 'decision-audit.json')
    if change == 'metadata': audit.pop('replacement_diagnostic')
    elif change == 'assisted': audit['replacement_diagnostic']['assisted'] = True
    elif change == 'version': audit['replacement_diagnostic']['version'] = 2
    elif change == 'totals': audit['totals']['replacement_eligible'] = 1
    elif change == 'suite':
        data = runner.read_json(directory / 'suite.json'); data['cases'][0]['seed'] += 1
        write(directory / 'suite.json', data)
    elif change == 'override':
        audit['totals']['replacement_overrides_changed'] = 1
    elif change == 'traces_missing': (directory / 'decision-traces.jsonl').unlink()
    elif change == 'traces_partial': audit['traced_decisions'] = 1
    write(directory / 'decision-audit.json', audit)
    assert runner.cached_check(root, plan, check) is None


def test_partial_retry_preserves_old_diagnostics_and_model_files(prepared, monkeypatch):
    import firered_rl.curriculum_checks as checks
    import firered_rl.shared_move_policy as shared
    import firered_rl.replacement_diagnostics as diagnostics
    import firered_rl.staged_team_lab as staged
    root, plan = prepared; check = plan['checks'][0]
    partial = root / 'checks' / check['label']; partial.mkdir(parents=True)
    write(partial / 'summary.json', {'status': 'failed'})
    before = {kind: runner.checksum(root / info['checkpoint']) for kind, info in plan['models'].items()}
    monkeypatch.setattr(checks, '_load_model', lambda path: SimpleNamespace(num_timesteps=runner.checkpoint_steps(path),
        policy=SimpleNamespace(features_extractor=SimpleNamespace(relational_enabled=True))))
    monkeypatch.setattr(shared, 'policy_architecture', lambda model: 'matchup-moves')
    monkeypatch.setattr(diagnostics, 'make_candidate', lambda model, assisted, project: model)
    def score(project, directory, suite_data, candidate, reward, **kwargs):
        assert candidate.num_timesteps == 128
        assert kwargs['pairs'] == 2 and kwargs['audit_factory'] is diagnostics.ReplacementAudit
        return saved_check(directory, check, suite_data)
    monkeypatch.setattr(staged, 'score_stage', score)
    result = runner.run_check(root, plan, check)
    assert '-retry-' in result['result_directory']
    assert runner.read_json(partial / 'summary.json') == {'status': 'failed'}
    assert before == {kind: runner.checksum(root / info['checkpoint']) for kind, info in plan['models'].items()}
    monkeypatch.setattr(staged, 'score_stage', lambda *args, **kwargs: pytest.fail('Complete check must not rerun'))
    assert runner.run_check(root, plan, check)['result_directory'] == result['result_directory']


def test_finish_requires_all_twenty_saved_checks_and_collection_labels_hybrid(prepared):
    root, plan = prepared
    with pytest.raises(RuntimeError, match='Not all'): runner.finish(root, plan)
    for check in plan['checks']: cached_fixture(root, plan, check)
    result = runner.finish(root, plan)
    assert len(result['checks']) == 20 and result['no_training']
    assert result['source_selected'] == plan['source_selected']
    runner.collect(root, plan)
    text = (root / 'report.md').read_text()
    assert 'hybrids' in text and 'Hybrid original' in text and '20/20' in text
    with ZipFile(root / 'replacement-results.zip') as archive:
        assert 'paired-comparisons.json' in archive.namelist()
        assert not any(name.endswith('.zip') or name.startswith('console/') for name in archive.namelist())
    comparisons = runner.read_json(root / 'paired-comparisons.json')
    assert len(comparisons) == 12
    assert all(value['win_rate_difference'] == 0 for value in comparisons.values())


def test_completed_launch_never_starts_worker_and_jobs_are_bounded(prepared, monkeypatch):
    root, plan = prepared
    for check in plan['checks']: cached_fixture(root, plan, check)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Completed checks should be reused'))
    runner.launch(root, plan, 2)
    with pytest.raises(ValueError, match='one or two'): runner.launch(root, plan, 3)


def test_inspector_shows_actual_moves_and_source_pool_role(prepared, capsys):
    root, plan = prepared
    runner.inspect(root, plan, 'heldout', 1, True)
    text = capsys.readouterr().out
    assert 'no training' in text and 'Surf' in text and 'Blastoise' in text
