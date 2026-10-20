"""Controlled trained-policy migration, bounded budgets, validation selection and evidence."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_switch_feature_ablation as runner
from test_presentation_check import source
from test_transfer_lab import write
from test_replacement_diagnostic_runner import saved_check


def migrated(path, steps, enabled, start=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(version=1, enabled=enabled, template_timesteps=start,
        optimizer='preserved-existing/new-empty', seed=261120)
    with ZipFile(path, 'w') as archive:
        archive.writestr('data', json.dumps(dict(num_timesteps=steps, policy_architecture='matchup-switches', switch_ablation=meta)))
    return path


def source_checks(source):
    plan, result = runner.read_json(source / 'plan.json'), runner.read_json(source / 'result.json')
    record = result['policies']['control']['best']
    frozen = dict(models=dict(candidate=dict(checkpoint=record['checkpoint'], sha256=record['checkpoint_sha256'],
        timesteps=record['timesteps'])), suites=plan['suites'], pins=plan['pins'], implementation={})
    for condition in runner.CONDITIONS:
        check = dict(label='source-'+condition, checkpoint='candidate', baseline=None, pool=condition,
            order='reversed' if condition.endswith('reversed') else 'original', assisted=False, suite=condition)
        suite, identity = runner.bench.diagnostic.check_context(source, frozen, check)
        directory = source / 'checks' / check['label']
        value = saved_check(directory, check, suite)
        value.update(identity, result_directory=str(directory.relative_to(source)))
        write(directory / 'summary.json', value)
        record['checks'][condition] = value
    write(source / 'result.json', result)


@pytest.fixture
def prepared(source, tmp_path, monkeypatch):
    from sb3_contrib import MaskablePPO
    import firered_rl.matchup_switch_policy
    source_checks(source)
    monkeypatch.setattr(MaskablePPO, 'load', lambda *args, **kwargs: object())
    monkeypatch.setattr(firered_rl.matchup_switch_policy, 'create_switch_checkpoint',
        lambda template, path, **kwargs: migrated(path, 128, kwargs['enabled']))
    monkeypatch.setattr(runner, 'initial_identity', lambda *args: dict(initial_outputs_equal=True,
        parameter_tensors_equal=True, optimizer_states_equal=True))
    monkeypatch.setattr(runner, 'implementation_identity', lambda: {'runner-test': 'fixed'})
    root = tmp_path / 'switch'
    plan = runner.freeze(root, SimpleNamespace(source_run=source, smoke=True, timesteps=500000))
    return root, plan


def training(root, plan):
    from firered_rl.evaluation import digest
    suite = runner.read_json(root / plan['suites']['validation'])
    for job in plan['jobs']:
        run = root / 'runs' / job['name']
        for kind, steps in (('best', job['starting_timesteps']+128), ('final', job['target_timesteps'])):
            migrated(run / 'checkpoints' / (kind+'.zip'), steps, job['bench_features_enabled'])
        write(run / 'checkpoints/best.json', dict(requested=len(suite['cases']), wins=1,
            timesteps=job['starting_timesteps']+128, suite_sha256=digest(suite),
            checkpoint_sha256=runner.checksum(run / 'checkpoints/best.zip')))
        migration = runner.checkpoint_metadata(root / job['checkpoint'])
        write(run / 'config.json', dict(session=dict(id=1), shuffle_bench_order=False,
            policy_architecture='matchup-switches', resume_checkpoint=dict(sha256=plan['pins'][job['checkpoint']]),
            switch_features=dict(version=1, enabled=job['bench_features_enabled'], whole_policy_training=True,
                scope='shared residual on multi-option forced replacements only', migration=migration)))
        write(run / 'evaluation-history.json', dict(identity=dict(suite_sha256=digest(suite)), records=[
            dict(status='completed', checkpoint_sha256=runner.checksum(run / 'checkpoints/final.zip'),
                 timesteps=job['target_timesteps'], requested=len(suite['cases']), wins=0)]))
        write(run / 'summary.json', dict(status='completed', stop_reason='timestep_budget',
            player_timesteps=job['target_timesteps'], elapsed_seconds=1))


def simulator(monkeypatch):
    calls = []
    def frozen(root, plan, check):
        existing = runner.bench.diagnostic.cached_check(root, plan, check)
        if existing is not None: return existing
        calls.append(check['label'])
        suite, identity = runner.bench.diagnostic.check_context(root, plan, check)
        base = root / 'checks' / check['label']
        directory = base if not base.exists() else base.with_name(base.name+'-retry-unit')
        result = saved_check(directory, check, suite)
        result.update(identity, result_directory=str(directory.relative_to(root)))
        write(directory / 'summary.json', result)
        return result
    monkeypatch.setattr(runner.bench.diagnostic, 'run_check', frozen)
    return calls


def test_freeze_uses_selected_best_and_imports_complete_evidence(prepared, source):
    root, plan = prepared
    before = {str(path): runner.checksum(path) for path in source.rglob('*') if path.is_file()}
    runner.verify(root, plan)
    assert plan['settings']['starting_timesteps'] == 128
    assert plan['settings']['seed'] == 261120
    assert plan['settings']['timesteps'] == 256
    assert plan['settings']['early_stopping'] is False
    assert [job['bench_features_enabled'] for job in plan['jobs']] == [False, True]
    assert all(job['target_timesteps'] == 384 and not job['shuffle_bench_order'] for job in plan['jobs'])
    assert (root / 'inputs/source.zip').read_bytes() == (source / 'runs/control/checkpoints/best.zip').read_bytes()
    initial = runner.read_json(root / 'initial.json')
    assert initial['imported'] and len(initial['checks']) == 5
    runner.bench.validate_saved_checks(root, [initial])
    assert before == {str(path): runner.checksum(path) for path in source.rglob('*') if path.is_file()}


def test_commands_use_migrated_architecture_same_budget_and_no_shuffle(prepared):
    root, plan = prepared
    values = [runner.command(root, plan, job) for job in plan['jobs']]
    for command, job in zip(values, plan['jobs']):
        assert command[command.index('--resume')+1] == str(root / job['checkpoint'])
        assert command[command.index('--policy-architecture')+1] == 'matchup-switches'
        assert command[command.index('--total-timesteps')+1] == '256'
        assert '--no-shuffle-bench-order' in command and '--no-eval-stop' in command
        assert command[command.index('--eval-suite')+1] == str(root / plan['suites']['validation'])
    normalized = [[part.replace('switch-features', 'control') for part in command] for command in values]
    assert normalized[0] == normalized[1]


@pytest.mark.parametrize('change', ['source', 'flag', 'budget', 'identity', 'membership', 'implementation'])
def test_verify_rejects_changed_plan_or_inputs(prepared, monkeypatch, change):
    root, plan = prepared
    if change == 'source': (root / 'inputs/source.zip').write_bytes(b'changed')
    elif change == 'flag': plan['jobs'][1]['bench_features_enabled'] = False
    elif change == 'budget': plan['settings']['timesteps'] += 1
    elif change == 'identity': plan['initial_identity']['initial_outputs_equal'] = False
    elif change == 'membership': write(root / 'inputs/extra.json', {})
    elif change == 'implementation': monkeypatch.setattr(runner, 'implementation_identity', lambda: {})
    with pytest.raises(ValueError): runner.verify(root, plan)


def test_resume_only_remaining_budget(prepared, monkeypatch):
    root, plan = prepared; job = plan['jobs'][0]
    write(root / 'runs/control/summary.json', dict(status='interrupted'))
    monkeypatch.setattr(runner.transfer, 'recovered_steps', lambda _: 256)
    command = runner.command(root, plan, job)
    assert command[command.index('--total-timesteps')+1] == '128'
    assert command[command.index('--if-exists')+1] == 'continue'
    write(root / 'runs/control/summary.json', dict(status='completed'))
    monkeypatch.setattr(runner.transfer, 'recovered_steps', lambda _: 384)
    assert runner.command(root, plan, job) is None


@pytest.mark.parametrize('change', ['resume', 'flag', 'architecture', 'shuffle', 'optimizer', 'budget'])
def test_actual_trainer_provenance_is_required(prepared, change):
    root, plan = prepared; training(root, plan)
    job = plan['jobs'][1]; path = root / 'runs' / job['name'] / 'config.json'
    config = runner.read_json(path)
    if change == 'resume': config['resume_checkpoint']['sha256'] = 'bad'
    elif change == 'flag': config['switch_features']['enabled'] = False
    elif change == 'architecture': config['policy_architecture'] = 'matchup-moves'
    elif change == 'shuffle': config['shuffle_bench_order'] = True
    elif change == 'optimizer': config['switch_features']['migration']['optimizer'] = 'reset'
    else: config['switch_features']['migration']['template_timesteps'] = 0
    write(path, config)
    with pytest.raises(ValueError): runner.training_provenance(root, plan, job, {})


def test_finish_selects_on_validation_exports_and_reuses_complete_checks(prepared, monkeypatch):
    root, plan = prepared; training(root, plan); calls = simulator(monkeypatch)
    result = runner.finish(root, plan)
    assert result['selected_by_validation']['name'] == 'control'  # Earlier arm retains validation tie.
    assert len(calls) == 20
    assert len(result['saved_evidence']['checked_directories']) == 25
    assert len(result['exports']) == 5
    for model in result['exports'].values(): assert runner.checksum(root / model['checkpoint']) == model['sha256']
    assert (root / 'models/selected.zip').read_bytes() == (root / 'runs/control/checkpoints/best.zip').read_bytes()
    runner.finish(root, plan)
    assert len(calls) == 20
    runner.collect(root, plan)
    with ZipFile(root / 'switch-feature-results.zip') as archive:
        assert 'models/manifest.json' in archive.namelist()
        assert not any(name.endswith('.zip') for name in archive.namelist())
    assert 'imported start' in (root / 'report.md').read_text()


def test_finish_rejects_wrong_validation_identity(prepared):
    root, plan = prepared; training(root, plan)
    path = root / 'runs/control/checkpoints/best.json'; meta = runner.read_json(path)
    meta['suite_sha256'] = 'changed'; write(path, meta)
    with pytest.raises(ValueError, match='pinned balanced validation'): runner.finish(root, plan)


def test_finish_rejects_early_success_stop(prepared, monkeypatch):
    root, plan = prepared; training(root, plan); simulator(monkeypatch)
    path = root / 'runs/control/summary.json'; meta = runner.read_json(path)
    meta['stop_reason'] = 'validation_success'; write(path, meta)
    with pytest.raises(ValueError, match='fixed timestep budget'): runner.finish(root, plan)


def test_lost_imported_baseline_evidence_blocks_finish(prepared):
    root, plan = prepared; training(root, plan)
    (root / 'checks/imported-start-mirror/episodes.jsonl').unlink()
    with pytest.raises(RuntimeError, match='incomplete'): runner.finish(root, plan)


def test_partial_check_retry_preserves_old_files(prepared, monkeypatch):
    root, plan = prepared; training(root, plan); calls = simulator(monkeypatch)
    runner.finish(root, plan)
    directory = root / 'checks/control-best-seen-original'
    (directory / 'decision-traces.jsonl').unlink()
    result = runner.finish(root, plan)
    assert directory.exists() and not (directory / 'decision-traces.jsonl').exists()
    assert result['policies']['control']['best']['checks']['seen-original']['result_directory'].endswith('-retry-unit')
    assert len(calls) == 21


def test_checkpoint_flag_and_metadata_are_strict(tmp_path):
    path = migrated(tmp_path / 'model.zip', 256, True)
    assert runner.checkpoint_metadata(path, enabled=True, starting_timesteps=128)['enabled']
    with pytest.raises(ValueError, match='feature flag'): runner.checkpoint_metadata(path, enabled=False)
    with pytest.raises(ValueError, match='source timestep'): runner.checkpoint_metadata(path, starting_timesteps=0)


def test_identity_checks_real_migrated_outputs(tmp_path):
    import gymnasium as gym
    import torch
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import policy_space
    from firered_rl.matchup_move_policy import MatchupMovePolicy
    from firered_rl.matchup_switch_policy import create_switch_checkpoint
    torch.set_num_threads(1)
    source = MaskablePPO(MatchupMovePolicy, None, _init_setup_model=False, seed=12)
    source.observation_space = policy_space('v2'); source.action_space = gym.spaces.Discrete(11)
    source.n_envs = 1; source._setup_model(); source.num_timesteps = 128
    models = [create_switch_checkpoint(source, tmp_path / f'{enabled}.zip', enabled=enabled, seed=261120)
              for enabled in (False, True)]
    evidence = runner.initial_identity(source, models)
    assert evidence['initial_outputs_equal'] and evidence['max_source_output_differences'] == [0., 0.]
    assert evidence['probes'] == 6
    with torch.no_grad(): next(models[1].policy.parameters()).add_(1.)
    with pytest.raises(ValueError, match='parameter tensors'): runner.initial_identity(source, models)


def test_final_validation_requires_same_checkpoint_and_suite(prepared, monkeypatch):
    root, plan = prepared; training(root, plan); simulator(monkeypatch)
    path = root / 'runs/control/evaluation-history.json'; history = runner.read_json(path)
    history['records'][0]['checkpoint_sha256'] = 'wrong-model'; write(path, history)
    with pytest.raises(ValueError, match='matching completed validation history'): runner.finish(root, plan)
