"""Controlled initialization comparison: pinning, budgets and selection discipline."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('transfer_lab_tests', ROOT / 'run_transfer_lab.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def checkpoint(path, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'w') as archive:
        archive.writestr('data', json.dumps({'num_timesteps': steps}))
    return path


def settings():
    return dict(seed=261080, timesteps=1000000, environments=16, learning_rate=.0001,
        ent_coef=.01, eval_every=50000, max_steps=100, early_stop_consecutive=2,
        validation_threshold=.65, heldout_threshold=.60, rotation_max_drop=.03,
        mirror_max_drop=.10, max_truncation_rate=.01,
        selection='Highest validation wins/requested; earlier arm retains ties.')


def saved_plan(tmp_path):
    root = tmp_path / 'new-run'
    checkpoint(root / 'inputs/foundation-mirror.zip', 253952)
    checkpoint(root / 'inputs/fresh-initial.zip', 0)
    config = settings(); write(root / 'inputs/settings.json', config)
    teams = {}
    for group in ('train', 'mirror'):
        teams[group] = []
        for number in range(2):
            path = write(root / 'inputs/teams' / group / f'{number}.json', {'party': []})
            teams[group].append(str(path.relative_to(root)))
    plan = dict(version=1, experiment='initialization-transfer', settings=config,
        teams=teams, suites={'validation': 'inputs/suites/validation.json'},
        jobs=runner.make_jobs(root, config), smoke=False,
        pins={str(path.relative_to(root)): runner.checksum(path) for path in (root / 'inputs').rglob('*') if path.is_file()})
    return root, plan


def source_run(tmp_path, smoke=False):
    root = tmp_path / 'old-run'
    model = checkpoint(root / 'inputs/foundation-mirror.zip', 253952)
    write(root / 'inputs/reward.json', {})
    entries = {}
    for name in ('04-type-matchups', 'mirror'):
        paths = []
        for number in range(2):
            path = write(root / 'inputs/curriculum-teams' / name / 'train' / f'{number}.json', {'party': []})
            paths.append(str(path.relative_to(root / 'inputs')))
        entries[name] = dict(root='/old/absolute/path/inputs', train_paths=paths,
            train_dir=f'curriculum-teams/{name}/train',
            sha256={relative: runner.checksum(root / 'inputs' / relative) for relative in paths})
    suites = {runner.STAGE: {}, '02-fixed-mirror': {}}
    for name in runner.CONDITIONS:
        path = write(root / 'inputs/suites' / (name + '.json'), {'assets': {}, 'cases': []})
        suites['02-fixed-mirror' if name == 'mirror' else runner.STAGE]['test' if name == 'mirror' else name] = str(path.relative_to(root))
    write(root / 'stages/02-fixed-mirror/result.json', dict(status='smoke_completed' if smoke else 'passed',
        selected='inputs/foundation-mirror.zip', checkpoint_sha256=runner.checksum(model)))
    pins = {str(path.relative_to(root)): runner.checksum(path) for path in (root / 'inputs').rglob('*') if path.is_file()}
    write(root / 'plan.json', dict(version=1, smoke=smoke, imports={'mirror': 'inputs/foundation-mirror.zip'},
        assets={runner.STAGE: entries[runner.STAGE]}, fixed={'mirror': entries['mirror']}, suites=suites, pins=pins))
    return root


def test_source_only_requires_stage_four_and_qualified_mirror(tmp_path):
    source = source_run(tmp_path)
    evidence = runner.source_inputs(source)
    assert set(evidence['teams']) == {'train', 'mirror'}
    assert set(evidence['suites']) == set(runner.CONDITIONS)
    assert evidence['model'] == source / 'inputs/foundation-mirror.zip'
    assert not (source / 'stages/04-type-matchups/result.json').exists()


def test_source_rejects_unqualified_smoke_and_mutated_inputs(tmp_path):
    source = source_run(tmp_path, smoke=True)
    with pytest.raises(ValueError, match='smoke source'):
        runner.source_inputs(source)
    runner.source_inputs(source, True)
    (source / 'inputs/reward.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='pinned input changed'):
        runner.source_inputs(source, True)


def test_source_rejects_extra_training_team(tmp_path):
    source = source_run(tmp_path)
    write(source / 'inputs/curriculum-teams/04-type-matchups/train/extra.json', {})
    with pytest.raises(ValueError, match='membership changed'):
        runner.source_inputs(source)


def test_source_rejects_changed_foundation_hash(tmp_path):
    source = source_run(tmp_path)
    checkpoint(source / 'inputs/foundation-mirror.zip', 700000)
    with pytest.raises(ValueError, match='pinned input changed'):
        runner.source_inputs(source)


def test_equal_new_budgets_common_flags_different_initialization(tmp_path):
    root, plan = saved_plan(tmp_path)
    a, b = plan['jobs']
    assert a['target_timesteps'] - a['starting_timesteps'] == b['target_timesteps'] - b['starting_timesteps'] == 1000000
    cmds = [runner.command(root, plan, job) for job in plan['jobs']]
    for command in cmds:
        assert command[command.index('--total-timesteps') + 1] == '1000000'
        assert command[command.index('--learning-rate') + 1] == '0.0001'
        assert command[command.index('--seed') + 1] == '261080'
        assert command[command.index('--eval-every') + 1] == '50000'
        assert command[command.index('--eval-stop-win-rate') + 1] == '0.65'
        assert command[command.index('--eval-stop-consecutive') + 1] == '2'
    assert cmds[0][cmds[0].index('--resume') + 1].endswith('foundation-mirror.zip')
    assert cmds[1][cmds[1].index('--resume') + 1].endswith('fresh-initial.zip')


def test_continue_preserves_budget_and_skips_success(tmp_path):
    root, plan = saved_plan(tmp_path); job = plan['jobs'][0]
    run = root / 'runs' / job['name']
    checkpoint(run / 'checkpoints/latest.zip', job['starting_timesteps'] + 200000)
    write(run / 'summary.json', dict(status='interrupted'))
    command = runner.command(root, plan, job)
    assert command[-1] == '800000'
    assert '--if-exists' in command and 'continue' in command
    assert '--resume' not in command and '--team-mixture' not in command
    write(run / 'summary.json', dict(status='completed', stop_reason='validation_success'))
    assert runner.command(root, plan, job) is None


def test_budget_edge_retries_incomplete_final_validation_only(tmp_path):
    root, plan = saved_plan(tmp_path); job = plan['jobs'][1]
    run = root / 'runs' / job['name']
    checkpoint(run / 'checkpoints/latest.zip', job['target_timesteps'] + 448)
    write(run / 'summary.json', dict(status='interrupted'))
    assert runner.command(root, plan, job)[-1] == '1'
    write(run / 'summary.json', dict(status='completed', stop_reason='timestep_budget'))
    assert runner.command(root, plan, job) is None


def test_verify_rejects_changed_saved_budgets_and_extra_teams(tmp_path):
    root, plan = saved_plan(tmp_path)
    runner.verify(root, plan)
    plan['jobs'][0]['target_timesteps'] += 1
    with pytest.raises(ValueError, match='budgets changed'):
        runner.verify(root, plan)
    plan['jobs'] = runner.make_jobs(root, plan['settings'])
    write(root / 'inputs/teams/train/extra.json', {})
    with pytest.raises(ValueError, match='membership changed'):
        runner.verify(root, plan)


def test_selection_uses_requested_validation_denominator_and_no_test(tmp_path):
    root, plan = saved_plan(tmp_path)
    for job, wins in zip(plan['jobs'], (60, 70)):
        path = checkpoint(root / 'runs' / job['name'] / 'checkpoints/best.zip', 300000)
        write(path.with_suffix('.json'), dict(checkpoint_sha256=runner.checksum(path), wins=wins, requested=100,
            win_rate=.99 if wins == 60 else .70))
    write(root / 'checks/tempting-heldout/summary.json', {'candidate': 'specialist', 'win_rate': 1.0})
    result = runner.selection(root, plan)
    assert result['selected']['name'] == 'fresh'
    assert result['selected']['validation_rate'] == .7
    assert result['heldout_used_for_selection'] is False


def outcome(wins, truncated=0):
    return dict(status='completed', requested_battles=100, attempted_battles=100,
        completed_battles=100-truncated, truncated=truncated, candidate_wins=wins)


def test_readiness_requires_common_specialist_mirror_target():
    checks = dict(validation=outcome(70), heldout=outcome(65), permuted=outcome(65), mirror=outcome(50))
    issues = runner.readiness(checks, outcome(100), settings())
    assert issues == ['mirror below original specialist minus 10 percentage points']
    checks['mirror'] = outcome(90)
    assert runner.readiness(checks, outcome(100), settings()) == []


def test_fresh_checkpoint_has_reproducible_new_weights_and_empty_optimizer(tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(1)
    pytest.importorskip('sb3_contrib')
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import policy_space
    template = SimpleNamespace(observation_space=policy_space('v2'), action_space=gym.spaces.Discrete(11), reward_settings={'scale': .001})
    paths = [tmp_path / 'fresh1.zip', tmp_path / 'fresh2.zip']
    for path in paths: runner.fresh_checkpoint(template, path, settings())
    first, second = [MaskablePPO.load(path, device='cpu') for path in paths]
    assert first.num_timesteps == second.num_timesteps == 0
    assert not first.policy.optimizer.state
    assert first.n_steps == 128 and first.n_epochs == 4 and first.seed == 261080
    assert all(torch.equal(v, second.policy.state_dict()[k]) for k, v in first.policy.state_dict().items())


def test_missing_legacy_immunity_fields_are_not_reported_as_zero():
    assert runner.immunity_counts({'audit_totals': {'move_actions': 100}}) == ['N/A', 'N/A', 'N/A']
    assert runner.immunity_counts({'audit_totals': {'matchup_classified_move_actions': 100, 'current_target_immune_move_actions': 0, 'current_target_immune_with_nonimmune_alternative': 0}}) == ['100', '0', '0']
