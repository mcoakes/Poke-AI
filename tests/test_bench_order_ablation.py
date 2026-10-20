"""Controlled continuation budgets, balanced selection, and frozen artifact integrity."""
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
spec = importlib.util.spec_from_file_location('bench_order_runner_tests', ROOT / 'run_bench_order_ablation.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
from test_transfer_lab import checkpoint, write
from test_replacement_diagnostic_runner import source, saved_check
from test_replacement_diagnostics import suite
from firered_rl.evaluation import validate_suite
from firered_rl.replacement_diagnostics import transform_suite


@pytest.fixture
def prepared(source, tmp_path):
    root = tmp_path / 'bench-lab'
    plan = runner.freeze(root, SimpleNamespace(source_run=source, smoke=False, timesteps=500000))
    return root, plan


def test_validation_balances_bench_orders_without_changing_seed_or_seat():
    original = suite(); before = copy.deepcopy(original)
    reversed_suite = transform_suite(original)
    combined = runner.combined_validation(original, reversed_suite)
    validate_suite(combined)
    assert original == before
    assert len(combined['cases']) == 4
    assert {case['episode'] for case in combined['cases']} == {1, 2, 3, 4}
    assert combined['cases'][:2] == original['cases']
    for old, new in zip(reversed_suite['cases'], combined['cases'][2:]):
        assert all(old[key] == new[key] for key in ('assets', 'seed', 'candidate_player'))
    assert combined['diagnostic']['equal_order_weight']
    assert len(combined['diagnostic']['paired_orders']) == 2


def test_validation_rejects_unpaired_schedules():
    original = suite(); reversed_suite = transform_suite(original)
    for case in reversed_suite['cases']: case['seed'] += 1
    with pytest.raises(ValueError, match='schedules differ'):
        runner.combined_validation(original, reversed_suite)


def test_freeze_preserves_full_checkpoint_bytes_and_source(prepared, source):
    root, plan = prepared
    source_checkpoint = source / 'runs/matchup/checkpoints/best.zip'
    assert (root / 'inputs/start.zip').read_bytes() == source_checkpoint.read_bytes()
    assert all(job['checkpoint'] == 'inputs/start.zip' for job in plan['jobs'])
    assert all(job['starting_timesteps'] == 128 and job['target_timesteps'] == 500128 for job in plan['jobs'])
    assert plan['jobs'][0]['seed'] == plan['jobs'][1]['seed'] == 261100
    assert set(plan['suites']) == set(runner.CONDITIONS) and len(plan['suites']) == 7
    assert runner.read_json(root / 'inputs/mixture.json')['groups']['fixed-mirror']['weight'] == .1
    assert runner.read_json(root / 'inputs/mixture.json')['groups']['current']['weight'] == .9
    assert plan['no_automatic_promotion'] and plan['settings']['early_stopping'] is False
    runner.verify(root, plan)


def test_default_budget_positive_and_smoke_override(source, tmp_path):
    with pytest.raises(ValueError, match='positive'):
        runner.freeze(tmp_path / 'invalid', SimpleNamespace(source_run=source, smoke=False, timesteps=0))
    root = tmp_path / 'smoke'
    plan = runner.freeze(root, SimpleNamespace(source_run=source, smoke=True, timesteps=12345))
    assert plan['settings']['timesteps'] == 256 and plan['settings']['environments'] == 2
    assert plan['settings']['eval_every'] == 128


@pytest.mark.parametrize('change', ['input', 'settings', 'jobs', 'suite_mapping', 'implementation', 'membership'])
def test_verify_rejects_mutated_frozen_configuration(prepared, change, monkeypatch):
    root, plan = prepared
    if change == 'input': (root / 'inputs/start.zip').write_bytes(b'changed')
    elif change == 'settings': plan['settings']['timesteps'] += 1
    elif change == 'jobs': plan['jobs'][0]['shuffle_bench_order'] = True
    elif change == 'suite_mapping': plan['suites']['validation'] = plan['suites']['heldout-original']
    elif change == 'implementation': monkeypatch.setattr(runner, 'implementation_identity', lambda: {})
    elif change == 'membership': write(root / 'inputs/teams/train/extra.json', {})
    with pytest.raises(ValueError): runner.verify(root, plan)


def test_commands_differ_only_in_bench_flag_and_job_output(prepared):
    root, plan = prepared
    commands = [runner.command(root, plan, job) for job in plan['jobs']]
    control, treatment = commands
    assert '--no-shuffle-bench-order' in control and '--shuffle-bench-order' in treatment
    for cmd in commands:
        assert cmd[cmd.index('--resume')+1] == str(root / 'inputs/start.zip')
        assert cmd[cmd.index('--policy-architecture')+1] == 'matchup-moves'
        assert cmd[cmd.index('--total-timesteps')+1] == '500000'
        assert cmd[cmd.index('--eval-suite')+1] == str(root / plan['suites']['validation'])
        assert '--no-eval-stop' in cmd and '--eval-stop-win-rate' not in cmd
        assert '--eval-stop-consecutive' not in cmd
    normalized = [[part.replace('bench-shuffle', 'control').replace('--shuffle-bench-order', '--no-shuffle-bench-order')
        for part in cmd] for cmd in commands]
    assert normalized[0] == normalized[1]


def test_resume_only_remaining_budget_and_no_early_success_shortcut(prepared, monkeypatch):
    root, plan = prepared; job = plan['jobs'][1]
    run = root / 'runs' / job['name']
    write(run / 'summary.json', dict(status='completed', stop_reason='validation_success'))
    monkeypatch.setattr(runner.transfer, 'recovered_steps', lambda _: 200128)
    cmd = runner.command(root, plan, job)
    assert cmd[cmd.index('--total-timesteps')+1] == '300000'
    assert cmd[cmd.index('--if-exists')+1] == 'continue'
    assert '--shuffle-bench-order' not in cmd  # Immutable native configuration restores it.
    monkeypatch.setattr(runner.transfer, 'recovered_steps', lambda _: 500128)
    assert runner.command(root, plan, job) is None


def test_incomplete_final_validation_at_budget_gets_one_rollout_request(prepared, monkeypatch):
    root, plan = prepared; job = plan['jobs'][0]
    write(root / 'runs' / job['name'] / 'summary.json', dict(status='interrupted'))
    monkeypatch.setattr(runner.transfer, 'recovered_steps', lambda _: job['target_timesteps'])
    cmd = runner.command(root, plan, job)
    assert cmd[cmd.index('--total-timesteps')+1] == '1'


def test_frozen_adapter_never_enables_rules_assistance(prepared, monkeypatch):
    root, plan = prepared
    calls = []
    def frozen(run, frozen_plan, check):
        calls.append((frozen_plan, check))
        return dict(candidate_wins=1, requested_battles=2, wins_per_requested=.5, truncated=0)
    monkeypatch.setattr(runner.diagnostic, 'run_check', frozen)
    runner.frozen(root, plan, 'initial-seen-original', 'seen-original', root / 'inputs/start.zip')
    frozen_plan, check = calls[0]
    assert check['assisted'] is False and check['baseline'] is None
    assert frozen_plan['models']['candidate']['sha256'] == plan['source_start_sha256']
    assert frozen_plan['implementation'] == plan['implementation']


def install_fake_simulator(monkeypatch):
    def frozen(root, plan, check):
        cached = runner.diagnostic.cached_check(root, plan, check)
        if cached is not None: return cached
        suite_data, identity = runner.diagnostic.check_context(root, plan, check)
        base = root / 'checks' / check['label']
        directory = base if not base.exists() else base.with_name(base.name + '-retry-unit')
        result = saved_check(directory, check, suite_data)
        result.update(identity, result_directory=str(directory.relative_to(root)))
        write(directory / 'summary.json', result)
        return result
    monkeypatch.setattr(runner.diagnostic, 'run_check', frozen)


def complete_training_fixture(root, plan):
    requested = len(runner.read_json(root / plan['suites']['validation'])['cases'])
    for job in plan['jobs']:
        run = root / 'runs' / job['name']
        checkpoint(run / 'checkpoints/best.zip', job['starting_timesteps']+128)
        checkpoint(run / 'checkpoints/final.zip', job['target_timesteps'])
        write(run / 'checkpoints/best.json', dict(requested=requested, wins=1,
            checkpoint_sha256=runner.checksum(run / 'checkpoints/best.zip')))
        write(run / 'config.json', dict(session=dict(id=1), shuffle_bench_order=job['shuffle_bench_order'],
            resume_checkpoint=dict(sha256=plan['source_start_sha256'])))
        summary = dict(status='completed', stop_reason='timestep_budget', session_id=1,
            player_timesteps=job['target_timesteps'], elapsed_seconds=1)
        if job['shuffle_bench_order']:
            summary['bench_order_shuffle'] = dict(version=1, scope='candidate_player_1_nonfixed_groups',
                lead='unchanged', opponent='unchanged', moves='unchanged',
                rehearsal='fixed-pair mixture groups are excluded',
                counts=dict(episode_selections=2, eligible=2, skipped_fixed_pair=0, identity=1, changed=1))
        write(run / 'summary.json', summary)


def test_initial_checks_shared_and_finish_uses_validation_only(prepared, monkeypatch):
    root, plan = prepared
    install_fake_simulator(monkeypatch)
    initial = runner.evaluate_initial(root, plan)
    assert len(initial['checks']) == 7 and initial['shared_by_both_arms']
    complete_training_fixture(root, plan)
    result = runner.finish(root, plan, initial)
    assert result['selected_by_validation']['name'] == 'control'  # Ties retain first arm.
    assert runner.read_json(root / 'selection.json')['heldout_used_for_selection'] is False
    assert result['comparison']['heldout_average_difference'] == 0
    assert result['no_automatic_promotion']
    assert result['saved_evidence']['complete_replacement_audits']
    for arm in result['policies'].values():
        for record in arm.values():
            assert set(record['checks']) == set(runner.CONDITIONS)
            assert all(check['replacement_assistance'] is False for check in record['checks'].values())
    runner.collect(root, plan)
    assert 'Heldout original' in (root / 'presentation-summary.md').read_text()
    assert 'no Rules assistance' in (root / 'report.md').read_text()
    with ZipFile(root / 'bench-order-results.zip') as archive:
        assert 'presentation-summary.md' in archive.namelist()
        assert not any(name.endswith('.zip') or name.startswith('console/') for name in archive.namelist())


def test_finish_rejects_wrong_selection_denominator(prepared):
    root, plan = prepared
    complete_training_fixture(root, plan)
    path = root / 'runs/control/checkpoints/best.json'
    meta = runner.read_json(path); meta['requested'] -= 1; write(path, meta)
    with pytest.raises(ValueError, match='balanced validation'):
        runner.finish(root, plan, {})


def test_finish_rejects_unmatched_budget(prepared, monkeypatch):
    root, plan = prepared
    install_fake_simulator(monkeypatch)
    initial = runner.evaluate_initial(root, plan)
    complete_training_fixture(root, plan)
    path = root / 'runs/control/summary.json'
    summary = runner.read_json(path); summary['stop_reason'] = 'validation_success'; write(path, summary)
    with pytest.raises(ValueError, match='fixed timestep budget'):
        runner.finish(root, plan, initial)


def test_saved_evidence_gate_rejects_lost_episode_rows(prepared, monkeypatch):
    root, plan = prepared; install_fake_simulator(monkeypatch)
    initial = runner.evaluate_initial(root, plan)
    runner.validate_saved_checks(root, [initial])
    directory = root / initial['checks']['heldout-original']['result_directory']
    (directory / 'episodes.jsonl').write_text('')
    with pytest.raises(RuntimeError, match='incomplete'):
        runner.validate_saved_checks(root, [initial])


@pytest.mark.parametrize('change', ['flag', 'source_hash', 'missing_counts', 'scope', 'counts', 'no_change'])
def test_actual_training_provenance_must_match_treatment(prepared, change):
    root, plan = prepared; complete_training_fixture(root, plan)
    job = plan['jobs'][1]; run = root / 'runs' / job['name']
    config = runner.read_json(run / 'config.json'); summary = runner.read_json(run / 'summary.json')
    if change == 'flag': config['shuffle_bench_order'] = False
    elif change == 'source_hash': config['resume_checkpoint']['sha256'] = 'changed'
    elif change == 'missing_counts': summary.pop('bench_order_shuffle')
    elif change == 'scope': summary['bench_order_shuffle']['scope'] = 'both_players'
    elif change == 'counts': summary['bench_order_shuffle']['counts']['eligible'] += 1
    elif change == 'no_change': summary['bench_order_shuffle']['counts'].update(changed=0, identity=2)
    write(run / 'config.json', config)
    with pytest.raises(ValueError): runner.training_provenance(root, plan, job, summary)


def test_resume_checks_first_session_source_and_aggregates_shuffle_counts(prepared):
    root, plan = prepared; complete_training_fixture(root, plan)
    job = plan['jobs'][1]; run = root / 'runs' / job['name']
    first = runner.read_json(run / 'config.json'); previous = runner.read_json(run / 'summary.json')
    write(run / 'sessions/000001/config.json', first)
    write(run / 'sessions/000001/summary.json', previous)
    current = copy.deepcopy(previous); current['session_id'] = 2
    current['bench_order_shuffle']['counts'].update(changed=0, identity=2)
    first['session']['id'] = 2; first['resume_checkpoint']['sha256'] = 'latest-checkpoint'
    write(run / 'config.json', first)
    provenance = runner.training_provenance(root, plan, job, current)
    assert provenance['sessions'] == 2
    assert provenance['aggregate_episode_start_counts']['changed'] == 1
    assert provenance['aggregate_episode_start_counts']['eligible'] == 4


def test_point_comparison_reports_tradeoffs_without_promotion():
    control = {condition: {'wins_per_requested': .4} for condition in runner.CONDITIONS}
    treatment = copy.deepcopy(control)
    treatment['heldout-original']['wins_per_requested'] = .5
    treatment['heldout-reversed']['wins_per_requested'] = .36
    treatment['mirror']['wins_per_requested'] = .3
    policies = dict(control=dict(best=dict(checks=control)), **{'bench-shuffle': dict(best=dict(checks=treatment))})
    result = runner.comparison(policies, dict(mirror_tradeoff=.05, single_order_tradeoff=.03, practical_gain_goal=.05))
    assert result['heldout_average_difference'] == pytest.approx(.03)
    assert len(result['tradeoffs']) == 2 and result['practical_goal_met'] is False
    assert result['no_automatic_promotion']


def test_launch_caps_jobs_and_reuses_completed_training(prepared, monkeypatch):
    root, plan = prepared
    monkeypatch.setattr(runner, 'command', lambda *args: None)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('No completed run should retrain'))
    runner.launch(root, plan, 2)
    with pytest.raises(ValueError, match='one or two'): runner.launch(root, plan, 3)


def test_inspector_includes_source_moves_and_mixture(prepared, capsys):
    root, plan = prepared
    runner.inspect(root, plan, 'validation', 1, True)
    output = capsys.readouterr().out
    assert 'Canonical training mixture' in output and 'Surf' in output and 'Blastoise' in output
