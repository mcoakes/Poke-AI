"""Ablation provenance and arm controls, independent from policy mechanics tests."""
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
spec = importlib.util.spec_from_file_location('matchup_ablation_tests', ROOT / 'run_matchup_ablation.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
from test_transfer_lab import checkpoint, write, settings


def source_fixture(tmp_path, smoke=False):
    root = tmp_path / 'source'
    checkpoint(root / 'inputs/fresh-initial.zip', 0)
    checkpoint(root / 'inputs/foundation-mirror.zip', 253952)
    write(root / 'inputs/reward.json', {'win': 5000, 'loss': -5000})
    config = settings(); write(root / 'inputs/settings.json', config)
    teams = {}
    for name in ('train', 'mirror'):
        teams[name] = []
        for i in range(2):
            path = write(root / 'inputs/teams' / name / f'team-{i}.json', {'party': []})
            teams[name].append(str(path.relative_to(root)))
    suites = {}
    for name in runner.CONDITIONS:
        path = write(root / 'inputs/suites' / f'{name}.json', {'assets': {}, 'cases': []})
        suites[name] = str(path.relative_to(root))
    plan = dict(version=1, experiment='initialization-transfer', settings=config, teams=teams,
        suites=suites, smoke=smoke, pins={str(p.relative_to(root)): runner.checksum(p)
            for p in (root / 'inputs').rglob('*') if p.is_file()})
    write(root / 'plan.json', plan)
    return root


def saved_fixture(tmp_path):
    root = tmp_path / 'run'
    config = settings(); config['seed'] = 261090
    write(root / 'inputs/settings.json', config)
    checkpoint(root / 'inputs/foundation-mirror.zip', 253952)
    for arm in ('control', 'matchup'): checkpoint(root / 'inputs' / f'{arm}-initial.zip', 0)
    teams = {}
    for name in ('train', 'mirror'):
        path = write(root / 'inputs/teams' / name / 'team.json', {'party': []})
        teams[name] = [str(path.relative_to(root))]
    identity = dict(state_tensors_equal=True, initial_timesteps=0, optimizers_empty=True)
    write(root / 'inputs/initial-identity.json', identity)
    plan = dict(version=1, experiment='matchup-feature-ablation', settings=config, teams=teams,
        suites={'validation': 'inputs/suites/validation.json'}, jobs=runner.make_jobs(config), smoke=False,
        initial_identity=identity, foundation_reference='inputs/foundation-mirror.zip',
        pins={str(p.relative_to(root)): runner.checksum(p) for p in (root / 'inputs').rglob('*') if p.is_file()})
    write(root / 'plan.json', plan)
    return root, plan


def test_source_read_preserves_inputs_and_original_fresh_initialization(tmp_path):
    root = source_fixture(tmp_path)
    before = {str(p.relative_to(root)): runner.checksum(p) for p in root.rglob('*') if p.is_file()}
    result = runner.source_inputs(root)
    assert result['fresh'] == root / 'inputs/fresh-initial.zip'
    assert result['foundation'] == root / 'inputs/foundation-mirror.zip'
    assert set(result['suites']) == set(runner.CONDITIONS)
    assert before == {str(p.relative_to(root)): runner.checksum(p) for p in root.rglob('*') if p.is_file()}


def test_reject_missing_pinned_foundation_instead_of_weakening_retention(tmp_path):
    root = source_fixture(tmp_path)
    (root / 'inputs/foundation-mirror.zip').unlink()
    with pytest.raises(ValueError, match='pinned input changed'):
        runner.source_inputs(root)


def test_reject_smoke_source_and_source_mutation(tmp_path):
    root = source_fixture(tmp_path, smoke=True)
    with pytest.raises(ValueError, match='smoke source'):
        runner.source_inputs(root)
    runner.source_inputs(root, smoke=True)
    (root / 'inputs/reward.json').write_text('{}')
    with pytest.raises(ValueError, match='pinned input changed'):
        runner.source_inputs(root, smoke=True)


def test_reject_source_team_membership_change(tmp_path):
    root = source_fixture(tmp_path)
    write(root / 'inputs/teams/train/extra.json', {})
    with pytest.raises(ValueError, match='membership changed'):
        runner.source_inputs(root)


def test_both_arms_same_command_except_checkpoint_and_output(tmp_path):
    root, plan = saved_fixture(tmp_path)
    commands = [runner.command(root, plan, job) for job in plan['jobs']]
    normalized = []
    for command in commands:
        assert command[command.index('--policy-architecture') + 1] == 'matchup-moves'
        assert command[command.index('--seed') + 1] == '261090'
        assert command[command.index('--total-timesteps') + 1] == '1000000'
        assert command[command.index('--eval-every') + 1] == '50000'
        assert command[command.index('--eval-stop-win-rate') + 1] == '0.65'
        item = list(command)
        for flag in ('--resume', '--output-dir'): item[item.index(flag)+1] = '<arm-path>'
        normalized.append(item)
    assert normalized[0] == normalized[1]
    assert [job['relational_enabled'] for job in plan['jobs']] == [False, True]
    assert {job['starting_timesteps'] for job in plan['jobs']} == {0}


def test_resume_remaining_steps_and_completed_run_is_idempotent(tmp_path):
    root, plan = saved_fixture(tmp_path); job = plan['jobs'][0]
    run = root / 'runs/control'
    checkpoint(run / 'checkpoints/latest.zip', 200704)
    write(run / 'summary.json', {'status': 'interrupted'})
    command = runner.command(root, plan, job)
    assert command[-1] == str(1000000-200704)
    assert '--resume' not in command and '--policy-architecture' not in command
    write(run / 'summary.json', {'status': 'completed', 'stop_reason': 'validation_success'})
    assert runner.command(root, plan, job) is None
    checkpoint(run / 'checkpoints/latest.zip', 1000448)
    write(run / 'summary.json', {'status': 'completed', 'stop_reason': 'timestep_budget'})
    assert runner.command(root, plan, job) is None


def test_verify_rejects_changed_treatment_and_does_not_drop_reference(tmp_path):
    root, plan = saved_fixture(tmp_path)
    runner.verify(root, plan)
    plan['jobs'][0]['relational_enabled'] = True
    with pytest.raises(ValueError, match='budgets changed'): runner.verify(root, plan)
    plan['jobs'] = runner.make_jobs(plan['settings'])
    plan['foundation_reference'] = None
    with pytest.raises(ValueError, match='retention reference'): runner.verify(root, plan)


def test_collection_keeps_metrics_and_identity_but_not_weights(tmp_path):
    root, plan = saved_fixture(tmp_path)
    write(root / 'runs/control/config.json', {'policy_architecture': 'matchup-moves'})
    write(root / 'runs/control/updates.jsonl', {'metrics': {}})
    write(root / 'checks/example/decision-audit.json', {'totals': {}})
    runner.collect(root, plan)
    with ZipFile(root / 'matchup-results.zip') as archive:
        assert archive.testzip() is None
        assert 'inputs/initial-identity.json' in archive.namelist()
        assert 'runs/control/updates.jsonl' in archive.namelist()
        assert 'checks/example/decision-audit.json' in archive.namelist()
        assert 'inputs/control-initial.zip' not in archive.namelist()
    assert 'control' in (root / 'report.md').read_text()
    assert 'N/A' in (root / 'report.md').read_text()


def test_prepare_only_does_not_start_training_or_evaluation(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    root, plan = saved_fixture(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['run_matchup_ablation.py', '--output-dir', str(root), '--prepare-only'])
    monkeypatch.setattr(runner, 'freeze', lambda *args: plan)
    def forbidden(*args): raise AssertionError('prepare-only executed work')
    monkeypatch.setattr(runner, 'evaluate', forbidden)
    monkeypatch.setattr(runner, 'launch', forbidden)
    assert runner.main() == 0
    assert runner.read_json(root / 'status.json')['status'] == 'prepared'


def test_real_migration_identity_and_architecture_detection(tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(1)
    pytest.importorskip('sb3_contrib')
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import policy_space
    from firered_rl.matchup_move_policy import create_ablation_checkpoint
    from firered_rl.shared_move_policy import policy_architecture
    template = SimpleNamespace(observation_space=policy_space('v2'), action_space=gym.spaces.Discrete(11), reward_settings={'scale': .001})
    original = tmp_path / 'original.zip'
    runner.transfer.fresh_checkpoint(template, original, settings())
    model = MaskablePPO.load(original, device='cpu')
    assert policy_architecture(model) == 'shared-moves'
    before = runner.checksum(original)
    arms = [create_ablation_checkpoint(model, tmp_path / f'{enabled}.zip', enabled=enabled, seed=261090)
            for enabled in (False, True)]
    evidence = runner.initial_identity(arms)
    assert evidence['state_tensors_equal'] and evidence['optimizers_empty']
    assert evidence['max_logit_difference'] <= 1e-7
    assert evidence['max_probability_difference'] <= 1e-7
    assert all(policy_architecture(arm) == 'matchup-moves' for arm in arms)
    assert before == runner.checksum(original)
    arms[1].policy.features_extractor.relational_enabled = False
    with pytest.raises(ValueError, match='features off'): runner.initial_identity(arms)


def saved_check(root):
    summary = dict(status='completed', result_directory='checks/accepted',
        candidate_identity={'kind': 'checkpoint', 'sha256': 'candidate'},
        opponent_identity={'kind': 'rules'}, input_suite_sha256='input', effective_suite_sha256='effective',
        reward_sha256='reward', suite_sha256='suite', requested_battles=2, attempted_battles=2,
        completed_battles=2, truncated=0, candidate_wins=1, opponent_wins=1, draws=0,
        audit_totals={'move_actions': 12})
    directory = root / summary['result_directory']
    write(directory / 'summary.json', summary)
    (directory / 'episodes.jsonl').write_text('\n'.join(json.dumps(dict(episode=i+1,
        candidate_outcome=outcome, truncated=False)) for i, outcome in enumerate(('won', 'lost'))) + '\n')
    return summary


def test_saved_evidence_gate_accepts_complete_referenced_checks(tmp_path):
    summary = saved_check(tmp_path)
    result = dict(initial={'control': {'heldout': summary}}, policies={'control': {'best': {'checks': {'heldout': summary}}}})
    proof = runner.verify_saved_checks(tmp_path, result)
    assert proof == dict(checked_directories=['checks/accepted'], complete_episode_logs=True)


def test_saved_evidence_gate_rejects_partial_missing_mismatched_or_outside_records(tmp_path):
    summary = saved_check(tmp_path)
    result = dict(initial={'control': {'heldout': summary}})
    log = tmp_path / 'checks/accepted/episodes.jsonl'
    original = log.read_text()
    log.write_text(original.splitlines()[0] + '\n')
    with pytest.raises(RuntimeError, match='--continue'): runner.verify_saved_checks(tmp_path, result)
    log.unlink()
    with pytest.raises(RuntimeError, match='--continue'): runner.verify_saved_checks(tmp_path, result)
    log.write_text(original)
    changed = dict(summary, candidate_identity={'kind': 'checkpoint', 'sha256': 'changed'})
    write(tmp_path / 'checks/accepted/summary.json', changed)
    with pytest.raises(RuntimeError, match='inconsistent'): runner.verify_saved_checks(tmp_path, result)
    write(tmp_path / 'checks/accepted/summary.json', dict(summary, status='running'))
    with pytest.raises(RuntimeError, match='incomplete'): runner.verify_saved_checks(tmp_path, result)
    summary['result_directory'] = '../outside'
    with pytest.raises(RuntimeError, match='--continue'): runner.verify_saved_checks(tmp_path, result)
