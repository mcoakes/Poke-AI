#!/usr/bin/env python3
"""Gated Rules-only experiments: fixed tasks, stability, order, then composition."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_fixed_team_lab import checksum, write_json, interrupt
from run_parallel_training import checkpoint_steps, recovered_steps, pump

DEFAULT_CONFIG = {
    'version': 1, 'seed': 261040, 'environments': 16,
    'learning_rates': [0.0001, 0.00002], 'entropy': 0.01,
    'eval_every': 5000, 'validation_battles': 256, 'test_battles': 1000,
    'calibration_battles': 200, 'consecutive_successes': 2,
    'budgets': {'move-choice': 50000, 'mirror-stability': 100000,
                'mirror-replication': 200000, 'order': 300000, 'composition': 400000},
    'thresholds': {'move-choice': .99, 'mirror-stability': .90,
                   'mirror-replication': .90, 'order': .70, 'composition': .60},
    'heldout_threshold': .55, 'maximum_rotation_drop': .03,
    'maximum_stability_drop': .05,
}
STAGE_SCENARIOS = {'move-choice': 'move-choice', 'mirror-stability': 'mirror',
                   'mirror-replication': 'mirror', 'order': 'order', 'composition': 'composition'}


def read_config(path=None, smoke=False):
    config = json.loads(json.dumps(DEFAULT_CONFIG)) if path is None else json.loads(Path(path).read_text())
    if set(config) != set(DEFAULT_CONFIG) or config.get('version') != 1:
        raise ValueError('Configuration must contain exactly the fields in config/experiments/staged-team-lab.json.')
    for name in ('seed', 'environments', 'eval_every', 'validation_battles', 'test_battles',
                 'calibration_battles', 'consecutive_successes'):
        if type(config[name]) is not int or config[name] < (0 if name == 'seed' else 1):
            raise ValueError(f'Invalid {name}')
    if config['seed'] >= 2**31-100 or config['environments'] % 2:
        raise ValueError('Use a seed below 2**31-100 and an even environment count.')
    for key in ('validation_battles', 'test_battles', 'calibration_battles'):
        if config[key] % 2:
            raise ValueError(f'{key} must be even.')
    rates = config['learning_rates']
    if not isinstance(rates, list) or len(rates) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in rates):
        raise ValueError('Supply two positive finite learning rates.')
    if set(config['budgets']) != set(STAGE_SCENARIOS) or set(config['thresholds']) != set(STAGE_SCENARIOS):
        raise ValueError('Budgets and thresholds must list all five stages.')
    if any(type(v) is not int or v <= 0 for v in config['budgets'].values()):
        raise ValueError('Budgets must be positive integers.')
    for value in [*config['thresholds'].values(), config['heldout_threshold'], config['maximum_rotation_drop'],
                  config['maximum_stability_drop'], config['entropy']]:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Thresholds, drops and entropy must be finite and in [0,1].')
    if smoke:
        config.update(environments=2, eval_every=128, validation_battles=8,
                      test_battles=8, calibration_battles=8)
        config['budgets'] = {name: 256 for name in STAGE_SCENARIOS}
    return config


def verify_pins(root, pins):
    for name, digest in pins.items():
        if checksum(root / name) != digest:
            raise ValueError(f'Pinned input changed: {name}')


def freeze(root, args):
    from firered_rl.staged_team_lab import write_assets, make_stage_suite
    from firered_rl.showdown_env import validate_showdown
    from firered_rl.linked_reward import LinkedRewardConfig
    from dataclasses import asdict
    validate_showdown(ROOT)
    config = read_config(args.config, args.smoke)
    source = None if args.fresh_start else args.source_lab.expanduser().resolve()
    if source is not None and not (source / 'mirror-fresh/checkpoints/best.zip').is_file():
        raise ValueError(f'Missing saved mirror best in {source}. Use --fresh-start for a new curriculum.')
    reward_path = args.reward_config or (source / 'inputs/reward.json' if source else ROOT / 'config/rewards/damage_knockout_win.json')
    reward = LinkedRewardConfig.load(reward_path)
    imported_sources = {}
    if source:
        from sb3_contrib import MaskablePPO
        from firered_rl.shared_move_policy import SharedMovePolicy
        # Check all source models before creating a destination directory.
        for scenario in ('move-choice', 'mirror'):
            original = source / f'{scenario}-fresh/checkpoints/best.zip'
            meta = json.loads(original.with_suffix('.json').read_text())
            if checksum(original) != meta['checkpoint_sha256']:
                raise ValueError(f'Saved best hash disagrees with metadata: {original}')
            model = MaskablePPO.load(original, device='cpu')
            if not isinstance(model.policy, SharedMovePolicy):
                raise ValueError('Imported checkpoint must use shared-moves.')
            del model
            imported_sources[scenario] = (original, meta)
        final = source / 'mirror-fresh/checkpoints/final.zip'
        if final.is_file():
            model = MaskablePPO.load(final, device='cpu')
            if not isinstance(model.policy, SharedMovePolicy):
                raise ValueError('Imported final checkpoint must use shared-moves.')
            del model
            recorded = source / 'checks/mirror-fresh-final-test/summary.json'
            if recorded.is_file() and json.loads(recorded.read_text()).get('checkpoint_sha256') != checksum(final):
                raise ValueError('Imported mirror final disagrees with its recorded test hash')
            imported_sources['mirror-final'] = (final, None)
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs'
    inputs.mkdir()
    (root / 'console').mkdir()
    write_json(inputs / 'reward.json', asdict(reward))
    assets = write_assets(inputs)
    imports = {}
    for scenario, (original, meta) in imported_sources.items():
        target = inputs / f'imported-{scenario}.zip'
        shutil.copyfile(original, target)
        if meta is not None:
            write_json(inputs / f'imported-{scenario}-best.json', meta)
        imports[scenario] = str(target.relative_to(root))
    if source:
        # Keep the old collapse's update history; no old battle/console bulk logs.
        for path in (source / 'mirror-fresh').rglob('*'):
            if path.is_file() and path.name in ('updates.jsonl', 'battle-blocks.jsonl', 'progress.csv',
                    'summary.json', 'config.json', 'evaluation-history.json', 'evaluations.md', 'best.json'):
                relative = path.relative_to(source / 'mirror-fresh')
                target = inputs / 'source-mirror-diagnostics' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    stages = ['move-choice', 'mirror-stability', 'mirror-replication', 'order', 'composition'] if source else [
        'move-choice', 'mirror-replication', 'mirror-stability', 'order', 'composition']
    suites = {}
    # Seeds are fixed, disjoint by purpose, and independent of training seeds.
    for i, stage in enumerate(STAGE_SCENARIOS):
        scenario = STAGE_SCENARIOS[stage]
        suites[stage] = {}
        for label, offset, count, permuted, heldout in [
            ('calibration', 0, config['calibration_battles'], False, False),
            ('validation', 1, config['validation_battles'], False, False),
            ('test', 2, config['test_battles'], False, False),
            ('permuted', 2, config['test_battles'], True, False),
            *([('heldout', 3, config['test_battles'], False, True)] if stage == 'composition' else [])]:
            suite = make_stage_suite(ROOT, assets[scenario], count, 842000 + i*10 + offset,
                                     scenario, permute_moves=permuted, heldout=heldout)
            path = inputs / f'{stage}-{label}.json'
            write_json(path, suite)
            suites[stage][label] = str(path.relative_to(root))
    plan = dict(version=1, smoke=args.smoke, config=config, assets=assets, imports=imports,
                stages=stages, suites=suites, source_lab=str(source) if source else None,
                pins={str(p.relative_to(root)): checksum(p) for p in inputs.rglob('*') if p.is_file()})
    write_json(root / 'plan.json', plan)
    # Precreate all queued log files for a stable tail glob.
    for stage in stages:
        for suffix in ('a', 'b'):
            (root / 'console' / f'{stage}-{suffix}.log').touch()
    return plan


def training_command(root, plan, job):
    run = root / 'runs' / job['name']
    base = [sys.executable, '-u', str(ROOT / 'mgba/train_ppo.py')]
    if run.exists():
        summary_path = run / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summary.get('status') == 'completed' and (
                summary.get('stop_reason') == 'validation_success' or recovered_steps(run) >= job['target_timesteps']):
            return None
        remaining = job['target_timesteps'] - recovered_steps(run)
        return base + ['--output-dir', str(run), '--if-exists', 'continue', '--total-timesteps', str(max(1, remaining))]
    c = plan['config']
    entry = plan['assets'][STAGE_SCENARIOS[job['stage']]]
    team_args = (['--teams', *[str(Path(entry['root']) / p) for p in entry['train_paths']]]
                 if entry['fixed'] else ['--team-dir', str(Path(entry['root']) / entry['train_dir'])])
    resume = ['--resume', str(root / job['checkpoint'])] if job.get('checkpoint') else []
    stop = [] if job['stage'] == 'mirror-stability' else [
        '--eval-stop-win-rate', str(c['thresholds'][job['stage']]),
        '--eval-stop-consecutive', str(c['consecutive_successes'])]
    return base + resume + team_args + stop + [
        '--backend', 'showdown', '--policy-architecture', 'shared-moves', '--observation-version', 'v2',
        '--rule-based-opponent', '--reward-config', str(root / 'inputs/reward.json'),
        '--learning-rate', str(job['learning_rate']), '--ent-coef', str(c['entropy']),
        '--rollout-steps', '128', '--batch-size', '128', '--epochs', '4',
        '--seed', str(job['seed']), '--environments', str(c['environments']),
        '--showdown-workers', 'process', '--device', 'cpu', '--torch-threads', '1',
        '--max-steps', '100', '--total-timesteps', str(c['budgets'][job['stage']]),
        '--eval-suite', str(root / plan['suites'][job['stage']]['validation']),
        '--eval-every', str(c['eval_every']), '--eval-environments', '4', '--keep-evaluations', '2',
        '--latest-every-updates', '1', '--checkpoint-every', str(c['eval_every']), '--keep-checkpoints', '3',
        '--battle-log', '--no-readable-log', '--no-step-logs', '--log-max-mb', '2', '--log-backups', '1',
        '--output-dir', str(run), '--if-exists', 'cancel']


def launch(root, plan, jobs, concurrency):
    pending, active = list(jobs), []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        while pending or active:
            while pending and len(active) < concurrency:
                job = pending.pop(0)
                command = training_command(root, plan, job)
                if command is None:
                    print(f"Already completed: {job['name']}", flush=True)
                    continue
                print(f"Starting {job['name']}\n{shlex.join(command)}", flush=True)
                process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                reader = threading.Thread(target=pump, args=(process, root / 'console' / f"{job['name']}.log"))
                reader.start()
                active.append((job, process, reader))
            for row in list(active):
                job, process, reader = row
                code = process.poll()
                if code is None:
                    continue
                reader.join()
                active.remove(row)
                if code:
                    raise RuntimeError(f"{job['name']} exited {code}; see console/{job['name']}.log")
                summary = json.loads((root / 'runs' / job['name'] / 'summary.json').read_text())
                if summary.get('status') != 'completed':
                    raise RuntimeError(f"{job['name']} stopped: {summary.get('status')}")
            if active:
                time.sleep(.25)
    finally:
        for _, process, _ in active:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        if active:
            print('Waiting for trainer cleanup; keep this terminal open.', flush=True)
        for _, process, reader in active:
            process.wait()
            reader.join()


def evaluate_checkpoint(root, plan, label, suite_path, checkpoint=None, baseline=None):
    from firered_rl.staged_team_lab import score_stage
    from firered_rl.fixed_team_lab import GreedyDamage, RuleBasedOpponent
    from sb3_contrib import MaskablePPO
    suite_sha = checksum(suite_path)
    identity = checksum(checkpoint) if checkpoint else baseline
    base = root / 'checks' / label
    # Completed retry directories also count when resuming.
    for directory in [base, *sorted(base.parent.glob(base.name + '-retry-*'))]:
        path = directory / 'summary.json'
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get('status') == 'completed' and saved.get('input_suite_sha256') == suite_sha and saved.get('candidate_identity') == identity:
                return saved
    run = base if not base.exists() else base.with_name(base.name + f'-retry-{time.time_ns()}')
    if checkpoint:
        model = MaskablePPO.load(checkpoint, device='cpu')
        model.policy.set_training_mode(False)
    else:
        model = {'random': lambda: None, 'greedy': lambda: GreedyDamage(ROOT),
                 'rules': lambda: RuleBasedOpponent(ROOT)}[baseline]()
    summary = score_stage(ROOT, run, json.loads(suite_path.read_text()), model,
                          root / 'inputs/reward.json', pairs=2, trace_cases=16, trace_decisions=24)
    summary.update(candidate_identity=identity, input_suite_sha256=suite_sha,
                   checkpoint=str(checkpoint) if checkpoint else None, result_directory=str(run.relative_to(root)))
    write_json(run / 'summary.json', summary)
    return summary


def passing(result, threshold):
    return (result.get('status') == 'completed' and result['truncated'] == 0
            and result['completed_battles'] == result['requested_battles']
            and result['candidate_wins'] / result['requested_battles'] >= threshold)


def check_gate(stage, checks, config, initial_rate=None):
    threshold = config['thresholds'][stage]
    failures = []
    for condition in ('test', 'permuted'):
        if not passing(checks[condition], threshold):
            failures.append(f'{condition} below {threshold:.0%} or incomplete/truncated')
    if checks['test']['candidate_win_rate'] - checks['permuted']['candidate_win_rate'] > config['maximum_rotation_drop'] + 1e-12:
        failures.append('move-slot rotation drop exceeds limit')
    if stage == 'move-choice' and any(checks[k]['first_surf_rate'] < threshold for k in ('test', 'permuted')):
        failures.append('first-turn Surf below success threshold')
    if stage == 'mirror-stability' and initial_rate is not None and checks['test']['candidate_win_rate'] < initial_rate - config['maximum_stability_drop'] - 1e-12:
        failures.append('final policy lost more than the allowed win-rate margin from its frozen start')
    if stage == 'composition' and not passing(checks['heldout'], config['heldout_threshold']):
        failures.append('unseen team compositions below heldout threshold or incomplete/truncated')
    return failures


def pin_checkpoint(root, source, target):
    path = root / target
    digest = checksum(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if checksum(path) != digest:
            raise ValueError(f'Pinned stage checkpoint changed: {path}')
    else:
        shutil.copyfile(source, path)
    return str(path.relative_to(root)), digest


def prepare_jobs(root, plan, stage, anchor):
    saved = root / 'stages' / stage / 'jobs.json'
    if saved.exists():
        data = json.loads(saved.read_text())
        verify_pins(root, data['pins'])
        return data['jobs']
    saved.parent.mkdir(parents=True, exist_ok=True)
    pins, checkpoint = {}, None
    if anchor and stage != 'mirror-replication':
        checkpoint, digest = pin_checkpoint(root, root / anchor, f'inputs/stage-starts/{stage}.zip')
        pins[checkpoint] = digest
    c = plan['config']
    rates = [c['learning_rates'][0]] if stage == 'move-choice' else c['learning_rates']
    if stage == 'mirror-replication':
        rates = [c['learning_rates'][0]] * 2
    jobs = []
    for i, rate in enumerate(rates):
        start = checkpoint_steps(root / checkpoint) if checkpoint else 0
        # Replications differ in seed. Configuration comparisons share a seed and starting weights/optimizer.
        seed = c['seed'] + (i if stage == 'mirror-replication' else 10 + list(STAGE_SCENARIOS).index(stage))
        jobs.append(dict(name=f"{stage}-{'ab'[i]}", stage=stage, learning_rate=rate, seed=seed,
                         checkpoint=checkpoint, starting_timesteps=start,
                         target_timesteps=start+c['budgets'][stage]))
    write_json(saved, dict(jobs=jobs, pins=pins))
    return jobs


def run_stage(root, plan, stage, anchor, concurrency):
    from firered_rl.staged_team_lab import validate_assets
    for entry in plan['assets'].values():
        validate_assets(entry)
    c = plan['config']
    stage_dir = root / 'stages' / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    suites = {k: root / v for k, v in plan['suites'][stage].items()}
    print(f'\nStage: {stage}', flush=True)
    for baseline in ('random', 'greedy', 'rules'):
        evaluate_checkpoint(root, plan, f'{stage}-baseline-{baseline}', suites['calibration'], baseline=baseline)
    if stage == 'mirror-stability' and plan['imports'].get('mirror-final'):
        for label in ('mirror', 'mirror-final'):
            evaluate_checkpoint(root, plan, f'original-{label}-test', suites['test'], root / plan['imports'][label])
    if stage == 'move-choice' and plan['imports'].get('move-choice'):
        selected = root / plan['imports']['move-choice']
        selected_name = 'imported-move-choice'
        candidates = []
    else:
        jobs = prepare_jobs(root, plan, stage, anchor)
        launch(root, plan, jobs, concurrency)
        candidates = []
        for job in jobs:
            run = root / 'runs' / job['name']
            history = json.loads((run / 'evaluation-history.json').read_text())
            # Native evaluation history is a list of completed records.
            records = history if isinstance(history, list) else history['records']
            key = 'final' if stage == 'mirror-stability' else 'best'
            if key == 'best':
                record = json.loads((run / 'checkpoints/best.json').read_text())
            else:
                steps = checkpoint_steps(run / 'checkpoints/final.zip')
                record = next(r for r in reversed(records) if r['status'] == 'completed' and r['timesteps'] == steps)
            candidates.append(dict(name=job['name'], checkpoint=run / 'checkpoints' / f'{key}.zip',
                                   validation_rate=record['win_rate']))
        # Choose using validation ONLY, before looking at test/heldout results.
        winner = max(candidates, key=lambda row: row['validation_rate'])
        selected, selected_name = winner['checkpoint'], winner['name']
    write_json(stage_dir / 'selection.json', dict(name=selected_name, checkpoint=str(selected),
        checkpoint_sha256=checksum(selected), criterion='final validation for stability; best validation otherwise',
        candidates=[{**r, 'checkpoint': str(r['checkpoint'])} for r in candidates]))
    initial_rate = None
    if stage == 'mirror-stability':
        start_result = evaluate_checkpoint(root, plan, f'{stage}-start-test', suites['test'], root / anchor)
        initial_rate = start_result['candidate_win_rate']
    all_checks = {}
    to_check = [(selected_name, selected)]
    for row in candidates:
        run = root / 'runs' / row['name']
        to_check += [(row['name']+'-'+key, run / 'checkpoints' / f'{key}.zip') for key in ('best', 'final')]
    cache = {}
    for name, path in to_check:
        identity = checksum(path)
        if identity in cache:
            all_checks[name] = cache[identity]
            continue
        values = {}
        for label in ('test', 'permuted', *(['heldout'] if stage == 'composition' else [])):
            values[label] = evaluate_checkpoint(root, plan, f'{name}-{label}', suites[label], path)
        cache[identity] = values
        all_checks[name] = values
    checks = all_checks[selected_name]
    failures = check_gate(stage, checks, c, initial_rate)
    if stage == 'mirror-replication':
        for row in candidates:
            failures += [row['name']+': '+s for s in check_gate(stage, cache[checksum(row['checkpoint'])], c)]
    if stage in ('order', 'composition'):
        # Forgetting checks are diagnostic; they do not select the winning configuration.
        evaluate_checkpoint(root, plan, f'{stage}-selected-mirror-retention',
                            root / plan['suites']['mirror-replication']['test'], selected)
    status = 'smoke_completed' if plan['smoke'] else ('blocked' if failures else 'passed')
    selected_relative, digest = pin_checkpoint(root, selected, f'inputs/selected/{stage}.zip')
    result = dict(stage=stage, status=status, failures=failures, selected=selected_relative,
                  checkpoint_sha256=digest, selected_run=selected_name, checks=checks,
                  initial_test_rate=initial_rate, completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(stage_dir / 'result.json', result)
    print(f"Stage {stage}: {status}; selected {selected_name}; {'; '.join(failures) or 'all gates passed'}", flush=True)
    return result


def memory_monitor(root, stopped):
    path = root / 'memory.csv'
    with path.open('a', buffering=1) as stream:
        writer = csv.writer(stream)
        if path.stat().st_size == 0:
            writer.writerow(['unix_seconds', 'mem_total_kib', 'mem_available_kib', 'swap_total_kib', 'swap_free_kib'])
        while not stopped.is_set():
            try:
                values = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
                writer.writerow([time.time(), *[values[k] for k in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')]])
            except (OSError, ValueError, KeyError):
                return
            stopped.wait(2)


def collect(root, plan):
    lines = ['# Staged team lab', '', 'SMOKE TEST ONLY: no learning claims or promotion gates apply.' if plan['smoke'] else
             'All battles are against Rules. Selection uses validation; frozen checks gate progression.', '',
             '| Stage | Status | Selected | Test | Rotated | Heldout |', '|---|---|---|---:|---:|---:|']
    gate_notes = []
    for stage in plan['stages']:
        path = root / 'stages' / stage / 'result.json'
        if not path.exists():
            lines.append(f'| {stage} | pending | — | — | — | — |')
            continue
        r = json.loads(path.read_text())
        rates = [f"{r['checks'][k]['candidate_win_rate']:.1%}" if k in r['checks'] else '—' for k in ('test','permuted','heldout')]
        lines.append(f"| {stage} | {r['status']} | {r['selected_run']} | {' | '.join(rates)} |")
        for reason in r['failures']:
            gate_notes.append(f'- {stage}: {reason}')
    lines += ['', *gate_notes, '', '## Frozen check results', '',
              '| Check | Wins/requested | Win rate | Truncated |', '|---|---:|---:|---:|']
    for path in sorted((root / 'checks').glob('*/summary.json')):
        row = json.loads(path.read_text())
        if row.get('status') == 'completed':
            lines.append(f"| {path.parent.name} | {row['candidate_wins']}/{row['requested_battles']} | {row['candidate_win_rate']:.1%} | {row['truncated']} |")
    lines += ['', 'Each job has console logs, evaluations.md, updates.jsonl, and PPO CSV metrics.',
              'checks/* contains frozen summaries, bounded decision traces/audits, and episode results.',
              'memory.csv records machine-wide available RAM and swap, including validation peaks.']
    (root / 'report.md').write_text('\n'.join(lines)+'\n')
    temp = root / 'staged-team-results.tmp'
    with ZipFile(temp, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            rel = path.relative_to(root)
            if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.md', '.csv'):
                continue
            if rel.parts[0] in ('inputs', 'checks', 'stages') or path.name in (
                'plan.json', 'report.md', 'status.json', 'memory.csv', 'config.json', 'summary.json', 'failure.json',
                'best.json', 'evaluations.md', 'evaluations.csv', 'evaluation-history.json', 'updates.jsonl',
                'battle-blocks.jsonl', 'progress.csv', 'switching.json'):
                archive.write(path, rel)
    temp.replace(root / 'staged-team-results.zip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('.local/run/staged-team-lab-01'))
    parser.add_argument('--source-lab', type=Path, default=Path('.local/run/fixed-team-lab-01'))
    parser.add_argument('--fresh-start', action='store_true', help='Build from fresh weights without importing prior lab models')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--reward-config', type=Path)
    parser.add_argument('--jobs', type=int, choices=(1,2), default=2)
    parser.add_argument('--continue', dest='continuing', action='store_true', help='Resume pinned stages; completed/blocked stages are not retrained')
    parser.add_argument('--smoke', action='store_true', help='Tiny integration test; labels output SMOKE and bypasses learning gates')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    root = args.output_dir.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    plan = None
    locked = False
    lock = (root.parent / f'.{root.name}.staged.lock').open('a')
    stopped, monitor = threading.Event(), None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        if args.continuing:
            plan = json.loads((root / 'plan.json').read_text())
            if plan.get('version') != 1:
                raise ValueError('Unsupported staged plan version')
            verify_pins(root, plan['pins'])
        else:
            plan = freeze(root, args)
        monitor = threading.Thread(target=memory_monitor, args=(root, stopped), daemon=True)
        monitor.start()
        print(f"Output: {root}\nWatch: tail -n 20 -F {shlex.quote(str(root / 'console'))}/*.log\nMemory: tail -F {shlex.quote(str(root / 'memory.csv'))}", flush=True)
        anchor = plan['imports'].get('mirror')
        for stage in plan['stages']:
            path = root / 'stages' / stage / 'result.json'
            if path.exists():
                result = json.loads(path.read_text())
                if checksum(root / result['selected']) != result['checkpoint_sha256']:
                    raise ValueError('Selected checkpoint changed')
            else:
                write_json(root / 'status.json', dict(status='running', stage=stage))
                stage_anchor = None if stage == 'move-choice' else anchor
                result = run_stage(root, plan, stage, stage_anchor, args.jobs)
            collect(root, plan)
            if result['status'] == 'blocked':
                write_json(root / 'status.json', dict(status='blocked', stage=stage, reasons=result['failures']))
                print('Progression stopped at the failed gate. Inspect report.md; --continue preserves this decision.', flush=True)
                return 2
            if stage != 'move-choice':
                # With a prior lab, replication tests reproducibility but must not replace
                # the selected stable continuation with a less stable fresh checkpoint.
                if stage != 'mirror-replication' or anchor is None:
                    anchor = result['selected']
        write_json(root / 'status.json', dict(status='smoke_completed' if plan['smoke'] else 'completed'))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if plan and locked:
            write_json(root / 'status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', message=str(error)))
        print(f'Stopped: {error or "interrupted"}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        stopped.set()
        if monitor:
            monitor.join()
        if plan and locked:
            collect(root, plan)
            print(f"Report: {root / 'report.md'}\nUpload: {root / 'staged-team-results.zip'}", flush=True)
        signal.signal(signal.SIGTERM, previous)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
