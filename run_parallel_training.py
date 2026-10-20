#!/usr/bin/env python3
"""Run seeded Showdown experiments with bounded concurrency and logs.

Run from the Poke-AI project root using its training venv. No extra dependencies.
Default: four seeds of one setup. --experiments loads explicit strategy variants.
1,000,000 additional transitions per run, at most four runs together.
Ctrl+C stops the active trainers gracefully; rerun with --continue-batch to finish
only the remaining budgets. Completed seeds are skipped. Source models are copied.
"""
import argparse
import fcntl
import hashlib
import json
import logging
import math
import re
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile


def checkpoint_steps(path):
    with ZipFile(path) as archive:
        value = json.loads(archive.read('data'))['num_timesteps']
    if not isinstance(value, int) or value < 0:
        raise ValueError(f'Invalid timestep counter: {path}')
    return value


def recovered_steps(run):
    paths = [run/'checkpoints/latest.zip', run/'checkpoints/final.zip']
    paths += list((run/'checkpoints').glob('ppo_*.zip'))
    values = []
    for path in paths:
        if path.is_file():
            try:
                values.append(checkpoint_steps(path))
            except (ValueError, KeyError, OSError, BadZipFile):
                continue
    if not values:
        raise ValueError(f'No readable progress checkpoint in {run}. Keep diagnostics and start a new batch directory.')
    return max(values)


DEFAULT_TRAINING = dict(learning_rate=0.0001, ent_coef=0.01, rules_weight=1.0,
                        frozen_reference_weight=0.0)


def jobs_for(plan):
    """Existing four-seed batch plans retain their original directories/settings."""
    if 'experiments' in plan:
        return plan['experiments']
    return [dict(name=f'seed-{seed}', seed=seed, **DEFAULT_TRAINING)
            for seed in plan['seeds']]


def read_experiments(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('version') != 1 or set(data) != {'version', 'experiments'}:
        raise ValueError('Experiment file requires version=1 and experiments only.')
    entries = data['experiments']
    if not isinstance(entries, list) or not entries:
        raise ValueError('experiments must be a nonempty list.')
    jobs, names = [], set()
    allowed = {'name', 'seeds'} | set(DEFAULT_TRAINING)
    for entry in entries:
        if not isinstance(entry, dict) or set(entry)-allowed:
            raise ValueError(f'Unknown experiment fields; allowed: {sorted(allowed)}')
        name = entry.get('name', '')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9-]{0,47}', name):
            raise ValueError('Experiment names must be short lowercase letters, digits or hyphens.')
        seeds = entry.get('seeds')
        if not isinstance(seeds, list) or not seeds or any(type(v) is not int or not 0 <= v < 2**31 for v in seeds):
            raise ValueError(f'{name}: seeds must be integers in [0, 2**31).')
        settings = dict(DEFAULT_TRAINING)
        for key in DEFAULT_TRAINING:
            value = entry.get(key, settings[key])
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}: {key} must be finite and nonnegative.')
            settings[key] = float(value)
        if settings['learning_rate'] <= 0 or settings['rules_weight'] + settings['frozen_reference_weight'] <= 0:
            raise ValueError(f'{name}: learning_rate and total opponent weight must be positive.')
        for seed in seeds:
            label = f'{name}-seed-{seed}'
            if label in names:
                raise ValueError(f'Duplicate experiment run: {label}')
            names.add(label)
            jobs.append(dict(name=label, strategy=name, seed=seed, **settings))
    return jobs


def pool_payload(job):
    opponents = {}
    if job['rules_weight'] > 0:
        opponents['rules'] = dict(kind='rule_based', weight=job['rules_weight'])
    if job['frozen_reference_weight'] > 0:
        opponents['reference'] = dict(kind='checkpoint', group='learned',
            label='Frozen batch starting checkpoint', path='reference.zip',
            weight=job['frozen_reference_weight'])
    return dict(version=1, opponents=opponents)


def command(project, root, plan, name):
    # Accept legacy numeric callers as well as explicit run names.
    if isinstance(name, int): name = f'seed-{name}'
    job = next(job for job in jobs_for(plan) if job['name'] == name)
    run = root/name
    if run.exists():
        if not (run/'config.json').is_file():
            raise ValueError(f'Incomplete run directory: {run}; inspect it before recovery.')
        remaining = plan['target_timesteps'] - recovered_steps(run)
        if remaining <= 0:
            return None
        return [sys.executable, '-u', str(project/'mgba/train_ppo.py'),
                '--output-dir', str(run), '--if-exists', 'continue',
                '--total-timesteps', str(remaining)]
    opponent = (['--opponent-pool', str(root/'inputs'/f'{name}-opponents.json')]
                if job['frozen_reference_weight'] else ['--rule-based-opponent'])
    return [sys.executable, '-u', str(project/'mgba/train_ppo.py'),
            '--backend', 'showdown', '--resume', str(root/'inputs/reference.zip'),
            '--policy-architecture', 'shared-moves', '--observation-version', 'v2',
            *opponent, '--team-dir', plan['team_dir'],
            '--reward-config', str(root/'inputs/reward.json'),
            '--learning-rate', str(job['learning_rate']), '--ent-coef', str(job['ent_coef']),
            '--rollout-steps', '128', '--batch-size', '128', '--epochs', '4',
            '--seed', str(job['seed']), '--environments', str(plan['environments']),
            '--showdown-workers', 'process', '--device', 'cpu', '--torch-threads', '1',
            '--total-timesteps', str(plan['additional_timesteps']),
            '--eval-suite', plan['eval_suite'], '--eval-every', str(plan['eval_every']),
            '--eval-environments', str(plan['eval_environments']),
            '--battle-log', '--no-readable-log', '--no-step-logs',
            '--log-max-mb', '2', '--log-backups', '1',
            '--latest-every-updates', '5', '--checkpoint-every', '100000',
            '--keep-checkpoints', '2', '--keep-evaluations', '2',
            '--output-dir', str(run), '--if-exists', 'cancel']


def pump(process, path):
    handler = RotatingFileHandler(path, maxBytes=8*1024*1024, backupCount=1)
    handler.setFormatter(logging.Formatter('%(message)s'))
    try:
        for line in process.stdout:
            record = logging.LogRecord('trainer', logging.INFO, '', 0, line.rstrip('\n'), (), None)
            handler.handle(record)
    finally:
        process.stdout.close()
        handler.close()


def collect(root, plan, states):
    jobs = jobs_for(plan)
    lines = ['# Parallel shared-moves training', '',
             'All runs start from the same weights. Settings and seeds are recorded below.', '',
             '| Run | Seed | Learning rate | Entropy | Rules / frozen weight | Status |',
             '|---|---:|---:|---:|---|---|']
    for job in jobs:
        name = job['name']
        lines.append(f"| {name} | {job['seed']} | {job['learning_rate']} | {job['ent_coef']} | "
                     f"{job['rules_weight']} / {job['frozen_reference_weight']} | {states.get(name, 'not started')} |")
    for job in jobs:
        name = job['name']
        table = root/name/'evaluations.md'
        if table.exists():
            lines += ['', f'## {name}', '', table.read_text()]
    (root/'batch-report.md').write_text('\n'.join(lines)+'\n')
    with ZipFile(root/'batch-results.tmp', 'w', ZIP_DEFLATED) as archive:
        for name in ['batch-plan.json', 'batch-report.md']:
            archive.write(root/name, name)
        for job in jobs:
            pool = root/'inputs'/f"{job['name']}-opponents.json"
            if pool.exists(): archive.write(pool, str(pool.relative_to(root)))
            for name in ['config.json', 'summary.json', 'evaluations.md', 'evaluations.csv',
                         'evaluation-history.json', 'checkpoints/best.json', 'failure.json']:
                path = root/job['name']/name
                if path.is_file():
                    archive.write(path, str(path.relative_to(root)))
    (root/'batch-results.tmp').replace(root/'batch-results.zip')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('.local/run/shared-long-seeds-01'))
    p.add_argument('--checkpoint', type=Path, default=Path('models/observation-v2-shared-moves-400k-reference-01.zip'))
    p.add_argument('--team-dir', type=Path, default=Path('.local/run/diversity-v2-01/teams/training'))
    p.add_argument('--reward-config', type=Path, default=Path('.local/run/diversity-v2-01/inputs/reward.json'))
    p.add_argument('--eval-suite', type=Path, default=Path('.local/run/validation-rules-and-agents-01/suite.json'))
    p.add_argument('--experiments', type=Path, help='JSON strategy definitions; each lists explicit seeds and overrides')
    p.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=[261020, 261021, 261022, 261023])
    p.add_argument('--total-timesteps', type=int, default=1000000)
    p.add_argument('--eval-every', type=int, default=200000)
    p.add_argument('--environments', type=int, default=64)
    p.add_argument('--eval-environments', type=int, default=16)
    p.add_argument('--continue-batch', action='store_true', help='Restore plan and finish remaining budgets; only --jobs may change')
    p.add_argument('--dry-run', action='store_true', help='Validate inputs and print commands without creating files')
    args = p.parse_args()
    project = Path.cwd().resolve()
    root = args.output_dir.resolve()
    if not (project/'mgba/train_ppo.py').is_file():
        p.error('Run from your Poke-AI project root with the training venv active.')
    if args.continue_batch:
        if args.experiments is not None:
            p.error('--continue-batch restores its saved experiments; omit --experiments.')
        # Restore the entire original plan, never silently add another full budget.
        plan = json.loads((root/'batch-plan.json').read_text())
        if plan['project'] != str(project):
            p.error('Batch belongs to a different project directory.')
        if hashlib.sha256((root/'inputs/reference.zip').read_bytes()).hexdigest() != plan['checkpoint_sha256']:
            p.error('Pinned starting checkpoint changed.')
        if 'reward_sha256' in plan and hashlib.sha256((root/'inputs/reward.json').read_bytes()).hexdigest() != plan['reward_sha256']:
            p.error('Pinned reward config changed.')
        for job in jobs_for(plan):
            if job['frozen_reference_weight']:
                pool = root/'inputs'/f"{job['name']}-opponents.json"
                if json.loads(pool.read_text()) != pool_payload(job):
                    p.error(f'Pinned opponent pool changed: {pool}')
    else:
        if root.exists():
            p.error('Output directory exists. Use --continue-batch or a new directory; nothing is overwritten.')
        if not args.seeds or len(set(args.seeds)) != len(args.seeds):
            p.error('Seeds must be distinct.')
        if min(args.total_timesteps, args.eval_every, args.environments, args.eval_environments) <= 0:
            p.error('Budgets and worker counts must be positive.')
        if args.environments % 2 or args.eval_environments % 2:
            p.error('Environment counts must be even.')
        if any(seed < 0 or seed >= 2**31 for seed in args.seeds):
            p.error('Seeds must be in [0, 2**31).')
        for file in [args.checkpoint, args.reward_config, args.eval_suite]:
            if not file.is_file():
                p.error(f'Missing input: {file}')
        if not args.team_dir.is_dir() or not any(args.team_dir.rglob('*.json')):
            p.error(f'No training teams: {args.team_dir}')
        try:
            experiments = (read_experiments(args.experiments) if args.experiments else
                [dict(name=f'seed-{seed}', seed=seed, **DEFAULT_TRAINING) for seed in args.seeds])
        except (ValueError, OSError) as error:
            p.error(str(error))
        start = checkpoint_steps(args.checkpoint)
        plan = dict(project=str(project), seeds=args.seeds, experiments=experiments, starting_timesteps=start,
                    target_timesteps=start+args.total_timesteps, additional_timesteps=args.total_timesteps,
                    team_dir=str(args.team_dir.resolve()), eval_suite=str(args.eval_suite.resolve()),
                    environments=args.environments, eval_environments=args.eval_environments,
                    eval_every=args.eval_every,
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    reward_sha256=hashlib.sha256(args.reward_config.read_bytes()).hexdigest())
    if args.dry_run:
        for job in jobs_for(plan):
            print(f"Run {job['name']}: seed={job['seed']}, lr={job['learning_rate']}, entropy={job['ent_coef']}, "
                  f"rules/frozen={job['rules_weight']}/{job['frozen_reference_weight']}")
            cmd = command(project, root, plan, job['name'])
            print(shlex.join(cmd) if cmd else f"{job['name']}: complete, skip")
        return 0
    if not args.continue_batch:
        root.mkdir(parents=True)
        (root/'inputs').mkdir()
        shutil.copyfile(args.checkpoint, root/'inputs/reference.zip')
        shutil.copyfile(args.reward_config, root/'inputs/reward.json')
        for job in jobs_for(plan):
            if job['frozen_reference_weight']:
                (root/'inputs'/f"{job['name']}-opponents.json").write_text(json.dumps(pool_payload(job), indent=2)+'\n')
        (root/'batch-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    # Prevent two launchers from managing this batch together. Trainer locks also apply.
    lock = (root/'batch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('This batch already has an active launcher.')
    (root/'console').mkdir(exist_ok=True)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    pending = [job['name'] for job in jobs_for(plan)]
    active, states = {}, {}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    failed = False
    try:
        while pending or active:
            while pending and len(active) < args.jobs and not stop.is_set():
                seed = pending.pop(0)
                cmd = command(project, root, plan, seed)
                if cmd is None:
                    states[seed] = 'budget reached'; continue
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, start_new_session=True, env=env)
                thread = threading.Thread(target=pump, args=(process, root/f'console/{seed}.log'))
                thread.start()
                active[seed] = [process, thread, False]
                states[seed] = 'running'
                print(f'Started {seed}, PID {process.pid}; {root / seed / "evaluations.md"}', flush=True)
            if stop.is_set():
                pending.clear()
                for process, thread, signalled in active.values():
                    if not signalled and process.poll() is None:
                        try:
                            process.send_signal(signal.SIGINT)
                        except ProcessLookupError:
                            pass
                for state in active.values(): state[2] = True
            for seed, (process, thread, _) in list(active.items()):
                if process.poll() is None: continue
                thread.join()
                path = root/seed/'summary.json'
                status = json.loads(path.read_text()).get('status') if path.is_file() else 'no summary'
                states[seed] = status if process.returncode == 0 else f'failed (exit {process.returncode})'
                if (process.returncode != 0 or status != 'completed') and not stop.is_set(): failed = True
                print(f'{seed}: {states[seed]}', flush=True)
                del active[seed]
                collect(root, plan, states)
            if stop.is_set() and not active: break
            time.sleep(.25)
    finally:
        # Also handle unexpected launcher errors without abandoning active trainers.
        for process, _, _ in active.values():
            if process.poll() is None:
                try: process.send_signal(signal.SIGINT)
                except ProcessLookupError: pass
        for process, thread, _ in active.values():
            process.wait(); thread.join()
        collect(root, plan, states)
        for sig, handler in previous.items(): signal.signal(sig, handler)
        lock.close()
    print(f'Reports: {root / "batch-results.zip"}', flush=True)
    return 130 if stop.is_set() else int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
