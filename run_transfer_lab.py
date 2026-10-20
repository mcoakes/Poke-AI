#!/usr/bin/env python3
"""Compare fixed-team specialist and fresh initialization on the same stage-4 task."""
import argparse
import copy
import fcntl
import json
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
from run_staged_team_lab import memory_monitor
from run_team_curriculum import complete, rate

STAGE = '04-type-matchups'
CONDITIONS = ('validation', 'test', 'heldout', 'permuted', 'mirror')


def read_json(path):
    return json.loads(Path(path).read_text())


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Input path escapes run directory: {relative}')
    return path


def source_inputs(source, smoke=False):
    """Validate only the imported stage-4 inputs, never unrelated later stages."""
    source = Path(source).expanduser().resolve()
    plan = read_json(source / 'plan.json')
    if plan.get('version') != 1:
        raise ValueError('Expected a version-1 team curriculum source run')
    if plan.get('smoke') and not smoke:
        raise ValueError('A smoke source cannot supply a normal transfer experiment')
    qualified = read_json(source / 'stages/02-fixed-mirror/result.json')
    if qualified.get('status') != 'passed' and not (smoke and qualified.get('status') == 'smoke_completed'):
        raise ValueError('The source fixed-mirror foundation must have passed')
    model_name = plan['imports']['mirror']
    if qualified['selected'] != model_name:
        raise ValueError('Qualified mirror checkpoint differs from the source import')
    files = {}

    def pinned(relative, digest=None):
        path = inside(source, relative)
        expected = plan['pins'].get(relative)
        if not expected or checksum(path) != expected or (digest is not None and digest != expected):
            raise ValueError(f'Source pinned input changed: {relative}')
        files[relative] = expected
        return path

    model = pinned(model_name, qualified['checkpoint_sha256'])
    reward = pinned('inputs/reward.json')
    teams = {}
    for label, entry in [('train', plan['assets'][STAGE]), ('mirror', plan['fixed']['mirror'])]:
        values = entry['train_paths']
        directory = inside(source / 'inputs', entry['train_dir'])
        expected_paths = {inside(source / 'inputs', value) for value in values}
        if not values or len(values) != len(expected_paths) or set(directory.glob('*.json')) != expected_paths:
            raise ValueError(f'Source {label} team membership changed')
        teams[label] = [pinned('inputs/' + value, entry['sha256'][value]) for value in values]
    if len(teams['mirror']) != 2:
        raise ValueError('Fixed mirror requires exactly two team files')
    suites = {label: pinned(plan['suites'][STAGE][label]) for label in CONDITIONS if label != 'mirror'}
    suites['mirror'] = pinned(plan['suites']['02-fixed-mirror']['test'])
    return dict(source=str(source), model=model, reward=reward, teams=teams, suites=suites,
                qualified=qualified, source_pins=files, source_smoke=bool(plan.get('smoke')))


def fresh_checkpoint(template, destination, settings):
    """Initialize new weights and a new optimizer; template supplies spaces only."""
    import gymnasium as gym
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from firered_rl.shared_move_policy import SharedMovePolicy

    class SpaceOnly(gym.Env):
        def __init__(self):
            self.observation_space = copy.deepcopy(template.observation_space)
            self.action_space = copy.deepcopy(template.action_space)
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {k: np.zeros(v.shape, dtype=v.dtype) for k, v in self.observation_space.spaces.items()}, {}
        def step(self, action):
            raise RuntimeError('Initialization environment must never run battles')

    vec = DummyVecEnv([SpaceOnly for _ in range(settings['environments'] // 2)])
    try:
        model = MaskablePPO(SharedMovePolicy, vec, device='cpu', seed=settings['seed'],
            n_steps=128, batch_size=128, n_epochs=4, learning_rate=settings['learning_rate'],
            ent_coef=settings['ent_coef'], gamma=.99, gae_lambda=.95, target_kl=.03,
            verbose=0, policy_kwargs={})
        model.observation_version = 'v2'
        model.policy_architecture = 'shared-moves'
        model.policy_architecture_version = model.policy.ARCHITECTURE_VERSION
        model.battle_backend = 'showdown'
        model.reward_settings = template.reward_settings
        model.save(destination)
        if model.num_timesteps != 0 or model.policy.optimizer.state:
            raise ValueError('Fresh initialization unexpectedly contains training state')
    finally:
        vec.close()


def make_jobs(root, settings):
    jobs = []
    for name, relative in [('specialist', 'inputs/foundation-mirror.zip'), ('fresh', 'inputs/fresh-initial.zip')]:
        start = checkpoint_steps(root / relative)
        jobs.append(dict(name=name, checkpoint=relative, starting_timesteps=start,
            target_timesteps=start + settings['timesteps'], seed=settings['seed']))
    return jobs


def freeze(root, args):
    from sb3_contrib import MaskablePPO
    from firered_rl.shared_move_policy import SharedMovePolicy
    from firered_rl.showdown_env import validate_showdown
    from firered_rl.evaluation import validate_suite
    evidence = source_inputs(args.source_run, args.smoke)
    validate_showdown(ROOT)
    model = MaskablePPO.load(evidence['model'], device='cpu')
    if not isinstance(model.policy, SharedMovePolicy) or getattr(model, 'observation_version', None) != 'v2':
        raise ValueError('Foundation must use shared-moves policy with v2 observations')
    settings = dict(seed=261080, timesteps=256 if args.smoke else 1000000,
        environments=2 if args.smoke else 16, learning_rate=.0001, ent_coef=.01,
        eval_every=128 if args.smoke else 50000, max_steps=100, early_stop_consecutive=2,
        validation_threshold=.65, heldout_threshold=.60, rotation_max_drop=.03,
        mirror_max_drop=.10, max_truncation_rate=.01,
        selection='Highest validation wins/requested; earlier arm retains ties. No automatic promotion.',
        comparison='Same maximum new-step budget, success stopping rule and seed; fresh policy has new weights and optimizer.')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs'; inputs.mkdir()
    (inputs / 'suites').mkdir(); (root / 'console').mkdir()
    shutil.copyfile(evidence['model'], inputs / 'foundation-mirror.zip')
    shutil.copyfile(evidence['reward'], inputs / 'reward.json')
    write_json(inputs / 'settings.json', settings)
    teams = {}
    for label, paths in evidence['teams'].items():
        folder = inputs / 'teams' / label; folder.mkdir(parents=True)
        teams[label] = []
        for index, path in enumerate(paths):
            target = folder / f'team-{index+1:04}.json'
            shutil.copyfile(path, target); teams[label].append(str(target.relative_to(root)))
    mixture = dict(version=1, groups={
        'current': dict(weight=.9, directory=str(inputs / 'teams/train')),
        'fixed-mirror': dict(weight=.1, teams=[str(root / value) for value in teams['mirror']])})
    write_json(inputs / 'mixture.json', mixture)
    suites = {}
    for label, path in evidence['suites'].items():
        data = read_json(path)
        if args.smoke:
            data['cases'] = data['cases'][:8]
            data['diagnostic'] = dict(data.get('diagnostic', {}), transfer_smoke=True,
                source_cases=len(read_json(path)['cases']), actual_battles=len(data['cases']))
        validate_suite(data)
        target = inputs / 'suites' / f'{label}.json'
        write_json(target, data); suites[label] = str(target.relative_to(root))
    fresh_checkpoint(model, inputs / 'fresh-initial.zip', settings)
    del model
    write_json(inputs / 'source-evidence.json', dict(source_run=evidence['source'],
        source_pins=evidence['source_pins'], qualified=evidence['qualified'],
        source_smoke=evidence['source_smoke'], note='Original curriculum and its blocked result are unchanged.'))
    jobs = make_jobs(root, settings)
    for job in jobs: (root / 'console' / (job['name'] + '.log')).touch()
    plan = dict(version=1, experiment='initialization-transfer', smoke=args.smoke,
        settings=settings, teams=teams, suites=suites, jobs=jobs,
        pins={str(p.relative_to(root)): checksum(p) for p in inputs.rglob('*') if p.is_file()})
    write_json(root / 'plan.json', plan)
    return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != 'initialization-transfer':
        raise ValueError('Unsupported transfer plan')
    for name, digest in plan['pins'].items():
        if checksum(inside(root, name)) != digest:
            raise ValueError(f'Pinned input changed: {name}')
    if plan['settings'] != read_json(root / 'inputs/settings.json') or plan['jobs'] != make_jobs(root, plan['settings']):
        raise ValueError('Saved experiment settings or budgets changed')
    if checkpoint_steps(root / 'inputs/fresh-initial.zip') != 0:
        raise ValueError('Fresh checkpoint is not an untrained initialization')
    for label, values in plan['teams'].items():
        expected = {inside(root, relative) for relative in values}
        if len(expected) != len(values) or set((root / 'inputs/teams' / label).glob('*.json')) != expected:
            raise ValueError(f'Pinned team membership changed: {label}')


def command(root, plan, job):
    base = [sys.executable, '-u', str(ROOT / 'mgba/train_ppo.py')]
    run = root / 'runs' / job['name']
    if run.exists():
        summary = read_json(run / 'summary.json') if (run / 'summary.json').exists() else {}
        steps = recovered_steps(run)
        if summary.get('status') == 'completed' and (steps >= job['target_timesteps'] or summary.get('stop_reason') == 'validation_success'):
            return None
        return base + ['--output-dir', str(run), '--if-exists', 'continue',
            '--total-timesteps', str(max(1, job['target_timesteps'] - steps))]
    s = plan['settings']
    return base + ['--backend', 'showdown', '--resume', str(root / job['checkpoint']),
        '--policy-architecture', 'shared-moves', '--observation-version', 'v2', '--rule-based-opponent',
        '--team-mixture', str(root / 'inputs/mixture.json'), '--reward-config', str(root / 'inputs/reward.json'),
        '--learning-rate', str(s['learning_rate']), '--ent-coef', str(s['ent_coef']),
        '--seed', str(job['seed']), '--environments', str(s['environments']),
        '--rollout-steps', '128', '--batch-size', '128', '--epochs', '4',
        '--torch-threads', '1', '--device', 'cpu', '--showdown-workers', 'process',
        '--max-steps', str(s['max_steps']), '--total-timesteps', str(s['timesteps']),
        '--eval-suite', str(root / plan['suites']['validation']), '--eval-every', str(s['eval_every']),
        '--eval-environments', '4', '--eval-stop-win-rate', str(s['validation_threshold']),
        '--eval-stop-consecutive', str(s['early_stop_consecutive']), '--latest-every-updates', '1',
        '--checkpoint-every', str(s['eval_every']), '--keep-checkpoints', '3', '--keep-evaluations', '2',
        '--battle-log', '--no-readable-log', '--no-step-logs', '--log-max-mb', '2', '--log-backups', '1',
        '--output-dir', str(run), '--if-exists', 'cancel']


def launch(root, plan, concurrency):
    if concurrency not in (1, 2):
        raise ValueError('Transfer experiments allow one or two concurrent jobs')
    pending, active = list(plan['jobs']), []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        while pending or active:
            while pending and len(active) < concurrency:
                job = pending.pop(0); cmd = command(root, plan, job)
                if cmd is None: continue
                print(f"Starting {job['name']}\n{shlex.join(cmd)}", flush=True)
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, start_new_session=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                reader = threading.Thread(target=pump, args=(proc, root / 'console' / (job['name'] + '.log')))
                reader.start(); active.append((job, proc, reader))
            for row in list(active):
                job, proc, reader = row; code = proc.poll()
                if code is None: continue
                reader.join(); active.remove(row)
                if code or read_json(root / 'runs' / job['name'] / 'summary.json').get('status') != 'completed':
                    raise RuntimeError(f"{job['name']} failed; inspect its console log")
            if active: time.sleep(.25)
    finally:
        for _, proc, _ in active:
            if proc.poll() is None: proc.send_signal(signal.SIGINT)
        if active: print('Waiting for trainers to save and stop workers.', flush=True)
        for _, proc, reader in active: proc.wait(); reader.join()


def selection(root, plan):
    candidates = []
    for job in plan['jobs']:
        run = root / 'runs' / job['name']
        meta = read_json(run / 'checkpoints/best.json'); path = run / 'checkpoints/best.zip'
        if checksum(path) != meta['checkpoint_sha256']:
            raise ValueError(f"Best checkpoint hash mismatch: {job['name']}")
        if not meta.get('requested'):
            raise ValueError('Missing validation denominator')
        candidates.append(dict(name=job['name'], checkpoint=str(path.relative_to(root)),
            checkpoint_sha256=checksum(path), validation_rate=meta['wins'] / meta['requested']))
    result = dict(selected=max(candidates, key=lambda c: c['validation_rate']), candidates=candidates,
        criterion=plan['settings']['selection'], heldout_used_for_selection=False)
    write_json(root / 'selection.json', result)
    return result


def readiness(checks, mirror_start, settings):
    issues = []
    for condition, floor in [('validation', settings['validation_threshold']),
                             ('heldout', settings['heldout_threshold']), ('permuted', settings['heldout_threshold'])]:
        if not complete(checks[condition], settings['max_truncation_rate']) or rate(checks[condition]) < floor:
            issues.append(f'{condition}: below {floor:.0%} or incomplete/excess truncation')
    if rate(checks['heldout']) - rate(checks['permuted']) > settings['rotation_max_drop'] + 1e-12:
        issues.append('move rotation loses more than 3 percentage points')
    if not complete(checks['mirror'], settings['max_truncation_rate']) or not complete(mirror_start, settings['max_truncation_rate']):
        issues.append('mirror check incomplete or excessive truncation')
    elif rate(checks['mirror']) < rate(mirror_start) - settings['mirror_max_drop'] - 1e-12:
        issues.append('mirror below original specialist minus 10 percentage points')
    return issues


def evaluate(root, plan):
    from firered_rl.curriculum_checks import frozen_check
    suites = {k: root / v for k, v in plan['suites'].items()}
    # All baselines use the exact same held-out cases as the policies.
    baselines = {name: frozen_check(ROOT, root, 'baseline-' + name, suites['heldout'], baseline=name)
                 for name in ('random', 'greedy', 'rules')}
    initial = {job['name']: {label: frozen_check(ROOT, root, job['name'] + '-initial-' + label,
        suite, checkpoint=root / job['checkpoint']) for label, suite in suites.items()} for job in plan['jobs']}
    write_json(root / 'initial.json', dict(baselines=baselines, policies=initial))
    return baselines, initial


def finish(root, plan, baselines, initial):
    from firered_rl.curriculum_checks import frozen_check
    selected = selection(root, plan)  # Before any trained-policy held-out checks.
    results, cache = {}, {}
    for job in plan['jobs']:
        results[job['name']] = {}
        for kind in ('best', 'final'):
            path = root / 'runs' / job['name'] / 'checkpoints' / (kind + '.zip')
            digest = checksum(path)
            if digest not in cache:
                cache[digest] = {label: frozen_check(ROOT, root, job['name'] + '-' + kind + '-' + label,
                    root / suite, checkpoint=path) for label, suite in plan['suites'].items()}
            checks = cache[digest]
            issues = readiness(checks, initial['specialist']['mirror'], plan['settings'])
            results[job['name']][kind] = dict(checkpoint=str(path.relative_to(root)), checkpoint_sha256=digest,
                checks=checks, readiness='not_ready' if issues else 'ready', issues=issues,
                mirror_change_from_own_start=rate(checks['mirror']) - rate(initial[job['name']]['mirror']))
    training = {}
    for job in plan['jobs']:
        summary = read_json(root / 'runs' / job['name'] / 'summary.json')
        training[job['name']] = dict(starting_timesteps=job['starting_timesteps'],
            ending_timesteps=summary['player_timesteps'],
            added_timesteps=summary['player_timesteps'] - job['starting_timesteps'],
            max_new_timesteps=plan['settings']['timesteps'], elapsed_seconds=summary['elapsed_seconds'],
            stop_reason=summary['stop_reason'])
    result = dict(status='smoke_completed' if plan['smoke'] else 'completed', diagnostic_only=True, training=training,
        selected_by_validation=selected['selected'], policies=results, baselines=baselines,
        initial=initial, no_automatic_promotion=True)
    write_json(root / 'result.json', result)
    return result


def inspect(root, plan, label, limit, full):
    labels = ('train', 'mirror', 'validation', 'heldout') if label == 'all' else (label,)
    print('Training mixture: ' + str(root / 'inputs/mixture.json'))
    print((root / 'inputs/mixture.json').read_text())
    for name in labels:
        if name in plan['teams']:
            values = [(str(root / path), read_json(root / path)) for path in plan['teams'][name]]
        else:
            path = root / plan['suites'][name]
            values = [(str(path) + ' asset ' + key, value['team']) for key, value in read_json(path)['assets'].items()]
        print(f'\n{name}: {len(values)} teams')
        for path, team in values[:limit]:
            print(path)
            if full: print(json.dumps(team, indent=2))
            else:
                for mon in team['party']:
                    print(f"  {mon['species']}: {', '.join(mon['moves'])} | {mon.get('nature', 'Hardy')} | {mon.get('held_item', 'None')}")


def immunity_counts(check):
    totals = check.get('audit_totals', {})
    return [str(totals[key]) if key in totals else 'N/A' for key in (
        'matchup_classified_move_actions', 'current_target_immune_move_actions',
        'current_target_immune_with_nonimmune_alternative')]


def collect(root, plan):
    result = read_json(root / 'result.json') if (root / 'result.json').exists() else {}
    initial = read_json(root / 'initial.json').get('policies', {}) if (root / 'initial.json').exists() else {}
    lines = ['# Type-transfer experiment', '', 'SMOKE ONLY: tiny budgets; not learning evidence.' if plan['smoke'] else
        'Specialist and fresh policies receive the same maximum new-step budget, success stopping rule, seed, teams and PPO settings. All battles are against Rules.', '',
        'Checkpoint selection uses validation only. Readiness is diagnostic; no model is promoted and the source run is unchanged.', '',
        '| Policy | Checkpoint | Validation | Seen teams | Heldout | Rotated | Fixed mirror | Readiness |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for job in plan['jobs']:
        name = job['name']
        records = [('initial', dict(checks=initial.get(name, {}), readiness='baseline'))]
        records += list(result.get('policies', {}).get(name, {}).items())
        for kind, record in records:
            checks = record['checks']
            values = [f'{rate(checks[k]):.1%}' if k in checks else '—' for k in CONDITIONS]
            lines.append(f"| {name} | {kind} | {' | '.join(values)} | {record['readiness']} |")
    lines += ['', '## Held-out move immunity audit', '',
        'All candidate battles; current defender before the opponent action. Unsupported moves are excluded. A legal alternative has no classified current immunity, not guaranteed damage.', '',
        '| Policy | Checkpoint | Classified moves | Immune choices | With legal alternative |', '|---|---|---:|---:|---:|']
    for job in plan['jobs']:
        name = job['name']
        records = [('initial', dict(checks=initial.get(name, {})))]
        records += list(result.get('policies', {}).get(name, {}).items())
        for kind, record in records:
            values = immunity_counts(record['checks'].get('heldout', {}))
            lines.append(f"| {name} | {kind} | {' | '.join(values)} |")
    lines += ['', '## Training budgets', '', '| Policy | Added steps | Maximum budget | Minutes | Stop reason |', '|---|---:|---:|---:|---|']
    for job in plan['jobs']:
        path = root / 'runs' / job['name'] / 'summary.json'
        value = read_json(path) if path.exists() else {}
        added = value.get('player_timesteps', job['starting_timesteps']) - job['starting_timesteps']
        lines.append(f"| {job['name']} | {added:,} | {plan['settings']['timesteps']:,} | {value.get('elapsed_seconds', 0)/60:.1f} | {value.get('stop_reason', 'pending')} |")
    lines += ['', 'Readiness issues:']
    for name, records in result.get('policies', {}).items():
        for kind, record in records.items():
            lines += [f'- {name} {kind}: {issue}' for issue in record['issues']]
    lines += ['', 'Training: 90% current stage and 10% original fixed mirror. Sampling counts are recorded in each run summary.',
        'Fresh mirror change is measured against its own initial score; readiness still requires the original specialist retention target. Reused held-out suites are development checks, not a new final generalization estimate.',
        'All baselines and held-out policy checks use exactly the same saved schedule.', '',
        '## Frozen checks', '', '| Check | Wins/requested | Rate | Truncated |', '|---|---:|---:|---:|']
    for path in sorted((root / 'checks').glob('*/summary.json')):
        value = read_json(path)
        if value.get('status') == 'completed':
            lines.append(f"| {path.parent.name} | {value['candidate_wins']}/{value['requested_battles']} | {rate(value):.1%} | {value['truncated']} |")
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    temp = root / 'transfer-results.tmp'
    with ZipFile(temp, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.md', '.csv'): continue
            if relative.parts[0] in ('inputs', 'checks') or path.name in (
                'plan.json', 'status.json', 'report.md', 'memory.csv', 'result.json', 'initial.json', 'selection.json',
                'config.json', 'summary.json', 'failure.json', 'updates.jsonl', 'battle-blocks.jsonl', 'progress.csv',
                'switching.json', 'evaluation-history.json', 'evaluations.md', 'evaluations.csv', 'best.json'):
                archive.write(path, relative)
    temp.replace(root / 'transfer-results.zip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('.local/run/team-curriculum-01'))
    parser.add_argument('--output-dir', type=Path, default=Path('.local/run/type-transfer-01'))
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-teams', nargs='?', const='all', choices=('all', 'train', 'mirror', 'validation', 'heldout'))
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--full-json', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Tiny integration test; never claim learning success')
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.limit < 0: parser.error('--limit cannot be negative')
    if args.inspect_teams is not None:
        plan = read_json(root / 'plan.json'); verify(root, plan)
        inspect(root, plan, args.inspect_teams, args.limit, args.full_json)
        return 0
    import torch
    torch.set_num_threads(1)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = (root.parent / ('.' + root.name + '.transfer.lock')).open('a')
    plan = None; owned = False; stop = threading.Event(); monitor = None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); owned = True
        if args.continuing:
            plan = read_json(root / 'plan.json'); verify(root, plan)
        else: plan = freeze(root, args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {shlex.quote(str(root / 'console'))}/*.log\nMemory: watch -n 2 free -h\nInspect: python run_transfer_lab.py --output-dir {shlex.quote(str(root))} --inspect-teams train", flush=True)
        if args.prepare_only:
            write_json(root / 'status.json', dict(status='prepared')); return 0
        monitor = threading.Thread(target=memory_monitor, args=(root, stop), daemon=True); monitor.start()
        write_json(root / 'status.json', dict(status='running', phase='initial-checks'))
        baselines, initial = evaluate(root, plan)
        collect(root, plan)
        write_json(root / 'status.json', dict(status='running', phase='training'))
        launch(root, plan, args.jobs)
        verify(root, plan)
        write_json(root / 'status.json', dict(status='running', phase='frozen-checks'))
        result = finish(root, plan, baselines, initial)
        write_json(root / 'status.json', dict(status=result['status'], diagnostic_only=True))
        return 0
    except (KeyboardInterrupt, Exception) as error:
        if plan is not None and owned:
            write_json(root / 'status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', message=str(error)))
        print(f'Stopped: {error or "interrupted"}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        stop.set()
        if monitor: monitor.join()
        if plan is not None and owned:
            collect(root, plan)
            print(f"Report: {root / 'report.md'}\nUpload: {root / 'transfer-results.zip'}", flush=True)
        signal.signal(signal.SIGTERM, previous); lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
