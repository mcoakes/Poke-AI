#!/usr/bin/env python3
"""Compare identical fresh policies with matchup features disabled/enabled.

Run at most two jobs. Watch: tail -n 20 -F RUN/console/*.log
Memory: watch -n 2 free -h. Ctrl+C saves; --continue resumes remaining budgets.
"""
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
import run_transfer_lab as transfer
from run_fixed_team_lab import checksum, write_json, interrupt
from run_parallel_training import checkpoint_steps, pump
from run_staged_team_lab import memory_monitor
from run_team_curriculum import rate, complete

read_json, inside = transfer.read_json, transfer.inside
CONDITIONS = transfer.CONDITIONS


def source_inputs(source, smoke=False):
    source = Path(source).expanduser().resolve()
    plan = read_json(source / 'plan.json')
    if plan.get('version') != 1 or plan.get('experiment') != 'initialization-transfer':
        raise ValueError('Source must be a version-1 initialization-transfer run')
    if plan.get('smoke') and not smoke:
        raise ValueError('A smoke source cannot supply a normal matchup experiment')
    pins = {}

    def pinned(relative):
        path = inside(source, relative)
        expected = plan['pins'].get(relative)
        if not expected or not path.is_file() or checksum(path) != expected:
            raise ValueError(f'Source pinned input changed: {relative}')
        pins[relative] = expected
        return path

    fresh = pinned('inputs/fresh-initial.zip')
    if checkpoint_steps(fresh) != 0:
        raise ValueError('Source fresh initialization must contain zero training steps')
    settings = read_json(pinned('inputs/settings.json'))
    if settings != plan['settings']:
        raise ValueError('Source settings differ from their pinned snapshot')
    reward = pinned('inputs/reward.json')
    teams = {}
    for name in ('train', 'mirror'):
        values = plan['teams'][name]
        expected = {inside(source, value) for value in values}
        if not values or len(values) != len(expected) or set((source / 'inputs/teams' / name).glob('*.json')) != expected:
            raise ValueError(f'Source {name} team membership changed')
        teams[name] = [pinned(value) for value in values]
    if len(teams['mirror']) != 2:
        raise ValueError('Fixed mirror needs exactly two team files')
    suites = {name: pinned(plan['suites'][name]) for name in CONDITIONS}
    foundation = pinned('inputs/foundation-mirror.zip')
    return dict(source=source, settings=settings, fresh=fresh, reward=reward, teams=teams,
        suites=suites, foundation=foundation, source_pins=pins, source_smoke=bool(plan.get('smoke')))


def initial_identity(models):
    """Prove both treatment arms share tensors; exercise logits on valid-shaped probes."""
    import numpy as np
    import torch
    control, matchup = models
    states = [model.policy.state_dict() for model in models]
    if states[0].keys() != states[1].keys() or any(not torch.equal(value, states[1][key]) for key, value in states[0].items()):
        raise ValueError('Ablation arms must have identical initial parameter/buffer tensors')
    if any(model.num_timesteps != 0 or model.policy.optimizer.state for model in models):
        raise ValueError('Ablation arms must be untrained with empty optimizers')
    enabled = [model.policy.features_extractor.relational_enabled for model in models]
    if enabled != [False, True]:
        raise ValueError('Expected control features off and matchup features on')
    probes = []
    for move_type, defender_type in ((10, 12), (4, 2), (0, 7)):
        obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in control.observation_space.spaces.items()}
        obs['action_mask'][:4] = 1
        obs['active'][0] = [1, 6, 50, 100, 100, 10, 2, 0, 1, 0]
        obs['active'][1] = [1, 9, 50, 100, 100, defender_type, defender_type, 0, 1, 0]
        obs['party'][0] = [1, 6, 50, 100, 100, 10, 2, 0, 0, 0]
        obs['opponent_party'][0] = [1, 9]
        obs['v2_party'][:, 0] = [1, 50, 100, 100, defender_type, defender_type, 0, 1, 0, 0]
        obs['v2_stats'][:, 0] = [100, 100, 100, 100, 100]
        obs['v2_active'][:] = [100, 100, 0, 1]
        obs['v2_active_stats'][:] = [100, 100, 100, 100, 100]
        for slot in range(4):
            obs['moves'][0, slot] = [50 + slot, move_type, 10, 10]
            obs['v2_move_details'][0, 0, slot] = [50 + slot * 20, 100, 0, 1, 0, move_type]
        probes.append(obs)
    batch = {key: torch.as_tensor(np.stack([obs[key] for obs in probes])) for key in probes[0]}
    outputs = []
    for model in models:
        model.policy.set_training_mode(False)
        with torch.no_grad():
            features = model.policy.extract_features(batch)
            outputs.append((model.policy.action_net(features), model.policy.predict_values(batch),
                model.policy.get_distribution(batch, action_masks=batch['action_mask'].bool()).distribution.probs))
    differences = [float((a-b).abs().max()) for a, b in zip(*outputs)]
    if any(value > 1e-7 for value in differences):
        raise ValueError(f'Initial outputs differ despite controlled initialization: {differences}')
    return dict(state_tensors_equal=True, same_parameter_count=True,
        parameters=sum(p.numel() for p in control.policy.parameters()),
        relational_enabled=dict(control=False, matchup=True), initial_timesteps=0,
        optimizers_empty=True, probes=len(probes), max_logit_difference=differences[0],
        max_value_difference=differences[1], max_probability_difference=differences[2],
        note='Same widened architecture and copied original fresh weights; extra input columns start at zero. Only the serialized feature toggle differs.')


def make_jobs(settings):
    return [dict(name=name, checkpoint=f'inputs/{name}-initial.zip', starting_timesteps=0,
        target_timesteps=settings['timesteps'], seed=settings['seed'], relational_enabled=enabled)
        for name, enabled in [('control', False), ('matchup', True)]]


def freeze(root, args):
    from sb3_contrib import MaskablePPO
    from firered_rl.shared_move_policy import policy_architecture
    from firered_rl.matchup_move_policy import create_ablation_checkpoint
    from firered_rl.showdown_env import validate_showdown
    from firered_rl.evaluation import validate_suite
    evidence = source_inputs(args.source_run, args.smoke)
    validate_showdown(ROOT)
    template = MaskablePPO.load(evidence['fresh'], device='cpu')
    if policy_architecture(template) != 'shared-moves' or getattr(template, 'observation_version', None) != 'v2':
        raise ValueError('Source fresh policy must use original shared-moves/v2 architecture')
    if template.num_timesteps != 0 or template.policy.optimizer.state:
        raise ValueError('Source fresh policy must have zero steps and empty optimizer state')
    settings = copy.deepcopy(evidence['settings'])
    settings.update(seed=261090, timesteps=256 if args.smoke else 1000000,
        environments=2 if args.smoke else 16, learning_rate=.0001, ent_coef=.01,
        eval_every=128 if args.smoke else 50000, early_stop_consecutive=2,
        selection='Highest validation wins/requested; control retains ties. No automatic promotion.',
        comparison='Same original fresh weights, widened architecture, optimizer state and budget; matchup features off/on only.')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs'; inputs.mkdir(); (inputs / 'suites').mkdir(); (root / 'console').mkdir()
    shutil.copyfile(evidence['fresh'], inputs / 'original-fresh.zip')
    shutil.copyfile(evidence['reward'], inputs / 'reward.json')
    shutil.copyfile(evidence['foundation'], inputs / 'foundation-mirror.zip')
    reference = 'inputs/foundation-mirror.zip'
    write_json(inputs / 'settings.json', settings)
    teams = {}
    for name, paths in evidence['teams'].items():
        folder = inputs / 'teams' / name; folder.mkdir(parents=True)
        teams[name] = []
        for index, path in enumerate(paths):
            target = folder / f'team-{index+1:04}.json'; shutil.copyfile(path, target)
            teams[name].append(str(target.relative_to(root)))
    write_json(inputs / 'mixture.json', dict(version=1, groups={
        'current': dict(weight=.9, directory=str(inputs / 'teams/train')),
        'fixed-mirror': dict(weight=.1, teams=[str(root / p) for p in teams['mirror']])}))
    suites = {}
    for name, path in evidence['suites'].items():
        data = read_json(path)
        if args.smoke:
            data['cases'] = data['cases'][:8]
            data['diagnostic'] = dict(data.get('diagnostic', {}), matchup_smoke=True, actual_battles=len(data['cases']))
        validate_suite(data)
        target = inputs / 'suites' / f'{name}.json'; write_json(target, data)
        suites[name] = str(target.relative_to(root))
    models = [create_ablation_checkpoint(template, inputs / f'{name}-initial.zip', enabled=enabled,
        seed=settings['seed'], training_settings=dict(n_steps=128, batch_size=128, n_epochs=4,
            learning_rate=settings['learning_rate'], ent_coef=settings['ent_coef'], gamma=.99,
            gae_lambda=.95, target_kl=.03, device='cpu'))
        for name, enabled in [('control', False), ('matchup', True)]]
    identity = initial_identity(models)
    write_json(inputs / 'initial-identity.json', identity)
    del models, template
    write_json(inputs / 'source-evidence.json', dict(source_run=str(evidence['source']),
        source_pins=evidence['source_pins'], source_smoke=evidence['source_smoke'],
        note='Source results, checkpoints, reward and team definitions are unchanged; heldout suites are reused development checks.'))
    jobs = make_jobs(settings)
    for job in jobs: (root / 'console' / (job['name'] + '.log')).touch()
    plan = dict(version=1, experiment='matchup-feature-ablation', smoke=args.smoke,
        settings=settings, teams=teams, suites=suites, jobs=jobs, foundation_reference=reference,
        initial_identity=identity, pins={str(p.relative_to(root)): checksum(p) for p in inputs.rglob('*') if p.is_file()})
    write_json(root / 'plan.json', plan)
    return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != 'matchup-feature-ablation':
        raise ValueError('Unsupported matchup ablation plan')
    for name, digest in plan['pins'].items():
        if checksum(inside(root, name)) != digest:
            raise ValueError(f'Pinned input changed: {name}')
    if plan['settings'] != read_json(root / 'inputs/settings.json') or plan['jobs'] != make_jobs(plan['settings']):
        raise ValueError('Saved settings or arm budgets changed')
    if plan['initial_identity'] != read_json(root / 'inputs/initial-identity.json'):
        raise ValueError('Initial identity evidence changed')
    if plan.get('foundation_reference') != 'inputs/foundation-mirror.zip':
        raise ValueError('The original specialist retention reference is required')
    for job in plan['jobs']:
        if checkpoint_steps(root / job['checkpoint']) != 0:
            raise ValueError('Ablation initial checkpoint contains training steps')
    for name, values in plan['teams'].items():
        expected = {inside(root, value) for value in values}
        if len(values) != len(expected) or set((root / 'inputs/teams' / name).glob('*.json')) != expected:
            raise ValueError(f'Pinned team membership changed: {name}')


def command(root, plan, job):
    result = transfer.command(root, plan, job)
    if result is not None and '--policy-architecture' in result:
        result[result.index('--policy-architecture') + 1] = 'matchup-moves'
    return result


def launch(root, plan, concurrency):
    if concurrency not in (1, 2): raise ValueError('Only one or two concurrent jobs are allowed')
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
                    raise RuntimeError(f"{job['name']} failed; inspect console log")
            if active: time.sleep(.25)
    finally:
        for _, proc, _ in active:
            if proc.poll() is None: proc.send_signal(signal.SIGINT)
        if active: print('Waiting for trainers to save and stop workers.', flush=True)
        for _, proc, reader in active: proc.wait(); reader.join()


def frozen(root, label, suite, **kwargs):
    from firered_rl.curriculum_checks import frozen_check
    print(f'Frozen check: {label}; no training.', flush=True)
    result = frozen_check(ROOT, root, label, suite, **kwargs)
    print(f"  {result['candidate_wins']}/{result['requested_battles']} wins ({rate(result):.1%}); truncated={result['truncated']}", flush=True)
    return result


def evaluate(root, plan):
    suites = {key: root / value for key, value in plan['suites'].items()}
    baselines = {name: frozen(root, 'baseline-' + name, suites['heldout'], baseline=name)
                 for name in ('random', 'greedy', 'rules')}
    initial = {job['name']: {label: frozen(root, job['name'] + '-initial-' + label,
        suite, checkpoint=root / job['checkpoint']) for label, suite in suites.items()} for job in plan['jobs']}
    reference = frozen(root, 'foundation-mirror-reference', suites['mirror'],
        checkpoint=root / plan['foundation_reference'])
    write_json(root / 'initial.json', dict(baselines=baselines, policies=initial, foundation_reference=reference))
    return baselines, initial, reference



def verify_saved_checks(root, result):
    """Do not declare completion from in-memory results without complete saved evidence."""
    from firered_rl.curriculum_checks import _complete_episode_log
    fields = ('status', 'candidate_identity', 'opponent_identity', 'input_suite_sha256',
        'effective_suite_sha256', 'reward_sha256', 'suite_sha256', 'requested_battles',
        'attempted_battles', 'completed_battles', 'truncated', 'candidate_wins', 'opponent_wins', 'draws')
    verified = set()

    def visit(value):
        if isinstance(value, dict):
            if 'result_directory' in value:
                relative = value['result_directory']
                try:
                    directory = inside(root, relative)
                    if directory.relative_to(root.resolve()).parts[0] != 'checks':
                        raise ValueError('not a check directory')
                    saved = read_json(directory / 'summary.json')
                    if (saved.get('status') != 'completed' or not _complete_episode_log(directory, saved)
                            or any(key not in value or key not in saved or value[key] != saved[key] for key in fields)
                            or ('audit_totals' in value and saved.get('audit_totals') != value['audit_totals'])):
                        raise ValueError('incomplete or inconsistent saved summary/episode log')
                except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                    raise RuntimeError(f'Saved frozen check is incomplete or inconsistent: {relative}. '
                        'Keep partial diagnostics and rerun with --continue to retry checks without repeating completed training.') from error
                verified.add(str(directory.relative_to(root.resolve())))
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)

    visit(result)
    if not verified:
        raise RuntimeError('No saved frozen checks found; rerun with --continue before declaring completion.')
    return dict(checked_directories=sorted(verified), complete_episode_logs=True)


def finish(root, plan, baselines, initial, reference):
    selected = transfer.selection(root, plan)  # Recorded before trained-policy held-out checks.
    results, cache, training = {}, {}, {}
    for job in plan['jobs']:
        results[job['name']] = {}
        for kind in ('best', 'final'):
            path = root / 'runs' / job['name'] / 'checkpoints' / (kind + '.zip'); digest = checksum(path)
            if digest not in cache:
                cache[digest] = {label: frozen(root, job['name'] + '-' + kind + '-' + label,
                    root / suite, checkpoint=path) for label, suite in plan['suites'].items()}
            checks = cache[digest]
            issues = transfer.readiness(checks, reference, plan['settings'])
            results[job['name']][kind] = dict(checkpoint=str(path.relative_to(root)), checkpoint_sha256=digest,
                checks=checks, readiness='not_ready' if issues else 'ready', issues=issues,
                mirror_reference='original_specialist',
                mirror_change_from_own_start=rate(checks['mirror']) - rate(initial[job['name']]['mirror']))
        summary = read_json(root / 'runs' / job['name'] / 'summary.json')
        training[job['name']] = dict(starting_timesteps=0, ending_timesteps=summary['player_timesteps'],
            added_timesteps=summary['player_timesteps'], max_new_timesteps=plan['settings']['timesteps'],
            elapsed_seconds=summary['elapsed_seconds'], stop_reason=summary['stop_reason'])
    result = dict(status='smoke_completed' if plan['smoke'] else 'completed', diagnostic_only=True,
        selected_by_validation=selected['selected'], policies=results, baselines=baselines, initial=initial,
        training=training, foundation_reference=reference, initial_identity=plan['initial_identity'],
        no_automatic_promotion=True, heldout_role='reused development checks')
    result['saved_evidence'] = verify_saved_checks(root, result)
    write_json(root / 'result.json', result)
    return result


def collect(root, plan):
    result = read_json(root / 'result.json') if (root / 'result.json').exists() else {}
    initial = read_json(root / 'initial.json').get('policies', {}) if (root / 'initial.json').exists() else {}
    lines = ['# Matchup-feature ablation', '', 'SMOKE ONLY; not learning evidence.' if plan['smoke'] else
        'Same fresh weights, optimizer state, architecture, teams, reward and PPO settings; control disables matchup features, treatment enables them.', '',
        'All battles are against Rules. Validation selects checkpoints. Reused held-out suites are development checks; no automatic promotion.', '',
        '| Policy | Checkpoint | Validation | Seen teams | Heldout | Rotated | Fixed mirror | Readiness |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    records = []
    for job in plan['jobs']:
        name = job['name']
        records.append((name, 'initial', dict(checks=initial.get(name, {}), readiness='baseline')))
        records += [(name, kind, record) for kind, record in result.get('policies', {}).get(name, {}).items()]
    for name, kind, record in records:
        values = [f'{rate(record["checks"][key]):.1%}' if key in record['checks'] else '—' for key in CONDITIONS]
        lines.append(f"| {name} | {kind} | {' | '.join(values)} | {record['readiness']} |")
    lines += ['', '## Held-out immunity audit', '',
        'All candidate battles; current defender before the opponent action. Unsupported moves excluded. Legal alternatives are not guaranteed hits or damage.', '',
        '| Policy | Checkpoint | Classified moves | Immune choices | With legal alternative |', '|---|---|---:|---:|---:|']
    for name, kind, record in records:
        lines.append(f"| {name} | {kind} | {' | '.join(transfer.immunity_counts(record['checks'].get('heldout', {})))} |")
    lines += ['', '## Training budgets', '', '| Policy | Added steps | Maximum | Minutes | Stop reason |', '|---|---:|---:|---:|---|']
    for job in plan['jobs']:
        path = root / 'runs' / job['name'] / 'summary.json'; summary = read_json(path) if path.exists() else {}
        lines.append(f"| {job['name']} | {summary.get('player_timesteps', 0):,} | {plan['settings']['timesteps']:,} | {summary.get('elapsed_seconds', 0)/60:.1f} | {summary.get('stop_reason', 'pending')} |")
    lines += ['', 'Readiness issues:', '']
    for name, kind, record in records:
        lines += [f'- {name} {kind}: {issue}' for issue in record.get('issues', [])]
    lines += ['', 'Mirror readiness uses the original specialist reference; each fresh arm also reports its change from the common untrained starting state.',
        'Both arms train on 90% current stage and 10% fixed mirror. Original source data and models are unchanged.', '',
        '## Frozen checks', '', '| Check | Wins/requested | Rate | Truncated |', '|---|---:|---:|---:|']
    for path in sorted((root / 'checks').glob('*/summary.json')):
        value = read_json(path)
        if value.get('status') == 'completed':
            lines.append(f"| {path.parent.name} | {value['candidate_wins']}/{value['requested_battles']} | {rate(value):.1%} | {value['truncated']} |")
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    temporary = root / 'matchup-results.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if not path.is_file() or path.suffix not in ('.json', '.jsonl', '.md', '.csv'): continue
            if relative.parts[0] in ('inputs', 'checks') or path.name in (
                'plan.json', 'status.json', 'report.md', 'memory.csv', 'result.json', 'initial.json', 'selection.json',
                'config.json', 'summary.json', 'failure.json', 'updates.jsonl', 'battle-blocks.jsonl', 'progress.csv',
                'switching.json', 'evaluation-history.json', 'evaluations.md', 'evaluations.csv', 'best.json'):
                archive.write(path, relative)
    temporary.replace(root / 'matchup-results.zip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('.local/run/type-transfer-01'))
    parser.add_argument('--output-dir', type=Path, default=Path('.local/run/matchup-ablation-01'))
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-teams', nargs='?', const='all', choices=('all', 'train', 'mirror', 'validation', 'heldout'))
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--full-json', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Tiny integration test; not evidence of learning')
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.limit < 0: parser.error('--limit cannot be negative')
    if args.inspect_teams is not None:
        plan = read_json(root / 'plan.json'); verify(root, plan)
        transfer.inspect(root, plan, args.inspect_teams, args.limit, args.full_json)
        return 0
    import torch
    torch.set_num_threads(1)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = (root.parent / ('.' + root.name + '.matchup.lock')).open('a')
    plan = None; owned = False; stop = threading.Event(); monitor = None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); owned = True
        if args.continuing:
            plan = read_json(root / 'plan.json'); verify(root, plan)
        else: plan = freeze(root, args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {shlex.quote(str(root / 'console'))}/*.log\nMemory: watch -n 2 free -h\nInspect: python run_matchup_ablation.py --output-dir {shlex.quote(str(root))} --inspect-teams train", flush=True)
        if args.prepare_only:
            write_json(root / 'status.json', dict(status='prepared')); return 0
        monitor = threading.Thread(target=memory_monitor, args=(root, stop), daemon=True); monitor.start()
        write_json(root / 'status.json', dict(status='running', phase='initial-checks'))
        baselines, initial, reference = evaluate(root, plan)
        collect(root, plan)
        write_json(root / 'status.json', dict(status='running', phase='training'))
        launch(root, plan, args.jobs)
        verify(root, plan)
        write_json(root / 'status.json', dict(status='running', phase='frozen-checks'))
        result = finish(root, plan, baselines, initial, reference)
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
            print(f"Report: {root / 'report.md'}\nUpload: {root / 'matchup-results.zip'}", flush=True)
        signal.signal(signal.SIGTERM, previous); lock.close()


if __name__ == '__main__': raise SystemExit(main())
