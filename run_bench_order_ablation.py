#!/usr/bin/env python3
"""Compare fixed versus shuffled candidate bench order during matched PPO continuations.

Watch training: tail -n 20 -F RUN/console/*.log
Watch memory: watch -n 2 free -h. Ctrl+C pauses; --continue resumes saved budgets.
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
import run_replacement_diagnostic as diagnostic
import run_transfer_lab as transfer
from run_fixed_team_lab import checksum, write_json, interrupt
from run_parallel_training import checkpoint_steps, pump
from run_staged_team_lab import memory_monitor

read_json, inside = transfer.read_json, transfer.inside
EXPERIMENT = 'bench-order-ablation'
CONDITIONS = ('validation', 'validation-original', 'seen-original', 'seen-reversed',
              'heldout-original', 'heldout-reversed', 'mirror')


def implementation_identity():
    result = diagnostic.implementation_identity()
    for path in (Path(__file__), ROOT / 'mgba/train_ppo.py', ROOT / 'src/firered_rl/bench_order.py',
                 ROOT / 'src/firered_rl/training_run.py'):
        if path.exists(): result[str(path.relative_to(ROOT))] = checksum(path)
    return result


def combined_validation(original, reversed_suite):
    """Equal-weight original/reversed orders; battle seeds remain paired across orders."""
    from firered_rl.evaluation import validate_suite
    original, reversed_suite = validate_suite(original), validate_suite(reversed_suite)
    if len(original['cases']) != len(reversed_suite['cases']):
        raise ValueError('Balanced validation requires equal original and reversed budgets')
    combined = copy.deepcopy(original)
    combined['assets'].update(copy.deepcopy(reversed_suite['assets']))
    cases = copy.deepcopy(original['cases'])
    max_matchup = max(case['matchup'] for case in cases)
    mapping = []
    for left, right in zip(original['cases'], reversed_suite['cases']):
        if any(left[key] != right[key] for key in ('episode', 'matchup', 'seed', 'candidate_player')):
            raise ValueError('Original/reversed validation schedules differ')
        case = copy.deepcopy(right)
        case['episode'] = len(cases) + 1
        case['matchup'] += max_matchup
        cases.append(case)
        mapping.append(dict(original_episode=left['episode'], reversed_episode=case['episode'],
            seed=left['seed'], candidate_player=left['candidate_player']))
    combined.update(cases=cases, swap_sides=False)
    combined['diagnostic'] = dict(combined.get('diagnostic', {}),
        role='primary balanced original/reversed validation; checkpoint selection only',
        bench_orders=['original', 'candidate-bench-reversed'], equal_order_weight=True,
        actual_battles=len(cases), paired_orders=mapping)
    return validate_suite(combined)


def make_jobs(settings):
    return [dict(name=name, checkpoint='inputs/start.zip', starting_timesteps=settings['starting_timesteps'],
        target_timesteps=settings['starting_timesteps'] + settings['timesteps'],
        seed=settings['seed'], shuffle_bench_order=shuffle)
        for name, shuffle in (('control', False), ('bench-shuffle', True))]


def freeze(root, args):
    from firered_rl.evaluation import validate_suite
    from firered_rl.replacement_diagnostics import transform_suite
    from firered_rl.showdown_env import validate_showdown
    evidence = diagnostic.source_inputs(args.source_run, args.smoke)
    validate_showdown(ROOT)
    if root == evidence['source'] or root.is_relative_to(evidence['source']):
        raise ValueError('Output must be outside the source experiment')
    if type(args.timesteps) is not int or args.timesteps < 1:
        raise ValueError('Additional timesteps must be a positive integer')
    start = evidence['models']['best']
    settings = dict(seed=261100, starting_timesteps=start['timesteps'],
        timesteps=256 if args.smoke else args.timesteps, environments=2 if args.smoke else 16,
        learning_rate=.0001, ent_coef=.01, rollout_steps=128, batch_size=128, epochs=4,
        eval_every=128 if args.smoke else 50000, max_steps=100,
        selection='Highest wins/requested on balanced original/reversed validation; control retains ties. Heldout never selects.',
        early_stopping=False, no_automatic_promotion=True, current_weight=.9, mirror_weight=.1,
        practical_gain_goal=.05, mirror_tradeoff=.05, single_order_tradeoff=.03,
        comparison='Same validation-best weights and optimizer state, seed, budget, architecture, reward and canonical team mixture. Treatment only shuffles the candidate bench during nonfixed training episodes.')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs'; inputs.mkdir(); (inputs / 'suites').mkdir(); (root / 'console').mkdir()
    shutil.copyfile(start['path'], inputs / 'start.zip')
    shutil.copyfile(evidence['reward'], inputs / 'reward.json')
    write_json(inputs / 'settings.json', settings)
    teams = {}
    for group, paths in evidence['teams'].items():
        directory = inputs / 'teams' / group; directory.mkdir(parents=True)
        teams[group] = []
        for index, path in enumerate(paths):
            target = directory / f'team-{index+1:04}.json'; shutil.copyfile(path, target)
            teams[group].append(str(target.relative_to(root)))
    write_json(inputs / 'mixture.json', dict(version=1, groups={
        'current': dict(weight=.9, directory=str(inputs / 'teams/train')),
        'fixed-mirror': dict(weight=.1, teams=[str(root / path) for path in teams['mirror']])}))
    suites = {}
    for name, path in evidence['suites'].items():
        original = read_json(path)
        if args.smoke:
            original['cases'] = original['cases'][:8]
            original['diagnostic'] = dict(original.get('diagnostic', {}), bench_order_smoke=True,
                actual_battles=len(original['cases']))
        validate_suite(original)
        if original['max_steps'] != settings['max_steps']:
            raise ValueError('Source suite max_steps must match controlled training settings')
        if name == 'mirror': values = {'mirror': original}
        else:
            reversed_suite = transform_suite(original)
            group = 'seen' if name == 'test' else name
            values = {group + '-original': original}
            if name == 'validation': values['validation'] = combined_validation(original, reversed_suite)
            else: values[group + '-reversed'] = reversed_suite
        for label, data in values.items():
            target = inputs / 'suites' / f'{label}.json'; write_json(target, data)
            suites[label] = str(target.relative_to(root))
    jobs = make_jobs(settings)
    source = dict(source_run=str(evidence['source']), source_pins=evidence['source_pins'],
        source_plan_sha256=evidence['source_plan_sha256'], source_result_sha256=evidence['source_result_sha256'],
        source_selected=evidence['source_selected'], source_smoke=evidence['source_smoke'],
        start_checkpoint=str(start['path']), start_sha256=start['sha256'], starting_timesteps=start['timesteps'],
        initial_identity='Both jobs load the exact same checkpoint bytes, including optimizer state. No migration or optimizer reset.',
        preservation='Source experiments, source final checkpoint and 400k reference are unchanged.')
    write_json(inputs / 'source-evidence.json', source)
    write_json(inputs / 'configuration.json', dict(settings=settings, teams=teams, suites=suites, jobs=jobs))
    for job in jobs: (root / 'console' / (job['name'] + '.log')).touch()
    plan = dict(version=1, experiment=EXPERIMENT, smoke=bool(args.smoke), settings=settings,
        teams=teams, suites=suites, jobs=jobs, implementation=implementation_identity(),
        source_selected=evidence['source_selected'], source_start_sha256=start['sha256'],
        heldout_role='Reused development checks; not an untouched generalization benchmark.',
        no_automatic_promotion=True, pins={str(path.relative_to(root)): checksum(path)
            for path in inputs.rglob('*') if path.is_file()})
    write_json(root / 'plan.json', plan)
    return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != EXPERIMENT:
        raise ValueError('Unsupported bench-order ablation plan')
    if plan.get('implementation') != implementation_identity():
        raise ValueError('Experiment implementation changed; use a new output directory')
    expected = {inside(root, path) for path in plan['pins']}
    if expected != {path.resolve() for path in (root / 'inputs').rglob('*') if path.is_file()}:
        raise ValueError('Pinned input membership changed')
    for path, expected_hash in plan['pins'].items():
        if checksum(inside(root, path)) != expected_hash:
            raise ValueError(f'Pinned input changed: {path}')
    config = read_json(root / 'inputs/configuration.json')
    if any(plan[key] != config[key] for key in ('settings', 'teams', 'suites', 'jobs')):
        raise ValueError('Saved experiment configuration changed')
    if plan['settings'] != read_json(root / 'inputs/settings.json') or plan['jobs'] != make_jobs(plan['settings']):
        raise ValueError('Saved budgets or training arms changed')
    if set(plan['suites']) != set(CONDITIONS): raise ValueError('Frozen condition membership changed')
    if (plan['pins']['inputs/start.zip'] != plan['source_start_sha256'] or
            checkpoint_steps(root / 'inputs/start.zip') != plan['settings']['starting_timesteps']):
        raise ValueError('Initial checkpoint provenance changed')
    if plan.get('no_automatic_promotion') is not True or plan['settings']['early_stopping'] is not False:
        raise ValueError('This comparison has fixed budgets and no automatic promotion')
    for path in [*plan['suites'].values(), *[p for group in plan['teams'].values() for p in group]]:
        if path not in plan['pins']: raise ValueError(f'Unpinned suite or team: {path}')


def command(root, plan, job):
    base = [sys.executable, '-u', str(ROOT / 'mgba/train_ppo.py')]
    run = root / 'runs' / job['name']
    if run.exists():
        summary = read_json(run / 'summary.json') if (run / 'summary.json').exists() else {}
        steps = transfer.recovered_steps(run)
        if summary.get('status') == 'completed' and steps >= job['target_timesteps']: return None
        return base + ['--output-dir', str(run), '--if-exists', 'continue',
            '--total-timesteps', str(max(1, job['target_timesteps'] - steps))]
    s = plan['settings']
    return base + ['--backend', 'showdown', '--resume', str(root / job['checkpoint']),
        '--policy-architecture', 'matchup-moves', '--observation-version', 'v2', '--rule-based-opponent',
        '--team-mixture', str(root / 'inputs/mixture.json'), '--reward-config', str(root / 'inputs/reward.json'),
        '--learning-rate', str(s['learning_rate']), '--ent-coef', str(s['ent_coef']),
        '--seed', str(job['seed']), '--environments', str(s['environments']),
        '--rollout-steps', str(s['rollout_steps']), '--batch-size', str(s['batch_size']), '--epochs', str(s['epochs']),
        '--torch-threads', '1', '--device', 'cpu', '--showdown-workers', 'process',
        '--max-steps', str(s['max_steps']), '--total-timesteps', str(s['timesteps']),
        '--eval-suite', str(root / plan['suites']['validation']), '--eval-every', str(s['eval_every']),
        '--eval-environments', '4', '--no-eval-stop', '--latest-every-updates', '1', '--checkpoint-every', str(s['eval_every']),
        '--keep-checkpoints', '3', '--keep-evaluations', '2', '--battle-log', '--no-readable-log',
        '--no-step-logs', '--log-max-mb', '2', '--log-backups', '1',
        '--shuffle-bench-order' if job['shuffle_bench_order'] else '--no-shuffle-bench-order',
        '--output-dir', str(run), '--if-exists', 'cancel']


def launch(root, plan, jobs):
    if jobs not in (1, 2): raise ValueError('Only one or two concurrent training jobs are allowed')
    pending, active = list(plan['jobs']), []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        while pending or active:
            while pending and len(active) < jobs:
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


def frozen(root, plan, label, condition, checkpoint):
    """Use the strict replacement diagnostic cache, with assistance always disabled."""
    relative = str(Path(checkpoint).relative_to(root))
    frozen_plan = dict(models=dict(candidate=dict(checkpoint=relative, sha256=checksum(checkpoint),
        timesteps=checkpoint_steps(checkpoint))), suites=plan['suites'], pins=plan['pins'],
        implementation=plan['implementation'])
    check = dict(label=label, checkpoint='candidate', baseline=None, pool=condition,
        order='reversed' if condition.endswith('-reversed') else 'original', assisted=False, suite=condition)
    print(f'Frozen check: {label}; learned PPO only, no Rules assistance.', flush=True)
    result = diagnostic.run_check(root, frozen_plan, check)
    print(f"  {result['candidate_wins']}/{result['requested_battles']} wins ({result['wins_per_requested']:.1%}); truncated={result['truncated']}", flush=True)
    return result


def evaluate_initial(root, plan):
    checks = {condition: frozen(root, plan, 'initial-' + condition, condition, root / 'inputs/start.zip')
              for condition in CONDITIONS}
    result = dict(checkpoint='inputs/start.zip', checkpoint_sha256=plan['source_start_sha256'],
        shared_by_both_arms=True, checks=checks)
    write_json(root / 'initial.json', result)
    return result


def validate_saved_checks(root, records):
    fields = ('status', 'candidate_identity', 'opponent_identity', 'input_suite_sha256',
        'effective_suite_sha256', 'reward_sha256', 'implementation', 'replacement_assistance',
        'suite_sha256', 'requested_battles', 'attempted_battles', 'completed_battles',
        'truncated', 'candidate_wins', 'opponent_wins', 'draws', 'audit_totals')
    checked = set()
    for record in records:
        for value in record['checks'].values():
            directory = inside(root, value['result_directory'])
            if directory.relative_to(root).parts[0] != 'checks': raise ValueError('Invalid saved check directory')
            saved = read_json(directory / 'summary.json')
            if (not diagnostic.complete_audit(directory, saved)
                    or any(key not in saved or value.get(key) != saved[key] for key in fields)):
                raise RuntimeError(f'Saved frozen check incomplete or inconsistent: {directory}. Use --continue to preserve and retry it.')
            checked.add(value['result_directory'])
    if not checked: raise RuntimeError('No complete saved frozen checks found')
    return dict(checked_directories=sorted(checked), complete_episode_logs=True, complete_replacement_audits=True)


def comparison(policies, settings):
    control, treatment = (policies[name]['best']['checks'] for name in ('control', 'bench-shuffle'))
    differences = {condition: treatment[condition]['wins_per_requested'] - control[condition]['wins_per_requested']
        for condition in CONDITIONS}
    average = (differences['heldout-original'] + differences['heldout-reversed']) / 2
    tradeoffs = []
    if differences['mirror'] < -settings['mirror_tradeoff']:
        tradeoffs.append('Fixed-mirror win rate fell more than 5 percentage points versus control.')
    if any(differences[condition] < -settings['single_order_tradeoff'] for condition in ('heldout-original', 'heldout-reversed')):
        tradeoffs.append('At least one heldout bench order fell more than 3 percentage points versus control.')
    return dict(primary='Validation-selected checkpoints; heldout win rate averaged equally across original and reversed bench orders.',
        heldout_average_difference=average, per_condition_differences=differences,
        practical_gain_goal=settings['practical_gain_goal'], practical_goal_met=average >= settings['practical_gain_goal'],
        tradeoffs=tradeoffs, no_automatic_promotion=True,
        scope='One paired continuation, reused development pools. Point differences are not statistical confidence or independent seed replication. Lowest-slot frequency alone is not a quality target.')


def training_provenance(root, plan, job, summary):
    run = root / 'runs' / job['name']
    config = read_json(run / 'config.json')
    first_path = run / 'sessions/000001/config.json'
    first = read_json(first_path) if first_path.exists() else config
    if first.get('session', {}).get('id') != 1:
        raise ValueError('Missing first-session checkpoint provenance')
    if first.get('resume_checkpoint', {}).get('sha256') != plan['source_start_sha256']:
        raise ValueError('Training did not start from the shared validation-best checkpoint')
    if any(item.get('shuffle_bench_order') is not job['shuffle_bench_order'] for item in (config, first)):
        raise ValueError('Actual trainer bench-order flag differs from the saved treatment')
    if not job['shuffle_bench_order']:
        if summary.get('bench_order_shuffle') is not None:
            raise ValueError('Control unexpectedly reports bench-order shuffling')
        return dict(initial_checkpoint_sha256=plan['source_start_sha256'], shuffle_bench_order=False)
    if summary.get('bench_order_shuffle') is None:
        raise ValueError('Treatment has no actual bench-order shuffle evidence')
    sessions = {summary.get('session_id', 1): summary}
    for path in sorted((run / 'sessions').glob('*/summary.json')):
        value = read_json(path)
        sessions.setdefault(value.get('session_id', int(path.parent.name)), value)
    aggregate = dict.fromkeys(('episode_selections', 'eligible', 'skipped_fixed_pair', 'identity', 'changed'), 0)
    for value in sessions.values():
        shuffle = value.get('bench_order_shuffle', {})
        if (shuffle.get('version') != 1 or shuffle.get('scope') != 'candidate_player_1_nonfixed_groups'
                or any(shuffle.get(key) != 'unchanged' for key in ('lead', 'opponent', 'moves'))
                or shuffle.get('rehearsal') != 'fixed-pair mixture groups are excluded'):
            raise ValueError('Treatment shuffle scope differs from the controlled experiment')
        counts = shuffle.get('counts', {})
        if any(type(counts.get(key)) is not int or counts[key] < 0 for key in aggregate):
            raise ValueError('Invalid bench-order shuffle counters')
        if (counts['episode_selections'] != counts['eligible'] + counts['skipped_fixed_pair']
                or counts['eligible'] != counts['identity'] + counts['changed']):
            raise ValueError('Inconsistent bench-order shuffle counters')
        for key in aggregate: aggregate[key] += counts[key]
    if not plan['smoke'] and (aggregate['eligible'] == 0 or aggregate['changed'] == 0):
        raise ValueError('Normal treatment run did not actually shuffle eligible benches')
    return dict(initial_checkpoint_sha256=plan['source_start_sha256'], shuffle_bench_order=True,
        sessions=len(sessions), aggregate_episode_start_counts=aggregate)


def finish(root, plan, initial):
    requested = len(read_json(root / plan['suites']['validation'])['cases'])
    for job in plan['jobs']:
        metadata = read_json(root / 'runs' / job['name'] / 'checkpoints/best.json')
        if metadata.get('requested') != requested:
            raise ValueError('Best checkpoint was not selected on the balanced validation budget')
    selected = transfer.selection(root, plan)  # Persisted before trained heldout checks.
    policies, training = {}, {}
    cache = {initial['checkpoint_sha256']: initial['checks']}
    for job in plan['jobs']:
        policies[job['name']] = {}
        summary = read_json(root / 'runs' / job['name'] / 'summary.json')
        if (summary.get('status') != 'completed' or summary.get('stop_reason') != 'timestep_budget'
                or summary['player_timesteps'] < job['target_timesteps']):
            raise ValueError('Matched continuation did not complete its fixed timestep budget')
        provenance = training_provenance(root, plan, job, summary)
        for kind in ('best', 'final'):
            path = root / 'runs' / job['name'] / 'checkpoints' / (kind + '.zip')
            sha = checksum(path)
            if kind == 'final' and checkpoint_steps(path) != summary['player_timesteps']:
                raise ValueError('Final checkpoint timestep count differs from completed training')
            if sha not in cache:
                cache[sha] = {condition: frozen(root, plan, job['name'] + '-' + kind + '-' + condition,
                    condition, path) for condition in CONDITIONS}
            policies[job['name']][kind] = dict(checkpoint=str(path.relative_to(root)),
                checkpoint_sha256=sha, timesteps=checkpoint_steps(path), checks=cache[sha],
                mirror_change_from_own_start=cache[sha]['mirror']['wins_per_requested']-initial['checks']['mirror']['wins_per_requested'])
        training[job['name']] = dict(starting_timesteps=job['starting_timesteps'],
            ending_timesteps=summary['player_timesteps'], added_timesteps=summary['player_timesteps']-job['starting_timesteps'],
            max_new_timesteps=plan['settings']['timesteps'], elapsed_seconds=summary['elapsed_seconds'],
            stop_reason=summary['stop_reason'], team_mixture_sampling=summary.get('team_mixture_sampling'),
            bench_order_shuffle=summary.get('bench_order_shuffle'), provenance=provenance)
    records = [initial, *[record for arm in policies.values() for record in arm.values()]]
    result = dict(status='smoke_completed' if plan['smoke'] else 'completed', diagnostic_only=True,
        no_automatic_promotion=True, selected_by_validation=selected['selected'], initial=initial,
        policies=policies, training=training, comparison=comparison(policies, plan['settings']),
        heldout_role=plan['heldout_role'], saved_evidence=validate_saved_checks(root, records))
    write_json(root / 'result.json', result)
    return result


def collect(root, plan):
    result = read_json(root / 'result.json') if (root / 'result.json').exists() else {}
    initial = read_json(root / 'initial.json') if (root / 'initial.json').exists() else {}
    status = read_json(root / 'status.json') if (root / 'status.json').exists() else {'status': 'prepared'}
    lines = ['# Bench-order training comparison', '', 'SMOKE ONLY; not learning evidence.' if plan['smoke'] else
        'Two matched continuations from the same validation-best matchup checkpoint, including its optimizer state.', '',
        'Control keeps source party order. Bench-shuffle randomly permutes only the candidate bench during nonfixed training episodes; the lead, opponent, moves, rewards and fixed-mirror rehearsal stay unchanged.', '',
        'All frozen scores below are learned PPO against Rules, with no Rules assistance. Validation averages original and reversed bench orders equally; heldout results never select checkpoints. Heldout pools are reused development checks.', '',
        '| Policy | Checkpoint | Total steps | Balanced validation | Original validation | Seen original | Seen reversed | Heldout original | Heldout reversed | Fixed mirror |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    records = []
    if initial: records.append(('shared', 'initial', dict(initial, timesteps=plan['settings']['starting_timesteps'])))
    for name in ('control', 'bench-shuffle'):
        records.extend((name, kind, record) for kind, record in result.get('policies', {}).get(name, {}).items())
    for name, kind, record in records:
        scores = [f"{record['checks'][condition]['wins_per_requested']:.1%}" for condition in CONDITIONS]
        lines.append(f"| {name} | {kind} | {record['timesteps']:,} | {' | '.join(scores)} |")
    if not records: lines.append('| — | pending | — | — | — | — | — | — | — | — |')
    lines += ['', f"Run status: {status['status']}.", '',
        '## Training budgets', '', '| Arm | New steps | Maximum | Minutes | Stop reason |', '|---|---:|---:|---:|---|']
    for job in plan['jobs']:
        path = root / 'runs' / job['name'] / 'summary.json'
        summary = read_json(path) if path.exists() else {}
        added = max(0, summary.get('player_timesteps', job['starting_timesteps'])-job['starting_timesteps'])
        lines.append(f"| {job['name']} | {added:,} | {plan['settings']['timesteps']:,} | {summary.get('elapsed_seconds', 0)/60:.1f} | {summary.get('stop_reason', 'pending')} |")
    lines += ['', '## Forced replacement choices', '',
        'Only forced replacements with at least two legal switch actions are counted. Lowest-slot frequency is descriptive; 50% is not an optimality target.', '',
        '| Policy | Checkpoint | Pool/order | Eligible | Lowest-slot choices | Rules disagreements | Overrides |', '|---|---|---|---:|---:|---:|---:|']
    for name, kind, record in records:
        for condition in ('seen-original', 'seen-reversed', 'heldout-original', 'heldout-reversed'):
            totals = record['checks'][condition]['audit_totals']
            lines.append(f"| {name} | {kind} | {condition} | {totals['replacement_eligible']} | {totals['replacement_actual_lowest_legal']} | {totals['replacement_base_rules_disagreements']} | {totals['replacement_overrides_changed']} |")
    lines += ['', '## Retention and immunity checks', '',
        '| Policy | Checkpoint | Mirror change from shared start | Heldout classified moves | Immune moves |',
        '|---|---|---:|---:|---:|']
    for name, kind, record in records:
        counts = record['checks']['heldout-original']['audit_totals']
        change = record.get('mirror_change_from_own_start', 0.)
        lines.append(f"| {name} | {kind} | {change*100:+.1f} pp | {counts.get('matchup_classified_move_actions', 'N/A')} | {counts.get('current_target_immune_move_actions', 'N/A')} |")
    if records:
        all_checks = {check['result_directory']: check for _, _, record in records for check in record['checks'].values()}
        lines += ['', f"Frozen saved checks: {len(all_checks)}; battles: {sum(check['requested_battles'] for check in all_checks.values())}; truncated: {sum(check['truncated'] for check in all_checks.values())}."]
    measured = result.get('comparison')
    if measured:
        lines += ['', '## Predeclared comparison', '',
            f"Validation-selected heldout mean across both orders: treatment minus control = {100*measured['heldout_average_difference']:+.1f} percentage points.",
            'The practical improvement goal is +5 points, with tradeoff flags for mirror loss over 5 points or either heldout order losing over 3 points. These are review criteria, not statistical guarantees or automatic promotion gates.',
            measured['scope'], '', *[f'- {note}' for note in measured['tradeoffs']]]
    lines += ['', 'All initial/source checkpoint bytes are pinned. Each arm receives the same additional-step budget; there is no early success stop. Fixed mirror remains 10% of training episodes.', '',
        'Watch jobs: `tail -n 20 -F ' + str(root / 'console') + '/*.log`',
        'Watch memory: `watch -n 2 free -h`',
        'Inspect teams: `python run_bench_order_ablation.py --output-dir ' + shlex.quote(str(root)) + ' --inspect-teams train --full-json`',
        'Settings: `cat ' + str(root / 'inputs/settings.json') + '`',
        'Pause with Ctrl+C once and wait for cleanup. Resume with `python run_bench_order_ablation.py --output-dir ' + shlex.quote(str(root)) + ' --continue --jobs 2`. Saved configuration and budgets stay fixed.',
        'Each job has console logs, evaluations.md, updates.jsonl and PPO CSV metrics. checks/ contains frozen results and all-battle replacement audits. memory.csv includes validation peaks.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    presentation = ['# Bench-order experiment summary', '',
        'Question: does shuffling the two bench Pokémon during training improve learned replacement decisions and robustness to party order?', '',
        'Both agents start from the same trained model and optimizer. Only candidate bench ordering during nonfixed training differs; there is no Rules assistance.', '',
        f"Status: {status['status']}. Additional budget: {plan['settings']['timesteps']:,} steps per arm.", '',
        'Best checkpoints are selected by balanced validation, not heldout scores. Final checkpoints are shown separately at the fixed training budget.', '',
        '## Results', '', *lines[8:8+len(records)+2]]
    if measured:
        presentation += ['', f"Primary heldout difference: {100*measured['heldout_average_difference']:+.1f} percentage points, averaged over both bench orders.",
            'One paired continuation on reused development teams; not independent seed replication or a final generalization benchmark.',
            'No automatic promotion. Review move choice, forced replacement behavior, mirror retention and both orders before expanding the task.']
    (root / 'presentation-summary.md').write_text('\n'.join(presentation) + '\n')
    temporary = root / 'bench-order-results.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if (path.is_file() and path.suffix in ('.json', '.jsonl', '.md', '.csv')
                    and relative.parts[0] != 'console'):
                archive.write(path, relative)
    temporary.replace(root / 'bench-order-results.zip')


def inspect(root, plan, group, limit, full):
    print('Canonical training mixture (bench-shuffle permutes candidate benches at episode start):')
    print((root / 'inputs/mixture.json').read_text())
    for name in ('train', 'mirror', 'validation', 'heldout') if group == 'all' else (group,):
        if name in plan['teams']:
            values = [(str(root / path), read_json(root / path)) for path in plan['teams'][name]]
        else:
            path = root / plan['suites'][name + '-original']
            values = [(str(path) + ' asset ' + key, value['team']) for key, value in read_json(path)['assets'].items()]
        print(f'\n{name}: {len(values)} teams')
        for path, team in values[:limit]:
            print(path)
            if full: print(json.dumps(team, indent=2))
            else:
                for mon in team['party']: print(f"  {mon['species']}: {', '.join(mon['moves'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('.local/run/matchup-ablation-01'))
    parser.add_argument('--output-dir', type=Path, default=Path('.local/run/bench-order-ablation-01'))
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--timesteps', type=int, default=None, help='Additional steps per arm when creating a run; default 500000')
    parser.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-teams', nargs='?', const='all', choices=('all', 'train', 'mirror', 'validation', 'heldout'))
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--full-json', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.limit < 0: parser.error('--limit cannot be negative')
    if args.timesteps is not None and args.timesteps < 1: parser.error('--timesteps must be positive')
    if args.continuing and args.timesteps is not None: parser.error('--continue uses the saved budget; do not supply --timesteps')
    if args.inspect_teams is not None:
        plan = read_json(root / 'plan.json'); verify(root, plan)
        inspect(root, plan, args.inspect_teams, args.limit, args.full_json); return 0
    if args.timesteps is None: args.timesteps = 500000
    import torch
    torch.set_num_threads(1)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = (root.parent / ('.' + root.name + '.bench-order.lock')).open('a')
    plan = None; owned = False; stopped = threading.Event(); monitor = None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); owned = True
        if args.continuing:
            plan = read_json(root / 'plan.json'); verify(root, plan)
        else: plan = freeze(root, args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {shlex.quote(str(root / 'console'))}/*.log\nMemory: watch -n 2 free -h\nInspect: python run_bench_order_ablation.py --output-dir {shlex.quote(str(root))} --inspect-teams train --full-json", flush=True)
        if args.prepare_only:
            write_json(root / 'status.json', dict(status='prepared')); return 0
        monitor = threading.Thread(target=memory_monitor, args=(root, stopped), daemon=True); monitor.start()
        write_json(root / 'status.json', dict(status='running', phase='initial-checks'))
        initial = evaluate_initial(root, plan)
        collect(root, plan)
        write_json(root / 'status.json', dict(status='running', phase='training'))
        launch(root, plan, args.jobs)
        verify(root, plan)
        write_json(root / 'status.json', dict(status='running', phase='frozen-checks'))
        result = finish(root, plan, initial)
        write_json(root / 'status.json', dict(status=result['status'], no_automatic_promotion=True))
        return 0
    except (KeyboardInterrupt, Exception) as error:
        if plan is not None and owned:
            write_json(root / 'status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', message=str(error)))
        print(f'Stopped: {error or "interrupted"}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        stopped.set()
        if monitor: monitor.join()
        if plan is not None and owned:
            collect(root, plan)
            print(f"Report: {root / 'report.md'}\nPresentation: {root / 'presentation-summary.md'}\nUpload: {root / 'bench-order-results.zip'}", flush=True)
        signal.signal(signal.SIGTERM, previous); lock.close()


if __name__ == '__main__': raise SystemExit(main())
