#!/usr/bin/env python3
"""Train two matched continuations with direct forced-replacement features off/on.

Watch: tail -n 20 -F RUN/console/*.log. Memory: watch -n 2 free -h.
Ctrl+C pauses safely; --continue resumes the saved bounded budget.
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
import run_bench_order_ablation as bench
import run_presentation_check as presentation
import run_transfer_lab as transfer
from run_fixed_team_lab import checksum, write_json, interrupt
from run_parallel_training import checkpoint_steps, pump
from run_staged_team_lab import memory_monitor
read_json, inside = transfer.read_json, transfer.inside
EXPERIMENT = 'switch-feature-ablation'
CONDITIONS = ('seen-original', 'seen-reversed', 'heldout-original', 'heldout-reversed', 'mirror')


def implementation_identity():
    result = bench.implementation_identity()
    for relative in ('run_switch_feature_ablation.py', 'run_presentation_check.py',
                     'src/firered_rl/matchup_switch_policy.py', 'src/firered_rl/matchup_move_policy.py',
                     'src/firered_rl/shared_move_policy.py'):
        result[relative] = checksum(ROOT / relative)
    return result


def make_jobs(settings):
    return [dict(name=name, checkpoint=f'inputs/{name}-initial.zip', starting_timesteps=settings['starting_timesteps'],
        target_timesteps=settings['starting_timesteps'] + settings['timesteps'], seed=settings['seed'],
        bench_features_enabled=enabled, shuffle_bench_order=False)
        for name, enabled in (('control', False), ('switch-features', True))]


def tensor_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict): return left.keys() == right.keys() and all(tensor_equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)): return type(left) is type(right) and len(left) == len(right) and all(tensor_equal(a, b) for a, b in zip(left, right))
    return left == right


def initial_identity(template, models):
    """Verify twins and equivalent source outputs before reusing source baseline scores."""
    import numpy as np
    import torch
    if not tensor_equal(models[0].policy.state_dict(), models[1].policy.state_dict()):
        raise ValueError('Feature off/on initial parameter tensors differ')
    if not tensor_equal(models[0].policy.optimizer.state_dict(), models[1].policy.optimizer.state_dict()):
        raise ValueError('Feature off/on initial optimizer states differ')
    if any(model.num_timesteps != template.num_timesteps for model in models):
        raise ValueError('Migration changed the starting timestep count')
    if [model.policy.features_extractor.bench_features_enabled for model in models] != [False, True]:
        raise ValueError('Expected direct bench features off/on')
    probes = []
    for forced in (False, True):
        for count in (2, 3, 5):
            obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in template.observation_space.spaces.items()}
            obs['context'][1] = int(forced)
            obs['opponent_active_index'][:] = 1
            obs['action_mask'][5:5+count] = 1
            if not forced: obs['action_mask'][:4] = 1
            obs['active'][0] = [1, 6, 50, 0 if forced else 100, 100, 10, 2, 0, 1, 0]
            obs['active'][1] = [1, 9, 50, 100, 100, 11, 11, 0, 1, 0]
            obs['v2_active'][:] = [100, 100, 0, 1]
            obs['v2_active_stats'][:] = [100, 100, 100, 100, 100]
            for slot in range(6):
                obs['party'][slot] = [1, 6+slot, 50, 100, 100, (slot+10)%18, (slot+10)%18, 0, 0, 0]
                obs['opponent_party'][slot] = [1, 9+slot]
                obs['v2_party'][:, slot] = [1, 50, 100, 100, 11, 11, 0, 1, 0, 0]
                obs['v2_stats'][:, slot] = [100, 100, 100, 100, 100]
                for move in range(4):
                    obs['moves'][slot, move] = [53, 10, 10, 10]
                    obs['opponent_moves'][slot, move] = [57, 11, 10, 10]
                    obs['v2_move_details'][0, slot, move] = [95, 100, 0, 1, 0, 10]
                    obs['v2_move_details'][1, slot, move] = [95, 100, 0, 1, 0, 11]
            if forced:
                obs['party'][0, 3] = 0
                obs['v2_party'][0, 0, 2] = 0
                obs['v2_party'][0, 0, 8] = 1
                obs['v2_active'][0, 0] = 0
            probes.append(obs)
    batch = {key: torch.as_tensor(np.stack([obs[key] for obs in probes])) for key in probes[0]}
    outputs = []
    for model in (template, *models):
        model.policy.set_training_mode(False)
        with torch.no_grad():
            features = model.policy.extract_features(batch)
            outputs.append((model.policy.action_net(features), model.policy.predict_values(batch),
                model.policy.get_distribution(batch, action_masks=batch['action_mask'].bool()).distribution.probs))
    differences = [max(float((base-value).abs().max()) for base, value in zip(outputs[0], output)) for output in outputs[1:]]
    if any(value > 1e-7 for value in differences): raise ValueError(f'Migration changed initial outputs: {differences}')
    return dict(parameter_tensors_equal=True, optimizer_states_equal=True, initial_outputs_equal=True,
        source_timesteps=template.num_timesteps, bench_features_enabled=[False, True], probes=len(probes),
        max_source_output_differences=differences, optimizer='preserved-existing/new-empty',
        parameters=sum(parameter.numel() for parameter in models[0].policy.parameters()))


def checkpoint_metadata(path, *, enabled=None, starting_timesteps=None):
    with ZipFile(path) as archive: data = json.loads(archive.read('data'))
    meta = data.get('switch_ablation', {})
    if (data.get('policy_architecture') != 'matchup-switches' or meta.get('version') != 1
            or meta.get('optimizer') != 'preserved-existing/new-empty'):
        raise ValueError('Missing direct switch-feature checkpoint provenance')
    if enabled is not None and meta.get('enabled') is not enabled:
        raise ValueError('Checkpoint feature flag differs from its planned arm')
    if starting_timesteps is not None and meta.get('template_timesteps') != starting_timesteps:
        raise ValueError('Checkpoint migration source timestep count differs')
    return meta


def import_initial(root, source, start):
    record = copy.deepcopy(source['result']['policies'][source['selected']['name']]['best'])
    checks = {}
    for condition in CONDITIONS:
        value = copy.deepcopy(record['checks'][condition])
        source_directory = inside(source['source'], value['result_directory'])
        relative = f'checks/imported-start-{condition}'
        target = root / relative; target.mkdir(parents=True)
        for name in ('summary.json', 'suite.json', 'episodes.jsonl', 'decision-audit.json', 'decision-traces.jsonl'):
            shutil.copyfile(source_directory / name, target / name)
        value['result_directory'] = relative
        checks[condition] = value
    result = dict(checkpoint='inputs/source.zip', checkpoint_sha256=start['sha256'], timesteps=start['timesteps'],
        checks=checks, imported=True, balanced_validation=source['selected']['validation_rate'],
        source_run=str(source['source']), source_checkpoint=start['source_checkpoint'],
        rationale='Migration identity check preserves source initial outputs; saved baseline evidence is imported rather than rerun.')
    bench.validate_saved_checks(root, [result])
    write_json(root / 'initial.json', result)
    return result


def freeze(root, args):
    from sb3_contrib import MaskablePPO
    from firered_rl.evaluation import validate_suite
    from firered_rl.matchup_switch_policy import create_switch_checkpoint
    from firered_rl.showdown_env import validate_showdown
    source = presentation.source_inputs(args.source_run, args.smoke)
    validate_showdown(ROOT)
    if root == source['source'] or root.is_relative_to(source['source']): raise ValueError('Output must be outside source experiment')
    if type(args.timesteps) is not int or args.timesteps < 1: raise ValueError('Additional timesteps must be positive')
    start = source['models']['official']
    settings = dict(seed=261120, starting_timesteps=start['timesteps'], timesteps=256 if args.smoke else args.timesteps,
        environments=2 if args.smoke else 16, learning_rate=.0001, ent_coef=.01, rollout_steps=128,
        batch_size=128, epochs=4, eval_every=128 if args.smoke else 50000, max_steps=100,
        selection='Highest wins/requested on balanced original/reversed validation; control retains ties. Heldout never selects.',
        early_stopping=False, no_automatic_promotion=True, current_weight=.9, mirror_weight=.1,
        comparison='Same source validation-best weights and old optimizer state; same expanded architecture, seed, teams, reward and budget. Only direct forced-replacement features differ.')
    root.mkdir(parents=True, exist_ok=False)
    for relative in ('inputs/suites', 'console', 'models'): (root / relative).mkdir(parents=True)
    shutil.copyfile(start['path'], root / 'inputs/source.zip')
    shutil.copyfile(source['reward'], root / 'inputs/reward.json')
    write_json(root / 'inputs/settings.json', settings)
    teams = {}
    for group, paths in source['teams'].items():
        (root / 'inputs/teams' / group).mkdir(parents=True)
        teams[group] = []
        for index, path in enumerate(paths):
            relative = f'inputs/teams/{group}/team-{index+1:04}.json'
            shutil.copyfile(path, root / relative); teams[group].append(relative)
    write_json(root / 'inputs/mixture.json', dict(version=1, groups={
        'current': dict(weight=.9, directory=str(root / 'inputs/teams/train')),
        'fixed-mirror': dict(weight=.1, teams=[str(root / path) for path in teams['mirror']])}))
    suites = {}
    for name in ('validation', *CONDITIONS):
        data = read_json(source['suites'][name])
        if args.smoke and len(data['cases']) > 8:
            if name == 'validation':
                # Preserve equal weights across original and reversed halves.
                from firered_rl.replacement_diagnostics import transform_suite
                original = read_json(source['suites']['validation-original'])
                original['cases'] = original['cases'][:8]
                data = bench.combined_validation(original, transform_suite(original))
            else: data['cases'] = data['cases'][:8]
        validate_suite(data)
        if data['max_steps'] != settings['max_steps']: raise ValueError('Source maximum battle length differs')
        relative = f'inputs/suites/{name}.json'; write_json(root / relative, data); suites[name] = relative
    template = MaskablePPO.load(root / 'inputs/source.zip', device='cpu')
    jobs = make_jobs(settings)
    models = [create_switch_checkpoint(template, root / job['checkpoint'], enabled=job['bench_features_enabled'],
        seed=settings['seed'], training_settings=dict(n_steps=128, batch_size=128, n_epochs=4,
            learning_rate=settings['learning_rate'], ent_coef=settings['ent_coef'], device='cpu')) for job in jobs]
    identity = initial_identity(template, models)
    write_json(root / 'inputs/initial-identity.json', identity)
    del models, template
    source_evidence = dict(source_run=str(source['source']), selected_by_validation=source['selected'],
        source_plan_sha256=checksum(source['source'] / 'plan.json'), source_result_sha256=checksum(source['source'] / 'result.json'),
        source_checkpoint=start['source_checkpoint'], source_sha256=start['sha256'],
        preservation='Source experiments and 400k reference are unchanged. Both arms migrate the same selected checkpoint.')
    write_json(root / 'inputs/source-evidence.json', source_evidence)
    write_json(root / 'inputs/configuration.json', dict(settings=settings, teams=teams, suites=suites, jobs=jobs))
    import_initial(root, source, start)
    for job in jobs: (root / 'console' / (job['name'] + '.log')).touch()
    plan = dict(version=1, experiment=EXPERIMENT, smoke=bool(args.smoke), settings=settings, teams=teams,
        suites=suites, jobs=jobs, initial_identity=identity, source_selected=source['selected'],
        source_start_sha256=start['sha256'], implementation=implementation_identity(), no_automatic_promotion=True,
        heldout_role='Reused development teams; not an untouched generalization benchmark.',
        pins={str(path.relative_to(root)): checksum(path) for path in (root / 'inputs').rglob('*') if path.is_file()})
    write_json(root / 'plan.json', plan)
    return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != EXPERIMENT: raise ValueError('Unsupported switch-feature plan')
    if plan.get('implementation') != implementation_identity(): raise ValueError('Implementation changed; use a new run directory')
    if {inside(root, path) for path in plan['pins']} != {path.resolve() for path in (root / 'inputs').rglob('*') if path.is_file()}:
        raise ValueError('Pinned input membership changed')
    for path, sha in plan['pins'].items():
        if checksum(inside(root, path)) != sha: raise ValueError(f'Pinned input changed: {path}')
    configuration = read_json(root / 'inputs/configuration.json')
    if any(plan[key] != configuration[key] for key in ('settings', 'teams', 'suites', 'jobs')):
        raise ValueError('Saved experiment configuration changed')
    if plan['settings'] != read_json(root / 'inputs/settings.json') or plan['jobs'] != make_jobs(plan['settings']):
        raise ValueError('Saved settings or arm budgets changed')
    if (plan['initial_identity'] != read_json(root / 'inputs/initial-identity.json')
            or plan['pins']['inputs/source.zip'] != plan['source_start_sha256']
            or checkpoint_steps(root / 'inputs/source.zip') != plan['settings']['starting_timesteps']):
        raise ValueError('Initial identity or source checkpoint changed')
    if set(plan['suites']) != set(('validation', *CONDITIONS)): raise ValueError('Frozen suite membership changed')
    for job in plan['jobs']:
        checkpoint_metadata(root / job['checkpoint'], enabled=job['bench_features_enabled'], starting_timesteps=job['starting_timesteps'])
        if checkpoint_steps(root / job['checkpoint']) != job['starting_timesteps']: raise ValueError('Initial checkpoint timestep count changed')
    if plan.get('no_automatic_promotion') is not True or plan['settings']['early_stopping'] is not False:
        raise ValueError('Fixed budget and no automatic source-model promotion are required')


def command(root, plan, job):
    # Reuse the bounded continuation command; architecture and each initial file differ.
    value = bench.command(root, plan, job)
    if value is not None and '--policy-architecture' in value:
        value[value.index('--policy-architecture')+1] = 'matchup-switches'
    return value


def launch(root, plan, concurrency):
    if concurrency not in (1, 2): raise ValueError('Only one or two concurrent training jobs are allowed')
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
                reader = threading.Thread(target=pump, args=(proc, root / 'console' / (job['name']+'.log')))
                reader.start(); active.append((job, proc, reader))
            for row in list(active):
                job, proc, reader = row
                if proc.poll() is None: continue
                reader.join(); active.remove(row)
                if proc.returncode or read_json(root / 'runs' / job['name'] / 'summary.json').get('status') != 'completed':
                    raise RuntimeError(f"{job['name']} failed; inspect console log")
            if active: time.sleep(.25)
    finally:
        for _, proc, _ in active:
            if proc.poll() is None: proc.send_signal(signal.SIGINT)
        for _, proc, reader in active: proc.wait(); reader.join()


def training_provenance(root, plan, job, summary):
    run = root / 'runs' / job['name']
    current = read_json(run / 'config.json')
    first_path = run / 'sessions/000001/config.json'
    first = read_json(first_path) if first_path.exists() else current
    expected = plan['pins'][job['checkpoint']]
    if first.get('session', {}).get('id') != 1 or first.get('resume_checkpoint', {}).get('sha256') != expected:
        raise ValueError('Training did not start from its pinned migrated checkpoint')
    if any(config.get('shuffle_bench_order') is not False or config.get('policy_architecture') != 'matchup-switches'
           for config in (first, current)) or summary.get('bench_order_shuffle') is not None:
        raise ValueError('Actual architecture or bench-order settings differ from the planned arm')
    for config in (first, current):
        actual = config.get('switch_features', {})
        migration = actual.get('migration', {})
        if (actual.get('version') != 1 or actual.get('enabled') is not job['bench_features_enabled']
                or actual.get('whole_policy_training') is not True
                or actual.get('scope') != 'shared residual on multi-option forced replacements only'
                or migration.get('enabled') is not job['bench_features_enabled']
                or migration.get('template_timesteps') != job['starting_timesteps']
                or migration.get('optimizer') != 'preserved-existing/new-empty'):
            raise ValueError('Actual trainer switch features differ from the planned arm')
    return dict(initial_checkpoint_sha256=expected, bench_features_enabled=job['bench_features_enabled'],
        architecture='matchup-switches', shuffle_bench_order=False)


def finish(root, plan):
    from firered_rl.evaluation import digest
    initial = read_json(root / 'initial.json')
    bench.validate_saved_checks(root, [initial])
    requested = len(read_json(root / plan['suites']['validation'])['cases'])
    validation_hash = digest(read_json(root / plan['suites']['validation']))
    metadata = {}
    for job in plan['jobs']:
        path = root / 'runs' / job['name'] / 'checkpoints/best.json'
        meta = read_json(path)
        if meta.get('requested') != requested or meta.get('suite_sha256') != validation_hash:
            raise ValueError('Best checkpoint was not selected on the pinned balanced validation suite')
        metadata[job['name']] = meta
    selected = transfer.selection(root, plan)  # Selection persists before later frozen checks.
    policies, training, cache = {}, {}, {}
    for job in plan['jobs']:
        run = root / 'runs' / job['name']; summary = read_json(run / 'summary.json')
        if (summary.get('status') != 'completed' or summary.get('stop_reason') != 'timestep_budget'
                or summary['player_timesteps'] < job['target_timesteps']):
            raise ValueError('Training did not complete its fixed timestep budget')
        provenance = training_provenance(root, plan, job, summary)
        policies[job['name']] = {}
        history = read_json(run / 'evaluation-history.json')
        if history.get('identity', {}).get('suite_sha256') != validation_hash:
            raise ValueError('Saved validation history suite differs from the pinned suite')
        for kind in ('best', 'final'):
            path = run / 'checkpoints' / (kind+'.zip')
            checkpoint_metadata(path, enabled=job['bench_features_enabled'], starting_timesteps=job['starting_timesteps'])
            steps, sha = checkpoint_steps(path), checksum(path)
            if steps != (metadata[job['name']]['timesteps'] if kind == 'best' else summary['player_timesteps']):
                raise ValueError('Checkpoint timestep count differs from saved training evidence')
            final_records = [record for record in history['records'] if record.get('status') == 'completed'
                and record.get('checkpoint_sha256') == sha and record.get('timesteps') == steps
                and record.get('requested') == requested]
            if kind == 'final' and not final_records:
                raise ValueError('Final checkpoint has no matching completed validation history record')
            validation_rate = metadata[job['name']]['wins']/requested if kind == 'best' else final_records[-1]['wins']/requested
            if sha not in cache:
                cache[sha] = {condition: bench.frozen(root, plan, job['name']+'-'+kind+'-'+condition, condition, path)
                              for condition in CONDITIONS}
            policies[job['name']][kind] = dict(checkpoint=str(path.relative_to(root)), checkpoint_sha256=sha,
                timesteps=steps, checks=cache[sha], bench_features_enabled=job['bench_features_enabled'],
                mirror_change_from_start=cache[sha]['mirror']['wins_per_requested']-initial['checks']['mirror']['wins_per_requested'],
                balanced_validation=validation_rate)
        training[job['name']] = dict(starting_timesteps=job['starting_timesteps'], ending_timesteps=summary['player_timesteps'],
            added_timesteps=summary['player_timesteps']-job['starting_timesteps'], elapsed_seconds=summary['elapsed_seconds'],
            stop_reason=summary['stop_reason'], team_mixture_sampling=summary.get('team_mixture_sampling'), provenance=provenance)
    verify(root, plan)
    records = [initial, *[record for arm in policies.values() for record in arm.values()]]
    saved = bench.validate_saved_checks(root, records)
    exports = {}
    for name, arm in policies.items():
        for kind, record in arm.items():
            relative = f'models/{name}-{kind}.zip'
            shutil.copyfile(root / record['checkpoint'], root / relative)
            if checksum(root / relative) != record['checkpoint_sha256']: raise RuntimeError('Exported model hash differs')
            exports[name+'-'+kind] = dict(checkpoint=relative, sha256=record['checkpoint_sha256'], timesteps=record['timesteps'])
    choice = selected['selected']; official = policies[choice['name']]['best']
    shutil.copyfile(root / official['checkpoint'], root / 'models/selected.zip')
    if checksum(root / 'models/selected.zip') != choice['checkpoint_sha256']: raise RuntimeError('Selected export hash differs')
    exports['selected'] = dict(checkpoint='models/selected.zip', sha256=choice['checkpoint_sha256'],
        source_arm=choice['name'], timesteps=official['timesteps'], criterion='balanced validation only')
    difference = sum(policies['switch-features']['best']['checks'][key]['wins_per_requested']-
        policies['control']['best']['checks'][key]['wins_per_requested'] for key in ('heldout-original', 'heldout-reversed'))/2
    result = dict(status='smoke_completed' if plan['smoke'] else 'completed', selected_by_validation=choice,
        initial=initial, policies=policies, training=training, exports=exports, saved_evidence=saved,
        no_automatic_promotion=True, comparison=dict(selected_heldout_mean_difference=difference,
            scope='One paired continuation on reused development teams. Scores do not measure seed variability or unrestricted play.'))
    write_json(root / 'models/manifest.json', exports)
    write_json(root / 'result.json', result)
    return result


def collect(root, plan):
    result = read_json(root / 'result.json') if (root / 'result.json').exists() else {}
    initial = read_json(root / 'initial.json')
    status = read_json(root / 'status.json') if (root / 'status.json').exists() else {'status': 'prepared'}
    lines = ['# Direct replacement feature comparison', '',
        'Two matched continuations from the recorded validation-best model. The expanded policy and old optimizer states start identically; new residual outputs are zero.',
        'Control disables direct bench features. Treatment adds observation-derived bench information only to learned forced-replacement scores with at least two legal choices. No Rules actions are supplied.',
        'Both use the same fixed teams, 90% current task / 10% fixed mirror, seed and bounded budget. No bench-order shuffling or early success stop.', '',
        f"Status: {status['status']}. Additional budget: {plan['settings']['timesteps']:,} steps per arm.",
        'Validation selects models using equal original/reversed bench orders. Frozen scores are against Rules; draws/truncations count as non-wins.', '',
        '| Arm | Checkpoint | Total steps | Balanced validation | Seen original | Seen reversed | Heldout original | Heldout reversed | Mirror |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    records = [('source', 'imported start', initial)]
    for name, arm in result.get('policies', {}).items(): records.extend((name, kind, rec) for kind, rec in arm.items())
    for name, kind, record in records:
        rates = [f"{record['checks'][condition]['wins_per_requested']:.1%}" for condition in CONDITIONS]
        validation = f"{record['balanced_validation']:.1%}" if record.get('balanced_validation') is not None else '—'
        lines.append(f"| {name} | {kind} | {record['timesteps']:,} | {validation} | {' | '.join(rates)} |")
    lines += ['', '## Forced replacements and retention', '',
        '| Arm | Checkpoint | Heldout order | Eligible | Lowest slot | Rules disagreements | Overrides | Mirror change |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for name, kind, record in records:
        for condition in ('heldout-original', 'heldout-reversed'):
            totals = record['checks'][condition]['audit_totals']
            lines.append(f"| {name} | {kind} | {condition} | {totals['replacement_eligible']} | {totals['replacement_actual_lowest_legal']} | {totals['replacement_base_rules_disagreements']} | {totals['replacement_overrides_changed']} | {100*record.get('mirror_change_from_start', 0):+.1f} pp |")
    if result:
        lines += ['', f"Validation-selected model: **{result['selected_by_validation']['name']}**, `models/selected.zip`.",
            f"Selected-checkpoint heldout mean difference, treatment minus control: {100*result['comparison']['selected_heldout_mean_difference']:+.1f} percentage points.",
            result['comparison']['scope'], 'The lowest legal slot is a descriptive measurement, not an optimality target. Source models and the 400k reference remain unchanged.']
    lines += ['', 'The imported start reuses complete source checks after migration output equivalence is verified. Best/final scores are separately audited; heldout never selects.',
        f"Logs: `tail -n 20 -F {root}/console/*.log`", 'Memory: `watch -n 2 free -h`',
        f"Teams: `python run_switch_feature_ablation.py --output-dir {shlex.quote(str(root))} --inspect-teams train --full-json`",
        f"Resume: `python run_switch_feature_ablation.py --output-dir {shlex.quote(str(root))} --continue --jobs 2`",
        'Each job has evaluations.md, updates.jsonl and PPO CSV metrics. memory.csv includes validation peaks.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    (root / 'play-summary.md').write_text('\n'.join(lines[:lines.index('## Forced replacements and retention')]))
    temporary = root / 'switch-feature-results.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.md', '.csv') and relative.parts[0] != 'console':
                archive.write(path, str(relative))
    temporary.replace(root / 'switch-feature-results.zip')


def inspect(root, plan, group, limit, full):
    print('Both arms use the same canonical teams, with no bench shuffling:')
    print((root / 'inputs/mixture.json').read_text())
    for name in ('train', 'mirror', 'validation', 'heldout') if group == 'all' else (group,):
        if name in plan['teams']: values = [(str(root / path), read_json(root / path)) for path in plan['teams'][name]]
        else:
            path = root / plan['suites']['heldout-original' if name == 'heldout' else 'validation']
            values = [(str(path)+' asset '+key, asset['team']) for key, asset in read_json(path)['assets'].items()]
        print(f'\n{name}: {len(values)} teams')
        for path, team in values[:limit]:
            print(path)
            if full: print(json.dumps(team, indent=2))
            else:
                for mon in team['party']: print(f"  {mon['species']}: {', '.join(mon['moves'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('.local/run/bench-order-ablation-01'))
    parser.add_argument('--output-dir', type=Path, default=Path('.local/run/switch-feature-ablation-01'))
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-teams', nargs='?', const='all', choices=('all', 'train', 'mirror', 'validation', 'heldout'))
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--full-json', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.limit < 0: parser.error('--limit cannot be negative')
    if args.timesteps is not None and args.timesteps < 1: parser.error('--timesteps must be positive')
    if args.continuing and args.timesteps is not None: parser.error('--continue uses the saved budget; omit --timesteps')
    if args.inspect_teams is not None:
        plan = read_json(root / 'plan.json'); verify(root, plan); inspect(root, plan, args.inspect_teams, args.limit, args.full_json); return 0
    if args.timesteps is None: args.timesteps = 500000
    import torch
    torch.set_num_threads(1)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = (root.parent / ('.'+root.name+'.switch-feature.lock')).open('a')
    plan = None; owned = False; stopped = threading.Event(); monitor = None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); owned = True
        if args.continuing: plan = read_json(root / 'plan.json'); verify(root, plan)
        else: plan = freeze(root, args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {shlex.quote(str(root / 'console'))}/*.log\nMemory: watch -n 2 free -h\nTeams: python run_switch_feature_ablation.py --output-dir {shlex.quote(str(root))} --inspect-teams train --full-json", flush=True)
        if args.prepare_only: write_json(root / 'status.json', dict(status='prepared')); return 0
        monitor = threading.Thread(target=memory_monitor, args=(root, stopped), daemon=True); monitor.start()
        write_json(root / 'status.json', dict(status='running', phase='training'))
        collect(root, plan)
        launch(root, plan, args.jobs); verify(root, plan)
        write_json(root / 'status.json', dict(status='running', phase='frozen-checks'))
        result = finish(root, plan)
        write_json(root / 'status.json', dict(status=result['status'], no_automatic_promotion=True)); return 0
    except (KeyboardInterrupt, Exception) as error:
        if plan is not None and owned: write_json(root / 'status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', message=str(error)))
        print(f'Stopped: {error or "interrupted"}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        stopped.set()
        if monitor: monitor.join()
        if plan is not None and owned:
            collect(root, plan); print(f"Report: {root / 'report.md'}\nUpload: {root / 'switch-feature-results.zip'}", flush=True)
        signal.signal(signal.SIGTERM, previous); lock.close()


if __name__ == '__main__': raise SystemExit(main())
