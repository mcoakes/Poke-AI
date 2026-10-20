#!/usr/bin/env python3
"""Compare a small training pool with a balanced expansion of familiar 3v3 teams.

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
import run_switch_feature_ablation as switches
import run_transfer_lab as transfer
from run_fixed_team_lab import checksum, write_json, interrupt
from run_parallel_training import checkpoint_steps, pump
from run_staged_team_lab import memory_monitor
read_json, inside = transfer.read_json, transfer.inside
EXPERIMENT = 'training-coverage'
CONDITIONS = switches.CONDITIONS


def implementation_identity():
    result = switches.implementation_identity()
    for relative in ('run_training_coverage.py', 'src/firered_rl/balanced_training_pool.py',
                     'src/firered_rl/broad_team_curriculum.py'):
        result[relative] = checksum(ROOT / relative)
    return result


def make_jobs(settings):
    return [dict(name=name, checkpoint='inputs/start.zip', starting_timesteps=settings['starting_timesteps'],
        target_timesteps=settings['starting_timesteps']+settings['timesteps'], seed=settings['seed'],
        bench_features_enabled=True, shuffle_bench_order=False, mixture=f'inputs/mixtures/{name}.json', weights=weights)
        for name, weights in [('small-pool', {'old': .9, 'fixed-mirror': .1}),
                              ('balanced-pool', {'old': .25, 'new': .65, 'fixed-mirror': .1})]]


def source_inputs(source, smoke=False):
    """Pin the predeclared feature-final continuation and the prior selected fallback."""
    source = Path(source).expanduser().resolve()
    plan, result = (read_json(source / name) for name in ('plan.json', 'result.json'))
    if plan.get('version') != 1 or plan.get('experiment') != 'switch-feature-ablation':
        raise ValueError('Source must be a version-1 switch-feature-ablation run')
    if plan.get('smoke') and not smoke: raise ValueError('Smoke source cannot supply a normal training run')
    if result.get('status') not in ('completed', 'smoke_completed'): raise ValueError('Source run must be completed')
    bench.validate_saved_checks(source, [result['initial'], *[record for arm in result['policies'].values() for record in arm.values()]])
    for relative, sha in plan['pins'].items():
        path = inside(source, relative)
        if not path.is_file() or checksum(path) != sha: raise ValueError(f'Source pinned input changed: {relative}')
    def pinned(relative):
        if relative not in plan['pins']: raise ValueError(f'Unpinned source input: {relative}')
        return inside(source, relative)
    teams = {name: [pinned(path) for path in plan['teams'][name]] for name in ('train', 'mirror')}
    for name, paths in teams.items():
        if not paths or len(set(paths)) != len(paths) or set((source / 'inputs/teams' / name).glob('*.json')) != set(paths):
            raise ValueError(f'Source {name} team membership changed')
    suites = {name: pinned(path) for name, path in plan['suites'].items()}
    if set(suites) != set(('validation', *CONDITIONS)): raise ValueError('Source suite membership differs')
    from firered_rl.evaluation import digest
    validation_hash = digest(read_json(suites['validation']))
    selected = result['selected_by_validation']
    requested = len(read_json(suites['validation'])['cases'])
    scores = {}
    for name in ('control', 'switch-features'):
        record = result['policies'][name]['best']
        meta = read_json(source / 'runs' / name / 'checkpoints/best.json')
        path = inside(source, record['checkpoint'])
        if (meta.get('requested') != requested or meta.get('suite_sha256') != validation_hash
                or type(meta.get('wins')) is not int or not 0 <= meta['wins'] <= requested
                or meta.get('checkpoint_sha256') != record['checkpoint_sha256']
                or checksum(path) != record['checkpoint_sha256'] or checkpoint_steps(path) != meta['timesteps']
                or meta['timesteps'] != record['timesteps']): raise ValueError('Source native best checkpoint evidence differs')
        scores[name] = meta['wins'] / requested
    if selected['name'] != max(scores, key=scores.get) or selected['validation_rate'] != scores[selected['name']]:
        raise ValueError('Source selection differs from native balanced validation')
    fallback_record = result['policies'][selected['name']]['best']
    if any(selected.get(key) != fallback_record.get(key) for key in ('checkpoint', 'checkpoint_sha256')):
        raise ValueError('Source selected checkpoint differs from selected arm best')
    record = result['policies']['switch-features']['final']
    path = inside(source, record['checkpoint'])
    if checksum(path) != record['checkpoint_sha256']: raise ValueError('Source feature-final checkpoint hash changed')
    steps = checkpoint_steps(path)
    if steps != record['timesteps'] or steps != result['training']['switch-features']['ending_timesteps']:
        raise ValueError('Source final timestep evidence differs')
    migration = switches.checkpoint_metadata(path, enabled=True, starting_timesteps=plan['settings']['starting_timesteps'])
    job = next(job for job in plan['jobs'] if job['name'] == 'switch-features')
    summary = read_json(source / 'runs/switch-features/summary.json')
    if (summary.get('status') != 'completed' or summary.get('stop_reason') != 'timestep_budget'
            or summary.get('player_timesteps') != steps):
        raise ValueError('Source feature training did not finish its budget')
    switches.training_provenance(source, plan, job, summary)
    # Validate the actual serialized policy toggle, not just its descriptive metadata.
    from sb3_contrib import MaskablePPO
    from firered_rl.shared_move_policy import policy_architecture
    model = MaskablePPO.load(path, device='cpu')
    if (policy_architecture(model) != 'matchup-switches' or not model.policy.features_extractor.relational_enabled
            or model.policy.features_extractor.bench_features_enabled is not True):
        raise ValueError('Source actual policy must have direct switch features enabled')
    del model
    return dict(source=source, plan=plan, result=result, teams=teams, suites=suites,
        reward=pinned('inputs/reward.json'), selected=selected,
        start=dict(path=path, sha256=record['checkpoint_sha256'], timesteps=steps, record=record,
            migration_template_timesteps=migration['template_timesteps']),
        fallback=dict(path=inside(source, fallback_record['checkpoint']), sha256=fallback_record['checkpoint_sha256'],
            timesteps=fallback_record['timesteps'], validation_rate=selected['validation_rate']))


def all_old_teams(source):
    values = [read_json(path) for paths in source['teams'].values() for path in paths]
    for path in source['suites'].values(): values.extend(asset['team'] for asset in read_json(path)['assets'].values())
    return values


def import_initial(root, source):
    record = copy.deepcopy(source['start']['record']); checks = {}
    for condition in CONDITIONS:
        value = copy.deepcopy(record['checks'][condition]); relative = f'checks/imported-start-{condition}'
        target = root / relative; target.mkdir(parents=True)
        origin = inside(source['source'], value['result_directory'])
        for name in ('summary.json', 'suite.json', 'episodes.jsonl', 'decision-audit.json', 'decision-traces.jsonl'):
            shutil.copyfile(origin / name, target / name)
        value['result_directory'] = relative; checks[condition] = value
    initial = dict(checkpoint='inputs/start.zip', checkpoint_sha256=source['start']['sha256'],
        timesteps=source['start']['timesteps'], balanced_validation=record['balanced_validation'], checks=checks,
        imported=True, source_checkpoint=record['checkpoint'], source_run=str(source['source']),
        rationale='Exact source feature-final bytes and optimizer state are shared by both arms; complete source checks are imported.')
    bench.validate_saved_checks(root, [initial]); write_json(root / 'initial.json', initial)
    return initial


def freeze(root, args):
    from firered_rl.balanced_training_pool import generate
    from firered_rl.evaluation import validate_suite
    from firered_rl.showdown_env import validate_showdown
    source = source_inputs(args.source_run, args.smoke)
    validate_showdown(ROOT)
    if root == source['source'] or root.is_relative_to(source['source']): raise ValueError('Output must be outside source experiment')
    if type(args.timesteps) is not int or args.timesteps < 1: raise ValueError('Additional timesteps must be positive')
    start = source['start']
    settings = dict(seed=261130, starting_timesteps=start['timesteps'], timesteps=256 if args.smoke else args.timesteps,
        migration_template_timesteps=start['migration_template_timesteps'], environments=2 if args.smoke else 16,
        learning_rate=.0001, ent_coef=.01, rollout_steps=128, batch_size=128, epochs=4,
        eval_every=128 if args.smoke else 50000, max_steps=100, new_teams=12 if args.smoke else 96,
        selection='Highest wins/requested on the unchanged balanced validation suite; small-pool retains ties. Heldout never selects.',
        early_stopping=False, no_automatic_promotion=True,
        comparison='Both continue the same feature-final checkpoint and optimizer. Small pool: 90% old/10% mirror. Balanced pool: 25% old/65% new balanced/10% mirror.')
    root.mkdir(parents=True, exist_ok=False)
    for relative in ('inputs/suites', 'inputs/mixtures', 'console', 'models'): (root / relative).mkdir(parents=True)
    shutil.copyfile(start['path'], root / 'inputs/start.zip')
    shutil.copyfile(source['fallback']['path'], root / 'inputs/previous-selected.zip')
    shutil.copyfile(source['fallback']['path'], root / 'models/previous-selected.zip')
    shutil.copyfile(source['reward'], root / 'inputs/reward.json')
    write_json(root / 'inputs/settings.json', settings)
    teams = {}
    for group, paths in source['teams'].items():
        name = 'old' if group == 'train' else group
        (root / 'inputs/teams' / name).mkdir(parents=True); teams[name] = []
        for index, path in enumerate(paths):
            relative = f'inputs/teams/{name}/team-{index+1:04}.json'
            shutil.copyfile(path, root / relative); teams[name].append(relative)
    config_path = getattr(args, 'config', ROOT / 'config/experiments/team-curriculum.json')
    config = read_json(config_path); presentation.validate_source_recipe(source, config)
    write_json(root / 'inputs/team-recipe.json', config)
    shutil.copyfile(inside(ROOT, config['catalog']), root / 'inputs/catalog.json')
    old_teams = all_old_teams(source); write_json(root / 'inputs/excluded-teams.json', old_teams)
    generated, inventory = generate(ROOT, config, old_teams, count=settings['new_teams'], seed=settings['seed'])
    teams['new'] = []; (root / 'inputs/teams/new').mkdir(parents=True)
    for index, team in enumerate(generated):
        relative = f'inputs/teams/new/team-{index+1:04}.json'
        write_json(root / relative, team); teams['new'].append(relative)
    inventory.update(config_path=str(Path(config_path).resolve()), config_sha256=checksum(config_path),
        catalog=config['catalog'], catalog_sha256=checksum(ROOT / config['catalog']))
    write_json(root / 'inputs/new-team-inventory.json', inventory)
    suites = {}
    for name in ('validation', *CONDITIONS):
        data = read_json(source['suites'][name])
        # Keep source suite bytes/selection fixed, including source smoke schedules.
        validate_suite(data)
        if data['max_steps'] != settings['max_steps']: raise ValueError('Source maximum battle length differs')
        relative = f'inputs/suites/{name}.json'; shutil.copyfile(source['suites'][name], root / relative); suites[name] = relative
    jobs = make_jobs(settings)
    for job in jobs:
        groups = {name: (dict(weight=weight, teams=[str(root / path) for path in teams['mirror']])
            if name == 'fixed-mirror' else dict(weight=weight, directory=str(root / 'inputs/teams' / name)))
            for name, weight in job['weights'].items()}
        write_json(root / job['mixture'], dict(version=1, groups=groups))
        (root / 'console' / (job['name']+'.log')).touch()
    fallback = dict(checkpoint='inputs/previous-selected.zip', sha256=source['fallback']['sha256'],
        timesteps=source['fallback']['timesteps'], validation_rate=source['fallback']['validation_rate'],
        original_selected=source['selected'])
    source_evidence = dict(source_run=str(source['source']), source_selected=source['selected'],
        source_plan_sha256=checksum(source['source'] / 'plan.json'), source_result_sha256=checksum(source['source'] / 'result.json'),
        start_checkpoint=source['start']['record']['checkpoint'], start_sha256=start['sha256'],
        start_choice='Predeclared development continuation from switch-features fixed-budget final; not selected by validation.',
        preservation='Original selected checkpoint is retained as fallback; source experiments and 400k reference are unchanged.')
    write_json(root / 'inputs/source-evidence.json', source_evidence)
    write_json(root / 'inputs/configuration.json', dict(settings=settings, teams=teams, suites=suites, jobs=jobs, fallback=fallback))
    import_initial(root, source)
    plan = dict(version=1, experiment=EXPERIMENT, smoke=bool(args.smoke), settings=settings, teams=teams,
        suites=suites, jobs=jobs, fallback=fallback, source_selected=source['selected'], source_start_sha256=start['sha256'],
        implementation=implementation_identity(), no_automatic_promotion=True,
        heldout_role='Reused development teams; not an untouched generalization benchmark.',
        pins={str(path.relative_to(root)): checksum(path) for path in (root / 'inputs').rglob('*') if path.is_file()})
    write_json(root / 'plan.json', plan)
    return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != EXPERIMENT: raise ValueError('Unsupported training coverage plan')
    if plan.get('implementation') != implementation_identity(): raise ValueError('Implementation changed; use a new run directory')
    if {inside(root, path) for path in plan['pins']} != {path.resolve() for path in (root / 'inputs').rglob('*') if path.is_file()}:
        raise ValueError('Pinned input membership changed')
    for path, sha in plan['pins'].items():
        if checksum(inside(root, path)) != sha: raise ValueError(f'Pinned input changed: {path}')
    configuration = read_json(root / 'inputs/configuration.json')
    if any(plan[key] != configuration[key] for key in ('settings', 'teams', 'suites', 'jobs', 'fallback')):
        raise ValueError('Saved experiment configuration changed')
    if plan['settings'] != read_json(root / 'inputs/settings.json') or plan['jobs'] != make_jobs(plan['settings']):
        raise ValueError('Saved settings, mixtures or arm budgets changed')
    if (plan['pins']['inputs/start.zip'] != plan['source_start_sha256']
            or checkpoint_steps(root / 'inputs/start.zip') != plan['settings']['starting_timesteps']):
        raise ValueError('Starting checkpoint provenance changed')
    switches.checkpoint_metadata(root / 'inputs/start.zip', enabled=True,
        starting_timesteps=plan['settings']['migration_template_timesteps'])
    if (plan['pins'][plan['fallback']['checkpoint']] != plan['fallback']['sha256']
            or checkpoint_steps(root / plan['fallback']['checkpoint']) != plan['fallback']['timesteps']):
        raise ValueError('Previous selected fallback changed')
    if set(plan['suites']) != set(('validation', *CONDITIONS)): raise ValueError('Frozen suite membership changed')
    inventory = read_json(root / 'inputs/new-team-inventory.json')
    if (checksum(Path(inventory['config_path'])) != inventory['config_sha256']
            or checksum(inside(ROOT, inventory['catalog'])) != inventory['catalog_sha256']
            or checksum(root / 'inputs/catalog.json') != inventory['catalog_sha256']):
        raise ValueError('Team recipe or catalog changed')
    from firered_rl.balanced_training_pool import validate_pool
    validate_pool(ROOT, read_json(root / 'inputs/team-recipe.json'), [read_json(root / path) for path in plan['teams']['new']],
        read_json(root / 'inputs/excluded-teams.json'), count=plan['settings']['new_teams'])
    if plan.get('no_automatic_promotion') is not True or plan['settings']['early_stopping'] is not False:
        raise ValueError('Fixed budget and no source-model promotion are required')


def command(root, plan, job):
    value = bench.command(root, plan, job)
    if value is not None and '--policy-architecture' in value:
        value[value.index('--policy-architecture')+1] = 'matchup-switches'
        value[value.index('--team-mixture')+1] = str(root / job['mixture'])
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
    run = root / 'runs' / job['name']; current = read_json(run / 'config.json')
    first_path = run / 'sessions/000001/config.json'
    first = read_json(first_path) if first_path.exists() else current
    if first.get('session', {}).get('id') != 1 or first.get('resume_checkpoint', {}).get('sha256') != plan['source_start_sha256']:
        raise ValueError('Training did not start from the common feature-final checkpoint')
    for config in (first, current):
        features = config.get('switch_features', {}); migration = features.get('migration', {})
        if (config.get('policy_architecture') != 'matchup-switches' or config.get('shuffle_bench_order') is not False
                or features.get('enabled') is not True or features.get('whole_policy_training') is not True
                or migration.get('enabled') is not True or migration.get('optimizer') != 'preserved-existing/new-empty'
                or migration.get('template_timesteps') != plan['settings']['migration_template_timesteps']):
            raise ValueError('Actual policy features or shuffle settings differ from the plan')
        actual = config.get('teams', {})
        if (actual.get('mode') != 'team_mixture' or actual.get('manifest', {}).get('sha256') != plan['pins'][job['mixture']]
                or {name: group['weight'] for name, group in actual.get('groups', {}).items()} != job['weights']):
            raise ValueError('Actual training mixture differs from the planned arm')
        for name, group in actual['groups'].items():
            expected_paths = {str(root / path) for path in plan['teams']['mirror' if name == 'fixed-mirror' else name]}
            if set(group.get('files', [])) != expected_paths or group.get('fixed_pair') is not (name == 'fixed-mirror'):
                raise ValueError('Actual mixture team membership or fixed-pair scope differs')
    if summary.get('bench_order_shuffle') is not None: raise ValueError('Unexpected bench-order shuffling')
    sessions = {summary.get('session_id', 1): summary}
    for path in sorted((run / 'sessions').glob('*/summary.json')):
        value = read_json(path); sessions.setdefault(value.get('session_id', int(path.parent.name)), value)
    counts = dict.fromkeys(job['weights'], 0)
    for value in sessions.values():
        mixture = value.get('team_mixture_sampling', {})
        observed = mixture.get('episode_selections', {})
        if mixture.get('weights') != job['weights'] or set(observed) != set(counts):
            raise ValueError('Saved mixture sampling evidence differs')
        if any(type(number) is not int or number < 0 for number in observed.values()) or sum(observed.values()) != mixture.get('total'):
            raise ValueError('Invalid actual episode group counts')
        for name, number in observed.items(): counts[name] += number
    if not plan['smoke'] and any(number == 0 for number in counts.values()):
        raise ValueError('Normal run did not sample every planned training group')
    return dict(initial_checkpoint_sha256=plan['source_start_sha256'], bench_features_enabled=True,
        architecture='matchup-switches', shuffle_bench_order=False, weights=job['weights'], episode_starts=counts)


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
            switches.checkpoint_metadata(path, enabled=True, starting_timesteps=plan['settings']['migration_template_timesteps'])
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
    shutil.copyfile(root / plan['fallback']['checkpoint'], root / 'models/previous-selected.zip')
    exports = {'previous-selected': dict(checkpoint='models/previous-selected.zip', sha256=plan['fallback']['sha256'],
        timesteps=plan['fallback']['timesteps'], validation_rate=plan['fallback']['validation_rate'])}
    for name, arm in policies.items():
        for kind, record in arm.items():
            relative = f'models/{name}-{kind}.zip'
            shutil.copyfile(root / record['checkpoint'], root / relative)
            if checksum(root / relative) != record['checkpoint_sha256']: raise RuntimeError('Exported model hash differs')
            exports[name+'-'+kind] = dict(checkpoint=relative, sha256=record['checkpoint_sha256'], timesteps=record['timesteps'])
    new_choice = selected['selected']; official = policies[new_choice['name']]['best']
    shutil.copyfile(root / official['checkpoint'], root / 'models/new-run-selected.zip')
    exports['new-run-selected'] = dict(checkpoint='models/new-run-selected.zip', sha256=new_choice['checkpoint_sha256'],
        source_arm=new_choice['name'], timesteps=official['timesteps'], criterion='best of new arms by unchanged validation')
    improves = new_choice['validation_rate'] > plan['fallback']['validation_rate']
    choice = new_choice if improves else dict(name='previous-selected', checkpoint=plan['fallback']['checkpoint'],
        checkpoint_sha256=plan['fallback']['sha256'], validation_rate=plan['fallback']['validation_rate'])
    selected_steps = official['timesteps'] if improves else plan['fallback']['timesteps']
    shutil.copyfile(root / choice['checkpoint'], root / 'models/selected.zip')
    if checksum(root / 'models/selected.zip') != choice['checkpoint_sha256']: raise RuntimeError('Selected export hash differs')
    exports['selected'] = dict(checkpoint='models/selected.zip', sha256=choice['checkpoint_sha256'],
        source_arm=choice['name'], timesteps=selected_steps, criterion='unchanged validation across new arms and previous fallback; fallback retains ties')
    write_json(root / 'selection.json', dict(selected=choice, new_run_selected=new_choice, previous_fallback=plan['fallback'],
        heldout_used_for_selection=False, fallback_retains_ties=True))
    difference = sum(policies['balanced-pool']['best']['checks'][key]['wins_per_requested']-
        policies['small-pool']['best']['checks'][key]['wins_per_requested'] for key in ('heldout-original', 'heldout-reversed'))/2
    result = dict(status='smoke_completed' if plan['smoke'] else 'completed', selected_by_validation=choice, new_run_selected=new_choice,
        initial=initial, policies=policies, training=training, exports=exports, saved_evidence=saved,
        fallback=plan['fallback'], new_selection_beats_previous=improves,
        no_automatic_promotion=True, comparison=dict(selected_heldout_mean_difference=difference,
            scope='One paired continuation on reused development teams. Scores do not measure seed variability or unrestricted play.'))
    write_json(root / 'models/manifest.json', exports)
    write_json(root / 'result.json', result)
    return result


def collect(root, plan):
    result = read_json(root / 'result.json') if (root / 'result.json').exists() else {}
    initial = read_json(root / 'initial.json')
    status = read_json(root / 'status.json') if (root / 'status.json').exists() else {'status': 'prepared'}
    lines = ['# Training coverage comparison', '',
        'Both jobs continue the same feature-enabled fixed-final checkpoint and optimizer state. This is an explicitly developmental continuation; the previous validation-selected fallback is retained.',
        'Small pool: 90% original teams / 10% fixed mirror. Balanced pool: 25% original teams / 65% new balanced teams / 10% fixed mirror. Both policies learn all their weights and have direct replacement features enabled.',
        'The new pool has 96 distinct compositions, with each familiar species appearing 24 times and occupying each party slot 8 times (smoke: 12 teams, 3 appearances, once per slot). All source train/validation/heldout/mirror compositions are excluded. No bench shuffling, reward changes, or early success stop.', '',
        f"Status: {status['status']}. Additional budget: {plan['settings']['timesteps']:,} steps per arm.",
        'The original validation suite and equal original/reversed order weighting remain fixed for selection. Frozen scores are against Rules; draws/truncations count as non-wins.', '',
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
        lines += ['', f"Previous selected fallback validation: {plan['fallback']['validation_rate']:.1%}; new selection beats it: {result['new_selection_beats_previous']}.", f"Validation-selected model: **{result['selected_by_validation']['name']}**, `models/selected.zip`.",
            f"Selected-checkpoint heldout mean difference, treatment minus control: {100*result['comparison']['selected_heldout_mean_difference']:+.1f} percentage points.",
            result['comparison']['scope'], 'The lowest legal slot is a descriptive measurement, not an optimality target. Source models and the 400k reference remain unchanged.']
    lines += ['', 'The imported start reuses complete source feature-final checks; both jobs load those exact checkpoint bytes. Best/final scores are separately audited; heldout never selects.',
        f"Logs: `tail -n 20 -F {root}/console/*.log`", 'Memory: `watch -n 2 free -h`',
        f"Teams: `python run_training_coverage.py --output-dir {shlex.quote(str(root))} --inspect-teams new --full-json`",
        f"Resume: `python run_training_coverage.py --output-dir {shlex.quote(str(root))} --continue --jobs 2`",
        'Each job has evaluations.md, updates.jsonl and PPO CSV metrics. memory.csv includes validation peaks.', '']
    lines += [f"Play after completion: `python mgba/play_gui.py --rom \"$PWD/FireRed.gba\" --source \"$PWD/FireRed.sav\" --workers 2 --speed 1 --checkpoint {shlex.quote(str(root / 'models/selected.zip'))}`",
        f"Previous fallback: `{root / 'models/previous-selected.zip'}`",
        f"Optional permanent copy: `mkdir -p models && cp --no-clobber {shlex.quote(str(root / 'models/selected.zip'))} models/{root.name}-selected.zip`",
        f"Fixed-final alternative: `cp --no-clobber {shlex.quote(str(root / 'models/balanced-pool-final.zip'))} models/{root.name}-balanced-final.zip`",
        'Original and new team definitions are in inputs/teams/old and inputs/teams/new. These pools are training data, not new heldout evidence.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    (root / 'play-summary.md').write_text('\n'.join(lines[:lines.index('## Forced replacements and retention')]))
    temporary = root / 'training-coverage-results.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.md', '.csv') and relative.parts[0] != 'console':
                archive.write(path, str(relative))
    temporary.replace(root / 'training-coverage-results.zip')


def inspect(root, plan, group, limit, full):
    for job in plan['jobs']:
        print(f"{job['name']}: {job['weights']}; {root / job['mixture']}")
    for name in ('old', 'new', 'mirror') if group == 'all' else (group,):
        paths = plan['teams'][name]; print(f'\n{name}: {len(paths)} teams')
        for relative in paths[:limit]:
            team = read_json(root / relative); print(root / relative)
            if full: print(json.dumps(team, indent=2))
            else:
                for mon in team['party']: print(f"  {mon['species']}: {', '.join(mon['moves'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('.local/run/switch-feature-ablation-01'))
    parser.add_argument('--output-dir', type=Path, default=Path('.local/run/training-coverage-01'))
    parser.add_argument('--config', type=Path, default=ROOT / 'config/experiments/team-curriculum.json')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-teams', nargs='?', const='all', choices=('all', 'old', 'new', 'mirror'))
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
    lock = (root.parent / ('.'+root.name+'.training-coverage.lock')).open('a')
    plan = None; owned = False; stopped = threading.Event(); monitor = None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); owned = True
        if args.continuing: plan = read_json(root / 'plan.json'); verify(root, plan)
        else: plan = freeze(root, args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {shlex.quote(str(root / 'console'))}/*.log\nMemory: watch -n 2 free -h\nTeams: python run_training_coverage.py --output-dir {shlex.quote(str(root))} --inspect-teams new --full-json", flush=True)
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
            collect(root, plan); print(f"Report: {root / 'report.md'}\nUpload: {root / 'training-coverage-results.zip'}", flush=True)
        signal.signal(signal.SIGTERM, previous); lock.close()


if __name__ == '__main__': raise SystemExit(main())
