#!/usr/bin/env python3
"""Compare Rules behavior-cloning warmstart plus PPO with matched PPO continuation.

Two jobs maximum. Watch console/*.log and memory.csv. Ctrl+C pauses; --continue
resumes the frozen preparation, validation and bounded PPO budgets.
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
import run_training_coverage as coverage
import run_switch_feature_ablation as switches
import run_transfer_lab as transfer
from run_fixed_team_lab import checksum, write_json, interrupt
from run_parallel_training import checkpoint_steps, pump
from run_staged_team_lab import memory_monitor
read_json, inside = transfer.read_json, transfer.inside
EXPERIMENT = 'teacher-training'
CONDITIONS = ('seen-original', 'seen-reversed', 'fresh-original', 'fresh-reversed', 'mirror')
WEIGHTS = {'old': .25, 'new': .65, 'fixed-mirror': .1}


def implementation_identity():
    value = coverage.implementation_identity()
    for relative in ('run_teacher_training.py', 'src/firered_rl/teacher_warmstart.py',
                     'src/firered_rl/teacher_training_suites.py'):
        value[relative] = checksum(ROOT / relative)
    return value


def make_jobs(settings):
    return [dict(name=name, checkpoint=f'inputs/{name}.zip', starting_timesteps=settings['starting_timesteps'],
        target_timesteps=settings['starting_timesteps']+settings['timesteps'], seed=settings['seed'],
        bench_features_enabled=True, shuffle_bench_order=False, mixture='inputs/mixture.json', weights=WEIGHTS.copy())
        for name in ('ppo-only', 'teacher-ppo')]


def source_inputs(source, smoke=False):
    source = Path(source).expanduser().resolve()
    plan, result = (read_json(source / name) for name in ('plan.json', 'result.json'))
    if plan.get('experiment') != 'training-coverage' or plan.get('version') != 1:
        raise ValueError('Source must be a completed version-1 training-coverage run')
    if plan.get('smoke') and not smoke: raise ValueError('Smoke source cannot supply a normal run')
    if result.get('status') not in ('completed', 'smoke_completed'): raise ValueError('Source is incomplete')
    coverage.verify(source, plan)
    records = [result['initial'], *[record for arm in result['policies'].values() for record in arm.values()]]
    bench.validate_saved_checks(source, records)
    from firered_rl.evaluation import digest
    suite = read_json(source / plan['suites']['validation']); requested = len(suite['cases'])
    scores = {}
    for job in plan['jobs']:
        name = job['name']; summary = read_json(source / 'runs' / name / 'summary.json')
        if (summary.get('status') != 'completed' or summary.get('stop_reason') != 'timestep_budget'
                or summary.get('player_timesteps', 0) < job['target_timesteps']):
            raise ValueError('Source training budget is incomplete')
        coverage.training_provenance(source, plan, job, summary)
        meta = read_json(source / 'runs' / name / 'checkpoints/best.json')
        best = result['policies'][name]['best']
        if (meta.get('suite_sha256') != digest(suite) or meta.get('requested') != requested
                or type(meta.get('wins')) is not int or not 0 <= meta['wins'] <= requested
                or meta.get('checkpoint_sha256') != best['checkpoint_sha256']
                or meta.get('timesteps') != best['timesteps']):
            raise ValueError('Source native best validation evidence differs')
        scores[name] = meta['wins'] / requested
        for kind, record in result['policies'][name].items():
            path = inside(source, record['checkpoint'])
            if checksum(path) != record['checkpoint_sha256'] or checkpoint_steps(path) != record['timesteps']:
                raise ValueError('Source checkpoint hash or timesteps differ')
            if kind == 'final' and record['timesteps'] != summary['player_timesteps']:
                raise ValueError('Source final steps differ from completed training')
    new_name = max(scores, key=scores.get)
    expected = (dict(name=new_name, **result['policies'][new_name]['best'], validation_rate=scores[new_name])
        if scores[new_name] > plan['fallback']['validation_rate'] else
        dict(name='previous-selected', checkpoint=plan['fallback']['checkpoint'],
             checkpoint_sha256=plan['fallback']['sha256'], validation_rate=plan['fallback']['validation_rate'],
             timesteps=plan['fallback']['timesteps']))
    selected = result['selected_by_validation']
    if any(selected.get(key) != expected[key] for key in ('name', 'checkpoint', 'checkpoint_sha256', 'validation_rate')):
        raise ValueError('Source selected model differs from recorded validation selection')
    parent = inside(source, selected['checkpoint']); sha = checksum(parent); steps = checkpoint_steps(parent)
    if sha != selected['checkpoint_sha256']: raise ValueError('Selected source model hash differs')
    smoke_override = None
    if smoke and plan.get('smoke'):
        record = result['policies']['balanced-pool']['best']
        parent = inside(source, record['checkpoint']); sha = checksum(parent); steps = checkpoint_steps(parent)
        smoke_override = dict(checkpoint=record['checkpoint'], sha256=sha,
            rationale='Smoke-only operational feature-enabled parent; source actual selection is preserved as evidence.')
    migration = switches.checkpoint_metadata(parent, enabled=True,
        starting_timesteps=plan['settings']['migration_template_timesteps'])
    from sb3_contrib import MaskablePPO
    from firered_rl.shared_move_policy import policy_architecture
    model = MaskablePPO.load(parent, device='cpu')
    if (policy_architecture(model) != 'matchup-switches' or not model.policy.features_extractor.relational_enabled
            or model.policy.features_extractor.bench_features_enabled is not True):
        raise ValueError('Common parent must have matchup and switch features enabled')
    del model
    fallback_records = {
        'current-selected': dict(path=parent, sha256=sha, timesteps=steps,
            source_checkpoint=smoke_override['checkpoint'] if smoke_override else selected['checkpoint']),
        'small-pool-best': dict(path=inside(source, result['policies']['small-pool']['best']['checkpoint']),
            sha256=result['policies']['small-pool']['best']['checkpoint_sha256'],
            timesteps=result['policies']['small-pool']['best']['timesteps'],
            source_checkpoint=result['policies']['small-pool']['best']['checkpoint']),
        'prior-feature-final': dict(path=inside(source, 'inputs/start.zip'), sha256=plan['pins']['inputs/start.zip'],
            timesteps=plan['settings']['starting_timesteps'], source_checkpoint='inputs/start.zip')}
    for record in fallback_records.values():
        if checksum(record['path']) != record['sha256'] or checkpoint_steps(record['path']) != record['timesteps']:
            raise ValueError('Frozen fallback identity changed')
    teams = {group: [inside(source, relative) for relative in paths] for group, paths in plan['teams'].items()}
    return dict(source=source, plan=plan, result=result, teams=teams,
        suites={name: inside(source, relative) for name, relative in plan['suites'].items()},
        reward=inside(source, 'inputs/reward.json'), selected=selected, fallbacks=fallback_records, smoke_source_override=smoke_override,
        start=dict(path=parent, sha256=sha, timesteps=steps, migration_template_timesteps=migration['template_timesteps']))


def freeze(root, args):
    from firered_rl.teacher_training_suites import freeze_teacher_suites
    from firered_rl.showdown_env import validate_showdown
    source = source_inputs(args.source_run, args.smoke); validate_showdown(ROOT)
    if root == source['source'] or root.is_relative_to(source['source']): raise ValueError('Output must be outside source run')
    if type(args.timesteps) is not int or args.timesteps < 1: raise ValueError('Timesteps must be positive')
    start = source['start']
    settings = dict(seed=261140, starting_timesteps=start['timesteps'], timesteps=256 if args.smoke else args.timesteps,
        migration_template_timesteps=start['migration_template_timesteps'], environments=2 if args.smoke else 16,
        learning_rate=.0001, ent_coef=.01, rollout_steps=128, batch_size=128, epochs=4,
        eval_every=128 if args.smoke else 50000, max_steps=100, early_stopping=False,
        selection='Broad frozen validation only; frozen fallbacks retain ties; endpoints never select.',
        comparison='All-action Rules behavior cloning then whole-policy PPO versus matched whole-policy PPO; both reset PPO optimizer.')
    warmstart = dict(battles=40 if args.smoke else 2000, max_rows=1024 if args.smoke else 50000,
        decisions_per_battle=24, epochs=1 if args.smoke else 8, batch_size=32 if args.smoke else 256,
        learning_rate=.0001, validation_fraction=.2, seed=261140)
    root.mkdir(parents=True, exist_ok=False)
    for relative in ('inputs/suites', 'inputs/fallbacks', 'console', 'models'): (root / relative).mkdir(parents=True)
    shutil.copyfile(start['path'], root / 'inputs/start.zip'); shutil.copyfile(source['reward'], root / 'inputs/reward.json')
    write_json(root / 'inputs/settings.json', settings); teams = {}
    for group in ('old', 'new', 'mirror'):
        (root / 'inputs/teams' / group).mkdir(parents=True); teams[group] = []
        for index, path in enumerate(source['teams'][group]):
            relative = f'inputs/teams/{group}/team-{index+1:04}.json'
            shutil.copyfile(path, root / relative); teams[group].append(relative)
    config = read_json(source['source'] / 'inputs/team-recipe.json')
    shutil.copyfile(source['source'] / 'inputs/team-recipe.json', root / 'inputs/team-recipe.json')
    shutil.copyfile(source['source'] / 'inputs/catalog.json', root / 'inputs/catalog.json')
    exclusions = [read_json(path) for paths in source['teams'].values() for path in paths]
    for path in source['suites'].values(): exclusions.extend(asset['team'] for asset in read_json(path)['assets'].values())
    exclusions.extend(read_json(source['source'] / 'inputs/excluded-teams.json'))
    write_json(root / 'inputs/excluded-teams.json', exclusions)
    prepared = freeze_teacher_suites(root, ROOT, config, exclusions,
        validation_teams=12 if args.smoke else 36, endpoint_teams=12 if args.smoke else 24,
        validation_battles=8 if args.smoke else 2048, endpoint_battles=8 if args.smoke else 1000,
        validation_seed=261140, endpoint_seed=261141, max_steps=settings['max_steps'])
    teams.update(prepared['teams']); suites = prepared['suites'].copy()
    for condition in ('seen-original', 'seen-reversed', 'mirror'):
        relative = f'inputs/suites/{condition}.json'; shutil.copyfile(source['suites'][condition], root / relative); suites[condition] = relative
    # Historical heldouts remain explicitly historical and do not become the fresh endpoint.
    for condition in ('heldout-original', 'heldout-reversed'):
        shutil.copyfile(source['suites'][condition], root / f'inputs/suites/historical-{condition}.json')
    groups = {name: (dict(weight=weight, teams=[str(root / path) for path in teams['mirror']]) if name == 'fixed-mirror'
        else dict(weight=weight, directory=str(root / 'inputs/teams' / name))) for name, weight in WEIGHTS.items()}
    write_json(root / 'inputs/mixture.json', dict(version=1, groups=groups))
    fallbacks = {}
    for name, record in source['fallbacks'].items():
        relative = f'inputs/fallbacks/{name}.zip'; shutil.copyfile(record['path'], root / relative)
        fallbacks[name] = dict(checkpoint=relative, sha256=record['sha256'], timesteps=record['timesteps'], source_checkpoint=record['source_checkpoint'])
    source_evidence = dict(source_run=str(source['source']), source_selected=source['selected'],
        start_checkpoint_sha256=start['sha256'], source_plan_sha256=checksum(source['source'] / 'plan.json'),
        source_result_sha256=checksum(source['source'] / 'result.json'), smoke_source_override=source['smoke_source_override'],
        parent_choice='Current coverage validation-selected model, declared before fresh suites or scores; common parent stays fixed.',
        source_preserved=True)
    write_json(root / 'inputs/source-evidence.json', source_evidence)
    jobs = make_jobs(settings)
    configuration = dict(settings=settings, warmstart=warmstart, teams=teams, suites=suites, jobs=jobs, fallbacks=fallbacks, source=source_evidence, teacher_suites=prepared)
    write_json(root / 'inputs/configuration.json', configuration)
    for name in ('launcher', 'warmstart', *[job['name'] for job in jobs]): (root / 'console' / (name+'.log')).touch()
    plan = dict(version=1, experiment=EXPERIMENT, smoke=bool(args.smoke), **configuration,
        source_start_sha256=start['sha256'], implementation=implementation_identity(), preparation=None,
        pins={str(path.relative_to(root)): checksum(path) for path in (root / 'inputs').rglob('*') if path.is_file()})
    write_json(root / 'plan.json', plan); return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != EXPERIMENT: raise ValueError('Unsupported teacher-training plan')
    if plan.get('implementation') != implementation_identity(): raise ValueError('Implementation changed; use a new run directory')
    expected = {inside(root, relative) for relative in plan['pins']}
    actual = {path.resolve() for path in (root / 'inputs').rglob('*') if path.is_file()}
    # Interrupted preparation may leave only these uncommitted generated checkpoints.
    pending = {root / f'inputs/{name}.zip' for name in ('ppo-only', 'teacher-ppo')} if plan['preparation'] is None else set()
    if actual - expected - pending or expected - actual: raise ValueError('Pinned input membership changed')
    for relative, sha in plan['pins'].items():
        if checksum(inside(root, relative)) != sha: raise ValueError(f'Pinned input changed: {relative}')
    config = read_json(root / 'inputs/configuration.json')
    if any(plan[key] != config[key] for key in ('settings', 'warmstart', 'teams', 'suites', 'jobs', 'fallbacks', 'source', 'teacher_suites')):
        raise ValueError('Frozen experiment configuration changed')
    if plan['settings'] != read_json(root / 'inputs/settings.json') or plan['jobs'] != make_jobs(plan['settings']):
        raise ValueError('Settings or matched budgets changed')
    if (plan['pins']['inputs/start.zip'] != plan['source_start_sha256']
            or checkpoint_steps(root / 'inputs/start.zip') != plan['settings']['starting_timesteps']):
        raise ValueError('Common parent identity changed')
    switches.checkpoint_metadata(root / 'inputs/start.zip', enabled=True,
        starting_timesteps=plan['settings']['migration_template_timesteps'])
    if set(plan['suites']) != {'validation', *CONDITIONS}: raise ValueError('Frozen suite membership differs')
    from firered_rl.teacher_training_suites import verify_frozen_teacher_suites
    verify_frozen_teacher_suites(root, ROOT, read_json(root / 'inputs/team-recipe.json'),
        read_json(root / 'inputs/excluded-teams.json'), plan['teacher_suites'])
    if checksum(root / 'inputs/catalog.json') != checksum(inside(ROOT, read_json(root / 'inputs/team-recipe.json')['catalog'])):
        raise ValueError('Current catalog differs from pinned recipe snapshot')
    from firered_rl.evaluation import validate_suite
    for relative in plan['suites'].values(): validate_suite(read_json(root / relative))
    if plan['preparation'] is not None:
        from firered_rl.teacher_warmstart import verify as verify_warmstart
        report = verify_warmstart(root, plan)
        if report != plan['preparation']['metadata']:
            raise ValueError('Warmstart report differs from frozen preparation evidence')
        validate_preparation_report(plan, report)
        for relative, sha in plan['preparation']['artifacts'].items():
            if checksum(inside(root, relative)) != sha: raise ValueError(f'Prepared artifact changed: {relative}')
        for job in plan['jobs']:
            if checkpoint_steps(root / job['checkpoint']) != job['starting_timesteps']:
                raise ValueError('Warmstart changed PPO timestep accounting')


def validate_preparation_report(plan, report):
    if (report.get('version') != 1 or report.get('status') != 'completed'
            or report.get('source_sha256') != plan['source_start_sha256']
            or report.get('source_timesteps') != plan['settings']['starting_timesteps']
            or report.get('no_evaluation_team_labels') is not True
            or report.get('no_game_outcome_labels') is not True):
        raise ValueError('Warmstart source or training-only evidence differs')
    fit = report.get('fit', {})
    if (fit.get('critic_parameters_unchanged') is not True or fit.get('shared_encoder_trained') is not True
            or not fit.get('frozen_critic_parameters') or fit.get('gradient_updates', 0) < 1):
        raise ValueError('Warmstart did not preserve critic heads or train actor/shared encoder')
    resets = report.get('optimizer_resets', {})
    if set(resets) != {'ppo-only', 'teacher-ppo'}: raise ValueError('Both PPO optimizer resets must be recorded')
    for record in resets.values():
        if (record.get('state_entries') != 0 or record.get('learning_rate') != plan['settings']['learning_rate']
                or record.get('num_timesteps') != plan['settings']['starting_timesteps']):
            raise ValueError('Prepared optimizer or timestep evidence differs')
    if resets['ppo-only'].get('parameters_sha256') != report.get('source_parameters_sha256'):
        raise ValueError('PPO-only reset changed source policy parameters')


def prepare(root, plan):
    if plan['preparation'] is not None: verify(root, plan); return plan['preparation']
    from firered_rl.teacher_warmstart import prepare as prepare_warmstart
    def announce(message):
        print(message, flush=True)
        with (root / 'console/warmstart.log').open('a') as handle: handle.write(str(message)+'\n')
    metadata = prepare_warmstart(root, plan, announce=announce)
    validate_preparation_report(plan, metadata)
    artifacts = {}
    for relative in ('inputs/ppo-only.zip', 'inputs/teacher-ppo.zip', 'models/teacher-only.zip', 'warmstart/report.json'):
        path = root / relative
        if not path.is_file(): raise ValueError(f'Warmstart omitted required artifact: {relative}')
        artifacts[relative] = checksum(path)
    if artifacts['inputs/teacher-ppo.zip'] != artifacts['models/teacher-only.zip']:
        raise ValueError('Teacher-only export differs from treatment initialization')
    for name in ('ppo-only', 'teacher-ppo'):
        path = root / f'inputs/{name}.zip'
        if checkpoint_steps(path) != plan['settings']['starting_timesteps']:
            raise ValueError('Warmstart changed PPO timestep accounting')
        switches.checkpoint_metadata(path, enabled=True, starting_timesteps=plan['settings']['migration_template_timesteps'])
    preparation = dict(metadata=metadata, artifacts=artifacts)
    plan['preparation'] = preparation
    for relative, sha in artifacts.items():
        if relative.startswith('inputs/'): plan['pins'][relative] = sha
    write_json(root / 'plan.json', plan); verify(root, plan); return preparation


def command(root, plan, job):
    if plan['preparation'] is None: raise ValueError('Warmstart preparation must complete before training')
    return coverage.command(root, plan, job)


def launch(root, plan, concurrency):
    if concurrency not in (1, 2): raise ValueError('At most two training jobs are allowed')
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


def frozen(root, plan, label, condition, path):
    relative = str(path.relative_to(root))
    frozen_plan = dict(models=dict(candidate=dict(checkpoint=relative, sha256=checksum(path),
        timesteps=checkpoint_steps(path))), suites=plan['suites'], pins=plan['pins'], implementation=plan['implementation'])
    check = dict(label=label, checkpoint='candidate', baseline=None, pool=condition,
        order='reversed' if condition.endswith('-reversed') else 'original', assisted=False, suite=condition)
    cached = bench.diagnostic.cached_check(root, frozen_plan, check)
    def announce(message):
        with (root / 'console/launcher.log').open('a') as handle:
            handle.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())+' '+message+'\n')
    if cached is None: announce('Starting frozen check: '+label)
    result = bench.frozen(root, plan, label, condition, path)
    if cached is None: announce(f"Completed {label}: {result['candidate_wins']}/{result['requested_battles']} wins; truncated={result['truncated']}")
    return result


def frozen_record(root, plan, name, checkpoint, *, validation=True, cache=None, conditions=None):
    cache = {} if cache is None else cache
    path = inside(root, checkpoint); sha = checksum(path)
    conditions = conditions if conditions is not None else (('validation', *CONDITIONS) if validation else CONDITIONS)
    checks = {}
    for condition in conditions:
        key = (sha, condition)
        if key not in cache: cache[key] = frozen(root, plan, name+'-'+condition, condition, path)
        checks[condition] = cache[key]
    return dict(checkpoint=checkpoint, checkpoint_sha256=sha, timesteps=checkpoint_steps(path), checks=checks,
        balanced_validation=checks['validation']['wins_per_requested'] if validation else None)


def evaluate_fallbacks(root, plan, *, complete=True):
    records, cache = {}, {}
    for name, record in plan['fallbacks'].items():
        records[name] = frozen_record(root, plan, name, record['checkpoint'], cache=cache,
            conditions=('validation', *CONDITIONS) if complete else ('validation',))
    bench.validate_saved_checks(root, list(records.values()))
    winner = max(records, key=lambda name: records[name]['balanced_validation'])
    value = dict(records=records, selected=winner, criterion='New broad validation only; endpoint scores never select',
        parent_checkpoint_sha256=plan['source_start_sha256'])
    write_json(root / 'fallbacks.json', value); return value


def training_provenance(root, plan, job, summary):
    adjusted = dict(plan, source_start_sha256=plan['pins'][job['checkpoint']])
    value = coverage.training_provenance(root, adjusted, job, summary)
    run = root / 'runs' / job['name']
    paths = [run / 'config.json']
    if (run / 'sessions/000001/config.json').exists(): paths.append(run / 'sessions/000001/config.json')
    for path in paths:
        actual = read_json(path); effective = actual.get('effective_ppo', {})
        expected = dict(learning_rate=plan['settings']['learning_rate'], ent_coef=plan['settings']['ent_coef'],
            n_steps=plan['settings']['rollout_steps'], batch_size=plan['settings']['batch_size'], n_epochs=plan['settings']['epochs'])
        session_id = actual.get('session', {}).get('id')
        if type(session_id) is not int or session_id < 1:
            raise ValueError('Invalid native training session identity')
        expected_seed = (job['seed'] + (session_id-1)*1000003) % (2**31)
        if (any(effective.get(key) != value for key, value in expected.items())
                or actual.get('seed') != expected_seed
                or actual.get('saved_args', {}).get('seed', actual.get('seed') if session_id == 1 else None) != job['seed']
                or actual.get('environments') != plan['settings']['environments']):
            raise ValueError('Actual PPO settings, session seed or environment count differ from the matched plan')
    value.update(common_parent_sha256=plan['source_start_sha256'], optimizer_reset=True,
        warmstart=(job['name']=='teacher-ppo'), preparation_sha256=plan['preparation']['artifacts']['warmstart/report.json'])
    return value


def finish(root, plan):
    from firered_rl.evaluation import digest
    verify(root, plan)
    fallbacks = evaluate_fallbacks(root, plan); records = fallbacks['records']
    cache = {(record['checkpoint_sha256'], condition): check for record in records.values() for condition, check in record['checks'].items()}
    teacher = frozen_record(root, plan, 'teacher-only', 'models/teacher-only.zip', cache=cache)
    requested = len(read_json(root / plan['suites']['validation'])['cases'])
    suite_hash = digest(read_json(root / plan['suites']['validation']))
    policies, training = {}, {}
    for job in plan['jobs']:
        name = job['name']; run = root / 'runs' / name
        summary = read_json(run / 'summary.json'); meta = read_json(run / 'checkpoints/best.json')
        if (summary.get('status') != 'completed' or summary.get('stop_reason') != 'timestep_budget'
                or summary.get('player_timesteps', 0) < job['target_timesteps']):
            raise ValueError('Training did not complete the fixed budget')
        if (meta.get('suite_sha256') != suite_hash or meta.get('requested') != requested
                or type(meta.get('wins')) is not int or not 0 <= meta['wins'] <= requested):
            raise ValueError('Best checkpoint did not use frozen broad validation')
        provenance = training_provenance(root, plan, job, summary)
        history = read_json(run / 'evaluation-history.json')
        if history.get('identity', {}).get('suite_sha256') != suite_hash: raise ValueError('Validation history suite differs')
        policies[name] = {}
        for kind in ('best', 'final'):
            relative = f'runs/{name}/checkpoints/{kind}.zip'; path = root / relative; sha = checksum(path); steps = checkpoint_steps(path)
            switches.checkpoint_metadata(path, enabled=True, starting_timesteps=plan['settings']['migration_template_timesteps'])
            if kind == 'best' and (sha != meta['checkpoint_sha256'] or steps != meta['timesteps']):
                raise ValueError('Native best checkpoint identity differs')
            if kind == 'final' and steps != summary['player_timesteps']: raise ValueError('Final timestep evidence differs')
            matching = [entry for entry in history['records'] if entry.get('status') == 'completed'
                and entry.get('checkpoint_sha256') == sha and entry.get('timesteps') == steps and entry.get('requested') == requested]
            if kind == 'final' and not matching: raise ValueError('Final lacks matching completed validation')
            wins = meta['wins'] if kind == 'best' else matching[-1]['wins']
            if type(wins) is not int or not 0 <= wins <= requested: raise ValueError('Invalid validation win count')
            record = frozen_record(root, plan, name+'-'+kind, relative, validation=False, cache=cache)
            record['balanced_validation'] = wins/requested
            record['mirror_change_from_parent'] = record['checks']['mirror']['wins_per_requested']-records['current-selected']['checks']['mirror']['wins_per_requested']
            policies[name][kind] = record
        training[name] = dict(starting_timesteps=job['starting_timesteps'], ending_timesteps=summary['player_timesteps'],
            added_timesteps=summary['player_timesteps']-job['starting_timesteps'], elapsed_seconds=summary['elapsed_seconds'], provenance=provenance)
    # Selection is defined entirely on the new validation, including the BC-only checkpoint.
    choices = [(name, record) for name, record in records.items()]
    choices += [('teacher-only', teacher)]
    choices += [(name, arm['best']) for name, arm in policies.items()]
    selected_name, selected = max(choices, key=lambda pair: pair[1]['balanced_validation'])
    selected_record = dict(name=selected_name, checkpoint=selected['checkpoint'], checkpoint_sha256=selected['checkpoint_sha256'],
        timesteps=selected['timesteps'], validation_rate=selected['balanced_validation'])
    exports = {}
    to_export = {**records, 'teacher-only': teacher, **{name+'-'+kind: record for name, arm in policies.items() for kind, record in arm.items()}, 'selected':selected}
    for name, record in to_export.items():
        export_name = {'current-selected':'fallback-current-selected', 'small-pool-best':'fallback-small-best',
            'prior-feature-final':'fallback-feature-final'}.get(name, name)
        target = root / f'models/{export_name}.zip'; origin = root / record['checkpoint']
        if target != origin: shutil.copyfile(origin, target)
        if checksum(target) != record['checkpoint_sha256']: raise RuntimeError('Exported checkpoint hash differs')
        exports[name] = dict(checkpoint=str(target.relative_to(root)), sha256=record['checkpoint_sha256'], timesteps=record['timesteps'])
    verify(root, plan)
    all_records = [*records.values(), teacher, *[record for arm in policies.values() for record in arm.values()]]
    saved = bench.validate_saved_checks(root, all_records)
    difference = sum(policies['teacher-ppo']['best']['checks'][key]['wins_per_requested']-
        policies['ppo-only']['best']['checks'][key]['wins_per_requested'] for key in ('fresh-original', 'fresh-reversed'))/2
    result = dict(status='smoke_completed' if plan['smoke'] else 'completed', selected_by_validation=selected_record,
        fallbacks=records, teacher_only=teacher, policies=policies, training=training, exports=exports, saved_evidence=saved,
        common_parent_sha256=plan['source_start_sha256'], endpoint_used_for_selection=False, fallback_retains_ties=True,
        comparison=dict(selected_fresh_mean_difference=difference,
            scope='One paired training seed, familiar species and simple moves; fresh frozen compositions. Not unrestricted competitive play.'))
    write_json(root / 'selection.json', dict(selected=selected_record, endpoint_used_for_selection=False, fallback_retains_ties=True))
    write_json(root / 'models/manifest.json', exports); write_json(root / 'result.json', result); return result


def collect(root, plan):
    result = read_json(root / 'result.json') if (root / 'result.json').exists() else {}
    status = read_json(root / 'status.json') if (root / 'status.json').exists() else {'status':'prepared'}
    lines = ['# Rules teacher warmstart and PPO', '',
        'Both jobs start from the same predeclared coverage-selected parent. Both reset PPO optimizer state; only teacher-ppo first learns to imitate legal Rules actions on old/new training teams; fixed mirror uses soft prior-policy rehearsal. BC updates actor/shared encoder with critic heads frozen; both PPO jobs then train the whole policy.',
        'Training stays 25% original teams / 65% balanced teams / 10% fixed mirror, with fixed party order and the same reward and PPO settings.',
        'Selection uses the new 36-composition validation suite (2048 battles, both seats and candidate bench orders). The separate 24-composition fresh endpoint never selects a model. Old heldout pools are retained as historical evidence. Smoke uses 12/12 teams and 8 battles per suite.', '',
        f"Status: {status['status']}. Additional PPO budget: {plan['settings']['timesteps']:,} steps per job.",
        'Frozen scores are against Rules; draws and truncations are non-wins. BC-only timesteps count prior PPO steps, not its supervised gradient updates.', '',
        '| Model | PPO steps | Broad validation | Seen original | Seen reversed | Fresh original | Fresh reversed | Mirror |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows = list(result.get('fallbacks', {}).items())
    if result.get('teacher_only'): rows.append(('teacher-only', result['teacher_only']))
    rows += [(name+'-'+kind, record) for name, arm in result.get('policies', {}).items() for kind, record in arm.items()]
    for name, record in rows:
        rates = [f"{record['checks'][condition]['wins_per_requested']:.1%}" for condition in CONDITIONS]
        lines.append(f"| {name} | {record['timesteps']:,} | {record['balanced_validation']:.1%} | {' | '.join(rates)} |")
    if result:
        lines += ['', f"Selected by broad validation: **{result['selected_by_validation']['name']}**, `models/selected.zip`.",
            'Frozen fallbacks win ties. The common training parent never changes in response to new validation or endpoint results.',
            f"Validation-selected treatment minus control fresh mean: {100*result['comparison']['selected_fresh_mean_difference']:+.2f} percentage points.", result['comparison']['scope']]
    lines += ['', 'Warmstart diagnostics and training-only teacher agreement: warmstart/report.json. Each trainer writes evaluations.md, updates.jsonl and PPO CSV metrics.',
        f"Logs: `tail -n 20 -F {root}/console/*.log`", 'Memory: `watch -n 2 free -h`',
        f"Validation progress: `watch -n 5 'for f in {root}/runs/*/evaluations.md; do echo \"$f\"; cat \"$f\"; done'`",
        f"Teams: `python run_teacher_training.py --output-dir {shlex.quote(str(root))} --inspect-teams all --limit 3 --full-json`",
        f"Resume: `python run_teacher_training.py --output-dir {shlex.quote(str(root))} --continue --jobs 2`",
        f"Play: `python mgba/play_gui.py --rom \"$PWD/FireRed.gba\" --source \"$PWD/FireRed.sav\" --workers 2 --speed 1 --checkpoint {shlex.quote(str(root / 'models/selected.zip'))}`",
        f"Permanent copy: `mkdir -p models && cp --no-clobber {shlex.quote(str(root / 'models/selected.zip'))} models/{root.name}-selected.zip`",
        'Source runs, models and the original 400k reference are unchanged. Upload archive excludes model weights, bulk logs and raw teacher arrays.', '']
    (root / 'report.md').write_text('\n'.join(lines)); (root / 'play-summary.md').write_text('\n'.join(lines))
    temporary = root / 'teacher-training-results.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if path.is_file() and path.suffix in ('.json','.jsonl','.md','.csv') and relative.parts[0] != 'console':
                archive.write(path, str(relative))
    temporary.replace(root / 'teacher-training-results.zip')


def inspect(root, plan, group, limit, full):
    print(f'Training mixture: {WEIGHTS}; {root / "inputs/mixture.json"}')
    for name in plan['teams'] if group == 'all' else (group,):
        paths = plan['teams'][name]; print(f'\n{name}: {len(paths)} teams')
        for relative in paths[:limit]:
            team = read_json(root / relative); print(root / relative)
            if full: print(json.dumps(team, indent=2))
            else:
                for mon in team['party']: print(f"  {mon['species']}: {', '.join(mon['moves'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('.local/run/training-coverage-01'))
    parser.add_argument('--output-dir', '--run-dir', dest='run_dir', type=Path, default=Path('.local/run/teacher-training-01'))
    parser.add_argument('--jobs', type=int, choices=(1,2), default=2)
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-teams', nargs='?', const='all', choices=('all','old','new','mirror','validation','fresh-test'))
    parser.add_argument('--limit', type=int, default=3); parser.add_argument('--full-json', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); root = args.run_dir.expanduser().resolve()
    if args.limit < 0 or args.timesteps is not None and args.timesteps < 1: parser.error('Limit must be nonnegative and timesteps positive')
    if args.continuing and args.timesteps is not None: parser.error('--continue keeps the saved budget; omit --timesteps')
    if args.inspect_teams is not None:
        plan = read_json(root / 'plan.json'); verify(root,plan); inspect(root,plan,args.inspect_teams,args.limit,args.full_json); return 0
    if args.timesteps is None: args.timesteps=500000
    import torch
    torch.set_num_threads(1); root.parent.mkdir(parents=True,exist_ok=True)
    lock = (root.parent / ('.'+root.name+'.teacher-training.lock')).open('a')
    plan=None;owned=False;stopped=threading.Event();monitor=None
    previous=signal.signal(signal.SIGTERM,interrupt)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owned=True
        if args.continuing: plan=read_json(root/'plan.json');verify(root,plan)
        else:plan=freeze(root,args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {root}/console/*.log\nMemory: watch -n 2 free -h\nTeams: python run_teacher_training.py --output-dir {root} --inspect-teams all",flush=True)
        if args.prepare_only:write_json(root/'status.json',dict(status='prepared'));return 0
        monitor=threading.Thread(target=memory_monitor,args=(root,stopped),daemon=True);monitor.start()
        write_json(root/'status.json',dict(status='running',phase='fallback-validation-and-checks'))
        evaluate_fallbacks(root,plan,complete=False)
        write_json(root/'status.json',dict(status='running',phase='teacher-warmstart'));prepare(root,plan)
        write_json(root/'status.json',dict(status='running',phase='training'));collect(root,plan)
        launch(root,plan,args.jobs)
        write_json(root/'status.json',dict(status='running',phase='frozen-checks'));result=finish(root,plan)
        write_json(root/'status.json',dict(status=result['status']));return 0
    except (KeyboardInterrupt,Exception) as error:
        if plan is not None and owned:write_json(root/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',message=str(error)))
        print(f'Stopped: {error or "interrupted"}',file=sys.stderr,flush=True);return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        stopped.set()
        if monitor:monitor.join()
        if plan is not None and owned:
            collect(root,plan);print(f"Report: {root/'report.md'}\nUpload: {root/'teacher-training-results.zip'}",flush=True)
        signal.signal(signal.SIGTERM,previous);lock.close()


if __name__=='__main__':raise SystemExit(main())
