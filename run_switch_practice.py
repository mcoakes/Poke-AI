#!/usr/bin/env python3
"""Compare longer PPO with targeted Rules decision practice plus matched PPO.

At most two jobs. Watch console/*.log and runs/*/evaluations.md; memory.csv records
available RAM and swap. Ctrl+C pauses safely; --continue keeps the frozen budget.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import signal
import sys
import threading
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_teacher_training as teacher
from run_fixed_team_lab import checksum, write_json, interrupt
from run_parallel_training import checkpoint_steps
from run_staged_team_lab import memory_monitor
bench, coverage, switches = teacher.bench, teacher.coverage, teacher.switches
read_json, inside = teacher.read_json, teacher.inside
CONDITIONS, WEIGHTS = teacher.CONDITIONS, teacher.WEIGHTS
EXPERIMENT = 'switch-practice'
command, launch, inspect = teacher.command, teacher.launch, teacher.inspect


def implementation_identity():
    value = teacher.implementation_identity()
    for relative in ('run_switch_practice.py', 'src/firered_rl/decision_weighted_warmstart.py'):
        value[relative] = checksum(ROOT / relative)
    return value


def practice_settings(smoke=False):
    return dict(battles=40 if smoke else 6000, max_rows=1024 if smoke else 50000,
        decisions_per_battle=24, epochs=1 if smoke else 8, batch_size=32 if smoke else 256,
        learning_rate=.0001, validation_fraction=.2, seed=261150,
        capacities=dict(move=614, forced=205, switch=103, mirror=102) if smoke else
                   dict(move=30000, forced=10000, switch=5000, mirror=5000),
        weights=dict(move=1., forced=2., switch=4.), mirror_mass=.1)


def make_jobs(settings):
    return [dict(name=name, checkpoint=f'inputs/{name}.zip', starting_timesteps=settings['starting_timesteps'],
        target_timesteps=settings['starting_timesteps']+settings['timesteps'], seed=settings['seed'],
        bench_features_enabled=True, shuffle_bench_order=False, mixture='inputs/mixture.json', weights=WEIGHTS.copy())
        for name in ('ppo-only', 'switch-practice')]


def source_inputs(source, smoke=False):
    source = Path(source).expanduser().resolve()
    plan, result = (read_json(source / name) for name in ('plan.json', 'result.json'))
    if plan.get('version') != 1 or plan.get('experiment') != 'teacher-training':
        raise ValueError('Source must be a completed version-1 teacher-training run')
    if plan.get('smoke') and not smoke: raise ValueError('Smoke source cannot supply normal training')
    if result.get('status') not in ('completed', 'smoke_completed'): raise ValueError('Source is incomplete')
    teacher.verify(source, plan)
    all_records = [*result['fallbacks'].values(), result['teacher_only'],
        *[record for arm in result['policies'].values() for record in arm.values()]]
    bench.validate_saved_checks(source, all_records)
    from firered_rl.evaluation import digest
    suite = read_json(source / plan['suites']['validation']); requested = len(suite['cases']); suite_hash = digest(suite)
    for job in plan['jobs']:
        name = job['name']; summary = read_json(source / 'runs' / name / 'summary.json')
        if (summary.get('status') != 'completed' or summary.get('stop_reason') != 'timestep_budget'
                or summary.get('player_timesteps', 0) < job['target_timesteps']):
            raise ValueError('Source training budget is incomplete')
        teacher.training_provenance(source, plan, job, summary)
        meta = read_json(source / 'runs' / name / 'checkpoints/best.json'); best = result['policies'][name]['best']
        if (meta.get('suite_sha256') != suite_hash or meta.get('requested') != requested
                or type(meta.get('wins')) is not int or not 0 <= meta['wins'] <= requested
                or meta.get('checkpoint_sha256') != best['checkpoint_sha256'] or meta.get('timesteps') != best['timesteps']
                or best['balanced_validation'] != meta['wins']/requested):
            raise ValueError('Source native best selection differs from frozen validation')
        final = result['policies'][name]['final']
        if final['timesteps'] != summary['player_timesteps']: raise ValueError('Source final timestep evidence differs')
    for record in all_records:
        path = inside(source, record['checkpoint'])
        if checksum(path) != record['checkpoint_sha256'] or checkpoint_steps(path) != record['timesteps']:
            raise ValueError('Source model hash or timestep count differs')
    choices = [*result['fallbacks'].items(), ('teacher-only', result['teacher_only']),
        *[(name, arm['best']) for name, arm in result['policies'].items()]]
    name, record = max(choices, key=lambda item: item[1]['balanced_validation'])
    selected = result['selected_by_validation']
    if (selected.get('name') != name or selected.get('validation_rate') != record['balanced_validation']
            or any(selected.get(key) != record[key] for key in ('checkpoint', 'checkpoint_sha256', 'timesteps'))
            or result.get('endpoint_used_for_selection') is not False or result.get('fallback_retains_ties') is not True):
        raise ValueError('Source global selection differs from validation and fallback tie rules')
    # All selection candidates must be measured on this exact broad validation.
    for old in [*result['fallbacks'].values(), result['teacher_only']]:
        check = old['checks']['validation']
        if (check['input_suite_sha256'] != checksum(source / plan['suites']['validation'])
                or check['requested_battles'] != requested
                or old['balanced_validation'] != check['candidate_wins']/requested):
            raise ValueError('Source fallback validation suite or score differs')
    path = inside(source, record['checkpoint'])
    migration = switches.checkpoint_metadata(path, enabled=True,
        starting_timesteps=plan['settings']['migration_template_timesteps'])
    from sb3_contrib import MaskablePPO
    from firered_rl.shared_move_policy import policy_architecture
    model = MaskablePPO.load(path, device='cpu')
    if (policy_architecture(model) != 'matchup-switches' or not model.policy.features_extractor.relational_enabled
            or model.policy.features_extractor.bench_features_enabled is not True):
        raise ValueError('Selected parent must have matchup and direct switch features enabled')
    del model
    return dict(source=source, plan=plan, result=result, selected=selected, record=record,
        start=dict(path=path, sha256=record['checkpoint_sha256'], timesteps=record['timesteps'],
            migration_template_timesteps=migration['template_timesteps']),
        teams={name:[inside(source, relative) for relative in paths] for name, paths in plan['teams'].items()},
        suites={name:inside(source, relative) for name, relative in plan['suites'].items()},
        reward=inside(source, 'inputs/reward.json'))


def import_initial(root, source):
    checks = {}
    for condition in CONDITIONS:
        value = copy.deepcopy(source['record']['checks'][condition]); relative = f'checks/imported-start-{condition}'
        target = root / relative; target.mkdir(parents=True)
        original = inside(source['source'], value['result_directory'])
        for name in ('summary.json', 'suite.json', 'episodes.jsonl', 'decision-audit.json', 'decision-traces.jsonl'):
            shutil.copyfile(original / name, target / name)
        value['result_directory'] = relative; checks[condition] = value
    record = dict(checkpoint='inputs/start.zip', checkpoint_sha256=source['start']['sha256'],
        timesteps=source['start']['timesteps'], balanced_validation=source['record']['balanced_validation'], checks=checks,
        imported=True, source_checkpoint=source['record']['checkpoint'], source_run=str(source['source']),
        rationale='Exact previously validation-selected policy; frozen check evidence is imported, with no new endpoint selection.')
    bench.validate_saved_checks(root, [record]); write_json(root / 'initial.json', record); return record


def freeze(root, args):
    from firered_rl.showdown_env import validate_showdown
    source = source_inputs(args.source_run, args.smoke); validate_showdown(ROOT)
    if root == source['source'] or root.is_relative_to(source['source']): raise ValueError('Output must be outside source run')
    if type(args.timesteps) is not int or args.timesteps < 1: raise ValueError('Additional steps must be positive')
    start = source['start']
    settings = dict(seed=261150, starting_timesteps=start['timesteps'], timesteps=256 if args.smoke else args.timesteps,
        migration_template_timesteps=start['migration_template_timesteps'], environments=2 if args.smoke else 16,
        learning_rate=.0001, ent_coef=.01, rollout_steps=128, batch_size=128, epochs=4,
        eval_every=128 if args.smoke else 100000, max_steps=100, early_stopping=False,
        selection='Unchanged broad validation only; previous selected fallback retains ties. Development checks never select.',
        comparison='Longer PPO versus capped decision-weighted Rules practice then PPO; both reset PPO optimizer state.')
    root.mkdir(parents=True, exist_ok=False)
    for relative in ('inputs/suites', 'console', 'models'): (root / relative).mkdir(parents=True)
    shutil.copyfile(start['path'], root / 'inputs/start.zip')
    shutil.copyfile(start['path'], root / 'models/previous-selected.zip')
    shutil.copyfile(source['reward'], root / 'inputs/reward.json'); write_json(root / 'inputs/settings.json', settings)
    teams = {}
    for group, paths in source['teams'].items():
        (root / 'inputs/teams' / group).mkdir(parents=True); teams[group] = []
        for index, path in enumerate(paths):
            relative = f'inputs/teams/{group}/team-{index+1:04}.json'
            shutil.copyfile(path, root / relative); teams[group].append(relative)
    suites = {}
    for name, path in source['suites'].items():
        relative = f'inputs/suites/{name}.json'; shutil.copyfile(path, root / relative); suites[name] = relative
    for name in ('team-recipe.json', 'catalog.json', 'excluded-teams.json', 'teacher-suite-inventory.json'):
        shutil.copyfile(source['source'] / 'inputs' / name, root / 'inputs' / name)
    groups = {name: (dict(weight=weight, teams=[str(root / path) for path in teams['mirror']]) if name == 'fixed-mirror'
        else dict(weight=weight, directory=str(root / 'inputs/teams' / name))) for name, weight in WEIGHTS.items()}
    write_json(root / 'inputs/mixture.json', dict(version=1, groups=groups))
    source_evidence = dict(source_run=str(source['source']), selected=source['selected'],
        start_checkpoint_sha256=start['sha256'], source_plan_sha256=checksum(source['source'] / 'plan.json'),
        source_result_sha256=checksum(source['source'] / 'result.json'),
        parent_choice='The source validation-selected checkpoint is fixed before practice; the source run and 400k reference remain unchanged.')
    write_json(root / 'inputs/source-evidence.json', source_evidence)
    jobs = make_jobs(settings)
    configuration = dict(settings=settings, practice=practice_settings(args.smoke), teams=teams, suites=suites, jobs=jobs,
        source=source_evidence, teacher_suites=copy.deepcopy(source['plan']['teacher_suites']))
    write_json(root / 'inputs/configuration.json', configuration)
    initial = import_initial(root, source)
    for name in ('launcher', 'practice', *[job['name'] for job in jobs]): (root / 'console' / (name+'.log')).touch()
    plan = dict(version=1, experiment=EXPERIMENT, smoke=bool(args.smoke), **configuration,
        source_start_sha256=start['sha256'], initial_sha256=checksum(root / 'initial.json'),
        implementation=implementation_identity(), preparation=None,
        development_role='The former fresh 24-team pool has now been inspected and is a reused development check, not untouched testing.',
        pins={str(path.relative_to(root)):checksum(path) for path in (root / 'inputs').rglob('*') if path.is_file()})
    write_json(root / 'plan.json', plan); return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != EXPERIMENT: raise ValueError('Unsupported switch-practice plan')
    if plan['implementation'] != implementation_identity(): raise ValueError('Implementation changed; use a new output directory')
    expected = {inside(root, relative) for relative in plan['pins']}
    actual = {path.resolve() for path in (root / 'inputs').rglob('*') if path.is_file()}
    pending = {root / f'inputs/{name}.zip' for name in ('ppo-only', 'switch-practice')} if plan['preparation'] is None else set()
    if actual-expected-pending or expected-actual: raise ValueError('Pinned input membership changed')
    for relative, sha in plan['pins'].items():
        if checksum(inside(root, relative)) != sha: raise ValueError(f'Pinned input changed: {relative}')
    config = read_json(root / 'inputs/configuration.json')
    if any(plan[key] != config[key] for key in ('settings', 'practice', 'teams', 'suites', 'jobs', 'source', 'teacher_suites')):
        raise ValueError('Frozen practice configuration changed')
    if plan['settings'] != read_json(root / 'inputs/settings.json') or plan['jobs'] != make_jobs(plan['settings']):
        raise ValueError('Matched budgets or PPO settings changed')
    if plan['practice'] != practice_settings(plan['smoke']): raise ValueError('Practice safeguards or fixed settings changed')
    if (plan['pins']['inputs/start.zip'] != plan['source_start_sha256']
            or checkpoint_steps(root / 'inputs/start.zip') != plan['settings']['starting_timesteps']):
        raise ValueError('Parent checkpoint identity changed')
    switches.checkpoint_metadata(root / 'inputs/start.zip', enabled=True,
        starting_timesteps=plan['settings']['migration_template_timesteps'])
    if checksum(root / 'initial.json') != plan['initial_sha256']: raise ValueError('Imported baseline record changed')
    initial = read_json(root / 'initial.json')
    if initial['checkpoint_sha256'] != plan['source_start_sha256']: raise ValueError('Baseline policy identity differs')
    bench.validate_saved_checks(root, [initial])
    if set(plan['suites']) != {'validation', *CONDITIONS}: raise ValueError('Frozen suite membership differs')
    from firered_rl.teacher_training_suites import verify_frozen_teacher_suites
    verify_frozen_teacher_suites(root, ROOT, read_json(root / 'inputs/team-recipe.json'),
        read_json(root / 'inputs/excluded-teams.json'), plan['teacher_suites'])
    if checksum(root / 'inputs/catalog.json') != checksum(inside(ROOT, read_json(root / 'inputs/team-recipe.json')['catalog'])):
        raise ValueError('Catalog differs from pinned source recipe')
    if plan['preparation'] is not None:
        from firered_rl.decision_weighted_warmstart import verify as verify_practice
        report = verify_practice(root, plan)
        if report != plan['preparation']['metadata']: raise ValueError('Practice report changed')
        validate_preparation_report(plan, report)
        for relative, sha in plan['preparation']['artifacts'].items():
            if checksum(inside(root, relative)) != sha: raise ValueError(f'Practice artifact changed: {relative}')
        for job in plan['jobs']:
            if checkpoint_steps(root / job['checkpoint']) != job['starting_timesteps']:
                raise ValueError('Practice changed PPO timestep accounting')


def validate_preparation_report(plan, report):
    # Shared guarantees use the teacher schema; only the treatment arm name differs.
    adjusted = copy.deepcopy(report)
    if 'switch-practice' in adjusted.get('optimizer_resets', {}):
        adjusted['optimizer_resets']['teacher-ppo'] = adjusted['optimizer_resets'].pop('switch-practice')
    teacher.validate_preparation_report(plan, adjusted)
    fit=report.get('fit',{})
    if (report.get('ppo_counters_unchanged') is not True
            or fit.get('weights')!=plan['practice']['weights']
            or fit.get('mirror_objective_mass')!=plan['practice']['mirror_mass']):
        raise ValueError('Practice counters, capped weights or mirror objective differ')
    if not plan['smoke'] and any(fit.get('split_counts',{}).get(split,{}).get('switch',0)<1 for split in ('fit','holdout')):
        raise ValueError('Production practice needs ordinary-switch examples in both fitting and heldout training pairs')


def prepare(root, plan):
    if plan['preparation'] is not None: verify(root, plan); return plan['preparation']
    from firered_rl.decision_weighted_warmstart import prepare as prepare_practice
    def announce(message):
        print(message, flush=True)
        with (root / 'console/practice.log').open('a') as handle: handle.write(str(message)+'\n')
    metadata = prepare_practice(root, plan, announce=announce); validate_preparation_report(plan, metadata)
    artifacts = {}
    for relative in ('inputs/ppo-only.zip', 'inputs/switch-practice.zip', 'models/practice-only.zip', 'practice/report.json'):
        path = root / relative
        if not path.is_file(): raise ValueError(f'Practice omitted artifact: {relative}')
        artifacts[relative] = checksum(path)
    if artifacts['inputs/switch-practice.zip'] != artifacts['models/practice-only.zip']:
        raise ValueError('Practice-only export differs from treatment initialization')
    for name in ('ppo-only', 'switch-practice'):
        path = root / f'inputs/{name}.zip'
        if checkpoint_steps(path) != plan['settings']['starting_timesteps']: raise ValueError('Practice changed PPO counters')
        switches.checkpoint_metadata(path, enabled=True, starting_timesteps=plan['settings']['migration_template_timesteps'])
    plan['preparation'] = dict(metadata=metadata, artifacts=artifacts)
    for relative, sha in artifacts.items():
        if relative.startswith('inputs/'): plan['pins'][relative] = sha
    write_json(root / 'plan.json', plan); verify(root, plan); return plan['preparation']


def training_provenance(root, plan, job, summary):
    # Reuse the exact native session/seed/settings and mixture guards, without modifying an old runner.
    adjusted = dict(plan, preparation=dict(artifacts={'warmstart/report.json':plan['preparation']['artifacts']['practice/report.json']}))
    value = teacher.training_provenance(root, adjusted, job, summary)
    value['warmstart'] = job['name']=='switch-practice'
    value['practice_sha256'] = value.pop('preparation_sha256')
    return value


def finish(root, plan):
    from firered_rl.evaluation import digest
    verify(root, plan); initial = read_json(root / 'initial.json')
    cache = {(initial['checkpoint_sha256'], condition):check for condition,check in initial['checks'].items()}
    practice = teacher.frozen_record(root, plan, 'practice-only', 'models/practice-only.zip', cache=cache)
    suite = read_json(root / plan['suites']['validation']); requested=len(suite['cases']); suite_hash=digest(suite)
    policies, training = {}, {}
    for job in plan['jobs']:
        name=job['name']; run=root/'runs'/name; summary=read_json(run/'summary.json'); meta=read_json(run/'checkpoints/best.json')
        if (summary.get('status')!='completed' or summary.get('stop_reason')!='timestep_budget'
                or summary.get('player_timesteps',0)<job['target_timesteps']): raise ValueError('Training budget incomplete')
        if (meta.get('suite_sha256')!=suite_hash or meta.get('requested')!=requested
                or type(meta.get('wins')) is not int or not 0<=meta['wins']<=requested):
            raise ValueError('Native best did not use unchanged broad validation')
        provenance=training_provenance(root,plan,job,summary)
        history=read_json(run/'evaluation-history.json')
        if history.get('identity',{}).get('suite_sha256')!=suite_hash: raise ValueError('Validation history suite differs')
        policies[name]={}
        for kind in ('best','final'):
            relative=f'runs/{name}/checkpoints/{kind}.zip';path=root/relative;sha=checksum(path);steps=checkpoint_steps(path)
            switches.checkpoint_metadata(path,enabled=True,starting_timesteps=plan['settings']['migration_template_timesteps'])
            if kind=='best' and (meta.get('checkpoint_sha256')!=sha or meta.get('timesteps')!=steps):
                raise ValueError('Native best checkpoint differs')
            if kind=='final' and steps!=summary['player_timesteps']: raise ValueError('Final timestep evidence differs')
            matching=[entry for entry in history['records'] if entry.get('status')=='completed'
                and entry.get('checkpoint_sha256')==sha and entry.get('timesteps')==steps and entry.get('requested')==requested]
            if kind=='final' and not matching: raise ValueError('Final lacks completed validation evidence')
            wins=meta['wins'] if kind=='best' else matching[-1]['wins']
            if type(wins) is not int or not 0<=wins<=requested: raise ValueError('Invalid validation count')
            record=teacher.frozen_record(root,plan,name+'-'+kind,relative,validation=False,cache=cache)
            record.update(balanced_validation=wins/requested,
                mirror_change_from_parent=record['checks']['mirror']['wins_per_requested']-initial['checks']['mirror']['wins_per_requested'])
            policies[name][kind]=record
        training[name]=dict(starting_timesteps=job['starting_timesteps'],ending_timesteps=summary['player_timesteps'],
            added_timesteps=summary['player_timesteps']-job['starting_timesteps'],elapsed_seconds=summary['elapsed_seconds'],provenance=provenance)
    choices=[('previous-selected',initial),('practice-only',practice),*[(name,arm['best']) for name,arm in policies.items()]]
    selected_name,selected=max(choices,key=lambda item:item[1]['balanced_validation'])
    choice=dict(name=selected_name,checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],
        timesteps=selected['timesteps'],validation_rate=selected['balanced_validation'])
    exports={}
    for name,record in {'previous-selected':initial,'practice-only':practice,
            **{name+'-'+kind:rec for name,arm in policies.items() for kind,rec in arm.items()},'selected':selected}.items():
        path=root/f'models/{name}.zip';origin=root/record['checkpoint']
        if path!=origin:shutil.copyfile(origin,path)
        if checksum(path)!=record['checkpoint_sha256']:raise RuntimeError('Exported model hash differs')
        exports[name]=dict(checkpoint=str(path.relative_to(root)),sha256=record['checkpoint_sha256'],timesteps=record['timesteps'])
    verify(root,plan)
    saved=bench.validate_saved_checks(root,[initial,practice,*[rec for arm in policies.values() for rec in arm.values()]])
    difference=sum(policies['switch-practice']['best']['checks'][key]['wins_per_requested']-
        policies['ppo-only']['best']['checks'][key]['wins_per_requested'] for key in ('fresh-original','fresh-reversed'))/2
    result=dict(status='smoke_completed' if plan['smoke'] else 'completed',selected_by_validation=choice,
        initial=initial,practice_only=practice,policies=policies,training=training,exports=exports,saved_evidence=saved,
        common_parent_sha256=plan['source_start_sha256'],endpoint_used_for_selection=False,fallback_retains_ties=True,
        comparison=dict(selected_development_mean_difference=difference,
            scope='One matched training seed; reused development compositions, familiar species and simple moves. Not untouched testing or unrestricted play.'))
    write_json(root/'selection.json',dict(selected=choice,endpoint_used_for_selection=False,fallback_retains_ties=True))
    write_json(root/'models/manifest.json',exports);write_json(root/'result.json',result);return result


def collect(root,plan):
    result=read_json(root/'result.json') if (root/'result.json').exists() else {}
    status=read_json(root/'status.json') if (root/'status.json').exists() else {'status':'prepared'}
    lines=['# Targeted switch practice and longer PPO','',
        'Both jobs continue the same validation-selected teacher-trained parent with reset PPO optimizer state. PPO only supplies the longer-training control. Switch practice first imitates legal Rules choices on new training-only trajectories, with capped move/forced/voluntary-switch loss weights 1/2/4 and 10% source-soft mirror rehearsal.',
        'Practice retains bounded stratified examples from 6,000 battles, half current-student and half Rules trajectories. The actor/shared encoder trains with critic heads frozen; shared encoder changes can still alter predicted values. Both PPO arms subsequently train the whole policy.',
        'PPO uses the unchanged 25% old / 65% balanced / 10% fixed-mirror mixture, fixed benches, reward and settings. Selection uses the unchanged 36-team balanced validation suite. The former fresh 24-team endpoint is now explicitly a reused development check; it never selects.', '',
        f"Status: {status['status']}. Additional PPO budget: {plan['settings']['timesteps']:,} steps per arm.",
        'Frozen scores are against Rules; draws/truncations are non-wins. Previous selected fallback wins ties. Source runs and the original 400k reference remain unchanged.','',
        '| Model | PPO steps | Validation | Seen original | Seen reversed | Development original | Development reversed | Mirror |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    initial=read_json(root/'initial.json');records=[('previous-selected',initial)]
    if result.get('practice_only'):records.append(('practice-only',result['practice_only']))
    records += [(name+'-'+kind,rec) for name,arm in result.get('policies',{}).items() for kind,rec in arm.items()]
    for name,record in records:
        rates=[f"{record['checks'][condition]['wins_per_requested']:.1%}" for condition in CONDITIONS]
        lines.append(f"| {name} | {record['timesteps']:,} | {record['balanced_validation']:.1%} | {' | '.join(rates)} |")
    if result:
        lines += ['',f"Validation-selected model: **{result['selected_by_validation']['name']}**, `models/selected.zip`.",
            f"Selected treatment minus control development mean: {100*result['comparison']['selected_development_mean_difference']:+.2f} percentage points.",result['comparison']['scope']]
    lines += ['', 'Practice category counts, training-pair split, agreement and loss metrics: practice/report.json. BC-only PPO step count excludes supervised gradient updates.',
        'Each trainer has evaluations.md, updates.jsonl and PPO CSV metrics. memory.csv records machine-wide available RAM and swap, including validation.',
        f"Logs: `tail -n 20 -F {root}/console/*.log`",'Memory: `watch -n 2 free -h`',
        f"Validation: `watch -n 5 'for f in {root}/runs/*/evaluations.md; do echo \"$f\"; cat \"$f\"; done'`",
        f"Teams: `python run_switch_practice.py --output-dir {shlex.quote(str(root))} --inspect-teams all --limit 3 --full-json`",
        f"Resume: `python run_switch_practice.py --output-dir {shlex.quote(str(root))} --continue --jobs 2`",
        f"Play: `python mgba/play_gui.py --rom \"$PWD/FireRed.gba\" --source \"$PWD/FireRed.sav\" --workers 2 --speed 1 --checkpoint {shlex.quote(str(root/'models/selected.zip'))}`",
        f"Permanent copy: `mkdir -p models && cp --no-clobber {shlex.quote(str(root/'models/selected.zip'))} models/{root.name}-selected.zip`",
        'Upload archive excludes model weights, bulk console logs and raw teacher arrays.','']
    (root/'report.md').write_text('\n'.join(lines));(root/'play-summary.md').write_text('\n'.join(lines))
    temporary=root/'switch-practice-results.tmp'
    with ZipFile(temporary,'w',ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative=path.relative_to(root)
            if path.is_file() and path.suffix in ('.json','.jsonl','.md','.csv') and relative.parts[0]!='console':archive.write(path,str(relative))
    temporary.replace(root/'switch-practice-results.zip')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=Path('.local/run/teacher-training-01'))
    parser.add_argument('--output-dir','--run-dir',dest='run_dir',type=Path,default=Path('.local/run/switch-practice-01'))
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2);parser.add_argument('--timesteps',type=int,default=None)
    parser.add_argument('--continue',dest='continuing',action='store_true');parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--inspect-teams',nargs='?',const='all',choices=('all','old','new','mirror','validation','fresh-test'))
    parser.add_argument('--limit',type=int,default=3);parser.add_argument('--full-json',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.run_dir.expanduser().resolve()
    if args.limit<0 or args.timesteps is not None and args.timesteps<1:parser.error('Limit must be nonnegative and timesteps positive')
    if args.continuing and args.timesteps is not None:parser.error('--continue keeps the saved budget; omit --timesteps')
    if args.inspect_teams is not None:
        plan=read_json(root/'plan.json');verify(root,plan);inspect(root,plan,args.inspect_teams,args.limit,args.full_json);return 0
    if args.timesteps is None:args.timesteps=1000000
    import torch
    torch.set_num_threads(1);root.parent.mkdir(parents=True,exist_ok=True)
    lock=(root.parent/('.'+root.name+'.switch-practice.lock')).open('a')
    plan=None;owned=False;stopped=threading.Event();monitor=None;previous=signal.signal(signal.SIGTERM,interrupt)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owned=True
        if args.continuing:plan=read_json(root/'plan.json');verify(root,plan)
        else:plan=freeze(root,args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {root}/console/*.log\nMemory: watch -n 2 free -h\nTeams: python run_switch_practice.py --output-dir {root} --inspect-teams all",flush=True)
        if args.prepare_only:write_json(root/'status.json',dict(status='prepared'));return 0
        monitor=threading.Thread(target=memory_monitor,args=(root,stopped),daemon=True);monitor.start()
        write_json(root/'status.json',dict(status='running',phase='switch-practice'));prepare(root,plan)
        write_json(root/'status.json',dict(status='running',phase='training'));collect(root,plan);launch(root,plan,args.jobs)
        write_json(root/'status.json',dict(status='running',phase='development-checks'));result=finish(root,plan)
        write_json(root/'status.json',dict(status=result['status']));return 0
    except (KeyboardInterrupt,Exception) as error:
        if plan is not None and owned:write_json(root/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',message=str(error)))
        print(f'Stopped: {error or "interrupted"}',file=sys.stderr,flush=True);return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        stopped.set()
        if monitor:monitor.join()
        if plan is not None and owned:
            collect(root,plan);print(f"Report: {root/'report.md'}\nUpload: {root/'switch-practice-results.zip'}",flush=True)
        signal.signal(signal.SIGTERM,previous);lock.close()


if __name__=='__main__':raise SystemExit(main())
