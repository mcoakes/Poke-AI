#!/usr/bin/env python3
"""Export the validation-selected model and check it on frozen fresh 3v3 teams.

No training or score-based model selection. Watch: tail -n 20 -F RUN/console/*.log
Memory: watch -n 2 free -h. Ctrl+C pauses; --continue retries incomplete checks.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
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
from run_parallel_training import checkpoint_steps, pump
from run_staged_team_lab import memory_monitor
from run_transfer_lab import inside, read_json
import run_replacement_diagnostic as diagnostic

EXPERIMENT = 'frozen-presentation-check'
STAGE = '04-type-matchups'
SEED = 261110
INTRO_SPECIES = {'Typhlosion', 'Blastoise', 'Sceptile', 'Arcanine', 'Vaporeon', 'Venusaur',
    'Raichu', 'Golem', 'Machamp', 'Alakazam', 'Dodrio', 'Lapras'}


def implementation_identity():
    paths = [Path(__file__), ROOT / 'run_replacement_diagnostic.py', ROOT / 'run_bench_order_ablation.py',
        *[ROOT / 'src/firered_rl' / name for name in ('broad_team_curriculum.py', 'replacement_diagnostics.py',
            'staged_team_lab.py', 'curriculum_checks.py', 'evaluation.py', 'fixed_team_lab.py', 'rule_opponent.py')]]
    return {str(path.relative_to(ROOT)): checksum(path) for path in paths}


def source_inputs(source, smoke=False):
    """Verify recorded selection, saved checks, pinned pools and exported weights."""
    from run_bench_order_ablation import validate_saved_checks
    source = Path(source).expanduser().resolve()
    plan, result = (read_json(source / name) for name in ('plan.json', 'result.json'))
    if plan.get('version') != 1 or plan.get('experiment') != 'bench-order-ablation':
        raise ValueError('Source must be a version-1 bench-order-ablation run')
    if plan.get('smoke') and not smoke: raise ValueError('A smoke source cannot supply a normal check')
    if result.get('status') not in ('completed', 'smoke_completed'):
        raise ValueError('Source must have completed training and frozen checks')
    records = [result['initial'], *[record for arm in result['policies'].values() for record in arm.values()]]
    evidence = validate_saved_checks(source, records)
    for relative, sha in plan['pins'].items():
        path = inside(source, relative)
        if not path.is_file() or checksum(path) != sha:
            raise ValueError(f'Source pinned input changed: {relative}')

    def pinned(relative):
        if relative not in plan['pins']: raise ValueError(f'Unpinned source input: {relative}')
        return inside(source, relative)

    teams = {group: [pinned(path) for path in plan['teams'][group]] for group in ('train', 'mirror')}
    for group, paths in teams.items():
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(f'Empty or duplicate source {group} teams')
        if set((source / 'inputs/teams' / group).glob('*.json')) != set(paths):
            raise ValueError(f'Source {group} team membership changed')
    suites = {name: pinned(path) for name, path in plan['suites'].items()}
    selected = result['selected_by_validation']
    arm = selected['name']
    if arm not in ('control', 'bench-shuffle'): raise ValueError('Unknown selected source arm')
    best = result['policies'][arm]['best']
    if any(selected.get(key) != best.get(key) for key in ('checkpoint', 'checkpoint_sha256')):
        raise ValueError('Recorded selection differs from the source validation-best model')
    scores = {}
    requested = len(read_json(suites['validation'])['cases'])
    for candidate_arm in ('control', 'bench-shuffle'):
        meta = read_json(source / 'runs' / candidate_arm / 'checkpoints/best.json')
        record = result['policies'][candidate_arm]['best']
        if (meta.get('requested') != requested or meta.get('checkpoint_sha256') != record['checkpoint_sha256']
                or meta.get('timesteps') != record['timesteps'] or not 0 <= meta['wins'] <= requested):
            raise ValueError('Source balanced validation evidence differs')
        scores[candidate_arm] = meta['wins'] / requested
    if arm != max(scores, key=scores.get) or abs(selected['validation_rate'] - scores[arm]) > 1e-12:
        raise ValueError('Recorded selection is not the highest balanced validation score (control retains ties)')
    models = {}
    for name, source_arm, kind in (('official', arm, 'best'), ('control-final', 'control', 'final'),
                                  ('bench-shuffle-final', 'bench-shuffle', 'final')):
        record = result['policies'][source_arm][kind]
        path = inside(source, record['checkpoint'])
        if checksum(path) != record['checkpoint_sha256']:
            raise ValueError(f'Source {name} checkpoint hash changed')
        steps = checkpoint_steps(path)
        if steps != record['timesteps']: raise ValueError(f'Source {name} timestep evidence differs')
        if kind == 'best':
            metadata = read_json(source / 'runs' / source_arm / 'checkpoints/best.json')
            requested = len(read_json(suites['validation'])['cases'])
            if (metadata.get('checkpoint_sha256') != record['checkpoint_sha256']
                    or metadata.get('timesteps') != steps or metadata.get('requested') != requested
                    or abs(metadata['wins'] / requested - selected['validation_rate']) > 1e-12):
                raise ValueError('Selected model differs from recorded balanced validation selection')
        elif steps != result['training'][source_arm]['ending_timesteps']:
            raise ValueError(f'Source {name} final timestep evidence differs')
        models[name] = dict(path=path, sha256=record['checkpoint_sha256'], timesteps=steps,
            source_arm=source_arm, source_kind=kind, source_checkpoint=record['checkpoint'])
    return dict(source=source, plan=plan, result=result, reward=pinned('inputs/reward.json'),
        teams=teams, suites=suites, models=models, selected=selected, saved_evidence=evidence)


def exclusions(source):
    from firered_rl.broad_team_curriculum import composition_signature, semantic_signature
    entries = []
    for group, paths in source['teams'].items():
        entries.extend((f'{group}:{path.name}', read_json(path)) for path in paths)
    for name, path in source['suites'].items():
        suite = read_json(path)
        entries.extend((f'suite:{name}:{key}', asset['team']) for key, asset in suite['assets'].items())
    return dict(compositions=sorted({composition_signature(team) for _, team in entries}),
        semantics=sorted({semantic_signature(team) for _, team in entries}),
        sources=[label for label, _ in entries], scope='All pinned source train/rehearsal teams and all embedded assets in every source suite')


def fresh_teams(config, excluded, count, seed=SEED):
    """Same stage sampler, conditioned on compositions absent from all prior pools."""
    from firered_rl.broad_team_curriculum import (_catalog, _validate_config, _pokemon, _validate_mon,
        composition_signature, semantic_signature)
    catalog, species, moves, items = _catalog(ROOT, config)
    _validate_config(config, catalog, species, moves, items)
    stage = next(item for item in config['stages'] if item['id'] == STAGE)
    if (stage['team_size'] != 3 or stage['species_pool'] != 'intro' or stage['build_mode'] != 'simple'
            or stage['moves']['count'] != 2 or stage['moves']['attacks'] != 2
            or not stage['moves'].get('simple_stab_normal') or stage.get('curated_fraction', 0) != 0):
        raise ValueError('Fresh check requires the original simple 3v3 two-attack stage recipe')
    pool = config['species_pools'][stage['species_pool']]
    if len(pool) != 12 or set(pool) != INTRO_SPECIES:
        raise ValueError('Fresh check requires the original 12-species pool')
    if (config['builds']['simple'] != dict(natures=['Hardy'], items=['None'],
            ev_spreads=[dict.fromkeys(('hp', 'atk', 'def', 'spa', 'spd', 'spe'), 0)])
            or stage['moves'].get('min_stab_power', 40) != 40):
        raise ValueError('Fresh check requires the original simple build and attack recipe')
    old_comps, old_semantics = set(excluded['compositions']), set(excluded['semantics'])
    seen_comps, seen_semantics, result = set(), set(), []
    rng = random.Random(seed); attempts = 0
    while len(result) < count and attempts < 20000:
        attempts += 1
        party = [_validate_mon(_pokemon(name, stage, config, catalog, moves, items, rng, {}),
                              catalog, species, moves, items) for name in rng.sample(pool, stage['team_size'])]
        team = dict(name=f'presentation-fresh-{len(result)+1:04}', party=party)
        comp, semantic = composition_signature(team), semantic_signature(team)
        if comp in old_comps or comp in seen_comps or semantic in old_semantics or semantic in seen_semantics:
            continue
        result.append(team); seen_comps.add(comp); seen_semantics.add(semantic)
    if len(result) != count: raise ValueError('Insufficient unused team compositions for the frozen fresh pool')
    return result, dict(seed=seed, stage=STAGE, attempts=attempts, rejected=attempts-count,
        teams=count, compositions=sorted(seen_comps), semantics=sorted(seen_semantics), species_pool=pool,
        scope='Familiar species and the same simple moves/build recipe; novel compositions, not novel species or unrestricted competitive play')


def validate_source_recipe(source, config):
    """Check the frozen non-mirror training pool actually belongs to this task."""
    from firered_rl.broad_team_curriculum import _catalog, _validate_mon, _move_options
    catalog, species, moves, items = _catalog(ROOT, config)
    stage = next(item for item in config['stages'] if item['id'] == STAGE)
    stats = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
    for path in source['teams']['train']:
        team = read_json(path)
        if len(team['party']) != 3: raise ValueError('Source training recipe is not simple 3v3')
        for raw in team['party']:
            mon = _validate_mon(raw, catalog, species, moves, items)
            if (mon['species'] not in INTRO_SPECIES or mon.get('level') != 50 or mon.get('nature') != 'Hardy'
                    or mon.get('held_item') != 'None' or mon.get('evs') != dict.fromkeys(stats, 0)
                    or mon.get('ivs') != dict.fromkeys(stats, 31) or len(mon['moves']) != 2):
                raise ValueError('Source training build differs from the fresh 3v3 recipe')
            attacks, _, stab = _move_options(mon['species'], stage, config, catalog, moves)
            selected = mon['moves']
            if (len(set(selected)) != 2 or any(move not in attacks for move in selected)
                    or not any(move in stab and catalog['moves'][other]['type'] == 'Normal'
                        for move, other in (selected, selected[::-1]))):
                raise ValueError('Source training moves differ from the fresh STAB/Normal recipe')


def make_checks():
    return [dict(label=f'{name}-{order}', checkpoint='official' if name == 'official' else None,
        baseline=None if name == 'official' else name, pool='fresh', order=order, assisted=False,
        suite=f'fresh-{order}') for name in ('official', 'greedy', 'rules') for order in ('original', 'reversed')]


def freeze(root, args):
    from firered_rl.curriculum_checks import coverage_suite
    from firered_rl.replacement_diagnostics import transform_suite
    from firered_rl.showdown_env import validate_showdown
    source = source_inputs(args.source_run, args.smoke)
    if root == source['source'] or root.is_relative_to(source['source']):
        raise ValueError('Output must be outside the source experiment')
    validate_showdown(ROOT)
    root.mkdir(parents=True, exist_ok=False)
    for relative in ('inputs/suites', 'inputs/teams/fresh', 'models', 'demo-teams', 'console', 'locks'):
        (root / relative).mkdir(parents=True)
    shutil.copyfile(source['reward'], root / 'inputs/reward.json')
    models = {}
    for name, info in source['models'].items():
        relative = f"models/{'selected' if name == 'official' else name}.zip"
        shutil.copyfile(info['path'], root / relative)
        models[name] = {key: value for key, value in info.items() if key != 'path'}
        models[name]['checkpoint'] = relative
    # This declaration precedes generation and all fresh scores; never reselect below.
    declaration = dict(selected_by_validation=source['selected'], official=models['official'],
        selection_scope='Recorded source balanced validation only', fresh_scores_can_change_selection=False)
    write_json(root / 'inputs/official-selection.json', declaration)
    write_json(root / 'models/manifest.json', dict(models=models, declaration=declaration))
    teams = dict(demo=[], fresh=[])
    for index, path in enumerate(source['teams']['train']):
        relative = f'demo-teams/team-{index+1:04}.json'
        shutil.copyfile(path, root / relative); teams['demo'].append(relative)
    config_path = getattr(args, 'config', ROOT / 'config/experiments/team-curriculum.json')
    config = read_json(config_path)
    validate_source_recipe(source, config)
    write_json(root / 'inputs/team-recipe.json', config)
    excluded = exclusions(source)
    write_json(root / 'inputs/excluded-teams.json', excluded)
    generated, generation = fresh_teams(config, excluded, 4 if args.smoke else 64)
    for index, team in enumerate(generated):
        relative = f'inputs/teams/fresh/team-{index+1:04}.json'
        write_json(root / relative, team); teams['fresh'].append(relative)
    generation.update(config_path=str(Path(config_path).resolve()), config_sha256=checksum(config_path),
        catalog=config['catalog'], catalog_sha256=checksum(ROOT / config['catalog']))
    write_json(root / 'inputs/generation.json', generation)
    original = coverage_suite(ROOT, [root / path for path in teams['fresh']], 8 if args.smoke else 1000,
        SEED + 1, max_steps=100)
    suites = {}
    for order, suite in (('original', original), ('reversed', transform_suite(original))):
        relative = f'inputs/suites/fresh-{order}.json'
        write_json(root / relative, suite); suites['fresh-' + order] = relative
    write_json(root / 'inputs/source-plan.json', source['plan'])
    write_json(root / 'inputs/source-result.json', source['result'])
    write_json(root / 'inputs/source-evidence.json', dict(source=str(source['source']),
        source_plan_sha256=checksum(source['source'] / 'plan.json'), source_result_sha256=checksum(source['source'] / 'result.json'),
        source_pins=source['plan']['pins'], saved_evidence=source['saved_evidence']))
    checks = make_checks()
    settings = dict(seed=SEED, fresh_teams=len(generated), battles_per_order=len(original['cases']),
        max_jobs=2, no_training=True, no_reselection=True)
    configuration = dict(models=models, teams=teams, suites=suites, official_selection=declaration, settings=settings)
    write_json(root / 'inputs/configuration.json', configuration)
    for check in checks: (root / 'console' / (check['label'] + '.log')).touch()
    plan = dict(version=1, experiment=EXPERIMENT, smoke=bool(args.smoke), checks=checks,
        no_training=True, no_automatic_promotion=True, implementation=implementation_identity(), **configuration,
        pins={str(path.relative_to(root)): checksum(path) for directory in ('inputs', 'models', 'demo-teams')
            for path in (root / directory).rglob('*') if path.is_file()})
    write_json(root / 'plan.json', plan)
    return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != EXPERIMENT:
        raise ValueError('Unsupported presentation check plan')
    if (plan.get('checks') != make_checks() or plan.get('no_training') is not True
            or plan.get('no_automatic_promotion') is not True): raise ValueError('Frozen check conditions changed')
    if plan.get('implementation') != implementation_identity():
        raise ValueError('Implementation changed; use a new output directory')
    paths = {path.resolve() for directory in ('inputs', 'models', 'demo-teams')
        for path in (root / directory).rglob('*') if path.is_file()}
    if paths != {inside(root, name) for name in plan['pins']}: raise ValueError('Pinned membership changed')
    for name, sha in plan['pins'].items():
        if checksum(inside(root, name)) != sha: raise ValueError(f'Pinned input changed: {name}')
    configuration = read_json(root / 'inputs/configuration.json')
    if any(plan.get(key) != value for key, value in configuration.items()):
        raise ValueError('Saved selection, models, teams, settings or suites changed')
    for name, model in plan['models'].items():
        if (plan['pins'].get(model['checkpoint']) != model['sha256']
                or checkpoint_steps(inside(root, model['checkpoint'])) != model['timesteps']):
            raise ValueError(f'Exported {name} metadata differs')
    generation = read_json(root / 'inputs/generation.json')
    if (checksum(Path(generation['config_path'])) != generation['config_sha256']
            or checksum(inside(ROOT, generation['catalog'])) != generation['catalog_sha256']):
        raise ValueError('Team recipe or catalog implementation changed')
    if read_json(root / 'inputs/official-selection.json') != plan['official_selection']:
        raise ValueError('Official selection changed')


def progress(root, plan, active=()):
    results = {check['label']: value for check in plan['checks']
        if (value := diagnostic.cached_check(root, plan, check)) is not None}
    write_json(root / 'progress.json', dict(completed=len(results), requested=len(plan['checks']), active=list(active),
        checks={label: dict(wins=value['candidate_wins'], requested=value['requested_battles'],
            win_rate=value['wins_per_requested']) for label, value in results.items()}))
    return results


def launch(root, plan, jobs):
    if jobs not in (1, 2): raise ValueError('Only one or two evaluation jobs are supported')
    pending = [check for check in plan['checks'] if diagnostic.cached_check(root, plan, check) is None]
    active = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        while pending or active:
            changed = False
            while pending and len(active) < jobs:
                check = pending.pop(0)
                command = [sys.executable, str(Path(__file__).resolve()), '--output-dir', str(root), '--worker', check['label']]
                print(f"Starting {check['label']}; log: {root / 'console' / (check['label'] + '.log')}", flush=True)
                proc = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                reader = threading.Thread(target=pump, args=(proc, root / 'console' / (check['label'] + '.log')))
                reader.start(); active.append((check, proc, reader)); changed = True
            for row in list(active):
                check, proc, reader = row
                if proc.poll() is None: continue
                reader.join(); active.remove(row); changed = True
                if proc.returncode or diagnostic.cached_check(root, plan, check) is None:
                    raise RuntimeError(f"{check['label']} failed or saved incomplete evidence; inspect its console log")
                print(f"Completed {check['label']}", flush=True)
            if changed: progress(root, plan, [item[0]['label'] for item in active])
            if active: time.sleep(.25)
    finally:
        for _, proc, _ in active:
            if proc.poll() is None: proc.send_signal(signal.SIGINT)
        for _, proc, reader in active: proc.wait(); reader.join()


def finish(root, plan):
    verify(root, plan)
    results = {}
    for check in plan['checks']:
        result = diagnostic.cached_check(root, plan, check)
        if result is None: raise RuntimeError(f"Saved check incomplete: {check['label']}; --continue preserves and retries it")
        results[check['label']] = result
    # Independent reread after collection, including every episode and bounded audit.
    from run_bench_order_ablation import validate_saved_checks
    saved = validate_saved_checks(root, [dict(checks=results)])
    result = dict(status='smoke_completed' if plan['smoke'] else 'completed', no_training=True,
        no_reselection=True, official_selection=plan['official_selection'], checks=results, saved_evidence=saved,
        scope='Fresh compositions of familiar species under the same simple 3v3 recipe; one validation-selected policy, no training replication')
    write_json(root / 'result.json', result)
    return result


def collect(root, plan):
    results = progress(root, plan)
    selected = plan['models']['official']
    lines = ['# Frozen presentation check', '',
        f"Official model: **{selected['source_arm']} validation-best**, {selected['timesteps']:,} timesteps; `models/selected.zip`.",
        'Selection was declared before fresh-team generation. Fresh scores never change the selected model. No training occurred.', '',
        f"Fresh pool: {plan['settings']['fresh_teams']} novel compositions of the familiar 12 species; simple 3v3 teams with two attacks each.",
        'All compositions and semantic teams in the pinned source experiment training, validation, development-heldout and rehearsal pools were excluded.',
        'All opponents are Rules. Both orders keep the initial lead and opponent unchanged; only the candidate bench is reversed.',
        'The same matchups and seeds are paired across candidates, orders and candidate seats. Draws and truncations count as non-wins.', '',
        '| Frozen candidate | Original wins/requested | Reversed wins/requested | Mean across orders |',
        '|---|---:|---:|---:|']
    for name, title in (('official', 'Selected PPO'), ('greedy', 'Greedy damage baseline'), ('rules', 'Rules baseline')):
        values = [results.get(f'{name}-{order}') for order in ('original', 'reversed')]
        cells = [f"{value['candidate_wins']}/{value['requested_battles']} ({value['wins_per_requested']:.1%})" if value else 'pending' for value in values]
        mean = f"{sum(value['wins_per_requested'] for value in values)/2:.1%}" if all(values) else 'pending'
        lines.append(f"| {title} | {' | '.join(cells)} | {mean} |")
    lines += ['', f"Complete saved checks: {len(results)}/{len(plan['checks'])}; truncated battles: {sum(value['truncated'] for value in results.values())}.",
        'These scores measure this finite fresh pool, not unrestricted play or unseen-species generalization. A single selected policy does not measure training-seed variability.', '',
        '## Exported models', '', '| File | Source | Timesteps | SHA-256 |', '|---|---|---:|---|']
    for model in plan['models'].values():
        lines.append(f"| `{model['checkpoint']}` | {model['source_arm']} {model['source_kind']} | {model['timesteps']:,} | `{model['sha256']}` |")
    lines += ['', 'The final checkpoints are exported as named alternatives only; they are not evaluated or selected by this fresh check.',
        f"Familiar GUI teams: `demo-teams/` ({len(plan['teams']['demo'])} source training teams). These are not the fresh evaluation pool.", '',
        f"Logs: `tail -n 20 -F {root}/console/*.log`", 'Memory: `watch -n 2 free -h`',
        f"Teams: `python run_presentation_check.py --output-dir {shlex.quote(str(root))} --inspect-teams fresh --limit 3 --full-json`", '']
    (root / 'report.md').write_text('\n'.join(lines))
    (root / 'presentation-summary.md').write_text('\n'.join(lines[:lines.index('## Exported models')]))
    with ZipFile(root / 'presentation-results.zip', 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.md', '.csv') and relative.parts[0] not in ('console', 'locks'):
                archive.write(path, str(relative))


def inspect(root, plan, group, limit, full):
    for name in ('fresh', 'demo') if group == 'all' else (group,):
        paths = plan['teams'][name]
        print(f'\n{name}: {len(paths)} teams')
        for relative in paths[:limit]:
            team = read_json(root / relative)
            print(f"{root / relative}: {team.get('name', '')}")
            if full: print(json.dumps(team, indent=2))
            else:
                for mon in team['party']: print(f"  {mon['species']}: {', '.join(mon['moves'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('.local/run/bench-order-ablation-01'))
    parser.add_argument('--output-dir', type=Path, default=Path('.local/run/presentation-check-01'))
    parser.add_argument('--config', type=Path, default=ROOT / 'config/experiments/team-curriculum.json')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-teams', nargs='?', const='all', choices=('all', 'fresh', 'demo'))
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--full-json', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Four fresh teams, eight cases per check; not scientific evidence')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.limit < 0: parser.error('--limit cannot be negative')
    import torch
    torch.set_num_threads(1)
    if args.worker:
        plan = read_json(root / 'plan.json'); verify(root, plan)
        matches = [check for check in plan['checks'] if check['label'] == args.worker]
        if len(matches) != 1: parser.error('Unknown frozen check worker label')
        lock = (root / 'locks' / (args.worker + '.lock')).open('a')
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            diagnostic.run_check(root, plan, matches[0]); return 0
        finally: signal.signal(signal.SIGTERM, previous); lock.close()
    if args.inspect_teams is not None:
        plan = read_json(root / 'plan.json'); verify(root, plan)
        inspect(root, plan, args.inspect_teams, args.limit, args.full_json); return 0
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = (root.parent / ('.' + root.name + '.presentation.lock')).open('a')
    plan = None; owned = False; stopped = threading.Event(); monitor = None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); owned = True
        if args.continuing:
            plan = read_json(root / 'plan.json'); verify(root, plan)
        else: plan = freeze(root, args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {shlex.quote(str(root / 'console'))}/*.log\nMemory: watch -n 2 free -h\nTeams: python run_presentation_check.py --output-dir {shlex.quote(str(root))} --inspect-teams fresh --full-json", flush=True)
        if args.prepare_only:
            write_json(root / 'status.json', dict(status='prepared', no_training=True)); return 0
        monitor = threading.Thread(target=memory_monitor, args=(root, stopped), daemon=True); monitor.start()
        write_json(root / 'status.json', dict(status='running', no_training=True))
        launch(root, plan, args.jobs)
        result = finish(root, plan)
        write_json(root / 'status.json', dict(status=result['status'], no_training=True)); return 0
    except (KeyboardInterrupt, Exception) as error:
        if plan is not None and owned:
            write_json(root / 'status.json', dict(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', no_training=True, message=str(error)))
        print(f'Stopped: {error or "interrupted"}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        stopped.set()
        if monitor: monitor.join()
        if plan is not None and owned:
            collect(root, plan)
            print(f"Report: {root / 'report.md'}\nUpload: {root / 'presentation-results.zip'}", flush=True)
        signal.signal(signal.SIGTERM, previous); lock.close()


if __name__ == '__main__': raise SystemExit(main())
