#!/usr/bin/env python3
"""Frozen replacement/order diagnostic. No learning and no checkpoint promotion.

Watch jobs: tail -n 20 -F RUN/console/*.log
Watch memory: watch -n 2 free -h. Ctrl+C pauses; --continue reuses complete checks.
"""
import argparse
from collections import Counter
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
from run_parallel_training import checkpoint_steps, pump
from run_staged_team_lab import memory_monitor
from run_transfer_lab import inside, read_json

EXPERIMENT = 'frozen-replacement-diagnostic'
REPLACEMENT_COUNTERS = ('replacement_eligible', 'replacement_actual_lowest_legal',
    'replacement_base_lowest_legal', 'replacement_base_rules_disagreements',
    'replacement_override_eligible', 'replacement_overrides_changed')


def implementation_identity():
    paths = [Path(__file__), *[ROOT / 'src/firered_rl' / name for name in (
        'replacement_diagnostics.py', 'staged_team_lab.py', 'curriculum_checks.py',
        'evaluation.py', 'fixed_team_lab.py')]]
    return {str(path.relative_to(ROOT)): checksum(path) for path in paths}


def source_inputs(source, smoke=False):
    """Read only completed, pinned source evidence and the two named matchup models."""
    source = Path(source).expanduser().resolve()
    plan = read_json(source / 'plan.json')
    result = read_json(source / 'result.json')
    if plan.get('version') != 1 or plan.get('experiment') != 'matchup-feature-ablation':
        raise ValueError('Source must be a version-1 matchup-feature-ablation run')
    if plan.get('smoke') and not smoke:
        raise ValueError('A smoke source cannot supply a normal diagnostic')
    if result.get('status') not in ('completed', 'smoke_completed'):
        raise ValueError('Source experiment must have completed frozen checks')
    from run_matchup_ablation import verify_saved_checks
    verify_saved_checks(source, result)
    pins = {}

    def pinned(relative):
        path = inside(source, relative)
        expected = plan['pins'].get(relative)
        if not expected or not path.is_file() or checksum(path) != expected:
            raise ValueError(f'Source pinned input changed: {relative}')
        pins[relative] = expected
        return path

    reward = pinned('inputs/reward.json')
    teams = {name: [pinned(path) for path in plan['teams'][name]] for name in ('train', 'mirror')}
    for name, paths in teams.items():
        if not paths or len(set(paths)) != len(paths) or set((source / 'inputs/teams' / name).glob('*.json')) != set(paths):
            raise ValueError(f'Source {name} team membership changed')
    suites = {name: pinned(plan['suites'][name]) for name in ('test', 'heldout', 'validation', 'mirror')}
    models = {}
    best_meta = read_json(source / 'runs/matchup/checkpoints/best.json')
    for kind in ('best', 'final'):
        record = result['policies']['matchup'][kind]
        path = inside(source, record['checkpoint'])
        if checksum(path) != record['checkpoint_sha256']:
            raise ValueError(f'Source matchup {kind} checkpoint hash changed')
        steps = checkpoint_steps(path)
        expected_steps = best_meta['timesteps'] if kind == 'best' else result['training']['matchup']['ending_timesteps']
        if steps != expected_steps:
            raise ValueError(f'Source matchup {kind} timestep evidence differs')
        if kind == 'best' and best_meta['checkpoint_sha256'] != record['checkpoint_sha256']:
            raise ValueError('Source matchup best checkpoint differs from validation selection')
        models[kind] = dict(path=path, sha256=record['checkpoint_sha256'], timesteps=steps)
    selected = result['selected_by_validation']
    if selected.get('name') == 'matchup' and selected.get('checkpoint_sha256') != models['best']['sha256']:
        raise ValueError('Source selected matchup hash differs from matchup best')
    return dict(source=source, reward=reward, teams=teams, suites=suites, models=models,
        source_pins=pins, source_selected=selected, source_plan_sha256=checksum(source / 'plan.json'),
        source_result_sha256=checksum(source / 'result.json'), source_smoke=bool(plan.get('smoke')))


def make_checks():
    checks = []
    for kind in ('best', 'final'):
        for pool in ('seen', 'heldout'):
            for order in ('original', 'reversed'):
                for assisted in (False, True):
                    mode = 'rules-forced' if assisted else 'ppo'
                    checks.append(dict(label=f'{kind}-{pool}-{order}-{mode}', checkpoint=kind,
                        baseline=None, pool=pool, order=order, assisted=assisted,
                        suite=f'{pool}-{order}'))
    for baseline in ('greedy', 'rules'):
        for order in ('original', 'reversed'):
            checks.append(dict(label=f'baseline-{baseline}-{order}', checkpoint=None,
                baseline=baseline, pool='heldout', order=order, assisted=False,
                suite=f'heldout-{order}'))
    return checks


def freeze(root, args):
    from firered_rl.evaluation import validate_suite
    from firered_rl.replacement_diagnostics import transform_suite
    from firered_rl.showdown_env import validate_showdown
    source = source_inputs(args.source_run, args.smoke)
    validate_showdown(ROOT)
    if root == source['source'] or root.is_relative_to(source['source']):
        raise ValueError('Output must be outside the source experiment directory')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs'; inputs.mkdir(); (inputs / 'suites').mkdir()
    (inputs / 'models').mkdir(); (root / 'console').mkdir(); (root / 'locks').mkdir()
    shutil.copyfile(source['reward'], inputs / 'reward.json')
    models = {}
    for kind, info in source['models'].items():
        relative = f'inputs/models/matchup-{kind}.zip'
        shutil.copyfile(info['path'], root / relative)
        models[kind] = dict(checkpoint=relative, sha256=info['sha256'], timesteps=info['timesteps'])
    teams = {}
    for name, paths in source['teams'].items():
        directory = inputs / 'teams' / name; directory.mkdir(parents=True)
        teams[name] = []
        for index, path in enumerate(paths):
            target = directory / f'team-{index+1:04}.json'; shutil.copyfile(path, target)
            teams[name].append(str(target.relative_to(root)))
    suites = {}
    for name, path in source['suites'].items():
        data = read_json(path)
        if args.smoke:
            data['cases'] = data['cases'][:8]
            data['diagnostic'] = dict(data.get('diagnostic', {}), replacement_smoke=True, actual_battles=len(data['cases']))
        validate_suite(data)
        names = [(('seen' if name == 'test' else name) + '-original', data)] if name in ('test', 'heldout') else [(name, data)]
        if name in ('test', 'heldout'):
            names.append((('seen' if name == 'test' else name) + '-reversed', transform_suite(data)))
        for label, suite in names:
            target = inputs / 'suites' / f'{label}.json'; write_json(target, suite)
            suites[label] = str(target.relative_to(root))
    evidence = {key: value for key, value in source.items() if key not in ('reward', 'teams', 'suites', 'models')}
    evidence['source'] = str(evidence['source'])
    evidence.update(no_training=True, source_models={kind: {**info, 'path': str(info['path'])}
        for kind, info in source['models'].items()},
        note='Matchup best was selected by source validation. Final is a fixed-budget diagnostic, never promoted by these checks.')
    write_json(inputs / 'source-evidence.json', evidence)
    checks = make_checks()
    for check in checks: (root / 'console' / (check['label'] + '.log')).touch()
    write_json(inputs / 'configuration.json', dict(models=models, teams=teams, suites=suites,
        source_selected=source['source_selected']))
    plan = dict(version=1, experiment=EXPERIMENT, smoke=bool(args.smoke), models=models,
        teams=teams, suites=suites, checks=checks, implementation=implementation_identity(),
        no_training=True, no_automatic_promotion=True, source_selected=source['source_selected'],
        assistance='Rules chooses only forced replacements with at least two legal switch choices. Hybrid results are not learned PPO scores.',
        order='Only the candidate bench is reversed; initial lead, opponent team, moves and case seeds stay fixed.',
        pins={str(path.relative_to(root)): checksum(path) for path in inputs.rglob('*') if path.is_file()})
    write_json(root / 'plan.json', plan)
    return plan


def verify(root, plan):
    if plan.get('version') != 1 or plan.get('experiment') != EXPERIMENT:
        raise ValueError('Unsupported replacement diagnostic plan')
    if plan.get('checks') != make_checks() or plan.get('no_training') is not True or plan.get('no_automatic_promotion') is not True:
        raise ValueError('Saved diagnostic conditions changed')
    if plan.get('implementation') != implementation_identity():
        raise ValueError('Diagnostic implementation changed; use a new output directory')
    expected = {inside(root, name) for name in plan['pins']}
    if expected != {path.resolve() for path in (root / 'inputs').rglob('*') if path.is_file()}:
        raise ValueError('Pinned input membership changed')
    for name, value in plan['pins'].items():
        if checksum(inside(root, name)) != value:
            raise ValueError(f'Pinned input changed: {name}')
    configuration = read_json(root / 'inputs/configuration.json')
    if any(plan.get(key) != configuration[key] for key in ('models', 'teams', 'suites', 'source_selected')):
        raise ValueError('Saved model, suite, team or source selection configuration changed')
    for kind, model in plan['models'].items():
        path = inside(root, model['checkpoint'])
        if plan['pins'].get(model['checkpoint']) != model['sha256'] or checkpoint_steps(path) != model['timesteps']:
            raise ValueError(f'Pinned {kind} model metadata differs')
    for path in [*plan['suites'].values(), *[p for group in plan['teams'].values() for p in group]]:
        if path not in plan['pins']:
            raise ValueError(f'Unpinned suite or team: {path}')


def check_context(root, plan, check):
    from firered_rl.curriculum_checks import _baseline_identity
    from firered_rl.evaluation import digest, validate_suite
    from firered_rl.rule_opponent import RuleBasedOpponent
    suite_path = inside(root, plan['suites'][check['suite']])
    suite = validate_suite(read_json(suite_path))
    rules = RuleBasedOpponent(ROOT).identity
    if suite.get('rule_opponent') not in (None, rules):
        raise ValueError('Frozen source suite Rules identity differs from current Rules')
    if 'opponents' in suite or any('opponent_id' in case for case in suite['cases']):
        raise ValueError('Replacement diagnostic requires one Rules opponent')
    suite = copy.deepcopy(suite)
    suite['rule_opponent'] = rules
    suite['check_opponent'] = rules
    suite['diagnostic'] = dict(suite.get('diagnostic', {}), replacement_assistance=check['assisted'],
        candidate_kind='hybrid-ppo-rules-forced' if check['assisted'] else 'frozen-ppo' if check['checkpoint'] else check['baseline'],
        no_training=True, opponent_kind='rules')
    validate_suite(suite)
    model = plan['models'][check['checkpoint']] if check['checkpoint'] else None
    candidate = (dict(kind='hybrid-ppo-rules-forced' if check['assisted'] else 'checkpoint',
        sha256=model['sha256'], forced_replacement_assistance=check['assisted'])
        if model else _baseline_identity(ROOT, check['baseline']))
    identity = dict(candidate_identity=candidate, opponent_identity=rules,
        input_suite_sha256=checksum(suite_path), effective_suite_sha256=digest(suite),
        reward_sha256=plan['pins']['inputs/reward.json'], implementation=plan['implementation'],
        replacement_assistance=check['assisted'], no_training=True)
    return suite, identity


def complete_audit(directory, summary):
    """A summary alone does not prove that the requested diagnostic was saved."""
    from firered_rl.curriculum_checks import _complete_episode_log
    from firered_rl.evaluation import digest
    try:
        if not _complete_episode_log(directory, summary): return False
        suite = read_json(directory / 'suite.json')
        if (digest(suite) != summary.get('effective_suite_sha256')
                or digest(suite) != summary.get('suite_sha256')
                or len(suite['cases']) != summary['requested_battles']): return False
        scheduled = {case['episode']: case for case in suite['cases']}
        rows = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines() if line]
        if any(any(row[key] != scheduled[row['episode']][key] for key in
                   ('episode', 'matchup', 'assets', 'seed', 'candidate_player')) for row in rows): return False
        audit = read_json(directory / 'decision-audit.json')
        totals = audit['totals']; episodes = audit['by_episode']
        metadata = audit['replacement_diagnostic']
        if (metadata.get('version') != 1 or metadata.get('all_battle_counts') is not True
                or metadata.get('assisted') != summary.get('replacement_assistance')): return False
        traces = [json.loads(line) for line in (directory / 'decision-traces.jsonl').read_text().splitlines() if line]
        limits = audit['trace_limits']
        if len(traces) != audit['traced_decisions']: return False
        if any(not (1 <= row['episode'] <= min(limits['cases'], summary['requested_battles'])
                    and 1 <= row['decision'] <= limits['decisions_per_case']) for row in traces): return False
        if len({(row['episode'], row['decision']) for row in traces}) != len(traces): return False
        if set(episodes) != {str(i) for i in range(1, summary['requested_battles'] + 1)}: return False
        if any(key not in totals for key in REPLACEMENT_COUNTERS): return False
        if any(type(totals[key]) is not int or totals[key] < 0 for key in REPLACEMENT_COUNTERS): return False
        eligible = totals['replacement_eligible']
        if any(totals[key] > eligible for key in REPLACEMENT_COUNTERS[1:]): return False
        if totals['replacement_override_eligible'] != (eligible if summary.get('replacement_assistance') else 0): return False
        if totals['replacement_overrides_changed'] > totals['replacement_override_eligible']: return False
        combined = Counter()
        for values in episodes.values(): combined.update(values)
        if any(combined[key] != value for key, value in totals.items()): return False
        if totals != summary.get('audit_totals'): return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def cached_check(root, plan, check):
    _, identity = check_context(root, plan, check)
    base = root / 'checks' / check['label']
    for directory in [base, *sorted(base.parent.glob(base.name + '-retry-*'))]:
        if not directory.resolve().is_relative_to((root / 'checks').resolve()): continue
        try:
            result = read_json(directory / 'summary.json')
            if (result.get('status') == 'completed' and all(result.get(k) == v for k, v in identity.items())
                    and complete_audit(directory, result)):
                result['result_directory'] = str(directory.relative_to(root))
                return result
        except (OSError, ValueError, KeyError, TypeError): pass
    return None


def run_check(root, plan, check):
    from firered_rl.curriculum_checks import _load_model
    from firered_rl.replacement_diagnostics import make_candidate, ReplacementAudit
    from firered_rl.shared_move_policy import policy_architecture
    from firered_rl.staged_team_lab import baseline_policy, score_stage
    previous = cached_check(root, plan, check)
    if previous is not None: return previous
    suite, identity = check_context(root, plan, check)
    if check['checkpoint']:
        model_info = plan['models'][check['checkpoint']]
        model = _load_model(inside(root, model_info['checkpoint']))
        if policy_architecture(model) not in ('matchup-moves', 'matchup-switches') or not model.policy.features_extractor.relational_enabled:
            raise ValueError('Expected a matchup policy with relational features enabled')
        candidate = make_candidate(model, check['assisted'], ROOT)
    else:
        model = None
        candidate = baseline_policy(ROOT, check['baseline'])
    base = root / 'checks' / check['label']
    directory = base if not base.exists() else base.with_name(base.name + f'-retry-{time.time_ns()}')
    print(f"{check['label']}: frozen {'HYBRID' if check['assisted'] else 'unassisted'} check; {len(suite['cases'])} battles; no training", flush=True)
    result = score_stage(ROOT, directory, suite, candidate, root / 'inputs/reward.json',
        pairs=2, trace_cases=16, trace_decisions=24, audit_factory=ReplacementAudit)
    if model is not None and model.num_timesteps != model_info['timesteps']:
        raise RuntimeError('Frozen checkpoint timestep count changed')
    result.update(identity, result_directory=str(directory.relative_to(root)),
        condition=check, checkpoint=(plan['models'][check['checkpoint']]['checkpoint'] if check['checkpoint'] else None),
        hybrid=check['assisted'], wins_per_requested=result['candidate_wins'] / result['requested_battles'])
    write_json(directory / 'summary.json', result)
    if not complete_audit(directory, result):
        raise RuntimeError(f"Saved diagnostic evidence incomplete: {directory}; --continue will preserve and retry it")
    return result


def progress(root, plan, active=()):
    results = {check['label']: result for check in plan['checks'] if (result := cached_check(root, plan, check)) is not None}
    write_json(root / 'progress.json', dict(completed=len(results), requested=len(plan['checks']),
        active=list(active), checks={label: dict(wins=r['candidate_wins'], requested=r['requested_battles'],
            win_rate=r['wins_per_requested'], hybrid=r['replacement_assistance']) for label, r in results.items()}))
    return results


def launch(root, plan, jobs):
    if jobs not in (1, 2): raise ValueError('Only one or two concurrent evaluation jobs are allowed')
    pending = [check for check in plan['checks'] if cached_check(root, plan, check) is None]
    active = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        while pending or active:
            started = False
            while pending and len(active) < jobs:
                check = pending.pop(0)
                command = [sys.executable, str(Path(__file__).resolve()), '--output-dir', str(root), '--worker', check['label']]
                print(f"Starting {check['label']}; log: {root / 'console' / (check['label'] + '.log')}", flush=True)
                proc = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                reader = threading.Thread(target=pump, args=(proc, root / 'console' / (check['label'] + '.log')))
                reader.start(); active.append((check, proc, reader))
                started = True
            changed = False
            for row in list(active):
                check, proc, reader = row
                code = proc.poll()
                if code is None: continue
                reader.join(); active.remove(row); changed = True
                if code or cached_check(root, plan, check) is None:
                    raise RuntimeError(f"{check['label']} failed or has incomplete evidence; inspect console log")
                print(f"Completed {check['label']}", flush=True)
            if changed or started: progress(root, plan, [item[0]['label'] for item in active])
            if active: time.sleep(.25)
    finally:
        for _, proc, _ in active:
            if proc.poll() is None: proc.send_signal(signal.SIGINT)
        if active: print('Waiting for evaluation workers to save partial diagnostics and stop.', flush=True)
        for _, proc, reader in active: proc.wait(); reader.join()


def paired_comparisons(root, results):
    comparisons = {}
    for kind in ('best', 'final'):
        for pool in ('seen', 'heldout'):
            prefix = f'{kind}-{pool}'
            contrasts = [('bench-order-ppo', 'original-ppo', 'reversed-ppo'),
                ('assistance-original', 'original-ppo', 'original-rules-forced'),
                ('assistance-reversed', 'reversed-ppo', 'reversed-rules-forced')]
            for name, left, right in contrasts:
                a, b = results.get(prefix + '-' + left), results.get(prefix + '-' + right)
                if a is None or b is None: continue
                def rows(result):
                    return {r['episode']: r for line in (inside(root, result['result_directory']) / 'episodes.jsonl').read_text().splitlines() if (r := json.loads(line))}
                aa, bb = rows(a), rows(b)
                if aa.keys() != bb.keys(): raise ValueError('Paired check episode IDs differ')
                pairs = []
                for episode in sorted(aa):
                    x, y = aa[episode], bb[episode]
                    if any(x[key] != y[key] for key in ('episode', 'matchup', 'seed', 'candidate_player')):
                        raise ValueError('Paired check schedules differ')
                    pairs.append({key: x[key] for key in ('episode', 'matchup', 'seed', 'candidate_player')} | dict(
                        left_outcome=x['candidate_outcome'], right_outcome=y['candidate_outcome'],
                        win_difference=int(y['candidate_outcome'] == 'won') - int(x['candidate_outcome'] == 'won')))
                comparisons[prefix + '-' + name] = dict(left=prefix + '-' + left, right=prefix + '-' + right,
                    wins_difference=sum(row['win_difference'] for row in pairs), requested=len(pairs),
                    win_rate_difference=sum(row['win_difference'] for row in pairs) / len(pairs), cases=pairs)
    return comparisons


def finish(root, plan):
    results = progress(root, plan)
    if set(results) != {check['label'] for check in plan['checks']}:
        raise RuntimeError('Not all frozen diagnostic checks have complete saved evidence')
    paired = paired_comparisons(root, results)
    write_json(root / 'paired-comparisons.json', paired)
    result = dict(status='smoke_completed' if plan['smoke'] else 'completed', no_training=True,
        no_automatic_promotion=True, source_selected=plan['source_selected'], models=plan['models'],
        checks=results, saved_evidence=dict(complete_episode_logs=True, complete_replacement_audits=True,
            checked_directories=sorted(value['result_directory'] for value in results.values())),
        interpretation='Assisted scores are frozen PPO+Rules hybrids; they do not show the agent learned replacement decisions.')
    write_json(root / 'result.json', result)
    return result


def collect(root, plan):
    results = progress(root, plan)
    lines = ['# Frozen replacement diagnostic', '', 'SMOKE ONLY; not learning evidence.' if plan['smoke'] else
        'Frozen matchup policies against Rules. No learning, checkpoint selection, or promotion.', '',
        'Rules-assisted columns are hybrids: Rules intervenes only at forced replacements with multiple legal switches. Ordinary move choices and voluntary switches remain PPO decisions.', '',
        'Bench reversal changes only the candidate bench order, preserving the initial lead and all team members/moves. Both orders keep the same case IDs, battle seeds and candidate seats.', '',
        '| Checkpoint | Training steps | Pool | PPO original | PPO reversed | Hybrid original | Hybrid reversed | Hybrid gain, original |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for kind in ('best', 'final'):
        for pool in ('seen', 'heldout'):
            checks = [results.get(f'{kind}-{pool}-{order}-{mode}') for order, mode in (
                ('original', 'ppo'), ('reversed', 'ppo'), ('original', 'rules-forced'), ('reversed', 'rules-forced'))]
            values = [f"{x['candidate_wins']}/{x['requested_battles']} ({x['wins_per_requested']:.1%})" if x else 'pending' for x in checks]
            delta = f"{100*(checks[2]['wins_per_requested']-checks[0]['wins_per_requested']):+.1f} pp" if checks[0] and checks[2] else '—'
            lines.append(f"| {kind} | {plan['models'][kind]['timesteps']:,} | {pool} | {' | '.join(values)} | {delta} |")
    lines += ['', 'Best means source validation selected; final is the predeclared final-budget checkpoint. Neither is selected by this diagnostic.', '',
        '## All-battle replacement audit', '',
        '| Check | Eligible choices | Actual lowest slot | Base policy lowest slot | Base/Rules disagreements | Override opportunities | Changed overrides | Truncated |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for check in plan['checks']:
        value = results.get(check['label'])
        if value:
            counts = value['audit_totals']
            lines.append(f"| {check['label']} | {' | '.join(str(counts[key]) for key in REPLACEMENT_COUNTERS)} | {value['truncated']} |")
    lines += ['', 'Eligible choices have at least two legal forced switch actions. Lowest-slot counts describe slot preference, not whether the chosen replacement was strategically correct. Rules is a heuristic comparator, not an optimal-action oracle.', '',
        '## Order calibration baselines', '', '| Baseline | Original heldout | Reversed heldout |', '|---|---:|---:|']
    for baseline in ('greedy', 'rules'):
        values = [results.get(f'baseline-{baseline}-{order}') for order in ('original', 'reversed')]
        lines.append(f"| {baseline} | " + ' | '.join(f"{x['wins_per_requested']:.1%}" if x else 'pending' for x in values) + ' |')
    lines += ['', f'Complete saved checks: {len(results)}/{len(plan["checks"])}.',
        'paired-comparisons.json retains case-level paired outcomes. decision-audit.json contains all-battle counts; decision-traces.jsonl is bounded. memory.csv includes evaluation peaks.', '',
        'Watch jobs: `tail -n 20 -F ' + str(root / 'console') + '/*.log`',
        'Watch memory: `watch -n 2 free -h`',
        'Inspect teams: `python run_replacement_diagnostic.py --output-dir ' + shlex.quote(str(root)) + ' --inspect-teams train --full-json`',
        'Pause with Ctrl+C once and wait for worker cleanup. Resume with `python run_replacement_diagnostic.py --output-dir ' + shlex.quote(str(root)) + ' --continue --jobs 2`. Complete checks are reused; interrupted checks are preserved and retried.']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    temporary = root / 'replacement-results.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if (path.is_file() and path.suffix in ('.json', '.jsonl', '.md', '.csv')
                    and relative.parts[0] not in ('console', 'locks')):
                archive.write(path, relative)
    temporary.replace(root / 'replacement-results.zip')


def inspect(root, plan, group, limit, full):
    print('Frozen diagnostic: no training. train/mirror name the source training pools.')
    for name in ('train', 'mirror', 'validation', 'heldout') if group == 'all' else (group,):
        if name in plan['teams']:
            values = [(str(root / path), read_json(root / path)) for path in plan['teams'][name]]
        else:
            path = root / plan['suites']['heldout-original' if name == 'heldout' else name]
            values = [(str(path) + ' asset ' + key, value['team']) for key, value in read_json(path)['assets'].items()]
        print(f'\n{name}: {len(values)} teams')
        for path, team in values[:limit]:
            print(path)
            if full: print(json.dumps(team, indent=2))
            else:
                for mon in team['party']:
                    print(f"  {mon['species']}: {', '.join(mon['moves'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('.local/run/matchup-ablation-01'))
    parser.add_argument('--output-dir', type=Path, default=Path('.local/run/replacement-diagnostic-01'))
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--continue', dest='continuing', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--inspect-teams', nargs='?', const='all', choices=('all', 'train', 'mirror', 'validation', 'heldout'))
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--full-json', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Eight cases per check; not learning evidence')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.output_dir.expanduser().resolve()
    if args.limit < 0: parser.error('--limit cannot be negative')
    import torch
    torch.set_num_threads(1)
    if args.worker:
        plan = read_json(root / 'plan.json'); verify(root, plan)
        matches = [check for check in plan['checks'] if check['label'] == args.worker]
        if len(matches) != 1: parser.error('Unknown diagnostic worker label')
        lock = (root / 'locks' / (args.worker + '.lock')).open('a')
        previous = signal.signal(signal.SIGTERM, interrupt)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_check(root, plan, matches[0]); return 0
        finally:
            signal.signal(signal.SIGTERM, previous); lock.close()
    if args.inspect_teams is not None:
        plan = read_json(root / 'plan.json'); verify(root, plan)
        inspect(root, plan, args.inspect_teams, args.limit, args.full_json); return 0
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = (root.parent / ('.' + root.name + '.replacement.lock')).open('a')
    plan = None; owned = False; stopped = threading.Event(); monitor = None
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); owned = True
        if args.continuing:
            plan = read_json(root / 'plan.json'); verify(root, plan)
        else: plan = freeze(root, args)
        print(f"Output: {root}\nLogs: tail -n 20 -F {shlex.quote(str(root / 'console'))}/*.log\nMemory: watch -n 2 free -h\nInspect: python run_replacement_diagnostic.py --output-dir {shlex.quote(str(root))} --inspect-teams train --full-json", flush=True)
        if args.prepare_only:
            write_json(root / 'status.json', dict(status='prepared', no_training=True)); return 0
        monitor = threading.Thread(target=memory_monitor, args=(root, stopped), daemon=True); monitor.start()
        write_json(root / 'status.json', dict(status='running', no_training=True))
        progress(root, plan)
        launch(root, plan, args.jobs)
        verify(root, plan)
        result = finish(root, plan)
        write_json(root / 'status.json', dict(status=result['status'], no_training=True))
        return 0
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
            print(f"Report: {root / 'report.md'}\nUpload: {root / 'replacement-results.zip'}", flush=True)
        signal.signal(signal.SIGTERM, previous); lock.close()


if __name__ == '__main__': raise SystemExit(main())
