"""Select and verify deterministic replays of losses from a saved evaluation."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from .evaluation import digest, validate_suite
from .periodic_evaluation import suite_opponent_path


def prepare_review(source, output, *, matchups=12, seed=261001, sample_all=False):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f'output directory already exists: {output}')
    if not 1 <= matchups <= 100:
        raise ValueError('matchups must be between 1 and 100')
    suite = validate_suite(json.loads((source / 'suite.json').read_text()))
    config = json.loads((source / 'config.json').read_text())
    if suite['backend'] != 'showdown' or not suite['swap_sides']:
        raise ValueError('review requires a side-swapped Showdown evaluation')
    if config.get('suite_sha256') != digest(suite) or not config.get('deterministic'):
        raise ValueError('source must be deterministic and match its recorded suite hash')
    candidate = config['models']['candidate']
    if candidate.get('observation_version') != 'v2':
        raise ValueError('decision review requires an observation-v2 candidate')
    model_path = source / 'candidate.zip'
    if not model_path.is_file():
        model_path = Path(candidate['snapshot_path'] if candidate.get('snapshot_path') else candidate['path'])
    model = model_path.read_bytes()
    if hashlib.sha256(model).hexdigest() != candidate['sha256']:
        raise ValueError('candidate checkpoint hash mismatch')
    rows = {}
    for line in (source / 'episodes.jsonl').read_text().splitlines():
        row = json.loads(line)
        episode = row['episode']
        if episode in rows or not 1 <= episode <= len(suite['cases']):
            raise ValueError('duplicate or invalid source episode')
        case = suite['cases'][episode-1]
        if any(row.get(k) != v for k, v in case.items()):
            raise ValueError('source episode disagrees with suite')
        rows[episode] = row
    def is_rules(case):
        if suite.get('rule_opponent'):
            return True
        return suite.get('opponents', {}).get(case.get('opponent_id'), {}).get('kind') == 'rule_based'
    candidates = [i for i in range(0, len(suite['cases']), 2)
                  if is_rules(suite['cases'][i]) and all(e in rows for e in (i+1, i+2))
                  and (sample_all or any(rows[e]['candidate_outcome'] == 'lost' and not rows[e]['truncated']
                          for e in (i+1, i+2)))]
    if not candidates:
        raise ValueError('no eligible paired Rules results' if sample_all else 'no completed paired matchups containing losses against rules')
    selected = sorted(random.Random(seed).sample(candidates, min(matchups, len(candidates))))
    subset = deepcopy(suite)
    subset.pop('opponent_quotas', None)
    subset['cases'] = []
    mapping = []
    for index in selected:
        for old in suite['cases'][index:index+2]:
            new = deepcopy(old)
            new.update(episode=len(subset['cases'])+1, matchup=len(subset['cases'])//2+1)
            subset['cases'].append(new)
            mapping.append(dict(episode=new['episode'], source_episode=old['episode'],
                                expected_outcome=rows[old['episode']]['candidate_outcome'],
                                expected_truncated=rows[old['episode']]['truncated'],
                                expected_decisions=rows[old['episode']]['decisions']))
    used = {key for case in subset['cases'] for key in case['assets']}
    subset['assets'] = {k:v for k,v in subset['assets'].items() if k in used}
    blobs = {}
    if subset.get('opponents'):
        used = {c['opponent_id'] for c in subset['cases']}
        subset['opponents'] = {k:v for k,v in subset['opponents'].items() if k in used}
        for entry in subset['opponents'].values():
            if entry['kind'] == 'checkpoint':
                data = suite_opponent_path(source / 'suite.json', entry).read_bytes()
                if hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise ValueError('opponent checkpoint hash mismatch')
                blobs[entry['sha256']] = data
    validate_suite(subset)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'
    inputs.mkdir()
    (inputs / 'opponents').mkdir()
    for checksum, data in blobs.items():
        (inputs / 'opponents' / f'{checksum}.zip').write_bytes(data)
    (inputs / 'candidate.zip').write_bytes(model)
    (inputs / 'suite.json').write_text(json.dumps(subset, indent=2)+'\n')
    (inputs / 'reward.json').write_text(json.dumps(config['reward'], indent=2)+'\n')
    selection = dict(source=str(source), source_suite_sha256=digest(suite),
                     candidate_sha256=candidate['sha256'], seed=seed, sample_all=sample_all,
                     eligible_matchups=len(candidates), selected_matchups=len(selected), cases=mapping)
    (output / 'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    return selection


def report_review(output):
    output = Path(output)
    selection = json.loads((output / 'selection.json').read_text())
    replay = output / 'replay'
    rows = {r['episode']: r for r in map(json.loads, (replay / 'episodes.jsonl').read_text().splitlines())}
    audit = json.loads((replay / 'decision-audit.json').read_text())
    mismatches, totals = [], Counter()
    losses = {c['episode'] for c in selection['cases'] if c['expected_outcome'] == 'lost' and not c['expected_truncated']}
    table = []
    for case in selection['cases']:
        row = rows.get(case['episode'], {})
        matched = (row.get('candidate_outcome') == case['expected_outcome'] and
                   row.get('truncated') == case['expected_truncated'] and
                   row.get('decisions') == case['expected_decisions'])
        if not matched:
            mismatches.append(case['episode'])
        if case['episode'] in losses:
            totals.update(audit['by_episode'].get(str(case['episode']), {}))
        table.append(f"| {case['source_episode']} | {case['episode']} | {case['expected_outcome']} | {row.get('candidate_outcome', 'missing')} | {'yes' if matched else 'NO'} |")
    data = dict(selection=selection, replay_mismatches=mismatches,
                replay_verified=not mismatches, selected_loss_battles=len(losses),
                loss_decision_counts=dict(totals), examples=[r for r in audit['examples'] if r['episode'] in losses])
    (output / 'review.json').write_text(json.dumps(data, indent=2)+'\n')
    lines = ['# Rules loss review', '',
             f"Replay verification: **{'PASS' if not mismatches else 'MISMATCH — investigate before interpreting'}** (outcome, truncation and decision count).",
             f"Selected {selection['selected_matchups']} of {selection['eligible_matchups']} eligible matchups; {len(losses)} originally lost battles, plus paired-seat results.", '',
             'This is a loss-selected diagnostic sample, not a win-rate estimate. Heuristic disagreements are not proven mistakes.', '',
             '| Signal in originally lost battles | Decisions |', '|---|---:|']
    lines += [f'| {key} | {value} |' for key, value in sorted(totals.items())]
    lines += ['', 'Automatic/single-legal-action decisions are excluded from disagreement flags. Replacement choices are counted separately from voluntary switches.',
              'A lower-direct-damage flag means another legal move had an estimate at least 25% of enemy maximum HP higher. Setup, healing, unsupported mechanics and longer-term plans can justify the chosen action.', '',
              '| Source episode | Replay episode | Original | Replay | Matches |', '|---:|---:|---|---|---|', *table,
              '', '## Decision examples', '', 'Up to 40 early flagged examples are retained across all replay cases; these are illustrative, not ranked by severity.']
    for row in data['examples'][:20]:
        hp = row['hp']
        lines += ['', f"- Episode {row['episode']} (pair {row.get('pair')}), decision {row['decision']}, turn {row['turn']}: **{row['chosen']}**. Rules suggest **{row['suggestion']}**.",
                  f"  Active: {row['active_species'][0]} {hp[0][0]:g}/{hp[0][1]:g} HP vs {row['active_species'][1]} {hp[1][0]:g}/{hp[1][1]:g} HP. Flags: {', '.join(row['flags'])}."]
    lines += ['', 'See `replay/candidate-decisions.jsonl` (and rotated backups) for legal alternatives, scores, reasons and status bits. `replay/pair-*/battle.log` and `battle-readable.log` provide battle context. All decision counts survive trace rotation in `decision-audit.json`.',
              '', 'No counterfactual actions were executed. Estimated damage omits some mechanics and is not guaranteed damage or a guaranteed knockout. No training or reward changes occurred.']
    (output / 'review.md').write_text('\n'.join(lines)+'\n')
    return data
