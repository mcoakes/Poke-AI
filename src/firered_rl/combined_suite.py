"""Combine pinned Showdown validation schedules without resampling their cases."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .evaluation import digest, validate_suite
from .periodic_evaluation import suite_opponent_path


def combine_suites(paths, output, *, project):
    paths = [Path(path).expanduser().resolve() for path in paths]
    output = Path(output).expanduser().resolve()
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("provide at least two distinct source suites")
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    sources = [validate_suite(json.loads(path.read_text())) for path in paths]
    first = sources[0]
    if any(s['backend'] != 'showdown' or not s['swap_sides'] for s in sources):
        raise ValueError("combined validation requires side-swapped Showdown suites")
    if any(s['max_steps'] != first['max_steps'] for s in sources):
        raise ValueError("source suites must use the same max_steps")
    result = dict(version=1, backend='showdown', swap_sides=True,
                  seed=first['seed'], max_steps=first['max_steps'],
                  assets={}, opponents={}, cases=[])
    blobs, provenance = {}, []
    rule_identity = None
    for index, (path, source) in enumerate(zip(paths, sources), 1):
        prefix = f's{index}_'
        entries = source.get('opponents')
        if not entries:
            if not source.get('rule_opponent'):
                raise ValueError(f"source must pin its opponent pool or rules identity: {path}")
            entries = {'rules': dict(kind='rule_based', group='rules',
                                     identity=source['rule_opponent'], label='Rules')}
        for key, original in entries.items():
            entry = deepcopy(original)
            if entry['kind'] == 'checkpoint':
                data = suite_opponent_path(path, entry).read_bytes()
                checksum = hashlib.sha256(data).hexdigest()
                if checksum != entry['sha256']:
                    raise ValueError(f"opponent checkpoint hash mismatch: {path}: {key}")
                blobs[checksum] = data
                entry['path'] = str(output / 'opponents' / f'{checksum}.zip')
            elif entry['kind'] == 'rule_based':
                if rule_identity is None:
                    from .rule_opponent import RuleBasedOpponent
                    rule_identity = RuleBasedOpponent(project).identity
                if entry['identity'] != rule_identity:
                    raise ValueError(f"rule implementation differs from source suite: {path}")
            result['opponents'][prefix + key] = entry
        result['assets'].update(deepcopy(source['assets']))
        start = len(result['cases']) + 1
        for case in source['cases']:
            copied = deepcopy(case)
            original_key = case.get('opponent_id', 'rules')
            copied.update(episode=len(result['cases']) + 1,
                          matchup=len(result['cases']) // 2 + 1,
                          opponent_id=prefix + original_key,
                          opponent_group=entries[original_key]['group'],
                          opponent_seed=case.get('opponent_seed', case['seed'] ^ 0x47B9))
            result['cases'].append(copied)
        provenance.append(dict(path=str(path), sha256=digest(source),
                               first_episode=start, last_episode=len(result['cases'])))
    validate_suite(result)
    counts = dict(Counter(c['opponent_group'] for c in result['cases']))
    manifest = dict(sources=provenance, suite_sha256=digest(result),
                    battles=len(result['cases']), by_opponent_group=counts,
                    note='Cases retain source teams, battle seeds and seats. No resampling or training.')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'opponents').mkdir()
    for checksum, data in blobs.items():
        (output / 'opponents' / f'{checksum}.zip').write_bytes(data)
    (output / 'suite.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output / 'suite.json', manifest
