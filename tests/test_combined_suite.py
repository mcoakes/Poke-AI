import copy
import hashlib
import json
from pathlib import Path

import pytest

from firered_rl.combined_suite import combine_suites
from firered_rl.evaluation import digest, validate_suite
from firered_rl.periodic_evaluation import suite_opponent_path
from firered_rl.rule_opponent import RuleBasedOpponent

ROOT = Path(__file__).resolve().parents[1]


def sources(tmp_path):
    asset = {'team': {'party': [{'species': 'Pikachu', 'level': 50, 'moves': ['Thunderbolt']}]}}
    key = digest(asset)
    base = dict(version=1, backend='showdown', seed=21, max_steps=20, swap_sides=True,
                assets={key: asset}, cases=[dict(episode=i, matchup=1, assets=[key, key],
                                              seed=71, candidate_player=i) for i in (1, 2)])
    data = b'frozen checkpoint fixture'
    checksum = hashlib.sha256(data).hexdigest()
    original = tmp_path / 'original.zip'
    original.write_bytes(data)
    learned = copy.deepcopy(base)
    learned['opponents'] = {'smart': dict(kind='checkpoint', group='smart',
                                         path=str(original), sha256=checksum)}
    for case in learned['cases']:
        case.update(opponent_id='smart', opponent_group='smart', opponent_seed=19)
    rule = copy.deepcopy(base)
    rule['rule_opponent'] = RuleBasedOpponent(ROOT).identity
    paths = [tmp_path / 'learned.json', tmp_path / 'rules.json']
    for path, suite in zip(paths, (learned, rule)):
        path.write_text(json.dumps(suite))
    return paths, learned, rule, original


def test_preserves_cases_bundles_models_and_source_files(tmp_path):
    paths, learned, rules, original = sources(tmp_path)
    before = {p: p.read_bytes() for p in [*paths, original]}
    path, manifest = combine_suites(paths, tmp_path / 'combined', project=ROOT)
    result = validate_suite(json.loads(path.read_text()))
    assert manifest['by_opponent_group'] == {'smart': 2, 'rules': 2}
    assert manifest['suite_sha256'] == digest(result)
    assert [c['matchup'] for c in result['cases']] == [1, 1, 2, 2]
    for source, cases in zip((learned, rules), (result['cases'][:2], result['cases'][2:])):
        for a, b in zip(source['cases'], cases):
            for key in ('seed', 'assets', 'candidate_player'):
                assert a[key] == b[key]
    assert result['cases'][0]['opponent_seed'] == 19
    assert result['cases'][2]['opponent_seed'] == 71 ^ 0x47B9
    assert all(p.read_bytes() == data for p, data in before.items())
    original.unlink()
    entry = result['opponents']['s1_smart']
    assert suite_opponent_path(path, entry).read_bytes() == before[original]
    with pytest.raises(ValueError, match='already exists'):
        combine_suites(paths, path.parent, project=ROOT)


@pytest.mark.parametrize('change,message', [
    ('hash', 'checkpoint hash mismatch'), ('rules', 'rule implementation differs'),
    ('steps', 'same max_steps'), ('unbound', 'must pin'), ('duplicate', 'distinct'),
])
def test_bad_inputs_fail_before_creating_output(tmp_path, change, message):
    paths, learned, rules, original = sources(tmp_path)
    if change == 'hash':
        original.write_bytes(b'tampered')
    elif change == 'rules':
        rules['rule_opponent']['sha256'] = '0' * 64
        paths[1].write_text(json.dumps(rules))
    elif change == 'steps':
        rules['max_steps'] = 10
        paths[1].write_text(json.dumps(rules))
    elif change == 'unbound':
        rules.pop('rule_opponent')
        paths[1].write_text(json.dumps(rules))
    elif change == 'duplicate':
        paths = [paths[0], paths[0]]
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match=message):
        combine_suites(paths, output, project=ROOT)
    assert not output.exists()
