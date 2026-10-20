"""Bench augmentation changes order without changing the canonical experiment."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from firered_rl.bench_order import BenchOrderTeams, validate_options
from firered_rl.training_teams import TrainingTeams


SPECIES = ['Blastoise', 'Raichu', 'Golem', 'Venusaur', 'Arcanine', 'Lapras']


def write_team(path, index=0, size=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    team = dict(name=f'team-{index}', party=[dict(species=SPECIES[(slot+index) % 6],
        level=50, moves=['Strength', 'Protect'], held_item='None', nature='Hardy',
        ivs=dict(hp=31, atk=31, defense=31, spa=31, spd=31, spe=31))
        for slot in range(size)])
    # TeamSpec accepts the project's stat key 'def', not the Python-friendly spelling.
    for mon in team['party']:
        mon['ivs']['def'] = mon['ivs'].pop('defense')
    path.write_text(json.dumps(team) + '\n')
    return path


def mixture(tmp_path, size=3):
    train = tmp_path / 'train'
    for i in range(4): write_team(train / f'{i}.json', i, size)
    fixed = [write_team(tmp_path / 'mirror' / f'{i}.json', i, size) for i in range(2)]
    path = tmp_path / 'mixture.json'
    path.write_text(json.dumps(dict(version=1, groups=dict(
        current=dict(weight=.9, directory=str(train)),
        mirror=dict(weight=.1, teams=list(map(str, fixed)))))))
    return TrainingTeams(mixture=path, seed=123).snapshot(tmp_path / 'frozen')


def source_again(source, seed=123):
    return TrainingTeams(mixture=source.mixture, seed=seed)


def original_path(wrapper, path):
    if path.parent == wrapper.directory:
        return Path(wrapper.manifest['variants'][path.name]['source'])
    return path


def test_pair_sampling_group_distribution_lead_and_opponent_are_unchanged(tmp_path):
    pinned = mixture(tmp_path)
    control = source_again(pinned)
    source = source_again(pinned)
    description = copy.deepcopy(source.describe())
    wrapped = BenchOrderTeams(source, tmp_path / 'variants', seed=123)
    plain, augmented = control.for_pair(0), wrapped.for_pair(0)
    for _ in range(250):
        expected = plain(); actual = augmented()
        assert original_path(wrapped, actual[0]) == expected[0]
        assert actual[1] == expected[1]
        old, new = [json.loads(path.read_text()) for path in (expected[0], actual[0])]
        assert old['party'][0] == new['party'][0]
        assert sorted(map(lambda x: json.dumps(x, sort_keys=True), old['party'])) == sorted(
            map(lambda x: json.dumps(x, sort_keys=True), new['party']))
        assert {k:v for k,v in old.items() if k != 'party'} == {k:v for k,v in new.items() if k != 'party'}
        if str(expected[0]) in wrapped.fixed_files:
            assert actual == expected
    assert wrapped.describe() == description
    assert wrapped.sampling_summary() == control.sampling_summary()
    result = wrapped.get_summary(); counts = result['counts']
    assert counts['episode_selections'] == 250
    assert counts['skipped_fixed_pair'] == control.sampling_summary()['episode_selections']['mirror'] > 0
    assert counts['eligible'] == counts['identity'] + counts['changed']
    assert counts['episode_selections'] == counts['eligible'] + counts['skipped_fixed_pair']
    assert counts['identity'] > 0 and counts['changed'] > 0
    assert result['stored_variants'] <= result['maximum_variants'] == 4
    json.dumps(result, allow_nan=False)


def semantic_sequence(wrapper, choices, count):
    return [(str(original_path(wrapper, paths[0])), json.loads(paths[0].read_text()), str(paths[1]))
            for paths in (choices() for _ in range(count))]


def test_independent_pair_rng_reproducibility_and_invocation_seed(tmp_path):
    pinned = mixture(tmp_path, size=6)
    a = BenchOrderTeams(source_again(pinned), tmp_path / 'a', seed=123)
    b = BenchOrderTeams(source_again(pinned), tmp_path / 'b', seed=123)
    a0, a1, b0, b1 = a.for_pair(0), a.for_pair(1), b.for_pair(0), b.for_pair(1)
    # Interleaving another pair does not consume either pair's permutation RNG.
    expected0 = semantic_sequence(a, a0, 30)
    expected1 = semantic_sequence(a, a1, 30)
    actual0, actual1 = [], []
    for _ in range(30):
        actual1 += semantic_sequence(b, b1, 1)
        actual0 += semantic_sequence(b, b0, 1)
    assert actual0 == expected0 and actual1 == expected1
    assert a.get_summary()['maximum_variants'] == 4 * 119
    different = BenchOrderTeams(source_again(pinned), tmp_path / 'different', seed=999)
    changed = semantic_sequence(different, different.for_pair(0), 30)
    assert [x[0] for x in changed] == [x[0] for x in expected0]
    assert [x[2] for x in changed] == [x[2] for x in expected0]
    assert any(x[1] != y[1] for x, y in zip(changed, expected0))


def test_threaded_callbacks_keep_manifest_and_counts_consistent(tmp_path):
    pinned = mixture(tmp_path)
    wrapped = BenchOrderTeams(source_again(pinned), tmp_path / 'variants', seed=123)
    choosers = [wrapped.for_pair(i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda choose: [choose() for _ in range(40)], choosers))
    result = wrapped.get_summary()
    assert result['counts']['episode_selections'] == 160
    assert all(row['episode_selections'] == 40 for row in result['per_pair'].values())
    assert json.loads(wrapped.manifest_path.read_text()) == wrapped.manifest
    reopened = BenchOrderTeams(source_again(pinned), wrapped.directory, seed=1000126)
    assert reopened.get_summary()['counts']['episode_selections'] == 0
    assert reopened.get_summary()['stored_variants'] == result['stored_variants']


def test_missing_variant_recovers_but_modified_variant_is_rejected(tmp_path):
    pinned = mixture(tmp_path)
    wrapped = BenchOrderTeams(source_again(pinned), tmp_path / 'variants', seed=123)
    choose = wrapped.for_pair(0)
    for _ in range(30): choose()
    name = next(iter(wrapped.manifest['variants']))
    path = wrapped.directory / name
    expected = path.read_bytes(); path.unlink()
    BenchOrderTeams(source_again(pinned), wrapped.directory, seed=456)
    assert path.read_bytes() == expected
    path.write_text('{}')
    with pytest.raises(ValueError, match='frozen bench variant changed'):
        BenchOrderTeams(source_again(pinned), wrapped.directory, seed=456)


def test_canonical_mutation_is_rejected_even_before_variants_exist(tmp_path):
    pinned = mixture(tmp_path)
    source = source_again(pinned)
    source.files[0].write_text('{}')
    with pytest.raises(ValueError, match='canonical team changed'):
        BenchOrderTeams(source, tmp_path / 'variants', seed=123)


@pytest.mark.parametrize('size', [1, 2])
def test_short_parties_keep_all_padding_absent_and_order_unchanged(tmp_path, size):
    path = write_team(tmp_path / 'teams' / 'one.json', size=size)
    source = TrainingTeams(directory=path.parent, seed=1)
    wrapped = BenchOrderTeams(source, tmp_path / 'variants', seed=1)
    assert wrapped.for_pair(0)() == (path.resolve(), path.resolve())
    assert wrapped.get_summary()['counts']['identity'] == 1
    assert wrapped.get_summary()['stored_variants'] == 0


def test_fixed_source_is_excluded_and_ambiguous_unsnapshotted_mixture_rejected(tmp_path):
    files = [write_team(tmp_path / 'teams' / f'{i}.json', i) for i in range(2)]
    fixed = TrainingTeams(teams=files, seed=1)
    wrapped = BenchOrderTeams(fixed, tmp_path / 'fixed-variants', seed=1)
    assert wrapped.for_pair(0)() == tuple(path.resolve() for path in files)
    assert wrapped.get_summary()['counts']['skipped_fixed_pair'] == 1
    manifest = tmp_path / 'ambiguous.json'
    manifest.write_text(json.dumps(dict(version=1, groups=dict(
        current=dict(weight=.9, directory=str(files[0].parent)),
        mirror=dict(weight=.1, teams=list(map(str, files)))))))
    with pytest.raises(ValueError, match='snapshot the mixture first'):
        BenchOrderTeams(TrainingTeams(mixture=manifest), tmp_path / 'bad', seed=1)


@pytest.mark.parametrize('backend,rules,pool,checkpoint', [
    ('mgba', True, None, None), ('showdown', False, None, None),
    ('showdown', True, 'pool.json', None), ('showdown', True, None, 'model.zip'),
])
def test_cli_scope_requires_showdown_rules_only(backend, rules, pool, checkpoint):
    args = SimpleNamespace(shuffle_bench_order=True, backend=backend,
        rule_based_opponent=rules, opponent_pool=pool, opponent_checkpoint=checkpoint)
    with pytest.raises(ValueError, match='shuffle-bench-order requires'):
        validate_options(args)
    args.shuffle_bench_order = False
    validate_options(args)


def test_cli_scope_accepts_rules_and_preserves_inputs():
    args = SimpleNamespace(shuffle_bench_order=True, backend='showdown', rule_based_opponent=True)
    validate_options(args)
    assert args.shuffle_bench_order is True
