"""The next training pool must be new, reproducible and exactly slot-balanced."""
from collections import Counter
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path

import pytest

from firered_rl.balanced_training_pool import generate, validate_pool
from firered_rl.broad_team_curriculum import composition_signature, semantic_signature
from firered_rl.teams import PokemonSpec


PROJECT = Path(__file__).resolve().parents[1]

# These are the 41 distinct compositions in the previous experiment's 131
# input team records: train + mirror + every frozen-suite asset (both orders).
# Keep the regression independent of uploaded files and absolute user paths.
OLD_COMPOSITIONS = (
    ('Alakazam', 'Arcanine', 'Dodrio'),
    ('Alakazam', 'Blastoise', 'Raichu'),
    ('Alakazam', 'Blastoise', 'Vaporeon'),
    ('Alakazam', 'Golem', 'Typhlosion'),
    ('Alakazam', 'Machamp', 'Vaporeon'),
    ('Alakazam', 'Raichu', 'Typhlosion'),
    ('Arcanine', 'Blastoise', 'Golem'),
    ('Arcanine', 'Blastoise', 'Raichu'),
    ('Arcanine', 'Dodrio', 'Lapras'),
    ('Arcanine', 'Dodrio', 'Vaporeon'),
    ('Arcanine', 'Golem', 'Lapras'),
    ('Arcanine', 'Golem', 'Typhlosion'),
    ('Arcanine', 'Lapras', 'Raichu'),
    ('Arcanine', 'Machamp', 'Raichu'),
    ('Arcanine', 'Sceptile', 'Vaporeon'),
    ('Arcanine', 'Vaporeon', 'Venusaur'),
    ('Blastoise', 'Golem', 'Lapras'),
    ('Blastoise', 'Golem', 'Vaporeon'),
    ('Blastoise', 'Machamp', 'Raichu'),
    ('Blastoise', 'Machamp', 'Sceptile'),
    ('Blastoise', 'Machamp', 'Typhlosion'),
    ('Blastoise', 'Machamp', 'Vaporeon'),
    ('Blastoise', 'Machamp', 'Venusaur'),
    ('Blastoise', 'Raichu', 'Typhlosion'),
    ('Blastoise', 'Sceptile', 'Typhlosion'),
    ('Blastoise', 'Sceptile', 'Venusaur'),
    ('Dodrio', 'Golem', 'Machamp'),
    ('Dodrio', 'Machamp', 'Venusaur'),
    ('Dodrio', 'Raichu', 'Venusaur'),
    ('Dodrio', 'Typhlosion', 'Vaporeon'),
    ('Golem', 'Lapras', 'Raichu'),
    ('Golem', 'Lapras', 'Typhlosion'),
    ('Golem', 'Machamp', 'Sceptile'),
    ('Golem', 'Machamp', 'Vaporeon'),
    ('Golem', 'Raichu', 'Sceptile'),
    ('Golem', 'Raichu', 'Typhlosion'),
    ('Lapras', 'Machamp', 'Raichu'),
    ('Machamp', 'Raichu', 'Sceptile'),
    ('Machamp', 'Typhlosion', 'Vaporeon'),
    ('Raichu', 'Sceptile', 'Typhlosion'),
    ('Sceptile', 'Typhlosion', 'Venusaur'),
)

SIMPLE_MOVES = {
    'Alakazam': ['Psychic', 'Body Slam'],
    'Arcanine': ['Flamethrower', 'Body Slam'],
    'Blastoise': ['Surf', 'Strength'],
    'Dodrio': ['Drill Peck', 'Double Edge'],
    'Golem': ['Earthquake', 'Strength'],
    'Lapras': ['Surf', 'Body Slam'],
    'Machamp': ['Brick Break', 'Strength'],
    'Raichu': ['Thunderbolt', 'Strength'],
    'Sceptile': ['Leaf Blade', 'Strength'],
    'Typhlosion': ['Flamethrower', 'Strength'],
    'Vaporeon': ['Surf', 'Body Slam'],
    'Venusaur': ['Razor Leaf', 'Strength'],
}


def team_for(species, name='old'):
    return {'name': name, 'party': [dict(
        species=value, level=50, nature='Hardy', held_item='None',
        moves=list(SIMPLE_MOVES[value]),
        evs={stat: 0 for stat in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')},
        ivs={stat: 31 for stat in ('hp', 'atk', 'def', 'spa', 'spd', 'spe')},
    ) for value in species]}


@pytest.fixture(scope='module')
def config():
    return json.loads((PROJECT / 'config/experiments/team-curriculum.json').read_text())


@pytest.fixture(scope='module')
def old_teams():
    return [team_for(names, f'old-{index:04d}')
            for index, names in enumerate(OLD_COMPOSITIONS)]


@pytest.fixture(scope='module')
def production_pool(config, old_teams):
    return generate(PROJECT, config, old_teams, count=96, seed=261130)


def assert_balanced(teams, species, count):
    assert len(teams) == count
    assert all(len(team['party']) == 3 for team in teams)
    expected_total = {name: count // 4 for name in species}
    assert Counter(mon['species'] for team in teams for mon in team['party']) == expected_total
    for slot in range(3):
        assert Counter(team['party'][slot]['species'] for team in teams) == {
            name: count // 12 for name in species}
    assert len({composition_signature(team) for team in teams}) == count
    assert len({semantic_signature(team) for team in teams}) == count
    assert all(len({mon['species'] for mon in team['party']}) == 3 for team in teams)


def test_actual_previous_compositions_leave_a_feasible_balanced_96_pool(
        config, old_teams, production_pool):
    teams, metadata = production_pool
    assert len({composition_signature(team) for team in old_teams}) == 41
    assert_balanced(teams, config['species_pools']['intro'], 96)
    old_compositions = {composition_signature(team) for team in old_teams}
    old_semantics = {semantic_signature(team) for team in old_teams}
    assert not old_compositions.intersection(map(composition_signature, teams))
    assert not old_semantics.intersection(map(semantic_signature, teams))
    ledger = validate_pool(PROJECT, config, teams, old_teams, count=96)
    assert isinstance(ledger, dict) and isinstance(metadata, dict)
    assert ledger['version'] == 1 and ledger['stage'] == '04-type-matchups'
    assert ledger['teams'] == 96
    assert ledger['species_counts'] == {name: 24 for name in config['species_pools']['intro']}
    assert ledger['slot_counts'] == {
        str(slot): {name: 8 for name in config['species_pools']['intro']} for slot in range(3)}
    assert ledger['compositions'] == sorted(map(composition_signature, teams))
    assert ledger['semantics'] == sorted(map(semantic_signature, teams))
    assert metadata['seed'] == 261130 and metadata['eligible_compositions'] == 179
    assert len(ledger['team_ledger']) == 96
    for team, row in zip(teams, ledger['team_ledger']):
        assert row['composition'] == composition_signature(team)
        assert row['semantic'] == semantic_signature(team)
        assert row['party'] == [mon['species'] for mon in team['party']]
        assert row['moves'] == [mon['moves'] for mon in team['party']]
    json.dumps({'metadata': metadata, 'ledger': ledger}, allow_nan=False)


def test_smoke_pool_has_three_appearances_and_one_per_slot(config, old_teams):
    teams, _ = generate(PROJECT, config, old_teams, count=12, seed=261130)
    assert_balanced(teams, config['species_pools']['intro'], 12)
    validate_pool(PROJECT, config, teams, old_teams, count=12)


def test_generated_builds_keep_the_stage04_legal_simple_two_attack_recipe(config, production_pool):
    catalog = json.loads((PROJECT / config['catalog']).read_text())
    stage = next(stage for stage in config['stages'] if stage['id'] == '04-type-matchups')
    attack_pool = set(config['move_pools'][stage['moves']['attack_pool']])
    for team in production_pool[0]:
        for mon in team['party']:
            PokemonSpec.from_dict(mon)
            info = catalog['species'][mon['species']]
            assert mon['level'] == 50 and mon['nature'] == 'Hardy'
            assert mon['held_item'] == 'None'
            assert set(mon['evs'].values()) == {0}
            assert set(mon['ivs'].values()) == {31}
            assert len(mon['moves']) == len(set(mon['moves'])) == 2
            assert set(mon['moves']) <= set(info['moves'])
            assert set(mon['moves']) <= attack_pool
            moves = [catalog['moves'][move] for move in mon['moves']]
            assert all(move['power'] > 1 for move in moves)
            assert any(move['type'] == 'Normal' for move in moves)
            assert any(move['type'] in info['types'] and move['power'] >= 40 for move in moves)


def test_generation_is_seeded_independent_of_old_input_order_and_does_not_mutate(
        config, old_teams):
    before_config, before_old = deepcopy(config), deepcopy(old_teams)
    first = generate(PROJECT, config, old_teams, count=12, seed=261130)
    second = generate(PROJECT, config, list(reversed(old_teams)), count=12, seed=261130)
    assert first == second
    changed_seed = generate(PROJECT, config, old_teams, count=12, seed=261131)
    assert first[0] != changed_seed[0]
    assert config == before_config and old_teams == before_old


def test_cosmetic_member_and_move_order_changes_do_not_bypass_old_exclusions(config, old_teams):
    permuted = deepcopy(old_teams)
    for team in permuted:
        team['name'] = 'cosmetic label'
        team['party'].reverse()
        for mon in team['party']:
            mon['moves'].reverse()
    original, _ = generate(PROJECT, config, old_teams, count=12, seed=261130)
    reordered, _ = generate(PROJECT, config, permuted, count=12, seed=261130)
    assert reordered == original


def test_validation_rejects_slot_imbalance_without_composition_or_global_count_changes(
        config, old_teams, production_pool):
    changed = deepcopy(production_pool[0])
    changed[0]['party'][0], changed[0]['party'][1] = changed[0]['party'][1], changed[0]['party'][0]
    assert {composition_signature(team) for team in changed} == {
        composition_signature(team) for team in production_pool[0]}
    with pytest.raises(ValueError):
        validate_pool(PROJECT, config, changed, old_teams, count=96)


def test_validation_rejects_species_imbalance_even_when_every_composition_is_new(
        config, old_teams, production_pool):
    teams = deepcopy(production_pool[0])
    existing = {composition_signature(team) for team in [*teams, *old_teams]}
    examples = {mon['species']: mon for team in teams for mon in team['party']}
    replacement = None
    for index, team in enumerate(teams):
        for species, mon in examples.items():
            if species in {member['species'] for member in team['party']}:
                continue
            changed = deepcopy(team)
            changed['party'][0] = deepcopy(mon)
            if composition_signature(changed) not in existing:
                replacement = index, changed
                break
        if replacement is not None:
            break
    assert replacement is not None
    index, changed = replacement
    teams[index] = changed
    with pytest.raises(ValueError, match='species balance'):
        validate_pool(PROJECT, config, teams, old_teams, count=96)


@pytest.mark.parametrize('change', ['duplicate', 'old_composition', 'count', 'illegal_move', 'build'])
def test_validation_rejects_tampered_pool(config, old_teams, production_pool, change):
    teams = deepcopy(production_pool[0])
    if change == 'duplicate':
        teams[-1] = deepcopy(teams[0])
        teams[-1]['name'] = 'a new label does not make a new team'
    elif change == 'old_composition':
        teams[-1] = deepcopy(old_teams[0])
    elif change == 'count':
        teams.pop()
    elif change == 'illegal_move':
        teams[0]['party'][0]['moves'][0] = 'Not A Real Move'
    else:
        teams[0]['party'][0]['level'] = 100
    with pytest.raises(ValueError):
        validate_pool(PROJECT, config, teams, old_teams, count=96)


@pytest.mark.parametrize('count', [0, -12, 1, 13, 95, True, 12.0])
def test_generation_rejects_nonpositive_or_nonintegral_balanced_count(config, old_teams, count):
    with pytest.raises(ValueError):
        generate(PROJECT, config, old_teams, count=count, seed=261130)


def test_generation_rejects_insufficient_unseen_compositions(config, old_teams):
    # C(12, 3) = 220; the actual 41 exclusions leave only 179 possibilities.
    with pytest.raises(ValueError):
        generate(PROJECT, config, old_teams, count=180, seed=261130)


def test_generation_rejects_pool_that_cannot_balance_a_missing_species(config):
    species = config['species_pools']['intro']
    excluded = [team_for(names) for names in combinations(species, 3)
                if 'Alakazam' in names]
    # There are many available compositions, but none can contribute Alakazam.
    with pytest.raises(ValueError):
        generate(PROJECT, config, excluded, count=12, seed=261130)
