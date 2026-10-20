"""Fresh selection and endpoint splits must remain separate and reproducible."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from firered_rl.balanced_training_pool import generate
from firered_rl.broad_team_curriculum import composition_signature
from firered_rl.evaluation import digest, validate_suite
from firered_rl.teacher_training_suites import (
    build_teacher_suites, freeze_teacher_suites, verify_frozen_teacher_suites,
)
from test_balanced_training_pool import OLD_COMPOSITIONS, team_for, assert_balanced


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def config():
    return json.loads((PROJECT / 'config/experiments/team-curriculum.json').read_text())


@pytest.fixture(scope='module')
def exclusions(config):
    # Recreate the actual prior 41 compositions plus the 96-team expansion.
    old = [team_for(names, f'legacy-{index:04d}') for index, names in enumerate(OLD_COMPOSITIONS)]
    new, _ = generate(PROJECT, config, old, count=96, seed=261130)
    return old + new


@pytest.fixture(scope='module')
def production(config, exclusions):
    return build_teacher_suites(PROJECT, config, exclusions)


def test_tight_actual_137_exclusions_leave_disjoint_balanced_36_and_24(
        config, exclusions, production):
    previous = set(map(composition_signature, exclusions))
    selection = production['teams']['validation']
    endpoint = production['teams']['fresh-test']
    assert len(previous) == 137
    assert_balanced(selection, config['species_pools']['intro'], 36)
    assert_balanced(endpoint, config['species_pools']['intro'], 24)
    selection_keys, endpoint_keys = set(map(composition_signature, selection)), set(map(composition_signature, endpoint))
    assert not previous & selection_keys
    assert not previous & endpoint_keys
    assert not selection_keys & endpoint_keys
    inventory = production['inventory']
    assert inventory['validation']['eligible_compositions'] == 83
    assert inventory['endpoint']['eligible_compositions'] == 47
    assert 'checkpoint selection' in inventory['selection_role']
    assert 'never used' in inventory['endpoint_role']
    assert inventory['validation']['seed'] == 261140
    assert inventory['endpoint']['seed'] == 261141


def test_validation_exact_budget_unique_ids_balanced_orders_and_seats(production):
    suite = validate_suite(production['suites']['validation'])
    cases = suite['cases']
    assert len(cases) == 2048
    assert [case['episode'] for case in cases] == list(range(1, 2049))
    assert Counter(case['candidate_player'] for case in cases) == {1: 1024, 2: 1024}
    assert Counter(case['matchup'] for case in cases) == dict.fromkeys(range(1, 1025), 2)
    assert len(suite['diagnostic']['paired_orders']) == 1024
    assert suite['diagnostic']['equal_order_weight']
    assert suite['diagnostic']['complete_team_coverage']
    assert sum(suite['diagnostic']['candidate_team_exposures'].values()) == 2048
    for order in (cases[:1024], cases[1024:]):
        exposures = Counter(case['assets'][case['candidate_player'] - 1] for case in order)
        assert len(exposures) == 36
        assert max(exposures.values()) - min(exposures.values()) <= 2
    for left, right, mapping in zip(cases[:1024], cases[1024:], suite['diagnostic']['paired_orders']):
        assert (left['seed'], left['candidate_player']) == (right['seed'], right['candidate_player'])
        assert mapping['original_episode'] == left['episode']
        assert mapping['reversed_episode'] == right['episode']
        seat = left['candidate_player'] - 1
        assert left['assets'][1 - seat] == right['assets'][1 - seat]
        first = suite['assets'][left['assets'][seat]]['team']['party']
        second = suite['assets'][right['assets'][seat]]['team']['party']
        assert second == first[:1] + first[:0:-1]


def test_endpoint_ordering_keeps_cases_seeds_leads_opponents_and_move_order(production):
    original = validate_suite(production['suites']['fresh-original'])
    reversed_suite = validate_suite(production['suites']['fresh-reversed'])
    assert len(original['cases']) == len(reversed_suite['cases']) == 1000
    assert Counter(case['candidate_player'] for case in original['cases']) == {1: 500, 2: 500}
    for suite in (original, reversed_suite):
        exposures = Counter(case['assets'][case['candidate_player'] - 1] for case in suite['cases'])
        assert len(exposures) == 24
        assert max(exposures.values()) - min(exposures.values()) <= 2
        assert suite['diagnostic']['candidate_team_exposures'] == exposures
    assert original['swap_sides'] is True
    assert reversed_suite['swap_sides'] is False
    assert reversed_suite['replacement_transform']['source_suite_sha256'] == digest(original)
    for left, right in zip(original['cases'], reversed_suite['cases']):
        for key in ('episode', 'matchup', 'seed', 'candidate_player'):
            assert left[key] == right[key]
        seat = left['candidate_player'] - 1
        assert left['assets'][1 - seat] == right['assets'][1 - seat]
        first = original['assets'][left['assets'][seat]]['team']['party']
        second = reversed_suite['assets'][right['assets'][seat]]['team']['party']
        assert second == [first[0], first[2], first[1]]


def test_seeds_and_exclusions_are_reproducible_without_input_mutation(config, exclusions, production):
    before = deepcopy((config, exclusions))
    repeated = build_teacher_suites(PROJECT, config, list(reversed(exclusions)))
    assert repeated == production
    assert (config, exclusions) == before
    json.dumps(production, allow_nan=False)


def test_smoke_preserves_exact_eight_case_budget_and_labels_partial_coverage(config):
    bundle = build_teacher_suites(PROJECT, config, [], validation_teams=12,
        endpoint_teams=12, validation_battles=8, endpoint_battles=8)
    for name, suite in bundle['suites'].items():
        assert len(suite['cases']) == 8
        assert Counter(case['candidate_player'] for case in suite['cases']) == {1: 4, 2: 4}
        assert suite['diagnostic']['complete_team_coverage'] is False
    for group in bundle['teams'].values():
        assert_balanced(group, config['species_pools']['intro'], 12)
    selection = set(map(composition_signature, bundle['teams']['validation']))
    endpoint = set(map(composition_signature, bundle['teams']['fresh-test']))
    assert not selection & endpoint


def test_freeze_hashes_membership_and_provenance_detect_tampering(tmp_path, config):
    sentinel = tmp_path / 'inputs/suites/mirror.json'
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b'legacy suite bytes stay unchanged\n')
    kwargs = dict(validation_teams=12, endpoint_teams=12, validation_battles=8, endpoint_battles=8)
    frozen = freeze_teacher_suites(tmp_path, PROJECT, config, [], **kwargs)
    assert sentinel.read_bytes() == b'legacy suite bytes stay unchanged\n'
    assert verify_frozen_teacher_suites(tmp_path, PROJECT, config, [], frozen) == frozen['inventory']
    with pytest.raises(ValueError, match='destination already exists'):
        freeze_teacher_suites(tmp_path, PROJECT, config, [], **kwargs)
    extra = tmp_path / 'inputs/teams/validation/extra.json'
    extra.write_text('{}')
    with pytest.raises(ValueError, match='membership changed'):
        verify_frozen_teacher_suites(tmp_path, PROJECT, config, [], frozen)
    extra.unlink()
    path = tmp_path / frozen['suites']['fresh-reversed']
    original = path.read_bytes()
    path.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='input changed'):
        verify_frozen_teacher_suites(tmp_path, PROJECT, config, [], frozen)
    path.write_bytes(original)
    changed = deepcopy(frozen)
    changed['inventory']['selection_role'] = 'independent test'
    with pytest.raises(ValueError, match='provenance changed'):
        verify_frozen_teacher_suites(tmp_path, PROJECT, config, [], changed)


@pytest.mark.parametrize('kwargs', [dict(validation_battles=6), dict(endpoint_battles=7),
    dict(validation_battles=True), dict(max_steps=0), dict(validation_teams=35)])
def test_bad_counts_are_rejected_without_rounding_or_silent_pool_reduction(config, kwargs):
    with pytest.raises(ValueError):
        build_teacher_suites(PROJECT, config, [], **kwargs)
