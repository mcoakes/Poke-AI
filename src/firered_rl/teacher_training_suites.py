"""Disjoint, balanced selection and endpoint teams for teacher-initialized PPO.

The caller supplies every historical training and evaluation team as exclusions.
Selection is a new validation pool, not independent generalization evidence.
The separate fresh endpoint pool is frozen before fitting and never selects.
No training data, legacy suite, policy, or reward is modified here.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from .balanced_training_pool import generate, validate_pool
from .evaluation import digest, validate_suite
from .replacement_diagnostics import transform_suite
from .rule_opponent import RuleBasedOpponent


VERSION = 1
SUITE_NAMES = ('validation', 'fresh-original', 'fresh-reversed')
VALIDATION_ROLE = 'Broader validation used for checkpoint selection; not an independent test.'
ENDPOINT_ROLE = 'Frozen fresh endpoint; never used for fitting, checkpoint selection or early stopping.'


def _positive_multiple(value, multiple, name):
    if type(value) is not int or value < multiple or value % multiple:
        raise ValueError(f'{name} must be a positive multiple of {multiple}')


def _canonical_suite(teams, episodes, seed, max_steps, identity, role):
    """Exact budget, cyclic team coverage and paired candidate seats.

Production budgets cover every team repeatedly. Small smoke budgets deliberately
exercise only a schedule prefix; metadata reports that incomplete team coverage.
Candidate seats are paired in every case, including the smoke prefix.
"""
    _positive_multiple(episodes, 2, 'canonical battle count')
    assets = {digest(dict(team=team)): dict(team=deepcopy(team)) for team in teams}
    keys = list(assets)
    if len(keys) != len(teams) or len(keys) < 2:
        raise ValueError('A fresh suite requires at least two distinct teams')
    rng, battle_rng = random.Random(seed), random.Random(seed ^ 0x5C017)
    cases, exposures = [], Counter(dict.fromkeys(keys, 0))
    for index in range(episodes // 2):
        position = index % len(keys)
        if position == 0:
            order = list(range(len(keys)))
            rng.shuffle(order)
            shift = rng.randrange(1, len(keys))
        pair = [keys[order[position]], keys[order[(position + shift) % len(keys)]]]
        battle_seed = battle_rng.randrange(2**31 - 1)
        for seat in (1, 2):
            exposures[pair[seat - 1]] += 1
            cases.append(dict(episode=len(cases) + 1, matchup=index + 1,
                assets=list(pair), seed=battle_seed, candidate_player=seat))
    return validate_suite(dict(version=1, backend='showdown', seed=seed,
        swap_sides=True, max_steps=max_steps, assets=assets, cases=cases,
        rule_opponent=deepcopy(identity), diagnostic=dict(role=role,
            schedule='full-pool-cycles-with-paired-seats', actual_battles=episodes,
            pool_teams=len(teams), balanced_candidate_seats=True,
            candidate_team_exposures=dict(exposures),
            complete_team_coverage=all(exposures.values()),
            full_cycles_per_order=(episodes // 2) // len(keys))))


def _combined_validation(original, reversed_suite):
    """Preserve paired seeds/seats while giving each bench order equal weight."""
    combined = deepcopy(original)
    combined['assets'].update(deepcopy(reversed_suite['assets']))
    cases = combined['cases']
    offset = len(cases) // 2
    pairs = []
    for left, right in zip(original['cases'], reversed_suite['cases']):
        if any(left[key] != right[key] for key in ('episode', 'matchup', 'seed', 'candidate_player')):
            raise ValueError('Original and reversed validation schedules differ')
        case = deepcopy(right)
        case['episode'] = len(cases) + 1
        case['matchup'] += offset
        cases.append(case)
        pairs.append(dict(original_episode=left['episode'], reversed_episode=case['episode'],
            seed=left['seed'], candidate_player=left['candidate_player'],
            original_matchup=left['matchup']))
    combined['swap_sides'] = False  # Reversal changes the candidate asset in each seat.
    exposures = Counter(case['assets'][case['candidate_player'] - 1] for case in cases)
    combined['diagnostic'].update(role=VALIDATION_ROLE, actual_battles=len(cases),
        bench_orders=['original', 'candidate-bench-reversed'], equal_order_weight=True,
        candidate_team_exposures=dict(exposures),
        exposure_scope='Actual asset keys; canonical and reversed candidate orders have separate entries.',
        paired_orders=pairs, canonical_suite_sha256=digest(original),
        reversed_suite_sha256=digest(reversed_suite))
    return validate_suite(combined)


def _reverse(suite):
    result = transform_suite(suite)
    result['diagnostic']['candidate_team_exposures'] = dict(Counter(
        case['assets'][case['candidate_player'] - 1] for case in result['cases']))
    return result


def build_teacher_suites(project, config, excluded_teams, *, validation_teams=36,
        endpoint_teams=24, validation_battles=2048, endpoint_battles=1000,
        validation_seed=261140, endpoint_seed=261141, max_steps=100):
    """Return JSON-ready teams, suites and provenance without filesystem writes.

    Both generated pools balance each of the familiar 12 species exactly in
    each party position. Exclusions use unordered species composition, so names,
    move changes and party reversal cannot bypass the split. Endpoint generation
    additionally excludes the entire new validation pool.
    """
    _positive_multiple(validation_battles, 4, 'validation battle count')
    _positive_multiple(endpoint_battles, 2, 'endpoint battle count')
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    config = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else deepcopy(config)
    excluded_teams = list(excluded_teams)
    validation, selection_inventory = generate(project, config, excluded_teams,
        count=validation_teams, seed=validation_seed)
    endpoint, endpoint_inventory = generate(project, config, [*excluded_teams, *validation],
        count=endpoint_teams, seed=endpoint_seed)
    for label, teams, inventory in (('validation', validation, selection_inventory),
                                   ('fresh-test', endpoint, endpoint_inventory)):
        for index, (team, row) in enumerate(zip(teams, inventory['team_ledger']), 1):
            team['name'] = row['name'] = f'teacher-{label}-{index:04d}'
    identity = RuleBasedOpponent(project).identity
    canonical = _canonical_suite(validation, validation_battles // 2,
        validation_seed, max_steps, identity, VALIDATION_ROLE)
    original = _canonical_suite(endpoint, endpoint_battles,
        endpoint_seed, max_steps, identity, ENDPOINT_ROLE)
    suites = dict(validation=_combined_validation(canonical, _reverse(canonical)),
        **{'fresh-original': original, 'fresh-reversed': _reverse(original)})
    inventory = dict(version=VERSION, selection_role=VALIDATION_ROLE,
        endpoint_role=ENDPOINT_ROLE, validation=selection_inventory, endpoint=endpoint_inventory,
        config_sha256=digest(config), catalog_sha256=hashlib.sha256(
            (Path(project) / config['catalog']).read_bytes()).hexdigest(),
        rule_opponent=identity,
        settings=dict(validation_teams=validation_teams, endpoint_teams=endpoint_teams,
            validation_battles=validation_battles, endpoint_battles=endpoint_battles,
            validation_seed=validation_seed, endpoint_seed=endpoint_seed, max_steps=max_steps),
        suite_sha256={name: digest(suite) for name, suite in suites.items()},
        exclusions_scope='All supplied source training old/new/mirror teams and every historical suite asset; new validation also excluded from fresh endpoint.',
        inference_scope='Familiar 12 species, simple 3v3, two attacks per Pokemon; not unrestricted human play.')
    return dict(teams={'validation': validation, 'fresh-test': endpoint}, suites=suites, inventory=inventory)


def _payloads(bundle):
    payloads, teams, suites = {}, {}, {}
    for group, values in bundle['teams'].items():
        teams[group] = []
        for index, team in enumerate(values, 1):
            relative = f'inputs/teams/{group}/team-{index:04d}.json'
            teams[group].append(relative)
            payloads[relative] = team
    for name, suite in bundle['suites'].items():
        relative = f'inputs/suites/{name}.json'
        suites[name] = relative
        payloads[relative] = suite
    payloads['inputs/teacher-suite-inventory.json'] = bundle['inventory']
    return payloads, teams, suites


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def freeze_teacher_suites(root, project, config, excluded_teams, **settings):
    """Write new immutable inputs and return relative maps, inventory and pins.

    Existing destinations are rejected before any write. The caller owns legacy
    suite copies and can merge these pins into its wider experiment manifest.
    """
    root = Path(root)
    bundle = build_teacher_suites(project, config, excluded_teams, **settings)
    payloads, teams, suites = _payloads(bundle)
    if any((root / relative).exists() for relative in payloads):
        raise ValueError('Fresh teacher suite destination already exists')
    for group in teams:
        directory = root / 'inputs/teams' / group
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Fresh teacher team directory is not empty')
    pins = {}
    for relative, value in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _json_bytes(value)
        with path.open('xb') as handle:
            handle.write(data)
        pins[relative] = hashlib.sha256(data).hexdigest()
    return dict(teams=teams, suites=suites, inventory=bundle['inventory'], pins=pins)


def verify_frozen_teacher_suites(root, project, config, excluded_teams, frozen):
    """Check immutable contents, exact balance, split exclusion and schedule.

    Regeneration uses the recorded seeds/settings and current pinned recipe. It
    checks pool membership as well as hashes; adding a team cannot bypass pins.
    """
    root = Path(root)
    excluded_teams = list(excluded_teams)
    expected = build_teacher_suites(project, config, excluded_teams,
        **frozen['inventory']['settings'])
    payloads, teams, suites = _payloads(expected)
    if (frozen['teams'] != teams or frozen['suites'] != suites
            or frozen['inventory'] != expected['inventory'] or set(frozen['pins']) != set(payloads)):
        raise ValueError('Frozen teacher suite configuration or provenance changed')
    for relative, value in payloads.items():
        path = root / relative
        expected_hash = hashlib.sha256(_json_bytes(value)).hexdigest()
        if (not path.is_file() or frozen['pins'][relative] != expected_hash
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash):
            raise ValueError(f'Frozen teacher suite input changed: {relative}')
    for group, paths in teams.items():
        directory = root / 'inputs/teams' / group
        if {path.resolve() for path in directory.iterdir()} != {(root / p).resolve() for p in paths}:
            raise ValueError(f'Frozen teacher team membership changed: {group}')
    # Explicit invariant validation also documents the intended split contract.
    old = list(excluded_teams)
    validate_pool(project, config, expected['teams']['validation'], old,
        count=frozen['inventory']['settings']['validation_teams'])
    validate_pool(project, config, expected['teams']['fresh-test'],
        [*old, *expected['teams']['validation']], count=frozen['inventory']['settings']['endpoint_teams'])
    return expected['inventory']
