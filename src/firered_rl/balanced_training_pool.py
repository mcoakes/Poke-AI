"""Deterministic, new simple teams with exact species/party-position balance.

This prepares a finite canonical training pool; it does not alter runtime team
sampling or shuffle benches. All prior source teams and suite assets must be
supplied in old_teams, including mirror rehearsal and development checks.
"""
from collections import Counter, deque
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import random

from .broad_team_curriculum import (
    _catalog, _validate_config, _validate_mon, _pokemon, _move_options,
    composition_signature, semantic_signature,
)


VERSION = 1
STAGE = '04-type-matchups'
INTRO_SPECIES = frozenset({'Typhlosion', 'Blastoise', 'Sceptile', 'Arcanine',
    'Vaporeon', 'Venusaur', 'Raichu', 'Golem', 'Machamp', 'Alakazam', 'Dodrio', 'Lapras'})
STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')


def _recipe(project, config):
    config = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else deepcopy(config)
    catalog, species, moves, items = _catalog(project, config)
    _validate_config(config, catalog, species, moves, items)
    stages = [stage for stage in config['stages'] if stage['id'] == STAGE]
    if len(stages) != 1:
        raise ValueError('Balanced pool requires exactly one stage 04-type-matchups recipe')
    stage = stages[0]
    if (stage['team_size'] != 3 or stage['species_pool'] != 'intro'
            or stage['build_mode'] != 'simple' or stage['moves']['count'] != 2
            or stage['moves']['attacks'] != 2 or not stage['moves'].get('simple_stab_normal')
            or stage.get('curated_fraction', 0) != 0 or stage['moves'].get('min_stab_power', 40) != 40):
        raise ValueError('Balanced pool requires the original simple 3v3 two-attack stage recipe')
    pool = sorted(config['species_pools'][stage['species_pool']])
    if len(pool) != 12 or set(pool) != INTRO_SPECIES:
        raise ValueError('Balanced pool requires the original 12-species intro pool')
    expected_build = dict(natures=['Hardy'], items=['None'], ev_spreads=[dict.fromkeys(STATS, 0)])
    if config['builds']['simple'] != expected_build:
        raise ValueError('Balanced pool requires the original simple build recipe')
    return config, stage, pool, catalog, species, moves, items


def _exclusions(old_teams):
    return dict(compositions=sorted({composition_signature(team) for team in old_teams}),
        semantics=sorted({semantic_signature(team) for team in old_teams}), input_teams=len(old_teams))


def _count(count):
    if type(count) is not int or count < 12 or count % 12:
        raise ValueError('Team count must be a positive multiple of 12 for exact species/position balance')


def _balanced_compositions(edges, count, rng):
    """Bounded deterministic swaps toward an exactly regular 3-uniform pool."""
    target = count // 4
    if count > len(edges) or any(sum(i in edge for edge in edges) < target for i in range(12)):
        raise ValueError('Insufficient eligible compositions for the requested exact species balance')
    edge_sets = [frozenset(edge) for edge in edges]
    for restart in range(32):
        selected = set(rng.sample(range(len(edges)), count))
        degree = Counter(i for index in selected for i in edges[index])
        for step in range(201):
            if all(degree[i] == target for i in range(12)):
                return [edges[index] for index in sorted(selected)], dict(restart=restart, swaps=step)
            if step == 200:
                break
            best, candidates = 0, []
            for remove in sorted(selected):
                for add in range(len(edges)):
                    if add in selected:
                        continue
                    minus = edge_sets[remove] - edge_sets[add]
                    plus = edge_sets[add] - edge_sets[remove]
                    delta = (sum(-2 * (degree[i] - target) + 1 for i in minus)
                        + sum(2 * (degree[i] - target) + 1 for i in plus))
                    if delta < best:
                        best, candidates = delta, [(remove, add, minus, plus)]
                    elif delta == best:
                        candidates.append((remove, add, minus, plus))
            if not candidates:
                break
            remove, add, minus, plus = rng.choice(candidates)
            selected.remove(remove)
            selected.add(add)
            for i in minus:
                degree[i] -= 1
            for i in plus:
                degree[i] += 1
    raise ValueError('Could not construct exact species balance within the bounded search; no smaller pool was substituted')


def _assign_slot(edges, quota):
    """Unit-flow matching from each team to one member, with species capacity.

For a regular pool this matching exists: each remaining team has d edges and
each species has d*quota appearances. Any subset of species therefore touches
at least quota*subset_size teams. Applying the matching three times balances
every party position without random runtime permutations.
"""
    size = len(edges)
    source, first_species, sink = 0, size + 1, size + 13
    graph = [[] for _ in range(sink + 1)]
    capacity = {}

    def connect(a, b, amount):
        graph[a].append(b)
        graph[b].append(a)
        capacity[a, b] = amount
        capacity[b, a] = 0

    for team, species in enumerate(edges, start=1):
        connect(source, team, 1)
        for index in sorted(species):
            connect(team, first_species + index, 1)
    for index in range(12):
        connect(first_species + index, sink, quota)
    for _ in range(size):
        previous, queue = {source: None}, deque([source])
        while queue and sink not in previous:
            node = queue.popleft()
            for other in graph[node]:
                if other not in previous and capacity[node, other] > 0:
                    previous[other] = node
                    queue.append(other)
        if sink not in previous:
            raise ValueError('Exact party-position matching is infeasible')
        node = sink
        while previous[node] is not None:
            parent = previous[node]
            capacity[parent, node] -= 1
            capacity[node, parent] += 1
            node = parent
    assignments = []
    for team, species in enumerate(edges, start=1):
        chosen = [index for index in species if capacity[team, first_species + index] == 0]
        if len(chosen) != 1:
            raise RuntimeError('Invalid party-position matching')
        assignments.append(chosen[0])
    return assignments


def validate_pool(project, config, teams, old_teams, *, count=None):
    """Validate recipes, exclusions, unique compositions and all three slots."""
    teams, old_teams = list(teams), list(old_teams)
    count = len(teams) if count is None else count
    _count(count)
    if len(teams) != count:
        raise ValueError('Generated team count differs from requested count')
    config, stage, pool, catalog, species, moves, items = _recipe(project, config)
    excluded = _exclusions(old_teams)
    forbidden_compositions, forbidden_semantics = set(excluded['compositions']), set(excluded['semantics'])
    compositions, semantics, ledger = set(), set(), []
    total = Counter(dict.fromkeys(pool, 0))
    slots = [Counter(dict.fromkeys(pool, 0)) for _ in range(3)]
    for team in teams:
        if len(team['party']) != 3:
            raise ValueError('Balanced teams must contain exactly three Pokemon')
        names = []
        for index, raw in enumerate(team['party']):
            mon = _validate_mon(raw, catalog, species, moves, items)
            if (mon['species'] not in INTRO_SPECIES or mon.get('level') != 50
                    or mon.get('nature') != 'Hardy' or mon.get('held_item') != 'None'
                    or mon.get('evs') != dict.fromkeys(STATS, 0)
                    or mon.get('ivs') != dict.fromkeys(STATS, 31)
                    or len(mon['moves']) != 2 or len(set(mon['moves'])) != 2):
                raise ValueError('Generated Pokemon differs from the simple stage recipe')
            attacks, _, stab = _move_options(mon['species'], stage, config, catalog, moves)
            first, second = mon['moves']
            if (any(move not in attacks for move in mon['moves'])
                    or not ((first in stab and catalog['moves'][second]['type'] == 'Normal')
                        or (second in stab and catalog['moves'][first]['type'] == 'Normal'))):
                raise ValueError('Generated moves must be one useful STAB and a distinct Normal attack')
            names.append(mon['species'])
            total[mon['species']] += 1
            slots[index][mon['species']] += 1
        if len(set(names)) != 3:
            raise ValueError('A generated team repeats a species')
        composition, semantic = composition_signature(team), semantic_signature(team)
        if composition in forbidden_compositions or semantic in forbidden_semantics:
            raise ValueError('Generated team overlaps an excluded source composition or semantic team')
        if composition in compositions or semantic in semantics:
            raise ValueError('Generated teams repeat a composition or semantic team')
        compositions.add(composition)
        semantics.add(semantic)
        ledger.append(dict(name=team.get('name'), composition=composition, semantic=semantic,
            party=names, moves=[list(mon['moves']) for mon in team['party']]))
    if any(total[name] != count // 4 for name in pool):
        raise ValueError('Generated pool does not have exact species balance')
    if any(slot[name] != count // 12 for slot in slots for name in pool):
        raise ValueError('Generated pool does not have exact party-position balance')
    return dict(version=VERSION, stage=STAGE, teams=count, species_pool=pool,
        species_counts=dict(total), slot_counts={str(i): dict(slot) for i, slot in enumerate(slots)},
        compositions=sorted(compositions), semantics=sorted(semantics), team_ledger=ledger,
        exclusions=excluded)


def generate(project, config, old_teams, *, count=96, seed=261130):
    """Return teams and a JSON-ready ledger without creating any files."""
    _count(count)
    if type(seed) is not int:
        raise ValueError('Seed must be an integer')
    old_teams = list(old_teams)
    config, stage, pool, catalog, species, moves, items = _recipe(project, config)
    excluded = _exclusions(old_teams)
    forbidden = set(excluded['compositions'])
    edges = [edge for edge in combinations(range(12), 3)
        if composition_signature(dict(party=[dict(species=pool[i]) for i in edge])) not in forbidden]
    edges, search = _balanced_compositions(edges, count, random.Random(seed))
    remaining = [set(edge) for edge in edges]
    orders = [[] for _ in edges]
    for _ in range(3):
        chosen = _assign_slot(remaining, count // 12)
        for order, available, value in zip(orders, remaining, chosen):
            order.append(value)
            available.remove(value)
    # Composition repair never changes the move/build RNG stream.
    build_rng = random.Random(seed ^ 0xBA1A4CED)
    teams = [dict(name=f'balanced-training-{index+1:04}', party=[
        _validate_mon(_pokemon(pool[i], stage, config, catalog, moves, items, build_rng, {}),
                      catalog, species, moves, items) for i in order])
        for index, order in enumerate(orders)]
    metadata = validate_pool(project, config, teams, old_teams, count=count)
    eligible = sum(composition_signature(dict(party=[dict(species=pool[i]) for i in edge])) not in forbidden
        for edge in combinations(range(12), 3))
    metadata.update(seed=seed, eligible_compositions=eligible, composition_search=search,
        recipe='Existing simple stage-4 generator; 3v3, familiar 12 species, two attacks, Hardy, no item, zero EVs, 31 IVs. Canonical order fixed; every species occurs equally in each party position.')
    return teams, metadata
