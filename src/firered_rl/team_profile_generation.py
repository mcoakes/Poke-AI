"""Three reproducible Gen III roster families with exactly four moves per Pokémon."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random

from .team_generation import (COMMON_ITEMS, EXCLUDE_RANDOM_ITEMS, SPECIFIC_ITEMS,
                              common_item, load_catalog, write_batch)
from .team_profiles import CATEGORIES
from .teams import NATURES, STATS, PokemonSpec


def load_profile_catalog(directory):
    catalog, templates = load_catalog(directory)
    options = json.loads((directory / "team_profiles_gen3.json").read_text())
    templates = templates + options["additional_sets"]
    for entry in templates:
        name = entry["species"]
        if len(set(entry["moves"])) != 4 or not set(entry["moves"]) <= set(catalog["species"][name]["moves"]):
            raise ValueError(f"curated set must have four species-learnable moves: {name}")
        if not set(entry["items"]) <= COMMON_ITEMS or name not in options["roles"]:
            raise ValueError(f"missing role or unsupported item: {name}")
        PokemonSpec.from_dict({k: v for k, v in entry.items() if k != "items"} | {"level": 50})
    return catalog, templates, options["roles"]


def random_evs(rng, chunks):
    result = dict.fromkeys(STATS, 0)
    for _ in range(chunks):
        stat = rng.choice([s for s in STATS if result[s] < 252])
        result[stat] += 4
    return result


def random_item(rng, species, catalog):
    pool = sorted(name for name, entry in catalog["items"].items()
                  if entry["battle_effect"] and name not in COMMON_ITEMS | EXCLUDE_RANDOM_ITEMS
                  and (name not in SPECIFIC_ITEMS or species in SPECIFIC_ITEMS[name]))
    return rng.choice(pool)


def casual_moves(rng, species, catalog):
    choices = sorted(set(catalog["species"][species]["moves"]) - {"Struggle", "Sketch"})
    attacks = [m for m in choices if catalog["moves"][m]["category"] != "status"]
    ordinary = [m for m in attacks if 40 <= catalog["moves"][m]["power"] <= 100]
    stab = [m for m in ordinary if catalog["moves"][m]["type"] in catalog["species"][species]["types"]]
    picked = [rng.choice(stab or ordinary or attacks or choices)]
    coverage = [m for m in (ordinary or attacks) if m not in picked
                and catalog["moves"][m]["type"] != catalog["moves"][picked[0]]["type"]]
    if coverage:
        picked.append(rng.choice(coverage))
    support = [m for m in choices if m not in picked and catalog["moves"][m]["category"] == "status"]
    if support:
        picked.append(rng.choice(support))
    picked += rng.sample([m for m in choices if m not in picked], 4 - len(picked))
    return picked


def competitive_party(rng, templates, roles, catalog, size, archetype):
    patterns = {
        "balanced": ("physical", "special", "wall", "utility", "physical", "special"),
        "offense": ("physical", "special", "physical", "special", "utility", "physical"),
        "bulky": ("wall", "special", "wall", "physical", "utility", "special"),
    }
    pattern = list(patterns[archetype])
    # Small teams still contain attackers; singleton rosters cover every role.
    if size == 1:
        pattern = [rng.choice(("physical", "special", "wall", "utility"))]
    party, used, types = [], set(), Counter()
    for role in pattern[:size]:
        candidates = [t for t in templates if t["species"] not in used
                      and not ("Baton Pass" in t["moves"] and any("Baton Pass" in p["moves"] for p in party))]
        candidates = [t for t in candidates if roles[t["species"]] == role] or candidates
        weights = [1 / (1 + sum(types[t] for t in catalog["species"][entry["species"]]["types"])) ** 2
                   for entry in candidates]
        entry = deepcopy(rng.choices(candidates, weights=weights, k=1)[0])
        entry["held_item"] = rng.choice(entry.pop("items"))
        entry["ivs"] = dict.fromkeys(STATS, 31)
        party.append(entry)
        used.add(entry["species"])
        types.update(catalog["species"][entry["species"]]["types"])
    rng.shuffle(party)  # Vary the lead as well as the roster.
    return party


def generate_profiles(catalog, templates, roles, *, count=150, seed=20260913, level=50):
    """Yield one independently seeded batch for each category/size directory."""
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("--count must be between 1 and 10000 (teams per size directory)")
    if type(level) is not int or not 1 <= level <= 100:
        raise ValueError("--level must be between 1 and 100")
    all_species = sorted(catalog["species"])
    all_moves = sorted(set(catalog["moves"]) - {"Struggle", "Sketch"})
    learnable = [s for s in all_species if len(set(catalog["species"][s]["moves"]) & set(all_moves)) >= 4]
    for category_index, category in enumerate(CATEGORIES):
        for size in range(1, 7):
            batch_seed = seed + category_index * 1000003 + size * 1009
            rng = random.Random(batch_seed)
            common_slots = set(rng.sample(range(count * size), round(.8 * count * size)))
            teams, seen = [], set()
            for index in range(count):
                archetype = ("balanced", "offense", "bulky")[index % 3] if category == "competitive" else category
                for attempt in range(1000):
                    if category == "competitive":
                        party = competitive_party(rng, templates, roles, catalog, size, archetype)
                    else:
                        species = rng.sample(learnable if category == "casual" else all_species, size)
                        party = []
                        for slot, name in enumerate(species):
                            moves = casual_moves(rng, name, catalog) if category == "casual" else rng.sample(all_moves, 4)
                            member = dict(species=name, moves=moves, nature=rng.choice(NATURES),
                                          ivs={s: rng.randrange(32) for s in STATS},
                                          evs=random_evs(rng, rng.randint(20, 100) if category == "casual" else rng.randrange(128)))
                            member["held_item"] = (common_item(rng, member, catalog) if index * size + slot in common_slots
                                                   else random_item(rng, name, catalog))
                            party.append(member)
                    for member in party:
                        member["level"] = level
                        rng.shuffle(member["moves"])
                        PokemonSpec.from_dict(member)
                        assert len(set(member["moves"])) == 4
                    identity = json.dumps(party, sort_keys=True)
                    if identity not in seen:
                        seen.add(identity)
                        break
                else:
                    raise ValueError(f"could not generate {count} distinct {category}/{size} teams")
                teams.append(dict(name=f"{category}_{size}_{index + 1:03d}", party=party,
                                  generation=dict(category=category, team_size=size, seed=seed,
                                                  batch_seed=batch_seed, archetype=archetype)))
            species_counts = Counter(p["species"] for t in teams for p in t["party"])
            report = dict(category=category, team_size=size, seed=seed, batch_seed=batch_seed,
                          teams=count, pokemon=count * size, level=level,
                          species_counts=dict(sorted(species_counts.items())),
                          common_item_count=sum(p["held_item"] in COMMON_ITEMS for t in teams for p in t["party"]),
                          four_moves=True)
            yield Path(category) / str(size), teams, report


def write_profiles(directory, batches):
    directory = Path(directory)
    # Generate and validate every batch before writing any of them.
    batches = list(batches)
    if directory.exists():
        raise FileExistsError(f"output directory already exists: {directory}; choose a new --output-dir")
    directory.mkdir(parents=True)
    for relative, teams, report in batches:
        write_batch(teams, report, directory / relative)
    return sum(len(teams) for _, teams, _ in batches)
