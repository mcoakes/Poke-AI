"""Seeded Gen III training teams; no ROM, .NET or network needed for configs."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import re

from .teams import NATURES, STATS, PokemonSpec, stat_values

COMMON_ITEMS = {"Leftovers", "Choice Band", "Lum Berry", "Chesto Berry",
                "Salac Berry", "Petaya Berry", "Liechi Berry", "Light Ball", "Thick Club"}
SPECIAL_ITEMS = {"Pikachu": "Light Ball", "Cubone": "Thick Club", "Marowak": "Thick Club"}
EXCLUDE_RANDOM_ITEMS = {"Enigma Berry", "Macho Brace", "Lucky Egg", "Soothe Bell",
                        "Amulet Coin", "Cleanse Tag", "Smoke Ball", "Everstone"}
SPECIFIC_ITEMS = {"Soul Dew": {"Latias", "Latios"}, "Deep Sea Tooth": {"Clamperl"},
                  "Deep Sea Scale": {"Clamperl"}, "Metal Powder": {"Ditto"},
                  "Stick": {"Farfetchd", "Farfetch'd"}, "Lucky Punch": {"Chansey"}}


def key(value):
    return re.sub(r"[^a-z0-9]", "", value.lower().replace("♀", "f").replace("♂", "m"))


def load_catalog(directory):
    data = json.loads((directory / "firered_us_v1_0.json").read_text())
    sets = json.loads((directory / "competitive_gen3.json").read_text())["sets"]
    for entry in sets:
        name = entry["species"]
        if not set(entry["moves"]) <= set(data["species"][name]["moves"]):
            raise ValueError(f"competitive template has moves outside the bundled learnset: {name}")
        if not set(entry["items"]) <= COMMON_ITEMS:
            raise ValueError(f"competitive template item is outside the common pool: {name}")
        PokemonSpec.from_dict({k: v for k, v in entry.items() if k != "items"} | {"level": 50})
    return data, sets


def canonical(value, table, label):
    matches = [name for name in table if key(name) == key(value)]
    if len(matches) != 1:
        raise ValueError(f"unknown Gen III {label}: {value}")
    return matches[0]


def common_item(rng, pokemon, catalog):
    if pokemon["species"] in SPECIAL_ITEMS:
        return SPECIAL_ITEMS[pokemon["species"]]
    categories = [catalog["moves"][move]["category"] for move in pokemon["moves"]]
    if len(categories) >= 3 and all(category == "physical" for category in categories):
        return rng.choice(["Choice Band", "Leftovers", "Lum Berry"])
    if "Rest" in pokemon["moves"]:
        return rng.choice(["Chesto Berry", "Leftovers"])
    return rng.choice(["Leftovers", "Leftovers", "Lum Berry"])


def generate(catalog, templates, *, count=40, team_size=6, level=50, seed=42,
             competitive_rate=.6, chaos_rate=.2, competitive_item_rate=.8,
             species_pool="all", iv_mode="perfect", ev_mode="trained", overrides=None,
             prefix="team"):
    """Percentages are exact rounded allocations over all Pokémon, not all teams."""
    for label, value, low, high in (("count", count, 1, 10000), ("team_size", team_size, 1, 6),
                                     ("level", level, 1, 100)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{label} must be an integer between {low} and {high}")
    rates = (competitive_rate, chaos_rate, competitive_item_rate)
    if any(not math.isfinite(rate) or not 0 <= rate <= 1 for rate in rates):
        raise ValueError("rates must be between 0 and 1")
    if competitive_rate + chaos_rate > 1:
        raise ValueError("competitive-rate + chaos-rate cannot exceed 1")
    if competitive_item_rate < competitive_rate:
        raise ValueError("competitive-item-rate must cover all competitive sets")
    if species_pool not in ("all", "fully-evolved", "kanto"):
        raise ValueError("unknown species pool")
    if iv_mode not in ("perfect", "random") or ev_mode not in ("trained", "zero", "random"):
        raise ValueError("unknown IV/EV mode")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("prefix must contain only letters, digits, _ or -")
    overrides = deepcopy(overrides or {})
    if set(overrides) - {"evs", "ivs", "held_item", "nature"}:
        raise ValueError("overrides support evs, ivs, held_item and nature")
    if "held_item" in overrides:
        overrides["held_item"] = canonical(overrides["held_item"], catalog["items"], "item")
    # Explicit spreads replace the generated spread. Omitted IVs=31; omitted EVs=0.
    for label, default, maximum in (("evs", 0, 255), ("ivs", 31, 31)):
        if label in overrides:
            overrides[label] = dict(zip(STATS, stat_values(overrides[label], default=default,
                                        maximum=maximum, label=label)))
    if "nature" in overrides:
        overrides["nature"] = canonical(overrides["nature"], NATURES, "nature")
    eligible = sorted(name for name, data in catalog["species"].items()
                      if species_pool == "all" or (species_pool == "fully-evolved" and data["fully_evolved"])
                      or (species_pool == "kanto" and data["id"] <= 151))
    competitive = [entry for entry in templates if entry["species"] in eligible]
    if len(eligible) < team_size or (competitive_rate and len({s['species'] for s in competitive}) < team_size):
        raise ValueError("species pool too small for unique species per team")
    rng = random.Random(seed)
    total = count * team_size
    n_comp = int(total * competitive_rate + .5)
    n_chaos = min(total - n_comp, int(total * chaos_rate + .5))
    styles = ["competitive"] * n_comp + ["chaos"] * n_chaos + ["learnable_random"] * (total - n_comp - n_chaos)
    rng.shuffle(styles)
    common_slots = {i for i, style in enumerate(styles) if style == "competitive"}
    common_slots.update(rng.sample([i for i in range(total) if i not in common_slots],
                                  int(total * competitive_item_rate + .5) - n_comp))
    all_moves = sorted(set(catalog["moves"]) - {"Struggle", "Sketch"})
    teams = []
    for team_index in range(count):
        members = [None] * team_size
        metadata = [None] * team_size
        used = set()
        # Select curated species first so a random slot cannot exhaust their pool.
        order = sorted(range(team_size), key=lambda slot: styles[team_index * team_size + slot] != "competitive")
        for slot in order:
            offset = team_index * team_size + slot
            style = styles[offset]
            if style == "competitive":
                template = deepcopy(rng.choice([entry for entry in competitive if entry["species"] not in used]))
                items = template.pop("items")
                member = template | {"level": level}
                item = rng.choice(items)
            else:
                name = rng.choice([name for name in eligible if name not in used])
                choices = all_moves if style == "chaos" else sorted(set(catalog["species"][name]["moves"]) - {"Struggle", "Sketch"})
                if not choices:
                    raise ValueError(f"no usable moves in learnset: {name}")
                attacks = [move for move in choices if catalog["moves"][move]["category"] != "status"]
                # Put an attack first when the species has one, for simple baseline policies.
                first = rng.choice(attacks or choices)
                moves = [first] + rng.sample([move for move in choices if move != first], min(3, len(choices) - 1))
                member = {"species": name, "level": level, "moves": moves, "nature": rng.choice(NATURES)}
                member["evs"] = dict.fromkeys(STATS, 0)
                if ev_mode == "trained":
                    category = catalog["moves"][first]["category"]
                    offense = "atk" if category == "physical" else "spa"
                    member["evs"].update({offense: 252, "spe": 252, "hp": 4})
                item = common_item(rng, member, catalog)
            if offset not in common_slots:
                choices = sorted(name for name, data in catalog["items"].items()
                    if data["battle_effect"] and name not in COMMON_ITEMS | EXCLUDE_RANDOM_ITEMS
                    and (name not in SPECIFIC_ITEMS or member["species"] in SPECIFIC_ITEMS[name]))
                item = rng.choice(choices)
            member["held_item"] = item
            member["ivs"] = {stat: 31 if iv_mode == "perfect" else rng.randrange(32) for stat in STATS}
            if ev_mode == "zero":
                member["evs"] = dict.fromkeys(STATS, 0)
            elif ev_mode == "random":
                member["evs"] = dict.fromkeys(STATS, 0)
                # 508 EVs in four-point chunks, no stat above 252.
                for _ in range(127):
                    stat = rng.choice([stat for stat in STATS if member["evs"][stat] < 252])
                    member["evs"][stat] += 4
            member.update(deepcopy(overrides))
            PokemonSpec.from_dict(member)
            members[slot] = member
            metadata[slot] = {"slot": slot + 1, "style": style,
                              "common_item": member["held_item"] in COMMON_ITEMS}
            used.add(member["species"])
        teams.append({"name": f"{prefix}_{team_index + 1:03d}", "party": members,
                      "generation": {"seed": seed, "members": metadata}})
    counts = Counter(entry["style"] for team in teams for entry in team["generation"]["members"])
    item_counts = Counter(member["held_item"] for team in teams for member in team["party"])
    return teams, {"profile": catalog["profile"], "seed": seed, "teams": count, "pokemon": total,
        "style_counts": dict(counts), "common_item_count": sum(n for name, n in item_counts.items() if name in COMMON_ITEMS),
        "item_counts": dict(sorted(item_counts.items())), "overrides": overrides,
        "options": {"level": level, "team_size": team_size, "species_pool": species_pool,
                    "competitive_rate": competitive_rate, "chaos_rate": chaos_rate,
                    "competitive_item_rate": competitive_item_rate, "iv_mode": iv_mode, "ev_mode": ev_mode}}


def write_batch(teams, report, directory):
    """Refuse existing directories so an old batch is never mixed with a new one."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for team in teams:
        (directory / f"{team['name']}.json").write_text(json.dumps(team, indent=2) + "\n")
    # Outside *.json, so build_states --team-dir cannot mistake it for a team.
    report = report | {"files": [f"{team['name']}.json" for team in teams]}
    (directory / "generation-summary.txt").write_text(json.dumps(report, indent=2) + "\n")
    return report
