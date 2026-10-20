"""Reproducible, configuration-driven Gen III curriculum pools.

Compositions (species multisets) have one global split owner. Labels and party /
move order do not count toward diversity. Individual moves are checked against
our bundled learnsets; this is not a tournament or event-combination validator.
Abilities are the simulator's species default, not configurable in TeamSpec.
"""
import copy
import hashlib
import json
from pathlib import Path
import random

from .teams import PokemonSpec, TeamSpec

SPLITS = ("train", "validation", "heldout", "benchmark")


def _key(value):
    return "".join(c for c in str(value).lower() if c.isalnum())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def semantic_signature(team):
    """Ignore cosmetic team labels, member order, and move order."""
    mons = []
    for mon in team["party"]:
        value = dict(mon)
        value["species"] = _key(value["species"])
        value["moves"] = sorted(_key(move) for move in value["moves"])
        value["held_item"] = _key(value.get("held_item", "None"))
        value["nature"] = _key(value.get("nature", "Hardy"))
        mons.append(value)
    return _digest(sorted(mons, key=lambda value: json.dumps(value, sort_keys=True)))


def composition_signature(team):
    return "/".join(sorted(_key(mon["species"]) for mon in team["party"]))


def _write(path, value):
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"pinned curriculum asset differs: {path}")
    if not path.exists():
        path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def _load_config(config):
    if isinstance(config, (str, Path)):
        return json.loads(Path(config).read_text())
    return copy.deepcopy(config)


def _catalog(project, config):
    path = Path(project) / config["catalog"]
    catalog = json.loads(path.read_text())
    species = {_key(name): name for name in catalog["species"]}
    moves = {_key(name): name for name in catalog["moves"]}
    items = {_key(name): name for name in catalog["items"]}
    return catalog, species, moves, items


def _validate_mon(mon, catalog, species, moves, items):
    mon = copy.deepcopy(mon)
    try:
        mon["species"] = species[_key(mon["species"])]
        mon["moves"] = [moves[_key(move)] for move in mon["moves"]]
        mon["held_item"] = items[_key(mon.get("held_item", "None"))]
    except KeyError as exc:
        raise ValueError(f"unknown species, move or item in {mon}") from exc
    permitted = set(catalog["species"][mon["species"]]["moves"])
    illegal = set(mon["moves"]) - permitted
    if illegal:
        raise ValueError(f"{mon['species']}: moves absent from bundled learnset: {sorted(illegal)}")
    PokemonSpec.from_dict(mon)
    return mon


def _validate_config(config, catalog, species, moves, items):
    if config.get("version") != 1:
        raise ValueError("team curriculum config version must be 1")
    fractions = config["split_fractions"]
    if set(fractions) != set(SPLITS) or any(v <= 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError("split_fractions must contain positive train/validation/heldout/benchmark summing to 1")
    for name, pool in config["species_pools"].items():
        if len(set(map(_key, pool))) != len(pool) or not pool:
            raise ValueError(f"empty or duplicate species in pool {name}")
        if any(_key(value) not in species for value in pool):
            raise ValueError(f"unknown species in pool {name}")
    for name, pool in config["move_pools"].items():
        if len(set(map(_key, pool))) != len(pool) or not pool:
            raise ValueError(f"empty or duplicate moves in pool {name}")
        if any(_key(value) not in moves for value in pool):
            raise ValueError(f"unknown move in pool {name}")
    ids = [stage["id"] for stage in config["stages"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate curriculum stage id")
    for stage in config["stages"]:
        if not stage["id"] or "/" in stage["id"] or "\\" in stage["id"] or stage["id"] in (".", ".."):
            raise ValueError("unsafe curriculum stage id")
        if type(stage["team_size"]) is not int or not 1 <= stage["team_size"] <= 6:
            raise ValueError("team_size must be 1..6")
        if stage["team_size"] > len(config["species_pools"][stage["species_pool"]]):
            raise ValueError("species pool is smaller than team_size")
        if set(stage["counts"]) != set(SPLITS) or any(type(n) is not int or n < (0 if split == "benchmark" else 2) for split, n in stage["counts"].items()):
            raise ValueError("each stage needs at least two teams in every split")
        spec = stage["moves"]
        if not 1 <= spec["attacks"] <= spec["count"] <= 4:
            raise ValueError("moves: require 1 <= attacks <= count <= 4")
        for field in ("attack_pool", "support_pool"):
            if spec[field] not in config["move_pools"]:
                raise ValueError(f"unknown {field}")
        attacks = config["move_pools"][spec["attack_pool"]]
        support = config["move_pools"][spec["support_pool"]]
        if any(catalog["moves"][moves[_key(m)]]["power"] <= 1 for m in attacks):
            raise ValueError("attack pool must contain direct damaging moves with power > 1")
        if any(catalog["moves"][moves[_key(m)]]["power"] != 0 for m in support):
            raise ValueError("support pool must contain status moves (power 0)")
        if not 0 <= stage.get("curated_fraction", 0) <= 1:
            raise ValueError("curated_fraction must be between 0 and 1")
        build = config["builds"][stage["build_mode"]]
        if not all(build.get(k) for k in ("natures", "items", "ev_spreads")):
            raise ValueError("build requires nonempty natures/items/ev_spreads")
        for nature in build["natures"]:
            for evs in build["ev_spreads"]:
                PokemonSpec.from_dict(dict(species="check", level=50, moves=["check"], nature=nature, evs=evs))
        if any(_key(item) not in items for item in build["items"]):
            raise ValueError("unknown item in build")


def _fixed_teams(config, catalog, species, moves, items):
    result = {}
    for name, entry in config["fixed_stages"].items():
        if len(entry["teams"]) != 2:
            raise ValueError(f"fixed stage {name} requires two teams")
        result[name] = []
        for team in entry["teams"]:
            if not 1 <= len(team["party"]) <= 6:
                raise ValueError("fixed team must have 1..6 Pokemon")
            result[name].append(dict(name=team["name"], party=[
                _validate_mon(mon, catalog, species, moves, items) for mon in team["party"]]))
    return result


def write_fixed_assets(project, output_inputs, config):
    """Write JSON-defined introductory fixed teams; return staged-lab-compatible entries."""
    config = _load_config(config)
    catalog, species, moves, items = _catalog(project, config)
    root = Path(output_inputs).resolve()
    result = {}
    for name, pair in _fixed_teams(config, catalog, species, moves, items).items():
        base = Path("curriculum-teams") / name / "train"
        entry = dict(root=str(root), fixed=True, train_dir=str(base), train_paths=[],
                     holdout_dir=None, holdout_paths=[], sha256={}, pair_sampling="fixed")
        for i, team in enumerate(pair):
            relative = str(base / f"p{i+1}.json")
            entry["sha256"][relative] = _write(root / relative, team)
            entry["train_paths"].append(relative)
        result[name] = entry
    return result


def _split_for(composition, config, reserved):
    if composition in reserved:
        return "train"
    value = int(_digest([config["seed"], "composition-split", composition])[:16], 16) / 2**64
    cumulative = 0.
    for split in SPLITS:
        cumulative += config["split_fractions"][split]
        if value < cumulative:
            return split
    return "benchmark"


def _move_options(name, stage, config, catalog, moves):
    info = catalog["species"][name]
    permitted = set(info["moves"])
    spec = stage["moves"]
    attacks = [moves[_key(m)] for m in config["move_pools"][spec["attack_pool"]]
               if moves[_key(m)] in permitted]
    support = [moves[_key(m)] for m in config["move_pools"][spec["support_pool"]]
               if moves[_key(m)] in permitted]
    stab = [m for m in attacks if catalog["moves"][m]["type"] in info["types"]
            and catalog["moves"][m]["power"] >= spec.get("min_stab_power", 40)]
    if not stab:
        raise ValueError(f"{stage['id']} / {name}: configured attacks have no useful STAB")
    if len(attacks) < spec["attacks"] or len(support) < spec["count"] - spec["attacks"]:
        raise ValueError(f"{stage['id']} / {name}: not enough learnable moves in configured pools")
    return attacks, support, stab


def _pokemon(name, stage, config, catalog, moves, items, rng, curated):
    spec = stage["moves"]
    attacks, support, stab = _move_options(name, stage, config, catalog, moves)
    selected = [rng.choice(stab)]
    remaining = [m for m in attacks if m not in selected]
    if spec.get("simple_stab_normal"):
        remaining = [m for m in remaining if catalog["moves"][m]["type"] == "Normal"]
        if not remaining:
            raise ValueError(f"{name}: simple stage needs a distinct Normal attack")
    else:
        coverage = [m for m in remaining if catalog["moves"][m]["type"] != catalog["moves"][selected[0]]["type"]]
        if spec["attacks"] > 1 and not coverage:
            raise ValueError(f"{name}: configured attacks lack type coverage")
        if spec["attacks"] > 1:
            selected.append(rng.choice(coverage))
            remaining = [m for m in remaining if m not in selected]
    selected += rng.sample(remaining, spec["attacks"] - len(selected))
    selected += rng.sample(support, spec["count"] - len(selected))
    build = config["builds"][stage["build_mode"]]
    # Match offensive spreads to the selected STAB's Gen III damage category;
    # defensive spreads remain available. Nature is kept aligned with that spread.
    spreads = build["ev_spreads"]
    category = catalog["moves"][selected[0]]["category"]
    preferred = [v for v in spreads if not (v.get("atk", 0) >= 200 and category == "special")
                 and not (v.get("spa", 0) >= 200 and category == "physical")]
    evs = copy.deepcopy(rng.choice(preferred or spreads))
    nature_pool = build["natures"]
    if evs.get("atk", 0) >= 200:
        nature_pool = [n for n in nature_pool if n in ("Adamant", "Jolly", "Brave", "Hardy")] or nature_pool
    elif evs.get("spa", 0) >= 200:
        nature_pool = [n for n in nature_pool if n in ("Modest", "Timid", "Quiet", "Hardy")] or nature_pool
    item_pool = [item for item in build["items"] if _key(item) != "chestoberry" or "Rest" in selected]
    mon = dict(species=name, level=50, nature=rng.choice(nature_pool),
               held_item=items[_key(rng.choice(item_pool or build["items"]))], moves=selected,
               evs=evs, ivs={stat: 31 for stat in ("hp", "atk", "def", "spa", "spd", "spe")})
    choices = curated.get(name, ())
    if choices and rng.random() < stage.get("curated_fraction", 0):
        template = rng.choice(choices)
        mon.update(moves=list(template["moves"]), nature=template["nature"],
                   evs=copy.deepcopy(template["evs"]), held_item=rng.choice(template["items"]))
    rng.shuffle(mon["moves"])
    return mon


def _diversity(teams):
    party = [mon for team in teams for mon in team["party"]]
    return dict(teams=len(teams), semantic_teams=len({semantic_signature(t) for t in teams}),
                compositions=len({composition_signature(t) for t in teams}),
                species=len({m["species"] for m in party}), moves=len({v for m in party for v in m["moves"]}),
                movesets=len({(m["species"], tuple(sorted(m["moves"]))) for m in party}),
                natures=len({m["nature"] for m in party}), items=len({m["held_item"] for m in party}))


def build_pools(project, output_inputs, config):
    """Build stages 04..10 with immutable split files and input-relative paths.

    Result is JSON-ready, keyed by stage id. Each entry has root, *_dir, *_paths,
    sha256, semantic_signatures/composition_signatures per split, and diversity.
    heldout and holdout fields are aliases for existing staged-lab callers.
    """
    project, root = Path(project).resolve(), Path(output_inputs).resolve()
    config = _load_config(config)
    catalog, species, moves, items = _catalog(project, config)
    _validate_config(config, catalog, species, moves, items)
    fixed = _fixed_teams(config, catalog, species, moves, items)
    reserved = {composition_signature(team) for pair in fixed.values() for team in pair}
    curated = {}
    curated_path = project / config["curated_sets"]
    for template in json.loads(curated_path.read_text())["sets"]:
        candidate = dict(species=template["species"], level=50, moves=template["moves"],
                         evs=template["evs"], nature=template["nature"], held_item=template["items"][0])
        try:
            candidate = _validate_mon(candidate, catalog, species, moves, items)
        except ValueError:
            continue
        # Curated teams need an effective STAB and at least two direct attacks.
        attacks = [m for m in candidate["moves"] if catalog["moves"][m]["power"] > 1]
        if len(attacks) < 2 or not any(catalog["moves"][m]["type"] in catalog["species"][candidate["species"]]["types"]
                                     and catalog["moves"][m]["power"] >= 40 for m in attacks):
            continue
        legal_items = [items[_key(item)] for item in template["items"] if _key(item) in items]
        if legal_items:
            curated.setdefault(candidate["species"], []).append(dict(moves=candidate["moves"],
                evs=candidate["evs"], nature=candidate["nature"], items=legal_items))
    result = {}
    for stage in config["stages"]:
        sid = stage["id"]
        base = Path("curriculum-teams") / sid
        entry = dict(root=str(root), fixed=False, sha256={}, diversity={}, semantic_signatures={},
                     composition_signatures={}, pair_sampling="two_distinct_pool_entries_without_replacement",
                     split_policy="global_species_composition_hash", ability_policy="natural_species_default",
                     catalog_sha256=hashlib.sha256((project / config["catalog"]).read_bytes()).hexdigest(),
                     curated_sha256=hashlib.sha256(curated_path.read_bytes()).hexdigest())
        pool = [species[_key(n)] for n in config["species_pools"][stage["species_pool"]]]
        for name in pool:
            _move_options(name, stage, config, catalog, moves)
        for split in SPLITS:
            rng = random.Random(int(_digest([config["seed"], sid, split]), 16))
            generated, signatures, compositions = [], set(), set()
            entry[f"{split}_dir"] = str(base / split)
            entry[f"{split}_paths"] = []
            count = stage["counts"][split]
            (root / base / split).mkdir(parents=True, exist_ok=True)
            attempts = 0
            while len(generated) < count:
                attempts += 1
                if attempts > max(20000, count * 1000):
                    raise ValueError(f"{sid}/{split}: cannot create requested unique compositions; expand the species pool")
                names = rng.sample(pool, stage["team_size"])
                composition = "/".join(sorted(map(_key, names)))
                if composition in compositions or _split_for(composition, config, reserved) != split:
                    continue
                team = dict(name=f"{sid}-{split}-{len(generated)+1:04d}", party=[
                    _pokemon(name, stage, config, catalog, moves, items, rng, curated) for name in names])
                team["party"] = [_validate_mon(mon, catalog, species, moves, items) for mon in team["party"]]
                signature = semantic_signature(team)
                if signature in signatures:
                    continue
                signatures.add(signature)
                compositions.add(composition)
                generated.append(team)
            for i, team in enumerate(generated):
                relative = str(base / split / f"team-{i+1:04d}.json")
                entry["sha256"][relative] = _write(root / relative, team)
                entry[f"{split}_paths"].append(relative)
            entry["diversity"][split] = _diversity(generated)
            entry["semantic_signatures"][split] = sorted(signatures)
            entry["composition_signatures"][split] = sorted(compositions)
        entry["holdout_dir"] = entry["heldout_dir"]
        entry["holdout_paths"] = list(entry["heldout_paths"])
        result[sid] = entry
    _write(root / "curriculum-teams" / "index.json", dict(config_sha256=_digest(config), stages=result))
    return result


def validate_pools(assets):
    """Reject mutation or extra JSON teams in any pinned broad curriculum pool."""
    for entry in assets.values():
        root = Path(entry["root"])
        for split in SPLITS:
            actual = set((root / entry[f"{split}_dir"]).glob("*.json"))
            expected = {root / p for p in entry[f"{split}_paths"]}
            if actual != expected:
                raise ValueError(f"pinned pool membership changed: {entry[f'{split}_dir']}")
        for relative, expected in entry["sha256"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"pinned team changed: {relative}")
