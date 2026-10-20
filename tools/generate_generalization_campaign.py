#!/usr/bin/env python3
"""Build reproducible groups of Gen III teams for a generalization curriculum."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

from firered_rl.curriculum import load_curriculum, party_signature
from firered_rl.team_generation import load_catalog
from firered_rl.teams import STATS, TeamSpec

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("01_basic_types", "Three-member parties: mono-type species, STAB and Normal attacks, no held items."),
    ("02_immunities", "Three-member parties: more mono-types, including Ground, Ghost and Normal immunities."),
    ("03_dual_types", "Three-member parties: dual types, resistances and compounded weaknesses."),
    ("04_coverage", "Three- and four-member parties: off-type coverage; species alone no longer predicts attacks."),
    ("05_status", "Four-member parties: paralysis, poison, burns or sleep alongside damaging moves."),
    ("06_recovery", "Four-member parties: recovery, Protect, residual damage and HP management."),
    ("07_setup", "Five-member parties: setup, Substitute, Encore, Taunt and stat management."),
    ("08_field_control", "Five-member parties: weather, hazards, Rapid Spin, phazing and Choice Band."),
    ("09_battle_roles", "Six-member parties: curated attacking, defensive and support sets together."),
    ("10_broad_transfer", "Parties of one to six: broader species, learnable moves, varied stats and held items."),
]
RESERVED_SPECIES = {"Arbok", "Armaldo", "Bellossom", "Cradily", "Delcatty", "Donphan", "Granbull",
                    "Lanturn", "Mantine", "Masquerain", "Octillery", "Politoed", "Sandslash", "Shiftry", "Xatu"}
LEGENDARIES = {"Articuno", "Zapdos", "Moltres", "Mewtwo", "Mew", "Raikou", "Entei", "Suicune",
              "Lugia", "Ho Oh", "Celebi", "Regirock", "Regice", "Registeel", "Latias", "Latios",
              "Kyogre", "Groudon", "Rayquaza", "Jirachi", "Deoxys"}
BASIC_TYPES = {"Fire", "Water", "Grass", "Electric", "Ice", "Fighting"}
# Introduce timing/locking mechanics after the initial type lessons.
COMPLEX_ATTACKS = {"Hidden Power", "Weather Ball", "Explosion", "Self Destruct", "Hyper Beam", "Solar Beam",
    "Fly", "Dig", "Dive", "Bounce", "Sky Attack", "Skull Bash", "Razor Wind", "Focus Punch", "Blast Burn",
    "Hydro Cannon", "Frenzy Plant", "Outrage", "Thrash", "Petal Dance", "Rollout", "Ice Ball", "Fury Cutter",
    "Future Sight", "Doom Desire", "Dream Eater", "Spit Up", "Present", "Magnitude", "Frustration"}
STATUS = {"Thunder Wave", "Toxic", "Will O Wisp", "Sleep Powder", "Hypnosis", "Stun Spore", "Spore"}
RECOVERY = {"Recover", "Soft Boiled", "Slack Off", "Milk Drink", "Moonlight", "Morning Sun", "Synthesis",
            "Rest", "Wish", "Protect", "Leech Seed"}
SETUP = {"Swords Dance", "Calm Mind", "Bulk Up", "Dragon Dance", "Agility", "Growth", "Curse",
         "Substitute", "Encore", "Taunt", "Haze"}
FIELD = {"Spikes", "Rapid Spin", "Roar", "Whirlwind", "Rain Dance", "Sunny Day", "Sandstorm", "Hail"}


def attack_pool(species, catalog, *, broad=False):
    return [m for m in catalog["species"][species]["moves"]
            if catalog["moves"][m]["power"] >= 35 and catalog["moves"][m]["category"] != "status"
            and (broad or m not in COMPLEX_ATTACKS)]


def member(species, stage, rng, catalog, curated):
    info = catalog["species"][species]
    moves = attack_pool(species, catalog, broad=stage >= 8)
    stab = [m for m in moves if catalog["moves"][m]["type"] in info["types"]]
    chosen = []
    if stab:
        chosen.append(rng.choice(stab))
    if stage <= 3:
        allowed = [m for m in moves if catalog["moves"][m]["type"] in (*info["types"], "Normal")]
        chosen += rng.sample([m for m in allowed if m not in chosen], 4-len(chosen))
    else:
        coverage = [m for m in moves if catalog["moves"][m]["type"] not in info["types"]]
        if coverage:
            chosen.append(rng.choice(coverage))
        focus = (STATUS if stage == 5 else RECOVERY if stage == 6 else SETUP if stage == 7
                 else FIELD if stage == 8 else set())
        support = sorted(set(info["moves"]) & focus)
        if support:
            move = rng.choice(support)
            if move not in chosen:
                chosen.append(move)
        filler = [m for m in moves if m not in chosen]
        chosen += rng.sample(filler, 4-len(chosen))
    item, nature = "None", "Hardy"
    evs = dict.fromkeys(STATS, 0)
    if stage >= 4:
        physical = sum(catalog["moves"][m]["category"] == "physical" for m in chosen) >= 2
        attack = "atk" if physical else "spa"
        nature = ("Adamant" if physical else "Modest")
        evs[attack], evs["hp" if stage in (6, 7) else "spe"] = 252, 252
        evs["spd"] = 4
    if stage >= 5:
        item = rng.choice(["Leftovers", "Lum Berry"])
    if stage >= 8 and all(catalog["moves"][m]["category"] == "physical" for m in chosen):
        item = rng.choice(["Choice Band", "Leftovers"])
    if stage >= 9 and species in curated and (stage == 9 or rng.random() < .6):
        template = rng.choice(curated[species])
        chosen = list(template["moves"])
        nature, evs, item = template["nature"], deepcopy(template["evs"]), rng.choice(template["items"])
    if stage == 10 and species not in curated:
        choices = [m for m in info["moves"] if m not in {"Sketch", "Struggle", "Hidden Power"}]
        # Keep an attack available; the remaining slots can include unusual legal moves.
        chosen = [rng.choice(moves)]
        chosen += rng.sample([m for m in choices if m not in chosen], 3)
    rng.shuffle(chosen)
    ivs = {s: (rng.randint(20, 31) if stage == 10 else 31) for s in STATS}
    if stage == 10 and rng.random() < .25:
        evs = dict.fromkeys(STATS, 0)
    return dict(species=species, level=50, moves=chosen, held_item=item, nature=nature, ivs=ivs, evs=evs)


def generate(output, *, seed=20260918, train_count=24, validation_count=12, holdout_count=40):
    output = Path(output)
    if any(n < 1 for n in (train_count, validation_count)) or holdout_count < 2:
        raise ValueError("team counts must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty; existing campaigns are never overwritten")
    catalog, sets = load_catalog(ROOT / "config/generation")
    curated = {}
    for entry in sets:
        curated.setdefault(entry["species"], []).append(entry)
    eligible = sorted(s for s, info in catalog["species"].items() if len(attack_pool(s, catalog)) >= 4)
    normal = [s for s in eligible if s not in RESERVED_SPECIES]
    seen = set()
    manifest = dict(version=1, name="pokemon_generalization", opponent_mode="previous-stage",
                    opponent_switch=dict(chance=0, cooldown=2), stages=[], next_team_dir="holdout")
    audit = []

    def make_group(folder, stage, count, group_seed, split):
        rng = random.Random(group_seed)
        folder.mkdir(parents=True)
        pool = [s for s in normal if stage == 10 or catalog["species"][s]["fully_evolved"]]
        if stage < 10:
            pool = [s for s in pool if s not in LEGENDARIES]
        if stage <= 2:
            pool = [s for s in pool if len(catalog["species"][s]["types"]) == 1]
        if stage == 1:
            pool = [s for s in pool if catalog["species"][s]["types"][0] in BASIC_TYPES]
        if stage <= 3:
            pool = [s for s in pool if len([m for m in attack_pool(s, catalog)
                    if catalog["moves"][m]["type"] in (*catalog["species"][s]["types"], "Normal")]) >= 4]
        if stage == 9:
            pool = [s for s in pool if s in curated]
        group_species = set()
        for index in range(count):
            if split == "unseen_species":
                pool = sorted(s for s in eligible if s in RESERVED_SPECIES)
            size = (3 if stage <= 3 else rng.choice([3, 4]) if stage == 4 else 4 if stage <= 6
                    else 5 if stage <= 8 else 6 if stage == 9 else index % 6 + 1)
            for _ in range(2000):
                species = rng.sample(pool, size)
                # Early parties span three different primary types; later sets span roles.
                if stage <= 3 and len({catalog["species"][s]["types"][0] for s in species}) < 3:
                    continue
                party = [member(s, stage, rng, catalog, curated) for s in species]
                if stage in (5, 6, 7, 8):
                    required = {5: STATUS, 6: RECOVERY, 7: SETUP, 8: FIELD}[stage]
                    if not any(required.intersection(p["moves"]) for p in party):
                        continue
                rng.shuffle(party)
                data = dict(name=f"Generalization {stage:02d} {split} {index+1:03d}", party=party)
                target = folder / f"team_{index+1:03d}.json"
                target.write_text(json.dumps(data, indent=2)+"\n")
                signature = party_signature(target)
                if signature not in seen:
                    seen.add(signature)
                    break
            else:
                raise ValueError("unable to construct enough distinct teams")
            TeamSpec.load(target)
            group_species.update(species)
        audit.append(dict(directory=str(folder.relative_to(output)), teams=count,
                          species=sorted(group_species)))

    for stage, (slug, focus) in enumerate(STAGES, 1):
        base = Path("stages") / slug
        make_group(output / base / "train", stage, train_count, seed+stage*1000, "train")
        make_group(output / base / "validation", stage, validation_count, seed+stage*1000+501, "validation")
        manifest["stages"].append(dict(stage=stage, team_dir=str(base / "train"),
                                      validation_dir=str(base / "validation"), focus=focus))
    # Half remixed sets and half reserved species: never part of the training or promotion pools.
    make_group(output / "holdout/recombinations", 10, (holdout_count+1)//2, seed+90001, "recombination")
    make_group(output / "holdout/unseen_species", 10, holdout_count//2, seed+90002, "unseen_species")
    (output / "curriculum.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (output / "generation.json").write_text(json.dumps(dict(seed=seed, train_count=train_count,
        validation_count=validation_count, holdout_count=holdout_count, groups=audit,
        reserved_species=sorted(RESERVED_SPECIES)), indent=2)+"\n")
    load_curriculum(output / "curriculum.json", grouped=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty campaign config directory")
    parser.add_argument("--seed", type=int, default=20260918, help="Generation seed (default: 20260918)")
    parser.add_argument("--train-count", type=int, default=24, help="Training teams per stage (default: 24)")
    parser.add_argument("--validation-count", type=int, default=12, help="Validation teams per stage (default: 12)")
    parser.add_argument("--holdout-count", type=int, default=40, help="Final test teams (default: 40; at least 2)")
    args = parser.parse_args()
    if args.holdout_count < 2:
        parser.error("--holdout-count must be at least 2")
    try:
        generate(args.output_dir, seed=args.seed, train_count=args.train_count,
                 validation_count=args.validation_count, holdout_count=args.holdout_count)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"campaign generation: {exc}\n")
    print(f"Created ten stage groups: {args.output_dir / 'curriculum.json'}")


if __name__ == "__main__":
    main()
