#!/usr/bin/env python3
"""Generate reproducible mixed Gen III team JSONs for build_states.py."""
import argparse
from pathlib import Path

from firered_rl.team_generation import generate, load_catalog, write_batch


def spread(value):
    result = {}
    try:
        for part in value.split(","):
            name, number = part.split("=")
            name = name.strip().lower()
            if name in result:
                raise ValueError("duplicate stat")
            result[name] = int(number)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use hp=252,def=252,spd=4; each stat only once") from error
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40, help="Number of team files (default: 40)")
    parser.add_argument("--team-size", type=int, default=6)
    parser.add_argument("--level", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--competitive-rate", type=float, default=.6)
    parser.add_argument("--chaos-rate", type=float, default=.2,
                        help="Unrestricted moves; remaining slots use species learnsets")
    parser.add_argument("--competitive-item-rate", type=float, default=.8)
    parser.add_argument("--species-pool", choices=["all", "fully-evolved", "kanto"], default="all")
    parser.add_argument("--iv-mode", choices=["perfect", "random"], default="perfect")
    parser.add_argument("--ev-mode", choices=["trained", "zero", "random"], default="trained")
    parser.add_argument("--evs", type=spread, help="Override EVs for every Pokemon; omitted stats become 0")
    parser.add_argument("--ivs", type=spread, help="Override IVs for every Pokemon; omitted stats become 31")
    parser.add_argument("--held-item", help="Override all held items; supersedes the 80%% allocation")
    parser.add_argument("--nature", help="Override the nature for every Pokemon")
    parser.add_argument("--prefix", default="team", help="Filename prefix; use a new one for separate state batches")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for generated JSONs")
    args = parser.parse_args()
    values = vars(args).copy()
    directory = values.pop("output_dir")
    values["overrides"] = {name: values.pop(name) for name in ("evs", "ivs", "held_item", "nature")}
    values["overrides"] = {name: value for name, value in values["overrides"].items() if value is not None}
    try:
        data, templates = load_catalog(root / "config/generation")
        teams, report = generate(data, templates, **values)
        write_batch(teams, report, directory)
    except (ValueError, OSError) as error:
        parser.exit(1, f"generate_teams: {error}\n")
    print(f"Generated {report['teams']} teams / {report['pokemon']} Pokemon: {directory}")
    print(f"Sets: {report['style_counts']}")
    print(f"Common competitive items: {report['common_item_count']}/{report['pokemon']}")
    print("Edit any team JSON, then pass this directory to build_states.py --team-dir.")


if __name__ == "__main__":
    main()
