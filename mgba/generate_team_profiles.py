#!/usr/bin/env python3
"""Generate competitive/casual/random JSON teams in size directories 1–6."""
import argparse
from pathlib import Path

from firered_rl.team_profile_generation import generate_profiles, load_profile_catalog, write_profiles


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=150, help="Teams per category/size directory (default: 150)")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--level", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing files are never replaced")
    args = parser.parse_args()
    try:
        catalog, templates, roles = load_profile_catalog(root / "config/generation")
        count = write_profiles(args.output_dir, generate_profiles(catalog, templates, roles,
                              count=args.count, seed=args.seed, level=args.level))
    except (OSError, ValueError) as error:
        parser.exit(1, f"generate_team_profiles: {error}\n")
    print(f"Generated {count} teams in 18 directories: {args.output_dir}")
    print("Every Pokémon has four distinct moves. These JSONs are inputs to build_states.py.")


if __name__ == "__main__":
    main()
