#!/usr/bin/env python3
"""Prepare disjoint train/validation/test teams and reproducible launch commands."""
import argparse
import math
from pathlib import Path
import sys


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="Showdown PPO run with best.zip, pinned frozen opponent and periodic validation")
    parser.add_argument("--output-dir", type=Path, required=True, help="New experiment directory; never overwrites an existing directory")
    parser.add_argument("--train-teams", type=positive, default=3000)
    parser.add_argument("--validation-teams", type=positive, default=500)
    parser.add_argument("--test-teams", type=positive, default=500)
    parser.add_argument("--team-size", type=int, choices=range(1, 7), default=3)
    parser.add_argument("--seed", type=int, default=26092801, help="Generation/split seed; training retains the source run's base seed")
    parser.add_argument("--episodes-per-opponent", type=positive, default=1000, help="Even battle count per category in each validation/test suite")
    parser.add_argument("--total-timesteps", type=positive, default=500000, help="Additional training budget written to train.sh")
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--environments", type=positive, default=64)
    parser.add_argument("--exclude-team-dir", type=Path, action="append", default=[], help="Other previously used team directory; recursive and repeatable")
    parser.add_argument("--exclude-suite", type=Path, action="append", default=[], help="Other previously used Showdown suite; repeatable")
    args = parser.parse_args()
    try:
        if sum((args.train_teams, args.validation_teams, args.test_teams)) > 10000:
            raise ValueError("at most 10,000 teams total per experiment")
        if args.environments % 2 or args.episodes_per_opponent % 2:
            raise ValueError("environments and episodes-per-opponent must be even")
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("learning-rate must be positive and finite")
        from firered_rl.diversity_experiment import setup
        manifest = setup(Path(__file__).resolve().parents[1], args.source_run, args.output_dir,
                         counts=(args.train_teams, args.validation_teams, args.test_teams),
                         team_size=args.team_size, seed=args.seed, episodes=args.episodes_per_opponent,
                         timesteps=args.total_timesteps, learning_rate=args.learning_rate, environments=args.environments,
                         exclude_dirs=args.exclude_team_dir, exclude_suites=args.exclude_suite)
        print(f"Prepared {manifest['counts']}; rejected generation candidates: {manifest['rejected']}")
        print(f"Manifest: {args.output_dir}/experiment.json\nTrain: bash {args.output_dir}/train.sh")
        print("No training or evaluation has started. Reserve test.sh until checkpoint selection is finished.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Experiment preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
