#!/usr/bin/env python3
"""Compare two best checkpoints on new teams using their fixed opponent pool."""
import argparse
import math
from pathlib import Path
import subprocess
import sys


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True, help="Reference PPO run with best.zip and periodic validation")
    parser.add_argument("--candidate-run", type=Path, required=True, help="New PPO run with best.zip and the same validation suite/rewards")
    parser.add_argument("--team-dir", type=Path, required=True, help="New team JSON directory, searched recursively")
    parser.add_argument("--exclude-team-dir", type=Path, action="append", default=[], help="Additional previously used team directory; repeatable")
    parser.add_argument("--exclude-suite", type=Path, action="append", default=[], help="Additional previously used evaluation suite; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True, help="New comparison directory; existing directories are never overwritten")
    parser.add_argument("--episodes-per-opponent", type=positive, default=1000, help="Even battles per category, including both seats (default: 1000)")
    parser.add_argument("--seed", type=int, default=26092703, help="New test schedule seed (default: 26092703)")
    parser.add_argument("--environments", type=positive, default=48, help="Even player count (default: 48)")
    parser.add_argument("--showdown-workers", choices=("thread", "process"), default="process")
    parser.add_argument("--node", default="node", help="Node executable (default: node)")
    parser.add_argument("--device", default="cpu", help="Policy inference device (default: cpu)")
    parser.add_argument("--torch-threads", type=positive, default=1)
    parser.add_argument("--battle-log", action=argparse.BooleanOptionalAction, default=True, help="Keep bounded spectator logs (default: enabled)")
    parser.add_argument("--log-max-mb", type=float, default=2, help="Positive per-log rotation limit in MiB (default: 2)")
    parser.add_argument("--log-backups", type=int, default=1, help="Rotated files per log, at least 0 (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and overlaps without writing files or running battles")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    try:
        if args.episodes_per_opponent % 2 or args.environments % 2:
            raise ValueError("episodes-per-opponent and environments must be even")
        if not math.isfinite(args.log_max_mb) or args.log_max_mb <= 0 or args.log_backups < 0:
            raise ValueError("log-max-mb must be positive and finite; log-backups must be nonnegative")
        if output.exists():
            raise ValueError(f"output already exists; use a new directory: {output}")
        for source in (args.previous_run, args.candidate_run, args.team_dir, *args.exclude_team_dir):
            if output.is_relative_to(source.expanduser().resolve()):
                raise ValueError("output must be outside input run and team directories")
        from firered_rl.generalization import prepare, publish
        plan = prepare(args.previous_run, args.candidate_run, args.team_dir,
                       exclude_dirs=args.exclude_team_dir, exclude_suites=args.exclude_suite)
        print(f"Teams: {len(plan['teams'])} accepted; {len(plan['excluded'])} overlapping/duplicate files excluded.", flush=True)
        print(f"Each model: {3 * args.episodes_per_opponent} battles; Random/Dumb/Smart equally weighted; both player seats.", flush=True)
        if args.dry_run:
            print("Checks passed. No files written; no battles started.")
            return 0
        models = publish(plan, output, episodes_per_opponent=args.episodes_per_opponent, seed=args.seed)
        script = Path(__file__).resolve().with_name("evaluate_ppo.py")
        for label, model in models.items():
            print(f"Evaluating {label}: {model['source']}", flush=True)
            command = [sys.executable, str(script), "--checkpoint", model["path"],
                       "--suite", str(output / "suite.json"),
                       "--reward-config", str(output / "inputs/reward.json"),
                       "--output-dir", str(output / label),
                       "--environments", str(args.environments), "--showdown-workers", args.showdown_workers,
                       "--node", args.node, "--device", args.device, "--torch-threads", str(args.torch_threads),
                       "--no-readable-log", "--no-step-logs", "--keep-episode-logs", "0",
                       "--battle-log" if args.battle_log else "--no-battle-log",
                       "--log-max-mb", str(args.log_max_mb), "--log-backups", str(args.log_backups)]
            subprocess.run(command, check=True)
        from firered_rl.baseline_report import build_baseline_report, write_baseline_report
        report = build_baseline_report({label: {"new_teams": output / label} for label in models},
                                       starting_label="previous", seed=args.seed)
        _, report_path = write_baseline_report(report, output)
        print(f"Comparison complete: {report_path}\nTeam overlap audit: {output / 'team-check.json'}")
        return 0
    except KeyboardInterrupt:
        print("Comparison interrupted; partial results retained. Use a new output directory to retry.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Generalization comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
