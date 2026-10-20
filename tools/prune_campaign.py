#!/usr/bin/env python3
"""Preview or apply conservative cleanup of inactive curriculum campaign data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firered_rl.retention import (apply_campaign_retention, campaign_retention_lock,
                                  plan_campaign_retention)


def count(value):
    number = int(value)
    if number < -1:
        raise argparse.ArgumentTypeError("must be -1 (all), zero, or a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing campaign directory; must be inactive")
    parser.add_argument("--keep-attempts", type=count, default=2,
                        help="Keep candidate checkpoints from latest N evaluated attempts per stage (-1: all; default: 2)")
    parser.add_argument("--keep-best", type=count, default=2,
                        help="Also keep top N evaluated candidate checkpoints per stage/rule cohort by win rate (-1: all; default: 2)")
    parser.add_argument("--keep-debug-attempts", type=count, default=0,
                        help="Keep detailed logs from latest N attempts per stage (-1: all; default: 0); newest unfinished attempt always kept")
    parser.add_argument("--apply", action="store_true", help="Delete previewed files; omission is read-only")
    args = parser.parse_args()
    try:
        with campaign_retention_lock(args.run_dir):
            print("Scanning campaign metadata and diagnostic files; large runs may take a while...", flush=True)
            plan = plan_campaign_retention(args.run_dir, keep_attempts=args.keep_attempts,
                    keep_best=args.keep_best, keep_debug_attempts=args.keep_debug_attempts)
            print(f"{'APPLY' if args.apply else 'DRY RUN'}: {plan['run_dir']}")
            for category, values in plan["totals"].items():
                print(f"  {category}: {values['files']:,} files; {values['bytes']:,} bytes")
            print(f"Total: {plan['files']:,} files; {plan['bytes']:,} bytes ({plan['bytes']/1024**3:.3f} GiB)")
            print("Inputs, promoted models, resume checkpoints, compact evaluation results and PPO summaries are retained.")
            for action in plan["actions"][:10]:
                print(f"  {action['path']}: {action['files']:,} files / {action['bytes']:,} bytes")
                for sample in action["sample"][:2]:
                    print(f"    {sample}")
            if len(plan["actions"]) > 10:
                print(f"  ... {len(plan['actions'])-10} more cleanup targets")
            for warning in plan["warnings"][:10]:
                print("Warning: " + warning)
            if args.apply:
                result = apply_campaign_retention(plan, assume_locked=True)
                print(f"Removed {result['files']:,} files / {result['bytes']:,} bytes; audit: retention-history.jsonl")
            else:
                print("No campaign data deleted. Add --apply to perform this cleanup.")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"Campaign cleanup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
