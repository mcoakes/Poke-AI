#!/usr/bin/env python3
"""Audit a saved checkpoint's move-slot sensitivity on an outcome-unfiltered Rules sample."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from firered_rl.loss_review import prepare_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--matchups', type=int, default=32, help='1–100 Rules seat pairs, sampled regardless of outcome (default: 32)')
    parser.add_argument('--seed', type=int, default=261002, help='Sample seed, not battle RNG (default: 261002)')
    parser.add_argument('--environments', type=int, default=2)
    parser.add_argument('--showdown-workers', choices=('thread','process'), default='process')
    parser.add_argument('--node', default='node')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--log-max-mb', type=float, default=2)
    parser.add_argument('--log-backups', type=int, default=1)
    args = parser.parse_args()
    if args.environments < 2 or args.environments % 2 or not math.isfinite(args.log_max_mb) or args.log_max_mb <= 0 or args.log_backups < 0:
        parser.error('use even positive environments, a finite positive log cap, and nonnegative backups')
    try:
        selection = prepare_review(args.evaluation_dir, args.output_dir, matchups=args.matchups, seed=args.seed, sample_all=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    run = args.output_dir.expanduser().resolve()
    command = [sys.executable, str(Path(__file__).with_name('evaluate_ppo.py')),
               '--checkpoint', str(run/'inputs/candidate.zip'), '--suite', str(run/'inputs/suite.json'),
               '--reward-config', str(run/'inputs/reward.json'), '--move-order-audit',
               '--environments', str(args.environments), '--showdown-workers', args.showdown_workers,
               '--node', args.node, '--device', args.device, '--battle-log', '--no-readable-log', '--no-step-logs',
               '--log-max-mb', str(args.log_max_mb), '--log-backups', str(args.log_backups),
               '--output-dir', str(run/'replay')]
    if subprocess.run(command).returncode:
        return 1
    rows = {r['episode']:r for r in map(json.loads, (run/'replay/episodes.jsonl').read_text().splitlines())}
    mismatches = [c['episode'] for c in selection['cases'] if
                  any(rows.get(c['episode'], {}).get(key) != c[expected] for key, expected in
                      [('candidate_outcome','expected_outcome'),('truncated','expected_truncated'),('decisions','expected_decisions')])]
    verification = dict(verified=not mismatches, mismatches=mismatches, cases=len(selection['cases']),
                        candidate_sha256=selection['candidate_sha256'], source_suite_sha256=selection['source_suite_sha256'])
    (run/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    heading = f"Replay verification: {'PASS' if not mismatches else 'MISMATCH — investigate before interpreting'}; {len(selection['cases'])} battles.\n\n"
    report = run/'report.md'
    report.write_text(heading+(run/'replay/move-order-audit.md').read_text())
    print(f'Report: {report}')
    return 1 if mismatches else 0


if __name__ == '__main__':
    raise SystemExit(main())
