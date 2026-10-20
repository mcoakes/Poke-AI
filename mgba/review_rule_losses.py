#!/usr/bin/env python3
"""Sample saved losses against Rules, replay both seats, and summarize candidate decisions."""
import argparse
from pathlib import Path
import subprocess
import sys

from firered_rl.loss_review import prepare_review, report_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True, help='Saved deterministic v2 evaluation with suite, config, episodes and candidate snapshot')
    parser.add_argument('--output-dir', type=Path, required=True, help='New review directory; never overwritten')
    parser.add_argument('--matchups', type=int, default=12, help='Number of loss-containing seat pairs, 1–100 (default: 12; capped by available pairs)')
    parser.add_argument('--seed', type=int, default=261001, help='Loss sample seed only; original battle seeds are preserved')
    parser.add_argument('--environments', type=int, default=2, help='Even concurrent player count (default: 2 for easy sequential logs)')
    parser.add_argument('--showdown-workers', choices=('thread', 'process'), default='process')
    parser.add_argument('--node', default='node', help='Node executable (default: node)')
    parser.add_argument('--device', default='cpu', help='Inference device (default: cpu)')
    parser.add_argument('--log-max-mb', type=float, default=2, help='Positive per-file log cap in MiB (default: 2)')
    parser.add_argument('--log-backups', type=int, default=1, help='Rotated backups per log (default: 1)')
    args = parser.parse_args()
    import math
    if args.environments < 2 or args.environments % 2:
        parser.error('--environments must be positive and even')
    if not math.isfinite(args.log_max_mb) or args.log_max_mb <= 0 or args.log_backups < 0:
        parser.error('use a finite positive log cap and nonnegative backups')
    try:
        selection = prepare_review(args.evaluation_dir, args.output_dir, matchups=args.matchups, seed=args.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    run = args.output_dir.expanduser().resolve()
    print(f"Reviewing {selection['selected_matchups']} matchups; checkpoint hash verified. No training.", flush=True)
    command = [sys.executable, str(Path(__file__).with_name('evaluate_ppo.py')),
               '--checkpoint', str(run/'inputs/candidate.zip'), '--suite', str(run/'inputs/suite.json'),
               '--reward-config', str(run/'inputs/reward.json'), '--decision-audit',
               '--environments', str(args.environments), '--showdown-workers', args.showdown_workers,
               '--node', args.node, '--device', args.device, '--rule-log', '--readable-log', '--battle-log',
               '--no-step-logs', '--log-max-mb', str(args.log_max_mb), '--log-backups', str(args.log_backups),
               '--output-dir', str(run/'replay')]
    result = subprocess.run(command)
    if result.returncode:
        print(f'Replay failed; partial diagnostics: {run}/replay', file=sys.stderr)
        return 1
    report = report_review(run)
    print(f"Report: {run / 'review.md'}")
    return 0 if report['replay_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
