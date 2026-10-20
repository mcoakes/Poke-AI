#!/usr/bin/env python3
"""Combine saved validation suites, preserving cases and bundling pinned opponents."""
import argparse
import json
from pathlib import Path

from firered_rl.combined_suite import combine_suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suites', type=Path, nargs='+', required=True,
                        help='Two or more distinct, side-swapped Showdown suite.json files; opponents must be pinned')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for suite.json, manifest.json and verified opponent copies; never overwritten')
    args = parser.parse_args()
    try:
        path, manifest = combine_suites(args.suites, args.output_dir,
                                       project=Path(__file__).resolve().parents[1])
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f'Combined suite: {path}')
    print(f"Battles: {manifest['battles']}; groups: {json.dumps(manifest['by_opponent_group'], sort_keys=True)}")
    print('No battles run. Use this file with evaluate_ppo.py --suite or train_ppo.py --eval-suite.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
