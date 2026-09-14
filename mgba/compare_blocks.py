#!/usr/bin/env python3
"""Compare non-overlapping battle blocks from a current or older PPO run."""
import argparse
import json
from pathlib import Path

from firered_rl.battle_blocks import BattleBlocks, format_block


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='PPO run containing episodes.jsonl')
    parser.add_argument('--block-size', type=positive_int, default=500, help='Ended episodes per block (default: 500)')
    parser.add_argument('--json', action='store_true', help='Print a JSON array instead of readable rows')
    args = parser.parse_args()
    blocks = BattleBlocks(args.block_size)
    results = []
    try:
        with (args.run_dir / 'episodes.jsonl').open() as stream:
            for line in stream:
                # The writer can still be appending its final record in a live run.
                if not line.endswith('\n'):
                    break
                event = json.loads(line)
                result = blocks.add(event)
                if result is not None:
                    results.append(result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if blocks.rows:
        results.append(blocks.snapshot())
    if args.json:
        print(json.dumps(results, indent=2))
    elif results:
        for row in results:
            print(format_block(row))
    else:
        print('No ended battles recorded yet.')


if __name__ == '__main__':
    main()
