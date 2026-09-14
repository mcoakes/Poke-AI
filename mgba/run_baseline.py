#!/usr/bin/env python3
"""Run automated masked actions against one NPC/wild battle checkpoint."""
import argparse
from datetime import datetime
from pathlib import Path

from firered_rl.baseline import replay_actions, run_baseline
from firered_rl.turn_env import FireRedTurnEnv


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=positive, default=55356)
    parser.add_argument("--emulator-pid", type=positive, required=True)
    parser.add_argument("--episodes", type=positive, default=10)
    parser.add_argument("--max-steps", type=positive, default=200)
    parser.add_argument("--decision-timeout", type=positive, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--allow-escape", action="store_true")
    parser.add_argument("--moves-first", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(".local/run") / datetime.now().strftime("baseline-%Y%m%d-%H%M%S-%f"))
    parser.add_argument("--replay-log", type=Path)
    parser.add_argument("--replay-episode", type=positive, default=1)
    args = parser.parse_args()
    replay = replay_actions(args.replay_log, args.replay_episode) if args.replay_log else None
    env = FireRedTurnEnv(args.emulator_pid, port=args.port,
                        max_steps=args.max_steps, decision_timeout=args.decision_timeout)
    print(f"Logs: {args.output}", flush=True)
    try:
        summary = run_baseline(env, args.output, episodes=1 if replay is not None else args.episodes,
            seed=args.seed, allow_escape=args.allow_escape, moves_first=args.moves_first, replay=replay)
        print("Outcomes:", summary["outcomes"])
        return 1 if summary["stopped_on_error"] else 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
