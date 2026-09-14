#!/usr/bin/env python3
"""Watch a manually played battle through Gymnasium; NOOP inputs only."""
import argparse
import time

from firered_rl.backends.mgba_battle import MGBABattleBackend
from firered_rl.env import FireRedBattleEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=55355)
    parser.add_argument("--max-steps", type=int, default=20000)
    args = parser.parse_args()
    env = FireRedBattleEnv(backend_factory=lambda: MGBABattleBackend(port=args.port),
                          frame_skip=1, max_episode_steps=args.max_steps)
    try:
        obs, info = env.reset()
        print("Reset passed. Play the visible emulator manually; this monitor sends NOOP.", flush=True)
        last = None
        total = 0.0
        for step in range(args.max_steps):
            obs, reward, terminated, truncated, info = env.step(0)
            assert env.observation_space.contains(obs), "observation outside declared space"
            total += reward
            state = info["battle_state"]
            if state != last or terminated or truncated:
                print(f"step={step+1} reward={reward:g} total={total:g} "
                      f"HP={state.player_hp}/{state.opponent_hp} "
                      f"fainted={state.player_fainted}/{state.opponent_fainted} "
                      f"outcome={state.outcome_reason} terminated={terminated} truncated={truncated}", flush=True)
                last = state
            if terminated or truncated:
                break
            time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()
