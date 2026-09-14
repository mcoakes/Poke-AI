#!/usr/bin/env python3
"""Interactive semantic actions: one Gymnasium step per battle decision."""
import argparse
import json
from pathlib import Path
from firered_rl.turn_env import FireRedTurnEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=55356)
    parser.add_argument("--emulator-pid", type=int, required=True,
                        help="actual mgba-qt PID shown by the visible launcher")
    args = parser.parse_args()
    names = json.loads((Path(__file__).parent / "profiles/firered_us_v1_0_names.json").read_text())
    env = FireRedTurnEnv(args.emulator_pid, port=args.port)
    try:
        obs, info = env.reset()
        print("0–3: move slots 1–4; 4–9: switch to party slots 1–6; 10: escape; q: quit")
        while True:
            for i, seen in enumerate(obs["opponent_party"]):
                if seen[0]:
                    moves = [f"{names['moves'].get(str(int(row[0])), str(int(row[0])))} "
                             f"PP {row[2]}/{row[3]}"
                             for row in obs["opponent_moves"][i] if row[0]]
                    print("Opponent", names["species"].get(str(int(seen[1])), str(int(seen[1]))),
                          f"(party slot {i+1}) moves:", ", ".join(moves) or "none")
            print("Available actions:", [i for i, enabled in enumerate(obs["action_mask"]) if enabled])
            text = input("Action (emulator suspended while waiting): ").strip()
            if text.lower() == "q":
                break
            try:
                action = int(text)
                obs, reward, terminated, truncated, info = env.step(action)
            except ValueError as error:
                print(error)
                continue
            print(f"reward={reward:g} terminated={terminated} truncated={truncated} info={info}")
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
