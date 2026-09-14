#!/usr/bin/env python3
"""Play a visible FireRed link battle against a trained PPO checkpoint."""
import argparse
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import traceback

from launch_link import LinkProcess, add_launch_arguments, validate_launch
from firered_rl.human_play import PPOOpponent, play_battle
from firered_rl.link_connect import connect_pair
from firered_rl.linked_env import LinkedBattleSession
from firered_rl.training_states import TrainingStates, add_state_arguments
from firered_rl.launch_options import speed_value
from firered_rl.checkpoints import resolve_checkpoint


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser, root)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint path or latest (newest saved PPO checkpoint)")
    add_state_arguments(parser)
    parser.add_argument("--human-player", type=int, choices=[1, 2], default=1)
    parser.add_argument("--ports", type=int, nargs=2, default=[55355, 55356])
    parser.add_argument("--stochastic", action="store_true", help="Sample the trained policy; default: highest-probability legal action")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=200, help="Battle decision limit, as in training (default: 200)")
    parser.add_argument("--output-dir", type=Path, help="New directory for this match and its report")
    parser.add_argument("--no-wait", action="store_true", help="Close the windows after the match without an Enter prompt")
    parser.add_argument("--setup-speed", type=speed_value,
                        help="Speed for link setup; switch to --speed at battle readiness (requires runtime-speed patch)")
    args = parser.parse_args()
    try:
        if args.headless:
            raise ValueError("human play needs visible windows; remove --headless")
        if args.lobby_only:
            raise ValueError("human play needs a battle; remove --lobby-only")
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if args.output_dir is not None and args.output_dir.expanduser().exists():
            raise ValueError("--output-dir must be a new directory")
        checkpoint = resolve_checkpoint(args.checkpoint, root)
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint not found: {checkpoint}")
        selection = TrainingStates.from_args(args, root)
        states = selection.for_pair(0)()
        validate_launch(args.rom, args.emulator, states, args.ports, False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        import torch
        from sb3_contrib import MaskablePPO
    except ImportError as error:
        parser.exit(1, f"PPO dependencies missing: {error}\nRun: python -m pip install -e '.[train]'\n")
    # Load once, before launching any processes. This runner never calls learn/save.
    try:
        torch.set_num_threads(1)
        checkpoint_bytes = checkpoint.read_bytes()
        model = MaskablePPO.load(BytesIO(checkpoint_bytes), device=args.device)
        model.set_random_seed(args.seed)
        opponent = PPOOpponent(model, stochastic=args.stochastic)
    except Exception as error:
        parser.exit(1, f"cannot load PPO opponent: {error}\n")
    run = (args.output_dir.expanduser().resolve() if args.output_dir else
           root / ".local/run" / datetime.now().strftime("human-%Y%m%d-%H%M%S-%f"))
    run.mkdir(parents=True)
    config = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                  training_timesteps=model.num_timesteps, human_player=args.human_player,
                  states=[str(p) for p in states], state_source=selection.describe(), ports=args.ports,
                  speed=args.speed, setup_speed=args.setup_speed, stochastic=args.stochastic, seed=args.seed, max_steps=args.max_steps)
    (run / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    runtime = LinkProcess(root, run / "match", args.ports, states)

    def launch():
        setup_speed = args.setup_speed if args.setup_speed is not None else args.speed
        runtime.start(args.rom, args.emulator, setup_speed, False, visible_player=args.human_player)
        runtime.wait_ready(args.startup_timeout)
        if args.setup_speed is not None:
            runtime.set_speed(setup_speed)  # Fail early on an unpatched binary.
        connect_pair(runtime.ports, runtime.session_id, runtime.run / "connection.jsonl",
                     timeout=args.connect_timeout, start_battle=True,
                     announce=lambda msg: print(msg, flush=True), process_state=runtime.process_status)
        if args.setup_speed is not None:
            runtime.set_speed(args.speed)
        return runtime

    session = LinkedBattleSession(launch, human_player=args.human_player-1, max_steps=args.max_steps)
    print(f"Human versus PPO: {run}\nCheckpoint training timesteps: {model.num_timesteps}\n"
          "Connecting both games automatically. Wait for READY before using the controls.", flush=True)
    try:
        result = play_battle(session, opponent, args.human_player-1, announce=lambda msg: print(msg, flush=True))
        (run / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        ending = "Battle limit reached" if result["truncated"] else f"You {result['human_outcome']}"
        print(f"{ending}. Human reward: {result['human_reward']:+g}; AI reward: {result['ai_reward']:+g}.\n"
              f"Report: {run / 'result.json'}", flush=True)
        if not args.no_wait:
            try:
                input("Press Enter to close both windows (or Ctrl+C). ")
            except EOFError:
                pass
        return 0
    except KeyboardInterrupt:
        if not (run / "result.json").is_file():
            (run / "result.json").write_text(json.dumps(dict(status="interrupted", decisions=session.steps), indent=2)+"\n")
        print("Match stopped.", flush=True)
        return 0
    except Exception as error:
        runtime.capture_failure(error)
        (run / "failure.json").write_text(json.dumps(dict(error=str(error), traceback=traceback.format_exc()), indent=2)+"\n")
        print(f"Human play failed: {error}\nDiagnostics: {run}", flush=True)
        return 1
    finally:
        session.close()
        runtime.close()  # Also handles failure during launch before the session owns it.
        for i in (1, 2):
            (runtime.run / f"player{i}" / "FireRed.gba").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
