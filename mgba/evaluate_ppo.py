#!/usr/bin/env python3
"""Evaluate a frozen PPO policy against checkpoints or random legal actions on a replayable suite."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
from threading import Event
import traceback

from launch_link import add_launch_arguments, validate_launch, positive_seconds
from train_ppo import EpisodeFactory, positive_int
from firered_rl.checkpoints import resolve_checkpoints
from firered_rl.evaluation import create_suite, digest, evaluate, materialize_assets, validate_suite
from firered_rl.linked_env import LinkedBattleSession
from firered_rl.linked_reward import LinkedRewardConfig
from firered_rl.ppo_features import policy_space
from firered_rl.training_states import TrainingStates, add_state_arguments
from firered_rl.training_teams import TrainingTeams


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser, root, require_rom=False)
    selection = add_state_arguments(parser, include_teams=True, required=True)
    selection.add_argument("--suite", type=Path, help="Replay suite.json from a previous evaluation")
    parser.add_argument("--backend", choices=("showdown", "mgba"), help="Default: showdown, or the replay suite's backend")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Candidate checkpoint path or latest")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--opponent-checkpoint", "--player2-checkpoint", type=Path,
                           help="Reference checkpoint path or latest; omit for a suite with a saved opponent pool")
    reference.add_argument("--random-opponent", action="store_true", help="Uniform random legal actions, seeded per battle")
    parser.add_argument("--episodes", type=positive_int, help="Total battle attempts (default: 1000); even when swapping sides")
    parser.add_argument("--environments", type=positive_int, default=8, help="Even number of concurrent players (default: 8)")
    parser.add_argument("--seed", type=int, help="Suite/team RNG seed (default: 10001); preserved in suite.json")
    parser.add_argument("--swap-sides", action=argparse.BooleanOptionalAction, default=None,
                        help="Each matchup is played with the candidate on both sides (default: on)")
    parser.add_argument("--max-steps", type=positive_int, help="Decision limit per battle (default: 200)")
    parser.add_argument("--base-port", type=int, default=55455, help="First adapter port for mGBA evaluation (default: 55455)")
    parser.add_argument("--command-timeout", type=positive_seconds, default=120)
    parser.add_argument("--node", default="node", help="Node.js executable")
    parser.add_argument("--showdown-workers", choices=("thread", "process"), default="thread")
    parser.add_argument("--device", default="cpu", help="PyTorch inference device")
    parser.add_argument("--torch-threads", type=positive_int, default=1)
    parser.add_argument("--reward-config", type=Path, default=root / "config/rewards/damage_knockout_win.json",
                        help="Reward points to report; no learning or model updates occur")
    parser.add_argument("--readable-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--battle-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--step-logs", action=argparse.BooleanOptionalAction, default=True,
                        help="Showdown per-decision traces and episode team copies (default: on)")
    parser.add_argument("--output-dir", type=Path, help="New evaluation directory; default: .local/run/evaluation-<timestamp>")
    parser.add_argument("--verbose", action="store_true", help="Print mGBA NPC connection phases")
    args = parser.parse_args()
    try:
        if args.environments < 2 or args.environments % 2:
            raise ValueError("--environments must be even and at least 2")
        if args.lobby_only:
            raise ValueError("evaluation requires battles; remove --lobby-only")
        checkpoints = resolve_checkpoints([args.checkpoint] + ([args.opponent_checkpoint] if args.opponent_checkpoint else []), root)
        reward_config = LinkedRewardConfig.load(args.reward_config)
        if args.suite:
            if any(value is not None for value in (args.episodes, args.seed, args.swap_sides, args.max_steps, args.team_size)) or args.mirror_teams:
                raise ValueError("--suite preserves episodes, seeds, sides, teams and max-steps; omit those options")
            suite = validate_suite(json.loads(args.suite.expanduser().read_text()))
            if args.backend and args.backend != suite["backend"]:
                raise ValueError("--backend disagrees with the replay suite")
            args.backend = suite["backend"]
            args.seed, args.max_steps = suite["seed"], suite["max_steps"]
        else:
            args.backend = args.backend or "showdown"
            args.seed = args.seed if args.seed is not None else 10001
            args.max_steps = args.max_steps or 200
            if args.backend == "mgba" and (args.teams or args.team_dir or args.training_profile):
                raise ValueError("mGBA evaluation uses states, not JSON teams")
            source = (TrainingTeams if args.backend == "showdown" else TrainingStates).from_args(args, root)
            suite = validate_suite(create_suite(source, backend=args.backend, episodes=args.episodes or 1000,
                seed=args.seed, swap_sides=args.swap_sides is not False, max_steps=args.max_steps))
        if args.backend == "showdown":
            from firered_rl.showdown_env import validate_showdown
            validate_showdown(root, args.node)
        else:
            if args.rom is None:
                raise ValueError("mGBA evaluation requires --rom")
            if not args.step_logs:
                raise ValueError("--no-step-logs currently applies to Showdown evaluation")
            if args.showdown_workers != "thread":
                raise ValueError("--showdown-workers process requires Showdown")
            states = [Path(suite["assets"][k]["path"]) for k in suite["cases"][0]["assets"]]
            validate_launch(args.rom, args.emulator, [states[i % 2] for i in range(args.environments)],
                            list(range(args.base_port, args.base_port+args.environments)), args.headless)
        pool = suite.get("opponents")
        if pool and (args.opponent_checkpoint or args.random_opponent):
            raise ValueError("this suite pins an opponent pool; omit --opponent-checkpoint/--random-opponent")
        if not pool and not (args.opponent_checkpoint or args.random_opponent):
            raise ValueError("choose --opponent-checkpoint or --random-opponent, or replay a pooled suite")
        snapshots = {path: path.read_bytes() for path in set(checkpoints)}
        import torch
        from sb3_contrib import MaskablePPO
        torch.set_num_threads(args.torch_threads)
        cache = {}

        def load_model(path, expected=None):
            path = Path(path).expanduser().resolve()
            data = snapshots[path] if path in snapshots else path.read_bytes()
            checksum = hashlib.sha256(data).hexdigest()
            if expected and checksum != expected:
                raise ValueError(f"opponent checkpoint hash mismatch: {path}")
            if checksum not in cache:
                model = MaskablePPO.load(BytesIO(data), device=args.device)
                if model.observation_space != policy_space() or model.action_space.n != 11:
                    raise ValueError("checkpoint does not match the battle observation/action schema")
                model.policy.set_training_mode(False)
                cache[checksum] = model
            model = cache[checksum]
            return model, dict(path=str(path), sha256=checksum, training_timesteps=model.num_timesteps)

        candidate, candidate_info = load_model(checkpoints[0])
        models, opponent_models, pool_info = [candidate], None, None
        model_info = dict(candidate=candidate_info)
        if pool:
            opponent_models, pool_info = {}, {}
            for key, entry in pool.items():
                if entry["kind"] == "random":
                    model, info = None, dict(kind="random")
                else:
                    model, info = load_model(entry["path"], entry["sha256"])
                opponent_models[key] = model
                pool_info[key] = {**entry, **info}
        elif args.random_opponent:
            models.append(None)
            model_info["opponent"] = dict(kind="random", rng="per-case-seed")
        else:
            model, info = load_model(checkpoints[1])
            models.append(model)
            model_info["opponent"] = info
        run = (args.output_dir or root / ".local/run" / datetime.now().strftime("evaluation-%Y%m%d-%H%M%S-%f")).expanduser().resolve()
        run.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    (run / "suite.json").write_text(json.dumps(suite, indent=2)+"\n")
    config = dict(backend=args.backend, deterministic=True, environments=args.environments,
                  showdown_workers=args.showdown_workers, reward=asdict(reward_config), suite_sha256=digest(suite),
                  models=model_info, step_logs=args.step_logs)
    if pool_info is not None:
        config["opponents"] = pool_info
    (run / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    paths = materialize_assets(suite, run)
    cancel = Event()
    slots = min(args.environments//2, len(suite["cases"]))
    selected = [[] for _ in range(slots)]
    factories, sessions = [], []
    print(f"Evaluation: {run}\nPPO weights are frozen; PPO actions are deterministic, random-policy actions use per-case seeds.\n"
          f"Candidate: {checkpoints[0]}\nOpponent: {'saved pool' if pool else ('random legal policy' if args.random_opponent else checkpoints[1])}\n"
          f"{len(suite['cases'])} battles; side swaps={suite['swap_sides']}; {slots} concurrent pairs.\n"
          f"Replay schedule: {run / 'suite.json'}", flush=True)
    try:
        for i in range(slots):
            ports = [args.base_port+2*i, args.base_port+2*i+1]
            directory = run / f"pair-{i+1}"
            choose = lambda i=i: tuple(selected[i])
            if args.backend == "showdown" and args.showdown_workers == "process":
                from firered_rl.showdown_process import ProcessShowdownSession
                session = ProcessShowdownSession(root, directory, ports, choose, args, cancel, reward_config)
                factory = session
            elif args.backend == "showdown":
                from firered_rl.showdown_env import ShowdownFactory, ShowdownBattleSession
                factory = ShowdownFactory(root, directory, ports, choose, args, cancel)
                session = ShowdownBattleSession(factory, max_steps=args.max_steps, reward_config=reward_config,
                    readable_log=directory / "battle-readable.log" if args.readable_log else None,
                    battle_log=directory / "battle.log" if args.battle_log else None)
            else:
                factory = EpisodeFactory(root, directory, ports, choose, args, cancel)
                session = LinkedBattleSession(factory.launch, release_episode=factory.release, cancel=cancel,
                    timeout=args.command_timeout, reset_timeout=args.startup_timeout+2*args.connect_timeout+30,
                    max_steps=args.max_steps, reward_config=reward_config,
                    readable_log=directory / "battle-readable.log" if args.readable_log else None,
                    battle_log=directory / "battle.log" if args.battle_log else None)
            factories.append(factory)
            sessions.append(session)
        result = evaluate(sessions, selected, suite, paths, models, run, cancel,
                          aborters=[f.stop for f in factories], announce=lambda msg: print(msg, flush=True), opponents=opponent_models)
        print(f"Completed: {result['completed_battles']} battles; truncated={result['truncated']}\n"
              f"Report: {run / 'summary.json'}", flush=True)
        return 0
    except KeyboardInterrupt:
        print(f"Evaluation stopped. Partial results: {run}", flush=True)
        return 130
    except Exception as error:
        (run / "failure.json").write_text(json.dumps(dict(error=str(error), traceback=traceback.format_exc()), indent=2)+"\n")
        print(f"Evaluation failed: {error}\nDiagnostics: {run}", flush=True)
        return 1
    finally:
        cancel.set()
        for factory in factories:
            factory.stop()
        for session in sessions:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
