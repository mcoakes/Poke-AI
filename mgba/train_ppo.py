#!/usr/bin/env python3
"""Train masked PPO through self-play or against a frozen player-2 checkpoint."""
import argparse
from datetime import datetime
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from threading import Event, Lock
import traceback

from launch_link import LinkProcess, add_launch_arguments, validate_launch, positive_seconds
from firered_rl.link_connect import connect_pair
from firered_rl.linked_env import LinkedBattleSession
from firered_rl.training_states import TrainingStates, add_state_arguments
from firered_rl.training_log import TrainingLog
from firered_rl.linked_reward import LinkedRewardConfig
from firered_rl.checkpoints import resolve_checkpoints
from firered_rl.switch_opponent import SwitchingOpponent, switch_settings


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative and finite")
    return value


class EpisodeFactory:
    def __init__(self, root, run, ports, choose, args, cancel):
        self.root, self.run, self.ports = root, run, ports
        self.choose, self.args, self.cancel = choose, args, cancel
        self.lock = Lock()
        self.active = None
        self.episode = 0
        self.states = None

    def launch(self):
        if self.cancel.is_set():
            raise RuntimeError("training cancelled")
        self.episode += 1
        self.states = self.choose()
        runtime = LinkProcess(self.root, self.run / f"episode-{self.episode:06d}", self.ports, self.states)
        with self.lock:
            self.active = runtime
        try:
            runtime.start(self.args.rom, self.args.emulator, self.args.speed, self.args.headless)
            if self.cancel.is_set():
                raise RuntimeError("training cancelled")
            runtime.wait_ready(self.args.startup_timeout)
            connect_pair(runtime.ports, runtime.session_id, runtime.run / "connection.jsonl",
                         timeout=self.args.connect_timeout, start_battle=True,
                         announce=(lambda msg: print(msg, flush=True)) if self.args.verbose else None,
                         process_state=runtime.process_status)
            if self.cancel.is_set():
                raise RuntimeError("training cancelled")
            return runtime
        except BaseException as error:
            if not self.cancel.is_set():
                runtime.capture_failure(error)
            self.release(runtime)
            raise

    def stop(self):
        with self.lock:
            runtime = self.active
        if runtime is not None:
            runtime.close()

    def release(self, runtime):
        runtime.close()
        for player in (1, 2):
            (runtime.run / f"player{player}" / "FireRed.gba").unlink(missing_ok=True)
        with self.lock:
            if self.active is runtime:
                self.active = None


def save_model(model, path):
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(str(temporary))
    temporary.replace(path)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser, root, require_rom=False)
    add_state_arguments(parser, include_teams=True)
    parser.add_argument("--backend", choices=("mgba", "showdown"), default="mgba",
                        help="Battle engine (default: mgba); Showdown uses team JSON files directly")
    parser.add_argument("--node", default="node", help="Node.js executable for the Showdown backend")
    parser.add_argument("--showdown-workers", choices=("thread", "process"), default="thread",
                        help="Python environment workers per pair; Node simulations are separate processes in both modes")
    parser.add_argument("--environments", type=positive_int, default=4, help="Even number of Gymnasium players (default: 4)")
    parser.add_argument("--base-port", type=int, default=55355)
    parser.add_argument("--train-battles", type=positive_int, help="Finish the PPO update reaching this many ended battles (including truncations); replaces timestep budget")
    parser.add_argument("--total-timesteps", type=positive_int, default=10000, help="Additional learning-player transitions, rounded up to a full rollout")
    parser.add_argument("--rollout-steps", type=positive_int, default=None, help="Steps per learning player; new model default 64; explicit value overrides resume")
    parser.add_argument("--batch-size", type=positive_int, default=None, help="Minibatch size; new model default 64; explicit value overrides resume")
    parser.add_argument("--epochs", type=positive_int, default=None, help="PPO epochs; new model default 4; explicit value overrides resume")
    parser.add_argument("--learning-rate", type=positive_seconds, default=None, help="Learning rate; new model default 0.0003; explicit value overrides resume")
    parser.add_argument("--ent-coef", type=nonnegative, default=None, help="Entropy coefficient; new model default 0.01; explicit value overrides resume")
    parser.add_argument("--max-steps", type=positive_int, default=200, help="Decision limit per battle; unfinished battles truncate")
    parser.add_argument("--command-timeout", type=positive_seconds, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda (default: cpu)")
    parser.add_argument("--torch-threads", type=positive_int, default=1)
    parser.add_argument("--checkpoint-every", type=positive_int, default=1024, help="Learning-player steps between numbered checkpoints; latest saves after every rollout")
    parser.add_argument("--resume", type=Path, help="Checkpoint path or latest; resume model/optimizer; explicit PPO flags override saved values")
    parser.add_argument("--opponent-checkpoint", "--player2-checkpoint", type=Path,
                        help="Checkpoint path or latest; freeze player 2 in every pair; only player 1 trains")
    parser.add_argument("--opponent-stochastic", action="store_true", help="Sample the frozen opponent policy (weights stay fixed)")
    parser.add_argument("--opponent-switch-chance", type=nonnegative, default=0,
                        help="Chance (0–1) of a type-based player-2 switch check; requires a frozen opponent (default: 0)")
    parser.add_argument("--opponent-switch-cooldown", type=int, default=2,
                        help="Normal decisions to skip after a voluntary opponent switch (default: 2)")
    parser.add_argument("--reward-config", type=Path, default=root / "config/rewards/damage_knockout_win.json")
    parser.add_argument("--readable-log", action=argparse.BooleanOptionalAction, default=True,
                        help="Write live observations/actions to each pair's battle-readable.log (default: on)")
    parser.add_argument("--battle-log", action=argparse.BooleanOptionalAction, default=True,
                        help="Write compact actual battle text and HP events to battle.log (default: on)")
    parser.add_argument("--output-dir", type=Path, help="New directory for this run; default: timestamped .local/run/ppo-...")
    parser.add_argument("--verbose", action="store_true", help="Print NPC connection phases")
    args = parser.parse_args()
    hyperparameters = {"rollout_steps": ("n_steps", 64), "batch_size": ("batch_size", 64),
                       "epochs": ("n_epochs", 4), "learning_rate": ("learning_rate", 3e-4),
                       "ent_coef": ("ent_coef", .01)}
    overrides = {}
    for flag, (parameter, default) in hyperparameters.items():
        value = getattr(args, flag)
        if value is not None:
            overrides[parameter] = value
        else:
            setattr(args, flag, default)
    try:
        if args.lobby_only:
            raise ValueError("PPO requires battles; remove --lobby-only")
        if args.environments < 2 or args.environments % 2:
            raise ValueError("--environments must be even and at least 2")
        args.resume, args.opponent_checkpoint = resolve_checkpoints(
            [args.resume, args.opponent_checkpoint], root)
        reward_config = LinkedRewardConfig.load(args.reward_config)
        learners = args.environments//2 if args.opponent_checkpoint else args.environments
        if args.opponent_checkpoint and not args.opponent_checkpoint.expanduser().is_file():
            raise ValueError(f"opponent checkpoint not found: {args.opponent_checkpoint}")
        if args.opponent_stochastic and not args.opponent_checkpoint:
            raise ValueError("--opponent-stochastic requires --opponent-checkpoint")
        switch_config = switch_settings(args.opponent_switch_chance, args.opponent_switch_cooldown)
        if switch_config["chance"] and not args.opponent_checkpoint:
            raise ValueError("--opponent-switch-chance requires --opponent-checkpoint")
        ports = list(range(args.base_port, args.base_port+args.environments))
        if args.backend == "showdown":
            from firered_rl.training_teams import TrainingTeams
            from firered_rl.showdown_env import ShowdownFactory, ShowdownBattleSession, validate_showdown
            validate_showdown(root, args.node)
            source = TrainingTeams.from_args(args, root)
        else:
            if args.showdown_workers != "thread":
                raise ValueError("--showdown-workers process requires --backend showdown")
            if args.rom is None:
                raise ValueError("--backend mgba requires --rom")
            if args.teams is not None or args.team_dir is not None or args.training_profile is not None:
                raise ValueError("--teams/--team-dir require --backend showdown; mGBA uses --states/--state-dir")
            source = TrainingStates.from_args(args, root)
            preview = [source.files[i % len(source.files)] for i in range(args.environments)]
            validate_launch(args.rom, args.emulator, preview, ports, args.headless)
        if args.resume is not None and not args.resume.expanduser().is_file():
            raise ValueError(f"model checkpoint not found: {args.resume}")
        if args.resume is None and (args.batch_size < 2 or args.rollout_steps*learners % args.batch_size):
            raise ValueError("--batch-size must be >= 2 and divide --rollout-steps * learning players "
                             "(half of --environments with a frozen opponent)")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        import numpy as np
        import torch
        import stable_baselines3
        import sb3_contrib
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.logger import configure
        from firered_rl.ppo_features import BattleFeatures
        from firered_rl.ppo_vec_env import LinkedVecEnv
        from firered_rl.frozen_opponent import FrozenOpponentVecEnv
    except ImportError as error:
        parser.exit(1, f"PPO dependencies missing: {error}\nRun: python -m pip install -e '.[train]'\n")
    torch.set_num_threads(args.torch_threads)
    run = (args.output_dir or root / ".local/run" / datetime.now().strftime("ppo-%Y%m%d-%H%M%S-%f")).expanduser().resolve()
    try:
        run.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.exit(1, f"training output must be a new directory: {error}\n")
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["teams" if args.backend == "showdown" else "states"] = source.describe()
    if args.backend == "showdown":
        config["simulator"] = dict(package="pokemon-showdown", version="0.11.11", format="gen3customgame",
                                   pp_ups=0, observation_schema="firered_us_v1_0", bag="empty")
    config.update(algorithm="MaskablePPO", policy="frozen player 2" if args.opponent_checkpoint else "shared self-play",
                  reward=asdict(reward_config), learning_environments=learners,
                  torch=torch.__version__, stable_baselines3=stable_baselines3.__version__, sb3_contrib=sb3_contrib.__version__)
    cancel = Event()
    factory_type = ShowdownFactory if args.backend == "showdown" else EpisodeFactory
    factories = [factory_type(root, run / f"pair-{i+1}", ports[i*2:i*2+2], source.for_pair(i), args, cancel)
                 for i in range(args.environments//2)]
    if args.backend == "showdown":
        sessions = [ShowdownBattleSession(factory, max_steps=args.max_steps, reward_config=reward_config,
                    readable_log=run / f"pair-{i+1}" / "battle-readable.log" if args.readable_log else None,
                    battle_log=run / f"pair-{i+1}" / "battle.log" if args.battle_log else None)
                    for i, factory in enumerate(factories)]
        if args.showdown_workers == "process":
            from firered_rl.showdown_process import ProcessShowdownSession
            # The parent retains team selection and central PPO; workers own simulation,
            # observations, rewards and per-pair logs. No policy copies are trained there.
            sessions = [ProcessShowdownSession(root, run / f"pair-{i+1}", ports[i*2:i*2+2],
                        source.for_pair(i), args, cancel, reward_config)
                        for i in range(args.environments//2)]
            factories = sessions
    else:
        sessions = [LinkedBattleSession(factory.launch, release_episode=factory.release, cancel=cancel,
                timeout=args.command_timeout, reset_timeout=args.startup_timeout+2*args.connect_timeout+30,
                max_steps=args.max_steps, reward_config=reward_config,
                readable_log=run / f"pair-{i+1}" / "battle-readable.log" if args.readable_log else None,
                battle_log=run / f"pair-{i+1}" / "battle.log" if args.battle_log else None)
                for i, factory in enumerate(factories)]
    progress = TrainingLog(run, announce=lambda msg: print(msg, flush=True),
                           learning_players=(0,) if args.opponent_checkpoint else (0, 1))

    def completed(pair, **fields):
        factory = factories[pair]
        progress.episode(pair, states=factory.states, episode=factory.episode, **fields)

    vec = LinkedVecEnv(sessions, cancel=cancel, aborters=[f.stop for f in factories], on_episode=completed,
                       reward_scale=reward_config.scale)
    model, status, iterations = None, "completed", 0
    speed_label = "unthrottled (simulation has no frame-speed setting)" if args.backend == "showdown" else args.speed
    print(f"PPO run: {run}\nBackend: {args.backend}\n{args.environments} players / {len(sessions)} battles; {learners} learning players.\n"
          f"Speed: {speed_label}; team source: {source.label}.\n"
          f"Rewards (reported points): {asdict(reward_config)}\n"
          "Each vector step collects one transition per learning player; item-use actions are disabled.", flush=True)
    if args.backend == "showdown":
        print(f"Showdown workers: {args.showdown_workers}; {len(sessions)} Node processes, "
              + (f"{len(sessions)} Python environment processes + central PPO." if args.showdown_workers == "process"
                 else "Python environment threads in the central PPO process."), flush=True)
    if args.readable_log:
        print(f"Readable logs: {run}/pair-N/battle-readable.log (one continuous file per pair)", flush=True)
    if args.battle_log:
        print(f"Spectator logs: {run}/pair-N/battle.log (game messages and HP changes)", flush=True)
    try:
        snapshots = {str(p): p.read_bytes() for p in {args.resume, args.opponent_checkpoint} if p is not None}
        if args.opponent_checkpoint:
            frozen_path = args.opponent_checkpoint.expanduser().resolve()
            frozen_bytes = snapshots[str(frozen_path)]
            opponent = MaskablePPO.load(BytesIO(frozen_bytes), device=args.device)
            opponent.set_random_seed(args.seed)
            config["frozen_opponent"] = dict(path=str(frozen_path), sha256=hashlib.sha256(frozen_bytes).hexdigest(),
                training_timesteps=opponent.num_timesteps, stochastic=args.opponent_stochastic)
            switching = SwitchingOpponent(root, vec.num_envs//2, seed=args.seed, **switch_config) if switch_config["chance"] else None
            vec = FrozenOpponentVecEnv(vec, opponent, stochastic=args.opponent_stochastic,
                                       switching=switching, switch_log=run / "opponent-switches.jsonl")
            config["opponent_switch"] = dict(**switch_config, heuristic_version=1,
                scope="training_only", minimum_hp_fraction=.35, minimum_score_gain=1)
            if switching:
                print(f"Opponent switch checks: {100*switch_config['chance']:g}% chance; "
                      f"{switch_config['cooldown']} normal decisions cooldown. Evaluation policies remain unmodified.", flush=True)
            print(f"Player 2 frozen at: {frozen_path} ({opponent.num_timesteps} training timesteps).", flush=True)
        if args.resume:
            model = MaskablePPO.load(BytesIO(snapshots[str(args.resume)]), env=vec, device=args.device, **overrides)
            if "learning_rate" in overrides:
                for group in model.policy.optimizer.param_groups:
                    group["lr"] = overrides["learning_rate"]
            config["resume_overrides"] = overrides
            config["resume_checkpoint"] = dict(path=str(args.resume),
                sha256=hashlib.sha256(snapshots[str(args.resume)]).hexdigest())
            print(f"Resuming {args.resume}; explicit PPO overrides: {overrides or 'none'}. Starting new battles.", flush=True)
        else:
            model = MaskablePPO("MultiInputPolicy", vec, device=args.device, seed=args.seed,
                n_steps=args.rollout_steps, batch_size=args.batch_size, n_epochs=args.epochs,
                learning_rate=args.learning_rate, ent_coef=args.ent_coef, gamma=.99, gae_lambda=.95,
                target_kl=.03, verbose=0, policy_kwargs=dict(features_extractor_class=BattleFeatures,
                    features_extractor_kwargs=dict(features_dim=128), normalize_images=False,
                    net_arch=dict(pi=[128, 64], vf=[128, 64])))
        model.set_logger(configure(str(run / "ppo"), ["csv"]))
        model.battle_backend = args.backend
        model.reward_settings = asdict(reward_config)
        rollout = model.n_steps*learners
        if model.batch_size < 2 or rollout % model.batch_size:
            raise ValueError("effective PPO batch size must divide n_steps * learning players; "
                             "frozen player 2 halves the learning-player count")
        config["effective_ppo"] = dict(n_steps=model.n_steps, batch_size=model.batch_size, n_epochs=model.n_epochs,
            learning_rate=float(model.lr_schedule(1)), gamma=model.gamma, ent_coef=model.ent_coef,
            starting_timesteps=model.num_timesteps, rollout_player_timesteps=rollout)
        print(f"Effective PPO: rollout steps={model.n_steps}, batch size={model.batch_size}, "
              f"epochs={model.n_epochs}, learning rate={model.lr_schedule(1):g}, "
              f"entropy coefficient={model.ent_coef:g}; {rollout} learning transitions/update", flush=True)
        (run / "config.json").write_text(json.dumps(config, indent=2)+"\n")
        start = model.num_timesteps
        target = start + math.ceil(args.total_timesteps/rollout)*rollout
        if args.train_battles:
            print(f"Battle budget: {args.train_battles}; finish the current rollout when reached (may overshoot).", flush=True)
        next_checkpoint = start + args.checkpoint_every
        while (progress.attempts < args.train_battles if args.train_battles else model.num_timesteps < target):
            print(f"Collecting {rollout} player transitions for update {iterations+1}; " +
                  (f"ended battles={progress.attempts}/{args.train_battles}; timesteps={model.num_timesteps}"
                   if args.train_battles else f"timesteps={model.num_timesteps}/{target}"), flush=True)
            model.learn(total_timesteps=rollout, reset_num_timesteps=False, log_interval=None)
            iterations += 1
            if isinstance(vec, FrozenOpponentVecEnv):
                switches = vec.switching_metrics()
                (run / "switching.json").write_text(json.dumps(switches, indent=2)+"\n")
                for key, value in switches.items():
                    model.logger.record("switching/"+key, value)
                print(f"Voluntary switches: P1 {switches['learner_voluntary_switch_rate']:.1%}, "
                      f"P2 {switches['opponent_voluntary_switch_rate']:.1%}; "
                      f"opponent heuristic overrides={switches['heuristic_overrides']}", flush=True)
            for key, value in progress.reward_metrics().items():
                if value is not None:
                    model.logger.record("rollout/"+key, value)
            metrics = {key: float(value) for key, value in model.logger.name_to_value.items()
                       if isinstance(value, (int, float, np.number))}
            progress.update(model.num_timesteps, iterations, metrics)
            model.logger.dump(model.num_timesteps)
            save_model(model, checkpoints / "latest.zip")
            if model.num_timesteps >= next_checkpoint:
                save_model(model, checkpoints / f"ppo_{model.num_timesteps:09d}.zip")
                next_checkpoint = model.num_timesteps + args.checkpoint_every
            print(f"PPO update {iterations}: timesteps={model.num_timesteps}, "
                  f"policy_loss={metrics.get('train/policy_gradient_loss', float('nan')):.5f}; latest.zip saved", flush=True)
        save_model(model, checkpoints / "final.zip")
    except KeyboardInterrupt:
        status = "interrupted"
        message = ("Latest completed PPO update preserved in checkpoints/latest.zip."
                   if (checkpoints / "latest.zip").is_file()
                   else "Stopped before the first PPO update was saved; no trained checkpoint was written.")
        print(f"Training interrupted. {message}", flush=True)
    except Exception as error:
        status = "failed"
        (run / "failure.json").write_text(json.dumps(dict(error=str(error), traceback=traceback.format_exc()), indent=2)+"\n")
        print(f"PPO failed: {error}\nDiagnostics: {run}", flush=True)
    finally:
        vec.close()
        summary = progress.finish(status)
        if isinstance(vec, FrozenOpponentVecEnv):
            summary["switching"] = vec.switching_metrics()
            (run / "switching.json").write_text(json.dumps(summary["switching"], indent=2)+"\n")
            temporary = run / "summary.tmp"
            temporary.write_text(json.dumps(summary, indent=2)+"\n")
            temporary.replace(run / "summary.json")
    print(f"Training {status}: {summary['completed_battles']} completed battles, "
          f"{summary['truncated_battles']} truncated; {summary['episodes_per_hour']:.1f} battles/hour\n"
          f"Checkpoints: {checkpoints}", flush=True)
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
