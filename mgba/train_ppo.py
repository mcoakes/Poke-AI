#!/usr/bin/env python3
"""Train masked PPO through self-play or against a frozen player-2 checkpoint."""
import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import math
from numbers import Real
from pathlib import Path
import re
import shutil
import sys
import tempfile
from threading import Event, Lock
import time
import traceback

from launch_link import LinkProcess, add_launch_arguments, validate_launch, positive_seconds
from firered_rl.link_connect import connect_pair
from firered_rl.linked_env import LinkedBattleSession
from firered_rl.training_states import TrainingStates, add_state_arguments
from firered_rl.training_log import TrainingLog
from firered_rl.linked_reward import LinkedRewardConfig
from firered_rl.checkpoints import resolve_checkpoints
from firered_rl.switch_opponent import SwitchingOpponent, switch_settings
from firered_rl.training_run import RunCancelled, select_training_run
from firered_rl.periodic_evaluation import EvaluationSetup, PeriodicEvaluation


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative and finite")
    return value


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return value


PPO_DIAGNOSTIC_METRICS = (
    "train/learning_rate", "train/entropy_loss", "train/approx_kl",
    "train/clip_fraction", "train/value_loss", "train/explained_variance",
    "train/policy_gradient_loss", "train/loss", "train/n_updates", "train/clip_range",
)


def update_metrics(logger_values):
    """Capture post-update numeric diagnostics as portable JSON scalars.

    Undefined explained variance (constant returns) is represented by null, as
    are missing expected metrics, rather than writing non-standard JSON NaN.
    """
    metrics = {key: (float(value) if math.isfinite(float(value)) else None)
               for key, value in logger_values.items() if isinstance(value, Real)}
    for key in PPO_DIAGNOSTIC_METRICS:
        metrics.setdefault(key, None)
    return metrics


def retention_count(value):
    value = int(value)
    if value < -1:
        raise argparse.ArgumentTypeError("must be >= -1 (-1 means unlimited)")
    return value


def add_log_arguments(parser):
    parser.add_argument("--step-logs", action=argparse.BooleanOptionalAction, default=None,
                        help="Full Showdown decision traces (default: off); mGBA keeps its existing diagnostics")
    parser.add_argument("--log-max-mb", type=nonnegative, default=10,
                        help="Showdown diagnostic log size before rotation; 0 unlimited (default: 10 MiB)")
    parser.add_argument("--log-backups", type=nonnegative_int, default=2,
                        help="Rotated backups per Showdown log (default: 2)")
    parser.add_argument("--keep-episode-logs", type=retention_count, default=2,
                        help="Completed Showdown trace directories per pair; 0 disables, -1 unlimited (default: 2)")


def prune_numbered_checkpoints(directory, keep):
    """Only numbered backups owned by this run; latest/final are never touched."""
    if keep < 0:
        return
    paths = sorted((p for p in directory.iterdir()
                    if re.fullmatch(r"ppo_\d+\.zip", p.name) and p.is_file() and not p.is_symlink()),
                   key=lambda p: int(p.stem[4:]), reverse=True)
    for path in paths[keep:]:
        path.unlink()


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


def copy_checkpoint_file(source, path):
    temporary = path.with_name(path.stem + ".tmp.zip")
    shutil.copyfile(source, temporary)
    temporary.replace(path)


def save_model(model, path, *copies):
    started = time.monotonic()
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(str(temporary))
    temporary.replace(path)
    for target in copies:
        copy_checkpoint_file(path, target)
    return time.monotonic() - started


def json_arguments(args):
    """Preserve original selectors separately from expanded source descriptions."""
    def convert(value):
        if isinstance(value, Path):
            return str(value.expanduser().resolve())
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    return {key: convert(value) for key, value in vars(args).items()}


def write_run_file(path, data):
    """Publish small metadata and pinned inputs without partially replacing them."""
    if isinstance(data, str):
        data = data.encode()
    if path.is_file() and path.read_bytes() == data:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def training_session(run, continuing, previous_status=None):
    """Keep prior invocation metadata and CSV/log files without copying large logs."""
    history = run / "sessions"
    history.mkdir(exist_ok=True)
    numbers = [int(path.name) for path in history.iterdir()
               if path.is_dir() and re.fullmatch(r"\d{6}", path.name)]
    previous = max(numbers, default=1) if continuing else 0
    if continuing:
        archived = history / f"{previous:06d}"
        archived.mkdir(exist_ok=True)
        for name in ("config.json", "summary.json", "switching.json", "failure.json"):
            if name == "failure.json" and previous_status != "failed":
                continue
            path = run / name
            if path.is_file():
                shutil.copyfile(path, archived / name)
    number = previous + 1
    metadata = history / f"{number:06d}"
    metadata.mkdir(exist_ok=False)
    # Initial run keeps its existing public layout; continuations get new logs.
    work = metadata if continuing else run
    return number, metadata, work


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser, root, require_rom=False)
    add_state_arguments(parser, include_teams=True, include_mixture=True, required=False)
    parser.add_argument("--backend", choices=("mgba", "showdown"), default="mgba",
                        help="Battle engine (default: mgba); Showdown uses team JSON files directly")
    parser.add_argument("--observation-version", choices=("v1", "v2"), default=None,
                        help="Policy observations: fresh flat defaults to v1, shared-moves/matchup-moves/matchup-switches to v2; auto-detected on resume; native v2 requires adapter 0.19.0")
    parser.add_argument("--policy-architecture", choices=("flat", "shared-moves", "matchup-moves", "matchup-switches"), default=None,
                        help="New model: flat (default), shared-moves, matchup-moves or matchup-switches (v2); resume auto-detects and rejects architecture changes")
    parser.add_argument("--shuffle-move-slots", action=argparse.BooleanOptionalAction, default=False,
                        help="Showdown: shuffle learner move presentation per battle and map actions back; frozen opponents retain original order (default: off)")
    parser.add_argument("--shuffle-bench-order", action=argparse.BooleanOptionalAction, default=False,
                        help="Showdown Rules training: shuffle player 1's occupied bench after team sampling; keep lead/opponent/fixed-pair rehearsal unchanged (default: off)")
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
    parser.add_argument("--latest-every-updates", type=positive_int, default=1,
                        help="Save latest.zip every N PPO updates, plus the first update and successful completion (default: 1)")
    parser.add_argument("--checkpoint-every", type=nonnegative_int, default=1024,
                        help="Learning-player steps between numbered backups, rounded to an update; 0 disables them (default: 1024)")
    parser.add_argument("--keep-checkpoints", type=retention_count, default=2,
                        help="Numbered backups to retain; 0 disables, -1 keeps all; latest/final always kept (default: 2)")
    parser.add_argument("--eval-suite", type=Path,
                        help="Fixed Showdown suite for periodic validation and best.zip selection; copied into this run")
    parser.add_argument("--eval-every", type=nonnegative_int, default=100000,
                        help="Additional learning transitions between validations, rounded to a PPO update; 0 pauses validation (default: 100000)")
    parser.add_argument("--eval-environments", type=positive_int,
                        help="Even number of validation players (default: training environment count)")
    parser.add_argument("--keep-evaluations", type=positive_int, default=2,
                        help="Recent validation result directories to retain, plus the best result; history and best.zip always kept (default: 2)")
    parser.add_argument("--eval-stop-win-rate", type=probability,
                        help="Stop successfully at this wins/requested validation fraction (0-1); requires zero truncations; omitted disables early stopping")
    parser.add_argument("--eval-stop-consecutive", type=positive_int, default=2,
                        help="Distinct consecutive successful validation checkpoints required to stop (default: 2); initial validation counts once")
    parser.add_argument("--no-eval-stop", action="store_const", dest="eval_stop_win_rate", const=None,
                        help="Explicitly disable validation early stopping when continuing a saved run")
    parser.add_argument("--resume", type=Path, help="Checkpoint path or latest; resume model/optimizer; explicit PPO flags override saved values")
    parser.add_argument("--opponent-checkpoint", "--player2-checkpoint", type=Path,
                        help="Checkpoint path or latest; freeze player 2 in every pair; only player 1 trains")
    parser.add_argument("--opponent-stochastic", action="store_true", help="Sample the frozen opponent policy (weights stay fixed)")
    parser.add_argument("--rule-based-opponent", action="store_true",
                        help="Showdown only: freeze player 2 as gen3-rules-v1; incompatible with --opponent-checkpoint")
    parser.add_argument("--opponent-pool", type=Path,
                        help="Showdown: weighted JSON pool of frozen checkpoints, rules and/or Random; sampled once per battle")
    parser.add_argument("--rule-log", action=argparse.BooleanOptionalAction, default=False,
                        help="Write bounded rule-decisions.jsonl with scores and reasons")
    parser.add_argument("--opponent-switch-chance", type=nonnegative, default=0,
                        help="Chance (0–1) of a type-based player-2 switch check; requires a frozen opponent (default: 0)")
    parser.add_argument("--opponent-switch-cooldown", type=int, default=2,
                        help="Normal decisions to skip after a voluntary opponent switch (default: 2)")
    parser.add_argument("--reward-config", type=Path, default=root / "config/rewards/damage_knockout_win.json")
    parser.add_argument("--readable-log", action=argparse.BooleanOptionalAction, default=True,
                        help="Write live observations/actions to each pair's battle-readable.log (default: on)")
    parser.add_argument("--battle-log", action=argparse.BooleanOptionalAction, default=True,
                        help="Write compact actual battle text and HP events to battle.log (default: on)")
    add_log_arguments(parser)
    parser.add_argument("--output-dir", type=Path, help="Run directory; existing runs use --if-exists (default: timestamped .local/run/ppo-...)")
    parser.add_argument("--if-exists", choices=("ask", "continue", "erase", "cancel"), default="ask",
                        help="Existing output directory: ask in a terminal, continue saved training, erase and restart, or cancel")
    parser.add_argument("--verbose", action="store_true", help="Print NPC connection phases")
    args = parser.parse_args()
    try:
        plan = select_training_run(parser, args, sys.argv[1:], root)
    except RunCancelled:
        print("Training cancelled; existing run unchanged.", flush=True)
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.step_logs is None:
        args.step_logs = args.backend == "mgba"
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
        from firered_rl.bench_order import validate_options as validate_bench_order_options
        validate_bench_order_options(args)
        if args.shuffle_move_slots and args.backend != "showdown":
            raise ValueError("--shuffle-move-slots requires --backend showdown")
        if args.lobby_only:
            raise ValueError("PPO requires battles; remove --lobby-only")
        if args.environments < 2 or args.environments % 2:
            raise ValueError("--environments must be even and at least 2")
        if args.eval_suite is None and (args.eval_environments is not None or
                                        ("eval_every" in plan.explicit and args.eval_every)):
            raise ValueError("--eval-every/--eval-environments require --eval-suite")
        if args.eval_stop_win_rate is not None and (args.eval_suite is None or not args.eval_every):
            raise ValueError("--eval-stop-win-rate requires --eval-suite and positive --eval-every")
        if "eval_stop_consecutive" in plan.explicit and args.eval_stop_win_rate is None:
            raise ValueError("--eval-stop-consecutive requires --eval-stop-win-rate")
        args.resume, args.opponent_checkpoint = resolve_checkpoints(
            [args.resume, args.opponent_checkpoint], root)
        reward_config = LinkedRewardConfig.load(args.reward_config)
        if args.opponent_pool and (args.backend != "showdown" or args.opponent_checkpoint or args.rule_based_opponent):
            raise ValueError("--opponent-pool requires Showdown and no other opponent selector")
        if args.rule_based_opponent and (args.backend != "showdown" or args.opponent_checkpoint):
            raise ValueError("--rule-based-opponent requires Showdown and no --opponent-checkpoint")
        if args.rule_log and not (args.rule_based_opponent or args.opponent_pool):
            raise ValueError("--rule-log requires --rule-based-opponent or --opponent-pool")
        learners = args.environments//2 if args.opponent_checkpoint or args.rule_based_opponent or args.opponent_pool else args.environments
        if args.opponent_checkpoint and not args.opponent_checkpoint.expanduser().is_file():
            raise ValueError(f"opponent checkpoint not found: {args.opponent_checkpoint}")
        if args.opponent_stochastic and not (args.opponent_checkpoint or args.opponent_pool):
            raise ValueError("--opponent-stochastic requires --opponent-checkpoint or --opponent-pool")
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
            if not args.step_logs:
                raise ValueError("--no-step-logs currently applies to Showdown training")
            if args.showdown_workers != "thread":
                raise ValueError("--showdown-workers process requires --backend showdown")
            if args.rom is None:
                raise ValueError("--backend mgba requires --rom")
            if args.teams is not None or args.team_dir is not None or args.training_profile is not None or args.team_mixture is not None:
                raise ValueError("--teams/--team-dir/--team-mixture require --backend showdown; mGBA uses --states/--state-dir")
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
        import torch
        from sb3_contrib import MaskablePPO
        from firered_rl.ppo_features import observation_version
        from firered_rl.shared_move_policy import policy_architecture
    except ImportError as error:
        parser.exit(1, f"PPO dependencies missing: {error}\nRun: python -m pip install -e '.[train]'\n")
    torch.set_num_threads(args.torch_threads)
    # Snapshot and load checkpoints before creating any workers or output directory.
    # The saved space is authoritative; legacy checkpoints need no version tag.
    resumed_model, opponent = None, None
    try:
        if args.device != "auto":
            torch.device(args.device)
        snapshots = {str(p): p.read_bytes() for p in {args.resume, args.opponent_checkpoint} if p is not None}
        opponent_version = None
        if args.opponent_pool:
            from firered_rl.opponent_pool import OpponentPool
            opponent = OpponentPool(root, args.opponent_pool, count=args.environments//2, seed=args.seed, device=args.device)
            opponent_version = observation_version(opponent.observation_space)
        if args.rule_based_opponent:
            from firered_rl.rule_opponent import RuleBasedOpponent
            opponent = RuleBasedOpponent(root)
            opponent_version = "v2"
        if args.opponent_checkpoint:
            opponent = MaskablePPO.load(BytesIO(snapshots[str(args.opponent_checkpoint)]), device=args.device)
            opponent_version = observation_version(opponent.observation_space)
            opponent.set_random_seed(args.seed)
        if args.resume:
            resumed_model = MaskablePPO.load(BytesIO(snapshots[str(args.resume)]), device=args.device,
                                            n_envs=learners, **overrides)
            saved_architecture = policy_architecture(resumed_model)
            if args.policy_architecture is not None and args.policy_architecture != saved_architecture:
                raise ValueError(f"--policy-architecture {args.policy_architecture} does not match the "
                                 f"{saved_architecture} resume checkpoint. Start a fresh model without "
                                 "--resume to change architecture; weights are not converted.")
            args.policy_architecture = saved_architecture
            saved_version = observation_version(resumed_model.observation_space)
            if args.observation_version is not None and args.observation_version != saved_version:
                raise ValueError(f"--observation-version {args.observation_version} does not match the "
                                 f"{saved_version} resume checkpoint; omit the flag to auto-detect, or start "
                                 "a new run without --resume. Checkpoints are not migrated between schemas.")
            args.observation_version = saved_version
        else:
            args.policy_architecture = args.policy_architecture or "flat"
            args.observation_version = args.observation_version or (
                "v2" if args.policy_architecture in ("shared-moves", "matchup-moves", "matchup-switches") else "v1")
        if args.policy_architecture in ("shared-moves", "matchup-moves", "matchup-switches") and args.observation_version != "v2":
            raise ValueError(f"--policy-architecture {args.policy_architecture} requires --observation-version v2")
        args.session_observation_version = ("v2" if "v2" in (args.observation_version, opponent_version)
                                            else "v1")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(f"checkpoint/observation configuration: {error}")
    run = plan.run
    source_description = source.describe()
    frozen_sha256 = (hashlib.sha256(snapshots[str(args.opponent_checkpoint)]).hexdigest()
                     if args.opponent_checkpoint else None)
    try:
        # Validate all effective PPO settings before an erase can remove anything.
        effective_steps = resumed_model.n_steps if resumed_model else args.rollout_steps
        effective_batch = resumed_model.batch_size if resumed_model else args.batch_size
        if effective_batch < 2 or effective_steps * learners % effective_batch:
            raise ValueError("effective PPO batch size must divide n_steps * learning players")
        plan.validate(source_description, asdict(reward_config), frozen_sha256)
        if args.team_mixture is not None and plan.action == "continue":
            expected = plan.saved_config.get("team_mixture_snapshot")
            if expected is None:
                raise ValueError("saved team mixture snapshot is missing")
            source.snapshot(run / "inputs/team-mixture", expected=expected)
        if args.rule_based_opponent and plan.action == "continue":
            if plan.saved_config.get("rule_opponent") != opponent.identity:
                raise ValueError("rule opponent implementation changed; start a new run with --resume")
        if args.opponent_pool and plan.action == "continue":
            if plan.saved_config.get("opponent_pool_identity") != opponent.identity:
                raise ValueError("opponent pool changed; start a new run with --resume")
        evaluation_setup = (EvaluationSetup(args, plan, asdict(reward_config), frozen_sha256)
                            if args.eval_suite is not None else None)
        inputs = [*source.files, args.reward_config, args.resume, args.opponent_checkpoint,
                  args.rom, args.emulator if args.backend == "mgba" else None, args.team_mixture]
        if args.opponent_pool:
            inputs.extend(opponent.inputs)
        if evaluation_setup is not None:
            inputs.extend(evaluation_setup.inputs)
        lease = plan.prepare(input_paths=[path for path in inputs if path is not None])
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        return execute_training(root, run, plan, args, source, source_description,
                                reward_config, snapshots, resumed_model, opponent,
                                opponent_version, switch_config, overrides, ports, learners, evaluation_setup)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    finally:
        lease.close()


def execute_training(root, run, plan, args, source, source_description, reward_config,
                     snapshots, resumed_model, opponent, opponent_version, switch_config,
                     overrides, ports, learners, evaluation_setup=None):
    import torch
    import stable_baselines3
    import sb3_contrib
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure
    from firered_rl.ppo_features import BattleFeatures
    from firered_rl.ppo_vec_env import LinkedVecEnv
    from firered_rl.frozen_opponent import FrozenOpponentVecEnv
    from firered_rl.showdown_env import ShowdownFactory, ShowdownBattleSession

    continuing = plan.action == "continue"
    previous_summary = (json.loads((run / "summary.json").read_text())
                        if continuing and (run / "summary.json").is_file() else {})
    previous_switching = (json.loads((run / "switching.json").read_text())
                          if continuing and (run / "switching.json").is_file() else {})
    # Reject corrupt history before creating a session or changing any run inputs.
    progress = TrainingLog(run, announce=lambda msg: print(msg, flush=True),
                           learning_players=(0,) if opponent is not None else (0, 1),
                           resume=continuing)
    session_id, metadata, work = training_session(run, continuing, previous_summary.get("status"))
    progress.session_id = session_id
    starting_attempts, starting_updates = progress.attempts, progress.updates
    if continuing:
        (run / "failure.json").unlink(missing_ok=True)
    evaluation_config = evaluation_setup.install(run, args) if evaluation_setup is not None else None
    saved_args = json_arguments(args)
    mixture_snapshot = None
    if args.team_mixture is not None:
        source = source.snapshot(run / "inputs/team-mixture",
                                 expected=plan.saved_config["team_mixture_snapshot"] if continuing else None)
        mixture_snapshot = source.describe()
    if continuing:
        # Keep the base seed in saved_args; each invocation starts fresh battles.
        args.seed = (args.seed + (session_id - 1) * 1000003) % (2**31)
        source.seed = args.seed
        if opponent is not None:
            opponent.set_random_seed(args.seed)
    if args.shuffle_bench_order:
        from firered_rl.bench_order import BenchOrderTeams
        source = BenchOrderTeams(source, run / "inputs/bench-order", seed=args.seed)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["saved_args"] = saved_args
    config["teams" if args.backend == "showdown" else "states"] = source_description
    if mixture_snapshot is not None:
        config["team_mixture_snapshot"] = mixture_snapshot
    if args.shuffle_bench_order:
        config["bench_order_shuffle"] = source.configuration()
        print("Learner bench order: randomized after canonical team sampling; lead, moves, "
              "Rules opponent and fixed-pair rehearsal stay unchanged.", flush=True)
    config["session"] = dict(id=session_id, continued=continuing, directory=str(work), seed=args.seed)
    if evaluation_config is not None:
        config["evaluation"] = evaluation_config
    campaign_child = run.name == "train" and re.fullmatch(r"attempt-\d+", run.parent.name)
    if not campaign_child:
        pinned = run / "inputs"
        pinned.mkdir(exist_ok=True)
        reward_path = pinned / "reward.json"
        write_run_file(reward_path, json.dumps(asdict(reward_config), indent=2) + "\n")
        config["reward_config"] = saved_args["reward_config"] = str(reward_path)
        if args.opponent_checkpoint:
            opponent_path = pinned / "frozen-opponent.zip"
            write_run_file(opponent_path, snapshots[str(args.opponent_checkpoint)])
            config["opponent_checkpoint"] = saved_args["opponent_checkpoint"] = str(opponent_path)
    if args.backend == "showdown":
        config["simulator"] = dict(package="pokemon-showdown", version="0.11.11", format="gen3customgame",
                                   pp_ups=0, observation_schema=("firered_us_v1_0" if args.session_observation_version == "v1"
                                                                else "firered_showdown_v2"), bag="empty")
    config.update(algorithm="MaskablePPO", policy="frozen player 2" if args.opponent_checkpoint else "shared self-play",
                  reward=asdict(reward_config), learning_environments=learners,
                  torch=torch.__version__, stable_baselines3=stable_baselines3.__version__, sb3_contrib=sb3_contrib.__version__)
    config["update_diagnostics"] = dict(path=str(run / "updates.jsonl"), schema_version=1,
        timing="after PPO optimization, before validation", expected_metrics=list(PPO_DIAGNOSTIC_METRICS),
        nonfinite_or_missing="null", metric_container="metrics")
    if args.rule_based_opponent:
        config.update(policy="rule-based player 2", rule_opponent=opponent.identity)
        opponent.configure_log(work, args)
    if args.opponent_pool:
        config.update(policy="episode opponent pool", opponent_pool_identity=opponent.identity)
        config["opponent_pool"] = saved_args["opponent_pool"] = str(opponent.install(run / "inputs/opponent-pool"))
        opponent.configure_log(work, args)
    cancel = Event()
    factory_type = ShowdownFactory if args.backend == "showdown" else EpisodeFactory
    factories = [factory_type(root, work / f"pair-{i+1}", ports[i*2:i*2+2], source.for_pair(i), args, cancel)
                 for i in range(args.environments//2)]
    if args.backend == "showdown":
        sessions = [ShowdownBattleSession(factory, max_steps=args.max_steps, reward_config=reward_config,
                    readable_log=work / f"pair-{i+1}" / "battle-readable.log" if args.readable_log else None,
                    battle_log=work / f"pair-{i+1}" / "battle.log" if args.battle_log else None)
                    for i, factory in enumerate(factories)]
        if args.showdown_workers == "process":
            from firered_rl.showdown_process import ProcessShowdownSession
            # The parent retains team selection and central PPO; workers own simulation,
            # observations, rewards and per-pair logs. No policy copies are trained there.
            sessions = [ProcessShowdownSession(root, work / f"pair-{i+1}", ports[i*2:i*2+2],
                        source.for_pair(i), args, cancel, reward_config)
                        for i in range(args.environments//2)]
            factories = sessions
    else:
        sessions = [LinkedBattleSession(factory.launch, release_episode=factory.release, cancel=cancel,
                observation_version=args.session_observation_version,
                timeout=args.command_timeout, reset_timeout=args.startup_timeout+2*args.connect_timeout+30,
                max_steps=args.max_steps, reward_config=reward_config,
                readable_log=work / f"pair-{i+1}" / "battle-readable.log" if args.readable_log else None,
                battle_log=work / f"pair-{i+1}" / "battle.log" if args.battle_log else None)
                for i, factory in enumerate(factories)]
    def completed(pair, **fields):
        factory = factories[pair]
        progress.episode(pair, states=factory.states, episode=factory.episode, **fields)

    vec = LinkedVecEnv(sessions, cancel=cancel, aborters=[f.stop for f in factories], on_episode=completed,
                       reward_scale=reward_config.scale, observation_version=args.session_observation_version)
    model, status, iterations = None, "completed", starting_updates
    validation = None
    stopping = None
    stop_reason = "battle_budget" if args.train_battles else "timestep_budget"
    opponent_vec = None
    session_iterations = 0
    checkpoint_seconds_total = previous_summary.get("checkpointing", {}).get("seconds", 0.0)
    latest_timesteps = None
    if continuing:
        # Normalize the local recovery point if the newest file was a numbered backup.
        temporary = checkpoints / "latest.tmp.zip"
        temporary.write_bytes(snapshots[str(args.resume)])
        temporary.replace(checkpoints / "latest.zip")
        latest_timesteps = resumed_model.num_timesteps
        print(f"Continuing session {session_id}: {starting_attempts} ended battles retained; "
              f"recovered {latest_timesteps} training timesteps. New battle seed: {args.seed}.", flush=True)
        if progress.player_steps > latest_timesteps:
            print("Prior logs include updates after the last saved checkpoint; those weights were not recovered.", flush=True)
    speed_label = "unthrottled (simulation has no frame-speed setting)" if args.backend == "showdown" else args.speed
    print(f"PPO run: {run}\nBackend: {args.backend}; observations: {args.observation_version}\n{args.environments} players / {len(sessions)} battles; {learners} learning players.\n"
          f"Speed: {speed_label}; team source: {source.label}.\n"
          f"Rewards (reported points): {asdict(reward_config)}\n"
          "Each vector step collects one transition per learning player; item-use actions are disabled.", flush=True)
    if args.backend == "showdown":
        print(f"Showdown workers: {args.showdown_workers}; {len(sessions)} Node processes, "
              + (f"{len(sessions)} Python environment processes + central PPO." if args.showdown_workers == "process"
                 else "Python environment threads in the central PPO process."), flush=True)
    if args.readable_log:
        print(f"Readable logs: {work}/pair-N/battle-readable.log", flush=True)
    if args.battle_log:
        print(f"Spectator logs: {work}/pair-N/battle.log (game messages and HP changes)", flush=True)
    print(f"Checkpoints: latest every {args.latest_every_updates} PPO update(s), plus first/final; "
          + (f"numbered backups every {args.checkpoint_every} learning transitions."
             if args.checkpoint_every and args.keep_checkpoints != 0 else "numbered backups disabled."), flush=True)
    if evaluation_config is not None:
        print(f"Validation: {evaluation_config['battles']} fixed battles; "
              + (f"every {args.eval_every} additional learning transitions, rounded to an update."
                 if args.eval_every else "paused (--eval-every 0)."), flush=True)
        print(f"Validation history: {run / 'evaluations.md'}; best weights: {checkpoints / 'best.zip'}. "
              "Training throughput includes validation pauses.", flush=True)
        if args.eval_stop_win_rate is not None:
            print(f"Success stop: at least {args.eval_stop_win_rate:.1%} wins/requested with zero truncations "
                  f"at {args.eval_stop_consecutive} consecutive validation checkpoints.", flush=True)
    print(f"PPO update diagnostics: {run / 'updates.jsonl'} (metrics object in each row).", flush=True)
    if args.backend == "showdown":
        limit = f"{args.log_max_mb:g} MiB" if args.log_max_mb else "unlimited size"
        traces = "on" if args.step_logs and args.keep_episode_logs != 0 else "off"
        print(f"Diagnostics: step traces {traces}; {limit} per log + {args.log_backups} backups; "
              f"completed episode trace limit={args.keep_episode_logs} (-1 means unlimited).", flush=True)
    try:
        if args.rule_based_opponent or args.opponent_pool:
            vec = FrozenOpponentVecEnv(vec, opponent, stochastic=args.opponent_stochastic,
                                       observation_version=args.observation_version)
            if continuing:
                for key in vec.switch_counts:
                    vec.switch_counts[key] = int(previous_switching.get(key, 0))
            if args.opponent_pool:
                print(f"Player 2 pool: {dict(zip(opponent.keys, opponent.probabilities.tolist()))}; sampled once per battle.", flush=True)
            else:
                print(f"Player 2: {opponent.VERSION} (observation-only rules; no learned weights).", flush=True)
        if args.opponent_checkpoint:
            frozen_path = args.opponent_checkpoint.expanduser().resolve()
            frozen_bytes = snapshots[str(frozen_path)]
            config["frozen_opponent"] = dict(path=config["opponent_checkpoint"], source_path=str(frozen_path),
                sha256=hashlib.sha256(frozen_bytes).hexdigest(),
                training_timesteps=opponent.num_timesteps, stochastic=args.opponent_stochastic,
                observation_version=opponent_version)
            switching = SwitchingOpponent(root, vec.num_envs//2, seed=args.seed, **switch_config) if switch_config["chance"] else None
            vec = FrozenOpponentVecEnv(vec, opponent, stochastic=args.opponent_stochastic,
                                       switching=switching, switch_log=work / "opponent-switches.jsonl",
                                       observation_version=args.observation_version)
            if continuing:
                for key in vec.switch_counts:
                    vec.switch_counts[key] = int(previous_switching.get(key, 0))
            config["opponent_switch"] = dict(**switch_config, heuristic_version=1,
                scope="training_only", minimum_hp_fraction=.35, minimum_score_gain=1)
            if switching:
                print(f"Opponent switch checks: {100*switch_config['chance']:g}% chance; "
                      f"{switch_config['cooldown']} normal decisions cooldown. Evaluation policies remain unmodified.", flush=True)
            print(f"Player 2 frozen at: {config['opponent_checkpoint']} ({opponent.num_timesteps} training timesteps).", flush=True)
        opponent_vec = vec if isinstance(vec, FrozenOpponentVecEnv) else None
        if args.shuffle_move_slots:
            from firered_rl.move_slot_shuffle import MoveSlotShuffle
            vec = MoveSlotShuffle(vec, seed=args.seed)
            print("Learner move slots: shuffled per Pokemon per battle; simulator actions mapped back. "
                  "Frozen opponents and validation use original move order.", flush=True)
        if args.resume:
            model = resumed_model
            model.set_env(vec)
            if continuing or "seed" in plan.explicit:
                # A requested branch seed must also reseed policy sampling/PPO.
                model.set_random_seed(args.seed)
            elif model.seed is not None:
                vec.seed(model.seed)
            if "learning_rate" in overrides:
                for group in model.policy.optimizer.param_groups:
                    group["lr"] = overrides["learning_rate"]
            config["resume_overrides"] = overrides
            config["resume_checkpoint"] = dict(path=str(args.resume),
                sha256=hashlib.sha256(snapshots[str(args.resume)]).hexdigest())
            print(f"Resuming {args.resume}; explicit PPO overrides: {overrides or 'none'}. Starting new battles.", flush=True)
        else:
            if args.policy_architecture == "matchup-switches":
                from firered_rl.matchup_switch_policy import MatchupSwitchPolicy
                policy_class, policy_kwargs = MatchupSwitchPolicy, {}
            elif args.policy_architecture == "matchup-moves":
                from firered_rl.matchup_move_policy import MatchupMovePolicy
                policy_class, policy_kwargs = MatchupMovePolicy, {}
            elif args.policy_architecture == "shared-moves":
                from firered_rl.shared_move_policy import SharedMovePolicy
                policy_class, policy_kwargs = SharedMovePolicy, {}
            else:
                policy_class = "MultiInputPolicy"
                policy_kwargs = dict(features_extractor_class=BattleFeatures,
                    features_extractor_kwargs=dict(features_dim=128), normalize_images=False,
                    net_arch=dict(pi=[128, 64], vf=[128, 64]))
            model = MaskablePPO(policy_class, vec, device=args.device, seed=args.seed,
                n_steps=args.rollout_steps, batch_size=args.batch_size, n_epochs=args.epochs,
                learning_rate=args.learning_rate, ent_coef=args.ent_coef, gamma=.99, gae_lambda=.95,
                target_kl=.03, verbose=0, policy_kwargs=policy_kwargs)
        model.set_logger(configure(str(work / "ppo"), ["csv"]))
        model.observation_version = args.observation_version
        model.policy_architecture = args.policy_architecture
        model.policy_architecture_version = getattr(model.policy, "ARCHITECTURE_VERSION", 1)
        config["policy_architecture_version"] = model.policy_architecture_version
        config["policy_parameters"] = sum(p.numel() for p in model.policy.parameters())
        if args.policy_architecture == "matchup-switches":
            from firered_rl.matchup_switch_policy import FEATURE_NAMES
            config["switch_features"] = dict(version=1,
                enabled=model.policy.features_extractor.bench_features_enabled,
                scope="shared residual on multi-option forced replacements only",
                feature_names=list(FEATURE_NAMES),
                whole_policy_training=True, migration=getattr(model, "switch_ablation", None))
        print(f"Policy architecture: {args.policy_architecture} "
              f"({config['policy_parameters']:,} parameters); observations: {args.observation_version}", flush=True)
        model.battle_backend = args.backend
        model.reward_settings = asdict(reward_config)
        if args.shuffle_move_slots:
            config["move_slot_shuffle"] = dict(version=vec.VERSION, seed=vec.shuffle_seed,
                scope="learners_only", schedule="per_party_member_per_episode",
                action_mapping="presented_to_simulator", validation="original_order")
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
        write_run_file(run / "config.json", json.dumps(config, indent=2)+"\n")
        write_run_file(metadata / "config.json", json.dumps(config, indent=2)+"\n")
        start = model.num_timesteps
        # A resumed model may pass initial validation without another update.
        # Its summary must still report the actual checkpoint training counter.
        progress.player_steps = start
        target = start + math.ceil(args.total_timesteps/rollout)*rollout
        if args.train_battles:
            print(f"Battle budget: {args.train_battles}; finish the current rollout when reached (may overshoot).", flush=True)
        next_checkpoint = start + args.checkpoint_every
        validation = PeriodicEvaluation(root, run, args, evaluation_setup) if evaluation_setup is not None else None
        if validation is not None and validation.due(model.num_timesteps):
            checkpoint_seconds_total += save_model(model, checkpoints / "latest.zip")
            latest_timesteps = model.num_timesteps
            validation.evaluate(checkpoints / "latest.zip", model.num_timesteps, reason="initial")
        if validation is not None:
            stopping = validation.stopping_status(model.num_timesteps)
        while (not (stopping and stopping["achieved"]) and
               (progress.attempts - starting_attempts < args.train_battles if args.train_battles else model.num_timesteps < target)):
            update_started = time.monotonic()
            print(f"Collecting {rollout} player transitions for update {iterations+1}; " +
                  (f"ended battles this session={progress.attempts-starting_attempts}/{args.train_battles}; timesteps={model.num_timesteps}"
                   if args.train_battles else f"timesteps={model.num_timesteps}/{target}"), flush=True)
            model.learn(total_timesteps=rollout, reset_num_timesteps=False, log_interval=None)
            iterations += 1
            session_iterations += 1
            if opponent_vec is not None:
                if args.opponent_pool:
                    write_run_file(work / "opponent-pool-metrics.json", json.dumps(opponent.metrics(), indent=2) + "\n")
                switches = opponent_vec.switching_metrics()
                (run / "switching.json").write_text(json.dumps(switches, indent=2)+"\n")
                for key, value in switches.items():
                    model.logger.record("switching/"+key, value)
                print(f"Voluntary switches: P1 {switches['learner_voluntary_switch_rate']:.1%}, "
                      f"P2 {switches['opponent_voluntary_switch_rate']:.1%}; "
                      f"opponent heuristic overrides={switches['heuristic_overrides']}", flush=True)
            for key, value in progress.reward_metrics().items():
                if value is not None:
                    model.logger.record("rollout/"+key, value)
            targets = []
            validation_due = validation is not None and validation.due(model.num_timesteps)
            if session_iterations == 1 or session_iterations % args.latest_every_updates == 0 or validation_due:
                targets.append(checkpoints / "latest.zip")
            if args.checkpoint_every and args.keep_checkpoints != 0 and model.num_timesteps >= next_checkpoint:
                targets.append(checkpoints / f"ppo_{model.num_timesteps:09d}.zip")
            checkpoint_seconds = save_model(model, *targets) if targets else 0.0
            if targets:
                prune_numbered_checkpoints(checkpoints, args.keep_checkpoints)
            checkpoint_seconds_total += checkpoint_seconds
            if checkpoints / "latest.zip" in targets:
                latest_timesteps = model.num_timesteps
            if args.checkpoint_every and model.num_timesteps >= next_checkpoint:
                next_checkpoint = model.num_timesteps + args.checkpoint_every
            update_seconds = time.monotonic() - update_started
            model.logger.record("time/checkpoint_seconds", checkpoint_seconds)
            model.logger.record("time/checkpoint_seconds_total", checkpoint_seconds_total)
            model.logger.record("time/update_seconds", update_seconds)
            model.logger.record("time/checkpoint_fraction", checkpoint_seconds / update_seconds if update_seconds else 0)
            metrics = update_metrics(model.logger.name_to_value)
            progress.update(model.num_timesteps, iterations, metrics)
            model.logger.dump(model.num_timesteps)
            saved = (f"{', '.join(path.name for path in targets)} saved ({checkpoint_seconds:.3f}s)"
                     if targets else "checkpoint save skipped")
            policy_loss = metrics.get("train/policy_gradient_loss")
            print(f"PPO update {iterations}: timesteps={model.num_timesteps}, "
                  f"policy_loss={policy_loss if policy_loss is not None else float('nan'):.5f}; {saved}", flush=True)
            if validation_due:
                validation.evaluate(checkpoints / "latest.zip", model.num_timesteps, reason="interval")
                stopping = validation.stopping_status(model.num_timesteps)
        if latest_timesteps != model.num_timesteps:
            checkpoint_seconds_total += save_model(model, checkpoints / "latest.zip", checkpoints / "final.zip")
            latest_timesteps = model.num_timesteps
        else:
            copy_started = time.monotonic()
            copy_checkpoint_file(checkpoints / "latest.zip", checkpoints / "final.zip")
            checkpoint_seconds_total += time.monotonic() - copy_started
        if validation is not None and validation.needs_final(model.num_timesteps):
            validation.evaluate(checkpoints / "final.zip", model.num_timesteps, reason="final")
            stopping = validation.stopping_status(model.num_timesteps)
        if stopping and stopping["achieved"]:
            stop_reason = "validation_success"
            print(f"Success target reached at {model.num_timesteps} timesteps; "
                  "training stopped and final checkpoint saved.", flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "interrupted"
        message = (f"Last saved PPO update preserved in checkpoints/latest.zip (timesteps={latest_timesteps}); "
                   "updates since that save are not preserved."
                   if (checkpoints / "latest.zip").is_file()
                   else "Stopped before the first PPO update was saved; no trained checkpoint was written.")
        print(f"Training interrupted. {message}", flush=True)
    except Exception as error:
        status = "failed"
        stop_reason = "failed"
        (run / "failure.json").write_text(json.dumps(dict(error=str(error), traceback=traceback.format_exc()), indent=2)+"\n")
        shutil.copyfile(run / "failure.json", metadata / "failure.json")
        print(f"PPO failed: {error}\nDiagnostics: {run}", flush=True)
    finally:
        vec.close()
        summary = progress.finish(status)
        if args.team_mixture is not None:
            summary["team_mixture_sampling"] = source.sampling_summary()
        if args.shuffle_bench_order:
            summary["bench_order_shuffle"] = source.get_summary()
        summary["stop_reason"] = stop_reason
        if stopping is not None:
            summary["early_stopping"] = stopping
        if opponent_vec is not None:
            summary["switching"] = opponent_vec.switching_metrics()
            (run / "switching.json").write_text(json.dumps(summary["switching"], indent=2)+"\n")
        summary["checkpointing"] = dict(latest_every_updates=args.latest_every_updates,
            checkpoint_every=args.checkpoint_every, keep_checkpoints=args.keep_checkpoints,
            latest_timesteps=latest_timesteps,
            seconds=checkpoint_seconds_total,
            fraction=checkpoint_seconds_total / summary["elapsed_seconds"] if summary["elapsed_seconds"] else 0)
        summary["session"] = dict(id=session_id, directory=str(work),
            starting_attempts=starting_attempts, ended_battles=progress.attempts-starting_attempts,
            updates=session_iterations, starting_timesteps=config.get("effective_ppo", {}).get("starting_timesteps"),
            recovered_checkpoint=str(args.resume) if args.resume else None, seed=args.seed)
        if validation is not None:
            summary["evaluation"] = dict(best=validation.best(), evaluations=len(validation.records),
                                         session_seconds=validation.seconds, training_rates_include_validation=True)
        temporary = run / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2)+"\n")
        temporary.replace(run / "summary.json")
        (metadata / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        if (run / "switching.json").exists():
            shutil.copyfile(run / "switching.json", metadata / "switching.json")
    print(f"Training {status}: {summary['completed_battles']} completed battles, "
          f"{summary['truncated_battles']} truncated; {summary['episodes_per_hour']:.1f} battles/hour\n"
          f"Checkpoints: {checkpoints}\n"
          f"Checkpoint writes: {checkpoint_seconds_total:.3f}s ({summary['checkpointing']['fraction']:.1%} of run).", flush=True)
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
