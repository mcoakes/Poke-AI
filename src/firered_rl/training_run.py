"""Select and safely reopen standalone PPO runs without importing training dependencies.

Selection is read-only. Call ``validate`` after loading inputs, then ``prepare``
only after all normal model/backend validation has succeeded. Keep its lease
open until workers and checkpoint writers have stopped.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import uuid
import zipfile


class RunCancelled(Exception):
    """The user chose to leave an existing run untouched."""


_HYPERPARAMETERS = {"rollout_steps", "batch_size", "epochs", "learning_rate", "ent_coef"}
_SELECTORS = {"states", "state_dir", "teams", "team_dir", "team_profile", "training_profile", "team_mixture"}
_IMMUTABLE = {"backend", "observation_version", "mirror_teams", "max_steps", "seed",
              "opponent_stochastic", "opponent_switch_chance", "opponent_switch_cooldown", "rule_based_opponent",
              "shuffle_move_slots", "shuffle_bench_order", "policy_architecture"}
_NOT_RESTORED = _HYPERPARAMETERS | {"resume", "output_dir", "if_exists", "session_observation_version"}
_CHECKPOINT = re.compile(r"(?:latest|final|ppo_\d+)\.zip\Z")


def explicit_arguments(parser, argv):
    """Return destinations supplied by the caller, including aliases and --no-X."""
    result = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is None and option.startswith("--") and parser.allow_abbrev:
            # Match argparse's unambiguous long-option abbreviation handling.
            matches = {a.dest: a for key, a in parser._option_string_actions.items()
                       if key.startswith(option)}
            if len(matches) == 1:
                action = next(iter(matches.values()))
        if action is not None:
            result.add(action.dest)
    return frozenset(result)


def checkpoint_timesteps(path):
    """Read public scalar metadata, never unpickle an SB3 archive during selection."""
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("data")
            if info.file_size > 32 * 1024 * 1024:
                raise ValueError("checkpoint metadata is too large")
            value = json.loads(archive.read(info))["num_timesteps"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid num_timesteps")
        if not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError("invalid num_timesteps")
        return int(value)
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot read training checkpoint metadata: {path}: {error}") from error


def newest_run_checkpoint(run):
    directory = Path(run) / "checkpoints"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("existing run has no regular checkpoints directory")
    candidates = []
    for path in directory.iterdir():
        if not _CHECKPOINT.fullmatch(path.name) or path.is_symlink() or not path.is_file():
            continue
        try:
            steps = checkpoint_timesteps(path)
        except ValueError:
            continue
        preference = 2 if path.name == "final.zip" else 1 if path.name == "latest.zip" else 0
        candidates.append((steps, preference, path.name, path))
    if not candidates:
        raise ValueError(f"no readable PPO checkpoint in {directory}; use erase or choose another run")
    return max(candidates)[-1]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path, directory):
    return path == directory or directory in path.parents


def _guard_path(run):
    return run.parent / f".{run.name}.training.lock"


def _lock_file(path, *, create):
    flags = os.O_RDWR | (os.O_CREAT if create else 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileNotFoundError:
        if not create:
            return None
        raise
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as error:
        os.close(fd)
        raise ValueError(f"training run is already active/locked: {path.parent / path.name[1:-14]}") from error
    return fd


def _assert_unlocked(run):
    fd = _lock_file(_guard_path(run), create=False)
    if fd is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _assert_owned_paths(run):
    """Files that continuation can append/replace must stay inside this run."""
    root_files = ("config.json", "summary.json", "summary.tmp", "switching.json",
                  "failure.json", "episodes.jsonl", "updates.jsonl", "battle-blocks.jsonl",
                  "evaluation-history.json", "evaluations.csv", "evaluations.md")
    for name in root_files:
        path = run / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"run output must be a regular file, not a symlink: {path}")
    for name in ("inputs", "sessions", "checkpoints", "evaluations"):
        directory = run / name
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError(f"run output must be a regular directory, not a symlink: {directory}")
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError(f"refusing to write through a linked run output: {path}")
            if name == "inputs" and path.name == "evaluation" and path.is_dir():
                for child in path.rglob("*"):
                    if child.is_symlink():
                        raise ValueError(f"refusing linked validation input: {child}")
            if name == "sessions" and path.is_dir() and re.fullmatch(r"\d{6}", path.name):
                for filename in ("config.json", "summary.json", "switching.json", "failure.json"):
                    target = path / filename
                    if target.is_symlink() or (target.exists() and not target.is_file()):
                        raise ValueError(f"session metadata must be a regular file: {target}")


def _check_safe_run(run, project):
    if run.is_symlink() or any(parent.is_symlink() for parent in run.parents):
        raise ValueError("refusing to continue or erase a run through a symlink")
    run = run.resolve()
    protected = {Path("/"), Path.home().resolve(), Path.cwd().resolve(), project.resolve()}
    if any(_inside(path, run) for path in protected):
        raise ValueError(f"refusing to modify a project, home, working directory, or its parent: {run}")
    if not run.is_dir():
        raise ValueError(f"training output already exists and is not a directory: {run}")
    if run.name == "train" and (re.fullmatch(r"attempt-\d+", run.parent.name)
                                 or re.fullmatch(r"stage-\d+", run.parent.name)):
        raise ValueError("campaign-managed training must be continued with run_curriculum.py")
    for parent in run.parents:
        if any((parent / name).is_file() for name in ("campaign.json", "curriculum-run.json", "curriculum.json")):
            raise ValueError("campaign-managed training must be continued with run_curriculum.py")
    _assert_owned_paths(run)
    config_path = run / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("refusing to modify an unrecognized directory: PPO config.json is missing")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read saved PPO configuration: {config_path}") from error
    if (not isinstance(config, dict) or config.get("algorithm") != "MaskablePPO"
            or config.get("backend") not in {"showdown", "mgba"}
            or not isinstance(config.get("teams" if config.get("backend") == "showdown" else "states"), dict)):
        raise ValueError("refusing to modify a directory without a recognized PPO training configuration")
    _assert_unlocked(run)
    return config


def _saved_arguments(config):
    saved = dict(config)
    original = config.get("saved_args")
    if isinstance(original, dict):
        saved.update(original)
    backend = config["backend"]
    description = config["teams" if backend == "showdown" else "states"]
    entries = description.get("teams" if backend == "showdown" else "checkpoints", [])
    for key in _SELECTORS:
        saved[key] = None
    mode = description.get("mode")
    if mode == "fixed":
        if len(entries) != 2:
            raise ValueError("saved fixed team/state selection must contain two files")
        saved["teams" if backend == "showdown" else "states"] = [p["path"] for p in entries]
    elif mode == "team_profile":
        saved["team_profile"] = description.get("team_profile")
        saved["team_size"] = description.get("team_size")
    elif mode == "team_mixture" and backend == "showdown":
        saved["team_mixture"] = description["manifest"]["path"]
    elif mode == "random_directory":
        selector = "team_dir" if backend == "showdown" else "state_dir"
        supplied = original if isinstance(original, dict) else config
        if backend == "showdown" and supplied.get("training_profile"):
            saved["training_profile"] = supplied["training_profile"]
        elif supplied.get(selector):
            saved[selector] = supplied[selector]
        elif entries and len({str(Path(p["path"]).parent) for p in entries}) == 1:
            saved[selector] = str(Path(entries[0]["path"]).parent)
        else:
            raise ValueError("cannot reconstruct saved team/state directory; start a new run with --resume")
    else:
        raise ValueError(f"unrecognized saved team/state selection mode: {mode}")
    saved["mirror_teams"] = description.get("mirror_teams", config.get("mirror_teams", False))
    saved["backend"] = config["backend"]
    saved["observation_version"] = config.get("observation_version") or "v1"
    saved["policy_architecture"] = config.get("policy_architecture") or "flat"
    frozen = config.get("frozen_opponent")
    saved["opponent_checkpoint"] = frozen.get("path") if frozen else config.get("opponent_checkpoint")
    saved["rule_based_opponent"] = bool(config.get("rule_based_opponent", False))
    saved["shuffle_move_slots"] = bool(config.get("shuffle_move_slots", False))
    saved["shuffle_bench_order"] = bool(config.get("shuffle_bench_order", False))
    saved["opponent_pool"] = config.get("opponent_pool")
    return saved


def _convert_saved(action, value):
    if value is None or action.type is None:
        return value
    if action.nargs in ("+", "*") or isinstance(action.nargs, int):
        return [action.type(str(item)) for item in value]
    return action.type(str(value))


def _restore(parser, args, explicit, config):
    saved = _saved_arguments(config)
    for key in _IMMUTABLE & explicit:
        expected = saved.get(key, getattr(args, key))
        if getattr(args, key) != expected:
            raise ValueError(f"--{key.replace('_', '-')} differs from this run; use --resume with a new --output-dir")
    supplied_selection = bool(_SELECTORS & explicit)
    for action in parser._actions:
        key = action.dest
        if key in explicit or key in _NOT_RESTORED or key not in saved:
            continue
        if supplied_selection and key in _SELECTORS:
            continue
        setattr(args, key, _convert_saved(action, saved[key]))
    if "total_timesteps" in explicit and "train_battles" not in explicit:
        args.train_battles = None
    if "train_battles" in explicit and "total_timesteps" in explicit:
        raise ValueError("choose --train-battles or --total-timesteps for this continuation")
    was_frozen = bool(saved.get("opponent_checkpoint"))
    if bool(getattr(args, "opponent_pool", None)) != bool(saved.get("opponent_pool")):
        raise ValueError("cannot change opponent-pool mode while continuing; start a new run with --resume")
    if bool(getattr(args, "opponent_checkpoint", None)) != was_frozen:
        raise ValueError("cannot change frozen-opponent/self-play mode while continuing the same run")
    if was_frozen and str(args.opponent_checkpoint).lower() in {"latest", "later"}:
        # Repeating the original command must not promote the frozen opponent
        # to a newer global learner checkpoint. Its saved fingerprint is still
        # checked against loaded bytes during validate().
        args.opponent_checkpoint = Path(saved["opponent_checkpoint"])


@dataclass
class RunLease:
    run: Path
    fd: int

    def close(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class RunPlan:
    run: Path
    action: str
    saved_config: dict | None
    explicit: frozenset
    project: Path
    checkpoint: Path | None = None
    config_sha256: str | None = None
    checkpoint_sha256: str | None = None
    validated: bool = False

    def validate(self, source_description, reward, frozen_sha256=None):
        """Compare loaded immutable inputs, not merely command-line spellings."""
        if self.action == "continue":
            config = self.saved_config
            saved_source = config["teams" if config["backend"] == "showdown" else "states"]
            if saved_source != source_description:
                raise ValueError("team/state selection or file content changed; start a new run with --resume")
            if config.get("reward") != reward:
                raise ValueError("reward configuration changed; start a new run with --resume")
            frozen = config.get("frozen_opponent")
            expected = frozen.get("sha256") if frozen else None
            if expected != frozen_sha256:
                raise ValueError("frozen opponent checkpoint changed; start a new run with --resume")
        self.validated = True

    def prepare(self, *, input_paths=()):
        """Create/continue/erase only after all input and model validation succeeds.

        Callers must include source files/directories, ROM, reward file, and any
        resume/opponent models in input_paths. Erase refuses inputs it would delete.
        """
        if not self.validated:
            raise ValueError("run inputs must be validated before preparing the output directory")
        if self.action == "erase":
            for path in input_paths:
                if path is not None and _inside(Path(path).expanduser().resolve(), self.run):
                    raise ValueError(f"cannot erase a run containing a selected training input: {path}")
        self.run.parent.mkdir(parents=True, exist_ok=True)
        fd = _lock_file(_guard_path(self.run), create=True)
        lease = RunLease(self.run, fd)
        try:
            if self.action in {"continue", "erase"}:
                _assert_owned_paths(self.run)
                # Check again under the exclusive lease. Selection/preflight may
                # have taken long enough for another writer to finish a session.
                if self.run.is_symlink() or _sha256(self.run / "config.json") != self.config_sha256:
                    raise ValueError("run configuration changed during preflight; retry the command")
                if self.action == "continue" and (
                        _sha256(self.checkpoint) != self.checkpoint_sha256
                        or newest_run_checkpoint(self.run) != self.checkpoint):
                    raise ValueError("run checkpoint changed during preflight; retry the command")
            if self.action == "erase":
                quarantine = self.run.with_name(f".{self.run.name}.erasing-{uuid.uuid4().hex}")
                self.run.rename(quarantine)
                try:
                    self.run.mkdir()
                except BaseException:
                    quarantine.rename(self.run)
                    raise
                shutil.rmtree(quarantine)
            elif self.action == "new":
                self.run.mkdir(exist_ok=False)
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps({"pid": os.getpid(), "run": str(self.run)}).encode())
            return lease
        except BaseException:
            lease.close()
            raise


def _prompt(input_fn, message, run):
    try:
        return input_fn(message).strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise RunCancelled(f"Cancelled; existing run left unchanged: {run}") from error


def select_training_run(parser, args, argv, project, *, input_fn=input, stream=None):
    """Select a run and restore omitted flags before dependencies/models are loaded."""
    project = Path(project).resolve()
    explicit = explicit_arguments(parser, argv)
    supplied = getattr(args, "output_dir", None)
    run = Path(supplied or project / ".local/run" / datetime.now().strftime("ppo-%Y%m%d-%H%M%S-%f")).expanduser().absolute()
    action = getattr(args, "if_exists", "ask")
    if not run.exists() and not run.is_symlink():
        args.output_dir = run
        return RunPlan(run, "new", None, explicit, project)
    if action == "cancel":
        raise RunCancelled(f"Cancelled; existing run left unchanged: {run}")
    if action == "ask":
        stream = stream or sys.stdin
        if not stream.isatty():
            raise ValueError("output directory exists; non-interactive runs require --if-exists continue, erase, or cancel")
        answer = _prompt(input_fn, f"Run exists: {run}\n[c]ontinue / [e]rase / [q]cancel [q]: ", run).lower()
        action = {"c": "continue", "continue": "continue", "e": "erase", "erase": "erase"}.get(answer)
        if action is None:
            raise RunCancelled(f"Cancelled; existing run left unchanged: {run}")
        if action == "erase":
            confirmation = _prompt(input_fn, f"Erase this run and all its contents? Type {run.name!r} to confirm: ", run)
            if confirmation not in {run.name, str(run)}:
                raise RunCancelled(f"Cancelled; existing run left unchanged: {run}")
    config = _check_safe_run(run, project)
    run = run.resolve()
    plan = RunPlan(run, action, config, explicit, project, config_sha256=_sha256(run / "config.json"))
    if action == "continue":
        checkpoint = newest_run_checkpoint(run)
        selected_resume = getattr(args, "resume", None)
        if "resume" in explicit and selected_resume is not None and str(selected_resume).lower() not in {"latest", "later"}:
            if Path(selected_resume).expanduser().resolve() != checkpoint:
                raise ValueError("same-run continuation uses its most advanced checkpoint; omit --resume or start a new run")
        _restore(parser, args, explicit, config)
        args.resume = checkpoint
        plan.checkpoint = checkpoint
        plan.checkpoint_sha256 = _sha256(checkpoint)
    args.output_dir = run
    return plan
