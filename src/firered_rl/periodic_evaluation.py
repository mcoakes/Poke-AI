"""Fixed-suite validation between PPO updates, with bounded results and best weights.

Evaluation runs in a child process: its model loads, RNGs and simulation workers
cannot change the learner or the training opponent. JSON history is authoritative;
CSV/Markdown and best.zip can be reconstructed after an interrupted publication.
"""
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .runtime_logs import RotatingLog


def atomic_write(path, data):
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"output must be a regular file: {path}")
    if isinstance(data, str):
        data = data.encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def suite_opponent_path(suite_path, entry):
    """Prefer content-addressed, hash-checked copies beside a saved suite."""
    if suite_path is not None:
        bundled = Path(suite_path).expanduser().parent / "opponents" / f"{entry['sha256']}.zip"
        if bundled.is_file():
            return bundled
    return Path(entry["path"]).expanduser()


def preserve_candidate(run, data, info):
    """Keep the exact loaded bytes, rather than reserializing the inference model."""
    if sha256(data) != info["sha256"]:
        raise ValueError("evaluation candidate snapshot disagrees with loaded model")
    target = Path(run) / "candidate.zip"
    atomic_write(target, data)
    info["snapshot_path"] = str(target.resolve())
    return target


class EvaluationSetup:
    """Read-only preflight before a training run is created or reopened."""
    def __init__(self, args, plan, reward, frozen_sha256):
        from .evaluation import digest, validate_suite

        self.suite_path = Path(args.eval_suite).expanduser().resolve()
        self.suite = validate_suite(json.loads(self.suite_path.read_text()))
        if args.backend != "showdown" or self.suite["backend"] != "showdown":
            raise ValueError("periodic --eval-suite currently requires Showdown training and a Showdown suite")
        environments = args.eval_environments or args.environments
        if environments < 2 or environments % 2:
            raise ValueError("--eval-environments must be even and at least 2")
        rules = self.suite.get("rule_opponent")
        if rules:
            from .rule_opponent import RuleBasedOpponent
            if rules != RuleBasedOpponent(plan.project).identity:
                raise ValueError("rule opponent implementation differs from the saved evaluation suite")
        if not self.suite.get("opponents") and not rules and not args.opponent_checkpoint:
            raise ValueError("--eval-suite needs a saved opponent pool, rule opponent, or a frozen --opponent-checkpoint")
        self.inputs = [self.suite_path]
        self.reward = reward
        self.reference = None
        if not self.suite.get("opponents") and not rules:
            self.reference = Path(args.opponent_checkpoint).read_bytes()
            if sha256(self.reference) != frozen_sha256:
                raise ValueError("evaluation reference changed during preflight")
        self.opponents = {}
        for entry in self.suite.get("opponents", {}).values():
            if entry["kind"] == "random":
                continue
            if entry["kind"] == "rule_based":
                from .rule_opponent import RuleBasedOpponent
                if entry["identity"] != RuleBasedOpponent(plan.project).identity:
                    raise ValueError("pooled rule opponent differs from the saved evaluation suite")
                continue
            path = suite_opponent_path(self.suite_path, entry).resolve()
            pinned = plan.run / "inputs/evaluation/opponents" / f"{entry['sha256']}.zip"
            if plan.action == "continue" and pinned.is_file():
                path = pinned
            data = path.read_bytes()
            if sha256(data) != entry["sha256"]:
                raise ValueError(f"evaluation opponent checkpoint hash mismatch: {path}")
            self.inputs.append(path)
            self.opponents[entry["sha256"]] = data
        self.identity = dict(suite_sha256=digest(self.suite), reward_sha256=digest(reward),
                             reference_sha256=None if self.suite.get("opponents") or rules else frozen_sha256,
                             selection_metric="wins / requested battles")
        previous = (plan.saved_config or {}).get("evaluation") if plan.action == "continue" else None
        if previous and previous.get("identity") != self.identity:
            raise ValueError("evaluation suite or reference changed; use --resume with a new --output-dir")
        self.history = dict(identity=self.identity, records=[])
        history_path = plan.run / "evaluation-history.json"
        if plan.action == "continue" and history_path.exists():
            self.history = json.loads(history_path.read_text())
            if self.history.get("identity") != self.identity or not isinstance(self.history.get("records"), list):
                raise ValueError("saved evaluation history disagrees with this suite")
            for record in self.history["records"]:
                if not re.fullmatch(r"evaluation-\d{6}", record.get("directory", "")):
                    raise ValueError("invalid evaluation history directory")

    def install(self, run, args):
        directory = run / "inputs" / "evaluation"
        directory.mkdir(parents=True, exist_ok=True)
        opponents = directory / "opponents"
        opponents.mkdir(exist_ok=True)
        for checksum, data in self.opponents.items():
            target = opponents / f"{checksum}.zip"
            if not target.is_file() or sha256(target.read_bytes()) != checksum:
                atomic_write(target, data)
        if self.reference is not None:
            atomic_write(directory / "reference.zip", self.reference)
            self.reference = None
        atomic_write(directory / "reward.json", json.dumps(self.reward, indent=2) + "\n")
        self.opponents.clear()
        args.eval_suite = directory / "suite.json"
        atomic_write(args.eval_suite, json.dumps(self.suite, indent=2) + "\n")
        return dict(identity=self.identity, suite=str(args.eval_suite), battles=len(self.suite["cases"]))


class PeriodicEvaluation:
    def __init__(self, root, run, args, setup):
        self.root, self.run, self.args = Path(root), Path(run), args
        self.history = setup.history
        self.requested = len(setup.suite["cases"])
        self.records = self.history["records"]
        self.directory = self.run / "evaluations"
        self.directory.mkdir(exist_ok=True)
        self.seconds = 0.0
        self._publish_views()

    def completed(self):
        return [r for r in self.records if r["status"] == "completed"]

    def best(self):
        # Python max preserves the earlier record on ties.
        return max(self.completed(), key=lambda r: r["wins"], default=None)

    def due(self, timesteps):
        completed = self.completed()
        return bool(self.args.eval_every and
                    (not completed or timesteps >= completed[-1]["timesteps"] + self.args.eval_every))

    def needs_final(self, timesteps):
        completed = self.completed()
        return bool(self.args.eval_every and
                    (not completed or timesteps != completed[-1]["timesteps"]))

    def stopping_status(self, timesteps):
        """Reconstruct a success streak from validated history, including on resume.

        Only the current checkpoint can trigger a stop. Repeated evaluations of
        the same timestep cannot count as evidence from another PPO update.
        Failed, interrupted, truncated, or below-target evaluations end a streak.
        """
        threshold = getattr(self.args, "eval_stop_win_rate", None)
        required = getattr(self.args, "eval_stop_consecutive", 2)
        result = dict(enabled=threshold is not None, threshold=threshold,
                      required_consecutive=required, consecutive=0, achieved=False,
                      timesteps=int(timesteps), evaluation=None, checkpoint_sha256=None)
        if threshold is None or not self.records or self.records[-1].get("timesteps") != timesteps:
            return result
        previous_step = None
        for record in reversed(self.records):
            steps, wins, requested = (record.get(key) for key in ("timesteps", "wins", "requested"))
            if (record.get("status") != "completed" or record.get("truncated") != 0
                    or type(steps) is not int or steps < 0
                    or (previous_step is not None and steps >= previous_step)
                    or type(wins) is not int or type(requested) is not int
                    or requested != self.requested or not 0 <= wins <= requested
                    or requested <= 0 or wins / requested < threshold):
                break
            result["consecutive"] += 1
            previous_step = steps
        result["achieved"] = result["consecutive"] >= required
        result["evaluation"] = self.records[-1].get("evaluation")
        result["checkpoint_sha256"] = self.records[-1].get("checkpoint_sha256")
        return result

    def _publish_views(self):
        best = self.best()
        if best is not None:
            target = self.run / "checkpoints" / "best.zip"
            if not target.is_file() or sha256(target.read_bytes()) != best["checkpoint_sha256"]:
                source = self.directory / best["directory"] / "candidate.zip"
                data = source.read_bytes()
                if sha256(data) != best["checkpoint_sha256"]:
                    raise ValueError("best evaluation candidate hash mismatch")
                atomic_write(target, data)
            atomic_write(target.with_suffix(".json"), json.dumps(
                dict(**best, **self.history["identity"]), indent=2) + "\n")
        fields = ["evaluation", "timesteps", "reason", "status", "wins", "requested", "win_rate",
                  "random_win_rate", "dumb_win_rate", "smart_win_rate", "learned_win_rate", "rules_win_rate", "truncated", "reward_mean",
                  "new_best", "seconds", "directory", "checkpoint_sha256"]
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.records)
        atomic_write(self.run / "evaluations.csv", stream.getvalue())
        lines = ["# Training validation history", "",
                 "Win rates use all requested battles; draws and truncations count as non-wins. "
                 "Ties retain the earlier best checkpoint.", "",
                 "| Evaluation | Timesteps | Overall | Random | Dumb | Smart | Learned | Rules | Truncated | Result |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        def percentage(value):
            return "—" if value is None else f"{value:.1%}"
        for row in self.records:
            rates = " | ".join(percentage(row.get(key)) for key in
                               ("win_rate", "random_win_rate", "dumb_win_rate", "smart_win_rate", "learned_win_rate", "rules_win_rate"))
            result = "BEST" if row.get("new_best") else row["status"]
            lines.append(f"| {row['evaluation']} | {row['timesteps']:,} | {rates} | "
                         f"{row.get('truncated', '—')} | {result} |")
        atomic_write(self.run / "evaluations.md", "\n".join(lines) + "\n")

    def _command(self, checkpoint, output):
        a = self.args
        command = [sys.executable, str(self.root / "mgba/evaluate_ppo.py"),
                   "--checkpoint", str(checkpoint), "--suite", str(a.eval_suite),
                   "--reward-config", str(Path(a.eval_suite).parent / "reward.json"), "--backend", "showdown",
                   "--environments", str(a.eval_environments or a.environments),
                   "--showdown-workers", a.showdown_workers, "--device", a.device,
                   "--torch-threads", str(a.torch_threads), "--node", a.node,
                   "--command-timeout", str(a.command_timeout), "--output-dir", str(output),
                   "--no-readable-log", "--no-step-logs", "--log-max-mb", str(a.log_max_mb),
                   "--log-backups", str(a.log_backups), "--keep-episode-logs", "0",
                   "--battle-log" if a.battle_log else "--no-battle-log"]
        if self.history["identity"]["reference_sha256"] is not None:
            command += ["--opponent-checkpoint", str(Path(a.eval_suite).parent / "reference.zip")]
        return command

    def _execute(self, command):
        log = RotatingLog(self.directory / "worker.log", max_bytes=2*1024*1024, backups=1)
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, start_new_session=True) as process:
            try:
                for line in process.stdout:
                    log.append(line)
                return process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise

    def evaluate(self, checkpoint, timesteps, *, reason):
        from .evaluation import digest

        existing = [int(p.name.split("-")[1]) for p in self.directory.iterdir()
                    if re.fullmatch(r"evaluation-\d{6}", p.name)]
        number = max(existing + [r["evaluation"] for r in self.records], default=0) + 1
        output = self.directory / f"evaluation-{number:06d}"
        checksum = sha256(Path(checkpoint).read_bytes())
        record = dict(evaluation=number, timesteps=int(timesteps), reason=reason, status="failed",
                      directory=output.name, checkpoint_sha256=checksum, new_best=False,
                      utc=datetime.now(timezone.utc).isoformat())
        started = time.monotonic()
        print(f"Validation {number}: timesteps={timesteps}; training paused; results: {output}", flush=True)
        try:
            code = self._execute(self._command(checkpoint, output))
            if code != 0:
                raise ValueError(f"evaluation exited {code}; see {self.directory / 'worker.log'}")
            summary = json.loads((output / "summary.json").read_text())
            config = json.loads((output / "config.json").read_text())
            candidate = output / "candidate.zip"
            if (summary["status"] != "completed" or summary["attempted_battles"] != summary["requested_battles"]
                    or summary["requested_battles"] != self.requested
                    or summary["suite_sha256"] != self.history["identity"]["suite_sha256"]
                    or digest(config["reward"]) != self.history["identity"]["reward_sha256"]
                    or config["models"]["candidate"]["sha256"] != checksum
                    or config["models"]["candidate"]["training_timesteps"] != timesteps
                    or sha256(candidate.read_bytes()) != checksum):
                raise ValueError("incomplete or mismatched validation result; best checkpoint unchanged")
            counts = [summary[k] for k in ("candidate_wins", "opponent_wins", "draws", "truncated")]
            if (any(type(n) is not int or n < 0 for n in counts) or sum(counts) != self.requested
                    or sum(counts[:3]) != summary["completed_battles"]):
                raise ValueError("invalid validation counts; best checkpoint unchanged")
            wins, requested = summary["candidate_wins"], summary["requested_battles"]
            record.update(status="completed", wins=wins, requested=requested, win_rate=wins/requested,
                          truncated=summary["truncated"], reward_mean=summary["candidate_reward_mean"],
                          new_best=self.best() is None or wins > self.best()["wins"])
            for group in ("random", "dumb", "smart", "learned", "rules"):
                record[group + "_win_rate"] = summary.get("by_opponent_group", {}).get(group, {}).get("win_rate")
        except BaseException as error:
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            record["error"] = str(error)
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            self.seconds += record["seconds"]
            self.records.append(record)
            atomic_write(self.run / "evaluation-history.json", json.dumps(self.history, indent=2) + "\n")
            self._publish_views()
        best = self.best()
        print(f"Validation {number}: wins={wins}/{requested} ({wins/requested:.1%}); "
              f"{'NEW BEST' if record['new_best'] else 'best unchanged'}; "
              f"best={best['win_rate']:.1%} at {best['timesteps']} timesteps", flush=True)
        keep = {r["directory"] for r in self.records[-self.args.keep_evaluations:]}
        keep.add(best["directory"])
        for old in self.records:
            path = self.directory / old["directory"]
            if old["directory"] not in keep and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
        return record
