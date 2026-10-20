#!/usr/bin/env python3
"""Train and validate a growing Showdown team pool, stopping when a stage needs work."""
import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from firered_rl.checkpoints import resolve_checkpoints
from firered_rl.curriculum import (DEFAULT_RULES, OPPONENT_MODES, digest, evaluation_rates, load_curriculum,
    promotion_report, read_json, report_markdown, stage_suites, validate_rules, write_json, team_groups)
from firered_rl.linked_reward import LinkedRewardConfig
from firered_rl.switch_opponent import switch_settings
from firered_rl.evaluation_pool import (copy_checkpoint, settings as evaluation_settings,
    pool_promotion, prepare_pool, verify_pool)
from firered_rl.retention import plan_campaign_retention, apply_campaign_retention


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRICULUM = ROOT / "config/curricula/type_progression/curriculum.json"
EXECUTION = dict(environments=48, showdown_workers="process", timesteps=100000,
                 rollout_steps=128, batch_size=128, epochs=None, learning_rate=None, ent_coef=None,
                 device="cpu", torch_threads=1, node="node", command_timeout=120,
                 readable_log=True, battle_log=True, latest_every_updates=1, checkpoint_every=1024,
                 keep_checkpoints=2, step_logs=False, log_max_mb=10, log_backups=2,
                 keep_episode_logs=2, retention="compact", keep_attempts=2, keep_best=2,
                 keep_debug_attempts=0)
IMMUTABLE_FLAGS = ("curriculum", "resume", "opponent_checkpoint", "reward_config",
                   "promotion_rules", "eval_episodes", "max_steps", "seed", "opponent_mode", "train_battles",
                   "opponent_switch_chance", "opponent_switch_cooldown")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def retention_count(value):
    value = int(value)
    if value < -1:
        raise argparse.ArgumentTypeError("must be >= -1 (-1 means unlimited)")
    return value


def nonnegative(value):
    import math
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update_evaluation_settings(args, run, config):
    """Change future gates without editing reports or a stage's training opponent."""
    old = dict(promotion_win_rate=config.get("promotion_win_rate", .75),
               evaluation=evaluation_settings(config))
    updated = {**old, "evaluation": old["evaluation"].copy()}
    mapping = dict(eval_opponents="mode", promotion_basic_win_rate="basic_win_rate",
                   promotion_smart_win_rate="smart_win_rate", promotion_require="require")
    for flag, key in mapping.items():
        value = getattr(args, flag, None)
        if value is not None:
            updated["evaluation"][key] = value
    rate = getattr(args, "promotion_win_rate", None)
    if rate is not None:
        updated["promotion_win_rate"] = rate
    evaluation = updated["evaluation"]
    if any(not 0 < value <= 1 for value in (updated["promotion_win_rate"],
            evaluation["basic_win_rate"], evaluation["smart_win_rate"])):
        raise ValueError("promotion win rates must be fractions in (0, 1], such as .75")
    pool_flags = (*mapping, "eval_smart_dir", "eval_dumb_checkpoint")
    changing = rate is not None or any(getattr(args, flag, None) is not None for flag in pool_flags)
    if (changing or evaluation["mode"] == "mixed") and not config.get("train_battles"):
        raise ValueError("editable promotion gates and mixed evaluation require a --train-battles campaign; "
                         "start a new run to switch from the original four-evaluation mode")
    if evaluation["mode"] == "mixed":
        if rate is not None:
            raise ValueError("mixed evaluation uses --promotion-basic-win-rate and --promotion-smart-win-rate; "
                             "omit --promotion-win-rate")
        if config["eval_episodes"] % 20:
            raise ValueError("mixed evaluation needs the saved --eval-episodes to be divisible by 20")
    elif any(getattr(args, flag, None) is not None for flag in pool_flags if flag != "eval_opponents"):
        raise ValueError("pool options require --eval-opponents mixed (or a campaign already using it)")
    smart_dir = getattr(args, "eval_smart_dir", None)
    if smart_dir is not None:
        smart_dir = smart_dir.expanduser().resolve()
        if not smart_dir.is_dir() or not any(smart_dir.glob("smart_v*.zip")):
            raise ValueError("--eval-smart-dir must contain saved smart_v*.zip checkpoints")
        evaluation["smart_dir"] = str(smart_dir)
    dumb = getattr(args, "eval_dumb_checkpoint", None)
    if dumb is not None:
        source = resolve_checkpoints([dumb], ROOT)[0]
        target = run / "evaluation-models" / f"{sha256(source)}.zip"
        checksum = copy_checkpoint(source, target)
        evaluation["dumb_checkpoint"] = dict(path=str(target), sha256=checksum)
    # The JSON history is authoritative; the readable JSONL is rebuilt on save.
    if updated != old:
        revision = config.get("settings_revision", 0) + 1
        config.setdefault("rule_changes", []).append(dict(revision=revision,
            time=datetime.now(timezone.utc).isoformat(), before=old, after=updated))
        config["settings_revision"] = revision
        print(f"Evaluation rules revision {revision}: applies to the next evaluation; prior results retained.", flush=True)
    config.update(updated)


def save_campaign(run, config):
    write_json(run / "campaign.json", config)
    target = run / "rule-changes.jsonl"
    temporary = target.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(change) + "\n" for change in config.get("rule_changes", [])))
    temporary.replace(target)


@contextmanager
def campaign_lock(run):
    with (run / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another curriculum command is running in {run}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_child(command, root):
    process = subprocess.Popen(command, cwd=root)
    try:
        code = process.wait()
    except KeyboardInterrupt:
        # Let the child close its Python workers and Node sessions, including workers
        # with their own process groups. The campaign never advances on interruption.
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    if code:
        raise RuntimeError(f"{Path(command[1]).name} exited with status {code}")


class Campaign:
    def __init__(self, run, config, execution, *, execute=None):
        self.run, self.config, self.execution = Path(run), config, execution
        self.execute = execute or run_child
        self.opponent = self.run / "inputs/opponent.zip"
        # Old campaigns did not store a mode; their original reference stays fixed.
        self.opponent_mode = config.get("opponent_mode", "fixed")
        if self.opponent_mode not in OPPONENT_MODES:
            raise ValueError("saved opponent mode must be fixed or previous-stage")
        self.stage_opponent = None
        self.reward = self.run / "inputs/reward.json"
        self.paths = [self.run / path for path in config["teams"]]
        self.groups = ([[self.run / p for p in group] for group in config["team_groups"]]
                       if "team_groups" in config else team_groups(self.paths))
        self.evaluation_groups = ([[self.run / p for p in group] for group in config["validation_groups"]]
                                  if config.get("validation_groups") else self.groups)
        self.count = len(config["curriculum"]["stages"])
        if (len(self.groups) != self.count + 1 or len(self.evaluation_groups) != self.count + 1
                or any(not g for g in self.groups + self.evaluation_groups)
                or [p for g in self.groups for p in g] != self.paths):
            raise ValueError("saved campaign team groups disagree with its stages/teams")
        if any(str(p.relative_to(self.run)) not in config["input_hashes"]
               for group in self.groups + self.evaluation_groups for p in group):
            raise ValueError("saved campaign team group references an unverified input")
        for relative, expected in config["input_hashes"].items():
            if sha256(self.run / relative) != expected:
                raise ValueError(f"saved curriculum input changed: {relative}; start a new --run-dir")

    def stage_dir(self, stage):
        return self.run / f"stage-{stage:02d}"

    def next_stage(self):
        for stage in range(1, self.count + 1):
            path = self.stage_dir(stage) / "result.json"
            if not path.is_file() or read_json(path)["status"] != "promote":
                return stage
            report = read_json(path)
            if sha256(report["candidate"]) != report["candidate_sha256"]:
                raise ValueError(f"promoted stage {stage} checkpoint changed")
            opponent = report.get("opponent")
            if opponent:
                if (opponent["mode"] != self.opponent_mode
                        or sha256(opponent["path"]) != opponent["sha256"]):
                    raise ValueError(f"promoted stage {stage} opponent changed")
            elif self.opponent_mode != "fixed":
                raise ValueError("previous-stage mode requires a new campaign with recorded opponents")
        return self.count + 1

    def command(self, script, arguments):
        command = [sys.executable, str(ROOT / "mgba" / script), *map(str, arguments)]
        with (self.run / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(command=command, cwd=str(ROOT))) + "\n")
        print("\n" + shlex.join(command), flush=True)
        self.execute(command, ROOT)

    def common(self, *, pooled=False):
        result = ["--reward-config", self.reward]
        if not pooled:
            result += ["--opponent-checkpoint", self.opponent]
        for key in ("environments", "showdown_workers", "node", "device", "torch_threads", "command_timeout",
                    "log_max_mb", "log_backups", "keep_episode_logs"):
            result += ["--" + key.replace("_", "-"), self.execution[key]]
        result.append("--step-logs" if self.execution["step_logs"] else "--no-step-logs")
        return result

    def evaluation(self, parent, label, suite_path, candidate):
        """Reuse only matching weights and a matching, fully completed schedule."""
        suite = read_json(suite_path)
        pooled = bool(suite.get("opponents"))
        if pooled:
            verify_pool(suite)
        models = dict(candidate=sha256(candidate))
        if not pooled:
            models["opponent"] = sha256(self.opponent)
        existing = sorted(parent.glob(label + "-*/config.json"))
        for config_path in existing:
            config = read_json(config_path)
            directory = config_path.parent
            if (config.get("suite_sha256") == digest(suite)
                and all(config.get("models", {}).get(k, {}).get("sha256") == v for k, v in models.items())
                and (directory / "summary.json").is_file()
                and read_json(directory / "summary.json").get("status") == "completed"):
                return directory, evaluation_rates(directory, suite)
        index = max([int(p.name.rsplit("-", 1)[1]) for p in parent.glob(label + "-*") if p.is_dir()], default=0) + 1
        directory = parent / f"{label}-{index:03d}"
        self.command("evaluate_ppo.py", [*self.common(pooled=pooled), "--checkpoint", candidate,
                     "--suite", suite_path, "--output-dir", directory])
        return directory, evaluation_rates(directory, suite)

    def pool_sources(self, stage, settings):
        dumb = self.run / "inputs/opponent.zip"
        if settings["dumb_checkpoint"]:
            pinned = settings["dumb_checkpoint"]
            dumb = Path(pinned["path"])
            if sha256(dumb) != pinned["sha256"]:
                raise ValueError("saved Dumb evaluation checkpoint changed")
        smart = [("smart_v1", self.run / "inputs/start.zip")]
        for previous in range(1, stage):
            report = read_json(self.stage_dir(previous) / "result.json")
            if report["status"] != "promote" or sha256(report["candidate"]) != report["candidate_sha256"]:
                raise ValueError(f"promoted Smart model changed at stage {previous}")
            smart.append((f"smart_v{previous+1}", Path(report["candidate"])))
        if settings["smart_dir"]:
            extra = sorted(Path(settings["smart_dir"]).glob("smart_v*.zip"))
            if not extra:
                raise ValueError("saved --eval-smart-dir no longer contains smart_v*.zip checkpoints")
            smart += [(path.stem, path) for path in extra]
        return dumb, smart

    def evaluation_pool(self, stage, attempt):
        settings = evaluation_settings(self.config)
        # A partial evaluation already owns copies of its models. Resume it even
        # when a discovery directory was moved or gained additional models.
        saved = (attempt / "evaluation-pool/plan.json").is_file()
        dumb, smart = (None, []) if saved else self.pool_sources(stage, settings)
        index = int(attempt.name.split("-")[1])
        path, suite = prepare_pool(attempt / "evaluation-pool", self.evaluation_groups, stage=stage,
            episodes=self.config["eval_episodes"], seed=self.config["seed"] + stage*1000003 + index*1009,
            max_steps=self.config["max_steps"], dumb=dumb, smart=smart)
        sizes = {group: sum(c["opponent_group"] == group for c in suite["cases"])
                 for group in ("random", "dumb", "smart")}
        print(f"Evaluation pool: {sizes['random']} Random / {sizes['dumb']} Dumb / {sizes['smart']} Smart; "
              f"promote with {settings['require']} gates: Basic >= {settings['basic_win_rate']:.1%}, "
              f"Smart >= {settings['smart_win_rate']:.1%}.\nPlan: {path}", flush=True)
        return path

    def prepare_stage(self, stage):
        directory = self.stage_dir(stage)
        directory.mkdir(exist_ok=True)
        source = self.run / "inputs/start.zip"
        if stage > 1:
            previous = read_json(self.stage_dir(stage-1) / "result.json")
            if previous["status"] != "promote":
                raise ValueError("the previous stage must be promoted before starting this stage")
            source = Path(previous["candidate"])
            if sha256(source) != previous["candidate_sha256"]:
                raise ValueError("the previous stage's promoted checkpoint changed")
        marker = directory / "stage.json"
        anchor = directory / "anchor.zip"
        existing = marker.is_file()
        if existing:
            metadata = read_json(marker)
        else:
            anchor.write_bytes(source.read_bytes())
            metadata = dict(stage=stage, source=str(source), anchor_sha256=sha256(anchor))
        if sha256(anchor) != metadata["anchor_sha256"]:
            raise ValueError("stage anchor checkpoint changed")
        if metadata["anchor_sha256"] != sha256(source):
            raise ValueError("stage anchor does not match its entering checkpoint")
        promoted_opponent = self.opponent_mode == "previous-stage" and stage > 1
        opponent_source = anchor if promoted_opponent else self.run / "inputs/opponent.zip"
        opponent = directory / "opponent.zip"
        expected = dict(mode=self.opponent_mode, path=str(opponent), sha256=sha256(opponent_source),
                        source_stage=stage-1 if promoted_opponent else None,
                        source=str(source if promoted_opponent else opponent_source))
        saved = metadata.get("opponent")
        if saved and saved != expected:
            raise ValueError("saved stage opponent disagrees with this campaign")
        if existing and not saved and self.opponent_mode != "fixed":
            raise ValueError("cannot change an existing stage's opponent mode; start a new campaign")
        if saved and not opponent.is_file():
            raise ValueError("saved stage opponent checkpoint is missing")
        if opponent.exists():
            if sha256(opponent) != expected["sha256"]:
                raise ValueError("saved stage opponent checkpoint changed")
        else:
            opponent.write_bytes(opponent_source.read_bytes())
        metadata["opponent"] = expected
        if not existing or not saved:
            # Fixed-mode stages created by older versions can adopt a copy of the
            # identical reference. Evaluation reuse is checked by content hash.
            write_json(marker, metadata)
        self.opponent, self.stage_opponent = opponent, expected
        pool = directory / "teams"
        pool.mkdir(exist_ok=True)
        training = [p for group in self.groups[:stage] for p in group]
        expected_names = {p.name for p in training}
        if any(p.name not in expected_names for p in pool.glob("*.json")):
            raise ValueError(f"unexpected team in saved stage pool: {pool}")
        for path in training:
            target = pool / path.name
            if target.exists() and target.read_bytes() != path.read_bytes():
                raise ValueError(f"saved stage team changed: {target}")
            if not target.exists():
                target.write_bytes(path.read_bytes())
        current, expanded = stage_suites(self.evaluation_groups, stage=stage,
            episodes=self.config["eval_episodes"], seed=self.config["seed"] + 100000 + stage,
            max_steps=self.config["max_steps"])
        for name, suite in (("current-suite.json", current), ("expanded-suite.json", expanded)):
            target = directory / name
            if target.exists() and read_json(target) != suite:
                raise ValueError(f"saved stage suite changed: {target}")
            if not target.exists():
                write_json(target, suite)
        return directory, pool, anchor

    def run_stage(self, stage):
        directory, pool, anchor = self.prepare_stage(stage)
        training_count = sum(len(g) for g in self.groups[:stage])
        validation_count = sum(len(g) for g in self.evaluation_groups[:stage])
        print(f"\nStage {stage}/{self.count}: {training_count} training teams "
              f"({len(self.groups[stage-1])} introduced this stage).", flush=True)
        if self.config.get("validation_groups"):
            print(f"Promotion uses {validation_count} held-out validation teams; none enter training.", flush=True)
        if not self.config.get("train_battles"):
            print(f"Next-group validation: {len(self.evaluation_groups[stage])} teams.", flush=True)
        print(self.config["curriculum"]["stages"][stage-1].get("focus", ""), flush=True)
        opponent_label = (f"promoted stage {self.stage_opponent['source_stage']}"
                          if self.stage_opponent["source_stage"] is not None else "initial reference")
        print(f"Frozen opponent: {opponent_label} ({self.opponent_mode}); "
              f"sha256={self.stage_opponent['sha256']}\nSnapshot: {self.opponent}", flush=True)
        # Pin this baseline for all attempts of the stage, avoiding cumulative
        # regression hidden by a newly lowered baseline on every retry.
        simple = bool(self.config.get("train_battles"))
        settings = evaluation_settings(self.config)
        mixed = settings["mode"] == "mixed"
        before_dir = before_expanded_dir = before = before_expanded = None
        if not simple:
            before_dir, before = self.evaluation(directory, "before-current", directory / "current-suite.json", anchor)
            before_expanded_dir, before_expanded = self.evaluation(directory, "before-expanded", directory / "expanded-suite.json", anchor)
        attempts = sorted(directory.glob("attempt-*/attempt.json"))
        resume, attempt = anchor, None
        if attempts:
            last = attempts[-1].parent
            report_path = last / "report.json"
            if report_path.is_file():
                previous = read_json(report_path)
                if previous["status"] == "promote":
                    write_json(directory / "result.json", previous)
                    self.track_record()
                    return previous
                resume = Path(previous["candidate"])
                if sha256(resume) != previous["candidate_sha256"]:
                    raise ValueError("last attempt checkpoint changed")
            elif (last / "train/checkpoints/final.zip").is_file() and (last / "train/summary.json").is_file() and read_json(last / "train/summary.json").get("status") == "completed":
                # Training completed but an evaluation was interrupted: do not retrain.
                attempt = last
            else:
                saved = last / "train/checkpoints/latest.zip"
                resume = saved if saved.is_file() else Path(read_json(last / "attempt.json")["resume"])
        if attempt is None:
            index = max([int(p.name.split("-")[1]) for p in directory.glob("attempt-*") if p.is_dir()], default=0) + 1
            attempt = directory / f"attempt-{index:03d}"
            attempt.mkdir()
            write_json(attempt / "attempt.json", dict(resume=str(resume), resume_sha256=sha256(resume),
                       opponent=self.stage_opponent, execution=self.execution,
                       settings_revision=self.config.get("settings_revision", 0), evaluation=settings))
            arguments = [*self.common(), "--backend", "showdown", "--team-dir", pool,
                "--resume", resume, "--total-timesteps", self.execution["timesteps"],
                "--max-steps", self.config["max_steps"], "--seed", self.config["seed"] + stage*1000 + index,
                "--output-dir", attempt / "train"]
            if simple:
                pos = arguments.index("--total-timesteps")
                arguments[pos:pos+2] = ["--train-battles", self.config["train_battles"]]
            for key in ("rollout_steps", "batch_size", "epochs", "learning_rate", "ent_coef",
                        "latest_every_updates", "checkpoint_every", "keep_checkpoints"):
                if self.execution[key] is not None:
                    arguments += ["--" + key.replace("_", "-"), self.execution[key]]
            for key in ("readable_log", "battle_log"):
                arguments.append("--" + ("" if self.execution[key] else "no-") + key.replace("_", "-"))
            switches = switch_settings(**self.config.get("opponent_switch", {}))
            if switches["chance"]:
                arguments += ["--opponent-switch-chance", switches["chance"],
                              "--opponent-switch-cooldown", switches["cooldown"]]
            self.command("train_ppo.py", arguments)
            if read_json(attempt / "train/summary.json").get("status") != "completed":
                raise RuntimeError("training did not complete; rerun this stage to resume its saved update")
        candidate = attempt / "train/checkpoints/final.zip"
        suite_path = self.evaluation_pool(stage, attempt) if mixed else directory / "current-suite.json"
        after_dir, after = self.evaluation(attempt, "eval-current", suite_path, candidate)
        expanded_dir = expanded = None
        if mixed:
            report = pool_promotion(after, settings)
            report["evaluation_plan"] = str(suite_path)
        elif simple:
            rates = after["all"]
            rate = rates["wins"] / rates["attempts"]
            threshold = self.config["promotion_win_rate"]
            report = dict(status="promote" if rate >= threshold else "hold", current=after,
                          mode="battle-budget", win_rate=rate, checks=[dict(check="current pool wins / all attempts",
                          value=rate, threshold=threshold, direction="at_least", passed=rate >= threshold)])
        else:
            expanded_dir, expanded = self.evaluation(attempt, "eval-expanded", directory / "expanded-suite.json", candidate)
            report = promotion_report(before, before_expanded, after, expanded, self.config["rules"])
        report.update(stage=stage, candidate=str(candidate), candidate_sha256=sha256(candidate),
            opponent=self.stage_opponent,
            training_teams=training_count, evaluation_teams=validation_count,
            held_out_validation=bool(self.config.get("validation_groups")),
            settings_revision=self.config.get("settings_revision", 0), evaluation_settings=settings,
            evaluations=dict(before_current=str(before_dir), before_expanded=str(before_expanded_dir),
                             current=str(after_dir), expanded=str(expanded_dir)))
        report["evaluations"] = {k: v for k, v in report["evaluations"].items() if v != "None"}
        report["attempt"] = int(attempt.name.split("-")[1])
        if report["status"] == "promote":
            export = self.run / "models" / f"smart_v{stage+1}.zip"
            export.parent.mkdir(exist_ok=True)
            if export.exists() and sha256(export) != sha256(candidate):
                raise ValueError(f"named checkpoint already exists with different weights: {export}")
            export.write_bytes(candidate.read_bytes())
            report["named_checkpoint"] = str(export)
        write_json(attempt / "report.json", report)
        (attempt / "report.md").write_text(report_markdown(report))
        write_json(directory / "result.json", report)
        (directory / "report.md").write_text(report_markdown(report))
        self.track_record()
        fmt = lambda values: "n/a" if values["win_rate"] is None else f"{values['win_rate']:.1%}"
        overall_rate = report.get("win_rate", after["all"]["win_rate"])
        current_text = "n/a" if overall_rate is None else f"{overall_rate:.1%}"
        print(f"\n{report['status'].upper()} stage {stage}: current={current_text}, "
              f"result={report['status']}", flush=True)
        if mixed:
            print("  " + " | ".join(f"{group.title()}={fmt(after[group])} "
                  f"({after[group]['wins']}/{after[group]['attempts']})"
                  for group in ("random", "dumb", "basic", "smart")), flush=True)
        for check in report["checks"]:
            if not check["passed"]:
                print(f"  Below threshold: {check['check']} = {check['value']}; {check['direction']} {check['threshold']}", flush=True)
        print(f"Report: {directory / 'report.md'}\nCheckpoint: {candidate}", flush=True)
        return report

    def track_record(self):
        """Rebuild from durable reports so resumes never duplicate rows."""
        rows = []
        for path in sorted(self.run.glob("stage-*/attempt-*/report.json")):
            report = read_json(path)
            rates = report["current"]["all"]
            row = dict(stage=report["stage"], attempt=report.get("attempt", int(path.parent.name.split("-")[1])),
                candidate=f"smart_v{report['stage']+1}", opponent=(f"smart_v{report['opponent']['source_stage']+1}"
                if report.get("opponent", {}).get("source_stage") else "initial reference"),
                wins=rates["wins"], evaluations=rates["attempts"],
                win_rate=report.get("win_rate", rates["win_rate"]), result=report["status"],
                checkpoint=report.get("named_checkpoint", report["candidate"]),
                settings_revision=report.get("settings_revision", 0),
                evaluation_mode=report.get("mode", "original"),
                promotion_require=report.get("promotion_require", ""))
            for group in ("random", "dumb", "basic", "smart"):
                values = report["current"].get(group, {})
                for key in ("wins", "attempts", "win_rate"):
                    row[f"{group}_{key}"] = values.get(key)
            rows.append(row)
        fields = ["stage", "attempt", "candidate", "opponent", "wins", "evaluations", "win_rate", "result", "checkpoint",
                  "settings_revision", "evaluation_mode", "promotion_require"]
        fields += [f"{group}_{key}" for group in ("random", "dumb", "basic", "smart")
                   for key in ("wins", "attempts", "win_rate")]
        target = self.run / "evaluations.csv"
        with target.with_suffix(".tmp").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        target.with_suffix(".tmp").replace(target)
        fmt = lambda value: "—" if value is None else f"{value:.1%}"
        lines = ["# Campaign evaluation record", ""]
        if any(row["evaluation_mode"] == "mixed-evaluation" for row in rows):
            lines += ["Basic combines Random + Dumb. Rates include draws and truncations as non-wins. "
                      "Rules identifies the saved revision and whether either or both gates were required.", "",
                      "| Stage | Attempt | Candidate | Overall | Random | Dumb | Basic | Smart | Rules | Result |",
                      "|---:|---:|---|---:|---:|---:|---:|---:|---|---|"]
            for row in rows:
                rates = " | ".join(fmt(row[f"{group}_win_rate"]) for group in ("random", "dumb", "basic", "smart"))
                rules = f"r{row['settings_revision']} {row['promotion_require'] or 'stage'}"
                lines.append(f"| {row['stage']} | {row['attempt']} | {row['candidate']} | {fmt(row['win_rate'])} | {rates} | {rules} | {row['result'].upper()} |")
            lines += ["", "Older stage evaluations have no category breakdown. "
                      "See evaluations.csv for counts, training opponents and checkpoint paths; "
                      "rule-changes.jsonl records threshold changes."]
        else:
            lines += ["| Stage | Attempt | Candidate | Frozen opponent | Wins / evaluated | Win rate | Result |",
                      "|---:|---:|---|---|---:|---:|---|"]
            for row in rows:
                lines.append(f"| {row['stage']} | {row['attempt']} | {row['candidate']} | {row['opponent']} | {row['wins']} / {row['evaluations']} | {fmt(row['win_rate'])} | {row['result'].upper()} |")
        (self.run / "evaluations.md").write_text("\n".join(lines)+"\n")


def initialize(args, run):
    if not args.resume or not args.opponent_checkpoint:
        raise ValueError("a new curriculum run needs --resume and --opponent-checkpoint")
    if any(p.name != ".lock" for p in run.iterdir()):
        raise ValueError("new curriculum output must be empty; choose a new --run-dir")
    manifest, groups, validation = load_curriculum(args.curriculum or DEFAULT_CURRICULUM, grouped=True)
    teams = [p for group in groups for p in group]
    opponent_mode = getattr(args, "opponent_mode", None) or manifest.get("opponent_mode", "fixed")
    if opponent_mode not in OPPONENT_MODES:
        raise ValueError("opponent mode must be fixed or previous-stage")
    switches = dict(manifest.get("opponent_switch", {}))
    for flag, key in (("opponent_switch_chance", "chance"), ("opponent_switch_cooldown", "cooldown")):
        if getattr(args, flag, None) is not None:
            switches[key] = getattr(args, flag)
    switches = switch_settings(**switches)
    episodes, max_steps = args.eval_episodes or 1000, args.max_steps or 200
    if episodes % 4:
        raise ValueError("--eval-episodes must be divisible by four")
    train_battles = getattr(args, "train_battles", None)
    promotion_rate = getattr(args, "promotion_win_rate", None)
    if promotion_rate is not None and (not train_battles or not 0 < promotion_rate <= 1):
        raise ValueError("--promotion-win-rate requires --train-battles and a fraction in (0, 1]")
    if train_battles and args.promotion_rules:
        raise ValueError("--promotion-rules applies to the original mode; use --promotion-win-rate with --train-battles")
    rules = validate_rules(read_json(args.promotion_rules) if args.promotion_rules else DEFAULT_RULES.copy())
    reward_path = args.reward_config or ROOT / "config/rewards/damage_knockout_win.json"
    LinkedRewardConfig.load(reward_path)
    checkpoints = resolve_checkpoints([args.resume, args.opponent_checkpoint], ROOT)
    data = {path: path.read_bytes() for path in set(checkpoints)}
    inputs = run / "inputs"
    (inputs / "teams").mkdir(parents=True)
    snapshots = []
    for name, source in zip(("start.zip", "opponent.zip"), checkpoints):
        (inputs / name).write_bytes(data[source])
        snapshots.append(dict(source=str(source), sha256=hashlib.sha256(data[source]).hexdigest()))
    (inputs / "reward.json").write_bytes(Path(reward_path).read_bytes())
    team_names = []
    for i, source in enumerate(teams):
        target = inputs / "teams" / f"{i+1:02d}_{source.name}"
        target.write_bytes(source.read_bytes())
        team_names.append(str(target.relative_to(run)))
    saved_paths = dict(zip(teams, team_names))
    validation_names = None
    if validation:
        (inputs / "validation").mkdir()
        for source in (p for group in validation for p in group):
            if source not in saved_paths:
                target = inputs / "validation" / f"{len(saved_paths)+1:04d}_{source.name}"
                target.write_bytes(source.read_bytes())
                saved_paths[source] = str(target.relative_to(run))
        validation_names = [[saved_paths[p] for p in group] for group in validation]
    config = dict(version=1, curriculum=manifest, teams=team_names, checkpoints=snapshots, opponent_mode=opponent_mode,
        team_groups=[[saved_paths[p] for p in group] for group in groups], validation_groups=validation_names,
        rules=rules, eval_episodes=episodes, max_steps=max_steps, seed=args.seed if args.seed is not None else 1,
        input_hashes={str(p.relative_to(run)): sha256(p) for p in inputs.rglob("*") if p.is_file()},
        execution=EXECUTION.copy())
    config["train_battles"] = getattr(args, "train_battles", None)
    config["opponent_switch"] = switches
    config["promotion_win_rate"] = getattr(args, "promotion_win_rate", None) or 0.75
    if config["promotion_win_rate"] > 1:
        raise ValueError("--promotion-win-rate must be in (0, 1]")
    models = run / "models"
    models.mkdir()
    (models / "smart_v1.zip").write_bytes((inputs / "start.zip").read_bytes())
    (models / "dumb.zip").write_bytes((inputs / "opponent.zip").read_bytes())
    write_json(run / "campaign.json", config)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-battles", type=positive_int, help="New run: use battle-budget mode; train at least N ended battles, then one current-pool evaluation")
    parser.add_argument("--promotion-win-rate", type=nonnegative, help="Stage evaluation gate (default .75); can change on resume; requires battle-budget mode")
    parser.add_argument("--eval-opponents", choices=("stage", "mixed"), help="Evaluation opponents: stage's frozen model (default) or 30%% Random / 30%% Dumb / 40%% Smart; can change on resume")
    parser.add_argument("--promotion-basic-win-rate", type=nonnegative, help="Mixed gate: wins / all Random + Dumb battles (default .75)")
    parser.add_argument("--promotion-smart-win-rate", type=nonnegative, help="Mixed gate: wins / all Smart battles (default .60)")
    parser.add_argument("--promotion-require", choices=("either", "both"), help="Mixed gate: either threshold or both required (default both)")
    parser.add_argument("--eval-smart-dir", type=Path, help="Add smart_v*.zip models to the campaign's starting and promoted Smart models; snapshot each evaluation")
    parser.add_argument("--eval-dumb-checkpoint", type=Path, help="Mixed evaluation Dumb model: path or latest, copied once (default: initial campaign reference)")
    parser.add_argument("--attempts-per-stage", type=positive_int, default=1, help="Maximum attempts per stage in this invocation; stop early on promotion (default 1)")
    parser.add_argument("--run-dir", type=Path, required=True, help="New or existing curriculum directory")
    parser.add_argument("--stage", type=positive_int, help="Run this stage; default: first stage not promoted")
    parser.add_argument("--through-stage", type=positive_int, help="Continue through this stage, stopping at the first HOLD")
    parser.add_argument("--curriculum", type=Path, help="New run: manifest with team files/directories per stage (default: config/curricula/type_progression/curriculum.json)")
    parser.add_argument("--resume", type=Path, help="New run: candidate checkpoint path or latest; copied once")
    parser.add_argument("--opponent-checkpoint", "--player2-checkpoint", type=Path, help="New run: initial reference path or latest; copied once")
    parser.add_argument("--opponent-mode", choices=OPPONENT_MODES,
                        help="New run: fixed reference or previous-stage promoted model; overrides manifest (default: fixed)")
    parser.add_argument("--opponent-switch-chance", type=nonnegative,
                        help="New run: training opponent type-switch check chance (0–1); overrides curriculum; normally 0")
    parser.add_argument("--opponent-switch-cooldown", type=int,
                        help="New run: normal decisions between opponent switch checks after a voluntary switch (default: 2)")
    parser.add_argument("--reward-config", type=Path, help="New run: reward JSON (default: damage_knockout_win.json)")
    parser.add_argument("--promotion-rules", type=Path, help="New run: promotion threshold JSON; see curriculum guide")
    parser.add_argument("--eval-episodes", type=positive_int, help="New run: battles per suite, multiple of four, or 20 for mixed opponents (default: 1000)")
    parser.add_argument("--max-steps", type=positive_int, help="New run: decision cap for training and evaluation (default: 200)")
    parser.add_argument("--seed", type=int, help="New run: base seed (default: 1); suites use separate fixed seeds")
    for key, help_text in (("environments", "Even number of players; half learn (initial default: 48)"),
        ("timesteps", "Additional transitions per attempt (initial default: 100000; rounded to a rollout)"),
        ("rollout_steps", "Steps per learning player (initial default: 128)"),
        ("batch_size", "Optimization minibatch (initial default: 128)"),
        ("epochs", "PPO epochs; omitted keeps checkpoint value"),
        ("latest_every_updates", "Save latest.zip every N PPO updates, plus first/final (initial default: 1)"),
        ("torch_threads", "PyTorch CPU threads (initial default: 1)")):
        parser.add_argument("--" + key.replace("_", "-"), type=positive_int, help=help_text)
    parser.add_argument("--learning-rate", type=nonnegative, help="Positive PPO learning rate; omitted keeps checkpoint value")
    parser.add_argument("--checkpoint-every", type=nonnegative_int,
                        help="Learning transitions between numbered training backups; 0 disables (initial default: 1024)")
    parser.add_argument("--keep-checkpoints", type=retention_count,
                        help="Numbered training backups per attempt; 0 disables, -1 unlimited (initial default: 2)")
    parser.add_argument("--retention", choices=("compact", "off"),
                        help="Prune completed attempts after reports are saved (new campaigns: compact; legacy: off)")
    for key, help_text in (("keep_attempts", "Recent candidate attempts per stage (initial default: 2)"),
                           ("keep_best", "Best candidates per stage/evaluation-rule cohort (initial default: 2)"),
                           ("keep_debug_attempts", "Recent attempt debug logs to retain per stage (initial default: 0)")):
        parser.add_argument("--" + key.replace("_", "-"), type=retention_count,
                            help=help_text + "; -1 retains all; applies with compact retention")
    parser.add_argument("--step-logs", action=argparse.BooleanOptionalAction, default=None,
                        help="Full Showdown decision traces in training and evaluation (initial default: off)")
    parser.add_argument("--log-max-mb", type=nonnegative,
                        help="Showdown diagnostic log rotation limit in MiB; 0 unlimited (initial default: 10)")
    parser.add_argument("--log-backups", type=nonnegative_int,
                        help="Rotated backups per Showdown diagnostic log (initial default: 2)")
    parser.add_argument("--keep-episode-logs", type=retention_count,
                        help="Completed trace directories per pair; 0 disables, -1 unlimited (initial default: 2)")
    parser.add_argument("--ent-coef", type=nonnegative, help="PPO entropy coefficient; omitted keeps checkpoint value")
    parser.add_argument("--command-timeout", type=positive_int, help="Seconds per paired decision (initial default: 120)")
    parser.add_argument("--showdown-workers", choices=("thread", "process"), help="Worker mode (initial default: process)")
    parser.add_argument("--node", help="Node executable (initial default: node)")
    parser.add_argument("--device", help="PyTorch device (initial default: cpu)")
    parser.add_argument("--readable-log", action=argparse.BooleanOptionalAction, default=None, help="Diagnostic battle logs (initial default: on)")
    parser.add_argument("--battle-log", action=argparse.BooleanOptionalAction, default=None, help="Spectator battle logs (initial default: on)")
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    run = args.run_dir.expanduser().resolve()
    try:
        run.mkdir(parents=True, exist_ok=True)
        with campaign_lock(run):
            if (run / "campaign.json").exists():
                if any(getattr(args, key) is not None for key in IMMUTABLE_FLAGS):
                    raise ValueError("this run already pins checkpoints, teams, budgets and original promotion-rules; "
                                     "omit new-run options. Evaluation-pool flags and --promotion-win-rate can change on resume")
                config = read_json(run / "campaign.json")
                if config.get("version") != 1:
                    raise ValueError("unsupported curriculum run version")
                # Upgrading an existing run must not silently opt it into historical deletion.
                config["execution"].setdefault("retention", "off")
            else:
                config = initialize(args, run)
            execution = {**EXECUTION, **config["execution"],
                         **{k: getattr(args, k) for k in EXECUTION if getattr(args, k) is not None}}
            if execution["environments"] < 2 or execution["environments"] % 2:
                raise ValueError("--environments must be even and at least two")
            if execution["batch_size"] < 2 or execution["rollout_steps"]*(execution["environments"]//2) % execution["batch_size"]:
                raise ValueError("--batch-size must divide --rollout-steps * (environments / 2)")
            if execution["learning_rate"] == 0:
                raise ValueError("--learning-rate must be positive")
            campaign = Campaign(run, config, execution)
            campaign.track_record()
            pending = campaign.next_stage()
            stage = args.stage or pending
            target = args.through_stage or stage
            if pending > campaign.count and not args.stage:
                print("All curriculum stages are promoted. Evaluate on a separate broad suite next.")
                return 0
            if stage > pending or stage > campaign.count or target > campaign.count or target < stage:
                raise ValueError(f"stages must advance in order; next stage is {pending}, last stage is {campaign.count}")
            if stage < pending:
                print(f"Stage {stage} is already promoted. Next stage: {pending}.")
                return 0
            update_evaluation_settings(args, run, config)
            config["execution"] = execution
            save_campaign(run, config)
            for selected in range(stage, target + 1):
                for _ in range(args.attempts_per_stage):
                    report = campaign.run_stage(selected)
                    if execution["retention"] == "compact":
                        plan = plan_campaign_retention(run, **{
                            key: execution[key] for key in ("keep_attempts", "keep_best", "keep_debug_attempts")})
                        result = apply_campaign_retention(plan, assume_locked=True)
                        if result["files"]:
                            print(f"Retention: removed {result['files']} files, "
                                  f"{result['bytes'] / 1024**3:.3f} GiB; reports and protected models retained.", flush=True)
                    if report["status"] == "promote":
                        break
                if report["status"] == "hold":
                    print(f"Stage {selected} held. Rerun with --run-dir {run} --stage {selected} to continue training.")
                    return 2
            print("Requested stages complete.")
            return 0
    except KeyboardInterrupt:
        print(f"Curriculum interrupted. Rerun --run-dir {run} to continue; completed artifacts are retained.")
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Curriculum failed: {error}\nRun: {run}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
