#!/usr/bin/env python3
"""Compare a campaign's starting, best and latest policies on frozen Showdown suites."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory

from evaluate_models import path_argument, positive_int, run_evaluator, validate_model
from train_ppo import nonnegative, nonnegative_int, retention_count
from firered_rl.baseline_setup import build_plan, save_plan, verify_plan
from firered_rl.baseline_report import build_baseline_report, load_baseline_evaluation, write_baseline_report
from firered_rl.curriculum import read_json, write_json
from firered_rl.evaluation import digest
from firered_rl.retention import campaign_retention_lock


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("starting", "best", "latest")
EXECUTION = dict(environments=48, showdown_workers="process", node="node", device="cpu",
    torch_threads=1, command_timeout=120, readable_log=True, battle_log=True, step_logs=False,
    log_max_mb=2, log_backups=1, keep_episode_logs=2, bootstrap_samples=4000, bootstrap_seed=1)
NEW_INPUTS = ("campaign_run", "stage", "episodes_per_opponent", "seed", "max_steps",
    "starting_checkpoint", "best_checkpoint", "latest_checkpoint", "dumb_checkpoint",
    "smart_checkpoint", "validation_dir")


@contextmanager
def baseline_lock(run):
    run.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock outside the output so a complete snapshot can be published atomically.
    with (run.parent / f".{run.name}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another baseline command is using {run}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def jobs_for(plan):
    groups = {}
    for role in ROLES:
        checksum = plan["candidates"][role]["sha256"]
        groups.setdefault(checksum, []).append(role)
    return [dict(candidate=roles[0], aliases=roles, split=split)
            for roles in groups.values() for split in ("training", "validation")]


def print_plan(plan):
    count = sum(len(entry["suite"]["cases"]) if "suite" in entry else entry["battles"]
                for entry in plan["suites"].values())
    unique = len({c["sha256"] for c in plan["candidates"].values()})
    print(f"Baseline: {unique} distinct candidate models; {unique*count:,} battles including side swaps.")
    for role in ROLES:
        candidate = plan["candidates"][role]
        print(f"  {role}: {candidate['source']} [{candidate['sha256'][:12]}]")
    for split, entry in plan["suites"].items():
        print(f"  {split}: {entry['team_count']} team templates")
    for key, opponent in plan["opponents"].items():
        print(f"  opponent {key}: {opponent.get('source', 'uniform random legal actions')}")
    for warning in plan.get("warnings", []):
        print("Warning: " + warning)


def runtime_identity(execution):
    """Avoid combining results across a changed simulator or policy implementation."""
    files = [*ROOT.joinpath("src/firered_rl").rglob("*.py"),
             *ROOT.joinpath("src/firered_rl").rglob("*.json"),
             *ROOT.joinpath("sim/showdown").glob("*.cjs"),
             ROOT / "sim/showdown/package-lock.json", ROOT / "mgba/evaluate_ppo.py",
             ROOT / "mgba/profiles/firered_us_v1_0_names.json",
             ROOT / "config/generation/firered_us_v1_0.json"]
    simulator = ROOT / "sim/showdown/node_modules/pokemon-showdown"
    files.extend(p for p in simulator.rglob("*") if p.is_file() and p.suffix in (".js", ".cjs", ".json"))
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    node = subprocess.run([execution["node"], "--version"], check=True, capture_output=True, text=True).stdout.strip()
    return dict(code=code, code_sha256=digest(code), python=platform.python_version(), node=node,
                packages={name: version(name) for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")},
                device=execution["device"], torch_threads=execution["torch_threads"])


def effective_device(requested):
    import torch
    from stable_baselines3.common.utils import get_device
    device = get_device(requested)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


def validate_result(output, plan, job):
    result = load_baseline_evaluation(output)
    expected = plan["suites"][job["split"]]["suite_sha256"]
    config = result["config"]
    if digest(result["suite"]) != expected:
        raise ValueError(f"baseline suite changed in {output}")
    if config["models"]["candidate"]["sha256"] != plan["candidates"][job["candidate"]]["sha256"]:
        raise ValueError(f"baseline candidate changed in {output}")
    if config["reward"] != plan["reward"]["value"]:
        raise ValueError(f"baseline rewards changed in {output}")
    return result


def run_baseline(run, plan, execution, *, execute=run_evaluator):
    jobs = jobs_for(plan)
    evaluations = {role: {} for role in ROLES}
    progress = dict(status="running", completed_jobs=0, total_jobs=len(jobs), evaluations=evaluations,
                    plan_sha256=plan["plan_sha256"])
    active = None
    try:
        for job in jobs:
            active = f"{job['candidate']}/{job['split']}"
            directory = run / "runs" / job["candidate"] / job["split"]
            attempts = sorted(directory.glob("attempt-*/summary.json"),
                              key=lambda p: int(p.parent.name.split("-")[1]))
            output = None
            for marker in attempts:
                if read_json(marker).get("status") == "completed":
                    validate_result(marker.parent, plan, job)
                    output = marker.parent
                    break
            progress["active"] = active
            write_json(run / "status.json", progress)
            if output is None:
                directory.mkdir(parents=True, exist_ok=True)
                index = max((int(p.name.split("-")[1]) for p in directory.glob("attempt-*")
                             if p.is_dir() and p.name.split("-")[1].isdigit()), default=0)+1
                output = directory / f"attempt-{index:03d}"
                log = directory / f"attempt-{index:03d}.log"
                command = [sys.executable, str(ROOT / "mgba/evaluate_ppo.py"),
                    "--checkpoint", plan["candidates"][job["candidate"]]["path"],
                    "--suite", plan["suites"][job["split"]]["path"],
                    "--reward-config", plan["reward"]["path"], "--output-dir", str(output)]
                for key in ("environments", "showdown_workers", "node", "device", "torch_threads",
                            "command_timeout", "log_max_mb", "log_backups", "keep_episode_logs"):
                    command += ["--" + key.replace("_", "-"), str(execution[key])]
                for key in ("readable_log", "battle_log", "step_logs"):
                    command += ["--" + ("" if execution[key] else "no-") + key.replace("_", "-")]
                print(f"[{progress['completed_jobs']+1}/{len(jobs)}] {active}; log: {log}", flush=True)
                with (run / "commands.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(job=active, command=command))+"\n")
                execute(command, log, output)
                validate_result(output, plan, job)
            else:
                print(f"Reusing validated evaluation: {active}", flush=True)
            for role in job["aliases"]:
                evaluations[role][job["split"]] = str(output)
            progress["completed_jobs"] += 1
            progress["active"] = None
            write_json(run / "status.json", progress)
        report = build_baseline_report(evaluations, bootstrap_samples=execution["bootstrap_samples"],
                                       seed=execution["bootstrap_seed"])
        report["baseline_plan_sha256"] = plan["plan_sha256"]
        report["selection_warnings"] = plan.get("warnings", [])
        write_baseline_report(report, run)
        progress.update(status="completed", active=None)
        write_json(run / "status.json", progress)
    except BaseException as error:
        progress.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        active=active, error=str(error))
        write_json(run / "status.json", progress)
        raise
    print(f"Baseline complete. Report: {run / 'baseline-report.md'}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=path_argument, required=True, help="New baseline directory, or existing baseline to resume")
    parser.add_argument("--campaign-run", type=path_argument, help="New run: source campaign; stop it while creating snapshots")
    parser.add_argument("--stage", type=positive_int, help="New run: campaign stage to inspect (default: 1)")
    parser.add_argument("--episodes-per-opponent", type=positive_int,
                        help="New run: battles per Random/Dumb/Smart group per split; even, includes both seats (default: 1000)")
    parser.add_argument("--seed", type=int, help="New run: evaluation schedule seed (default: 26001)")
    parser.add_argument("--max-steps", type=positive_int, help="New run: decision limit (default: source campaign's cap)")
    for role in (*ROLES, "dumb", "smart"):
        parser.add_argument(f"--{role}-checkpoint", type=Path,
                            help=f"New run: override {role} model with a checkpoint path or latest")
    parser.add_argument("--validation-dir", type=path_argument,
                        help="New run: separate validation JSON team directory; default campaign's cumulative validation groups")
    for key, help_text in (("environments", "Concurrent players, even (default: 48)"),
                           ("torch_threads", "PyTorch CPU threads (default: 1)"),
                           ("command_timeout", "Seconds per paired decision (default: 120)"),
                           ("bootstrap_samples", "Paired bootstrap resamples (default: 4000)")):
        parser.add_argument("--" + key.replace("_", "-"), type=positive_int, help=help_text)
    parser.add_argument("--bootstrap-seed", type=int, help="Report bootstrap RNG seed (default: 1)")
    parser.add_argument("--showdown-workers", choices=("thread", "process"), help="Python worker mode (default: process)")
    parser.add_argument("--node", help="Node executable (default: node)")
    parser.add_argument("--device", help="Inference device (default: cpu); fixed once evaluation begins")
    for key in ("readable_log", "battle_log", "step_logs"):
        parser.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction, default=None,
                            help=f"Bounded Showdown {key.replace('_', ' ')} (default: {'off' if key == 'step_logs' else 'on'})")
    parser.add_argument("--log-max-mb", type=nonnegative, help="MiB per diagnostic file; 0 unlimited (default: 2)")
    parser.add_argument("--log-backups", type=nonnegative_int, help="Rotated copies per diagnostic file (default: 1)")
    parser.add_argument("--keep-episode-logs", type=retention_count,
                        help="Completed trace directories per pair; 0 disables, -1 unlimited (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Show frozen selections and workload without running battles or writing baseline outputs")
    return parser, parser.parse_args()


def main():
    _, args = parse_args()
    run = args.run_dir
    try:
        existing = (run / "baseline.json").is_file()
        if existing and any(getattr(args, key) is not None for key in NEW_INPUTS):
            raise ValueError("baseline inputs are frozen; resume with --run-dir only or choose a new output directory")
        if not existing:
            if args.campaign_run is None:
                raise ValueError("a new baseline requires --campaign-run")
            if run.is_relative_to(args.campaign_run):
                raise ValueError("baseline output must be outside the source campaign")
            if run.exists() and any(run.iterdir()):
                raise ValueError("new baseline output must be empty; choose a new --run-dir")
        execution = {**EXECUTION, **(read_json(run / "execution.json") if existing else {}),
                     **{k: getattr(args, k) for k in EXECUTION if getattr(args, k) is not None}}
        if execution["environments"] < 2 or execution["environments"] % 2:
            raise ValueError("--environments must be even and at least two")
        if execution["bootstrap_samples"] < 100:
            raise ValueError("--bootstrap-samples must be at least 100")
        if existing:
            plan = verify_plan(run)
        else:
            with campaign_retention_lock(args.campaign_run):
                plan = build_plan(args.campaign_run, **{k: getattr(args, k) for k in NEW_INPUTS
                    if k != "campaign_run" and getattr(args, k) is not None})
        print_plan(plan)
        if args.dry_run:
            print("Dry run complete. No baseline outputs written and no battles started.")
            return 0
        from firered_rl.showdown_env import validate_showdown
        validate_showdown(ROOT, execution["node"])
        import torch
        torch.set_num_threads(execution["torch_threads"])
        execution["device"] = effective_device(execution["device"])
        identity = runtime_identity(execution)
        if existing and read_json(run / "runtime.json") != identity:
            raise ValueError("simulator, policy code, dependency versions or inference settings changed; create a new baseline run")
        with baseline_lock(run):
            if not existing:
                if (run / "baseline.json").exists():
                    raise ValueError("another process initialized this baseline; resume with --run-dir only")
                with campaign_retention_lock(args.campaign_run):
                    # Re-read under the source lock before snapshotting. A preview never authorizes stale inputs.
                    plan = build_plan(args.campaign_run, **{k: getattr(args, k) for k in NEW_INPUTS
                        if k != "campaign_run" and getattr(args, k) is not None})
                    unique = {c["sha256"]: c["source"] for c in plan["candidates"].values()}
                    unique.update({v["sha256"]: v["source"] for v in plan["opponents"].values() if v["kind"] == "checkpoint"})
                    for source in unique.values():
                        validate_model(Path(source))
                    # Commit the directory only after all required files are present.
                    with TemporaryDirectory(prefix=f".{run.name}-initializing-", dir=run.parent) as folder:
                        staging = Path(folder)
                        save_plan(plan, staging, published_dir=run)
                        write_json(staging / "runtime.json", identity)
                        write_json(staging / "execution.json", execution)
                        os.replace(staging, run)
                    plan = verify_plan(run)
            else:
                plan = verify_plan(run)
            write_json(run / "execution.json", execution)
            run_baseline(run, plan, execution)
        return 0
    except KeyboardInterrupt:
        if (run / "baseline.json").is_file():
            print(f"Baseline interrupted. Resume with --run-dir {run}; complete evaluations are retained.")
        else:
            print("Baseline initialization interrupted. Rerun the original command to create its snapshot.")
        return 130
    except (ValueError, OSError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print(f"Baseline failed: {error}\nRun: {run}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
