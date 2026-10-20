#!/usr/bin/env python3
"""Calibrate and train tiny fixed-team tasks against Rules, with two-job logging.

Each task compares a fresh shared-moves policy with a continuation of the 400k
reference. Run from the project root with the normal training venv activated.
"""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from run_parallel_training import checkpoint_steps, recovered_steps, pump


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interrupt(*_):
    raise KeyboardInterrupt


def training_command(root, plan, job):
    run = root / job["name"]
    base = [sys.executable, "-u", str(ROOT / "mgba/train_ppo.py")]
    # Older plans deliberately keep their original full-budget behavior.
    stopping = plan.get("early_stopping", {})
    stop_flags = (["--eval-stop-win-rate", str(stopping["win_rates"][job["scenario"]]),
                   "--eval-stop-consecutive", str(stopping["consecutive"])]
                  if stopping.get("enabled", False) else ["--no-eval-stop"])
    if run.exists():
        remaining = job["target_timesteps"] - recovered_steps(run)
        # A final validation interrupted at the target is retried with one extra
        # rollout by the native trainer, rather than silently called completed.
        summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
        if summary.get("status") == "completed" and (remaining <= 0 or
                                                       summary.get("stop_reason") == "validation_success"):
            return None
        return base + ["--output-dir", str(run), "--if-exists", "continue",
                       "--total-timesteps", str(max(1, remaining))] + stop_flags
    paths = [root / "inputs" / f"{job['scenario']}-{side}.json" for side in (1, 2)]
    resume = ["--resume", str(root / "inputs/reference.zip")] if job["initialization"] == "reference" else []
    return base + resume + stop_flags + [
        "--backend", "showdown", "--policy-architecture", "shared-moves",
        "--observation-version", "v2", "--rule-based-opponent", "--teams", *map(str, paths),
        "--reward-config", str(root / "inputs/reward.json"),
        "--learning-rate", "0.0001", "--ent-coef", "0.01",
        "--rollout-steps", "128", "--batch-size", "128", "--epochs", "4",
        "--environments", str(plan["environments"]), "--showdown-workers", "process",
        "--device", "cpu", "--torch-threads", "1", "--seed", str(plan["seed"]),
        "--max-steps", "100", "--total-timesteps", str(job["additional_timesteps"]),
        "--eval-suite", str(root / "inputs" / f"{job['scenario']}-validation.json"),
        "--eval-every", "25000", "--eval-environments", "8", "--keep-evaluations", "2",
        "--latest-every-updates", "5", "--checkpoint-every", "50000", "--keep-checkpoints", "2",
        "--battle-log", "--no-readable-log", "--no-step-logs", "--log-max-mb", "2", "--log-backups", "1",
        "--output-dir", str(run), "--if-exists", "cancel"]


def launch(root, plan, jobs):
    pending, active = list(plan["runs"]), []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        while pending or active:
            while pending and len(active) < jobs:
                job = pending.pop(0)
                command = training_command(root, plan, job)
                if command is None:
                    print(f"Already completed: {job['name']}", flush=True)
                    continue
                print(f"Starting {job['name']}\n{shlex.join(command)}", flush=True)
                process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                reader = threading.Thread(target=pump, args=(process, root / "console" / f"{job['name']}.log"))
                reader.start()
                active.append((job, process, reader))
            for row in list(active):
                job, process, reader = row
                code = process.poll()
                if code is None:
                    continue
                reader.join()
                active.remove(row)
                if code:
                    raise RuntimeError(f"{job['name']} exited {code}; see console/{job['name']}.log")
                summary = json.loads((root / job['name'] / "summary.json").read_text())
                if summary.get("status") != "completed":
                    raise RuntimeError(f"{job['name']} did not complete: {summary.get('status')}")
                print(f"Completed: {job['name']}", flush=True)
            if active:
                time.sleep(.5)
    finally:
        # Signal the trainer first: its own cleanup saves/retains checkpoints
        # and terminates validation children. Do not kill every worker mid-write.
        for _, process, _ in active:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        if active:
            print("Waiting for active trainers to stop; keep this terminal open.", flush=True)
        for _, process, reader in active:
            process.wait()
            reader.join()


def freeze_inputs(root, args):
    from firered_rl.fixed_team_lab import teams, make_suite
    from firered_rl.linked_reward import LinkedRewardConfig
    from firered_rl.showdown_env import validate_showdown
    from firered_rl.shared_move_policy import SharedMovePolicy
    from sb3_contrib import MaskablePPO

    validate_showdown(ROOT)
    reward = LinkedRewardConfig.load(args.reward_config)
    if not args.fresh_only:
        model = MaskablePPO.load(args.checkpoint, device="cpu")
        if not isinstance(model.policy, SharedMovePolicy):
            raise ValueError("Reference must use the shared-moves architecture.")
        del model
    # Refuse accidental replacement of prior work.
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    (root / "console").mkdir()
    write_json(inputs / "reward.json", asdict(reward))
    if not args.fresh_only:
        shutil.copyfile(args.checkpoint, inputs / "reference.zip")
    runs = []
    for scenario, values in teams().items():
        paths = [inputs / f"{scenario}-{i}.json" for i in (1, 2)]
        for path, value in zip(paths, values):
            write_json(path, value)
        for label, seed, count, permute in [
            ("calibration", 731701, args.calibration_battles, False),
            ("validation", 731702, args.validation_battles, False),
            ("test", 731703, args.test_battles, False),
            ("permuted", 731703, args.test_battles, True),
        ]:
            write_json(inputs / f"{scenario}-{label}.json",
                       make_suite(ROOT, paths, count, seed, scenario, permute_moves=permute))
        budget = args.move_steps if scenario == "move-choice" else args.mirror_steps
        for initialization in (["fresh"] if args.fresh_only else ["fresh", "reference"]):
            start = 0 if initialization == "fresh" else checkpoint_steps(inputs / "reference.zip")
            name = f"{scenario}-{initialization}"
            runs.append(dict(name=name, scenario=scenario, initialization=initialization,
                             additional_timesteps=budget, starting_timesteps=start,
                             target_timesteps=start + budget))
            # All logs exist from the beginning so tail -F console/*.log also
            # sees jobs which have not started yet.
            (root / "console" / f"{name}.log").touch()
    plan = dict(version=1, seed=args.seed, environments=args.environments, runs=runs,
                inputs={p.name: checksum(p) for p in inputs.iterdir()},
                early_stopping=dict(enabled=not getattr(args, "no_early_stop", False), consecutive=2,
                                    win_rates={"move-choice": .99, "mirror": .95}),
                thresholds=dict(move_choice_win_rate=.95, move_choice_first_surf_rate=.95,
                                mirror_score_note="Compare with measured Rules/greedy; >50% is not a guaranteed ceiling test."))
    write_json(root / "plan.json", plan)
    return plan


def calibration(root):
    from firered_rl.fixed_team_lab import score, GreedyDamage, RuleBasedOpponent
    stats = {}
    for scenario in ("move-choice", "mirror"):
        suite = json.loads((root / "inputs" / f"{scenario}-calibration.json").read_text())
        for label, model in [("random", None), ("greedy", GreedyDamage(ROOT)), ("rules", RuleBasedOpponent(ROOT))]:
            run = root / "calibration" / f"{scenario}-{label}"
            result = None
            for path in sorted(run.parent.glob(f"{run.name}*/summary.json")):
                saved = json.loads(path.read_text())
                if saved.get("status") == "completed" and "wins_per_requested" in saved:
                    result = saved
            if result is None:
                if run.exists():
                    run = run.with_name(run.name + f"-retry-{time.time_ns()}")
                result = score(ROOT, run, suite, model, root / "inputs/reward.json")
            stats[f"{scenario}-{label}"] = result
    good, random = stats["move-choice-greedy"], stats["move-choice-random"]
    if good["truncated"] or good["wins_per_requested"] < .95 or good["wins_per_requested"] - random["wins_per_requested"] < .2:
        raise ValueError("Move-choice calibration did not show a clear learnable advantage; inspect calibration before training.")
    return stats


def final_tests(root, plan):
    from sb3_contrib import MaskablePPO
    from firered_rl.fixed_team_lab import score
    for job in plan["runs"]:
        for checkpoint in ("best", "final"):
            path = root / job["name"] / "checkpoints" / f"{checkpoint}.zip"
            model = MaskablePPO.load(path, device="cpu")
            model.policy.set_training_mode(False)
            for condition in ("test", "permuted"):
                run = root / "checks" / f"{job['name']}-{checkpoint}-{condition}"
                if (run / "summary.json").exists():
                    result = json.loads((run / "summary.json").read_text())
                    if result.get("status") == "completed" and result.get("checkpoint_sha256") == checksum(path):
                        continue
                # Partial evaluations get a new directory on continuation.
                if run.exists():
                    run = run.with_name(run.name + f"-retry-{time.time_ns()}")
                suite = json.loads((root / "inputs" / f"{job['scenario']}-{condition}.json").read_text())
                result = score(ROOT, run, suite, model, root / "inputs/reward.json")
                result["checkpoint_sha256"] = checksum(path)
                write_json(run / "summary.json", result)
            del model


def collect(root):
    lines = ["# Fixed-team learnability lab", "", "Wins use all requested battles; truncations count as non-wins.", "",
             "| Evaluation | Wins | Win rate | Truncated | First-turn Surf (move-choice only) |",
             "|---|---:|---:|---:|---:|"]
    for directory in ("calibration", "checks"):
        for path in sorted((root / directory).glob("*/summary.json")):
            s = json.loads(path.read_text())
            if s.get("status") != "completed":
                lines.append(f"| {directory}/{path.parent.name} ({s.get('status', 'incomplete')}) | — | — | — | — |")
                continue
            surf = f"{s['first_surf_rate']:.1%}" if path.parent.name.startswith("move-choice") else "—"
            lines.append(f"| {directory}/{path.parent.name} | {s['candidate_wins']}/{s['requested_battles']} | "
                         f"{s['wins_per_requested']:.1%} | {s['truncated']} | {surf} |")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    temporary = root / "fixed-team-results.tmp"
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or relative.parts[0] == "console" or path.suffix not in (".json", ".jsonl", ".md", ".csv"):
                continue
            # Keep compact reports/configs; omit bulky per-step/training traces.
            if (relative.parts[0] in ("inputs", "calibration", "checks") or path.name in
                ("plan.json", "report.md", "config.json", "summary.json", "failure.json", "best.json",
                 "evaluations.md", "evaluations.csv", "evaluation-history.json", "updates.jsonl",
                 "battle-blocks.jsonl", "progress.csv", "switching.json")):
                archive.write(path, relative)
    temporary.replace(root / "fixed-team-results.zip")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path(".local/run/fixed-team-lab-01"))
    p.add_argument("--checkpoint", type=Path, default=Path("models/observation-v2-shared-moves-400k-reference-01.zip"))
    p.add_argument("--reward-config", type=Path, default=Path(".local/run/diversity-v2-01/inputs/reward.json"))
    p.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    p.add_argument("--environments", type=int, default=16)
    p.add_argument("--seed", type=int, default=261030)
    p.add_argument("--move-steps", type=int, default=50000)
    p.add_argument("--mirror-steps", type=int, default=200000)
    p.add_argument("--calibration-battles", type=int, default=200)
    p.add_argument("--validation-battles", type=int, default=256)
    p.add_argument("--test-battles", type=int, default=1000)
    p.add_argument("--fresh-only", action="store_true", help="Omit reference continuations (also useful for installation smoke tests)")
    p.add_argument("--no-early-stop", action="store_true",
                   help="New batches: train the full budget instead of stopping after two successful validations; continuation preserves its saved setting")
    p.add_argument("--calibrate-only", action="store_true", help="Prepare inputs and baselines; --continue later runs training")
    p.add_argument("--continue", dest="continuing", action="store_true", help="Restore pinned inputs/budgets; only jobs may change")
    args = p.parse_args()
    if args.environments < 2 or args.environments % 2 or min(args.move_steps, args.mirror_steps) < 1:
        p.error("Use an even environment count >= 2 and positive training budgets.")
    if any(n < 2 or n % 2 for n in (args.calibration_battles, args.validation_battles, args.test_battles)):
        p.error("Battle counts must be positive and even.")
    import torch
    torch.set_num_threads(1)
    root = args.output_dir.expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = (root.parent / f".{root.name}.lab.lock").open("a")
    initialized = False
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.continuing:
            plan = json.loads((root / "plan.json").read_text())
            if args.no_early_stop and plan.get("early_stopping", {}).get("enabled", False):
                raise ValueError("--continue preserves the saved early-stopping policy; --no-early-stop applies to a new batch")
            for name, sha in plan["inputs"].items():
                if checksum(root / "inputs" / name) != sha:
                    raise ValueError(f"Pinned input changed: {name}")
        else:
            plan = freeze_inputs(root, args)
        initialized = True
        print(f"Output: {root}\nLive logs: tail -n 20 -F '{root}/console/'*.log", flush=True)
        calibration(root)
        if not args.calibrate_only:
            launch(root, plan, args.jobs)
            final_tests(root, plan)
        collect(root)
        print(f"Report: {root / 'report.md'}\nUpload: {root / 'fixed-team-results.zip'}", flush=True)
        return 0
    except (KeyboardInterrupt, Exception) as error:
        if initialized:
            collect(root)
        print(f"Stopped: {error or 'interrupted'}\nDiagnostics: {root}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
