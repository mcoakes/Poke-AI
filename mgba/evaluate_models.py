#!/usr/bin/env python3
"""Compare a directory of PPO models in a saved Showdown round-robin tournament."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys

from firered_rl.curriculum import read_json, write_json
from firered_rl.tournament import (DEFAULTS, completed_matches, make_plan, read_match,
    save_plan, verify_plan, write_reports)


ROOT = Path(__file__).resolve().parents[1]
EXECUTION = dict(environments=48, showdown_workers="process", node="node", device="cpu",
                 torch_threads=1, command_timeout=120, keep_battle_logs=False)


def path_argument(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("path is empty; set your shell variable first")
    return Path(value).expanduser().resolve()


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


@contextmanager
def run_lock(run):
    with (run / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"another evaluator is already using {run}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_model(path):
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import policy_space
    try:
        model = MaskablePPO.load(path, device="cpu")
    except Exception as error:
        raise ValueError(f"cannot load PPO model {path}: {error}") from error
    if model.observation_space != policy_space() or getattr(model.action_space, "n", None) != 11:
        raise ValueError(f"model does not match the battle observation/action schema: {path}")


def run_evaluator(command, log_path, output):
    """Keep child output on disk; print brief progress while the current pair runs."""
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            previous = -1
            while True:
                try:
                    code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    try:
                        progress = read_json(output / "summary.json")
                    except (OSError, ValueError):
                        continue
                    count = progress.get("attempted_battles", 0)
                    if count != previous:
                        print(f"  {count}/{progress['requested_battles']} battles evaluated", flush=True)
                        previous = count
        except BaseException:
            if process.poll() is None:
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
        raise RuntimeError(f"evaluation exited with status {code}; see {log_path}")


def print_plan(plan):
    models, scenarios = plan["models"], plan["scenarios"]
    n, count, episodes = len(models), len(plan["matches"]), plan["config"]["episodes"]
    print(f"{plan['name']}: {n} participants, {len(scenarios)} team configurations.\n"
          f"{n*(n-1)//2} model pairs × {len(scenarios)} configurations × {episodes} battles = "
          f"{count*episodes:,} battles. Counts include both player sides.", flush=True)
    for model in models:
        aliases = ", ".join(a["name"] for a in model.get("aliases", []))
        print(f"  {model['id']}: {model['label']}" + (f" (identical copies: {aliases})" if aliases else ""), flush=True)


def run_tournament(run, plan, execution, *, execute=run_evaluator):
    results = completed_matches(run, plan)
    done = {r["id"] for r in results}
    models = {m["id"]:m for m in plan["models"]}
    scenarios = {s["id"]:s for s in plan["scenarios"]}
    ledger = run / "matches.jsonl"
    recorded = set()
    if ledger.exists():
        data = ledger.read_bytes()
        offset = 0
        for line in data.splitlines(keepends=True):
            try:
                recorded.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                if offset + len(line) != len(data):
                    raise ValueError(f"invalid match history: {ledger}") from None
                # Repair only the final interrupted event; results live in validated evaluations.
                with ledger.open("r+b") as stream:
                    stream.truncate(offset)
                break
            offset += len(line)
        if ledger.stat().st_size and not ledger.read_bytes().endswith(b"\n"):
            with ledger.open("ab") as stream:
                stream.write(b"\n")

    def record(result):
        if result["id"] not in recorded:
            with ledger.open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            recorded.add(result["id"])

    for result in results:
        record(result)
    write_reports(run, plan, results)
    active = None
    try:
        for match in plan["matches"]:
            if match["id"] in done:
                continue
            active = match["id"]
            a, b = models[match["a"]], models[match["b"]]
            scenario = scenarios[match["scenario"]]
            directory = run / "matches" / active
            directory.mkdir(parents=True, exist_ok=True)
            index = max((int(p.name.split("-")[1]) for p in directory.glob("attempt-*") if p.is_dir()), default=0)+1
            output = directory / f"attempt-{index:03d}"
            log_path = directory / f"attempt-{index:03d}.log"
            write_reports(run, plan, results, active=active)
            print(f"\n[{len(results)+1}/{len(plan['matches'])}] {scenario['label']}: {a['label']} vs {b['label']}\n"
                  f"Details: {log_path}", flush=True)
            command = [sys.executable, str(ROOT / "mgba/evaluate_ppo.py"), "--checkpoint", str(run / a["path"]),
                "--suite", str(run / scenario["path"]), "--output-dir", str(output)]
            command += (["--random-opponent"] if b["kind"] == "random"
                        else ["--opponent-checkpoint", str(run / b["path"])])
            for key in ("environments", "showdown_workers", "node", "device", "torch_threads", "command_timeout"):
                command += ["--"+key.replace("_", "-"), str(execution[key])]
            keep = execution["keep_battle_logs"]
            command += ["--step-logs" if keep else "--no-step-logs",
                        "--readable-log" if keep else "--no-readable-log",
                        "--battle-log" if keep else "--no-battle-log"]
            with (run / "commands.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(match=active, command=command))+"\n")
            execute(command, log_path, output)
            result = read_match(run, plan, match, output)
            results.append(result)
            done.add(active)
            record(result)
            write_reports(run, plan, results)
            print(f"  {a['label']} {result['wins_a']} wins / {b['label']} {result['wins_b']} wins / "
                  f"{result['draws']} draws / {result['truncated']} truncated", flush=True)
            active = None
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        write_reports(run, plan, results, status=status, active=active)
        raise
    report = write_reports(run, plan, results)
    print(f"\nRound robin complete. Leaderboard: {run / 'leaderboard.md'}", flush=True)
    for row in report["leaderboards"]["overall"]:
        print(f"  {row['rank']}. {row['label']}: score={row['score']:.1%}, "
              f"wins={row['win_rate']:.1%}, games={row['games']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=path_argument, required=True, help="New or existing tournament directory")
    parser.add_argument("--config", type=path_argument, help="New run: evaluation plan JSON (required)")
    parser.add_argument("--models-dir", type=path_argument, help="New run: directory of saved PPO ZIPs (required)")
    parser.add_argument("--model-glob", help="New run: relative discovery glob (default: *.zip; **/final.zip scans subdirectories)")
    parser.add_argument("--episodes", type=positive_int, help="New run: override battles per pair per configuration; must be even")
    parser.add_argument("--max-steps", type=positive_int, help="New run: override decision cap per battle")
    parser.add_argument("--seed", type=int, help="New run: override team/battle schedule seed")
    parser.add_argument("--include-random", action=argparse.BooleanOptionalAction, default=None,
                        help="New run: include uniform random legal policy (default: config value, normally on)")
    parser.add_argument("--environments", type=positive_int, help="Concurrent players within a model-pair test; even (initial default: 48)")
    parser.add_argument("--showdown-workers", choices=("thread", "process"), help="Python worker mode (initial default: process)")
    parser.add_argument("--node", help="Node executable (initial default: node)")
    parser.add_argument("--device", help="PyTorch inference device (initial default: cpu)")
    parser.add_argument("--torch-threads", type=positive_int, help="PyTorch CPU threads (initial default: 1)")
    parser.add_argument("--command-timeout", type=positive_int, help="Seconds per paired decision (initial default: 120)")
    parser.add_argument("--keep-battle-logs", action=argparse.BooleanOptionalAction, default=None,
                        help="Keep per-turn traces and battle logs (initial default: off; compact results always kept)")
    parser.add_argument("--dry-run", action="store_true", help="Show participants and battle count without writing files or running battles")
    args = parser.parse_args()
    run = args.run_dir
    new_inputs = ("config", "models_dir", "model_glob", *DEFAULTS)
    try:
        existing = (run / "tournament.json").is_file()
        if existing:
            if any(getattr(args, key) is not None for key in new_inputs):
                raise ValueError("this tournament already pins models, teams and seeds; use --run-dir alone or create a new directory")
            plan = read_json(run / "tournament.json")
            verify_plan(run, plan)
        else:
            if not args.config or not args.models_dir:
                raise ValueError("a new tournament requires --config and --models-dir")
            if run.exists() and any(p.name != ".lock" for p in run.iterdir()):
                raise ValueError("new tournament output must be empty; choose a new --run-dir")
            plan, suites = make_plan(ROOT, args.config, args.models_dir, model_glob=args.model_glob or "*.zip",
                                    overrides={key:getattr(args, key) for key in DEFAULTS})
        print_plan(plan)
        if args.dry_run:
            print("Dry run: model files hashed; PPO loading and battle execution skipped. No files written.")
            return 0
        from firered_rl.showdown_env import validate_showdown
        execution = {**EXECUTION, **(read_json(run / "execution.json") if existing else {})}
        execution.update({key:getattr(args, key) for key in EXECUTION if getattr(args, key) is not None})
        if execution["environments"] < 2 or execution["environments"] % 2:
            raise ValueError("--environments must be even and at least two")
        validate_showdown(ROOT, execution["node"])
        import torch
        torch.set_num_threads(execution["torch_threads"])
        run.mkdir(parents=True, exist_ok=True)
        with run_lock(run):
            if not existing:
                if (run / "tournament.json").exists():
                    raise ValueError("another command initialized this tournament; rerun using --run-dir only")
                save_plan(run, plan, suites, validate_model=validate_model)
            write_json(run / "execution.json", execution)
            run_tournament(run, plan, execution)
        return 0
    except KeyboardInterrupt:
        print(f"Evaluation interrupted. Resume with --run-dir {run}; completed model-pair tests are retained.")
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(f"Model evaluation failed: {error}\nRun: {run}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
