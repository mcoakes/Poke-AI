#!/usr/bin/env python3
"""Audit current observation/action correctness with local Showdown, without training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def log_size(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite; audit logs are always bounded")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty audit directory (no resume or overwrite)")
    parser.add_argument("--checkpoint", help="Optional frozen checkpoint path or latest (later is an alias); default disposable untrained policy")
    parser.add_argument("--observation-version", choices=("v1", "v2"), default=None,
                        help="Observation schema; default checkpoint schema, otherwise v1")
    parser.add_argument("--node", default="node", help="Node.js executable (default: node)")
    parser.add_argument("--seed", type=int, default=26002, help="Fixture and disposable-policy seed, 0..2**32-1 (default: 26002)")
    parser.add_argument("--command-timeout", type=positive, default=30, help="Seconds per simulator request (default: 30)")
    parser.add_argument("--torch-threads", type=positive, default=1, help="CPU inference threads (default: 1)")
    parser.add_argument("--log-max-mb", type=log_size, default=2, help="Positive MiB limit per diagnostic file (default: 2)")
    parser.add_argument("--log-backups", type=nonnegative, default=1, help="Rotated backups per diagnostic file (default: 1)")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in 0..2**32-1")
    return args


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def checkpoint_schema(checkpoint):
    """Read the serialized observation space, including legacy unversioned models."""
    from stable_baselines3.common.save_util import load_from_zip_file
    from firered_rl.ppo_features import observation_version
    data, _, _ = load_from_zip_file(checkpoint, device="cpu")
    if data is None or "observation_space" not in data:
        raise ValueError("checkpoint has no observation space")
    return observation_version(data["observation_space"])


def report_markdown(report):
    counts = report["counts"]
    lines = ["# Training observation/action audit", "",
        f"Result: **{report['result'].upper()}** — {counts.get('pass', 0)} passing checks, "
        f"{counts.get('fail', 0)} correctness failures, {counts.get('limitation', 0)} information limitations.", "",
        "This audit does not train, change rewards, modify the observation schema, or save model weights.",
        "Missing information is reported separately from incorrect values or broken execution.", "",
        f"Seed: `{report['config']['seed']}` · Elapsed: {report['elapsed_seconds']:.1f} seconds.", ""]
    selected_version = report["config"].get("observation_version", "v1")
    lines += [f"Audited observation schema: `{selected_version}`.", ""]
    if selected_version == "v2":
        legacy = report.get("regression_counts", {})
        lines += ["The result counts above cover v2 checks only. The unchanged v1 audit is run separately as a regression suite.",
                  f"V1 regression evidence: {legacy.get('pass', 0)} passing checks, "
                  f"{legacy.get('fail', 0)} failures and {legacy.get('limitation', 0)} legacy information limitations. "
                  "Those legacy limitations are not reported as current v2 gaps.", ""]
    checkpoint = report.get("checkpoint")
    if checkpoint:
        lines += [f"Frozen checkpoint: `{checkpoint['path']}`", "",
                  f"Actual checkpoint schema: `{checkpoint.get('observation_version', 'v1')}`", "",
                  f"Loaded SHA-256: `{checkpoint.get('loaded_sha256', 'unavailable: loading failed')}`", ""]
    else:
        lines += ["Policy: a disposable untrained MaskablePPO model tests inference wiring only.", ""]
    lines += ["Neither mode measures tactical skill, win rate, or generalization.", "", "## Correctness findings", ""]
    failures = [row for row in report["checks"] if row["status"] == "fail"]
    if failures:
        for row in failures:
            detail = row["details"].get("error") or row["details"].get("explanation") or json.dumps(row["details"], sort_keys=True)
            lines += [f"- **{escape(row['name'])}** ({row.get('scope', selected_version)}, {row['group']}): {escape(detail)}"]
    else:
        lines += ["All executed correctness checks passed. This is coverage of the listed fixtures, not proof of all battle mechanics."]
    inventory_heading = "V2 additions: information inventory" if selected_version == "v2" else "Information inventory"
    lines += ["", f"## {inventory_heading}", "", "| Field | Coverage | Policy representation | Limitation |",
              "|---|---|---|---|"]
    for row in report["inventory"]:
        lines.append("| " + " | ".join(escape(row.get(key, "")) for key in
                                      ("name", "status", "policy_representation", "limitation")) + " |")
    lines += ["", "## Checks", "", "| Scope | Group | Check | Result |", "|---|---|---|---|"]
    for row in report["checks"]:
        lines.append(f"| {escape(row.get('scope', selected_version))} | {escape(row['group'])} | {escape(row['name'])} | {row['status'].upper()} |")
    diagnostics = ("Controlled v2 diagnostics are in `controlled-v2/`; v2 vector/PPO logs are in `pipeline-v2/`. "
                   "The unchanged legacy suite's logs are in `legacy-v1/`. " if selected_version == "v2" else
                   "Controlled battle diagnostics are in `controlled/`; real vector/PPO logs are in `pipeline/`. ")
    lines += ["", "## Interpretation", "",
        "- A correctness failure requires inspecting its evidence before blaming learning or tuning hyperparameters.",
        "- Information limitations describe what this policy cannot directly observe. They are not automatically bugs or proof of the cause of weak validation results.",
        "- Synthetic fixtures intentionally isolate mechanics; some custom teams/moves are not legal competitive teams.",
        "- Player-side comparisons use nonrandom controlled outcomes; equal seeds do not guarantee mirrored stochastic trajectories.",
        "- Checkpoint probabilities and value predictions establish numerical validity and legality, not good decisions.",
        "- The bootstrap check verifies final-observation eligibility and the value path without collecting training rollouts.",
        "- This is a Showdown audit. Cartridge equivalence, all abilities/moves, tactical benchmarks and training stability remain separate work.", "",
        "Detailed evidence, provenance and inference probabilities are in `audit-report.json`. "
        + diagnostics + "Readable/spectator logs are retained with per-file rotation; full training step traces are disabled.", ""]
    return "\n".join(lines)


def save_report(output, report):
    for name, content in (("audit-report.json", json.dumps(report, indent=2) + "\n"),
                          ("audit-report.md", report_markdown(report))):
        path = output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def main():
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    try:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("--output-dir must be new or empty; choose another directory")
        import torch
        from firered_rl.audit_cases import run_battle_checks
        from firered_rl.audit_inventory import inventory_checks, observation_inventory
        from firered_rl.audit_pipeline import run_pipeline_checks
        from firered_rl.checkpoints import resolve_checkpoint
        from firered_rl.showdown_env import validate_showdown

        validate_showdown(ROOT, args.node)
        torch.set_num_threads(args.torch_threads)
        checkpoint = resolve_checkpoint(args.checkpoint, ROOT) if args.checkpoint else None
        actual_schema = checkpoint_schema(checkpoint) if checkpoint else None
        selected_version = args.observation_version or actual_schema or "v1"
        if actual_schema and selected_version != actual_schema:
            raise ValueError(f"--observation-version {selected_version} conflicts with checkpoint schema {actual_schema}")
        inventory = observation_inventory()
        if selected_version == "v2":
            from firered_rl.audit_v2 import (
                observation_inventory_v2, run_v2_battle_checks, run_v2_pipeline_checks,
            )
            inventory = observation_inventory_v2()
        output.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents two audit processes sharing an empty directory.
        with (output / "audit-owner.json").open("x") as stream:
            json.dump(dict(created_at=datetime.now(timezone.utc).isoformat()), stream)
        relevant = [*ROOT.joinpath("src/firered_rl").glob("*.py"),
                    *ROOT.joinpath("sim/showdown").glob("*.cjs"), ROOT / "sim/showdown/package-lock.json",
                    ROOT / "mgba/profiles/firered_us_v1_0_names.json", ROOT / "config/generation/firered_us_v1_0.json",
                    ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json", Path(__file__).resolve()]
        report = dict(version=1, status="running", result="pending", config=dict(
            seed=args.seed, observation_version=selected_version, node=args.node, command_timeout=args.command_timeout, torch_threads=args.torch_threads,
            log_max_mb=args.log_max_mb, log_backups=args.log_backups, readable_log=True, battle_log=True, step_logs=False),
            checkpoint=dict(path=str(checkpoint), observation_version=actual_schema) if checkpoint else None,
            inventory=inventory, checks=[], counts={},
            elapsed_seconds=0, created_at=datetime.now(timezone.utc).isoformat(), runtime=dict(
                python=platform.python_version(), node=subprocess.run([args.node, "--version"], check=True,
                capture_output=True, text=True).stdout.strip(), packages={name: version(name) for name in
                ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")},
                files_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in relevant}))
        started = time.monotonic()
        print(f"Training audit: {output}", flush=True)
        print(f"Policy: {checkpoint or 'disposable untrained model (wiring checks only)'}", flush=True)
        common = dict(node=args.node, timeout=args.command_timeout, seed=args.seed,
                      log_max_mb=args.log_max_mb, log_backups=args.log_backups)
        jobs = (("inventory", "v1", inventory_checks),
                ("controlled", "v1", lambda: run_battle_checks(ROOT, output / "controlled", **common)),
                ("pipeline", "v1", lambda: run_pipeline_checks(ROOT, output, checkpoint=checkpoint, **common)))
        if selected_version == "v2":
            jobs = (
                ("inventory", "v1_regression", inventory_checks),
                ("controlled", "v1_regression", lambda: run_battle_checks(ROOT, output / "legacy-v1" / "controlled", **common)),
                ("pipeline", "v1_regression", lambda: run_pipeline_checks(ROOT, output / "legacy-v1", **common)),
                ("controlled", "v2", lambda: run_v2_battle_checks(ROOT, output / "controlled-v2", **common)),
                ("pipeline", "v2", lambda: run_v2_pipeline_checks(ROOT, output, checkpoint=checkpoint, **common)),
            )
        try:
            for group, scope, function in jobs:
                print(f"Checking {scope} {group}...", flush=True)
                try:
                    rows = function()
                except Exception as error:
                    rows = [dict(name=f"{group}_execution", status="fail", details=dict(error=f"{type(error).__name__}: {error}"))]
                report["checks"].extend(dict(group=group, scope=scope, **row) for row in rows)
                report["counts"] = dict(Counter(row["status"] for row in report["checks"]
                                                if row["scope"] == selected_version))
                if selected_version == "v2":
                    report["regression_counts"] = dict(Counter(row["status"] for row in report["checks"]
                                                              if row["scope"] == "v1_regression"))
                report["elapsed_seconds"] = time.monotonic()-started
                save_report(output, report)
            if checkpoint:
                loaded = next((row for row in report["checks"] if row["name"] == ("v2_pipeline_policy_load" if selected_version == "v2" else "pipeline_policy_load")), None)
                if loaded and loaded["status"] == "pass":
                    report["checkpoint"]["loaded_sha256"] = loaded["details"].get("checkpoint_sha256")
                    report["checkpoint"]["training_timesteps"] = loaded["details"]["num_timesteps"]
                    print(f"Loaded checkpoint SHA-256: {report['checkpoint']['loaded_sha256']}", flush=True)
            report["status"] = "completed"
            report["result"] = "fail" if any(row["status"] == "fail" for row in report["checks"]) else "pass"
        except KeyboardInterrupt:
            report.update(status="interrupted", result="incomplete")
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic()-started
            save_report(output, report)
        counts = report["counts"]
        print(f"{report['result'].upper()}: {counts.get('pass', 0)} passed, {counts.get('fail', 0)} correctness failures, "
              f"{counts.get('limitation', 0)} information limitations.")
        if selected_version == "v2":
            legacy = report["regression_counts"]
            print(f"V1 regression scope: {legacy.get('pass', 0)} passed, {legacy.get('fail', 0)} failures, "
                  f"{legacy.get('limitation', 0)} legacy information limitations.")
        print(f"Report: {output / 'audit-report.md'}")
        return 1 if report["result"] == "fail" else 0
    except KeyboardInterrupt:
        print(f"Audit interrupted. Partial evidence: {output}; rerun in a new directory.", file=sys.stderr)
        return 130
    except (ValueError, OSError, ImportError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Audit could not start: {error}\nInstall Python training extras and the pinned Showdown dependency; see docs/training-audit.md.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
