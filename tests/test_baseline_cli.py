"""Real baseline snapshot -> local simulator -> interrupted/resumed evaluation."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

from firered_rl.curriculum import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "mgba" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_campaign(tmp_path):
    import gymnasium as gym
    import torch
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import BattleFeatures, policy_space
    torch.set_num_threads(1)

    class ShapeEnv(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)

    agent = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2, n_epochs=1,
                        policy_kwargs=dict(features_extractor_class=BattleFeatures), seed=2)
    starting = tmp_path / "starting.zip"
    agent.save(starting)
    opponent = tmp_path / "opponent.zip"
    agent.num_timesteps = 2
    agent.save(opponent)
    for directory, mons in (("train", [("Rattata", "Tackle"), ("Pidgey", "Tackle")]),
                            ("validation", [("Spearow", "Peck"), ("Sentret", "Scratch")]),
                            ("probe", [("Zigzagoon", "Tackle")])):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
        for species, move in mons:
            write_json(tmp_path / directory / f"{species}.json", dict(name=species,
                party=[dict(species=species, level=5, moves=[move, "Protect", "Rest", "Toxic"])]))
    manifest = tmp_path / "curriculum.json"
    write_json(manifest, dict(version=1, name="baseline-smoke", stages=[
        dict(stage=1, team_dir="train", validation_dir="validation", focus="test")], next_team_dir="probe"))
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / ".lock").touch()
    curriculum = module("baseline_campaign", "run_curriculum.py")
    config = curriculum.initialize(argparse.Namespace(resume=starting, opponent_checkpoint=opponent,
        curriculum=manifest, eval_episodes=4, max_steps=15, promotion_rules=None,
        reward_config=None, seed=1), campaign)
    stage = campaign / "stage-01"
    stage.mkdir()
    anchor = stage / "anchor.zip"
    shutil.copyfile(campaign / "inputs/start.zip", anchor)
    write_json(stage / "stage.json", dict(stage=1, anchor_sha256=sha(anchor)))
    attempt = stage / "attempt-001"
    checkpoints = attempt / "train/checkpoints"
    checkpoints.mkdir(parents=True)
    agent.num_timesteps = 4
    agent.save(checkpoints / "final.zip")
    write_json(attempt / "attempt.json", dict(resume=str(anchor), resume_sha256=sha(anchor)))
    write_json(attempt / "train/summary.json", dict(status="completed"))
    report = dict(stage=1, attempt=1, status="hold", candidate=str(checkpoints / "final.zip"),
                  candidate_sha256=sha(checkpoints / "final.zip"), win_rate=.6,
                  current={"all": dict(attempts=10, wins=6, win_rate=.6)}, settings_revision=0,
                  mode="battle-budget")
    write_json(attempt / "report.json", report)
    write_json(stage / "result.json", report)
    assert config["validation_groups"]
    return campaign


@pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(), reason="Showdown not installed")
def test_real_baseline_dry_run_interrupt_resume_and_reuse(tmp_path, monkeypatch):
    pytest.importorskip("sb3_contrib")
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    cli = module("baseline_cli", "establish_baseline.py")
    campaign = fixture_campaign(tmp_path)
    before = {str(p.relative_to(campaign)): p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    output = tmp_path / "baseline"
    args = ["establish_baseline", "--campaign-run", str(campaign), "--run-dir", str(output),
            "--episodes-per-opponent", "4", "--bootstrap-samples", "100",
            "--environments", "2", "--showdown-workers", "thread", "--log-max-mb", ".02"]
    monkeypatch.setattr(sys, "argv", args + ["--dry-run"])
    assert cli.main() == 0 and not output.exists()
    write = cli.write_json

    def interrupt_initialization(path, value):
        if path.name == "runtime.json":
            raise KeyboardInterrupt()
        write(path, value)

    monkeypatch.setattr(cli, "write_json", interrupt_initialization)
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 130 and not output.exists()
    assert not list(tmp_path.glob(".baseline-initializing-*"))
    monkeypatch.setattr(cli, "write_json", write)
    calls = []

    def execute(command, log, directory):
        calls.append(command)
        assert "--readable-log" in command and "--battle-log" in command
        assert "--no-step-logs" in command
        if len(calls) == 2:
            raise KeyboardInterrupt()
        cli.run_evaluator(command, log, directory)

    run = cli.run_baseline
    monkeypatch.setattr(cli, "run_baseline", lambda path, plan, execution: run(path, plan, execution, execute=execute))
    monkeypatch.setattr(sys, "argv", args)
    assert cli.main() == 130
    plan = read_json(output / "baseline.json")
    assert len(cli.jobs_for(plan)) == 4  # best/latest alias one immutable model.
    assert read_json(output / "status.json")["completed_jobs"] == 1
    assert before == {str(p.relative_to(campaign)): p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    # Snapshots are independently resumable after the original campaign is relocated.
    campaign.rename(tmp_path / "original-preserved")
    monkeypatch.setattr(sys, "argv", ["establish_baseline", "--run-dir", str(output)])
    assert cli.main() == 0
    assert len(calls) == 5  # Four evaluations, plus the deliberate interruption.
    status = read_json(output / "status.json")
    assert status["status"] == "completed" and status["completed_jobs"] == 4
    report = read_json(output / "baseline-report.json")
    assert report["candidates"]["best"]["identical_weights_to"] == ["latest"]
    assert report["candidates"]["best"]["evaluations"] == report["candidates"]["latest"]["evaluations"]
    assert set(report["splits"]) == {"training", "validation"}
    for split in report["splits"].values():
        assert split["candidates"]["starting"]["groups"]["all"]["attempts"] == 12
    assert (output / "baseline-report.md").is_file()
    assert list(output.glob("runs/*/*/attempt-*/pair-*/battle.log"))
    assert not list(output.glob("runs/*/*/attempt-*/pair-*/episode-*"))
    assert cli.main() == 0 and len(calls) == 5  # All results validated and reused.
    monkeypatch.setattr(sys, "argv", ["establish_baseline", "--run-dir", str(output), "--seed", "99"])
    assert cli.main() == 1  # Frozen suites cannot be silently resampled.
    monkeypatch.setattr(sys, "argv", ["establish_baseline", "--run-dir", str(output), "--torch-threads", "2"])
    assert cli.main() == 1  # Do not mix runtime identities in one report.


def test_effective_device_is_explicit(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    cli = module("baseline_device", "establish_baseline.py")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cli.effective_device("auto") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert cli.effective_device("auto") == "cuda:2"
    assert cli.effective_device("cuda:1") == "cuda:1"


def test_runtime_hash_covers_installed_simulator_and_mappings(tmp_path, monkeypatch):
    pytest.importorskip("sb3_contrib")
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    cli = module("baseline_identity", "establish_baseline.py")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    for filename in ("sim/showdown/package-lock.json", "mgba/evaluate_ppo.py",
                     "mgba/profiles/firered_us_v1_0_names.json", "config/generation/firered_us_v1_0.json",
                     "sim/showdown/node_modules/pokemon-showdown/dist/sim/battle.js"):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: argparse.Namespace(stdout="v22"))
    initial = cli.runtime_identity(cli.EXECUTION)
    path.write_text("changed installed simulator")
    assert cli.runtime_identity(cli.EXECUTION)["code_sha256"] != initial["code_sha256"]
