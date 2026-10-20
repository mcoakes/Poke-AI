"""Runner contracts: native metadata, validation selection, pins and recovery."""
import copy
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_staged_team_lab as runner


def checkpoint(path, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(dict(num_timesteps=steps)))


def result(rate=.95):
    return dict(status="completed", truncated=0, completed_battles=100,
                requested_battles=100, candidate_wins=round(rate*100),
                candidate_win_rate=rate, first_surf_rate=rate)


def test_recovery_skips_success_stop_but_retries_pending_final_validation(tmp_path):
    job = dict(name="order-a", target_timesteps=200)
    run = tmp_path / "runs/order-a"
    checkpoint(run / "checkpoints/latest.zip", 120)
    command = runner.training_command(tmp_path, {}, job)
    assert command[command.index("--total-timesteps")+1] == "80"
    runner.write_json(run / "summary.json", dict(status="completed", stop_reason="validation_success"))
    assert runner.training_command(tmp_path, {}, job) is None
    checkpoint(run / "checkpoints/latest.zip", 200)
    runner.write_json(run / "summary.json", dict(status="interrupted"))
    command = runner.training_command(tmp_path, {}, job)
    assert command[command.index("--total-timesteps")+1] == "1"
    runner.write_json(run / "summary.json", dict(status="completed", stop_reason="timestep_budget"))
    assert runner.training_command(tmp_path, {}, job) is None


def test_controlled_continuations_share_seed_and_pin_start(tmp_path):
    checkpoint(tmp_path / "inputs/champion.zip", 153600)
    plan = dict(config=copy.deepcopy(runner.DEFAULT_CONFIG))
    jobs = runner.prepare_jobs(tmp_path, plan, "mirror-stability", "inputs/champion.zip")
    assert len(jobs) == 2
    assert jobs[0]["seed"] == jobs[1]["seed"]
    assert jobs[0]["checkpoint"] == jobs[1]["checkpoint"]
    assert jobs[0]["learning_rate"] != jobs[1]["learning_rate"]
    assert all(j["target_timesteps"] == 253600 for j in jobs)
    assert runner.prepare_jobs(tmp_path, plan, "mirror-stability", "ignored") == jobs
    (tmp_path / jobs[0]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pinned input changed"):
        runner.prepare_jobs(tmp_path, plan, "mirror-stability", "ignored")


def test_replications_start_fresh_with_distinct_seeds(tmp_path):
    plan = dict(config=copy.deepcopy(runner.DEFAULT_CONFIG))
    jobs = runner.prepare_jobs(tmp_path, plan, "mirror-replication", "unavailable-champion.zip")
    assert len({j["seed"] for j in jobs}) == 2
    assert len({j["learning_rate"] for j in jobs}) == 1
    assert all(j["checkpoint"] is None and j["starting_timesteps"] == 0 for j in jobs)


def test_stability_gate_rejects_degradation_even_above_absolute_threshold():
    config = copy.deepcopy(runner.DEFAULT_CONFIG)
    checks = dict(test=result(.91), permuted=result(.91))
    failures = runner.check_gate("mirror-stability", checks, config, initial_rate=.98)
    assert any("frozen start" in failure for failure in failures)
    assert runner.check_gate("mirror-stability", checks, config, initial_rate=.95) == []


@pytest.mark.parametrize("field,value", [("truncated", 1), ("completed_battles", 99), ("status", "running")])
def test_gate_requires_completed_untruncated_battles(field, value):
    checks = dict(test=result(.99), permuted=result(.99))
    checks["test"][field] = value
    assert runner.check_gate("mirror-replication", checks, runner.DEFAULT_CONFIG)


def test_stage_reads_native_history_and_selects_before_test_results(tmp_path, monkeypatch):
    pytest.importorskip("sb3_contrib")
    stage = "mirror-stability"
    config = copy.deepcopy(runner.DEFAULT_CONFIG)
    suites = {key: f"inputs/{stage}-{key}.json" for key in ("calibration", "validation", "test", "permuted")}
    plan = dict(config=config, suites={stage: suites}, imports={}, assets={}, smoke=False)
    checkpoint(tmp_path / "inputs/champion.zip", 153600)
    jobs = [dict(name=f"{stage}-{suffix}") for suffix in ("a", "b")]
    for suffix, rate in (("a", .94), ("b", .96)):
        run = tmp_path / "runs" / f"{stage}-{suffix}"
        # Distinct byte identities despite matching final counters.
        for name, steps in (("best", 200000), ("final", 253600)):
            checkpoint(run / "checkpoints" / f"{name}.zip", steps)
            with ZipFile(run / "checkpoints" / f"{name}.zip", "a") as archive:
                archive.writestr("policy-identity", suffix+name)
        runner.write_json(run / "evaluation-history.json", dict(identity={}, records=[
            dict(status="completed", timesteps=253600, win_rate=rate)]))
    monkeypatch.setattr(runner, "prepare_jobs", lambda *args: jobs)
    monkeypatch.setattr(runner, "launch", lambda *args: None)

    def evaluate(root, plan, label, suite, candidate=None, baseline=None):
        if baseline:
            return result(.5)
        selection = json.loads((root / "stages" / stage / "selection.json").read_text())
        assert selection["name"] == "mirror-stability-b"
        # A has a stronger test outcome but must not replace validation winner B.
        return result(.99 if "-a" in label else .96)

    monkeypatch.setattr(runner, "evaluate_checkpoint", evaluate)
    actual = runner.run_stage(tmp_path, plan, stage, "inputs/champion.zip", 2)
    assert actual["selected_run"] == "mirror-stability-b"
    assert actual["status"] == "passed"
    assert actual["checks"]["test"]["candidate_win_rate"] == .96
