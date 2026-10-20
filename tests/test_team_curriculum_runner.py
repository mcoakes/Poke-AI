"""Runner policy: trusted imports, fixed budgets, validation-only selection."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("sb3_contrib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("team_curriculum_runner_tests", ROOT / "run_team_curriculum.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def config():
    return json.loads((ROOT / "config/experiments/team-curriculum.json").read_text())


def outcome(wins, requested=100, truncated=0):
    return dict(status="completed", attempted_battles=requested,
        requested_battles=requested, completed_battles=requested-truncated,
        candidate_wins=wins, truncated=truncated,
        candidate_win_rate=wins/(requested-truncated), wins_per_requested=wins/requested)


def source_fixture(tmp_path, smoke=False):
    root = tmp_path / "source-run"
    write(root / "plan.json", dict(smoke=smoke))
    write(root / "status.json", dict(status="blocked", stage="mirror-replication"))
    write(root / "inputs/reward.json", {})
    for stage in ("move-choice", "mirror-stability"):
        path = root / "inputs" / f"{stage}.zip"
        path.write_bytes(stage.encode())
        write(root / "stages" / stage / "result.json", dict(
            status="smoke_completed" if smoke else "passed", selected=str(path.relative_to(root)),
            checkpoint_sha256=runner.checksum(path)))
    write(root / "stages/mirror-replication/result.json", dict(status="blocked", issues=["seed b 61% < 90% gate"]))
    return root


def test_import_qualified_policy_preserves_failed_replication(tmp_path):
    source = source_fixture(tmp_path)
    evidence = runner.source_evidence(source)
    assert evidence["evidence"]["mirror-stability"]["status"] == "passed"
    assert evidence["evidence"]["mirror-replication"]["status"] == "blocked"
    assert set(evidence["models"]) == {"move-choice", "mirror"}
    evidence["models"]["mirror"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.source_evidence(source)


def test_import_requires_real_pass_and_smoke_cannot_promote(tmp_path):
    source = source_fixture(tmp_path, smoke=True)
    with pytest.raises(ValueError, match="smoke run cannot supply"):
        runner.source_evidence(source)
    assert runner.source_evidence(source, smoke=True)["models"]["mirror"].is_file()
    path = source / "stages/mirror-stability/result.json"
    data = json.loads(path.read_text()); data["status"] = "blocked"; write(path, data)
    with pytest.raises(ValueError, match="must have passed"):
        runner.source_evidence(source, smoke=True)


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c["training"].update(environments=3), "must be even"),
    (lambda c: c["training"].update(learning_rates=[.0001,float("nan")]), "learning rates"),
    (lambda c: c["training"].update(early_stop_consecutive=0), "early_stop_consecutive"),
    (lambda c: c["training"].update(max_truncation_rate=1.1), "fractions"),
    (lambda c: c["stages"][0]["training"].update(timesteps=0), "timesteps"),
    (lambda c: c["stages"][0].update(id="05-duplicate"), "04 through 10"),
    (lambda c: c["stages"][-1]["counts"].update(benchmark=0), "benchmark"),
])
def test_config_rejects_invalid_before_training(tmp_path, mutation, match):
    value = config(); mutation(value)
    with pytest.raises(ValueError, match=match):
        runner.read_config(write(tmp_path / "config.json", value))


def test_smoke_reduces_all_stages_without_changing_input(tmp_path):
    value = config()
    path = write(tmp_path / "config.json", value)
    small = runner.read_config(path, smoke=True)
    assert json.loads(path.read_text()) == value
    assert small["training"]["environments"] == 2
    assert all(s["training"]["timesteps"] == 256 for s in small["stages"])
    assert all(s["counts"]["train"] == 2 for s in small["stages"])


def job_fixture(tmp_path, monkeypatch):
    root = tmp_path / "run"
    anchor = "inputs/foundation.zip"
    (root / "inputs").mkdir(parents=True)
    (root / anchor).write_bytes(b"foundation")
    c = config(); stage = c["stages"][0]
    plan = dict(config=c, suites={stage["id"]: {"validation": "inputs/validation.json"}})
    monkeypatch.setattr(runner, "checkpoint_steps", lambda path: 401408)
    return root, anchor, plan, stage


def test_job_budgets_and_start_copy_are_pinned(tmp_path, monkeypatch):
    root, anchor, plan, stage = job_fixture(tmp_path, monkeypatch)
    jobs = runner.jobs_for(root, plan, stage, anchor)
    assert len(jobs) == 2 and jobs[0]["seed"] == jobs[1]["seed"]
    assert jobs[0]["learning_rate"] != jobs[1]["learning_rate"]
    assert all(j["target_timesteps"] == 401408 + stage["training"]["timesteps"] for j in jobs)
    changed_stage = copy.deepcopy(stage); changed_stage["training"]["timesteps"] *= 2
    assert runner.jobs_for(root, plan, changed_stage, anchor) == jobs
    (root / jobs[0]["checkpoint"]).write_bytes(b"corrupt-stage-start")
    with pytest.raises(ValueError, match="Stage start changed"):
        runner.jobs_for(root, plan, stage, anchor)


def test_command_uses_real_team_mixture_and_resume_remaining_budget(tmp_path, monkeypatch):
    root, anchor, plan, stage = job_fixture(tmp_path, monkeypatch)
    job = runner.jobs_for(root, plan, stage, anchor)[0]
    cmd = runner.command(root, plan, job)
    assert "--team-mixture" in cmd and "--team-dir" not in cmd
    assert cmd[cmd.index("--team-mixture")+1] == str(root / "inputs" / f"{stage['id']}-mixture.json")
    assert cmd[cmd.index("--resume")+1] == str(root / job["checkpoint"])
    assert cmd[cmd.index("--total-timesteps")+1] == str(stage["training"]["timesteps"])
    run = root / "runs" / job["name"]
    write(run / "summary.json", dict(status="interrupted"))
    monkeypatch.setattr(runner, "recovered_steps", lambda path: job["target_timesteps"] - 1234)
    resumed = runner.command(root, plan, job)
    assert resumed[resumed.index("--total-timesteps")+1] == "1234"
    assert resumed[resumed.index("--if-exists")+1] == "continue"
    monkeypatch.setattr(runner, "recovered_steps", lambda path: job["target_timesteps"])
    retry = runner.command(root, plan, job)
    assert retry[retry.index("--total-timesteps")+1] == "1"
    write(run / "summary.json", dict(status="completed", stop_reason="validation_success"))
    assert runner.command(root, plan, job) is None


def test_gate_counts_truncation_as_nonwin_and_checks_retention():
    c = config(); c["training"].update(max_truncation_rate=.2, retention_max_drop=.05)
    stage = copy.deepcopy(c["stages"][0]); stage["training"]["heldout_win_rate"] = .6
    check = outcome(55, truncated=10)
    assert check["candidate_win_rate"] > .6 and runner.rate(check) == .55
    issues = runner.broad_gate(stage, dict(heldout=check, permuted=check), {}, c)
    assert len(issues) == 2 and all("below 60%" in text for text in issues)
    checks = dict(heldout=outcome(90), permuted=outcome(90))
    assert runner.broad_gate(stage, checks, {"mirror": dict(before=outcome(99),after=outcome(90))}, c)
    assert not runner.broad_gate(stage, checks, {"mirror": dict(before=outcome(99),after=outcome(95))}, c)


@pytest.mark.parametrize("validation_wins,selected_test_wins,issue_text", [
    (91, 55, "heldout"), (55, 99, "validation threshold")])
def test_selection_precedes_frozen_checks_and_never_switches_after_failure(
        tmp_path, monkeypatch, validation_wins, selected_test_wins, issue_text):
    from firered_rl import curriculum_checks
    root, anchor, plan, stage = job_fixture(tmp_path, monkeypatch)
    plan["smoke"] = False
    original_mirror = "inputs/original-qualified-mirror.zip"
    (root / original_mirror).write_bytes(b"original-qualified-mirror")
    plan["imports"] = {"mirror": original_mirror}
    stage["training"]["heldout_win_rate"] = .6
    stage["training"]["validation_win_rate"] = .8
    plan["suites"][stage["id"]].update({key: f"inputs/{key}.json" for key in ("calibration","test","heldout","permuted")})
    plan["suites"][runner.FOUNDATION_IDS[1]] = {"test": "inputs/mirror.json"}
    jobs = runner.jobs_for(root, plan, stage, anchor)
    for i,job in enumerate(jobs):
        path = root / "runs" / job["name"] / "checkpoints"
        path.mkdir(parents=True)
        (path / "best.zip").write_bytes(f"model-{i}".encode())
        (path / "final.zip").write_bytes(f"model-{i}".encode())
        # Intentionally misleading legacy win_rate; requested counts win selection.
        write(path / "best.json", dict(checkpoint_sha256=runner.checksum(path / "best.zip"),
              wins=validation_wins if i == 0 else validation_wins-2, requested=100, win_rate=.01 if i == 0 else .99))
    monkeypatch.setattr(runner, "launch", lambda *args: None)
    inspected, checked_paths = [], {}

    def frozen(project, output, label, suite, checkpoint=None, baseline=None, **kwargs):
        if baseline:
            return outcome(50)
        selection = json.loads((root / "stages" / stage["id"] / "selection.json").read_text())
        inspected.append(selection["selected"]["name"])
        checked_paths[label] = checkpoint
        # Arm b wins frozen checks more often; must not displace validation winner a.
        return outcome(selected_test_wins if jobs[0]["name"] in str(checkpoint) else 99)

    monkeypatch.setattr(curriculum_checks, "frozen_check", frozen)
    result = runner.run_stage(root, plan, stage, anchor, None, 2)
    assert result["selected_run"] == jobs[0]["name"]
    assert result["status"] == "blocked"
    assert any(issue_text in issue for issue in result["issues"])
    assert set(inspected) == {jobs[0]["name"]}
    assert (root / result["selected"]).read_bytes() == b"model-0"
    assert checked_paths[f"{stage['id']}-retain-fixed-mirror-before"] == root / original_mirror


def test_inspect_teams_shows_saved_paths_moves_and_full_json(tmp_path, capsys):
    path = write(tmp_path / "inputs/train/team.json", {"name":"visible-team", "party":[
        {"species":"Blastoise", "moves":["Surf","Ice Beam"], "nature":"Modest", "held_item":"Leftovers"}]})
    entry = dict(root=str(tmp_path / "inputs"), train_paths=["train/team.json"])
    plan = dict(fixed={"move-choice":entry, "mirror":entry}, assets={"04-types":entry})
    runner.inspect(tmp_path, plan, "4", "train", 1, False)
    text = capsys.readouterr().out
    assert str(path) in text and "Surf, Ice Beam" in text and "Leftovers" in text
    runner.inspect(tmp_path, plan, "04", "train", 1, True)
    assert '"name": "visible-team"' in capsys.readouterr().out
