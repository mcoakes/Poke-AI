"""Guard experimental validity: seat changes must not reverse the advantage."""
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

pytest.importorskip("sb3_contrib")
from firered_rl.fixed_team_lab import teams, make_suite
from firered_rl.evaluation import validate_suite

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scenario", ["move-choice", "mirror"])
@pytest.mark.parametrize("permuted", [False, True])
def test_team_identity_and_seat_balance(tmp_path, scenario, permuted):
    originals = teams()[scenario]
    paths = [tmp_path / f"{i}.json" for i in range(2)]
    for path, value in zip(paths, originals):
        path.write_text(json.dumps(value))
    suite = make_suite(ROOT, paths, 16, 17, scenario, permute_moves=permuted)
    validate_suite(suite)
    assert sum(c["candidate_player"] == 1 for c in suite["cases"]) == 8
    surf_slots = set()
    for case in suite["cases"]:
        seat = case["candidate_player"] - 1
        candidate = suite["assets"][case["assets"][seat]]["team"]
        opponent = suite["assets"][case["assets"][1-seat]]["team"]
        assert opponent == originals[1]
        for actual, original in zip(candidate["party"], originals[0]["party"]):
            assert {k: v for k, v in actual.items() if k != "moves"} == {k: v for k, v in original.items() if k != "moves"}
            assert sorted(actual["moves"]) == sorted(original["moves"])
        if scenario == "move-choice":
            assert candidate["party"][0]["species"] == "Starmie"
            assert opponent["party"][0]["species"] == "Golem"
            surf_slots.add(candidate["party"][0]["moves"].index("Surf"))
    if scenario == "move-choice":
        assert len(surf_slots) == (4 if permuted else 1)


def test_validation_and_test_use_different_battle_seeds(tmp_path):
    paths = [tmp_path / f"{i}.json" for i in range(2)]
    for path, value in zip(paths, teams()["mirror"]):
        path.write_text(json.dumps(value))
    suites = [make_suite(ROOT, paths, 256, seed, "mirror") for seed in (731702, 731703)]
    assert not {c["seed"] for c in suites[0]["cases"]} & {c["seed"] for c in suites[1]["cases"]}


def test_resume_preserves_remaining_budget_and_retries_unfinished_final_validation(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT))
    from run_fixed_team_lab import training_command
    job = dict(name="mirror-reference", scenario="mirror", initialization="reference",
               starting_timesteps=401408, additional_timesteps=200000, target_timesteps=601408)
    plan = dict(seed=10, environments=16)
    command = training_command(tmp_path, plan, job)
    assert "--teams" in command and "--team-dir" not in command
    assert "--no-eval-stop" in command and "--eval-stop-win-rate" not in command
    assert "--resume" in command and command[command.index("--total-timesteps") + 1] == "200000"
    run = tmp_path / job["name"]
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/latest.zip"
    with ZipFile(checkpoint, "w") as z:
        z.writestr("data", json.dumps(dict(num_timesteps=501408)))
    command = training_command(tmp_path, plan, job)
    assert command[command.index("--total-timesteps") + 1] == "100000"
    with ZipFile(checkpoint, "w") as z:
        z.writestr("data", json.dumps(dict(num_timesteps=601408)))
    command = training_command(tmp_path, plan, job)
    assert command[command.index("--total-timesteps") + 1] == "1"
    (run / "summary.json").write_text(json.dumps(dict(status="completed")))
    assert training_command(tmp_path, plan, job) is None


@pytest.mark.parametrize("scenario,threshold", [("move-choice", .99), ("mirror", .95)])
def test_new_plan_success_stopping_and_native_continuation(tmp_path, scenario, threshold):
    import sys
    sys.path.insert(0, str(ROOT))
    from run_fixed_team_lab import training_command
    job = dict(name=f"{scenario}-fresh", scenario=scenario, initialization="fresh",
               starting_timesteps=0, additional_timesteps=200000, target_timesteps=200000)
    plan = dict(seed=10, environments=16, early_stopping=dict(enabled=True, consecutive=2,
                win_rates={"move-choice": .99, "mirror": .95}))
    command = training_command(tmp_path, plan, job)
    assert command[command.index("--eval-stop-win-rate") + 1] == str(threshold)
    assert command[command.index("--eval-stop-consecutive") + 1] == "2"
    run = tmp_path / job["name"]
    (run / "checkpoints").mkdir(parents=True)
    with ZipFile(run / "checkpoints/latest.zip", "w") as z:
        z.writestr("data", json.dumps(dict(num_timesteps=51200)))
    command = training_command(tmp_path, plan, job)
    assert command[command.index("--total-timesteps") + 1] == "148800"
    assert command[command.index("--eval-stop-win-rate") + 1] == str(threshold)
    (run / "summary.json").write_text(json.dumps(dict(status="completed", stop_reason="validation_success")))
    assert training_command(tmp_path, plan, job) is None
    (run / "summary.json").write_text(json.dumps(dict(status="interrupted", stop_reason="interrupted")))
    plan["early_stopping"]["enabled"] = False
    command = training_command(tmp_path, plan, job)
    assert "--no-eval-stop" in command and "--eval-stop-win-rate" not in command


def test_collector_includes_update_and_switching_diagnostics(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT))
    from run_fixed_team_lab import collect
    included = ["mirror-fresh/updates.jsonl", "mirror-fresh/battle-blocks.jsonl",
                "mirror-fresh/switching.json", "mirror-fresh/ppo/progress.csv",
                "mirror-fresh/sessions/000002/ppo/progress.csv"]
    excluded = ["mirror-fresh/episodes.jsonl", "mirror-fresh/pair-1/decisions.jsonl"]
    for name in included + excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    collect(tmp_path)
    with ZipFile(tmp_path / "fixed-team-results.zip") as archive:
        assert set(included) <= set(archive.namelist())
        assert not set(excluded) & set(archive.namelist())


@pytest.mark.parametrize("disabled", [False, True])
def test_new_batch_pins_early_stopping_policy(tmp_path, monkeypatch, disabled):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT))
    from run_fixed_team_lab import freeze_inputs
    monkeypatch.setattr("firered_rl.showdown_env.validate_showdown", lambda *args: None)
    args = SimpleNamespace(fresh_only=True, reward_config=ROOT / "config/rewards/damage_knockout_win.json",
        calibration_battles=4, validation_battles=4, test_battles=4, move_steps=50000,
        mirror_steps=200000, seed=261030, environments=16, no_early_stop=disabled)
    root = tmp_path / "new-batch"
    plan = freeze_inputs(root, args)
    saved = json.loads((root / "plan.json").read_text())
    assert saved["early_stopping"] == plan["early_stopping"] == dict(
        enabled=not disabled, consecutive=2, win_rates={"move-choice": .99, "mirror": .95})
