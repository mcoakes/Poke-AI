"""Exercise standalone continuation through the real PPO CLI and simulator."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO
from test_observation_v2_training import real_vec, small_model


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not shutil.which("node") or not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
    reason="Showdown is not installed",
)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def fingerprint(directory):
    return {str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def training(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    spec = importlib.util.spec_from_file_location("continue_training_cli", ROOT / "mgba/train_ppo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    teams = [tmp_path / f"team-{side}.json" for side in range(2)]
    for team, pokemon in zip(teams, [
            {"species": "Pikachu", "level": 100, "moves": ["Thunderbolt"]},
            {"species": "Magikarp", "level": 5, "moves": ["Splash"]}]):
        team.write_text(json.dumps({"party": [pokemon]}))
    # A legacy-schema frozen model tests that both schemas survive continuation.
    vec = real_vec(tmp_path / "frozen-source", teams, "v1")
    frozen = tmp_path / "frozen.zip"
    try:
        small_model(vec).save(frozen)
    finally:
        vec.close()
    reward = tmp_path / "reward.json"
    reward.write_text(json.dumps({**read(ROOT / "config/rewards/damage_knockout_win.json"), "turn_penalty": 37}))
    run = tmp_path / "run"

    def invoke(*flags):
        monkeypatch.setattr(sys, "argv", ["train_ppo", "--output-dir", str(run), *map(str, flags)])
        return cli.main()

    new_flags = ["--backend", "showdown", "--observation-version", "v2", "--showdown-workers", "process",
                 "--teams", *teams, "--opponent-checkpoint", frozen, "--reward-config", reward,
                 "--environments", 2, "--rollout-steps", 4, "--batch-size", 4, "--epochs", 1,
                 "--max-steps", 2, "--total-timesteps", 8, "--checkpoint-every", 4,
                 "--keep-checkpoints", 2, "--log-max-mb", 1, "--log-backups", 1]
    assert invoke(*new_flags) == 0
    return SimpleNamespace(cli=cli, run=run, teams=teams, frozen=frozen, reward=reward,
                           invoke=invoke, new_flags=new_flags)


@pytest.mark.parametrize("legacy_run", [False, True], ids=["current", "legacy-033"])
def test_continue_preserves_schema_frozen_inputs_history_and_recovers_newest(training, legacy_run):
    t, run = training, training.run
    if legacy_run:
        # v0.33 completed runs have no saved_args/sessions/input snapshots. Their
        # expanded source descriptions and root logs must still be recoverable.
        config = read(run / "config.json")
        config.pop("saved_args")
        config.pop("session")
        config["opponent_checkpoint"] = config["frozen_opponent"]["path"] = str(t.frozen)
        config["reward_config"] = str(t.reward)
        (run / "config.json").write_text(json.dumps(config))
        shutil.rmtree(run / "inputs")
        shutil.rmtree(run / "sessions")
        summary = read(run / "summary.json")
        for key in ("session", "session_count", "session_id", "session_elapsed_seconds"):
            summary.pop(key, None)
        (run / "summary.json").write_text(json.dumps(summary))
        for name in ("episodes.jsonl", "updates.jsonl"):
            events = rows(run / name)
            for event in events:
                event.pop("session_id", None)
            (run / name).write_text("".join(json.dumps(event) + "\n" for event in events))
    original_config = (run / "config.json").read_bytes()
    original_csv = (run / "ppo/progress.csv").read_bytes()
    original_log = (run / "pair-1/battle-readable.log").read_bytes()
    initial_checkpoint = (run / "checkpoints/final.zip").read_bytes()
    pinned = t.frozen.read_bytes()
    original_external_frozen = t.frozen.read_bytes()
    first = read(run / "summary.json")
    original_events = (run / "episodes.jsonl").read_bytes()
    original_updates = (run / "updates.jsonl").read_bytes()
    # A successful run may contain a stale diagnostic from an older invocation.
    (run / "failure.json").write_text(json.dumps({"error": "obsolete failure"}))

    # Only the output directory, continuation choice and additional budget needed.
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 8) == 0
    second = read(run / "summary.json")
    config = read(run / "config.json")
    loaded = MaskablePPO.load(run / "checkpoints/final.zip", device="cpu")
    assert loaded.num_timesteps == 16 and loaded.observation_version == "v2"
    assert (loaded.n_steps, loaded.batch_size, loaded.n_epochs) == (4, 4, 1)
    assert config["backend"] == "showdown" and config["showdown_workers"] == "process"
    assert config["learning_environments"] == 1 and config["observation_version"] == "v2"
    assert config["frozen_opponent"]["observation_version"] == "v1"
    assert config["reward"]["turn_penalty"] == 37
    assert config["resume_overrides"] == {} and config["effective_ppo"]["starting_timesteps"] == 8
    assert (run / "sessions/000001/config.json").read_bytes() == original_config
    assert (run / "ppo/progress.csv").read_bytes() == original_csv
    assert (run / "pair-1/battle-readable.log").read_bytes() == original_log
    assert (run / "sessions/000002/ppo/progress.csv").is_file()
    assert (run / "sessions/000002/pair-1/battle-readable.log").is_file()
    assert (run / "episodes.jsonl").read_bytes().startswith(original_events)
    assert (run / "updates.jsonl").read_bytes().startswith(original_updates)
    assert second["finished_attempts"] == first["finished_attempts"] * 2
    assert second["completed_battles"] == first["completed_battles"] * 2 > 0
    assert second["updates"] == first["updates"] * 2 == 4
    assert second["elapsed_seconds"] > first["elapsed_seconds"] > 0
    assert second["episodes_per_hour"] == pytest.approx(second["completed_battles"] * 3600 / second["elapsed_seconds"])
    assert second["session_count"] == 2 and second["session"]["ended_battles"] == first["finished_attempts"]
    assert {row.get("session_id", 1) for row in rows(run / "episodes.jsonl")} == {1, 2}
    assert not (run / "failure.json").exists()
    assert not (run / "sessions/000001/failure.json").exists()

    # Simulate a later numbered recovery point with stale latest/final aliases.
    assert (run / "checkpoints/ppo_000000016.zip").is_file()
    for name in ("latest.zip", "final.zip"):
        (run / "checkpoints" / name).write_bytes(initial_checkpoint)
    (run / "summary.json").write_text(json.dumps({**second, "status": "failed"}))
    failure = json.dumps({"error": "interrupted after a saved numbered checkpoint"}).encode()
    (run / "failure.json").write_bytes(failure)
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 4) == 0
    third = read(run / "summary.json")
    assert third["session"]["recovered_checkpoint"].endswith("ppo_000000016.zip")
    assert third["session"]["starting_timesteps"] == 16 and third["player_timesteps"] == 20
    assert third["updates"] == 5 and third["session_count"] == 3
    assert (run / "sessions/000002/failure.json").read_bytes() == failure
    assert not (run / "failure.json").exists()
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 4) == 0
    fourth = read(run / "summary.json")
    assert fourth["player_timesteps"] == 24 and fourth["updates"] == 6
    assert fourth["session_count"] == 4
    assert [read(run / f"sessions/{i:06d}/config.json")["session"]["seed"] for i in range(2, 5)] == [
        1000004, 2000007, 3000010]
    assert (run / "inputs/frozen-opponent.zip").read_bytes() == pinned == original_external_frozen
    assert t.frozen.read_bytes() == original_external_frozen


def test_existing_run_cancel_noninteractive_and_incompatible_flags_are_read_only(training, monkeypatch, capsys):
    t, run = training, training.run
    before = fingerprint(run)
    assert t.invoke("--if-exists", "cancel") == 0
    assert fingerprint(run) == before
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(SystemExit) as error:
        t.invoke()
    assert error.value.code == 2
    assert "non-interactive" in capsys.readouterr().err
    assert fingerprint(run) == before
    with pytest.raises(SystemExit) as error:
        t.invoke("--if-exists", "continue", "--observation-version", "v1")
    assert error.value.code == 2
    assert "differs from this run" in capsys.readouterr().err
    assert fingerprint(run) == before


def test_changed_reward_team_and_frozen_checkpoint_rejected_before_writes(training, capsys):
    t, run = training, training.run
    mutations = [
        (run / "inputs/reward.json", lambda p: p.write_text(json.dumps({**read(p), "turn_penalty": 38})),
         "reward configuration changed"),
        (t.teams[0], lambda p: p.write_text(json.dumps({"party": [{"species": "Magikarp", "level": 6,
                                                               "moves": ["Splash"]}]})),
         "team/state selection or file content changed"),
    ]
    for path, mutate, message in mutations:
        old = path.read_bytes()
        try:
            mutate(path)
            before = fingerprint(run)
            with pytest.raises(SystemExit) as error:
                t.invoke("--if-exists", "continue", "--total-timesteps", 4)
            assert error.value.code == 2 and message in capsys.readouterr().err
            assert fingerprint(run) == before
        finally:
            path.write_bytes(old)
    checkpoint = run / "inputs/frozen-opponent.zip"
    original = checkpoint.read_bytes()
    try:
        model = MaskablePPO.load(checkpoint, device="cpu")
        model.num_timesteps += 1
        model.save(checkpoint)
        before = fingerprint(run)
        with pytest.raises(SystemExit) as error:
            t.invoke("--if-exists", "continue", "--total-timesteps", 4)
        assert error.value.code == 2 and "frozen opponent checkpoint changed" in capsys.readouterr().err
        assert fingerprint(run) == before
    finally:
        checkpoint.write_bytes(original)
    events = run / "episodes.jsonl"
    original = events.read_bytes()
    try:
        events.write_bytes(b"{broken interior record}\n" + original)
        before = fingerprint(run)
        with pytest.raises(SystemExit) as error:
            t.invoke("--if-exists", "continue", "--total-timesteps", 4)
        assert error.value.code == 2 and "cannot continue metrics: malformed" in capsys.readouterr().err
        assert fingerprint(run) == before
    finally:
        events.write_bytes(original)


def test_erase_waits_for_valid_preflight_then_restarts_fresh(training, capsys):
    t, run = training, training.run
    marker = run / "keep-me.txt"
    marker.write_text("old experiment")
    before = fingerprint(run)
    with pytest.raises(SystemExit) as error:
        t.invoke(*t.new_flags, "--if-exists", "erase", "--batch-size", 3)
    assert error.value.code == 2 and "batch-size" in capsys.readouterr().err
    assert fingerprint(run) == before
    assert t.invoke(*t.new_flags, "--if-exists", "erase", "--total-timesteps", 4) == 0
    assert not marker.exists() and not (run / "sessions/000002").exists()
    summary = read(run / "summary.json")
    config = read(run / "config.json")
    assert summary["player_timesteps"] == 4 and summary["updates"] == 1 and summary["session_count"] == 1
    assert config["effective_ppo"]["starting_timesteps"] == 0 and "resume_checkpoint" not in config
    assert {entry["session_id"] for entry in rows(run / "episodes.jsonl")} == {1}


def test_campaign_attempt_cannot_be_continued_by_standalone_cli(training, monkeypatch, capsys, tmp_path):
    t = training
    attempt = tmp_path / "campaign/stage-01/attempt-001/train"
    attempt.parent.mkdir(parents=True)
    shutil.copytree(t.run, attempt)
    before = fingerprint(attempt)
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--output-dir", str(attempt), "--if-exists", "continue"])
    with pytest.raises(SystemExit) as error:
        t.cli.main()
    assert error.value.code == 2 and "run_curriculum.py" in capsys.readouterr().err
    assert fingerprint(attempt) == before


def test_success_stop_saves_updated_weights_and_survives_native_continue(training, monkeypatch):
    from firered_rl.evaluation import create_suite
    from firered_rl.training_teams import TrainingTeams
    t, run = training, training.run
    suite = create_suite(TrainingTeams(teams=t.teams), backend="showdown", episodes=4,
                         seed=778, swap_sides=True, max_steps=2)
    suite_path = run.parent / "validation-suite.json"
    suite_path.write_text(json.dumps(suite))
    observed = []
    original = t.cli.PeriodicEvaluation.evaluate

    def evaluate(controller, checkpoint, timesteps, *, reason):
        loaded = MaskablePPO.load(checkpoint, device="cpu")
        observed.append((timesteps, loaded.num_timesteps, loaded._n_updates))
        return original(controller, checkpoint, timesteps, reason=reason)

    monkeypatch.setattr(t.cli.PeriodicEvaluation, "evaluate", evaluate)
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 100,
                    "--eval-suite", suite_path, "--eval-every", 4,
                    "--eval-stop-win-rate", .5, "--eval-stop-consecutive", 2) == 0
    summary = read(run / "summary.json")
    assert summary["status"] == "completed" and summary["stop_reason"] == "validation_success"
    assert summary["early_stopping"]["consecutive"] == 2
    assert summary["player_timesteps"] == 12
    assert observed == [(8, 8, 2), (12, 12, 3)]
    final = MaskablePPO.load(run / "checkpoints/final.zip", device="cpu")
    assert final.num_timesteps == 12 and final._n_updates == 3
    assert (run / "checkpoints/final.zip").read_bytes() == (run / "checkpoints/latest.zip").read_bytes()
    old_updates = (run / "updates.jsonl").read_bytes()
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 100) == 0
    summary = read(run / "summary.json")
    assert summary["stop_reason"] == "validation_success" and summary["session"]["updates"] == 0
    assert summary["player_timesteps"] == 12
    assert (run / "updates.jsonl").read_bytes() == old_updates
    config = read(run / "config.json")
    assert config["eval_stop_win_rate"] == .5 and config["eval_stop_consecutive"] == 2
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 4, "--no-eval-stop") == 0
    assert read(run / "summary.json")["stop_reason"] == "timestep_budget"
    assert MaskablePPO.load(run / "checkpoints/final.zip", device="cpu").num_timesteps == 16
