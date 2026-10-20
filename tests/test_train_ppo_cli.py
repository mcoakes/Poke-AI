"""Run the actual CLI training/checkpoint loop with simulated emulator sessions."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO
from test_ppo import Pair
from firered_rl.link_inspection import pair_inspections
from firered_rl.linked_env import linked_observation


@pytest.mark.parametrize("interval,numbered,stop,keep", [
    (1, 8, None, 2), (3, 0, None, 2), (3, 16, None, 2),
    (3, 0, KeyboardInterrupt, 2), (3, 0, RuntimeError, 2),
    (1, 8, None, -1), (1, 8, None, 0),
])
def test_checkpoint_cadence_and_interrupt_safety(tmp_path, monkeypatch, interval, numbered, stop, keep):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "mgba"))
    spec = importlib.util.spec_from_file_location("cadence_cli", root / "mgba/train_ppo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    raw = json.loads((root / "tests/fixtures/link_master_follower.json").read_text())
    observations = [linked_observation(r) for r in pair_inspections(raw)]
    monkeypatch.setattr(cli, "validate_launch", lambda *args: None)

    class Factory:
        def __init__(self, root, run, ports, choose, args, cancel):
            self.choose, self.episode, self.states = choose, 0, None
        def launch(self):
            self.episode += 1
            self.states = self.choose()
        def stop(self): pass
        def release(self, runtime): pass

    class Session(Pair):
        def __init__(self, launch, **kwargs):
            super().__init__(observations, limit=2)
            self.launch = launch
        def reset_pair(self, seeds):
            self.launch()
            return super().reset_pair(seeds)

    monkeypatch.setattr(cli, "EpisodeFactory", Factory)
    monkeypatch.setattr(cli, "LinkedBattleSession", Session)
    states = [tmp_path / f"team{i}.ss1" for i in range(2)]
    for path in states:
        path.write_bytes(b"state")
    run = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--rom", "FireRed.gba", "--states", *map(str, states),
        "--environments", "2", "--rollout-steps", "4", "--batch-size", "4", "--epochs", "1",
        "--total-timesteps", "40", "--latest-every-updates", str(interval),
        "--checkpoint-every", str(numbered), "--keep-checkpoints", str(keep), "--output-dir", str(run)])
    writes = []
    original_save = cli.save_model
    def save(model, *paths):
        writes.append((model.num_timesteps, [p.name for p in paths]))
        return original_save(model, *paths)
    monkeypatch.setattr(cli, "save_model", save)
    if stop:
        original_learn = MaskablePPO.learn
        calls = 0
        def learn(model, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                # A partial rollout must not overwrite a complete saved update.
                model.num_timesteps += 1
                raise stop("interrupted collection")
            return original_learn(model, *args, **kwargs)
        monkeypatch.setattr(MaskablePPO, "learn", learn)
    assert cli.main() == (1 if stop is RuntimeError else 0)
    checkpoints = run / "checkpoints"
    latest = MaskablePPO.load(checkpoints / "latest.zip", device="cpu")
    assert latest.num_timesteps == (8 if stop else 40)
    assert latest.n_steps == 4 and latest.batch_size == 4  # No PPO change.
    assert (checkpoints / "final.zip").exists() == (stop is None)
    if not stop:
        assert (checkpoints / "final.zip").read_bytes() == (checkpoints / "latest.zip").read_bytes()
    assert [step for step, paths in writes if "latest.zip" in paths] == (
        [8] if stop else [8, 16, 24, 32, 40] if interval == 1 else [8, 24, 40])
    expected_backups = [] if not numbered else list(range(numbered, 41, numbered))
    if keep >= 0:
        expected_backups = expected_backups[-keep:] if keep else []
    assert sorted(p.name for p in checkpoints.glob("ppo_*.zip")) == [f"ppo_{n:09d}.zip" for n in expected_backups]
    updates = [json.loads(line) for line in (run / "updates.jsonl").read_text().splitlines()]
    assert len(updates) == (2 if stop else 5)
    assert updates[0]["metrics"]["time/checkpoint_seconds"] > 0
    assert all(set(cli.PPO_DIAGNOSTIC_METRICS) <= set(row["metrics"]) for row in updates)
    assert updates[-1]["metrics"]["train/n_updates"] == len(updates)
    if interval > 1 and not numbered:
        assert updates[1]["metrics"]["time/checkpoint_seconds"] == 0
    summary = json.loads((run / "summary.json").read_text())
    assert summary["checkpointing"]["latest_timesteps"] == latest.num_timesteps
    assert summary["checkpointing"]["keep_checkpoints"] == keep
    assert summary["checkpointing"]["seconds"] > 0
    assert 0 < summary["checkpointing"]["fraction"] <= 1
    assert summary["stop_reason"] == ("interrupted" if stop is KeyboardInterrupt else
                                      "failed" if stop else "timestep_budget")


def test_update_metrics_are_json_safe_and_probability_rejects_nonfinite(tmp_path, monkeypatch):
    import argparse
    import numpy as np
    from test_observation_v2_evaluation import load_cli
    from firered_rl.training_log import TrainingLog
    cli = load_cli("train_ppo", monkeypatch)
    metrics = cli.update_metrics({"train/explained_variance": np.float32("nan"),
        "train/value_loss": np.float64("inf"), "train/n_updates": np.int64(4),
        "train/entropy_loss": np.float32(-.5), "train/policy_gradient_loss": float("nan"),
        "other": object()})
    assert metrics["train/explained_variance"] is None
    assert metrics["train/value_loss"] is None
    assert metrics["train/n_updates"] == 4
    assert metrics["train/entropy_loss"] == -.5
    assert metrics["train/approx_kl"] is None
    assert metrics["train/policy_gradient_loss"] is None
    assert "other" not in metrics
    json.dumps(metrics, allow_nan=False)
    progress = TrainingLog(tmp_path, announce=lambda _: None)
    progress.update(128, 4, metrics)
    saved = json.loads((tmp_path / "updates.jsonl").read_text())
    assert saved["metrics"]["train/policy_gradient_loss"] is None
    restored = TrainingLog(tmp_path, announce=lambda _: None, resume=True)
    assert restored.player_steps == 128 and restored.updates == 4
    for value in ("nan", "inf", "-0.1", "1.1"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.probability(value)


def test_cli_updates_checkpoints_resume_and_episode_logs(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "mgba"))
    spec = importlib.util.spec_from_file_location("training_cli", root / "mgba/train_ppo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = json.loads((root / "tests/fixtures/link_master_follower.json").read_text())
    observations = [linked_observation(r) for r in pair_inspections(raw)]
    monkeypatch.setattr(module, "validate_launch", lambda *args: None)
    factories = []

    class Factory:
        def __init__(self, root, run, ports, choose, args, cancel):
            self.choose, self.episode, self.states = choose, 0, None
            factories.append(self)
        def launch(self):
            self.episode += 1
            self.states = self.choose()
        def stop(self): pass
        def release(self, runtime): pass

    class Session(Pair):
        def __init__(self, launch, **kwargs):
            super().__init__(observations, limit=2)
            self.launch = launch
        def reset_pair(self, seeds):
            self.launch()
            return super().reset_pair(seeds)

    monkeypatch.setattr(module, "EpisodeFactory", Factory)
    monkeypatch.setattr(module, "LinkedBattleSession", Session)
    directory = tmp_path / "states"
    directory.mkdir()
    for i in range(3):
        (directory / f"team{i}.ss1").write_bytes(b"checkpoint")
    first = tmp_path / "first"
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--rom", "FireRed.gba", "--state-dir", str(directory),
        "--environments", "2", "--rollout-steps", "4", "--batch-size", "4", "--epochs", "1",
        "--total-timesteps", "16", "--checkpoint-every", "8", "--output-dir", str(first)])
    assert module.main() == 0
    model_path = first / "checkpoints/latest.zip"
    assert model_path.is_file() and (first / "checkpoints/final.zip").is_file()
    assert len(list((first / "checkpoints").glob("ppo_*.zip"))) == 2
    summary = json.loads((first / "summary.json").read_text())
    assert summary["updates"] == 2 and summary["player_timesteps"] == 16
    assert summary["completed_battles"] == 4 and summary["player_episodes"] == 8
    assert summary["ep_rew_mean"] == 0
    assert summary["player1_ep_rew_mean"] == -1
    assert summary["player2_ep_rew_mean"] == 1
    assert summary["ep_len_mean"] == 2
    assert "rollout/ep_rew_mean" in (first / "ppo/progress.csv").read_text()
    events = [json.loads(row) for row in (first / "episodes.jsonl").read_text().splitlines()]
    assert all(len(event["states"]) == 2 and event["winner"] == "player2" for event in events)
    assert (first / "ppo/progress.csv").is_file()
    second = tmp_path / "resumed"
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--rom", "FireRed.gba", "--states",
        str(directory / "team0.ss1"), str(directory / "team1.ss1"), "--environments", "2",
        "--resume", str(model_path), "--total-timesteps", "8", "--output-dir", str(second)])
    assert module.main() == 0
    assert MaskablePPO.load(second / "checkpoints/final.zip", device="cpu").num_timesteps == 24
    config = json.loads((second / "config.json").read_text())
    assert config["effective_ppo"]["n_steps"] == 4  # saved hyperparameters, not CLI default 64
    assert config["states"]["mode"] == "fixed"
    assert config["reward"]["win"] == 5000 and config["reward"]["scale"] == .001
    assert config["reward"]["turn_penalty"] == 50 and config["reward"]["damage_cap_per_pokemon"] == 100
    loaded = MaskablePPO.load(second / "checkpoints/final.zip", device="cpu")
    assert loaded.reward_settings["turn_penalty"] == 50
    frozen = tmp_path / "frozen"
    previous_bytes = model_path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--rom", "FireRed.gba", "--state-dir", str(directory),
        "--environments", "2", "--resume", str(model_path), "--opponent-checkpoint", str(model_path),
        "--total-timesteps", "8", "--output-dir", str(frozen)])
    assert module.main() == 0
    assert MaskablePPO.load(frozen / "checkpoints/final.zip", device="cpu").num_timesteps == 24
    assert model_path.read_bytes() == previous_bytes
    config = json.loads((frozen / "config.json").read_text())
    assert config["learning_environments"] == 1 and config["frozen_opponent"]["training_timesteps"] == 16
    summary = json.loads((frozen / "summary.json").read_text())
    assert summary["ep_rew_mean"] == -1 and summary["player2_ep_rew_mean"] == 1
    assert summary["learning_players"] == [1]

    # Exercise the actual profile CLI through collection, an update and checkpoint saving.
    monkeypatch.setattr(module, "__file__", str(tmp_path / "mgba/train_ppo.py"))
    for category in ("competitive", "casual", "random"):
        target = tmp_path / "states/counter/profiles" / category / "3"
        target.mkdir(parents=True)
        for i in range(2):
            (target / f"team{i}.ss1").write_bytes(b"counter checkpoint")
    profiled = tmp_path / "profile-run"
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--rom", "FireRed.gba", "--team-profile", "mixed",
        "--team-size", "3", "--environments", "2", "--resume", str(model_path),
        "--reward-config", str(root / "config/rewards/damage_knockout_win.json"),
        "--total-timesteps", "8", "--output-dir", str(profiled)])
    assert module.main() == 0
    config = json.loads((profiled / "config.json").read_text())
    assert config["states"]["mode"] == "team_profile" and config["states"]["team_size"] == 3
    assert config["states"]["category_counts"] == dict(competitive=2, casual=2, random=2)
    assert MaskablePPO.load(profiled / "checkpoints/final.zip", device="cpu").num_timesteps == 24
    events = [json.loads(row) for row in (profiled / "episodes.jsonl").read_text().splitlines()]
    assert events and all(Path(path).parent.name == "3" for event in events for path in event["states"])
