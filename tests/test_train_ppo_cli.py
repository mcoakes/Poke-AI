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
