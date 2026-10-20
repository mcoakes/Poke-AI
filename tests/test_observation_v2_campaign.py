"""A new campaign inherits v2 from its initial checkpoint without a campaign flag."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
MaskablePPO = pytest.importorskip("sb3_contrib").MaskablePPO
import gymnasium as gym

from firered_rl.curriculum import read_json, write_json
from firered_rl.ppo_features import BattleFeatures, observation_version, policy_space

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(), reason="Showdown not installed")
def test_new_v2_campaign_infers_child_schemas_with_legacy_v1_reference(tmp_path):
    torch.set_num_threads(1)

    class ShapeEnv(gym.Env):
        action_space = gym.spaces.Discrete(11)
        def __init__(self, version):
            self.observation_space = policy_space(version)

    checkpoints = {}
    for version in ("v1", "v2"):
        model = MaskablePPO("MultiInputPolicy", ShapeEnv(version), n_steps=2, batch_size=2, n_epochs=1,
                            device="cpu", seed=9, policy_kwargs={"features_extractor_class": BattleFeatures})
        if version == "v2":
            model.observation_version = version
        # The v1 reference deliberately has no observation_version metadata.
        checkpoints[version] = tmp_path / f"initial-{version}.zip"
        model.save(checkpoints[version])
    original = {version: path.read_bytes() for version, path in checkpoints.items()}
    for i, species in enumerate(("Raticate", "Linoone"), 1):
        write_json(tmp_path / f"team-{i}.json", {"party": [{"species": species, "level": 1,
            "moves": ["Double Edge", "Body Slam", "Strength", "Facade"]}]})
    manifest = tmp_path / "curriculum.json"
    write_json(manifest, {"version": 1, "name": "v2-campaign-smoke", "opponent_mode": "fixed",
        "stages": [{"stage": 1, "team": "team-1.json", "focus": "schema propagation"}],
        "next_team": "team-2.json"})
    run = tmp_path / "campaign"
    command = [sys.executable, str(ROOT / "mgba/run_curriculum.py"), "--run-dir", str(run),
        "--curriculum", str(manifest),
        "--resume", str(checkpoints["v2"]), "--opponent-checkpoint", str(checkpoints["v1"]),
        "--stage", "1", "--environments", "2", "--timesteps", "2", "--rollout-steps", "2",
        "--batch-size", "2", "--epochs", "1", "--eval-episodes", "4", "--max-steps", "1",
        "--seed", "9", "--showdown-workers", "process", "--retention", "compact",
        "--checkpoint-every", "0", "--no-step-logs", "--keep-episode-logs", "0",
        "--no-battle-log", "--no-readable-log"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    # A one-turn decision cap holds the stage; the complete workflow still runs.
    assert result.returncode == 2, result.stdout + result.stderr
    report = read_json(run / "stage-01/result.json")
    assert report["status"] == "hold" and len(report["evaluations"]) == 4
    trained = MaskablePPO.load(report["candidate"], device="cpu")
    assert trained.num_timesteps == 2
    assert trained.observation_version == observation_version(trained.observation_space) == "v2"
    training = read_json(run / "stage-01/attempt-001/train/config.json")
    assert training["observation_version"] == training["session_observation_version"] == "v2"
    assert training["frozen_opponent"]["observation_version"] == "v1"
    for directory in report["evaluations"].values():
        evaluation = read_json(Path(directory) / "config.json")
        assert evaluation["session_observation_version"] == "v2"
        assert evaluation["models"]["candidate"]["observation_version"] == "v2"
        assert evaluation["models"]["opponent"]["observation_version"] == "v1"
        episodes = [json.loads(row) for row in (Path(directory) / "episodes.jsonl").read_text().splitlines()]
        assert len(episodes) == 4 and {row["candidate_player"] for row in episodes} == {1, 2}
        assert read_json(Path(directory) / "summary.json")["status"] == "completed"
    children = [json.loads(row)["command"] for row in (run / "commands.jsonl").read_text().splitlines()]
    assert len(children) == 5 and all("--observation-version" not in child for child in children)
    assert all("--no-step-logs" in child for child in children)
    assert not list(run.rglob("episode-*"))
    assert checkpoints["v2"].read_bytes() == (run / "inputs/start.zip").read_bytes() == original["v2"]
    assert checkpoints["v1"].read_bytes() == (run / "inputs/opponent.zip").read_bytes() == original["v1"]
