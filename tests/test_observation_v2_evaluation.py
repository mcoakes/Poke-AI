"""Schema projection, provenance, and native-backend guards for frozen policies."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from threading import Event
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO

from firered_rl.evaluation import evaluate
from firered_rl.human_play import PPOOpponent
from firered_rl.observation_v2 import versioned_raw_space
from firered_rl.ppo_features import BattleFeatures, policy_space


ROOT = Path(__file__).resolve().parents[1]


def load_cli(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    spec = importlib.util.spec_from_file_location(name + "_v2_test", ROOT / "mgba" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_v2():
    obs = {key: np.zeros(space.shape, dtype=space.dtype)
           for key, space in versioned_raw_space("v2").spaces.items()}
    obs["action_mask"][0] = 1
    obs["party"][0, 7] = (1 << 30) | 1
    obs["v2_party"][0, 0, 9] = (1 << 30) | 1
    obs["v2_boosts"][0, 0] = -2
    return obs


class SchemaPolicy:
    def __init__(self, version):
        self.observation_space = policy_space(version)
        self.version, self.calls = version, 0

    def predict(self, obs, action_masks, deterministic):
        assert set(obs) == set(self.observation_space.spaces)
        assert deterministic and action_masks.dtype == bool
        assert np.all(obs["status_bits"][:, 0, [0, 30]] == 1)
        assert np.all(obs["party"][:, 0, 7] == 0)
        if self.version == "v2":
            assert np.all(obs["v2_boosts"][:, 0, 0] == -2)
            assert np.all(obs["v2_status_bits"][:, 0, 0, [0, 30]] == 1)
        self.calls += len(action_masks)
        return action_masks.argmax(axis=1), None


class EvaluationPair:
    def reset_pair(self, seeds):
        return [(raw_v2(), {}), (raw_v2(), {})]

    def step_pair(self, actions):
        assert list(actions) == [0, 0]
        return [(raw_v2(), 1, True, False, {"outcome": "won"}),
                (raw_v2(), -1, True, False, {"outcome": "lost"})]


@pytest.mark.parametrize("candidate_version", ["v1", "v2"])
def test_each_pool_policy_gets_its_schema_on_both_sides(tmp_path, candidate_version):
    candidate, old, new = SchemaPolicy(candidate_version), SchemaPolicy("v1"), SchemaPolicy("v2")
    pool = {"old": old, "new": new, "random": None}
    cases = [dict(episode=i * 2 + side, matchup=i + 1, assets=["a", "b"], seed=71 + i,
                  candidate_player=side, opponent_id=key, opponent_group="random" if key == "random" else "smart")
             for i, key in enumerate(pool) for side in (1, 2)]
    suite = dict(backend="showdown", cases=cases,
                 opponents={key: {"label": key} for key in pool})
    stats = evaluate([EvaluationPair(), EvaluationPair()], [[], []], suite,
                     {"a": Path("a"), "b": Path("b")}, [candidate], tmp_path, Event(),
                     opponents=pool, announce=lambda _: None)
    assert stats["completed_battles"] == 6
    assert stats["candidate_wins"] == stats["opponent_wins"] == 3
    assert candidate.calls == 6 and old.calls == new.calls == 2


def model_double(version):
    return SimpleNamespace(observation_space=policy_space(version), action_space=gym.spaces.Discrete(11),
                           num_timesteps=0, policy=SimpleNamespace(set_training_mode=lambda _: None))


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_tournament_preflight_reports_model_schema_and_guards_mgba(tmp_path, monkeypatch, version):
    cli = load_cli("evaluate_models", monkeypatch)
    model = model_double(version)
    monkeypatch.setattr(MaskablePPO, "load", lambda *args, **kwargs: model)
    info = cli.validate_model(tmp_path / "model.zip")
    assert info["observation_version"] == version
    assert info["supported_backends"] == ["showdown", "mgba"]
    cli.validate_model(tmp_path / "model.zip", backend="mgba")
    assert PPOOpponent(model).model is model
    model.observation_space = gym.spaces.Dict({"partial": gym.spaces.Box(0, 1, (1,))})
    with pytest.raises(ValueError, match="observation space"):
        cli.validate_model(tmp_path / "model.zip")


def test_mgba_evaluation_accepts_v2_and_reaches_native_validation(tmp_path, monkeypatch, capsys):
    cli = load_cli("evaluate_ppo", monkeypatch)
    model = model_double("v2")
    monkeypatch.setattr(MaskablePPO, "load", lambda *args, **kwargs: model)
    def validate(*args, **kwargs):
        raise ValueError('native validation reached')
    monkeypatch.setattr(cli, "validate_launch", validate)
    checkpoint = tmp_path / "v2.zip"
    checkpoint.write_bytes(b"checkpoint")
    states = [tmp_path / f"state{i}.ss1" for i in range(2)]
    for state in states:
        state.write_bytes(b"state")
    run = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["evaluate_ppo", "--backend", "mgba", "--checkpoint", str(checkpoint),
        "--rom", str(tmp_path/'FireRed.gba'),
        "--states", *map(str, states), "--random-opponent", "--episodes", "2", "--output-dir", str(run)])
    with pytest.raises(SystemExit):
        cli.main()
    assert "native validation reached" in capsys.readouterr().err
    assert not run.exists()


@pytest.mark.skipif(not shutil.which("node") or not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
                    reason="Showdown not installed")
@pytest.mark.parametrize("candidate_version", ["v1", "v2"])
def test_real_mixed_checkpoint_replay_keeps_per_model_schema_and_identity(tmp_path, monkeypatch, candidate_version):
    import torch
    torch.set_num_threads(1)
    checkpoints = {}
    for version in ("v1", "v2"):
        class ShapeEnv(gym.Env):
            observation_space = policy_space(version)
            action_space = gym.spaces.Discrete(11)
        model = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2, seed=71,
                            policy_kwargs=dict(features_extractor_class=BattleFeatures))
        checkpoints[version] = tmp_path / f"{version}.zip"
        model.save(checkpoints[version])
    originals = {version: path.read_bytes() for version, path in checkpoints.items()}

    def no_training(*args, **kwargs):
        raise AssertionError("evaluation must never train or save weights")

    monkeypatch.setattr(MaskablePPO, "learn", no_training)
    monkeypatch.setattr(MaskablePPO, "save", no_training)
    teams = [tmp_path / f"team{i}.json" for i in range(2)]
    for path, species, move, level in zip(teams, ["Magikarp", "Blastoise"], ["Splash", "Surf"], [1, 50]):
        path.write_text(json.dumps(dict(party=[dict(species=species, moves=[move], level=level)])))
    opponent_version = "v1" if candidate_version == "v2" else "v2"
    cli = load_cli("evaluate_ppo", monkeypatch)
    common = ["evaluate_ppo", "--checkpoint", str(checkpoints[candidate_version]),
              "--opponent-checkpoint", str(checkpoints[opponent_version])]
    first, replay = tmp_path / "first", tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", [*common, "--teams", *map(str, teams), "--episodes", "4", "--max-steps", "4",
                                      "--environments", "4", "--output-dir", str(first)])
    assert cli.main() == 0
    for path in teams:
        path.unlink()
    monkeypatch.setattr(sys, "argv", [*common, "--suite", str(first / "suite.json"), "--environments", "2",
                                      "--showdown-workers", "process", "--output-dir", str(replay)])
    assert cli.main() == 0
    rows = lambda run: sorted((json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()),
                              key=lambda row: row["episode"])
    assert rows(first) == rows(replay)
    assert {row["candidate_player"] for row in rows(first)} == {1, 2}
    for run in (first, replay):
        info = json.loads((run / "config.json").read_text())
        assert (run / "candidate.zip").read_bytes() == originals[candidate_version]
        assert info["models"]["candidate"]["snapshot_path"] == str(run / "candidate.zip")
        assert info["observation_version"] == info["session_observation_version"] == "v2"
        for role, version in (("candidate", candidate_version), ("opponent", opponent_version)):
            assert info["models"][role]["observation_version"] == version
            assert info["models"][role]["sha256"] == hashlib.sha256(originals[version]).hexdigest()
    assert all(path.read_bytes() == originals[version] for version, path in checkpoints.items())
