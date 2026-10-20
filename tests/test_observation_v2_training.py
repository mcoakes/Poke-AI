"""Short real-simulator checks for schema selection, resume and mixed opponents."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
MaskablePPO = pytest.importorskip("sb3_contrib").MaskablePPO
from firered_rl.frozen_opponent import FrozenOpponentVecEnv
from firered_rl.ppo_features import BattleFeatures, observation_version, policy_space
from firered_rl.ppo_vec_env import LinkedVecEnv
from firered_rl.showdown_env import ShowdownBattleSession, ShowdownFactory

ROOT = Path(__file__).resolve().parents[1]
SHOWDOWN = shutil.which("node") and (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file()
requires_showdown = pytest.mark.skipif(not SHOWDOWN, reason="Showdown not installed")


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    spec = importlib.util.spec_from_file_location("observation_v2_cli", ROOT / "mgba/train_ppo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def teams(tmp_path):
    files = [tmp_path / f"team-{i}.json" for i in range(2)]
    for path in files:
        path.write_text(json.dumps({"party": [{"species": "Magikarp", "level": 5, "moves": ["Splash"]}]}))
    return files


def command(run, teams, *flags):
    return ["train_ppo", "--backend", "showdown", "--environments", "2", "--teams", *map(str, teams),
            "--total-timesteps", "8", "--max-steps", "2", "--checkpoint-every", "0",
            "--no-readable-log", "--no-battle-log", "--output-dir", str(run), *flags]


@requires_showdown
@pytest.mark.parametrize("workers", ["thread", "process"])
def test_v2_real_cli_learn_save_load_resume_and_schema_mismatch(tmp_path, monkeypatch, capsys, cli, teams, workers):
    first = tmp_path / "first"
    monkeypatch.setattr(sys, "argv", command(first, teams, "--observation-version", "v2",
        "--showdown-workers", workers, "--rollout-steps", "4", "--batch-size", "4", "--epochs", "1"))
    assert cli.main() == 0
    checkpoint = first / "checkpoints/final.zip"
    initial_bytes = checkpoint.read_bytes()
    saved = MaskablePPO.load(checkpoint, device="cpu")
    assert saved.num_timesteps == 8 and saved.observation_version == "v2"
    assert observation_version(saved.observation_space) == "v2"
    assert saved.policy.optimizer.state
    before = deepcopy(saved.policy.state_dict())
    resumed = tmp_path / "resumed"
    monkeypatch.setattr(sys, "argv", command(resumed, teams, "--resume", str(checkpoint),
                                             "--showdown-workers", workers))
    assert cli.main() == 0
    loaded = MaskablePPO.load(resumed / "checkpoints/final.zip", device="cpu")
    assert loaded.num_timesteps == 16 and loaded.observation_version == "v2"
    assert (loaded.n_steps, loaded.batch_size, loaded.n_epochs) == (4, 4, 1)
    assert all(torch.isfinite(value).all() for value in loaded.policy.state_dict().values())
    assert any(not torch.equal(value, loaded.policy.state_dict()[name]) for name, value in before.items())
    config = json.loads((resumed / "config.json").read_text())
    assert config["observation_version"] == config["session_observation_version"] == "v2"
    assert config["effective_ppo"]["starting_timesteps"] == 8
    assert checkpoint.read_bytes() == initial_bytes
    assert not list(first.glob("pair-*/episode-*"))

    # Reject a schema override before starting a simulator or creating a run.
    rejected = tmp_path / "rejected"
    monkeypatch.setattr(sys, "argv", command(rejected, teams, "--resume", str(checkpoint),
                                             "--observation-version", "v1"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not rejected.exists()
    assert "does not match the v2 resume checkpoint" in capsys.readouterr().err


def test_mgba_v2_reaches_native_launch_validation(tmp_path, monkeypatch, capsys, cli):
    def unexpected(*args):
        raise ValueError("native launch validation reached")
    monkeypatch.setattr(cli, "validate_launch", unexpected)
    run = tmp_path / "unsupported"
    (tmp_path / 'battle.ss1').write_bytes(b'state')
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--rom", str(tmp_path/'FireRed.gba'),
                                      "--observation-version", "v2", "--state-dir", str(tmp_path),
                                      "--output-dir", str(run)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not run.exists()
    assert "native launch validation reached" in capsys.readouterr().err


def real_vec(path, teams, version):
    args = SimpleNamespace(seed=3, node="node", command_timeout=15, observation_version=version)
    cancel = Event()
    factory = ShowdownFactory(ROOT, path, [55355, 55356], lambda: tuple(teams), args, cancel)
    session = ShowdownBattleSession(factory, max_steps=1)
    return LinkedVecEnv([session], cancel=cancel, aborters=[factory.stop], observation_version=version)


def small_model(vec):
    return MaskablePPO("MultiInputPolicy", vec, n_steps=4, batch_size=4, n_epochs=1, seed=7,
                       device="cpu", policy_kwargs={"features_extractor_class": BattleFeatures,
                                                     "net_arch": [32]})


@requires_showdown
@pytest.mark.parametrize("learner_version,opponent_version", [("v2", "v1"), ("v1", "v2")])
def test_mixed_frozen_versions_project_predictions_and_terminal_observations(
        tmp_path, teams, learner_version, opponent_version, monkeypatch):
    torch.set_num_threads(1)
    source = real_vec(tmp_path / "source", teams, opponent_version)
    try:
        opponent = small_model(source)
        checkpoint = tmp_path / "opponent.zip"
        opponent.save(checkpoint)
    finally:
        source.close()
    opponent = MaskablePPO.load(checkpoint, device="cpu")
    before_weights, before_bytes = deepcopy(opponent.policy.state_dict()), checkpoint.read_bytes()
    predicted = []
    original_predict = opponent.predict
    def predict(obs, *args, **kwargs):
        assert set(obs) == set(policy_space(opponent_version).spaces)
        predicted.append(True)
        return original_predict(obs, *args, **kwargs)
    monkeypatch.setattr(opponent, "predict", predict)
    paired = real_vec(tmp_path / "paired", teams, "v2")
    vec = FrozenOpponentVecEnv(paired, opponent, observation_version=learner_version)
    try:
        obs = vec.reset()
        assert set(obs) == set(policy_space(learner_version).spaces)
        assert vec.get_attr("observation_version") == [learner_version]
        assert vec.get_attr("observation_space") == [policy_space(learner_version)]
        _, _, dones, infos = vec.step(np.argmax(obs["action_mask"], axis=1))
        assert dones.tolist() == [True] and infos[0]["TimeLimit.truncated"]
        assert policy_space(learner_version).contains(infos[0]["terminal_observation"])
        learner = small_model(vec)
        before_learner = deepcopy(learner.policy.state_dict())
        learner.learn(4)
        assert predicted and learner.num_timesteps == 4 and opponent.num_timesteps == 0
        assert learner.rollout_buffer.rewards.shape == (4, 1)
        assert any(not torch.equal(value, learner.policy.state_dict()[name])
                   for name, value in before_learner.items())
        assert all(torch.equal(value, opponent.policy.state_dict()[name])
                   for name, value in before_weights.items())
        assert checkpoint.read_bytes() == before_bytes and not opponent.policy.training
    finally:
        vec.close()


@requires_showdown
def test_cli_legacy_v1_resume_against_v2_frozen_opponent(tmp_path, monkeypatch, cli, teams):
    # Older checkpoints have no observation_version metadata; their spaces decide.
    checkpoints = {}
    for version in ("v1", "v2"):
        vec = real_vec(tmp_path / f"source-{version}", teams, version)
        try:
            model = small_model(vec)
            checkpoints[version] = tmp_path / f"{version}.zip"
            model.save(checkpoints[version])
        finally:
            vec.close()
    run = tmp_path / "mixed-cli"
    monkeypatch.setattr(sys, "argv", command(run, teams, "--resume", str(checkpoints["v1"]),
                                             "--opponent-checkpoint", str(checkpoints["v2"])))
    assert cli.main() == 0
    config = json.loads((run / "config.json").read_text())
    assert config["observation_version"] == "v1" and config["session_observation_version"] == "v2"
    assert config["frozen_opponent"]["observation_version"] == "v2"
    loaded = MaskablePPO.load(run / "checkpoints/final.zip", device="cpu")
    assert loaded.observation_version == observation_version(loaded.observation_space) == "v1"
    assert loaded.num_timesteps == 8
