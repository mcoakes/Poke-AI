"""Real simulator/vector/policy audits must never train or save checkpoints."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
import gymnasium as gym
from sb3_contrib import MaskablePPO

from firered_rl.audit_pipeline import _policy_digest, run_pipeline_checks
from firered_rl.ppo_features import BattleFeatures, policy_space
from firered_rl.ppo_vec_env import LinkedVecEnv
from firered_rl.showdown_env import ShowdownFactory


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
    reason="install pinned simulator with npm --prefix sim/showdown ci --omit=optional")


@pytest.fixture
def no_training(monkeypatch):
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("an audit must not train or write a checkpoint")

    monkeypatch.setattr(MaskablePPO, "learn", forbidden)
    monkeypatch.setattr(MaskablePPO, "save", forbidden)


def test_real_pipeline_boundaries_masks_and_logs(tmp_path, no_training, monkeypatch):
    closed = []
    close = ShowdownFactory.close

    def track_close(self):
        close(self)
        closed.append(self)

    monkeypatch.setattr(ShowdownFactory, "close", track_close)
    checks = run_pipeline_checks(ROOT, tmp_path)
    json.dumps(checks, allow_nan=False)
    assert len(checks) == 9
    assert all(row["status"] == "pass" for row in checks), checks
    details = {row["name"]: row["details"] for row in checks}
    assert details["pipeline_policy_load"]["source"] == "temporary untrained MaskablePPO"
    assert details["pipeline_policy_load"]["num_timesteps"] == 0
    assert details["pipeline_terminal_autoreset"]["outcomes"] == ["lost", "won"]
    assert details["pipeline_forced_replacement_wait"]["forced_legal_actions"] == [[5], [10]]
    bootstrap = details["pipeline_truncation_bootstrap"]
    assert bootstrap["terminal_bootstrap_mask"] == [False, False]
    assert bootstrap["truncation_bootstrap_mask"] == [True, True]
    assert bootstrap["final_observation_turns"] == [1, 1]
    assert np.isfinite(bootstrap["final_observation_values"]).all()
    cases = details["pipeline_masked_inference"]["cases"]
    assert any(case["legal_actions"][0] == [0, 5] for case in cases)
    assert all(case["invalid_probability_mass"] == 0 for case in cases)
    for case in cases:
        probabilities = np.array(case["predicted_probabilities"])
        mask = np.array(case["observation_snapshot"]["action_mask"], dtype=bool)
        assert probabilities.shape == mask.shape == (2, 11)
        assert np.allclose(probabilities.sum(axis=1), 1)
        assert (probabilities[~mask] == 0).all()
    assert len(closed) == 3
    assert all(factory.process.poll() is not None for factory in closed)
    assert all(factory.stderr_reader is not None and not factory.stderr_reader.is_alive()
               for factory in closed)
    for filename in details["pipeline_bounded_logs"]["files"]:
        assert 0 < (tmp_path / filename).stat().st_size <= 2 * 1024 * 1024
    assert "used Surf!" in (tmp_path / "pipeline/terminal/battle.log").read_text()
    assert not list(tmp_path.rglob("*.zip"))
    assert not list(tmp_path.glob("pipeline/*/episode-*"))


def test_checkpoint_loaded_once_and_inference_is_read_only(tmp_path, monkeypatch):
    torch.set_num_threads(1)

    class ContractEnv(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)

    original = MaskablePPO("MultiInputPolicy", ContractEnv(), device="cpu", seed=7,
        n_steps=2, batch_size=2,
        policy_kwargs=dict(features_extractor_class=BattleFeatures, net_arch=[32]))
    original.num_timesteps = 123
    checkpoint = tmp_path / "existing.zip"
    original.save(checkpoint)
    before = checkpoint.read_bytes()
    modified = checkpoint.stat().st_mtime_ns
    expected_digest = _policy_digest(original.policy)
    load = MaskablePPO.load.__func__
    loaded = []

    def tracked_load(cls, *args, **kwargs):
        model = load(cls, *args, **kwargs)
        loaded.append(model)
        return model

    def forbidden(*args, **kwargs):
        raise AssertionError("training and checkpoint writing are forbidden")

    monkeypatch.setattr(MaskablePPO, "load", classmethod(tracked_load))
    monkeypatch.setattr(MaskablePPO, "learn", forbidden)
    monkeypatch.setattr(MaskablePPO, "save", forbidden)
    checks = run_pipeline_checks(ROOT, tmp_path / "audit", checkpoint=checkpoint)
    assert all(row["status"] == "pass" for row in checks), checks
    assert len(loaded) == 1
    assert loaded[0].num_timesteps == 123
    assert _policy_digest(loaded[0].policy) == expected_digest
    assert checkpoint.read_bytes() == before
    assert checkpoint.stat().st_mtime_ns == modified
    details = {row["name"]: row["details"] for row in checks}
    assert details["pipeline_policy_immutable"]["checkpoint_bytes_unchanged"] is True
    assert details["pipeline_policy_load"]["trained_checkpoint"] is True
    assert details["pipeline_policy_load"]["checkpoint_sha256"] == hashlib.sha256(before).hexdigest()
    assert not list((tmp_path / "audit").rglob("*.zip"))


def test_bad_checkpoint_reports_failure_without_suppressing_real_boundaries(tmp_path, no_training):
    checkpoint = tmp_path / "bad.zip"
    checkpoint.write_bytes(b"not a model")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checks = run_pipeline_checks(ROOT, tmp_path / "audit", checkpoint=checkpoint)
    status = {row["name"]: row["status"] for row in checks}
    assert status["pipeline_policy_load"] == "fail"
    assert status["pipeline_masked_inference"] == "fail"
    assert status["pipeline_policy_immutable"] == "fail"
    assert status["pipeline_terminal_autoreset"] == "pass"
    assert status["pipeline_forced_replacement_wait"] == "pass"
    assert status["pipeline_live_truncation"] == "pass"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest


def test_audit_detects_autoreset_overwriting_terminal_observation(tmp_path, no_training, monkeypatch):
    step_wait = LinkedVecEnv.step_wait

    def replace_final_with_reset(self):
        observations, rewards, dones, infos = step_wait(self)
        for index, info in enumerate(infos):
            if info.get("outcome") in ("won", "lost"):
                info["terminal_observation"] = {key: value[index].copy()
                                                for key, value in observations.items()}
        return observations, rewards, dones, infos

    monkeypatch.setattr(LinkedVecEnv, "step_wait", replace_final_with_reset)
    checks = run_pipeline_checks(ROOT, tmp_path)
    by_name = {row["name"]: row for row in checks}
    assert by_name["pipeline_terminal_autoreset"]["status"] == "fail"
    assert "pre-reset state" in by_name["pipeline_terminal_autoreset"]["details"]["error"]
    assert by_name["pipeline_live_truncation"]["status"] == "pass"
    assert by_name["pipeline_forced_replacement_wait"]["status"] == "pass"


@pytest.mark.parametrize("options", [dict(log_max_mb=0), dict(log_max_mb=-1),
    dict(log_max_mb=float("inf")), dict(log_max_mb=float("nan")), dict(log_backups=-1)])
def test_invalid_log_bounds_rejected_before_output_creation(tmp_path, options):
    with pytest.raises(ValueError, match="log_"):
        run_pipeline_checks(ROOT, tmp_path, **options)
    assert not list(tmp_path.iterdir())
