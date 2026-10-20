"""V2 audits use native oracles and read-only real-vector policy inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
import gymnasium as gym
from sb3_contrib import MaskablePPO

from firered_rl import audit_v2
from firered_rl.audit_cases import mon
from firered_rl.audit_pipeline import _policy_digest
from firered_rl.ppo_features import BattleFeatures, policy_space

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
    reason="install the pinned local Showdown simulator")


def cli_module():
    spec = importlib.util.spec_from_file_location("audit_training_v2_cli", ROOT / "mgba/audit_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def auditor(directory):
    return audit_v2.NativeV2Audit(ROOT, directory, node="node", timeout=10, seed=26002,
                                 log_max_mb=.05, log_backups=1)


def test_real_native_v2_fields_and_isolated_feature_channels(tmp_path):
    rows = audit_v2.run_v2_battle_checks(ROOT, tmp_path, log_max_mb=.05)
    assert len(rows) == 8
    assert all(row["status"] == "pass" for row in rows), rows
    by_name = {row["name"]: row["details"] for row in rows}
    assert by_name["v2_signed_boosts"]["positive_and_negative"] is True
    assert by_name["v2_weather_duration_and_expiry"]["expiry"] is True
    assert by_name["v2_damaged_status_bench"]["status"] == "tox"
    feature = by_name["v2_native_metadata_and_feature_sensitivity"]
    assert feature["high_and_low_bits_preserved"] is True
    assert feature["hidden_power_base_and_active"] is True
    assert {probe["field"] for probe in feature["probes"]} == {
        "v2_party", "v2_stats", "v2_active_stats", "v2_active", "v2_boosts",
        "v2_move_details", "v2_weather", "v2_volatiles"}
    for filename in ("audit-diagnostics.jsonl", "battle.log", "showdown-stderr.log"):
        paths = list(tmp_path.glob(filename+"*"))
        assert 1 <= len(paths) <= 2
        assert all(path.stat().st_size <= int(.05*1024*1024) for path in paths)


@pytest.mark.parametrize("field", ["v2_party", "v2_stats", "v2_active_stats", "v2_active",
    "v2_boosts", "v2_move_details", "v2_weather", "v2_volatiles"])
def test_independent_native_oracle_detects_zeroed_new_field(tmp_path, monkeypatch, field):
    audit = auditor(tmp_path)
    try:
        audit.reset([mon("Mew", ["Rain Dance", "Swords Dance", "Substitute"])],
                    [mon("Lapras", ["Splash"])])
        audit.step([0, 0])
        audit.step([1, 0])
        audit.step([2, 0])
        assert audit.observations[0][field].any()
        encoder = audit_v2.showdown_observation
        def dropped(report, terminal=False, version="v1"):
            result = encoder(report, terminal, version=version)
            result[field][:] = 0
            return result
        monkeypatch.setattr(audit_v2, "showdown_observation", dropped)
        with pytest.raises(AssertionError, match="expected"):
            audit.capture()
    finally:
        audit.close()


def test_native_oracle_detects_shape_valid_wrong_move_type(tmp_path, monkeypatch):
    audit = auditor(tmp_path)
    try:
        audit.reset([mon("Mew", ["Hidden Power"], ivs={"atk": 30, "def": 30})],
                    [mon("Lapras", ["Splash"])])
        encoder = audit_v2.showdown_observation
        def wrong_type(report, terminal=False, version="v1"):
            result = encoder(report, terminal, version=version)
            result["v2_move_details"][0, 0, 0, 5] = 0
            return result
        monkeypatch.setattr(audit_v2, "showdown_observation", wrong_type)
        with pytest.raises(AssertionError, match="native move metadata"):
            audit.capture()
    finally:
        audit.close()


def test_feature_probe_detects_dropped_boost_channel(tmp_path, monkeypatch):
    audit = auditor(tmp_path)
    try:
        audit.reset([mon("Mew", ["Splash"]), mon("Lapras", ["Surf"])],
                    [mon("Blastoise", ["Splash"]), mon("Lapras", ["Ice Beam"])])
        flatten = BattleFeatures._flatten_v2
        def dropped(self, obs):
            return flatten(self, {**obs, "v2_boosts": torch.zeros_like(obs["v2_boosts"])})
        monkeypatch.setattr(BattleFeatures, "_flatten_v2", dropped)
        with pytest.raises(AssertionError, match="v2_boosts.*does not reach"):
            audit_v2._feature_sensitivity(audit.observations[0])
    finally:
        audit.close()


def make_checkpoint(path, observation_version):
    class ContractEnv(gym.Env):
        observation_space = policy_space(observation_version)
        action_space = gym.spaces.Discrete(11)
    model = MaskablePPO("MultiInputPolicy", ContractEnv(), device="cpu", seed=7,
        n_steps=2, batch_size=2,
        policy_kwargs=dict(features_extractor_class=BattleFeatures, net_arch=[32]))
    model.num_timesteps = 123
    model.save(path)
    return model


def test_v2_checkpoint_autodetection_scope_and_read_only_inference(tmp_path, monkeypatch):
    path = tmp_path / "v2.zip"
    model = make_checkpoint(path, "v2")
    original = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    expected_digest = _policy_digest(model.policy)
    load = MaskablePPO.load.__func__
    loaded = []
    def track_load(cls, *args, **kwargs):
        result = load(cls, *args, **kwargs)
        loaded.append(result)
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError("audit must not train or save model checkpoints")
    monkeypatch.setattr(MaskablePPO, "load", classmethod(track_load))
    monkeypatch.setattr(MaskablePPO, "learn", forbidden)
    monkeypatch.setattr(MaskablePPO, "save", forbidden)
    output = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit_training", "--output-dir", str(output),
                                      "--checkpoint", str(path), "--log-max-mb", ".05"])
    assert cli_module().main() == 0
    report = json.loads((output / "audit-report.json").read_text())
    assert report["config"]["observation_version"] == "v2"
    assert report["counts"] == {"pass": 15}
    assert report["regression_counts"] == {"pass": 51, "limitation": 13}
    assert report["checkpoint"]["observation_version"] == "v2"
    assert report["checkpoint"]["training_timesteps"] == 123
    assert report["checkpoint"]["loaded_sha256"] == hashlib.sha256(original).hexdigest()
    assert all(row["scope"] == "v1_regression" for row in report["checks"] if row["status"] == "limitation")
    assert len(loaded) == 1 and _policy_digest(loaded[0].policy) == expected_digest
    assert path.read_bytes() == original and path.stat().st_mtime_ns == mtime
    assert not list(output.rglob("*.zip"))
    inference = next(row for row in report["checks"] if row["name"] == "v2_pipeline_masked_inference")
    for case in inference["details"]["cases"]:
        probs = np.asarray(case["predicted_probabilities"])
        assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1)
        assert case["invalid_probability_mass"] == 0
    markdown = (output / "audit-report.md").read_text()
    assert "not reported as current v2 gaps" in markdown
    assert "Actual checkpoint schema: `v2`" in markdown


@pytest.mark.parametrize("checkpoint_version,requested", [("v1", "v2"), ("v2", "v1")])
def test_explicit_checkpoint_schema_mismatch_fails_before_output(tmp_path, monkeypatch, capsys,
                                                               checkpoint_version, requested):
    checkpoint = tmp_path / "checkpoint.zip"
    make_checkpoint(checkpoint, checkpoint_version)
    output = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit_training", "--output-dir", str(output),
        "--checkpoint", str(checkpoint), "--observation-version", requested])
    assert cli_module().main() == 2
    assert not output.exists()
    assert "conflicts with checkpoint schema" in capsys.readouterr().err


def test_v2_vector_detects_lost_final_fields(tmp_path, monkeypatch):
    from firered_rl.ppo_vec_env import LinkedVecEnv
    step_wait = LinkedVecEnv.step_wait
    def dropped_final(self):
        observations, rewards, dones, infos = step_wait(self)
        for info in infos:
            if info.get("TimeLimit.truncated"):
                info["terminal_observation"]["v2_weather"][:] = 0
        return observations, rewards, dones, infos
    monkeypatch.setattr(LinkedVecEnv, "step_wait", dropped_final)
    rows = audit_v2.run_v2_pipeline_checks(ROOT, tmp_path)
    by_name = {row["name"]: row for row in rows}
    assert by_name["v2_pipeline_terminal_autoreset"]["status"] == "pass"
    assert by_name["v2_pipeline_truncation_bootstrap"]["status"] == "fail"
    assert "lost pre-reset v2 fields" in by_name["v2_pipeline_truncation_bootstrap"]["details"]["error"]
