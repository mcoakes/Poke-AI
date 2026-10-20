"""Real Showdown/PPO continuation with validation in an isolated subprocess."""
import json
from pathlib import Path
import shutil

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO

from test_periodic_evaluation import suite_file
from test_training_continue_cli import training, fingerprint, read


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(), reason="Showdown not installed")


def test_real_periodic_validation_preserves_learning_and_survives_continue(training, tmp_path):
    t = training
    # Preserve identical inputs/model/RNG settings for a fork used as a control.
    checkpoint = tmp_path / "initial.zip"
    checkpoint.write_bytes((t.run / "checkpoints/final.zip").read_bytes())
    control = tmp_path / "control"
    suite = suite_file(tmp_path, t.teams, t.frozen)
    frozen_before = t.frozen.read_bytes()
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 8,
                    "--eval-suite", suite, "--eval-every", 4, "--eval-environments", 2,
                    "--keep-evaluations", 1, "--checkpoint-every", 0) == 0
    history = read(t.run / "evaluation-history.json")
    assert [r["timesteps"] for r in history["records"]] == [8, 12, 16]
    assert [r["reason"] for r in history["records"]] == ["initial", "interval", "interval"]
    assert all(r["requested"] == 4 for r in history["records"])
    assert all(r["status"] == "completed" for r in history["records"])
    final = MaskablePPO.load(t.run / "checkpoints/final.zip", device="cpu")
    assert final.num_timesteps == 16
    assert t.frozen.read_bytes() == frozen_before
    best = read(t.run / "checkpoints/best.json")
    original_best = (t.run / "checkpoints/best.zip").read_bytes()
    assert len(list((t.run / "evaluations").glob("evaluation-*"))) <= 2
    assert best["requested"] == 4
    for directory in (t.run / "evaluations").glob("evaluation-*"):
        info = read(directory / "config.json")
        assert info["models"]["candidate"]["snapshot_path"] == str(directory / "candidate.zip")
    # Exactly the same model, seed and training settings without validation.
    flags = [*t.new_flags, "--resume", checkpoint, "--output-dir", control,
             "--seed", 1000004, "--total-timesteps", 8, "--checkpoint-every", 0]
    assert t.invoke(*flags) == 0
    control_model = MaskablePPO.load(control / "checkpoints/final.zip", device="cpu")
    for name, value in final.policy.state_dict().items():
        assert torch.equal(value, control_model.policy.state_dict()[name]), name
    # Copied suite/weights survive deletion of the external suite and opponent.
    suite.unlink()
    t.frozen.unlink()
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 4, "--eval-every", 10) == 0
    history = read(t.run / "evaluation-history.json")
    assert [r["timesteps"] for r in history["records"]] == [8, 12, 16, 20]
    assert history["records"][-1]["reason"] == "final"
    assert (t.run / "checkpoints/best.zip").read_bytes() == original_best  # Ties keep first winner.
    assert (t.run / "checkpoints/final.zip").read_bytes() != original_best
    assert t.invoke("--if-exists", "continue", "--total-timesteps", 4, "--eval-every", 0) == 0
    assert read(t.run / "evaluation-history.json") == history
    assert (t.run / "checkpoints/best.zip").read_bytes() == original_best
    before = fingerprint(t.run)
    # Suite changes are rejected before modifying existing run files.
    alternative = tmp_path / "different-suite.json"
    altered = read(t.run / "inputs/evaluation/suite.json")
    altered["seed"] += 1
    alternative.write_text(json.dumps(altered))
    with pytest.raises(SystemExit) as error:
        t.invoke("--if-exists", "continue", "--eval-suite", alternative)
    assert error.value.code == 2 and fingerprint(t.run) == before
