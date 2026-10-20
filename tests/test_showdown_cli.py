"""Real Showdown -> PPO update -> checkpoint -> frozen opponent resume."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(), reason="Showdown not installed")


@pytest.mark.parametrize("worker_mode", ["thread", "process"])
def test_real_training_and_frozen_resume(tmp_path, monkeypatch, worker_mode):
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    spec = importlib.util.spec_from_file_location("showdown_training_cli", ROOT / "mgba/train_ppo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    files = [tmp_path / f"team{i}.json" for i in range(2)]
    for path, species, level, move in zip(files, ["Magikarp","Blastoise"], [1,50], ["Splash","Surf"]):
        path.write_text(json.dumps(dict(party=[dict(species=species,level=level,moves=[move])])))
    run = tmp_path / "first"
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--backend", "showdown", "--environments", "4",
        "--teams", *map(str,files), "--rollout-steps", "4", "--batch-size", "8", "--epochs", "1",
        "--total-timesteps", "16", "--checkpoint-every", "8", "--output-dir", str(run),
        "--showdown-workers", worker_mode])
    assert cli.main() == 0
    checkpoint = run / "checkpoints/latest.zip"
    original = checkpoint.read_bytes()
    model = MaskablePPO.load(checkpoint)
    assert model.num_timesteps == 16 and model.battle_backend == "showdown"
    summary = json.loads((run / "summary.json").read_text())
    assert summary["completed_battles"] == 8
    assert summary["ep_rew_mean"] is not None and summary["episodes_per_hour"] > 0
    assert (run / "pair-1/battle.log").is_file() and (run / "pair-1/battle-readable.log").is_file()
    assert not list(run.glob("pair-*/episode-*"))  # Full reports are opt-in, metrics remain available.
    assert json.loads((run / "config.json").read_text())["step_logs"] is False
    workers = [json.loads(p.read_text()) for p in run.glob("pair-*/workers.json")]
    assert len(workers) == 2 and len({w["node_pid"] for w in workers}) == 2
    assert len({w["python_pid"] for w in workers}) == (2 if worker_mode == "process" else 1)
    assert all(w["mode"] == worker_mode for w in workers)

    # An older mGBA checkpoint has no backend tag. Same model/schema must still load.
    del model.battle_backend
    legacy = tmp_path / "legacy.zip"
    model.save(legacy)
    resumed = tmp_path / "resumed"
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--backend", "showdown", "--environments", "4",
        "--team-dir", str(tmp_path), "--resume", str(legacy), "--opponent-checkpoint", str(checkpoint),
        "--opponent-stochastic", "--total-timesteps", "16", "--output-dir", str(resumed),
        "--showdown-workers", worker_mode])
    assert cli.main() == 0
    trained = MaskablePPO.load(resumed / "checkpoints/latest.zip")
    assert trained.num_timesteps == 32
    assert (trained.n_steps, trained.batch_size, trained.n_epochs) == (4, 8, 1)
    assert checkpoint.read_bytes() == original
    config = json.loads((resumed / "config.json").read_text())
    assert config["frozen_opponent"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert config["frozen_opponent"]["stochastic"] and config["learning_environments"] == 2
    assert config["teams"]["mode"] == "random_directory"
    assert json.loads((resumed / "summary.json").read_text())["learning_players"] == [1]

    overridden = tmp_path / 'overridden'
    monkeypatch.setattr(sys, 'argv', ['train_ppo', '--backend', 'showdown', '--environments', '4',
        '--teams', *map(str, files), '--resume', str(checkpoint), '--opponent-checkpoint', str(checkpoint),
        '--rollout-steps', '8', '--batch-size', '4', '--epochs', '2', '--learning-rate', '0.0001',
        '--ent-coef', '0', '--total-timesteps', '16', '--output-dir', str(overridden),
        '--showdown-workers', worker_mode])
    assert cli.main() == 0
    altered = MaskablePPO.load(overridden / 'checkpoints/latest.zip')
    assert altered.num_timesteps == 32
    assert (altered.n_steps, altered.batch_size, altered.n_epochs, altered.ent_coef) == (8, 4, 2, 0)
    assert altered.lr_schedule(1) == pytest.approx(.0001)
    assert all(g['lr'] == pytest.approx(.0001) for g in altered.policy.optimizer.param_groups)
    assert altered.policy.optimizer.state  # Adam state survives resume and remains usable.
    cfg = json.loads((overridden / 'config.json').read_text())
    assert cfg['effective_ppo']['rollout_player_timesteps'] == 16
    assert cfg['resume_overrides']['ent_coef'] == 0
    assert checkpoint.read_bytes() == original
