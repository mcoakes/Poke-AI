"""Weighted rehearsal stays episode-scoped, reproducible, and pinned across resume."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from firered_rl.training_states import add_state_arguments
from firered_rl.training_teams import TrainingTeams


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mixture(tmp_path):
    for group, species in (("current", "Pikachu"), ("previous", "Charmander"), ("mirror", "Squirtle")):
        directory = tmp_path / group
        directory.mkdir()
        for index in range(3):
            (directory / f"{index}.json").write_text(json.dumps({"party": [
                {"species": species, "level": 50, "moves": ["Tackle"]}]}))
    path = tmp_path / "mixture.json"
    path.write_text(json.dumps({"version": 1, "groups": {
        "current": {"weight": 8, "directory": "current"},
        "previous": {"weight": 1, "directory": "previous"},
        "mirror": {"weight": 1, "teams": ["mirror/1.json", "mirror/0.json"]}}}))
    return path


def test_episode_scoped_weighting_reproducibility_and_fixed_roles(mixture):
    source = TrainingTeams(mixture=mixture, seed=191)
    first, second = source.for_pair(0), TrainingTeams(mixture=mixture, seed=191).for_pair(0)
    draws = [first() for _ in range(10000)]
    assert draws == [second() for _ in range(10000)]
    other_pair = TrainingTeams(mixture=mixture, seed=191).for_pair(1)
    assert draws[:20] != [other_pair() for _ in range(20)]
    counts = Counter()
    for pair in draws:
        assert pair[0].parent == pair[1].parent
        name = pair[0].parent.name
        counts[name] += 1
        if name == "mirror":
            assert [file.name for file in pair] == ["1.json", "0.json"]
        else:
            assert pair[0] != pair[1]
    assert 7800 < counts["current"] < 8200
    assert 850 < counts["previous"] < 1150
    assert 850 < counts["mirror"] < 1150
    assert source.sampling_summary()["episode_selections"] == dict(counts)


def test_snapshot_preserves_groups_and_detects_frozen_additions(mixture, tmp_path):
    source = TrainingTeams(mixture=mixture)
    frozen = source.snapshot(tmp_path / "snapshot")
    assert frozen.weights == source.weights
    original_draw, frozen_draw = source.for_pair(0), frozen.for_pair(0)
    for _ in range(100):
        assert [file.read_bytes() for file in original_draw()] == [file.read_bytes() for file in frozen_draw()]
    descriptor = frozen.describe()
    assert source.snapshot(tmp_path / "snapshot", expected=descriptor).describe() == descriptor
    group = tmp_path / "snapshot/group-000"
    shutil.copyfile(group / "team-00000.json", group / "added.json")
    with pytest.raises(ValueError, match="pinned team mixture inputs changed"):
        source.snapshot(tmp_path / "snapshot", expected=descriptor)


@pytest.mark.parametrize("change", ["weight", "content", "addition", "deletion"])
def test_original_descriptor_tracks_all_mutations(mixture, change):
    before = TrainingTeams(mixture=mixture).describe()
    path = mixture.parent / "current/0.json"
    if change == "weight":
        data = json.loads(mixture.read_text())
        data["groups"]["current"]["weight"] = 7
        mixture.write_text(json.dumps(data))
    elif change == "content":
        data = json.loads(path.read_text())
        data["party"][0]["level"] = 51
        path.write_text(json.dumps(data))
    elif change == "addition":
        shutil.copyfile(path, path.with_name("added.json"))
    else:
        path.unlink()
    assert TrainingTeams(mixture=mixture).describe() != before


@pytest.mark.parametrize("weight", [0, -1, True, float("nan"), float("inf"), "1"])
def test_bad_weights_rejected(mixture, weight):
    data = json.loads(mixture.read_text())
    data["groups"]["current"]["weight"] = weight
    mixture.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="weight must be positive and finite"):
        TrainingTeams(mixture=mixture)


def test_mixture_only_opt_in_and_mutually_exclusive(mixture):
    parser = argparse.ArgumentParser()
    add_state_arguments(parser, include_teams=True)
    with pytest.raises(SystemExit):
        parser.parse_args(["--team-mixture", str(mixture)])
    parser = argparse.ArgumentParser()
    add_state_arguments(parser, include_teams=True, include_mixture=True)
    args = parser.parse_args(["--team-mixture", str(mixture)])
    assert args.team_mixture == mixture
    with pytest.raises(SystemExit):
        parser.parse_args(["--team-mixture", str(mixture), "--team-dir", str(mixture.parent)])
    with pytest.raises(ValueError, match="cannot be combined"):
        TrainingTeams(mixture=mixture, mirror_teams=True)


@pytest.mark.skipif(not shutil.which("node") or not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
                    reason="Showdown is not installed")
def test_real_ppo_mixture_continue_and_fresh_branch(mixture, tmp_path, monkeypatch):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    spec = importlib.util.spec_from_file_location("mixture_training_cli", ROOT / "mgba/train_ppo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    run = tmp_path / "run"

    def invoke(directory, *flags):
        monkeypatch.setattr(sys, "argv", ["train_ppo", "--output-dir", str(directory), *map(str, flags)])
        return cli.main()

    flags = ["--backend", "showdown", "--policy-architecture", "shared-moves", "--showdown-workers", "process",
             "--team-mixture", mixture, "--rule-based-opponent", "--environments", 2,
             "--rollout-steps", 4, "--batch-size", 4, "--epochs", 1, "--max-steps", 2,
             "--total-timesteps", 8, "--no-readable-log", "--no-battle-log", "--no-step-logs"]
    assert invoke(run, *flags) == 0
    first = json.loads((run / "config.json").read_text())
    assert first["teams"]["mode"] == "team_mixture"
    assert first["team_mixture_snapshot"]["manifest"]["path"] == str(run / "inputs/team-mixture/mixture.json")
    assert invoke(run, "--if-exists", "continue", "--total-timesteps", 8) == 0
    assert MaskablePPO.load(run / "checkpoints/final.zip").num_timesteps == 16
    continued = json.loads((run / "config.json").read_text())
    assert continued["teams"] == first["teams"]
    assert continued["team_mixture_snapshot"] == first["team_mixture_snapshot"]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["team_mixture_sampling"]["total"] > 0
    assert summary["team_mixture_sampling"]["weights"] == {"current": .8, "previous": .1, "mirror": .1}

    # An explicit branch can use a new mixture without modifying the original run.
    altered = tmp_path / "branch-mixture.json"
    data = json.loads(mixture.read_text())
    data["groups"]["current"]["weight"] = 2
    altered.write_text(json.dumps(data))
    branch_flags = [altered if item == mixture else item for item in flags]
    assert invoke(tmp_path / "branch", *branch_flags, "--resume", run / "checkpoints/final.zip") == 0
    assert MaskablePPO.load(tmp_path / "branch/checkpoints/final.zip").num_timesteps == 24

    # Native continuation rejects external additions before changing config/checkpoint.
    previous_config = (run / "config.json").read_bytes()
    previous_checkpoint = (run / "checkpoints/final.zip").read_bytes()
    added = mixture.parent / "current/added.json"
    shutil.copyfile(mixture.parent / "current/0.json", added)
    with pytest.raises(SystemExit) as error:
        invoke(run, "--if-exists", "continue", "--total-timesteps", 8)
    assert error.value.code == 2
    assert (run / "config.json").read_bytes() == previous_config
    assert (run / "checkpoints/final.zip").read_bytes() == previous_checkpoint
    added.unlink()
    frozen = run / "inputs/team-mixture/group-000/team-00000.json"
    frozen.write_text(frozen.read_text() + "\n")
    with pytest.raises(SystemExit) as error:
        invoke(run, "--if-exists", "continue", "--total-timesteps", 8)
    assert error.value.code == 2
    assert (run / "config.json").read_bytes() == previous_config
