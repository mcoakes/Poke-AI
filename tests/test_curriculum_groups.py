import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from firered_rl.curriculum import load_curriculum, party_signature, read_json, stage_suites, write_json
from firered_rl.evaluation import digest, summary

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location("group_campaign", ROOT / "mgba/run_curriculum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def grouped(tmp_path):
    template = read_json(ROOT / "config/curricula/type_progression/teams/01_six_types.json")
    counter = 0
    for folder, count in (("one/train", 2), ("two/train/nested", 3), ("one/validation", 2),
                          ("two/validation", 3), ("probe", 2)):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        for i in range(count):
            counter += 1
            team = deepcopy(template)
            team["party"][0]["level"] = 20+counter
            write_json(directory / f"team_{i}.json", team)
    manifest = dict(version=1, name="group-test", next_team_dir="probe", stages=[
        dict(stage=1, team_dir="one/train", validation_dir="one/validation", focus="first"),
        dict(stage=2, team_dir="two/train", validation_dir="two/validation", focus="second")])
    path = tmp_path / "curriculum.json"
    write_json(path, manifest)
    return path


def test_groups_are_recursive_sorted_and_validation_is_separate(grouped):
    _, groups, validation = load_curriculum(grouped, grouped=True)
    assert [len(g) for g in groups] == [2, 3, 2]
    assert [len(g) for g in validation] == [2, 3, 2]
    assert len(load_curriculum(grouped)[1]) == 7
    assert groups[1] == sorted(groups[1])
    train = {party_signature(p) for g in groups[:-1] for p in g}
    assert not train.intersection(party_signature(p) for g in validation for p in g)
    current, expanded = stage_suites(validation, stage=1, episodes=40, seed=7, max_steps=200)
    incoming = {digest(dict(team=read_json(p))) for p in validation[1]}
    assert len(current["assets"]) == 2
    assert not incoming.intersection(current["assets"])
    assert expanded["cases"][:20] == current["cases"][:20]
    for a, b in zip(expanded["cases"][20::2], expanded["cases"][21::2]):
        assert incoming.intersection(a["assets"])
        assert a["assets"] == b["assets"] and a["seed"] == b["seed"]
        assert (a["candidate_player"], b["candidate_player"]) == (1, 2)


@pytest.mark.parametrize("problem", ["empty", "ambiguous", "invalid", "overlap", "partial_validation", "duplicate_party"])
def test_bad_groups_fail_before_training(grouped, problem):
    value = read_json(grouped)
    if problem == "empty":
        (grouped.parent / "empty").mkdir()
        value["stages"][0]["team_dir"] = "empty"
    elif problem == "ambiguous":
        value["stages"][0]["team"] = "one/train/team_0.json"
    elif problem == "invalid":
        write_json(grouped.parent / "one/train/broken.json", dict(party=[]))
    elif problem == "overlap":
        value["stages"][1]["team_dir"] = "one/train"
    elif problem == "partial_validation":
        value["stages"][1].pop("validation_dir")
    elif problem == "duplicate_party":
        team = read_json(grouped.parent / "one/train/team_0.json")
        team["name"] = "different name"
        team["party"].reverse()
        for mon in team["party"]:
            mon["moves"].reverse()
        write_json(grouped.parent / "two/validation/copy.json", team)
    write_json(grouped, value)
    with pytest.raises(ValueError):
        load_curriculum(grouped, grouped=True)


@pytest.mark.parametrize("mode", ["original", "battle-budget", "mixed"])
def test_grouped_campaign_snapshots_and_training_evaluation_flow(grouped, tmp_path, mode):
    cli = runner()
    start, opponent = tmp_path / "start.zip", tmp_path / "opponent.zip"
    start.write_bytes(b"candidate")
    opponent.write_bytes(b"opponent")
    run = tmp_path / "run"
    run.mkdir()
    config = cli.initialize(argparse.Namespace(resume=start, opponent_checkpoint=opponent, curriculum=grouped,
        eval_episodes=40, max_steps=200, promotion_rules=None, reward_config=None, seed=9,
        train_battles=None if mode == "original" else 2), run)
    if mode == "mixed":
        config["evaluation"] = dict(mode="mixed")
    train_names = {read_json(run / p)["party"][0]["level"] for g in config["team_groups"][:-1] for p in g}
    training_pools, evaluations = [], []

    def execute(command, root):
        values = command[2:]
        get = lambda name: Path(values[values.index(name)+1])
        output = get("--output-dir")
        if Path(command[1]).name == "train_ppo.py":
            assert "--opponent-switch-chance" not in values
            teams = sorted(get("--team-dir").glob("*.json"))
            training_pools.append(teams)
            assert all(read_json(p)["party"][0]["level"] in train_names for p in teams)
            (output / "checkpoints").mkdir(parents=True)
            (output / "checkpoints/final.zip").write_bytes(get("--resume").read_bytes()+b"updated")
            write_json(output / "summary.json", dict(status="completed"))
        else:
            suite = read_json(get("--suite"))
            assert all(a["team"]["party"][0]["level"] not in train_names for a in suite["assets"].values())
            evaluations.append(suite)
            output.mkdir(parents=True)
            write_json(output / "config.json", dict(suite_sha256=digest(suite), models=dict(
                candidate=dict(sha256=cli.sha256(get("--checkpoint"))),
                opponent=dict(sha256=cli.sha256(opponent)))))
            rows = [dict(c, candidate_outcome="won", truncated=False, candidate_reward=1,
                         opponent_reward=-1, decisions=1) for c in suite["cases"]]
            (output / "episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            write_json(output / "summary.json", summary(rows, requested=len(rows), elapsed=1,
                status="completed", suite_hash=digest(suite)))

    # Changing source directories after initialization cannot change this run.
    for p in grouped.parent.glob("one/*/*.json"):
        p.unlink()
    campaign = cli.Campaign(run, config, cli.EXECUTION, execute=execute)
    first = campaign.run_stage(1)
    assert first["status"] == "promote" and first["held_out_validation"]
    assert first["training_teams"] == first["evaluation_teams"] == 2
    second = campaign.run_stage(2)
    assert second["status"] == "promote"
    assert [len(p) for p in training_pools] == [2, 5]
    assert set(training_pools[0][i].name for i in range(2)) <= {p.name for p in training_pools[1]}
    assert campaign.next_stage() == 3
    cli.Campaign(run, config, cli.EXECUTION).prepare_stage(2)  # stable replay
    (run / config["validation_groups"][0][0]).write_text("{}")
    with pytest.raises(ValueError, match="saved curriculum input changed"):
        cli.Campaign(run, config, cli.EXECUTION)


def test_shipped_generalization_campaign_uses_valid_diverse_nonoverlapping_teams():
    from collections import Counter
    folder = ROOT / "config/curricula/pokemon_generalization"
    manifest, groups, validation = load_curriculum(folder / "curriculum.json", grouped=True)
    assert manifest["opponent_switch"]["chance"] == 0
    assert [len(g) for g in groups] == [24]*10+[40]
    assert [len(g) for g in validation] == [12]*10+[40]
    catalog = read_json(ROOT / "config/generation/firered_us_v1_0.json")
    types, species = set(), set()
    for paths in groups[:-1]+validation:
        for path in paths:
            team = read_json(path)
            for mon in team["party"]:
                info = catalog["species"][mon["species"]]
                assert len(mon["moves"]) == 4 and set(mon["moves"]) <= set(info["moves"])
                assert mon["held_item"] in catalog["items"]
                assert sum(mon["evs"].values()) <= 510
                types.update(info["types"])
                species.add(mon["species"])
    assert len(types) == 17 and len(species) > 100
    sizes = Counter(len(read_json(p)["party"]) for p in groups[9])
    assert sizes == dict.fromkeys(range(1, 7), 4)
    assert not set(read_json(folder / "generation.json")["reserved_species"]).intersection(
        mon["species"] for g in groups[:-1] for p in g for mon in read_json(p)["party"])


def test_real_showdown_group_campaign_round_trip(tmp_path, monkeypatch):
    import shutil
    import sys
    import gymnasium as gym
    import torch
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import BattleFeatures, policy_space
    if not shutil.which("node") or not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file():
        pytest.skip("Showdown not installed")
    torch.set_num_threads(1)

    class ShapeEnv(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)

    model = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2, n_epochs=1,
        seed=31, policy_kwargs=dict(features_extractor_class=BattleFeatures))
    checkpoint = tmp_path / "start.zip"
    model.save(checkpoint)
    before = checkpoint.read_bytes()
    run = tmp_path / "real-campaign"
    monkeypatch.setattr(sys, "argv", ["run_curriculum", "--run-dir", str(run),
        "--curriculum", str(ROOT / "config/curricula/pokemon_generalization/curriculum.json"),
        "--resume", str(checkpoint), "--opponent-checkpoint", str(checkpoint),
        "--train-battles", "2", "--eval-episodes", "20", "--eval-opponents", "mixed",
        "--max-steps", "4", "--stage", "1", "--environments", "2", "--showdown-workers", "process",
        "--rollout-steps", "2", "--batch-size", "2", "--epochs", "1", "--no-readable-log", "--no-battle-log"])
    assert runner().main() in (0, 2)
    attempt = run / "stage-01/attempt-001"
    assert read_json(attempt / "train/summary.json")["status"] == "completed"
    assert (attempt / "train/checkpoints/final.zip").is_file()
    report = read_json(attempt / "report.json")
    assert report["held_out_validation"] and report["training_teams"] == 24 and report["evaluation_teams"] == 12
    suite = read_json(report["evaluation_plan"])
    assert len(suite["cases"]) == 20
    assert all("validation" in a["team"]["name"] for a in suite["assets"].values())
    assert read_json(attempt / "train/switching.json")["heuristic_overrides"] == 0
    assert checkpoint.read_bytes() == before
