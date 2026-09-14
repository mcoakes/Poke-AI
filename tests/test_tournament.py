import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from firered_rl.curriculum import read_json, write_json
from firered_rl.evaluation import digest, summary
from firered_rl.tournament import (completed_matches, leaderboard, make_plan, rates,
    read_config, save_plan, verify_plan)


ROOT = Path(__file__).resolve().parents[1]


def cli_module():
    spec = importlib.util.spec_from_file_location("models_cli", ROOT / "mgba/evaluate_models.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def inputs(tmp_path):
    models = tmp_path / "models"; models.mkdir()
    (models / "smart_v1.zip").write_bytes(b"smart one")
    (models / "smart_v2.zip").write_bytes(b"smart two")
    (models / "same_as_v1.zip").write_bytes(b"smart one")
    teams = tmp_path / "teams"; teams.mkdir()
    for i, species in enumerate(("Raticate", "Linoone")):
        write_json(teams / f"{i}.json", dict(party=[dict(species=species, level=1,
            moves=["Double Edge", "Body Slam", "Strength", "Facade"])]))
    config = tmp_path / "config.json"
    write_json(config, dict(version=1, episodes=4, max_steps=8, scenarios=[
        dict(id="mirrors", team_dir="teams", mirror_teams=True),
        dict(id="fixed", teams=["teams/0.json", "teams/1.json"])]))
    return config, models


def test_round_robin_suites_models_and_snapshots_are_fixed(tmp_path):
    config, models = inputs(tmp_path)
    plan, suites = make_plan(ROOT, config, models)
    assert len(plan["models"]) == 3  # duplicate checkpoint collapsed, Random included
    assert len(plan["matches"]) == 6
    assert all(match["a"] != "random" for match in plan["matches"])
    assert len({m["id"] for m in plan["matches"]}) == 6
    assert sum(bool(m.get("aliases")) for m in plan["models"]) == 1
    for suite in suites.values():
        assert len(suite["cases"]) == 4
        for a,b in zip(suite["cases"][::2], suite["cases"][1::2]):
            assert a["seed"] == b["seed"] and a["assets"] == b["assets"]
            assert (a["candidate_player"], b["candidate_player"]) == (1,2)
    assert all(c["assets"][0] == c["assets"][1] for c in suites["mirrors"]["cases"])
    run = tmp_path / "run"; run.mkdir()
    save_plan(run, plan, suites, validate_model=lambda p:None)
    for path in models.glob("*.zip"):
        path.unlink()
    config.unlink()
    verify_plan(run, read_json(run / "tournament.json"))
    assert len(list(run.rglob("*.zip"))) == 2
    suite_path = run / plan["scenarios"][0]["path"]
    changed = read_json(suite_path); changed["max_steps"] += 1
    write_json(suite_path, changed)
    with pytest.raises(ValueError, match="suite changed"):
        verify_plan(run, plan)


@pytest.mark.parametrize("change", [dict(episodes=3), dict(include_random=1),
    dict(scenarios=[]), dict(scenarios=[dict(id="overall", training_profile="six_types")]),
    dict(scenarios=[dict(id="bad", training_profile="six_types", team_size=3)]),
    dict(scenarios=[dict(id="bad", teams=[None, 1])])])
def test_invalid_plans_fail_before_running(tmp_path, change):
    config, _ = inputs(tmp_path)
    write_json(config, {**read_json(config), **change})
    with pytest.raises(ValueError):
        read_config(config)


def test_rank_uses_draws_and_truncations_and_reverses_opponents():
    match = dict(a="a", b="b", games=4, wins_a=2, wins_b=0, draws=1, truncated=1)
    assert rates([match], "a")["score"] == .625
    assert rates([match], "b")["score"] == .125
    assert rates([match], "b")["losses"] == 2
    models = [dict(id="a", label="A"), dict(id="b", label="B"), dict(id="random", label="Random")]
    assert leaderboard([match], models)[0]["model"] == "a"
    assert leaderboard([match], models)[-1]["rank"] is None
    match.update(wins_a=0, draws=4, truncated=0)
    assert [r["rank"] for r in leaderboard([match], models)] == [1, 1, None]


def test_checkpoint_names_cannot_collide_with_random_agent_label(tmp_path):
    config, models = inputs(tmp_path)
    (models / "Random.zip").write_bytes(b"learned random baseline")
    (models / "Random [random].zip").write_bytes(b"another baseline")
    plan, _ = make_plan(ROOT, config, models)
    labels = [m["label"] for m in plan["models"]]
    assert len(set(labels)) == len(labels) == 5


def test_interruption_reuses_finished_pairs_without_duplicate_results(tmp_path):
    cli = cli_module()
    config, models = inputs(tmp_path)
    plan, suites = make_plan(ROOT, config, models)
    run = tmp_path / "run"; run.mkdir()
    save_plan(run, plan, suites, validate_model=lambda p:None)
    calls = []

    def execute(command, log, output):
        calls.append(output)
        output.mkdir(parents=True)
        if len(calls) == 2:
            (output / "summary.json").write_text('{"status":')
            raise KeyboardInterrupt
        get = lambda flag:Path(command[command.index(flag)+1])
        suite = read_json(get("--suite"))
        assert "--no-step-logs" in command
        model_info = dict(candidate=dict(sha256=cli_module_hash(get("--checkpoint"))))
        model_info["opponent"] = (dict(kind="random") if "--random-opponent" in command
                                  else dict(sha256=cli_module_hash(get("--opponent-checkpoint"))))
        write_json(output / "config.json", dict(models=model_info, suite_sha256=digest(suite), deterministic=True))
        rows = [dict(case, candidate_outcome=outcome, truncated=outcome=="truncated", terminated=outcome!="truncated",
            candidate_reward=0, opponent_reward=0, decisions=1)
            for case,outcome in zip(suite["cases"], ("won", "lost", "draw", "truncated"))]
        (output / "episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        write_json(output / "summary.json", summary(rows, requested=4, elapsed=1, status="completed", suite_hash=digest(suite)))

    with pytest.raises(KeyboardInterrupt):
        cli.run_tournament(run, plan, cli.EXECUTION, execute=execute)
    assert read_json(run / "summary.json")["status"] == "interrupted"
    assert read_json(run / "summary.json")["completed_matches"] == 1
    report = cli.run_tournament(run, plan, cli.EXECUTION, execute=execute)
    assert len(calls) == 7 and report["completed_matches"] == 6
    assert report["recorded_battles"] == 24 and report["status"] == "completed"
    lines = (run / "matches.jsonl").read_text().splitlines()
    assert len(lines) == 6 and len({json.loads(line)["id"] for line in lines}) == 6
    cli.run_tournament(run, plan, cli.EXECUTION, execute=execute)
    assert len(calls) == 7
    assert (run / "matches.jsonl").read_text().splitlines() == lines
    assert "Random" in (run / "matchups.md").read_text()
    # No summary-only shortcut may turn a missing battle into a ranking result.
    path = calls[0] / "episodes.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1])+"\n")
    with pytest.raises(ValueError, match="missing cases"):
        completed_matches(run, plan)


def cli_module_hash(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dry_run_is_read_only_and_empty_run_dir_is_rejected(tmp_path, monkeypatch):
    cli = cli_module()
    config, models = inputs(tmp_path)
    run = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["evaluate_models", "--run-dir", str(run), "--config", str(config),
                                     "--models-dir", str(models), "--dry-run"])
    assert cli.main() == 0 and not run.exists()
    with pytest.raises(Exception, match="path is empty"):
        cli.path_argument("")


def test_real_round_robin_and_resume_use_copied_models(tmp_path):
    pytest.importorskip("sb3_contrib")
    if not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").exists():
        pytest.skip("Showdown not installed")
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import BattleFeatures, policy_space
    import torch
    torch.set_num_threads(1)
    config, models = inputs(tmp_path)
    for path in models.glob("*.zip"):
        path.unlink()
    value = read_json(config); value["scenarios"] = value["scenarios"][:1]
    write_json(config, value)

    class ShapeEnv(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)

    for seed in (77, 88):
        model = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2, seed=seed,
                            policy_kwargs=dict(features_extractor_class=BattleFeatures))
        model.save(models / f"smart_{seed}.zip")
    originals = {p:p.read_bytes() for p in models.glob("*.zip")}
    run = tmp_path / "evaluation"
    command = [sys.executable, str(ROOT / "mgba/evaluate_models.py"), "--run-dir", str(run)]
    first = subprocess.run([*command, "--config", str(config), "--models-dir", str(models),
        "--environments", "4", "--showdown-workers", "process"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    report = read_json(run / "summary.json")
    assert report["status"] == "completed" and report["completed_matches"] == 3
    assert report["recorded_battles"] == 12
    assert all(r["games"] == 8 for r in report["leaderboards"]["overall"])
    assert not list(run.rglob("gym-steps.jsonl")) and not list(run.rglob("showdown-protocol.log"))
    assert not list(run.rglob("episode-*/teams.json"))
    assert len(list(run.rglob("*.zip"))) == 2
    assert all(path.read_bytes() == data for path,data in originals.items())
    for path in originals:
        path.unlink()
    config.unlink()
    before = (run / "matches.jsonl").read_bytes()
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (run / "matches.jsonl").read_bytes() == before
    assert report["leaderboards"] == read_json(run / "summary.json")["leaderboards"]
