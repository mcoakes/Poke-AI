"""Coverage, immutable inputs, and frozen benchmark cache identities."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("sb3_contrib")
from firered_rl import curriculum_checks as checks
from firered_rl.evaluation import validate_suite
from firered_rl.fixed_team_lab import pokemon

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def pool(tmp_path, count=5):
    choices = [("Typhlosion", "Flamethrower"), ("Blastoise", "Surf"),
               ("Sceptile", "LeafBlade"), ("Starmie", "Surf"), ("Venusaur", "RazorLeaf")]
    return [write(tmp_path / f"team-{i}.json", dict(name=f"team-{i}", party=[
        pokemon(species, ["Strength", move])])) for i, (species, move) in enumerate(choices[:count])]


def semantic_team(suite, case, side):
    team = copy.deepcopy(suite["assets"][case["assets"][side]]["team"])
    for mon in team["party"]:
        mon["moves"].sort()
    return team


def test_coverage_covers_each_team_before_first_role_repeats(tmp_path):
    paths = pool(tmp_path)
    suite = checks.coverage_suite(ROOT, paths, 1, 543, max_steps=240)
    assert len(suite["cases"]) == 2 * len(paths)
    assert suite["max_steps"] == 240
    assert suite["diagnostic"]["requested_budget"] == 1
    assert suite["diagnostic"]["actual_battles"] == 10
    first_keys = [case["assets"][0] for case in suite["cases"][::2]]
    assert len(set(first_keys)) == len(paths)
    assert set(suite["diagnostic"]["candidate_team_exposures"].values()) == {2}
    for left, right in zip(suite["cases"][::2], suite["cases"][1::2]):
        assert left["candidate_player"] == 1 and right["candidate_player"] == 2
        assert left["assets"] == right["assets"]
        assert left["seed"] == right["seed"]
        assert left["assets"][0] != left["assets"][1]
    assert suite == checks.coverage_suite(ROOT, paths, 1, 543, max_steps=240)


def test_coverage_rotation_preserves_pairings_and_rng(tmp_path):
    paths = pool(tmp_path)
    ordinary = checks.coverage_suite(ROOT, paths, 23, 777)
    rotated = checks.coverage_suite(ROOT, paths, 23, 777, permute_moves=True)
    assert len(ordinary["cases"]) == 24
    validate_suite(ordinary)
    validate_suite(rotated)
    changed = 0
    for a, b in zip(ordinary["cases"], rotated["cases"]):
        assert (a["seed"], a["candidate_player"], a["matchup"]) == (
            b["seed"], b["candidate_player"], b["matchup"])
        side = a["candidate_player"] - 1
        assert semantic_team(ordinary, a, side) == semantic_team(rotated, b, side)
        assert ordinary["assets"][a["assets"][1-side]] == rotated["assets"][b["assets"][1-side]]
        changed += a["assets"][side] != b["assets"][side]
    assert changed > 0


def test_pool_inventory_checks_every_split_and_semantic_variance(tmp_path):
    entry = dict(root=str(tmp_path), sha256={})
    for split in ("train", "validation", "heldout", "benchmark"):
        paths = pool(tmp_path / split, 2)
        entry[f"{split}_dir"] = split
        entry[f"{split}_paths"] = [str(p.relative_to(tmp_path)) for p in paths]
        entry["sha256"].update({str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in paths})
    inventory = checks.pool_inventory(entry)
    assert all(value["teams"] == 2 for value in inventory["splits"].values())
    assert inventory["splits"]["train"]["species"] == {"Blastoise": 1, "Typhlosion": 1}
    assert inventory["splits"]["train"]["moves"]["Strength"] == 2
    extra = tmp_path / "benchmark" / "untracked.json"
    extra.write_text((tmp_path / "benchmark/team-0.json").read_text())
    with pytest.raises(ValueError, match="membership changed"):
        checks.pool_inventory(entry)
    extra.unlink()
    (tmp_path / "heldout/team-0.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned team changed"):
        checks.pool_inventory(entry)


@pytest.fixture
def frozen_context(tmp_path, monkeypatch):
    root = tmp_path / "run"
    reward = write(root / "inputs/reward.json", {"test": 1})
    suite = write(tmp_path / "suite.json", checks.coverage_suite(ROOT, pool(tmp_path / "pool", 2), 4, 1))
    calls = []

    def score(project, run, schedule, candidate, reward_path, **kwargs):
        run.mkdir(parents=True)
        write(run / "suite.json", schedule)
        (run / "episodes.jsonl").write_text("".join(json.dumps(dict(episode=i+1,
            candidate_outcome=("won", "lost", "lost", "truncated")[i], truncated=i==3)) + "\n"
            for i in range(4)))
        calls.append(dict(run=run, suite=schedule, candidate=candidate, **kwargs))
        # Core candidate_win_rate excludes truncation. Wrapper must use requested.
        return dict(status="completed", requested_battles=4, completed_battles=3,
                    attempted_battles=4, candidate_wins=1, opponent_wins=2, draws=0,
                    truncated=1, candidate_win_rate=1/3)

    monkeypatch.setattr(checks, "score_stage", score)
    monkeypatch.setattr(checks, "_load_model", lambda path: ("loaded", str(path)))
    return root, suite, reward, calls


def test_frozen_cache_includes_candidate_opponent_suite_and_reward(frozen_context, tmp_path):
    root, suite, reward, calls = frozen_context
    candidate, opponent = tmp_path / "candidate.zip", tmp_path / "opponent.zip"
    candidate.write_bytes(b"candidate-one")
    opponent.write_bytes(b"opponent-one")
    kwargs = dict(checkpoint=candidate, opponent_kind="checkpoint", opponent_checkpoint=opponent)
    first = checks.frozen_check(ROOT, root, "final", suite, **kwargs)
    assert first["wins_per_requested"] == .25
    assert first["candidate_win_rate"] == 1/3
    assert first["opponent_kind"] == "checkpoint"
    assert "rule_opponent" not in calls[-1]["suite"]
    assert calls[-1]["suite"]["check_opponent"]["sha256"] == hashlib.sha256(opponent.read_bytes()).hexdigest()
    assert checks.frozen_check(ROOT, root, "final", suite, **kwargs) == first
    assert len(calls) == 1
    candidate.write_bytes(b"candidate-two")
    second = checks.frozen_check(ROOT, root, "final", suite, **kwargs)
    assert second["candidate_identity"] != first["candidate_identity"]
    opponent.write_bytes(b"opponent-two")
    third = checks.frozen_check(ROOT, root, "final", suite, **kwargs)
    assert third["opponent_identity"] != second["opponent_identity"]
    data = json.loads(suite.read_text())
    data["cases"][0]["seed"] += 1
    data["cases"][1]["seed"] += 1
    write(suite, data)
    checks.frozen_check(ROOT, root, "final", suite, **kwargs)
    write(reward, {"test": 2})
    checks.frozen_check(ROOT, root, "final", suite, **kwargs)
    assert len(calls) == 5
    assert len({row["run"] for row in calls}) == 5
    assert first["result_directory"] == "checks/final"


def test_partial_retried_and_greedy_benchmark_not_labelled_rules(frozen_context):
    root, suite, reward, calls = frozen_context
    write(root / "checks/baseline/summary.json", {"status": "running"})
    result = checks.frozen_check(ROOT, root, "baseline", suite, opponent_kind="greedy")
    assert "-retry-" in result["result_directory"]
    assert calls[0]["candidate"] is None
    assert result["candidate_identity"]["kind"] == "random"
    assert result["opponent_identity"]["kind"] == "greedy_damage"
    assert "rule_opponent" not in calls[0]["suite"]
    assert json.loads((root / "checks/baseline/summary.json").read_text())["status"] == "running"


def test_completed_summary_with_incomplete_episode_log_is_retried(frozen_context):
    root, suite, reward, calls = frozen_context
    original = checks.frozen_check(ROOT, root, "incomplete-log", suite)
    episodes = root / original["result_directory"] / "episodes.jsonl"
    episodes.write_text(episodes.read_text().splitlines()[0] + "\n")
    retry = checks.frozen_check(ROOT, root, "incomplete-log", suite)
    assert retry["result_directory"] != original["result_directory"]
    assert len(calls) == 2
    assert len(episodes.read_text().splitlines()) == 1
    assert checks.frozen_check(ROOT, root, "incomplete-log", suite) == retry
    assert len(calls) == 2


def test_wrong_rules_identity_rejected_before_reuse(frozen_context):
    root, suite, reward, calls = frozen_context
    data = json.loads(suite.read_text())
    data["rule_opponent"]["sha256"] = "0" * 64
    write(suite, data)
    with pytest.raises(ValueError, match="Rules identity differs"):
        checks.frozen_check(ROOT, root, "rules", suite)
    assert calls == []
