import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
from firered_rl.opponent_pool import OpponentPool
from firered_rl.evaluation import validate_suite
from firered_rl.ppo_features import policy_space
from test_rule_opponent import observation, move

ROOT = Path(__file__).resolve().parents[1]


def write_pool(path, entries):
    path.write_text(json.dumps(dict(version=1, opponents=entries)))
    return path


def test_episode_selection_not_per_turn_and_independent_pair_rng(tmp_path):
    path = write_pool(tmp_path / "pool.json", {"rules": dict(kind="rule_based", weight=3), "random": dict(kind="random", weight=7)})
    pool = OpponentPool(ROOT, path, count=3, seed=81)
    replica = OpponentPool(ROOT, path, count=3, seed=81)
    pool.reset_episodes(range(3))
    replica.reset_episodes(range(3))
    selected = pool.selected.copy()
    obs = observation(pool.models["rules"])
    move(pool.models["rules"], obs, 0, "Tackle")
    move(pool.models["rules"], obs, 1, "Surf")
    batch = {key: np.stack([value] * 3) for key, value in obs.items()}
    for _ in range(8):
        actions, _ = pool.predict(batch, action_masks=batch["action_mask"])
        assert pool.selected == selected
        assert all(batch["action_mask"][i, a] for i, a in enumerate(actions))
    pool.finish_episodes([1])
    pool.reset_episodes([1])
    replica.reset_episodes([1])
    assert pool.selected == replica.selected  # Action sampling did not consume assignment RNG.
    assert pool.selected[0] == selected[0] and pool.selected[2] == selected[2]
    assert sum(pool.started.values()) == 4 and sum(pool.completed.values()) == 1


def test_weighted_quotas_seat_pairs_and_invalid_pools(tmp_path):
    path = write_pool(tmp_path / "pool.json", {"rules": dict(kind="rule_based", weight=3), "random": dict(kind="random", weight=7)})
    pool = OpponentPool(ROOT, path)
    # Full suite validation is exercised by the real CLI below; inspect scheduling here.
    suite = dict(seed=42, swap_sides=True, cases=[dict(episode=i+1) for i in range(100)])
    scheduled = pool.evaluation_suite(suite)
    assert "opponents" not in suite
    assert scheduled["opponent_quotas"] == {"random": 70, "rules": 30}
    for a, b in zip(scheduled["cases"][::2], scheduled["cases"][1::2]):
        assert a["opponent_id"] == b["opponent_id"] and a["opponent_seed"] == b["opponent_seed"]
    assert scheduled == pool.evaluation_suite(suite)
    with pytest.raises(ValueError, match="already pins"):
        pool.evaluation_suite(scheduled)
    with pytest.raises(ValueError, match="budget too small"):
        pool.evaluation_suite({**suite, "cases": suite["cases"][:2]})
    for weight in (0, -1, float("nan"), True):
        write_pool(path, {"x": dict(kind="random", weight=weight)})
        with pytest.raises(ValueError, match="weight"):
            OpponentPool(ROOT, path)


def test_mixed_v1_v2_inference_routes_only_assigned_rows(tmp_path, monkeypatch):
    from sb3_contrib import MaskablePPO
    import gymnasium as gym
    class Policy:
        observation_space = policy_space("v1")
        action_space = gym.spaces.Discrete(11)
        num_timesteps = 123
        def __init__(self): self.policy, self.calls = self, []
        def set_training_mode(self, value): assert not value
        def set_random_seed(self, seed): pass
        def predict(self, obs, action_masks, deterministic):
            assert set(obs) == set(self.observation_space.spaces)
            self.calls.append(len(action_masks))
            return np.zeros(len(action_masks), dtype=int), None
    model = Policy()
    monkeypatch.setattr(MaskablePPO, "load", lambda *a, **k: model)
    (tmp_path / "frozen.zip").write_bytes(b"fixture")
    path = write_pool(tmp_path / "pool.json", {"old": dict(kind="checkpoint", path="frozen.zip"), "rules": dict(kind="rule_based")})
    pool = OpponentPool(ROOT, path, count=4)
    pool.selected = ["old", "rules", "old", "rules"]
    obs = observation(pool.models["rules"])
    move(pool.models["rules"], obs, 0, "Tackle")
    move(pool.models["rules"], obs, 1, "Surf")
    batch = {key: np.stack([value] * 4) for key, value in obs.items()}
    choices, _ = pool.predict(batch, action_masks=batch["action_mask"])
    assert choices.tolist() == [0, 1, 0, 1] and model.calls == [2]
    installed = pool.install(tmp_path / "pinned")
    (tmp_path / "frozen.zip").unlink()
    assert OpponentPool(ROOT, installed).identity == pool.identity


def test_real_mixed_training_evaluation_continuation_and_pinning(tmp_path, monkeypatch):
    from test_observation_v2_training import real_vec, small_model, SHOWDOWN
    if not SHOWDOWN:
        pytest.skip("Showdown not installed")
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    train = importlib.import_module("train_ppo")
    evaluate = importlib.import_module("evaluate_ppo")
    teams = []
    for i, (species, moves) in enumerate((("Pikachu", ["Thunderbolt", "Quick Attack"]), ("Squirtle", ["Surf", "Tackle"]))):
        path = tmp_path / f"team-{i}.json"
        path.write_text(json.dumps(dict(party=[dict(species=species, level=50, moves=moves)])))
        teams.append(path)
    vec = real_vec(tmp_path / "source", teams, "v1")
    frozen = tmp_path / "frozen.zip"
    try:
        small_model(vec).save(frozen)
    finally:
        vec.close()
    before = frozen.read_bytes()
    entries = {"rules": dict(kind="rule_based", weight=1),
               "frozen": dict(kind="checkpoint", path="frozen.zip", weight=1),
               "random": dict(kind="random", weight=1)}
    pool = write_pool(tmp_path / "pool.json", entries)
    def invoke(cli, flags):
        monkeypatch.setattr(sys, "argv", ["cli", *map(str, flags)])
        return cli.main()
    output = tmp_path / "evaluation"
    assert invoke(evaluate, ["--checkpoint", frozen, "--opponent-pool", pool, "--teams", *teams,
                            "--episodes", 12, "--max-steps", 3, "--environments", 4,
                            "--showdown-workers", "process", "--rule-log", "--output-dir", output]) == 0
    suite = validate_suite(json.loads((output / "suite.json").read_text()))
    assert suite["opponent_quotas"] == dict(frozen=4, random=4, rules=4)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary["by_opponent"][key]["attempts"] == 4 for key in entries)
    assert summary["by_opponent_group"]["rules"]["attempts"] == 4
    # Combine a pinned pool with a standalone rules schedule and replay it.
    from firered_rl.combined_suite import combine_suites
    from firered_rl.rule_opponent import RuleBasedOpponent
    rule_suite = {key: value for key, value in suite.items()
                  if key not in ("opponents", "opponent_quotas")}
    rule_suite["cases"] = [{key: value for key, value in case.items()
                            if key not in ("opponent_id", "opponent_group", "opponent_seed")}
                           for case in suite["cases"][:2]]
    rule_suite["rule_opponent"] = RuleBasedOpponent(ROOT).identity
    rule_path = tmp_path / "rules-suite.json"
    rule_path.write_text(json.dumps(rule_suite))
    combined, _ = combine_suites([output / "suite.json", rule_path], tmp_path / "combined", project=ROOT)
    combined_run = tmp_path / "combined-evaluation"
    assert invoke(evaluate, ["--checkpoint", frozen, "--suite", combined,
                            "--environments", 4, "--output-dir", combined_run]) == 0
    combined_summary = json.loads((combined_run / "summary.json").read_text())
    assert combined_summary["by_opponent_group"]["rules"]["attempts"] == 6
    old_rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    new_rows = [json.loads(line) for line in (combined_run / "episodes.jsonl").read_text().splitlines()]
    for old, new in zip(sorted(old_rows, key=lambda r:r["episode"]),
                        sorted(new_rows, key=lambda r:r["episode"])[:12]):
        for key in ("candidate_outcome", "candidate_reward", "decisions", "truncated"):
            assert old[key] == new[key]
    run = tmp_path / "train"
    assert invoke(train, ["--output-dir", run, "--backend", "showdown", "--opponent-pool", pool,
                         "--observation-version", "v2", "--teams", *teams, "--environments", 4,
                         "--rollout-steps", 4, "--batch-size", 8, "--epochs", 1, "--max-steps", 3,
                         "--total-timesteps", 8, "--showdown-workers", "process", "--rule-log",
                         "--eval-suite", combined, "--eval-every", 8]) == 0
    config = json.loads((run / "config.json").read_text())
    assert config["learning_environments"] == 2 and config["session_observation_version"] == "v2"
    assert (run / "checkpoints/best.zip").is_file()
    assert "Rules" in (run / "evaluations.md").read_text()
    assert frozen.read_bytes() == before
    # Continuing/replaying must work after original opponent files are removed.
    frozen.unlink()
    pool.unlink()
    assert invoke(train, ["--output-dir", run, "--if-exists", "continue", "--total-timesteps", 8]) == 0
    replay = tmp_path / "replay"
    assert invoke(evaluate, ["--checkpoint", output / "candidate.zip", "--suite", output / "suite.json",
                            "--environments", 2, "--showdown-workers", "process", "--output-dir", replay]) == 0
    again = json.loads((replay / "summary.json").read_text())
    for key in ("candidate_wins", "opponent_wins", "draws", "truncated", "suite_sha256", "by_opponent"):
        assert summary[key] == again[key]
    # Changing the mixture cannot silently alter an existing experiment.
    pinned = Path(json.loads((run / "config.json").read_text())["opponent_pool"])
    changed = json.loads(pinned.read_text())
    changed["opponents"]["rules"]["weight"] = 10
    pinned.write_text(json.dumps(changed))
    with pytest.raises(SystemExit) as error:
        invoke(train, ["--output-dir", run, "--if-exists", "continue", "--total-timesteps", 8])
    assert error.value.code == 2
