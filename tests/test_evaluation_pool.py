from collections import Counter
import copy
from pathlib import Path

import pytest

from firered_rl.curriculum import load_curriculum
from firered_rl.evaluation import validate_suite
from firered_rl.evaluation_pool import DEFAULT_EVALUATION, pool_promotion, prepare_pool, verify_pool


ROOT = Path(__file__).resolve().parents[1]


def test_pool_quotas_side_balance_model_snapshots_and_replay(tmp_path):
    _, teams = load_curriculum(ROOT / "config/curricula/type_progression/curriculum.json")
    dumb, smart1, smart2 = [tmp_path / name for name in ("dumb.zip", "smart_v1.zip", "smart_v2.zip")]
    for path in (dumb, smart1, smart2):
        path.write_bytes(path.name.encode())
    options = dict(stage=2, episodes=1000, seed=91, max_steps=90, dumb=dumb,
                   smart=[("v1", smart1), ("v2", smart2), ("duplicate", smart1)])
    path, suite = prepare_pool(tmp_path / "pool", teams, **options)
    assert Counter(case["opponent_group"] for case in suite["cases"]) == dict(random=300, dumb=300, smart=400)
    assert Counter((case["opponent_group"], case["candidate_player"]) for case in suite["cases"]) == {
        ("random", 1):150, ("random", 2):150, ("dumb", 1):150, ("dumb", 2):150,
        ("smart", 1):200, ("smart", 2):200}
    assert len(suite["opponents"]) == 4  # identical Smart weights get one sampling entry
    assert {c["opponent_id"] for c in suite["cases"]} == set(suite["opponents"])
    assert suite["max_steps"] == 90
    assert validate_suite(suite) is suite
    # The next attempt draws new seeds/opponents, keeping the requested quotas.
    _, following = prepare_pool(tmp_path / "next", teams, **{**options, "seed":92})
    assert following["cases"] != suite["cases"]
    for source in (dumb, smart1, smart2):
        source.unlink()
    replay_path, replay = prepare_pool(tmp_path / "pool", teams, **options)
    assert replay_path == path and replay == suite
    changed = copy.deepcopy(suite)
    changed["cases"][1]["opponent_seed"] += 1
    with pytest.raises(ValueError, match="side-swapped"):
        validate_suite(changed)
    Path(suite["opponents"]["smart_1"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        verify_pool(suite)


@pytest.mark.parametrize("basic,smart,require,expected", [
    (450, 240, "both", "promote"), (449, 240, "both", "hold"),
    (450, 239, "both", "hold"), (450, 0, "either", "promote"),
    (0, 240, "either", "promote"), (449, 239, "either", "hold")])
def test_promotion_thresholds_are_inclusive_and_use_all_attempts(basic, smart, require, expected):
    # Some battles truncated: the gate must still divide by all attempts.
    rates = dict(all=dict(wins=basic+smart, attempts=1000),
        random=dict(attempts=300), dumb=dict(attempts=300),
        basic=dict(wins=basic, attempts=600, completed=max(basic, 300)),
        smart=dict(wins=smart, attempts=400, completed=max(smart, 200)))
    result = pool_promotion(rates, {**DEFAULT_EVALUATION, "require":require})
    assert result["status"] == expected
    rates["random"]["attempts"] = 0
    assert pool_promotion(rates, {**DEFAULT_EVALUATION, "require":require})["status"] == "hold"


def test_real_pooled_evaluation_replays_random_policy_across_concurrency(tmp_path, monkeypatch):
    import importlib.util
    import json
    import shutil
    import sys
    import gymnasium as gym
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    import torch
    from firered_rl.ppo_features import BattleFeatures, policy_space
    if not shutil.which("node") or not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file():
        pytest.skip("Showdown not installed")
    torch.set_num_threads(1)

    class ShapeEnv(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)

    model = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2, seed=77,
                        policy_kwargs=dict(features_extractor_class=BattleFeatures))
    candidate, smart = tmp_path / "candidate.zip", tmp_path / "smart.zip"
    model.save(candidate)
    model = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2, seed=88,
                        policy_kwargs=dict(features_extractor_class=BattleFeatures))
    model.save(smart)
    originals = {p:p.read_bytes() for p in (candidate, smart)}
    teams = [tmp_path / f"team{i}.json" for i in range(3)]
    for path, species in zip(teams, ("Raticate", "Linoone", "Furret")):
        path.write_text(json.dumps(dict(party=[dict(species=species, level=1,
            moves=["Double Edge", "Body Slam", "Strength", "Facade"])])))
    suite_path, _ = prepare_pool(tmp_path / "pool", teams, stage=2, episodes=20, seed=71,
        max_steps=8, dumb=smart, smart=[("smart_v1", smart)])
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    spec = importlib.util.spec_from_file_location("pool_eval_cli", ROOT / "mgba/evaluate_ppo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def no_training(*args, **kwargs):
        raise AssertionError("evaluation must never train or save weights")
    monkeypatch.setattr(MaskablePPO, "learn", no_training)
    monkeypatch.setattr(MaskablePPO, "save", no_training)
    common = ["evaluate_ppo", "--checkpoint", str(candidate), "--suite", str(suite_path)]
    runs = [tmp_path / "thread", tmp_path / "process"]
    for mode, environments, run in zip(("thread", "process"), ("4", "2"), runs):
        monkeypatch.setattr(sys, "argv", [*common, "--showdown-workers", mode,
            "--environments", environments, "--output-dir", str(run)])
        assert cli.main() == 0
    rows = lambda p: sorted([json.loads(line) for line in (p / "episodes.jsonl").read_text().splitlines()], key=lambda r:r["episode"])
    assert rows(runs[0]) == rows(runs[1])
    result = json.loads((runs[0] / "summary.json").read_text())
    assert result["attempted_battles"] == 20 and result["completed_battles"] == 20
    assert {group:r["attempts"] for group,r in result["by_opponent_group"].items()} == dict(random=6, dumb=6, smart=8, basic=12)
    assert all(path.read_bytes() == data for path, data in originals.items())
