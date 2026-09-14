from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from firered_rl.switch_opponent import SwitchingOpponent, switch_settings
from firered_rl.turn_env import observation_space

ROOT = Path(__file__).resolve().parents[1]


def matchup():
    obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in observation_space().spaces.items()}
    # Opponent's own view: Venusaur facing Charizard, with Blastoise on the bench.
    obs["party"][0] = [1, 3, 50, 100, 100, 12, 3, 0, 0, 0]
    obs["party"][1] = [1, 9, 50, 100, 100, 11, 11, 0, 0, 0]
    obs["active"][0] = [1, 3, 50, 100, 100, 12, 3, 0, 1, 0]
    obs["active"][1] = [1, 6, 50, 100, 100, 10, 2, 0, 0, 0]
    obs["moves"][0, 0] = [75, 12, 25, 25]
    obs["moves"][1, 0] = [57, 11, 15, 15]
    obs["opponent_moves"][0, 0] = [53, 10, 15, 15]
    obs["opponent_active_index"][0] = 1
    obs["action_mask"][[0, 5]] = 1
    return obs


def batched(obs, n=1):
    return {key: np.stack([value.copy() for _ in range(n)]) for key, value in obs.items()}


def test_gen3_dual_types_immunities_and_better_legal_switch():
    chooser = SwitchingOpponent(ROOT, 1, chance=1)
    assert chooser.effectiveness(10, [12, 12]) == 2
    assert chooser.effectiveness(13, [11, 2]) == 4
    assert chooser.effectiveness(13, [11, 4]) == 0
    assert chooser.effectiveness(7, [8, 8]) == .5
    assert chooser.effectiveness(17, [8, 8]) == .5
    obs = matchup()
    result = chooser.best_switch(obs)
    assert result["action"] == 5 and result["species"] == "Blastoise"
    assert result["replacement"]["incoming"] == .5
    obs["active"][0] = [1, 9, 50, 100, 100, 11, 11, 0, 2, 0]
    obs["action_mask"][4:10] = [1, 0, 0, 0, 0, 0]
    assert chooser.best_switch(obs) is None


@pytest.mark.parametrize("case", ["trapped", "forced", "waiting", "fainted", "low_hp", "empty_pp", "neutral", "coverage"])
def test_only_eligible_healthy_improving_switches(case):
    obs = matchup()
    if case == "trapped": obs["action_mask"][5] = 0
    if case == "forced": obs["context"][1] = 1
    if case == "waiting": obs["action_mask"][:4] = 0
    if case == "fainted": obs["party"][1, 3] = 0
    if case == "low_hp": obs["party"][1, 3] = 34
    if case == "empty_pp": obs["moves"][1, :, 2] = 0
    if case == "neutral":
        obs["active"][0, 5:7] = [11, 11]
        obs["moves"][0] = obs["moves"][1]
    if case == "coverage":
        obs["opponent_moves"][0, 1] = [85, 13, 15, 15]
        obs["active"][0, 5:7] = [10, 10]
    assert SwitchingOpponent(ROOT, 1, chance=1).best_switch(obs) is None


def test_probability_cooldown_resets_and_input_immutability():
    obs = batched(matchup(), 2)
    original = deepcopy(obs)
    assert SwitchingOpponent(ROOT, 2, chance=0).choose(obs, np.array([0, 0]))[0].tolist() == [0, 0]
    chooser = SwitchingOpponent(ROOT, 2, chance=1, cooldown=2)
    assert chooser.choose(obs, np.array([0, 0]))[0].tolist() == [5, 5]
    assert chooser.choose(obs, np.array([0, 0]))[0].tolist() == [0, 0]
    chooser.reset([0])
    assert chooser.choose(obs, np.array([0, 0]))[0].tolist() == [5, 0]
    assert chooser.choose(obs, np.array([0, 0]))[0].tolist() == [0, 5]
    assert all(np.array_equal(obs[key], original[key]) for key in obs)
    first = SwitchingOpponent(ROOT, 2, chance=.3, cooldown=0, seed=44)
    second = SwitchingOpponent(ROOT, 2, chance=.3, cooldown=0, seed=44)
    a = [first.choose(obs, np.array([0, 0]))[0].tolist() for _ in range(40)]
    b = [second.choose(obs, np.array([0, 0]))[0].tolist() for _ in range(40)]
    assert a == b and any(5 in row for row in a) and any(0 in row for row in a)


@pytest.mark.parametrize("settings", [dict(chance=1.1), dict(chance=float("nan")), dict(chance=True), dict(cooldown=-1), dict(cooldown=1.5)])
def test_invalid_switch_settings(settings):
    with pytest.raises(ValueError):
        switch_settings(**settings)


def test_real_showdown_training_with_switching_opponent(tmp_path):
    if not (ROOT / "sim/showdown/node_modules/pokemon-showdown").is_dir():
        pytest.skip("install the pinned Showdown dependencies")
    torch = pytest.importorskip("torch")
    PPO = pytest.importorskip("sb3_contrib").MaskablePPO
    import gymnasium as gym
    from firered_rl.ppo_features import policy_space, BattleFeatures

    class SpacesOnly(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)

    torch.set_num_threads(1)
    model = PPO("MultiInputPolicy", SpacesOnly(), n_steps=4, batch_size=4, n_epochs=1,
                policy_kwargs=dict(features_extractor_class=BattleFeatures), device="cpu", seed=1)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.bias[0] = 10
    checkpoint = tmp_path / "frozen.zip"
    model.save(checkpoint)
    before = checkpoint.read_bytes()
    team = ROOT / "config/training_profiles/six_types/teams/six_types.json"
    run = tmp_path / "train"
    result = subprocess.run([sys.executable, str(ROOT / "mgba/train_ppo.py"),
        "--backend", "showdown", "--environments", "4", "--showdown-workers", "process",
        "--teams", str(team), str(team), "--resume", str(checkpoint), "--opponent-checkpoint", str(checkpoint),
        "--opponent-switch-chance", "1", "--opponent-switch-cooldown", "2",
        "--total-timesteps", "8", "--rollout-steps", "4", "--batch-size", "4", "--epochs", "1",
        "--no-readable-log", "--no-battle-log", "--output-dir", str(run)],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    stats = json.loads((run / "switching.json").read_text())
    assert stats["heuristic_overrides"] >= 2
    assert stats["learner_normal_decisions"] > 0
    assert stats["opponent_voluntary_switches"] >= stats["heuristic_overrides"]
    events = [json.loads(line) for line in (run / "opponent-switches.jsonl").read_text().splitlines()]
    assert {event["pair"] for event in events} == {1, 2}
    assert all(4 <= event["action"] <= 9 and event["action"] != event["policy_action"] for event in events)
    assert checkpoint.read_bytes() == before
    assert PPO.load(run / "checkpoints/final.zip", device="cpu").num_timesteps == 8
    assert "switching/learner_voluntary_switch_rate" in (run / "ppo/progress.csv").read_text()
