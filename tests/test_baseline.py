import json
import random

import gymnasium as gym
import numpy as np

from firered_rl.baseline import choose_action, replay_actions, run_baseline


class FakeEnv:
    observation_space = gym.spaces.Dict({"action_mask": gym.spaces.MultiBinary(11)})
    report = {"frame": 123}

    def __init__(self, fail=False):
        self.resets, self.actions, self.fail = 0, [], fail

    def reset(self, seed=None):
        self.resets += 1
        self.count = 0
        return {"action_mask": np.array([0, 1]+[0]*9, dtype=np.int8)}, {}

    def step(self, action):
        self.actions.append(action)
        if self.fail:
            raise TimeoutError("decision timed out")
        self.count += 1
        return {"action_mask": np.array([0, 1]+[0]*9, dtype=np.int8)}, 2.0, self.count == 2, False, {"outcome_reason": "won"}


def test_masked_policy_and_moves_first():
    mask = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    assert {choose_action(mask, random.Random(i)) for i in range(100)} == {0, 5}
    assert choose_action(mask, random.Random(1), moves_first=True) == 0


def test_reset_each_episode_and_replay_log(tmp_path):
    env = FakeEnv()
    result = run_baseline(env, tmp_path, episodes=2)
    assert env.resets == 2 and result["outcomes"] == {"won": 2}
    assert result["episodes"][0]["total_reward"] == 4
    assert replay_actions(tmp_path / "events.jsonl", 2) == [1, 1]


def test_failure_logs_attempt_before_step_and_stops(tmp_path):
    env = FakeEnv(fail=True)
    result = run_baseline(env, tmp_path, episodes=3)
    assert env.resets == 1 and result["outcomes"] == {"timeout": 1}
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-2]["event"] == "action_requested"
    assert events[-1]["last_inspection"] == {"frame": 123}
    assert replay_actions(tmp_path / "events.jsonl", 1) == [1]
