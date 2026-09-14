from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

from firered_rl.link_inspection import pair_inspections
from firered_rl.linked_env import linked_observation
from firered_rl.ppo_features import BattleFeatures, policy_observation, policy_space
from firered_rl.ppo_vec_env import LinkedVecEnv


@pytest.fixture
def raw_observations():
    raw = json.loads((Path(__file__).parents[1] / "tests/fixtures/link_master_follower.json").read_text())
    return [linked_observation(r) for r in pair_inspections(raw)]


class Pair:
    def __init__(self, observations, *, truncated=False, limit=3, barrier=None):
        self.initial = observations
        self.truncated, self.limit, self.barrier = truncated, limit, barrier
        self.resets = self.steps = 0
        self.closed = False

    def players(self):
        return [SimpleNamespace(render_mode=None), SimpleNamespace(render_mode=None)]

    def observe(self):
        observations = deepcopy(self.initial)
        for obs in observations:
            obs["context"][0] = self.turn
            obs["action_mask"][:] = 0
            obs["action_mask"][[0, 2] if self.turn % 2 == 0 else [1, 5]] = 1
        return observations

    def reset_pair(self, seeds):
        self.turn = 0
        self.resets += 1
        return [(obs, {"reset": self.resets}) for obs in self.observe()]

    def step_pair(self, actions):
        if self.barrier:
            self.barrier.wait(timeout=3)
        for action, obs in zip(actions, self.observe()):
            assert obs["action_mask"][action], "invalid PPO action reached the simulated emulator"
        self.steps += 1
        self.turn += 1
        done = self.turn >= self.limit
        observations = self.observe()
        results = []
        for i, obs in enumerate(observations):
            terminated = done and not self.truncated
            if terminated:
                obs["context"][2] = 1
                obs["action_mask"][:] = 0
            results.append((obs, (-1 if i == 0 else 1) if terminated else 0,
                            terminated, done and self.truncated,
                            {"outcome": ("lost" if i == 0 else "won") if terminated else "ongoing"}))
        return results

    def close(self):
        self.closed = True


def test_bitfields_survive_float_preprocessing(raw_observations):
    raw = deepcopy(raw_observations[0])
    raw["active"][0, 9] = (1 << 31) | 1
    raw["party"][0, 7] = (1 << 30) | 3
    obs = policy_observation(raw)
    assert obs["volatile_bits"][0, [0, 31]].tolist() == [1, 1]
    assert obs["status_bits"][0, [0, 1, 30]].tolist() == [1, 1, 1]
    assert policy_space().contains(obs)


def test_concurrent_pairs_autoreset_and_final_observations(raw_observations):
    barrier = Barrier(2)
    pairs = [Pair(raw_observations, truncated=True, limit=1, barrier=barrier),
             Pair(raw_observations, limit=1, barrier=barrier)]
    events = []
    vec = LinkedVecEnv(pairs, on_episode=lambda *a, **kw: events.append((a, kw)))
    try:
        vec.seed(5)
        before = vec.reset()
        assert is_masking_supported(vec)
        with pytest.raises(ValueError, match="unavailable"):
            vec.step_async(np.array([0, 0, 0, 10]))
        assert all(pair.steps == 0 for pair in pairs)
        obs, rewards, done, infos = vec.step(np.array([0, 2, 0, 2]))
        assert all(done) and rewards.tolist() == [0, 0, -1, 1]
        assert len(events) == 2 and all(pair.resets == 2 for pair in pairs)
        assert all(obs["context"][:, 0] == 0)
        assert all(info["terminal_observation"]["context"][0] == 1 for info in infos)
        assert [info["TimeLimit.truncated"] for info in infos] == [True, True, False, False]
        assert all(infos[i]["terminal_observation"]["action_mask"].any() for i in (0, 1))
        assert get_action_masks(vec).shape == (4, 11)
        assert all(vec.reset_infos[i]["reset"] == 2 for i in range(4))
        # Callers cannot accidentally change the cached action masks.
        obs["action_mask"][:] = 0
        assert get_action_masks(vec).any()
    finally:
        vec.close()
    assert all(pair.closed for pair in pairs)


def test_real_ppo_updates_masks_bootstrap_and_checkpoint_reload(raw_observations, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    pairs = [Pair(raw_observations, truncated=True, limit=2), Pair(raw_observations, limit=3)]
    vec = LinkedVecEnv(pairs)
    try:
        model = MaskablePPO("MultiInputPolicy", vec, n_steps=4, batch_size=8, n_epochs=2,
            learning_rate=1e-3, device="cpu", seed=7,
            policy_kwargs={"features_extractor_class": BattleFeatures, "net_arch": [32]})
        before = {name: p.detach().clone() for name, p in model.policy.named_parameters()}
        # Known bootstrap values prove time limits differ from game terminals.
        with monkeypatch.context() as context:
            context.setattr(model.policy, "predict_values", lambda obs: torch.full((len(obs["party"]), 1), .5))
            model.learn(16)
            rewards = model.rollout_buffer.rewards
            assert np.allclose(rewards[[1, 3], :2], .5*model.gamma)
            assert rewards[2, 2:].tolist() == [-1, 1]
        assert any(not torch.equal(before[name], p) for name, p in model.policy.named_parameters())
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        saved = tmp_path / "policy.zip"
        model.save(saved)
        loaded = MaskablePPO.load(saved, env=vec, device="cpu")
        assert loaded.policy.optimizer.state_dict()["state"]
        assert loaded.num_timesteps == 16
        loaded.learn(16, reset_num_timesteps=False)
        assert loaded.num_timesteps == 32
        obs = vec.reset()
        masks = get_action_masks(vec)
        for _ in range(20):
            actions, _ = loaded.predict(obs, action_masks=masks)
            assert all(masks[i, action] for i, action in enumerate(actions))
    finally:
        vec.close()
