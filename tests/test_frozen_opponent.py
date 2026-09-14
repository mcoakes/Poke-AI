from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
MaskablePPO = pytest.importorskip("sb3_contrib").MaskablePPO
from firered_rl.frozen_opponent import FrozenOpponentVecEnv
from firered_rl.link_inspection import pair_inspections
from firered_rl.linked_env import linked_observation
from firered_rl.ppo_features import BattleFeatures
from firered_rl.ppo_vec_env import LinkedVecEnv
from test_ppo import Pair


def test_only_player_one_trains_and_frozen_weights_and_checkpoint_stay_fixed(tmp_path):
    raw = json.loads((Path(__file__).parents[1] / "tests/fixtures/link_master_follower.json").read_text())
    obs = [linked_observation(r) for r in pair_inspections(raw)]
    torch.set_num_threads(1)
    initial = LinkedVecEnv([Pair(obs, limit=2)])
    try:
        opponent = MaskablePPO("MultiInputPolicy", initial, n_steps=2, batch_size=4, n_epochs=1,
                              policy_kwargs=dict(features_extractor_class=BattleFeatures))
        opponent.learn(4)
        checkpoint = tmp_path / "fixed.zip"
        opponent.save(checkpoint)
    finally:
        initial.close()
    opponent = MaskablePPO.load(checkpoint, device="cpu")
    before_file = checkpoint.read_bytes()
    before_weights = deepcopy(opponent.policy.state_dict())
    events = []
    paired = LinkedVecEnv([Pair(obs, limit=2), Pair(obs, limit=2)], reward_scale=.001,
                          on_episode=lambda pair, **data: events.append(data))
    vec = FrozenOpponentVecEnv(paired, opponent)
    try:
        learner = MaskablePPO("MultiInputPolicy", vec, n_steps=2, batch_size=4, n_epochs=1,
                             policy_kwargs=dict(features_extractor_class=BattleFeatures))
        initial_learner = deepcopy(learner.policy.state_dict())
        learner.learn(8)
        assert learner.num_timesteps == 8 and opponent.num_timesteps == 4
        assert learner.rollout_buffer.rewards.shape == (2, 2)  # two learners, NOT four players
        assert np.all(learner.rollout_buffer.rewards <= 0)  # player-2 wins are excluded
        assert np.min(learner.rollout_buffer.rewards) == pytest.approx(-.001)
        assert events and all(e["returns"] == [-1, 1] for e in events)  # reports remain raw points
        assert any(not torch.equal(v, learner.policy.state_dict()[k]) for k, v in initial_learner.items())
        assert all(torch.equal(v, opponent.policy.state_dict()[k]) for k, v in before_weights.items())
        assert checkpoint.read_bytes() == before_file
        assert not opponent.policy.training
    finally:
        vec.close()
