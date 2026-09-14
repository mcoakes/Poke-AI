"""Categorical embeddings and scaled battle values for the PPO policy."""
import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from firered_rl.turn_env import observation_space


def policy_space():
    spaces = {key: gym.spaces.Box(0, np.float32(space.high.max()) if hasattr(space, "high") else 1,
                                 shape=space.shape, dtype=np.float32)
              for key, space in observation_space().spaces.items()}
    spaces["status_bits"] = gym.spaces.Box(0, 1, (8, 32), dtype=np.float32)
    spaces["volatile_bits"] = gym.spaces.Box(0, 1, (2, 32), dtype=np.float32)
    return gym.spaces.Dict(spaces)


def policy_observation(raw):
    # Expand bitfields BEFORE SB3 converts integers to float32 (which loses low
    # bits in 32-bit status flags). All remaining IDs/values fit exactly.
    result = {key: np.asarray(value, dtype=np.float32).copy() for key, value in raw.items()}
    statuses = np.concatenate([raw["party"][:, 7], raw["active"][:, 7]]).astype(np.uint64)
    volatile = np.asarray(raw["active"][:, 9], dtype=np.uint64)
    shifts = np.arange(32, dtype=np.uint64)
    result["status_bits"] = ((statuses[:, None] >> shifts) & 1).astype(np.float32)
    result["volatile_bits"] = ((volatile[:, None] >> shifts) & 1).astype(np.float32)
    result["party"][:, 7] = 0
    result["active"][:, 7] = result["active"][:, 9] = 0
    return result


class BattleFeatures(BaseFeaturesExtractor):
    """Shared species/move/type/item embeddings; IDs are not ordinal scores."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.species = nn.Embedding(512, 12, padding_idx=0)
        self.move = nn.Embedding(512, 16, padding_idx=0)
        self.item = nn.Embedding(512, 4, padding_idx=0)
        self.types = nn.Embedding(32, 4)
        with torch.no_grad():
            sample = {key: torch.zeros((1, *space.shape)) for key, space in observation_space.spaces.items()}
            width = self._flatten(sample).shape[1]
        self.network = nn.Sequential(nn.Linear(width, 256), nn.ReLU(),
                                     nn.Linear(256, features_dim), nn.ReLU())

    @staticmethod
    def ids(value, size=512):
        return value.long().clamp(0, size-1)

    def _flatten(self, obs):
        features = []
        for name in ("party", "active"):
            p = obs[name]
            hp_fraction = (p[..., 3:4] / p[..., 4:5].clamp(min=1)).clamp(0, 1)
            numeric = torch.cat([p[..., :1], p[..., 2:3]/100, hp_fraction,
                                 p[..., 4:5]/1024], -1)
            mon = [self.species(self.ids(p[..., 1])), numeric,
                   self.types(self.ids(p[..., 5:7], 32)).flatten(-2)]
            if name == "party":
                mon.extend([self.item(self.ids(p[..., 8])), p[..., 9:10]])
            else:
                mon.append(p[..., 8:9]/6)
            features.append((torch.cat(mon, -1) * p[..., :1]).flatten(1))
        for name in ("moves", "opponent_moves"):
            m = obs[name]
            values = torch.cat([self.move(self.ids(m[..., 0])),
                                self.types(self.ids(m[..., 1], 32)), (m[..., 2:4]/64).clamp(0, 1)], -1)
            features.append((values * (m[..., :1] != 0)).flatten(1))
        p = obs["opponent_party"]
        features.append((torch.cat([p[..., :1], self.species(self.ids(p[..., 1]))], -1) * p[..., :1]).flatten(1))
        bag = obs["bag"]
        values = torch.cat([self.item(self.ids(bag[..., 0])), (bag[..., 1:2]/999).clamp(0, 1)], -1)
        features.append((values * (bag[..., :1] != 0)).flatten(1))
        scale = obs["context"].new_tensor([255, 1, 1, 10])
        features.extend([obs["context"]/scale, obs["action_mask"], obs["opponent_active_index"]/6,
                         obs["status_bits"].flatten(1), obs["volatile_bits"].flatten(1)])
        return torch.cat(features, dim=1)

    def forward(self, observations):
        return self.network(self._flatten(observations))
