"""Categorical embeddings and scaled battle values for the PPO policy."""
from functools import lru_cache

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from firered_rl.turn_env import observation_space
from firered_rl.observation_v2 import (
    raw_observation_version, validate_v2_fields, validate_version, v2_spaces,
)


def policy_space(version="v1"):
    validate_version(version)
    spaces = {key: gym.spaces.Box(0, np.float32(space.high.max()) if hasattr(space, "high") else 1,
                                 shape=space.shape, dtype=np.float32)
              for key, space in observation_space().spaces.items()}
    spaces["status_bits"] = gym.spaces.Box(0, 1, (8, 32), dtype=np.float32)
    spaces["volatile_bits"] = gym.spaces.Box(0, 1, (2, 32), dtype=np.float32)
    if version == "v2":
        spaces.update({key: gym.spaces.Box(space.low.astype(np.float32),
                                          space.high.astype(np.float32), dtype=np.float32)
                       for key, space in v2_spaces().items()})
        spaces["v2_status_bits"] = gym.spaces.Box(0, 1, (2, 6, 32), dtype=np.float32)
        spaces["v2_volatile_bits"] = gym.spaces.Box(0, 1, (2, 32), dtype=np.float32)
    return gym.spaces.Dict(spaces)


def policy_observation(raw, version=None):
    version = validate_version(raw_observation_version(raw) if version is None else version)
    extra = None
    if version == "v2":
        extra = validate_v2_fields({key: value for key, value in raw.items()
                                    if key.startswith("v2_")})
    elif raw_observation_version(raw) == "v2":
        raw = {key: raw[key] for key in observation_space().spaces}
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
    if extra is not None:
        statuses = extra["v2_party"][..., 9].astype(np.uint64)
        volatile = extra["v2_volatiles"].astype(np.uint64)
        result["v2_status_bits"] = ((statuses[..., None] >> shifts) & 1).astype(np.float32)
        result["v2_volatile_bits"] = ((volatile[:, None] >> shifts) & 1).astype(np.float32)
        result["v2_party"][..., 9] = 0
        result["v2_volatiles"][:] = 0
    return result


def observation_version(space):
    """Identify complete supported checkpoint spaces, rejecting partial schemas."""
    for version in ("v1", "v2"):
        expected = policy_space(version)
        if not isinstance(space, gym.spaces.Dict) or set(space.spaces) != set(expected.spaces):
            continue
        # Gym Box equality uses allclose for bounds; checkpoint contracts need
        # exact bounds as well as shapes/dtypes, including uint32-scale limits.
        if all(isinstance(space[key], gym.spaces.Box)
               and space[key].shape == target.shape
               and space[key].dtype == target.dtype
               and np.array_equal(space[key].low, target.low)
               and np.array_equal(space[key].high, target.high)
               for key, target in expected.spaces.items()):
            return version
    raise ValueError("unsupported checkpoint observation space; expected the complete v1 or v2 schema")


@lru_cache(maxsize=2)
def _policy_keys(version):
    return tuple(policy_space(version).spaces)


def policy_for_version(observation, version):
    """Project preprocessed policy data without re-expanding zeroed bitfields."""
    validate_version(version)
    source = raw_observation_version(observation)
    expected = set(_policy_keys(source))
    if set(observation) != expected:
        raise ValueError(f"incomplete or unsupported preprocessed {source} observation")
    if source == "v1" and version == "v2":
        raise ValueError("cannot upgrade a v1 observation to v2 without native Showdown fields")
    return {key: observation[key] for key in _policy_keys(version)}


class BattleFeatures(BaseFeaturesExtractor):
    """Shared species/move/type/item embeddings; IDs are not ordinal scores."""
    def __init__(self, observation_space, features_dim=128, include_moves=True):
        super().__init__(observation_space, features_dim)
        self.observation_version = observation_version(observation_space)
        self.include_moves = include_moves
        self.species = nn.Embedding(512, 12, padding_idx=0)
        self.move = nn.Embedding(512, 16, padding_idx=0)
        self.item = nn.Embedding(512, 4, padding_idx=0)
        self.types = nn.Embedding(32, 4)
        if self.observation_version == "v2":
            self.ability = nn.Embedding(256, 8, padding_idx=0)
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
            if not self.include_moves:
                continue
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
        if self.observation_version == "v2":
            features.extend(self._flatten_v2(obs))
        return torch.cat(features, dim=1)

    def _flatten_v2(self, obs):
        p = obs["v2_party"]
        hp_fraction = (p[..., 2:3] / p[..., 3:4].clamp(min=1)).clamp(0, 1)
        party = torch.cat([
            p[..., :1], p[..., 1:2]/100, p[..., 2:4]/1024, hp_fraction,
            self.types(self.ids(p[..., 4:6], 32)).flatten(-2),
            self.item(self.ids(p[..., 6])), self.ability(self.ids(p[..., 7], 256)),
            p[..., 8:9], obs["v2_stats"]/1024, obs["v2_status_bits"],
        ], -1)
        features = [(party * p[..., :1]).flatten(1)]

        active = obs["v2_active"]
        hp_fraction = (active[..., :1] / active[..., 1:2].clamp(min=1)).clamp(0, 1)
        values = torch.cat([
            active[..., :2]/1024, hp_fraction,
            self.item(self.ids(active[..., 2])),
            self.ability(self.ids(active[..., 3], 256)),
            obs["v2_active_stats"]/1024, obs["v2_boosts"]/6,
            obs["v2_volatile_bits"],
        ], -1)
        features.append((values * obs["active"][..., :1]).flatten(1))

        if self.include_moves:
            moves = obs["v2_move_details"]
            category = torch.nn.functional.one_hot(self.ids(moves[..., 3], 3), 3).float()
            values = torch.cat([moves[..., :1]/250, moves[..., 1:2]/100,
                                moves[..., 2:3], category, moves[..., 4:5]/7,
                                self.types(self.ids(moves[..., 5], 32))], -1)
            move_ids = torch.stack([obs["moves"][..., 0], obs["opponent_moves"][..., 0]], 1)
            mask = (move_ids != 0).unsqueeze(-1) * p[..., :1].unsqueeze(-2)
            features.append((values * mask).flatten(1))

        weather = obs["v2_weather"]
        category = torch.nn.functional.one_hot(self.ids(weather[..., 0], 5), 5).float()
        features.append(torch.cat([category, weather[..., 1:2]/8, weather[..., 2:3]], -1))
        return features

    def forward(self, observations):
        return self.network(self._flatten(observations))
