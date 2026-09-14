from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym

from firered_rl.backends.base import EmulatorBackend
from firered_rl.backends.mock import MockBattleBackend
from firered_rl.env import FireRedBattleEnv


def make_vector_env(
    num_envs: int = 4,
    backend_factory: Callable[[], EmulatorBackend] = MockBattleBackend,
    *,
    asynchronous: bool = True,
    frame_skip: int = 4,
    max_episode_steps: int = 5000,
) -> gym.vector.VectorEnv:
    if not 1 <= num_envs <= 4:
        raise ValueError("the initial system supports between 1 and 4 environments")

    def make_one():
        return FireRedBattleEnv(backend_factory, frame_skip, max_episode_steps)

    env_fns = [make_one for _ in range(num_envs)]
    cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return cls(env_fns)

