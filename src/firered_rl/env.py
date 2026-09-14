from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import gymnasium as gym
import numpy as np

from firered_rl.actions import ACTION_COUNT
from firered_rl.backends.base import EmulatorBackend
from firered_rl.backends.mock import MockBattleBackend
from firered_rl.reward import RewardConfig, battle_reward


class FireRedBattleEnv(gym.Env[np.ndarray, int]):
    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        backend_factory: Callable[[], EmulatorBackend] = MockBattleBackend,
        frame_skip: int = 4,
        max_episode_steps: int = 5000,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if frame_skip < 1 or max_episode_steps < 1:
            raise ValueError("frame_skip and max_episode_steps must be positive")
        self.backend = backend_factory()
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)
        self._state = None
        self._steps = 0
        self._done = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._steps = 0
        self._done = False
        self._state = self.backend.reset(seed)
        return self._state.as_array(), {"battle_state": self._state,
            "inspection": getattr(self.backend, "last_inspection", None)}

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        if self._done:
            raise RuntimeError("episode ended; call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        previous = self._state
        self._state = self.backend.step(int(action), self.frame_skip)
        self._steps += 1
        reward = battle_reward(previous, self._state, self.reward_config)
        terminated = self._state.terminal or self._state.outcome != 0
        truncated = self._steps >= self.max_episode_steps and not terminated
        self._done = terminated or truncated
        return self._state.as_array(), reward, terminated, truncated, {
            "battle_state": self._state,
            "frames_advanced": self.frame_skip,
            "outcome_reason": self._state.outcome_reason,
            "inspection": getattr(self.backend, "last_inspection", None),
        }

    def render(self):
        return self.backend.render()

    def close(self) -> None:
        self.backend.close()
