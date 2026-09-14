from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from firered_rl.backends.base import EmulatorBackend
from firered_rl.reward import RewardConfig, battle_reward
from firered_rl.state import BattleState


class ActionSource(Protocol):
    def __call__(self, observation: BattleState) -> int: ...


@dataclass(frozen=True, slots=True)
class LinkStep:
    states: tuple[BattleState, BattleState]
    rewards: tuple[float, float]
    terminated: bool


class LinkBattleSession:
    """Legacy/mock paired stepping; does not establish a native mGBA link.

    Use mgba/launch_link.py and mgba/test_link.py for the native connection test.
    Real linked turn scheduling requires a coordinator and is not implemented here.
    """

    def __init__(
        self,
        player1: EmulatorBackend,
        player2: EmulatorBackend,
        frame_quantum: int = 1,
        reward_config: RewardConfig | None = None,
    ) -> None:
        if frame_quantum < 1:
            raise ValueError("frame_quantum must be positive")
        self.backends = (player1, player2)
        self.frame_quantum = frame_quantum
        self.reward_config = reward_config or RewardConfig()
        self.states: tuple[BattleState, BattleState] | None = None

    def reset(self, seed: int | None = None) -> tuple[BattleState, BattleState]:
        second_seed = None if seed is None else seed + 1
        self.states = (self.backends[0].reset(seed), self.backends[1].reset(second_seed))
        return self.states

    def step(self, actions: tuple[int, int]) -> LinkStep:
        if self.states is None:
            raise RuntimeError("reset() must be called before step()")
        previous = self.states
        # Each bridge command advances one small frame quantum. The mGBA adapter
        # is responsible for link-I/O synchronization within that quantum.
        current = (
            self.backends[0].step(actions[0], self.frame_quantum),
            self.backends[1].step(actions[1], self.frame_quantum),
        )
        self.states = current
        rewards = tuple(
            battle_reward(old, new, self.reward_config) for old, new in zip(previous, current)
        )
        return LinkStep(current, rewards, any(state.outcome != 0 for state in current))

    def run_turn(
        self, player1_source: ActionSource, player2_source: ActionSource
    ) -> LinkStep:
        if self.states is None:
            raise RuntimeError("reset() must be called before run_turn()")
        return self.step((player1_source(self.states[0]), player2_source(self.states[1])))

    def close(self) -> None:
        for backend in self.backends:
            backend.close()
