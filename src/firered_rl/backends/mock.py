from __future__ import annotations

import numpy as np

from firered_rl.actions import GBAAction
from firered_rl.state import BattleState


class MockBattleBackend:
    """Deterministic backend for API tests; it is not a Pokemon simulator."""

    def __init__(self, max_turns: int = 100) -> None:
        self.max_turns = max_turns
        self._rng = np.random.default_rng()
        self._state = BattleState(100, 100, 100, 100)

    def reset(self, seed: int | None = None) -> BattleState:
        self._rng = np.random.default_rng(seed)
        self._state = BattleState(100, 100, 100, 100)
        return self._state

    def step(self, action: int, frames: int) -> BattleState:
        state = self._state
        attack = int(action) == int(GBAAction.A)
        dealt = int(self._rng.integers(8, 16)) if attack else 0
        received = int(self._rng.integers(3, 9))
        player_hp = max(0, state.player_hp - received)
        opponent_hp = max(0, state.opponent_hp - dealt)
        turn = state.turn + 1
        outcome = 1 if opponent_hp == 0 else (-1 if player_hp == 0 else 0)
        self._state = BattleState(
            player_hp,
            100,
            opponent_hp,
            100,
            player_fainted=int(player_hp == 0),
            opponent_fainted=int(opponent_hp == 0),
            turn=turn,
            outcome=outcome,
        )
        return self._state

    def render(self) -> np.ndarray:
        return np.zeros((160, 240, 3), dtype=np.uint8)

    def close(self) -> None:
        return None

