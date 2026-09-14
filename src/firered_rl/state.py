from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BattleState:
    player_hp: int
    player_max_hp: int
    opponent_hp: int
    opponent_max_hp: int
    player_active: int = 0
    opponent_active: int = 0
    player_fainted: int = 0
    opponent_fainted: int = 0
    menu_state: int = 0
    turn: int = 0
    outcome: int = 0  # -1 loss, 0 running/draw, 1 win
    terminal: bool = False
    outcome_reason: str = "ongoing"

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.player_hp / max(1, self.player_max_hp),
                self.opponent_hp / max(1, self.opponent_max_hp),
                self.player_active / 5.0,
                self.opponent_active / 5.0,
                self.player_fainted / 6.0,
                self.opponent_fainted / 6.0,
                self.menu_state / 15.0,
                min(self.turn, 255) / 255.0,
            ],
            dtype=np.float32,
        )
