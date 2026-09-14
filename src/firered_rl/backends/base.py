from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from firered_rl.state import BattleState


@runtime_checkable
class EmulatorBackend(Protocol):
    def reset(self, seed: int | None = None) -> BattleState: ...

    def step(self, action: int, frames: int) -> BattleState: ...

    def render(self) -> np.ndarray | None: ...

    def close(self) -> None: ...

