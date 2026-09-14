from __future__ import annotations

import json
import socket
from dataclasses import fields
from typing import Any

import numpy as np

from firered_rl.state import BattleState


class MGBAProtocolError(RuntimeError):
    pass


class MGBABridgeBackend:
    """Client for a Lua adapter running inside one mGBA instance."""

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self._socket: socket.socket | None = None
        self._reader: Any = None

    def _connect(self) -> None:
        if self._socket is not None:
            return
        self._socket = socket.create_connection((self.host, self.port), self.timeout)
        self._reader = self._socket.makefile("rb")

    def _request(self, command: str, **payload: Any) -> dict[str, Any]:
        self._connect()
        assert self._socket is not None
        message = json.dumps({"command": command, **payload}, separators=(",", ":")) + "\n"
        self._socket.sendall(message.encode())
        response = self._reader.readline()
        if not response:
            raise MGBAProtocolError("mGBA bridge disconnected")
        decoded = json.loads(response)
        if not decoded.get("ok", False):
            raise MGBAProtocolError(str(decoded.get("error", "unknown bridge error")))
        return decoded

    @staticmethod
    def _state(payload: dict[str, Any]) -> BattleState:
        if payload.get("state_ready") is False:
            raise MGBAProtocolError("battle memory is not implemented; use reset_response/step_response")
        source = payload["state"]
        return BattleState(**{field.name: source[field.name] for field in fields(BattleState) if field.name in source})

    def reset(self, seed: int | None = None) -> BattleState:
        return self._state(self.reset_response(seed))

    def reset_response(self, seed: int | None = None) -> dict[str, Any]:
        """Reload the configured state; seed does not reseed FireRed's RNG."""
        return self._request("reset", seed=seed)

    def ping(self) -> dict[str, Any]:
        """Check adapter connectivity without requiring a memory profile."""
        return self._request("ping")

    def status(self) -> dict[str, Any]:
        """Return adapter and ROM identity information."""
        return self._request("status")

    def inspect(self) -> dict[str, Any]:
        """Read party/bag diagnostics without changing game input or memory."""
        return self._request("inspect")

    def link_status(self) -> dict[str, Any]:
        return self._request("link_status")

    def link_connect(self, hold_at_service: bool = False) -> dict[str, Any]:
        return self._request("link_connect", hold_at_service=hold_at_service)

    def link_connect_continue(self) -> dict[str, Any]:
        return self._request("link_connect_continue")

    def link_connect_status(self) -> dict[str, Any]:
        return self._request("link_connect_status")

    def link_battle_start(self) -> dict[str, Any]:
        return self._request("link_battle_start")

    def link_connect_cancel(self) -> dict[str, Any]:
        return self._request("link_connect_cancel")

    def link_turn_prepare(self, request_id, slot, expected_turn, kind="switch",
                          replacement=False, skip_move_menu=False, human_opponent=False, battle_log=False) -> dict[str, Any]:
        return self._request("link_turn_prepare", kind=kind, request_id=request_id,
                             slot=slot, expected_turn=expected_turn, replacement=replacement,
                             skip_move_menu=skip_move_menu, human_opponent=human_opponent, battle_log=battle_log)

    def link_turn_commit(self, request_id) -> dict[str, Any]:
        return self._request("link_turn_commit", request_id=request_id)

    def link_turn_status(self, events_after=0) -> dict[str, Any]:
        return self._request("link_turn_status", events_after=events_after)

    def link_turn_cancel(self) -> dict[str, Any]:
        return self._request("link_turn_cancel")

    def step(self, action: int, frames: int) -> BattleState:
        return self._state(self.step_response(action, frames))

    def step_response(self, action: int, frames: int) -> dict[str, Any]:
        """Return the complete step response, including frame diagnostics."""
        return self._request("step", action=int(action), frames=int(frames))

    def render(self) -> np.ndarray | None:
        # Pixel transport is optional; RAM observations avoid this training overhead.
        return None

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._request("close")
            except (OSError, MGBAProtocolError, json.JSONDecodeError):
                # Cleanup must not mask an earlier timeout or protocol failure.
                pass
            finally:
                if self._reader is not None:
                    self._reader.close()
                self._socket.close()
                self._socket = None
                self._reader = None
