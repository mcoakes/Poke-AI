"""In-process FIFO matchmaking. Each match owns a separate native cable session."""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class PoolMember:
    agent_id: str
    port: int
    state: Path


class MatchPool:
    """Join once, then take disjoint pairs in arrival order.

    Pair before launching: mGBA attaches cores to their process's cable at startup.
    This queue is a local API, not a network service or a battle-turn scheduler.
    """
    def __init__(self):
        self._waiting = deque()
        self._ids = set()
        self._ports = set()
        self._lock = Lock()

    def join(self, member: PoolMember):
        with self._lock:
            if not member.agent_id or member.agent_id in self._ids:
                raise ValueError("agent IDs must be nonempty and unique")
            if not 1 <= member.port <= 65535 or member.port in self._ports:
                raise ValueError("adapter ports must be valid and unique")
            self._ids.add(member.agent_id)
            self._ports.add(member.port)
            self._waiting.append(member)

    def take_pair(self):
        with self._lock:
            if len(self._waiting) < 2:
                return None
            return self._waiting.popleft(), self._waiting.popleft()

    @property
    def waiting(self):
        with self._lock:
            return tuple(self._waiting)
