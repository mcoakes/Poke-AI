"""Bounded diagnostic logs; model files and root training metrics are never rotated."""
from collections import deque
import json
import math
from pathlib import Path
import shutil


def log_settings(args):
    """Shared defaults for thread/process simulator workers and older callers."""
    maximum = float(getattr(args, "log_max_mb", 10))
    backups = int(getattr(args, "log_backups", 2))
    episodes = int(getattr(args, "keep_episode_logs", 2))
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError("--log-max-mb must be finite and >= 0 (0 means unlimited)")
    if backups < 0:
        raise ValueError("--log-backups must be >= 0")
    if episodes < -1:
        raise ValueError("--keep-episode-logs must be >= -1 (-1 means unlimited)")
    return dict(max_bytes=max(1, int(maximum * 1024 * 1024)) if maximum else 0,
                backups=backups), episodes


class RotatingLog:
    """One writer per path. Retain current output plus a bounded number of backups.

    Append calls are kept together unless a single entry exceeds the cap. Such a
    text entry retains its tail, while an oversized JSONL entry is replaced with
    a JSON diagnostic so surviving JSONL files remain parseable.
    """
    def __init__(self, path, *, max_bytes=10 * 1024 * 1024, backups=2):
        self.path = Path(path)
        self.max_bytes, self.backups = max_bytes, backups
        if max_bytes < 0 or backups < 0:
            raise ValueError("log size and backup count must be nonnegative")

    def _rotate(self):
        if self.path.is_symlink():
            raise ValueError(f"refusing to rotate symlink: {self.path}")
        if self.backups == 0:
            self.path.unlink(missing_ok=True)
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
        oldest.unlink(missing_ok=True)
        for index in range(self.backups - 1, 0, -1):
            previous = self.path.with_name(f"{self.path.name}.{index}")
            if previous.is_symlink():
                raise ValueError(f"refusing to rotate symlink: {previous}")
            if previous.exists():
                previous.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def append(self, text):
        if not text:
            return
        data = text.encode("utf-8")
        if self.max_bytes and len(data) > self.max_bytes:
            if self.path.suffix == ".jsonl":
                data = (json.dumps(dict(event="log_entry_omitted", bytes=len(data),
                                        reason="entry exceeds log size limit")) + "\n").encode()
                if len(data) > self.max_bytes:
                    data = b"{}\n" if self.max_bytes >= 3 else b""
            else:
                notice = b"[oversized log entry; only its tail is retained]\n"
                room = max(0, self.max_bytes - len(notice))
                if room:
                    data = notice + data[-room:].decode("utf-8", errors="ignore").encode("utf-8")
                else:
                    data = data[-self.max_bytes:].decode("utf-8", errors="ignore").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError(f"refusing to write log through symlink: {self.path}")
        if self.max_bytes and self.path.exists() and self.path.stat().st_size + len(data) > self.max_bytes:
            self._rotate()
        with self.path.open("ab") as stream:
            stream.write(data)


class EpisodeLogs:
    """Retain only the latest N completed episode directories created by this owner.

    Active or failed episodes are not eligible for automatic removal. Callers
    may pass keep=-1 for unbounded debugging, or keep=0 to disable episode logs.
    """
    def __init__(self, directory, *, enabled=False, keep=2):
        if keep < -1:
            raise ValueError("episode log retention must be >= -1")
        self.directory = Path(directory)
        self.enabled, self.keep = bool(enabled) and keep != 0, keep
        self.completed = deque()
        self.owned = set()

    def start(self, episode):
        path = self.directory / f"episode-{episode:06d}"
        if self.enabled:
            path.mkdir(parents=True, exist_ok=False)
            self.owned.add(path)
        return path

    def complete(self, path):
        if not self.enabled:
            return []
        path = Path(path)
        if path not in self.owned or path.parent != self.directory or path.is_symlink():
            raise ValueError(f"episode log directory is not owned by this worker: {path}")
        if path in self.completed:
            return []
        (path / ".completed.json").write_text('{"completed": true}\n', encoding="utf-8")
        self.completed.append(path)
        removed = []
        while self.keep >= 0 and len(self.completed) > self.keep:
            old = self.completed[0]
            if old.is_symlink() or old.parent != self.directory or old not in self.owned:
                raise ValueError(f"refusing to remove unowned episode logs: {old}")
            shutil.rmtree(old)
            self.completed.popleft()
            self.owned.remove(old)
            removed.append(old)
        return removed
