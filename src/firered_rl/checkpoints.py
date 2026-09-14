"""Resolve checkpoint selectors once, before starting a run or match."""
from pathlib import Path
import re


def is_latest(value):
    return str(value).lower() in ("latest", "later")


def resolve_checkpoint(value, project, *, search_root=None):
    if not is_latest(value):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"checkpoint not found: {path}")
        return path
    root = Path(search_root or Path(project) / ".local/run").expanduser().resolve()
    candidates = []
    for path in root.rglob("*.zip"):
        # Exclude generated game files, temporary saves, and copied play snapshots.
        if path.parent.name != "checkpoints" and path.parent != root:
            continue
        if not re.fullmatch(r"latest\.zip|final\.zip|ppo_\d+\.zip", path.name):
            continue
        try:
            if path.is_file():
                candidates.append((path.stat().st_mtime_ns, str(path), path))
        except FileNotFoundError:
            continue  # A trainer can atomically replace latest.zip during discovery.
    if not candidates:
        raise ValueError(f"no completed PPO checkpoints under {root}; provide a checkpoint path")
    return max(candidates)[2].resolve()


def resolve_checkpoints(values, project, *, search_root=None):
    """Two `latest` selectors in one command must select the same file."""
    latest = None
    result = []
    for value in values:
        if value is None:
            result.append(None)
        elif is_latest(value):
            if latest is None:
                latest = resolve_checkpoint(value, project, search_root=search_root)
            result.append(latest)
        else:
            result.append(resolve_checkpoint(value, project, search_root=search_root))
    return result
