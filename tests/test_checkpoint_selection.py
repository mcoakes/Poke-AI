import os
from pathlib import Path

import pytest

from firered_rl.checkpoints import resolve_checkpoint, resolve_checkpoints


def test_latest_selects_newest_finished_save_not_largest_step_or_temporary(tmp_path):
    root = tmp_path / ".local/run"
    old = root / "ppo-old/checkpoints/ppo_999999999.zip"
    newest = root / "ppo-new/checkpoints/latest.zip"
    temp = root / "ppo-new/checkpoints/latest.tmp.zip"
    copy = root / "play-ui/checkpoint.zip"
    for i, path in enumerate((old, newest, temp, copy)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"policy")
        os.utime(path, (100+i, 100+i))
    assert resolve_checkpoint("latest", tmp_path) == newest
    assert resolve_checkpoint("later", tmp_path) == newest
    assert resolve_checkpoints([None, "latest", "later", old], tmp_path) == [None, newest, newest, old]
    assert resolve_checkpoint(old, tmp_path) == old


def test_latest_fails_clearly_and_accepts_custom_checkpoint_root(tmp_path):
    with pytest.raises(ValueError, match="no completed PPO"):
        resolve_checkpoint("latest", tmp_path)
    path = tmp_path / "models/final.zip"
    path.parent.mkdir()
    path.write_bytes(b"model")
    assert resolve_checkpoint("latest", tmp_path, search_root=path.parent) == path


def test_gui_latest_is_resolved_again_for_each_match(tmp_path, monkeypatch):
    from firered_rl.play_web import MatchJobs, PlaySettings, TeamCatalog
    root = Path(__file__).resolve().parents[1]
    first = tmp_path / ".local/run/ppo-first/checkpoints/latest.zip"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    settings = PlaySettings(project=tmp_path, rom=tmp_path / "FireRed.gba", source=tmp_path / "FireRed.sav",
                            emulator=tmp_path / "mgba", checkpoint=Path("latest"))
    jobs = MatchJobs(settings, TeamCatalog(root))
    seen = []
    monkeypatch.setattr(jobs, "_work", lambda run, teams, checkpoint, *rest: seen.append(checkpoint))
    assert jobs.models()[0]["id"] == "latest"
    payload = dict(teams=[jobs.catalog.sample("casual", 1)["team"]]*2, checkpoint="latest")
    try:
        jobs.start(payload)
        jobs.thread.join(2)
        assert seen == [first]
        second = tmp_path / ".local/run/ppo-second/checkpoints/latest.zip"
        second.parent.mkdir(parents=True)
        second.write_bytes(b"second")
        os.utime(second, (first.stat().st_mtime+10, first.stat().st_mtime+10))
        jobs.start(payload)
        jobs.thread.join(2)
        assert seen == [first, second]
    finally:
        jobs.shutdown()
