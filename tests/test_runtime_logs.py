"""Retention bounds must not remove active diagnostics, models or summaries."""
import json
from pathlib import Path
import shutil
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from firered_rl.runtime_logs import EpisodeLogs, RotatingLog, log_settings


def test_rotation_retains_recent_entries_and_bounds_files(tmp_path):
    path = tmp_path / "battle.log"
    log = RotatingLog(path, max_bytes=12, backups=2)
    for index in range(8):
        log.append(f"event {index}\n")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["battle.log", "battle.log.1", "battle.log.2"]
    assert path.read_text() == "event 7\n"
    assert path.with_name("battle.log.1").read_text() == "event 6\n"
    assert path.with_name("battle.log.2").read_text() == "event 5\n"
    assert all(p.stat().st_size <= 12 for p in tmp_path.iterdir())


def test_no_backups_unlimited_and_oversized_entries(tmp_path):
    path = tmp_path / "battle.log"
    log = RotatingLog(path, max_bytes=128, backups=0)
    for _ in range(4):
        log.append("Pokémon " * 100)
    assert path.stat().st_size <= 128
    assert "tail is retained" in path.read_text()
    assert list(tmp_path.iterdir()) == [path]
    path = tmp_path / "gym-steps.jsonl"
    RotatingLog(path, max_bytes=128, backups=1).append(json.dumps(dict(reports=["x" * 200])) + "\n")
    assert json.loads(path.read_text())["event"] == "log_entry_omitted"
    unlimited = tmp_path / "unlimited.log"
    RotatingLog(unlimited, max_bytes=0).append("x" * 10000)
    assert unlimited.stat().st_size == 10000


def test_log_symlink_target_is_untouched(tmp_path):
    protected = tmp_path / "checkpoint.zip"
    protected.write_bytes(b"checkpoint")
    link = tmp_path / "battle.log"
    link.symlink_to(protected)
    with pytest.raises(ValueError, match="symlink"):
        RotatingLog(link, max_bytes=12).append("changed")
    assert protected.read_bytes() == b"checkpoint"


def test_only_owned_completed_episodes_are_pruned(tmp_path):
    owner = EpisodeLogs(tmp_path, enabled=True, keep=2)
    external = tmp_path / "episode-999999"
    external.mkdir()
    summary = tmp_path / "summary.json"
    summary.write_text('{"battles": 123}')
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "final.zip").write_bytes(b"checkpoint")
    first, active, second, third = (owner.start(n) for n in range(1, 5))
    owner.complete(first)
    owner.complete(second)
    removed = owner.complete(third)
    assert removed == [first] and not first.exists()
    assert active.exists() and external.exists()
    assert second.exists() and third.exists()
    assert json.loads((third / ".completed.json").read_text())["completed"]
    assert summary.read_text() == '{"battles": 123}'
    assert (checkpoints / "final.zip").read_bytes() == b"checkpoint"
    with pytest.raises(ValueError, match="not owned"):
        owner.complete(external)


@pytest.mark.parametrize("keep", [0, 2, -1])
def test_episode_flags_and_defaults(tmp_path, keep):
    assert log_settings(SimpleNamespace()) == (dict(max_bytes=10*1024*1024, backups=2), 2)
    owner = EpisodeLogs(tmp_path, enabled=True, keep=keep)
    for index in range(5):
        path = owner.start(index)
        owner.complete(path)
    assert len(list(tmp_path.glob("episode-*"))) == {0: 0, 2: 2, -1: 5}[keep]
    disabled = EpisodeLogs(tmp_path / "disabled")
    disabled.start(1)
    assert not (tmp_path / "disabled").exists()


@pytest.mark.parametrize("values", [dict(log_max_mb=-1), dict(log_max_mb=float("nan")),
                                  dict(log_max_mb=float("inf")), dict(log_backups=-1),
                                  dict(keep_episode_logs=-2)])
def test_bad_limits_fail(values):
    with pytest.raises(ValueError):
        log_settings(SimpleNamespace(**values))


def test_stderr_is_drained_and_rotated(tmp_path):
    pytest.importorskip("gymnasium")
    from firered_rl.showdown_env import ShowdownFactory
    runner = tmp_path / "sim/showdown/runner.cjs"
    runner.parent.mkdir(parents=True)
    # A tiny pipe-protocol worker exercises stderr with no actual battle dependency.
    runner.write_text("import sys\nfor line in sys.stdin:\n"
                      " sys.stderr.write('worker diagnostic\\n' * 10000)\n"
                      " sys.stderr.flush()\n"
                      " print('{\"ok\": true}', flush=True)\n")
    args = SimpleNamespace(seed=1, node=sys.executable, command_timeout=10,
                           log_max_mb=.001, log_backups=1)
    factory = ShowdownFactory(tmp_path, tmp_path / "pair", [1, 2], lambda: (), args, Event())
    try:
        assert factory.request("ping")["ok"]
    finally:
        factory.close()
    logs = list((tmp_path / "pair").glob("showdown-stderr.log*"))
    assert 1 <= len(logs) <= 2
    assert all(p.stat().st_size <= int(.001 * 1024 * 1024) for p in logs)


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_real_showdown_trace_retention_and_active_episode(tmp_path, mode):
    pytest.importorskip("gymnasium")
    from firered_rl.showdown_env import ShowdownBattleSession, ShowdownFactory
    from firered_rl.showdown_process import ProcessShowdownSession
    from firered_rl.linked_reward import LinkedRewardConfig
    root = Path(__file__).resolve().parents[1]
    if not shutil.which("node") or not (root / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file():
        pytest.skip("install the pinned simulator")
    run = tmp_path / "pair"
    teams = []
    for number, (species, level, move) in enumerate([("Magikarp", 1, "Splash"), ("Blastoise", 50, "Surf")]):
        path = tmp_path / f"team{number}.json"
        path.write_text(json.dumps(dict(name=f"team{number}", party=[dict(species=species, level=level, moves=[move])])))
        teams.append(path)
    args = SimpleNamespace(seed=1, node="node", command_timeout=10, max_steps=10,
                           readable_log=True, battle_log=True, step_logs=True,
                           log_max_mb=.002, log_backups=1, keep_episode_logs=2)
    cancel = Event()
    if mode == "thread":
        factory = ShowdownFactory(root, run, [1, 2], lambda: teams, args, cancel)
        session = ShowdownBattleSession(factory, readable_log=run / "battle-readable.log", battle_log=run / "battle.log")
    else:
        session = ProcessShowdownSession(root, run, [1, 2], lambda: teams, args, cancel, LinkedRewardConfig())
    try:
        for _ in range(4):
            session.reset_pair([123, 456])
            result = session.step_pair([0, 0])
            assert result[0][2] and result[1][2]
            assert result[1][4]["outcome"] == "won"
        session.reset_pair([123, 456])
        assert sorted(p.name for p in run.glob("episode-*")) == [
            "episode-000003", "episode-000004", "episode-000005"]
        assert not (run / "episode-000005/.completed.json").exists()
    finally:
        session.close()
    for log in run.rglob("*.log*"):
        assert log.stat().st_size <= int(.002 * 1024 * 1024)
    for log in run.rglob("gym-steps.jsonl*"):
        assert log.stat().st_size <= int(.002 * 1024 * 1024)
        for line in log.read_text().splitlines():
            json.loads(line)
    assert (run / "battle-readable.log.1").exists()
    assert (run / "battle.log").exists()
