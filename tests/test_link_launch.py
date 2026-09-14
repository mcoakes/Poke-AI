import argparse
import json
from pathlib import Path
import pytest

from firered_rl.launch_options import add_speed_argument, speed_options


def test_speed_rejects_invalid_targets_and_defines_default():
    parser = argparse.ArgumentParser()
    add_speed_argument(parser)
    assert parser.parse_args([]).speed == 1.0
    for value in ("0", "-1", "nan", "inf", "1e100", "fast"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--speed", value])
    assert "audioSync=0" in speed_options(parser.parse_args(["--speed", "unlimited"]).speed)


def test_pairs_have_isolated_manifests_and_pass_speed_to_the_process(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "mgba"))
    import launch_link
    calls, stopped = [], []
    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 100+len(calls)
            calls.append((command, kwargs))
    monkeypatch.setattr(launch_link.subprocess, "Popen", Process)
    monkeypatch.setattr(launch_link, "stop_processes", lambda processes: stopped.extend(processes))
    rom = tmp_path / "source.gba"
    rom.write_bytes(b"synthetic ROM placeholder")
    states = [tmp_path / "fire.ss1", tmp_path / "water.ss1"]
    sessions = []
    try:
        for index, speed in enumerate((4.0, "unlimited")):
            session = launch_link.LinkProcess(root, tmp_path / f"pair{index}",
                [55355+index*2, 55356+index*2], states)
            sessions.append(session)
            session.start(rom, tmp_path / "mgba-qt", speed, headless=index == 1, visible_player=index+1)
        manifests = [json.loads((s.run / "manifest.json").read_text()) for s in sessions]
        assert len({p["rom"] for pair in manifests for p in pair}) == 4
        assert sessions[0].session_id != sessions[1].session_id
        for index, players in enumerate(manifests):
            assert calls[index][1]["env"]["MGBA_VISIBLE_LINK_ROM"] == players[index]["rom"]
        for session, players in zip(sessions, manifests):
            for i, player in enumerate(players):
                assert player["port"] == session.ports[i]
                assert player["state"] == str(states[i])
                assert session.session_id in Path(player["script"]).read_text()
        assert calls[0][0][0] == str(tmp_path / "mgba-qt")
        assert any(option.startswith("fpsTarget=238.") for option in calls[0][0])
        assert calls[1][0][:2] == ["xvfb-run", "-a"]
        assert "audioSync=0" in calls[1][0]
        assert calls[0][1]["env"]["MGBA_LINK_MANIFEST"] != calls[1][1]["env"]["MGBA_LINK_MANIFEST"]
    finally:
        for session in sessions:
            session.close()
    assert len(stopped) == 2 and all(s.log.closed for s in sessions)


def test_failure_snapshot_distinguishes_running_from_native_signal(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "mgba"))
    from launch_link import LinkProcess
    session = LinkProcess(root, tmp_path, (55355, 55356), (Path("a.ss1"), Path("b.ss1")))
    class Process:
        pid = 123
        returncode = None
        def poll(self):
            return self.returncode
    session.process = Process()
    assert session.process_status()["state"] == "running"
    session.process.returncode = -6
    snapshot = session.capture_failure("socket closed")
    assert snapshot["process"]["returncode"] == -6
    assert snapshot["process"]["signal"] == "SIGABRT"
    assert json.loads((tmp_path / "failure.json").read_text()) == snapshot


def test_runtime_speed_requires_both_fresh_acknowledgements(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "mgba"))
    import launch_link
    session = launch_link.LinkProcess(root, tmp_path, (55355, 55356), (Path("a"), Path("b")))
    class Process:
        def poll(self): return None
    session.process = Process()
    for i in (1, 2):
        (tmp_path / f"player{i}").mkdir()
        (tmp_path / f"player{i}/speed-applied.json").write_text('{"id":"stale","speed":1}')
    calls = []
    def acknowledge(_):
        request = (tmp_path / "speed.json").read_text()
        calls.append(request)
        (tmp_path / f"player{min(len(calls),2)}/speed-applied.json").write_text(request)
    monkeypatch.setattr(launch_link.time, "sleep", acknowledge)
    session.set_speed(1)
    assert len(calls) == 2
    assert json.loads(calls[0])["speed"] == 1
