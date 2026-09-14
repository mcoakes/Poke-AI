import json
import pytest

import firered_rl.link_connect as module


@pytest.fixture
def bridge_pair(monkeypatch):
    instances = []
    events = []
    class Bridge:
        fail_start = False
        bad_evidence = False
        disconnect = False
        battle_support = True
        bad_battle = False
        battle_menu_missing = False
        def __init__(self, host, port):
            self.index = port-55355
            self.cancelled = self.closed = False
            self.verifications = 0
            self.started = self.continued = False
            self.probe_polled = False
            self.battle_started = False
            instances.append(self)
        def status(self):
            return dict(auto_connect_supported=True, native_link_session=True,
                        battle_start_supported=self.battle_support,
                        staged_connect_supported=True, link_session_id="session", link_player=self.index+1)
        def link_connect(self, hold_at_service=False):
            assert hold_at_service
            if self.index == 1:
                assert instances[0].probe_polled and not instances[0].continued
            events.append((self.index, "start"))
            if self.fail_start and self.index == 1:
                raise RuntimeError("wrong counter")
            self.started = True
        def link_connect_continue(self):
            assert all(b.probe_polled for b in instances)
            self.continued = True
            events.append((self.index, "continue"))
        def link_connect_status(self):
            if self.disconnect:
                raise RuntimeError("mGBA bridge disconnected")
            self.probe_polled = True
            if self.battle_started:
                ready = not (self.battle_menu_missing and self.index == 1)
                return dict(phase="battle_ready", active=False, diagnostics=dict(command_ready=ready))
            return dict(phase="complete" if self.continued else "waiting_for_partner_probe",
                        active=not self.continued, probe_complete=True)
        def link_status(self):
            self.verifications += 1
            return dict(connection_established=not self.bad_evidence, player_count=2,
                        link_battle=self.battle_started and not self.bad_battle,
                        received_remote_players=True, local_id=self.index, link_type=0x2233,
                        players=[dict(id=0, trainer_id=1), dict(id=1, trainer_id=2)])
        def link_battle_start(self):
            assert all(b.verifications >= 3 for b in instances)
            self.battle_started = True
            events.append((self.index, "battle_start"))
        def link_connect_cancel(self):
            self.cancelled = True
        def close(self):
            self.closed = True
    monkeypatch.setattr(module, "MGBABridgeBackend", Bridge)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    Bridge.events = events
    return Bridge, instances


def test_requires_three_fresh_handshakes_and_closes_owners(bridge_pair, tmp_path):
    Bridge, instances = bridge_pair
    path = tmp_path / "events.jsonl"
    result = module.connect_pair([55355, 55356], "session", path, announce=lambda _: None)
    assert result["status"] == "connected"
    assert all(b.closed and not b.cancelled and b.verifications == 3 for b in instances)
    assert json.loads(path.read_text().splitlines()[-1])["event"] == "connected"
    assert Bridge.events[:2] == [(0, "start"), (1, "start")]


def test_partial_start_cancels_both_owners(bridge_pair, tmp_path):
    Bridge, instances = bridge_pair
    Bridge.fail_start = True
    with pytest.raises(RuntimeError, match="wrong counter"):
        module.connect_pair([55355, 55356], "session", tmp_path / "events.jsonl")
    assert all(b.cancelled and b.closed for b in instances)


def test_complete_without_handshake_times_out_and_cancels(bridge_pair, monkeypatch, tmp_path):
    Bridge, instances = bridge_pair
    Bridge.bad_evidence = True
    clock = iter(range(100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        module.connect_pair([55355, 55356], "session", tmp_path / "events.jsonl", timeout=12)
    assert all(b.cancelled and b.closed for b in instances)


@pytest.mark.parametrize("announce", [None, lambda _: None])
def test_disconnect_reports_port_operation_and_process_before_cleanup(bridge_pair, tmp_path, announce):
    Bridge, instances = bridge_pair
    Bridge.disconnect = True
    log = tmp_path / "events.jsonl"
    native = dict(state="exited", pid=123, returncode=-6, signal="SIGABRT")
    with pytest.raises(RuntimeError, match="port 55355 link_connect_status: mGBA bridge disconnected"):
        module.connect_pair([55355, 55356], "session", log,
                            process_state=lambda: native, announce=announce)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[-1]["process"] == native
    assert records[-2]["event"] == "request_failed"
    assert instances[0].cancelled and not instances[1].started
    assert all(b.closed for b in instances)


@pytest.mark.parametrize("announce", [None, print])
def test_battle_start_follows_verified_connection_and_requires_both_menus(bridge_pair, tmp_path, announce):
    Bridge, instances = bridge_pair
    log = tmp_path / "events.jsonl"
    result = module.connect_pair([55355, 55356], "session", log, start_battle=True, announce=announce)
    assert result["status"] == "battle_ready" and result["location"] == "battle"
    assert all(b.battle_started and b.verifications == 6 and b.closed for b in instances)
    assert json.loads(log.read_text().splitlines()[-1])["event"] == "battle_ready"


def test_old_adapter_rejected_before_automatic_movement(bridge_pair, tmp_path):
    Bridge, instances = bridge_pair
    Bridge.battle_support = False
    with pytest.raises(RuntimeError, match="0.10.0"):
        module.connect_pair([55355, 55356], "session", tmp_path / "events.jsonl", start_battle=True)
    assert all(not b.started and b.closed for b in instances)


def test_menu_ready_without_link_battle_evidence_is_not_success(bridge_pair, monkeypatch, tmp_path):
    Bridge, instances = bridge_pair
    Bridge.bad_battle = True
    clock = iter(range(100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        module.connect_pair([55355, 55356], "session", tmp_path / "events.jsonl", timeout=15, start_battle=True)
    assert all(b.battle_started and b.cancelled and b.closed for b in instances)


def test_one_missing_command_menu_prevents_pair_readiness(bridge_pair, monkeypatch, tmp_path):
    Bridge, instances = bridge_pair
    Bridge.battle_menu_missing = True
    clock = iter(range(100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        module.connect_pair([55355, 55356], "session", tmp_path / "events.jsonl", timeout=15, start_battle=True)
    assert all(b.verifications == 3 and b.cancelled and b.closed for b in instances)
