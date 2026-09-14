from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import firered_rl.linked_env as module
from firered_rl.link_inspection import terminal_inspections

ROOT = Path(__file__).resolve().parents[1]


def test_replacement_boundary_waits_for_peer_to_finish_submitting():
    states = [dict(phase="replacement_ready"), dict(phase="select_shift")]
    reports = [dict(battle=dict(menus=[dict(state="party")])),
               dict(battle=dict(menus=[dict(state="party")]))]
    payloads = [dict(kind="wait"), dict(kind="switch")]
    assert not module.actions_submitted(states, reports, payloads)
    states[1]["phase"] = "resolving"
    assert not module.actions_submitted(states, reports, payloads)  # A not consumed yet
    reports[1]["battle"]["menus"][0]["state"] = "busy_or_unknown"
    assert module.actions_submitted(states, reports, payloads)


def test_early_wait_boundary_does_not_cancel_peer_navigation(runtime):
    launch, _, _, bridges, raw = runtime
    session = module.LinkedBattleSession(launch, timeout=1)
    try:
        session.reset_pair()
        module.local_mon(session.reports[0])["current_hp"] = 0
        session.reports[0]["battle"]["menus"][0]["state"] = "busy_or_unknown"
        polls = [0, 0]
        def status(index):
            polls[index] += 1
            if index == 0:
                return dict(request_id=1, active=False, phase="replacement_ready")
            if polls[index] < 3:
                return dict(request_id=1, active=True, phase="select_shift")
            raw[0]["battle"]["menus"][0].update(state="party", submenu="choose_pokemon")
            raw[0]["battle"]["battlers"][0]["current_hp"] = 0
            raw[1]["battle"]["menus"][0]["state"] = "busy_or_unknown"
            return dict(request_id=1, active=True, phase="resolving")
        for i, bridge in enumerate(bridges):
            bridge.link_turn_status = lambda index=i: status(index)
        result = session.step_pair([10, 5])
        assert polls[1] >= 3
        assert result[0][4]["reason"] == "forced_replacement"
        assert result[1][0]["action_mask"][10] == 1
        assert not result[0][2] and not result[1][2]
    finally:
        session.close()


def test_readable_log_is_live_and_continues_across_episodes(runtime, tmp_path):
    launch, _, _, bridges, _ = runtime
    path = tmp_path / "pair-1" / "battle-readable.log"
    session = module.LinkedBattleSession(launch, timeout=1, readable_log=path)
    try:
        session.reset_pair()
        reset_text = path.read_text()
        assert "RESET" in reset_text and "PLAYER 2" in reset_text
        assert "Charizard" in reset_text and "Flamethrower" in reset_text
        assert "bench HP is not in observation" in reset_text
        original_prepare = bridges[0].link_turn_prepare
        def prepare(**payload):
            # The chosen action must be on disk before emulator execution starts.
            assert "CHOSEN ACTION: 0 = Move slot 1: Flamethrower" in path.read_text()
            return original_prepare(**payload)
        bridges[0].link_turn_prepare = prepare
        session.step_pair([0, 0])
        text = path.read_text()
        assert "COMPLETED" in text and "REWARD: -1 raw points" in text
        assert "terminated=True truncated=False" in text
        session.reset_pair()
        assert path.read_text().count("| RESET |") == 2
    finally:
        session.close()


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    initial = json.loads((ROOT/"tests/fixtures/link_master_follower.json").read_text())
    state = deepcopy(initial)
    instances, launches, closed = [], [], []
    class Bridge:
        def __init__(self, host, port):
            self.index = port-55355
            self.prepared = self.committed = False
            self.request_id = 0
            instances.append(self)
        def status(self):
            return dict(native_link_session=True, link_move_supported=True,
                        link_peer_simulation_supported=True, link_player=self.index+1, link_session_id="test")
        def link_status(self):
            return dict(connection_established=True, received_remote_players=True,
                        player_count=2, local_id=self.index, link_battle=True,
                        players=[dict(id=0),dict(id=1)], link_type=1)
        def inspect(self):
            return deepcopy(state[self.index])
        def link_turn_prepare(self, **payload):
            self.payload = payload
            self.request_id = payload["request_id"]
            self.prepared = True
            return dict(phase="prepared", request_id=self.request_id)
        def link_turn_commit(self, request_id):
            assert all(b.prepared for b in instances[-2:])
            self.committed = True
            return dict(ok=True)
        def link_turn_status(self):
            result = dict(request_id=self.request_id, active=True, phase="resolving")
            if self.index == 0 and all(b.committed for b in instances[-2:]):
                snapshot = deepcopy(state[0])
                for mon in snapshot["party"]:
                    if mon.get("valid"):
                        mon["current_hp"] = 0
                snapshot["battle"]["battlers"][0]["current_hp"] = 0
                snapshot["battle"]["player_counts"].update(remaining=0, fainted=2)
                snapshot["battle"].update(outcome_raw=2, outcome="lost", terminal=True)
                result.update(active=False, phase="battle_complete", terminal_inspection=snapshot)
            return result
        def link_turn_cancel(self):
            self.cancelled = True
        def close(self):
            self.closed = True
    monkeypatch.setattr(module, "MGBABridgeBackend", Bridge)
    def launch():
        state[:] = deepcopy(initial)
        launches.append(True)
        return SimpleNamespace(ports=[55355,55356], session_id="test", run=tmp_path,
                               close=lambda: closed.append(True))
    return launch, launches, closed, instances, state


def test_two_gym_players_reset_together_and_receive_opposite_terminal_results(runtime):
    launch, launches, closed, bridges, state = runtime
    session = module.LinkedBattleSession(launch, timeout=1)
    envs = session.players()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            for _ in range(2):
                futures = [executor.submit(e.reset, seed=1) for e in envs]
                reset = [f.result(timeout=3) for f in futures]
                for env, (obs, info) in zip(envs, reset):
                    assert env.observation_space.contains(obs) and not info["seed_applied"]
                futures = [executor.submit(e.step, 0) for e in envs]
                result = [f.result(timeout=3) for f in futures]
                assert [r[1] for r in result] == [-1, 1]
                assert all(r[2] and not r[3] for r in result)
                assert [r[4]["outcome"] for r in result] == ["lost", "won"]
                assert state[1]["battle"]["outcome_raw"] == 0  # no fabricated follower result needed
                assert all(not any(r[0]["action_mask"]) for r in result)
                assert all(e.observation_space.contains(r[0]) for e, r in zip(envs, result))
        assert len(launches) == 2 and len(closed) == 1
    finally:
        session.close()
    assert len(closed) == 2 and all(b.closed for b in bridges)


def test_forced_replacement_observation_and_partner_wait(runtime):
    _, _, _, _, raw = runtime
    reports = module.pair_inspections(raw, [55355,55356])
    mon = module.local_mon(reports[0])
    mon.update(current_hp=0, absent=True)
    reports[0]["party"][0]["current_hp"] = 0
    reports[0]["battle"]["menus"] = [dict(state="party", submenu="choose_pokemon")]
    reports[1]["battle"]["menus"] = [dict(state="busy_or_unknown")]
    first, second = [module.linked_observation(r) for r in reports]
    assert first["context"][1] and first["action_mask"][5]
    assert not any(first["action_mask"][:4])
    assert second["action_mask"][10] and sum(second["action_mask"]) == 1


@pytest.mark.parametrize("actor", [0, 1])
def test_baton_pass_yields_recipient_choice_then_resumes_same_turn(runtime, actor):
    launch, _, _, bridges, raw = runtime
    session = module.LinkedBattleSession(launch, timeout=1)
    try:
        session.reset_pair()
        # Reproduce the new episode's living, partially trapped Baton Pass user.
        battler_id = module.local_mon(session.reports[actor])["battler_id"]
        master_mon = next(m for m in raw[0]["battle"]["battlers"] if m["battler_id"] == battler_id)
        master_mon["volatile_status_raw"] = 0x2000
        raw[actor]["battle"]["menus"][0].update(
            state="party", submenu="choose_pokemon", switch_required=True)
        raw[1-actor]["battle"]["menus"][0]["state"] = "busy_or_unknown"
        for i, bridge in enumerate(bridges):
            bridge.link_turn_status = lambda index=i: dict(
                request_id=1, active=index != actor,
                phase="replacement_ready" if index == actor else "resolving")
        turn = session.reports[0]["battle"]["turn"]
        result = session.step_pair([0, 0])
        assert result[actor][4]["reason"] == "required_switch"
        assert result[actor][0]["context"][1] == 1
        assert result[actor][0]["active"][0][3] > 0
        mask = result[actor][0]["action_mask"]
        assert mask[5] and not mask[:5].any() and not mask[10]
        assert result[1-actor][0]["action_mask"].tolist() == [0]*10+[1]
        assert all(not r[2] and not r[3] for r in result)
        assert result[actor][4]["turn_end"] == turn
        target = deepcopy(session.reports[actor]["party"][1])

        def completed(index):
            for r in raw:
                mon = next(m for m in r["battle"]["battlers"] if m["battler_id"] == battler_id)
                mon.update(party_slot=2, species_id=target["species_id"],
                           current_hp=target["current_hp"], max_hp=target["max_hp"],
                           moves=target["moves"], type_ids=target["base_type_ids"])
                r["battle"]["menus"][0].update(state="command", submenu=None, switch_required=False)
            raw[0]["battle"]["turn"] = turn+1
            return dict(request_id=2, active=False, phase="turn_complete")
        for i, bridge in enumerate(bridges):
            bridge.link_turn_status = lambda index=i: completed(index)
        actions = [10, 10]
        actions[actor] = 5
        result = session.step_pair(actions)
        assert bridges[actor].payload["kind"] == "switch"
        assert bridges[actor].payload["replacement"] is True
        assert bridges[1-actor].payload["kind"] == "wait"
        assert result[actor][0]["active"][0][8] == 2
        assert all(not r[2] and r[4]["turn_end"] == turn+1 for r in result)
    finally:
        session.close()


def test_terminal_outcome_requires_remaining_counts_to_agree(runtime):
    _, _, _, _, raw = runtime
    snapshot = deepcopy(raw[0])
    snapshot["battle"]["outcome_raw"] = 2
    with pytest.raises(ValueError, match="remaining Pokemon"):
        terminal_inspections(snapshot, raw, 0, [55355,55356])


def test_terminal_error_names_invalid_side_and_checksum_warning(runtime):
    _, _, _, _, raw = runtime
    snapshot = deepcopy(raw[0])
    snapshot["battle"]["outcome_raw"] = 2
    snapshot["battle"]["opponent_counts"].update(
        valid=False, warnings=["slot 6: checksum failure"])
    with pytest.raises(ValueError, match="opponent.*slot 6: checksum failure"):
        terminal_inspections(snapshot, raw, 0, [55355,55356])


def test_terminal_retry_waits_past_peer_turn_menu_and_rewards_once(runtime):
    launch, _, _, bridges, _ = runtime
    session = module.LinkedBattleSession(launch, timeout=1)
    try:
        session.reset_pair()
        original_status = bridges[0].link_turn_status
        polls = [0]
        def status():
            polls[0] += 1
            if polls[0] <= 2:
                return dict(request_id=1, active=True, phase="capturing_terminal")
            return original_status()
        bridges[0].link_turn_status = status
        bridges[1].link_turn_status = lambda: dict(
            request_id=1, active=False, phase="turn_complete")
        # A regular inspection at this point may already see terminal data;
        # only the master's validated snapshot should be used.
        def unexpected_inspection(*args, **kwargs):
            raise AssertionError("ordinary turn inspection during terminal retry")
        session.read_reports = unexpected_inspection
        result = session.step_pair([0, 0])
        assert polls[0] == 3
        assert [r[1] for r in result] == [-1, 1]
        assert all(r[2] and not r[3] for r in result)
        log = [json.loads(line) for line in (session.runtime.run / "gym-steps.jsonl").read_text().splitlines()]
        assert sum(entry["event"] == "completed" for entry in log) == 1
    finally:
        session.close()


def test_time_limit_preserves_nonterminal_observation_for_value_bootstrap(runtime, monkeypatch):
    launch, _, _, bridges, state = runtime
    session = module.LinkedBattleSession(launch, timeout=1, max_steps=1)
    try:
        session.reset_pair()
        def completed(bridge):
            if bridge.index == 0:
                state[0]["battle"]["turn"] += 1
            return dict(request_id=bridge.request_id, active=False, phase="turn_complete")
        monkeypatch.setattr(type(bridges[0]), "link_turn_status", completed)
        results = session.step_pair([0, 0])
        assert all(not row[2] and row[3] for row in results)
        assert all(row[0]["context"][2] == 0 and row[0]["action_mask"].any() for row in results)
    finally:
        session.close()


def test_shaped_rewards_use_actual_paired_terminal_snapshot_and_pay_once(runtime):
    from firered_rl.linked_reward import LinkedRewardConfig
    launch, _, _, _, _ = runtime
    config = LinkedRewardConfig(damage=100, knockout=500, win=5000, loss=-5000, scale=.001)
    session = module.LinkedBattleSession(launch, reward_config=config)
    try:
        session.reset_pair()
        results = session.step_pair([0, 0])
        assert results[0][1] == -5000
        assert results[1][1] > 5500
        assert results[1][4]["reward_components"]["outcome"] == 5000
        assert sum(results[1][4]["reward_components"].values()) == results[1][1]
        with pytest.raises(RuntimeError, match="reset both"):
            session.step_pair([0, 0])
    finally:
        session.close()
