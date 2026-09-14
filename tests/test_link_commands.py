from copy import deepcopy
import json
from pathlib import Path
import pytest

import firered_rl.link_commands as module


ROOT = Path(__file__).resolve().parents[1]


def reports():
    result = []
    for i, species in enumerate(([6, 59], [9, 131, 134, 121, 62, 130])):
        party = [dict(slot=j+1, valid=True, is_egg=False, species_id=mid,
                      current_hp=100, max_hp=100, moves=[dict(slot=1, move_id=1, current_pp=10)])
                 for j, mid in enumerate(species)]
        local = dict(party[0], side="player", party_slot=1, absent=False, type_ids=[10, 10],
                     ability_id=0, restrictions={}, volatile_status_raw=0, battler_id=i)
        enemy = dict(local, side="opponent", species_id=9 if i == 0 else 6, battler_id=1-i)
        result.append(dict(frame=100, party=party, counts_valid=True,
                           battle=dict(active=True, type_flags=14 if i == 0 else 10, outcome_raw=0, turn=0,
                                       battlers=[local, enemy], menus=[dict(state="command")])) )
    return result


def test_plan_expands_half_fire_half_water_for_n_and_24_cycles():
    plan = module.load_commands(ROOT / "config/commands/switch_stress.json", 8)
    assert plan["total_turns"] == 120 and len(plan["players"]) == 8
    for i, player in enumerate(plan["players"]):
        assert module.switch_slot(player, 119) == (1 if i % 2 == 0 else 6)
    assert [module.switch_slot(plan["players"][0], i) for i in range(7)] == [2, 1, 2, 1, 2, 1, 2]
    assert [module.switch_slot(plan["players"][1], i) for i in range(7)] == [2, 3, 4, 5, 6, 2, 3]


@pytest.mark.parametrize("bad", [{"switch": True}, {"switch": 7}, {"switch": 2.5}, {"item": 1}])
def test_invalid_commands_rejected_before_launch(tmp_path, bad):
    path = tmp_path / "commands.json"
    path.write_text(json.dumps(dict(repeats=1, turns_per_repeat=1, players=[dict(cycle=[bad])]*2)))
    with pytest.raises(ValueError):
        module.load_commands(path, 4)


def test_preflight_catches_already_active_and_missing_slots():
    r = reports()[0]
    with pytest.raises(ValueError, match="already be active"):
        module.preflight(r, dict(cycle=[{"switch": 2}]), 2)
    with pytest.raises(ValueError, match="unavailable"):
        module.preflight(r, dict(cycle=[{"switch": 6}]), 1)


@pytest.fixture
def pair(monkeypatch):
    instances, state = [], reports()
    class Bridge:
        fail_prepare = False
        mismatch = False
        def __init__(self, host, port):
            self.index = port-55355
            self.prepared, self.committed, self.closed, self.cancelled = False, False, False, False
            self.request_id = 0
            instances.append(self)
        def status(self):
            return dict(link_turn_supported=True, link_peer_simulation_supported=True, native_link_session=True,
                        link_session_id="pair", link_player=self.index+1)
        def link_status(self):
            return dict(connection_established=True, received_remote_players=True,
                        player_count=2, local_id=self.index, link_battle=True,
                        players=[dict(id=0), dict(id=1)], link_type=1)
        def inspect(self):
            raw = deepcopy(state[self.index])
            if self.index == 1:
                raw["battle"]["turn"] = 0  # follower never runs the simulation
                for mon in raw["battle"]["battlers"]:
                    mon.update(valid=False, species_id=0, current_hp=0, max_hp=0)
            return raw
        def link_turn_prepare(self, request_id, slot, expected_turn):
            assert expected_turn == state[0]["battle"]["turn"]
            if self.fail_prepare and self.index == 1:
                raise RuntimeError("invalid target")
            self.request_id, self.slot, self.prepared = request_id, slot, True
            return dict(phase="prepared", request_id=request_id, active=True)
        def link_turn_commit(self, request_id):
            assert all(b.prepared and b.request_id == request_id for b in instances)
            self.committed = True
            return dict(ok=True)
        def link_turn_status(self):
            if all(b.committed for b in instances):
                for i, r in enumerate(state):
                    r["frame"] += 500
                    r["battle"]["turn"] = min(r["battle"]["turn"]+1, 255)
                    for side, player in (("player", i), ("opponent", 1-i)):
                        mon = module.active_mon(r, side)
                        target = state[player]["party"][instances[player].slot-1]
                        mon["party_slot"], mon["species_id"] = instances[player].slot, target["species_id"]
                if self.mismatch:
                    module.active_mon(state[0])["species_id"] = 999
                for b in instances:
                    b.prepared = b.committed = False
            return dict(phase="turn_complete", request_id=self.request_id, active=False)
        def link_turn_cancel(self):
            self.cancelled = True
        def close(self):
            self.closed = True
    monkeypatch.setattr(module, "MGBABridgeBackend", Bridge)
    return Bridge, instances


def test_120_paired_turns_and_final_inspector_verification(pair, tmp_path):
    Bridge, instances = pair
    plan = module.load_commands(ROOT / "config/commands/switch_stress.json", 2)
    result = module.run_pair_commands([55355, 55356], "pair", plan["players"], plan, tmp_path,
                                     announce=lambda _: None)
    assert result["status"] == "commands_passed" and result["turns"] == 120
    assert [(c["actual_slot"], c["species_id"]) for c in result["checks"]] == [(1, 6), (6, 130)]
    assert all(c["turn"] == 120 for c in result["checks"])
    assert all(b.closed and b.cancelled for b in instances)
    final = json.loads((tmp_path / "final-inspection-player2.json").read_text())["battle"]
    assert final["turn"] == 120 and final["turn_raw"] == 0
    assert final["simulation_source_port"] == 55355
    assert json.loads((tmp_path / "commands.jsonl").read_text().splitlines()[-1])["event"] == "passed"


def test_prepare_failure_never_commits_either_side(pair, tmp_path):
    Bridge, instances = pair
    Bridge.fail_prepare = True
    plan = module.load_commands(ROOT / "config/commands/switch_stress.json", 2)
    with pytest.raises(RuntimeError, match="invalid target"):
        module.run_pair_commands([55355, 55356], "pair", plan["players"], plan, tmp_path)
    assert all(not b.committed and b.cancelled and b.closed for b in instances)


def test_inspector_mismatch_stops_sequence_and_saves_failure(pair, tmp_path):
    Bridge, instances = pair
    Bridge.mismatch = True
    plan = module.load_commands(ROOT / "config/commands/switch_stress.json", 2)
    with pytest.raises(ValueError, match="expected slot"):
        module.run_pair_commands([55355, 55356], "pair", plan["players"], plan, tmp_path)
    assert (tmp_path / "failure-inspection-player1.json").is_file()
    assert not (tmp_path / "commands-result.json").exists()
    assert all(b.cancelled and b.closed for b in instances)
