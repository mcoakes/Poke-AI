"""Regression evidence: a real linked follower's gBattleMons records are zero."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from firered_rl.link_inspection import pair_inspections
from firered_rl.link_commands import active_mon, load_commands, preflight

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def snapshots():
    return json.loads((ROOT / "tests/fixtures/link_master_follower.json").read_text())


@pytest.mark.parametrize("reverse", [False, True])
def test_real_follower_is_decoded_from_master_with_local_perspective(snapshots, reverse):
    if reverse:
        snapshots.reverse()
    before = deepcopy(snapshots)
    result = pair_inspections(snapshots, [55355, 55356])
    assert snapshots == before  # raw evidence remains untouched
    expected = [9, 6] if reverse else [6, 9]
    for i, r in enumerate(result):
        assert active_mon(r)["species_id"] == expected[i]
        assert active_mon(r, "opponent")["species_id"] == expected[1-i]
        assert r["battle"]["menus"] == snapshots[i]["battle"]["menus"]
        assert r["battle"]["simulation_source_player"] == (2 if reverse else 1)
    follower = result[0 if reverse else 1]
    assert all(not m["valid"] for m in follower["battle"]["raw_battlers"])
    assert active_mon(follower)["data_source"] == "peer_simulation"
    assert active_mon(follower)["current_hp"] == 38


def test_real_snapshots_pass_switch_preflight(snapshots):
    result = pair_inspections(snapshots)
    plan = load_commands(ROOT / "config/commands/switch_stress.json", 2)
    assert [preflight(r, p, 120) for r, p in zip(result, plan["players"])] == [1, 6]
    result[0]["battle"]["turn"] = 200
    with pytest.raises(ValueError, match="255-turn"):
        preflight(result[0], plan["players"][0], 120)


def test_authority_and_local_indices_are_verified(snapshots):
    snapshots[1]["battle"]["battlers"][1]["party_slot"] = 2
    with pytest.raises(ValueError, match="party index disagrees"):
        pair_inspections(snapshots)
    snapshots[1]["battle"]["type_flags"] |= 4
    with pytest.raises(ValueError, match="exactly one simulation master"):
        pair_inspections(snapshots)


def test_follower_raw_turn_is_preserved(snapshots):
    snapshots[0]["battle"]["turn"] = 7
    result = pair_inspections(snapshots)
    assert result[1]["battle"]["turn"] == 7
    assert result[1]["battle"]["turn_raw"] == 0


def test_named_inspector_displays_both_perspectives(snapshots, capsys):
    spec = importlib.util.spec_from_file_location("inspect_state", ROOT / "mgba/inspect_state.py")
    inspector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inspector)
    for report in pair_inspections(snapshots, [55355, 55356]):
        inspector.display(inspector.decorate(report))
        output = capsys.readouterr().out
        assert "UNINITIALIZED/INVALID" not in output
        assert "Charizard" in output and "Blastoise" in output
        assert "Paired simulation data: port 55355" in output
