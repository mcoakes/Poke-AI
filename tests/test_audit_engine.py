"""The audit oracle reads native state without affecting production transitions."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not shutil.which("node")
    or not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
    reason="install the pinned Showdown simulator",
)


def run_worker(commands, worker="audit_runner.cjs"):
    result = subprocess.run(
        ["node", str(ROOT / "sim/showdown" / worker)],
        input="".join(json.dumps(command) + "\n" for command in commands),
        text=True, capture_output=True, check=True, timeout=30,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


def reset_command():
    return dict(command="reset", seed=[1, 2, 3, 4], teams=[
        dict(party=[
            dict(species="Charizard", level=50, moves=["Fly", "Splash"]),
            dict(species="Treecko", level=50, moves=["Absorb"], held_item="Leftovers"),
        ]),
        dict(party=[dict(species="Blissey", level=50, moves=["Splash"])]),
    ])


def test_native_reference_keeps_native_ids_and_original_order():
    reset, native, _, switched = run_worker([
        reset_command(), dict(command="audit_inspect"),
        dict(command="step", actions=[5, 0]), dict(command="audit_inspect"),
    ])
    assert native["audit_schema"] == 1
    assert len(native["rng_seed"]) == 4
    assert native["protocol"] == reset["protocol"]
    treecko = native["sides"][0]["party"][1]
    assert treecko["species"]["num"] == 252  # Native National Dex identifier.
    assert treecko["item"]["num"] == 234  # Native item ID, not cartridge ID 200.
    assert reset["reports"][0]["party"][1]["species_id"] == 277
    assert reset["reports"][0]["party"][1]["held_item_id"] == 200
    assert switched["sides"][0]["active_slot"] == 2
    assert switched["sides"][0]["order"] == [2, 1]
    assert [p["species"]["name"] for p in switched["sides"][0]["party"]] == [
        "Charizard", "Treecko"
    ]
    assert switched["sides"][0]["party"][1]["position"] == 0
    assert switched["sides"][0]["request"]["active"][0]["moves"][0]["id"] == "absorb"


def test_inspection_preserves_rng_log_offset_and_subsequent_transitions():
    commands = [reset_command()] + [dict(command="step", actions=actions) for actions in (
        [0, 0], [10, 0], [5, 0], [4, 0], [0, 0], [10, 0],
    )]
    expected = run_worker(commands, "runner.cjs")
    observed = run_worker([
        entry for command in commands
        for entry in (command, dict(command="audit_inspect"), dict(command="audit_inspect"))
    ])
    # Separate processes can cross a wall-clock second; only Showdown's |t:|
    # display messages vary for this reason. Compare every battle-state value
    # and every other protocol line exactly.
    def without_wallclock(rows):
        return [{**row, "protocol": [line for line in row.get("protocol", [])
                                     if not line.startswith("|t:|")]} for row in rows]
    assert without_wallclock(observed[::3]) == without_wallclock(expected)
    for first, second in zip(observed[1::3], observed[2::3]):
        assert first == second
        assert first["log_offset"] == len(first["protocol"])
    charging = observed[4]["sides"][0]["party"][0]
    assert charging["volatiles"]["twoturnmove"]["move"] == "fly"
    assert charging["trapped"] is True
    assert charging["moves"][0]["pp"] == 14


def test_inspection_avoids_event_getters_and_returns_detached_data():
    # Poison all tempting event-backed or adapter-like Pokemon getters. A raw
    # reference inspector must remain usable even when those APIs cannot run.
    code = """
      const assert = require('node:assert/strict');
      const worker = require('./sim/showdown/audit_runner.cjs');
      const {auditSource} = require('./sim/showdown/runner.cjs');
      worker.handle(JSON.parse(process.argv[1]));
      const {battle, original} = auditSource();
      const seed = JSON.stringify(battle.prng.rng.seed);
      for (const pokemon of original.flat()) {
        for (const name of ['getTypes', 'getMoves', 'getLockedMove',
          'getSemiLockedMove', 'getItem', 'getAbility', 'getMoveRequestData']) {
          pokemon[name] = () => { throw new Error(`Inspector called ${name}`); };
        }
      }
      battle.runEvent = battle.priorityEvent = () => { throw new Error('Inspector ran event'); };
      const before = worker.inspect();
      const changed = worker.inspect();
      changed.sides[0].party[0].hp = -1;
      changed.sides[0].party[0].types[0] = 'Water';
      changed.sides[0].request.active[0].moves[0].pp = 0;
      changed.protocol.push('changed');
      changed.rng_seed[0] = -1;
      assert.deepEqual(worker.inspect(), before);
      assert.equal(JSON.stringify(battle.prng.rng.seed), seed);
      console.log('ok');
    """
    result = subprocess.run(
        ["node", "-e", code, json.dumps(reset_command())], cwd=ROOT,
        text=True, capture_output=True, check=True, timeout=30,
    )
    assert result.stdout.strip() == "ok"


def test_audit_command_is_not_added_to_production_protocol():
    response = run_worker([dict(command="audit_inspect")], "runner.cjs")[0]
    assert response["ok"] is False
    assert response["error"] == "Unknown command: audit_inspect"
    response = run_worker([dict(command="audit_inspect")])[0]
    assert response["ok"] is False
    assert response["error"] == "Reset before inspecting"


def test_execution_evidence_preserves_native_state_protocol_and_rng():
    commands = [reset_command()] + [dict(command='step', actions=actions) for actions in (
        [0, 0], [10, 0], [5, 0], [4, 0],
    )]
    plain = run_worker([row for cmd in commands for row in (cmd, dict(command='audit_inspect'))])
    audited = run_worker([row for cmd in commands
                          for row in ({**cmd, 'audit_execution': True}, dict(command='audit_inspect'))])
    for index, (before, after) in enumerate(zip(plain, audited)):
        assert 'audit_execution' not in before
        evidence = after.pop('audit_execution', None)
        if index > 0 and index % 2 == 0:
            assert evidence is not None
        for row in (before, after):
            row['protocol'] = [line for line in row.get('protocol', []) if not line.startswith('|t:|')]
        assert before == after
