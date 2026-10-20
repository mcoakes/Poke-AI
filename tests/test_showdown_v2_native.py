"""Real Gen III state exercises for the additive, perspective-relative report."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "sim/showdown/audit_runner.cjs"
pytestmark = pytest.mark.skipif(
    not shutil.which("node")
    or not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
    reason="Showdown not installed",
)


def mon(species, moves, **kwargs):
    return dict(species=species, level=50, moves=moves, nature="Hardy", **kwargs)


class NativeWorker:
    def __init__(self):
        self.process = subprocess.Popen(
            ["node", str(RUNNER)], cwd=ROOT, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def call(self, command, **kwargs):
        self.process.stdin.write(json.dumps(dict(command=command, **kwargs)) + "\n")
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        assert result["ok"], result
        return result

    def reset(self, first, second, seed=(1, 2, 3, 4)):
        return self.call("reset", seed=seed, teams=[dict(party=first), dict(party=second)])

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)
        assert self.process.returncode == 0, self.process.stderr.read()


@pytest.fixture
def worker():
    value = NativeWorker()
    yield value
    value.close()


def views(response):
    values = [report["observation_v2"] for report in response["reports"]]
    assert response["reports"][0]["backend"] == "showdown"
    for field in values[0]:
        # Weather is the one global field; all other fields reverse perspectives.
        if field == "v2_weather":
            assert values[0][field] == values[1][field]
        else:
            assert values[0][field] == values[1][field][::-1]
    return values


def test_native_party_stats_ids_details_and_padding(worker):
    pikachu = mon("Pikachu", ["Quick Attack", "Bite", "Aerial Ace", "Counter"],
                  evs=dict(atk=252, spe=252, hp=6),
                  ivs={"hp": 0, "atk": 31, "def": 30, "spa": 2, "spd": 17, "spe": 25})
    pikachu["nature"] = "Adamant"
    value = views(worker.reset(
        [mon("Mew", ["Hidden Power", "Splash"], held_item="Leftovers"), pikachu],
        [mon("Blissey", ["Splash"], held_item="Oran Berry")],
    ))[0]
    shapes = {"v2_party": (2, 6, 10), "v2_stats": (2, 6, 5),
              "v2_active_stats": (2, 5), "v2_active": (2, 4),
              "v2_boosts": (2, 7), "v2_move_details": (2, 6, 4, 6),
              "v2_weather": (3,), "v2_volatiles": (2,)}
    assert set(value) == set(shapes)
    for name, shape in shapes.items():
        assert np.asarray(value[name]).shape == shape
    # FireRed item/type numbers and natural Gen III ability numbers.
    assert value["v2_party"][0][0] == [1, 50, 175, 175, 14, 14, 200, 28, 0, 0]
    assert value["v2_party"][1][0] == [1, 50, 330, 330, 0, 0, 139, 30, 0, 0]
    # Computed nature/IV/EV stats, rather than the species' 55/30/50/40/90 bases.
    assert value["v2_stats"][0][1] == [117, 50, 50, 53, 139]
    assert value["v2_active_stats"] == [[120] * 5, [30, 30, 95, 155, 75]]
    assert value["v2_active"] == [[175, 175, 200, 28], [330, 330, 139, 30]]
    assert value["v2_move_details"][0][0][:2] == [
        [70, 100, 0, 1, 0, 17], [0, 0, 1, 2, 0, 0],
    ]
    # Bite is special in Gen III. Priority is signed; always-hit is explicit.
    assert value["v2_move_details"][0][1] == [
        [40, 100, 0, 0, 1, 0], [60, 100, 0, 1, 0, 17],
        [60, 0, 1, 0, 0, 2], [0, 100, 0, 0, -5, 1],
    ]
    for field in ("v2_party", "v2_stats", "v2_move_details"):
        assert np.count_nonzero(np.asarray(value[field])[:, 2:]) == 0
    assert np.count_nonzero(np.asarray(value["v2_move_details"])[0, 0, 2:]) == 0
    assert value["v2_weather"] == [0, 0, 0]


@pytest.mark.parametrize("side", [0, 1])
def test_signed_boosts_and_original_party_slots(worker, side):
    teams = [[mon("Mew", ["Swords Dance", "Splash"]), mon("Pikachu", ["Splash"])],
             [mon("Gyarados", ["Splash"])]]
    if side:
        teams.reverse()
    before = views(worker.reset(*teams))[side]
    assert before["v2_boosts"][0] == [-1, 0, 0, 0, 0, 0, 0]
    after = views(worker.call("step", actions=[0, 0]))[side]
    assert after["v2_boosts"][0] == [1, 0, 0, 0, 0, 0, 0]
    assert after["v2_stats"] == before["v2_stats"]
    assert after["v2_active_stats"] == before["v2_active_stats"]
    actions = [0, 0]
    actions[side] = 5
    switched = views(worker.call("step", actions=actions))[side]
    assert switched["v2_stats"] == before["v2_stats"]
    assert switched["v2_active_stats"][0] == before["v2_stats"][0][1]
    assert switched["v2_boosts"][0] == [0] * 7


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("move,code", [("Thunder Wave", 64), ("Spore", None), ("Toxic", None)])
def test_major_status_raw_comes_from_native_state(worker, side, move, code):
    teams = [[mon("Alakazam", [move, "Splash"])], [mon("Blissey", ["Splash"])]]
    if side:
        teams.reverse()
    worker.reset(*teams)
    for _ in range(8):
        report = worker.call("step", actions=[0, 0])
        raw = worker.call("audit_inspect")
        target = raw["sides"][1 - side]["party"][0]
        if target["status"]:
            break
    assert target["status"]
    if move == "Spore":
        expected = max(1, min(7, target["status_state"]["time"]))
    elif move == "Toxic":
        expected = 128 | (min(15, target["status_state"]["stage"]) << 8)
    else:
        expected = code
    assert views(report)[side]["v2_party"][1][0][9] == expected


@pytest.mark.parametrize("side", [0, 1])
def test_item_exchange_and_consumption(worker, side):
    teams = [[mon("Mew", ["Trick", "Splash"], held_item="Leftovers")],
             [mon("Blissey", ["Splash"], held_item="Oran Berry")]]
    if side:
        teams.reverse()
    worker.reset(*teams)
    value = views(worker.call("step", actions=[0, 0]))[side]
    assert [value["v2_party"][i][0][6] for i in (0, 1)] == [139, 200]
    assert [value["v2_active"][i][2] for i in (0, 1)] == [139, 200]
    teams = [[mon("Mew", ["Splash"], held_item="Oran Berry")],
             [mon("Blissey", ["Super Fang"])]]
    if side:
        teams.reverse()
    worker.reset(*teams)
    for _ in range(8):
        value = views(worker.call("step", actions=[0, 0]))[side]
        if value["v2_active"][0][2] == 0:
            break
    assert value["v2_party"][0][0][6] == value["v2_active"][0][2] == 0


@pytest.mark.parametrize("move,weather", [("Sunny Day", 1), ("Rain Dance", 2),
                                         ("Sandstorm", 3), ("Hail", 4)])
def test_weather_remaining_turns_and_expiration(worker, move, weather):
    worker.reset([mon("Mew", [move, "Splash"])], [mon("Blissey", ["Splash"])])
    value = views(worker.call("step", actions=[0, 0]))[0]
    for remaining in (4, 3, 2, 1):
        raw = worker.call("audit_inspect")
        assert raw["field"]["weather_state"]["duration"] == remaining
        assert value["v2_weather"] == [weather, remaining, 0]
        value = views(worker.call("step", actions=[1, 0]))[0]
    assert value["v2_weather"] == [0, 0, 0]


@pytest.mark.parametrize("species,weather", [("Groudon", 1), ("Kyogre", 2), ("Tyranitar", 3)])
def test_ability_weather_is_permanent(worker, species, weather):
    response = worker.reset([mon(species, ["Splash"])], [mon("Blissey", ["Splash"])])
    assert views(response)[0]["v2_weather"] == [weather, 0, 1]
    for _ in range(6):
        response = worker.call("step", actions=[0, 0])
    assert views(response)[0]["v2_weather"] == [weather, 0, 1]


@pytest.mark.parametrize("side", [0, 1])
def test_transform_replaces_whole_move_row_but_preserves_base_stats(worker, side):
    teams = [[mon("Ditto", ["Transform", "Splash", "Quick Attack", "Surf"])],
             [mon("Blissey", ["Hidden Power"],
                  ivs={"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0})]]
    if side:
        teams.reverse()
    original = views(worker.reset(*teams))[side]
    copied = views(worker.call("step", actions=[0, 0]))[side]
    assert copied["v2_stats"] == original["v2_stats"]
    assert copied["v2_active_stats"][0] == original["v2_active_stats"][1]
    assert copied["v2_party"][0][0][4:6] == original["v2_party"][0][0][4:6]
    # All-zero IVs yield Fighting Hidden Power with base power 30.
    assert copied["v2_move_details"][0][0] == [[30, 100, 0, 0, 0, 1]] + [[0] * 6] * 3
    assert copied["v2_volatiles"][0] & (1 << 21)
    assert copied["v2_active"][0][3] == copied["v2_active"][1][3] == 30


def test_inspection_is_read_only_and_metadata_is_native(worker):
    worker.reset([mon("Mew", ["Confuse Ray", "Splash"])],
                 [mon("Blissey", ["Splash"], held_item="Leftovers")])
    report = worker.call("step", actions=[0, 0])
    first = worker.call("audit_inspect")
    second = worker.call("audit_inspect")
    assert first == second
    native = first["sides"][1]["party"][0]
    expected = max(1, min(7, native["volatiles"]["confusion"]["time"]))
    assert views(report)[0]["v2_volatiles"][1] & 7 == expected
    splash = native["moves"][0]
    assert splash["accuracy"] is True
    assert splash["priority"] == 0


def test_fainted_party_row_retains_identity_and_exact_zero_hp(worker):
    victim = mon("Magikarp", ["Splash"])
    victim["level"] = 1
    before = views(worker.reset([victim], [mon("Mew", ["Thunderbolt"])]))[0]
    result = worker.call("step", actions=[0, 0])
    assert result["ended"]
    after = views(result)[0]
    assert after["v2_party"][0][0][0] == 1
    assert after["v2_party"][0][0][2] == after["v2_active"][0][0] == 0
    assert after["v2_party"][0][0][8] == 1
    assert after["v2_stats"] == before["v2_stats"]


# This digest was captured from the pre-v2 worker with the pinned simulator.
# It covers exact legacy report values, protocol, legal masks and RNG state.
V1_TRACE_SCRIPT = r"""
Date.now = () => 1700000000000;
const production = require('./sim/showdown/runner.cjs');
const mon = (species, moves, item = '') => ({species, level:50, moves, held_item:item});
const battles = [
  [[mon('Mew', ['Confuse Ray','Toxic','Hidden Power','Splash'], 'Leftovers'),
    mon('Gyarados', ['Surf','Bite'])],
   [mon('Blissey', ['Seismic Toss','Soft-Boiled','Thunder Wave','Splash'], 'Oran Berry'),
    mon('Snorlax', ['Rest','Sleep Talk','Body Slam'])]],
  [[mon('Ditto', ['Transform','Splash','Quick Attack','Surf']), mon('Pikachu', ['Thunderbolt'])],
   [mon('Kyogre', ['Hidden Power','Water Pulse']), mon('Mew', ['Sunny Day','Splash'])]],
  [[mon('Charizard', ['Fly','Hyper Beam','Outrage','Splash']), mon('Treecko', ['Absorb'])],
   [mon('Lapras', ['Ice Beam','Rest','Sleep Talk']), mon('Blissey', ['Splash'])]],
];
const trace = [];
for (let b = 0; b < battles.length; b++) {
  let result = production.handle({command:'reset', seed:[1,2,3,b+4],
    teams:battles[b].map(party => ({party}))});
  for (let t = 0; t < 19; t++) {
    for (const report of result.reports) delete report.observation_v2;
    trace.push({result, rng:production.auditSource().battle.prng.rng.seed.slice()});
    if (result.ended || t === 18) break;
    const actions = result.reports.map((report, side) => {
      const legal = report.action_mask.map((allowed, index) => allowed ? index : -1)
        .filter(index => index >= 0);
      return legal[(t * 3 + side + b) % legal.length];
    });
    result = production.handle({command:'step', actions});
  }
}
production.auditSource().battle.destroy();
process.stdout.write(JSON.stringify(trace));
"""


def test_v1_reports_protocol_actions_and_rng_match_pre_v2_worker():
    result = subprocess.run(["node", "-e", V1_TRACE_SCRIPT], cwd=ROOT, check=True,
                            capture_output=True, text=True, timeout=30)
    expected = "0a93e932430019806baaa1fc58a4aafde6751eeb08a184d273460c68b5b5b875"
    assert hashlib.sha256(result.stdout.encode()).hexdigest() == expected
