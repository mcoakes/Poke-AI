"""Semantic curriculum split, paired checks, and bounded serializable traces."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
from firered_rl.evaluation import validate_suite
from firered_rl.staged_team_lab import (
    StageDecisionAudit, asset_paths, json_native, make_stage_suite,
    validate_assets, write_assets,
)

ROOT = Path(__file__).resolve().parents[1]


def composition(path):
    return frozenset(mon["species"] for mon in json.loads(path.read_text())["party"])


def test_curriculum_assets_split_compositions_not_just_order(tmp_path):
    assets = write_assets(tmp_path)
    json.dumps(assets)
    assert len(asset_paths(assets["order"])) == 6
    assert len({composition(p) for p in asset_paths(assets["order"])}) == 1
    train = asset_paths(assets["composition"])
    heldout = asset_paths(assets["composition"], heldout=True)
    assert len(train) == 36 and len(heldout) == 12
    train_compositions = {composition(p) for p in train}
    heldout_compositions = {composition(p) for p in heldout}
    assert len(train_compositions) == 6 and len(heldout_compositions) == 2
    assert not train_compositions & heldout_compositions
    assert frozenset(("Typhlosion", "Blastoise", "Sceptile")) in train_compositions
    assert assets == write_assets(tmp_path)


@pytest.mark.parametrize("stage,heldout", [("move-choice", False), ("mirror", False),
    ("order", False), ("composition", False), ("composition", True)])
def test_suite_rotation_preserves_semantic_pairing_and_seeds(tmp_path, stage, heldout):
    entry = write_assets(tmp_path)[stage]
    suite = make_stage_suite(ROOT, entry, 24, 17, stage, heldout=heldout)
    rotated = make_stage_suite(ROOT, entry, 24, 17, stage, heldout=heldout, permute_moves=True)
    validate_suite(suite)
    validate_suite(rotated)
    assert sum(c["candidate_player"] == 1 for c in suite["cases"]) == 12
    changed = 0
    for a, b in zip(suite["cases"], rotated["cases"]):
        assert (a["seed"], a["candidate_player"], a["matchup"]) == (
            b["seed"], b["candidate_player"], b["matchup"])
        seat = a["candidate_player"] - 1
        original = suite["assets"][a["assets"][seat]]["team"]
        modified = rotated["assets"][b["assets"][seat]]["team"]
        assert suite["assets"][a["assets"][1-seat]] == rotated["assets"][b["assets"][1-seat]]
        for actual, expected in zip(modified["party"], original["party"]):
            assert {k: v for k, v in actual.items() if k != "moves"} == {
                k: v for k, v in expected.items() if k != "moves"}
            assert sorted(actual["moves"]) == sorted(expected["moves"])
            changed += actual["moves"] != expected["moves"]
        if stage == "move-choice":
            assert original["party"][0]["species"] == "Starmie"
    assert changed > 0
    if stage in ("order", "composition"):
        for a, b in zip(suite["cases"][::2], suite["cases"][1::2]):
            assert a["assets"] == b["assets"]
            assert a["assets"][0] != a["assets"][1]


def test_suites_use_distinct_seed_streams_and_reject_modified_inputs(tmp_path):
    entry = write_assets(tmp_path)["order"]
    suites = [make_stage_suite(ROOT, entry, 256, seed, "order") for seed in (731702, 731703)]
    assert not {c["seed"] for c in suites[0]["cases"]} & {c["seed"] for c in suites[1]["cases"]}
    asset_paths(entry)[0].write_text("{}")
    with pytest.raises(ValueError, match="asset changed"):
        validate_assets(entry)


def test_pool_pinning_rejects_new_untracked_team(tmp_path):
    entry = write_assets(tmp_path)["order"]
    extra = Path(entry["root"]) / entry["train_dir"] / "untracked.json"
    extra.write_text(asset_paths(entry)[0].read_text())
    with pytest.raises(ValueError, match="asset list changed"):
        validate_assets(entry)


def test_audit_counts_forced_and_voluntary_and_bounds_traces(tmp_path):
    observer = StageDecisionAudit(ROOT, None, trace_cases=2, trace_decisions=2)
    observer.rules.choose = lambda obs: (int(np.flatnonzero(obs["action_mask"])[0]), [])
    obs = dict(action_mask=np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
               active=np.zeros((2, 10), dtype=np.float32),
               party=np.zeros((6, 10), dtype=np.float32),
               moves=np.zeros((6, 4, 4), dtype=np.float32),
               context=np.array([1, 0, 0, 0], dtype=np.float32),
               v2_active=np.zeros((2, 4), dtype=np.float32))
    obs["active"][0, 1], obs["active"][0, 8] = 157, 1
    obs["active"][1, 1], obs["party"][1, 1] = 9, 9
    obs["moves"][0, 0, 0] = 57
    case = dict(episode=1, matchup=1, seed=17, candidate_player=1)
    observer(case, 1, obs, np.int64(0))
    observer(case, 2, obs, np.int64(5))
    forced = copy.deepcopy(obs)
    forced["context"][1] = 1
    observer(case, 3, forced, np.int64(5))
    observer(dict(case, episode=3), 1, obs, 0)
    result = observer.finish(tmp_path)
    assert result["totals"]["voluntary_switches"] == 1
    assert result["totals"]["forced_switches"] == 1
    assert result["totals"]["voluntary_decisions"] == 3
    assert result["voluntary_switch_rate"] == pytest.approx(1/3)
    assert result["traced_decisions"] == 2
    traces = [json.loads(line) for line in (tmp_path / "decision-traces.jsonl").read_text().splitlines()]
    assert traces[0]["chosen"]["move_id"] == 57
    assert traces[1]["chosen"]["species_id"] == 9
    assert traces[1]["chosen"]["kind"] == "switch"
    json.dumps(json_native(dict(test=np.float32(1.5), nested=[np.int64(2)],
                               array=np.array([np.nan, np.inf]))), allow_nan=False)
