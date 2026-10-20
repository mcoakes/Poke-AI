"""Verify actual configured diversity, split isolation, legality and immutability."""
import copy
import json
from pathlib import Path

import pytest

from firered_rl.broad_team_curriculum import (
    SPLITS, build_pools, composition_signature, semantic_signature,
    validate_pools, write_fixed_assets,
)
from firered_rl.teams import TeamSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/team-curriculum.json"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    return build_pools(ROOT, tmp_path_factory.mktemp("broad-pools"), CONFIG)


def test_default_counts_diversity_and_all_individual_moves_are_legal(generated):
    config = json.loads(CONFIG.read_text())
    catalog = json.loads((ROOT / config["catalog"]).read_text())
    counts, species, moves = [], [], []
    for stage in config["stages"]:
        entry = generated[stage["id"]]
        for split in SPLITS:
            diversity = entry["diversity"][split]
            assert diversity["teams"] == stage["counts"][split]
            assert diversity["semantic_teams"] == diversity["teams"]
            assert diversity["compositions"] == diversity["teams"]
            for relative in entry[f"{split}_paths"]:
                path = Path(entry["root"]) / relative
                team = TeamSpec.load(path)
                assert len(team.party) == stage["team_size"]
                for mon in team.party:
                    assert len(mon.moves) == stage["moves"]["count"]
                    assert set(mon.moves) <= set(catalog["species"][mon.species]["moves"])
                    assert mon.held_item in catalog["items"]
                    attacks = [m for m in mon.moves if catalog["moves"][m]["power"] > 1]
                    assert len(attacks) >= 2
                    assert any(catalog["moves"][m]["type"] in catalog["species"][mon.species]["types"]
                               and catalog["moves"][m]["power"] >= 40 for m in attacks)
        counts.append(entry["diversity"]["train"]["teams"])
        species.append(entry["diversity"]["train"]["species"])
        moves.append(entry["diversity"]["train"]["moves"])
    assert counts == [16, 48, 96, 160, 256, 384, 1024]
    assert species == sorted(species) and species[-1] >= 60
    assert moves == sorted(moves) and moves[-1] >= 80
    assert generated["07-varied-builds"]["diversity"]["train"]["natures"] > 1
    assert generated["07-varied-builds"]["diversity"]["train"]["items"] > 1


def test_all_stages_share_disjoint_composition_splits_even_with_rehearsal(generated):
    compositions = {split: set() for split in SPLITS}
    signatures = {split: set() for split in SPLITS}
    for entry in generated.values():
        for split in SPLITS:
            compositions[split].update(entry["composition_signatures"][split])
            signatures[split].update(entry["semantic_signatures"][split])
    for i, split in enumerate(SPLITS):
        for other in SPLITS[i + 1:]:
            assert not compositions[split] & compositions[other]
            assert not signatures[split] & signatures[other]
    fixed = json.loads(CONFIG.read_text())["fixed_stages"]
    for value in fixed.values():
        for team in value["teams"]:
            for split in ("validation", "heldout", "benchmark"):
                assert composition_signature(team) not in compositions[split]


def test_semantic_diversity_does_not_count_order_or_labels():
    team = json.loads(CONFIG.read_text())["fixed_stages"]["mirror"]["teams"][0]
    modified = copy.deepcopy(team)
    modified["name"] = "different label"
    modified["party"].reverse()
    for mon in modified["party"]:
        mon["moves"].reverse()
    assert semantic_signature(team) == semantic_signature(modified)
    modified["party"][0]["moves"][0] = "Protect"
    assert semantic_signature(team) != semantic_signature(modified)


def test_rebuilding_is_reproducible_and_pinned(generated, tmp_path):
    other = build_pools(ROOT, tmp_path, CONFIG)
    for stage, entry in generated.items():
        assert other[stage]["sha256"] == entry["sha256"]
        assert other[stage]["diversity"] == entry["diversity"]
    assert build_pools(ROOT, tmp_path, CONFIG) == other
    validate_pools(other)
    entry = next(iter(other.values()))
    path = tmp_path / entry["train_paths"][0]
    original = path.read_text()
    path.write_text(original + " ")
    with pytest.raises(ValueError, match="pinned team changed"):
        validate_pools(other)
    with pytest.raises(ValueError, match="pinned curriculum asset differs"):
        build_pools(ROOT, tmp_path, CONFIG)
    path.write_text(original)
    extra = path.parent / "extra.json"
    extra.write_text(original)
    with pytest.raises(ValueError, match="membership changed"):
        validate_pools(other)


def test_invalid_configured_support_moves_fail_before_generation(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["move_pools"]["support"].append("Made Up Move")
    with pytest.raises(ValueError, match="unknown move"):
        build_pools(ROOT, tmp_path, config)
    config["move_pools"]["support"][-1] = "Surf"
    with pytest.raises(ValueError, match="status moves"):
        build_pools(ROOT, tmp_path, config)


def test_fixed_teams_are_editable_json_and_keep_original_matchups(tmp_path):
    from firered_rl.fixed_team_lab import teams
    config = json.loads(CONFIG.read_text())
    entries = write_fixed_assets(ROOT, tmp_path, config)
    for name, entry in entries.items():
        assert [semantic_signature(t) for t in config["fixed_stages"][name]["teams"]] == [semantic_signature(t) for t in teams()[name]]
        for relative, expected in zip(entry["train_paths"], config["fixed_stages"][name]["teams"]):
            actual = json.loads((tmp_path / relative).read_text())
            assert semantic_signature(actual) == semantic_signature(expected)
    assert json.loads((tmp_path / entries["move-choice"]["train_paths"][0]).read_text())["party"][0]["species"] == "Starmie"
    config["fixed_stages"]["mirror"]["teams"][0]["party"][0]["moves"] = ["Surf"]
    with pytest.raises(ValueError, match="absent from bundled learnset"):
        write_fixed_assets(ROOT, tmp_path / "bad", config)
