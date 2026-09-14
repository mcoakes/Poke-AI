import json
from pathlib import Path

import pytest

from firered_rl.team_generation import COMMON_ITEMS, generate, load_catalog, write_batch
from firered_rl.teams import PokemonSpec, TeamSpec


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(Path(__file__).parents[1] / "config/generation")


def test_batch_proportions_and_seed_reproduction(catalog, tmp_path):
    teams, summary = generate(*catalog, count=40, seed=42)
    assert (teams, summary) == generate(*catalog, count=40, seed=42)
    assert teams != generate(*catalog, count=40, seed=43)[0]
    assert summary["style_counts"] == {"competitive": 144, "learnable_random": 48, "chaos": 48}
    assert summary["common_item_count"] == 192
    data = catalog[0]
    for team in teams:
        assert len({p["species"] for p in team["party"]}) == 6
        for p, meta in zip(team["party"], team["generation"]["members"]):
            if meta["style"] != "chaos":
                assert set(p["moves"]) <= set(data["species"][p["species"]]["moves"])
            if meta["style"] == "competitive":
                assert p["held_item"] in COMMON_ITEMS
            assert len(set(p["moves"])) == len(p["moves"])
            assert sum(p["evs"].values()) == 508
    directory = tmp_path / "teams"
    write_batch(teams, summary, directory)
    assert len(list(directory.glob("*.json"))) == 40
    for file in directory.glob("*.json"):
        assert len(TeamSpec.load(file).party) == 6
    with pytest.raises(FileExistsError):
        write_batch(teams, summary, directory)


def test_stat_and_item_overrides_reach_json(catalog):
    teams, summary = generate(*catalog, count=2, iv_mode="random", ev_mode="random",
        overrides={"evs": {"hp": 252, "def": 252, "spd": 4}, "ivs": {"atk": 0},
                   "held_item": "LumBerry", "nature": "bold"})
    assert summary["common_item_count"] == 12
    for p in (p for t in teams for p in t["party"]):
        parsed = PokemonSpec.from_dict(p)
        assert parsed.evs == (252, 0, 252, 0, 4, 0)
        assert parsed.ivs == (31, 0, 31, 31, 31, 31)
        assert parsed.nature == "Bold" and parsed.held_item == "Lum Berry"


@pytest.mark.parametrize("options", [
    {"competitive_rate": .9, "chaos_rate": .2},
    {"competitive_item_rate": .2}, {"chaos_rate": float("nan")},
    {"overrides": {"evs": {"hp": 252, "atk": 252, "spe": 252}}},
    {"overrides": {"ivs": {"spe": 32}}},
    {"overrides": {"held_item": "Choice Scarf"}},
    {"prefix": "../../oops"},
])
def test_invalid_batch_fails_before_writing(catalog, options):
    with pytest.raises(ValueError):
        generate(*catalog, **options)


def test_generation_three_boundaries_and_random_modes(catalog):
    data = catalog[0]
    assert {p["id"] for p in data["species"].values()} == set(range(1, 387))
    assert data["items"]["Leftovers"]["id"] == 200
    assert data["items"]["Choice Band"]["id"] == 186
    assert data["moves"]["Shadow Ball"]["category"] == "physical"
    assert data["moves"]["Bite"]["category"] == "special"
    teams, report = generate(*catalog, count=5, species_pool="kanto", competitive_rate=0,
                             chaos_rate=1, competitive_item_rate=0, iv_mode="random", ev_mode="zero")
    assert report["common_item_count"] == 0
    for p in (p for t in teams for p in t["party"]):
        assert data["species"][p["species"]]["id"] <= 151
        assert sum(p["evs"].values()) == 0
        assert all(0 <= value <= 31 for value in p["ivs"].values())
        assert len(p["moves"]) == 4


def test_shipped_examples_match_seeded_generation(catalog):
    directory = Path(__file__).parents[1] / "config/teams/mixed_example"
    teams, _ = generate(*catalog, count=40, seed=42, prefix="mixed")
    assert [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))] == teams
