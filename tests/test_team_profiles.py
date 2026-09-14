import json
from pathlib import Path

import pytest

from firered_rl.team_generation import COMMON_ITEMS
from firered_rl.team_profile_generation import generate_profiles, load_profile_catalog, write_profiles
from firered_rl.team_profiles import CATEGORIES, profile_files
from firered_rl.teams import PokemonSpec

ROOT = Path(__file__).parents[1]


def test_complete_catalog_is_reproducible_varied_and_has_four_moves():
    catalog, templates, roles = load_profile_catalog(ROOT / "config/generation")
    root = ROOT / "config/teams/profiles"
    batches = list(generate_profiles(catalog, templates, roles))
    assert len(batches) == 18
    assert len(list(root.glob("*/*/*.json"))) == 2700
    for relative, teams, report in batches:
        category, size = relative.parts[0], int(relative.parts[1])
        paths = sorted((root / relative).glob("*.json"))
        assert len(paths) == len(teams) == 150
        assert [json.loads(p.read_text()) for p in paths] == teams
        assert len({json.dumps(t["party"], sort_keys=True) for t in teams}) == 150
        species = set()
        for team in teams:
            assert len(team["party"]) == size
            assert len({p["species"] for p in team["party"]}) == size
            for mon in team["party"]:
                PokemonSpec.from_dict(mon)
                species.add(mon["species"])
                assert len(set(mon["moves"])) == 4
                assert set(mon["moves"]) <= catalog["moves"].keys()
                assert not {"Sketch", "Struggle"} & set(mon["moves"])
                assert mon["held_item"] in catalog["items"] and mon["level"] == 50
                assert sum(mon["evs"].values()) <= 510 and max(mon["evs"].values()) <= 252
                if category != "random":
                    assert set(mon["moves"]) <= set(catalog["species"][mon["species"]]["moves"])
                if category == "competitive":
                    assert set(mon["ivs"].values()) == {31}
                    assert sum(mon["evs"].values()) == 508
                    assert mon["held_item"] in COMMON_ITEMS
        assert len(species) >= (30 if category == "competitive" else 75)
        assert report["common_item_count"] == (150 * size if category == "competitive" else 120 * size)


def test_seeds_change_rosters_and_writer_never_overwrites(tmp_path):
    catalog, templates, roles = load_profile_catalog(ROOT / "config/generation")
    first = list(generate_profiles(catalog, templates, roles, count=3, seed=1))
    second = list(generate_profiles(catalog, templates, roles, count=3, seed=2))
    assert all(a[1] != b[1] for a, b in zip(first, second))
    target = tmp_path / "new"
    assert write_profiles(target, first) == 54
    with pytest.raises(FileExistsError):
        write_profiles(target, second)
    assert json.loads(next((target / "competitive/1").glob("*.json")).read_text()) in first[0][1]


def test_profile_json_selection_filters_category_and_size():
    root = ROOT / "config/teams/profiles"
    groups = profile_files(root, "competitive", 3)
    assert list(groups) == ["competitive"] and len(groups["competitive"]) == 150
    assert all(p.parent.name == "3" for p in groups["competitive"])
    assert {key: len(files) for key, files in profile_files(root, "mixed").items()} == dict.fromkeys(CATEGORIES, 900)
