import json

import pytest

from firered_rl.teams import PokemonSpec, TeamPool, TeamSpec


def write_team(path, name="test", party=None):
    party = party or [{"species": "Pikachu", "level": 50, "moves": ["Thunderbolt"]}]
    path.write_text(json.dumps({"name": name, "party": party}), encoding="utf-8")


def test_load_team(tmp_path):
    path = tmp_path / "team.json"
    write_team(path)
    team = TeamSpec.load(path)
    assert team.name == "test"
    assert team.party == (PokemonSpec("Pikachu", 50, ("Thunderbolt",)),)


@pytest.mark.parametrize("level", [0, 101])
def test_rejects_invalid_level(tmp_path, level):
    path = tmp_path / "team.json"
    write_team(path, party=[{"species": "Pikachu", "level": level, "moves": ["Tackle"]}])
    with pytest.raises(ValueError, match="level"):
        TeamSpec.load(path)


def test_seeded_team_selection_is_reproducible(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"team{index}.json"
        write_team(path, name=f"team{index}")
        paths.append(path)
    pool = TeamPool(paths)
    assert pool.choose(42) == pool.choose(42)


@pytest.mark.parametrize("fields", [
    {"evs": {"hp": 256}}, {"evs": {"hp": 255, "def": 255, "spe": 1}},
    {"evs": {"spdef": 4}}, {"ivs": {"atk": -1}}, {"ivs": {"hp": True}},
    {"level": 50.5}, {"held_itme": "Leftovers"}, {"nature": "Clever"},
])
def test_stat_input_validation(fields):
    with pytest.raises(ValueError):
        PokemonSpec.from_dict({"species": "Pikachu", "level": 50, "moves": ["Thunderbolt"]} | fields)
