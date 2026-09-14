import json
from collections import Counter
import argparse
from pathlib import Path

import pytest

from firered_rl.training_states import TrainingStates, add_state_arguments
from firered_rl.teams import TeamSpec
from firered_rl.team_profiles import CATEGORIES


def profile_checkpoints(root):
    for category in CATEGORIES:
        for size in (1, 3, 6):
            directory = root / category / str(size)
            directory.mkdir(parents=True)
            # Deliberately unbalanced catalog: mixed must still weight categories equally.
            for i in range(1 if category == "competitive" else 9):
                (directory / f"{i}.ss1").write_bytes(f"{category}/{size}/{i}".encode())
            (directory / "0.ss1.json").write_text(json.dumps({"verified_party": [{}] * size}))
            (directory / "battery.sav").write_bytes(b"not a checkpoint")


def test_mixed_profile_is_balanced_seeded_and_filters_both_players(tmp_path):
    profile_checkpoints(tmp_path)
    source = TrainingStates(profile="mixed", team_size=3, profile_root=tmp_path, seed=71)
    choose, other = source.for_pair(0), source.for_pair(1)
    sequence = []
    for _ in range(3000):
        sequence.append(choose())
        other()
    replay = source.for_pair(0)
    assert sequence == [replay() for _ in sequence]
    counts = Counter(p.parent.parent.name for pair in sequence for p in pair)
    assert all(1800 < counts[c] < 2200 for c in CATEGORIES)
    assert all(p.parent.name == "3" for pair in sequence for p in pair)
    assert any(a.parent.parent != b.parent.parent for a, b in sequence)
    assert any(a.parent.parent == b.parent.parent for a, b in sequence)
    assert all(a != b or a.parent.parent.name == "competitive" for a, b in sequence)
    description = source.describe()
    assert description["mode"] == "team_profile" and description["team_size"] == 3
    assert description["category_counts"] == dict(competitive=1, casual=9, random=9)


@pytest.mark.parametrize("category", CATEGORIES)
def test_single_profile_without_size_can_draw_all_built_sizes(tmp_path, category):
    profile_checkpoints(tmp_path)
    choose = TrainingStates(profile=category, profile_root=tmp_path).for_pair(0)
    paths = [p for _ in range(100) for p in choose()]
    assert {p.parent.parent.name for p in paths} == {category}
    assert {p.parent.name for p in paths} == {"1", "3", "6"}


def test_profile_validation_and_missing_states_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="requires --team-profile"):
        TrainingStates(directory=tmp_path, team_size=3)
    with pytest.raises(ValueError, match="exactly one"):
        TrainingStates(directory=tmp_path, profile="mixed")
    with pytest.raises(ValueError, match="Build them first"):
        TrainingStates(profile="casual", team_size=3, profile_root=tmp_path)
    profile_checkpoints(tmp_path)
    with pytest.raises(ValueError, match="no counter states"):
        TrainingStates(profile="mixed", team_size=2, profile_root=tmp_path)
    path = tmp_path / "casual/3/0.ss1.json"
    path.write_text(json.dumps({"verified_party": [{}] * 6}))
    with pytest.raises(ValueError, match="metadata disagrees"):
        TrainingStates(profile="casual", team_size=3, profile_root=tmp_path)


@pytest.mark.parametrize("argv", [[], ["--team-profile", "mixed", "--states", "a.ss1", "b.ss1"],
                                  ["--state-dir", "states", "--team-profile", "random"],
                                  ["--team-profile", "random", "--team-size", "7"]])
def test_source_cli_rejects_missing_or_conflicting_selection(argv):
    parser = argparse.ArgumentParser()
    add_state_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_random_selection_is_per_pair_seeded_and_skips_metadata(tmp_path):
    for i in range(5):
        (tmp_path / f"team{i}.ss1").write_bytes(bytes([i]))
    (tmp_path / "team0.ss1.json").write_text("{}")
    (tmp_path / "battery.sav").write_bytes(b"save")
    source = TrainingStates(directory=tmp_path, seed=123)
    assert len(source.files) == 5
    first, second = source.for_pair(0), source.for_pair(1)
    sequence = []
    for _ in range(12):
        sequence.append(first())
        second()  # Other workers cannot change the first worker's RNG sequence.
    again = source.for_pair(0)
    assert sequence == [again() for _ in range(12)]
    assert all(a != b for a, b in sequence)
    assert len(set(sequence)) > 1
    assert len(source.describe()["checkpoints"][0]["sha256"]) == 64


def test_fixed_order_and_single_file_directory(tmp_path):
    first, second = tmp_path / "first.ss1", tmp_path / "second.ss1"
    first.write_bytes(b"a")
    solo = TrainingStates(directory=tmp_path).for_pair(0)
    assert solo() == (first, first)
    second.write_bytes(b"b")
    fixed = TrainingStates(states=[second, first]).for_pair(2)
    assert fixed() == fixed() == (second, first)
    with pytest.raises(ValueError):
        TrainingStates(states=[first])
    with pytest.raises(ValueError):
        TrainingStates(states=[first, second], directory=tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        TrainingStates(states=[first, tmp_path / "missing.ss1"])


def test_starter_teams_only_change_order_and_have_full_stat_budget():
    root = Path(__file__).parents[1]
    first = TeamSpec.load(root / "config/teams/starters_charizard_lead.json")
    second = TeamSpec.load(root / "config/teams/starters_venusaur_lead.json")
    assert [p.species for p in first.party] == ["Charizard", "Blastoise", "Venusaur"]
    assert [p.species for p in second.party] == ["Venusaur", "Blastoise", "Charizard"]
    assert {p.species:p for p in first.party} == {p.species:p for p in second.party}
    catalog = json.loads((root / "config/generation/firered_us_v1_0.json").read_text())
    for mon in first.party:
        assert mon.ivs == (31,)*6 and sum(mon.evs) == 510 and max(mon.evs) == 252
        assert mon.held_item == "Leftovers" and mon.level == 50
        assert set(mon.moves) <= set(catalog["species"][mon.species]["moves"])
