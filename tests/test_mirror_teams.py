import argparse
from pathlib import Path

import pytest

from firered_rl.training_states import TrainingStates, add_state_arguments
from firered_rl.training_teams import TrainingTeams


@pytest.mark.parametrize('backend', ['mgba', 'showdown'])
def test_random_mirrors_vary_and_repeat_with_seed(tmp_path, backend):
    if backend == 'mgba':
        for i in range(5):
            (tmp_path / f'{i}.ss1').write_bytes(b'fixture')
        cls = TrainingStates
    else:
        root = Path(__file__).resolve().parents[1]
        team = (root / 'config/teams/fire_vs_water.json').read_bytes()
        for i in range(5):
            (tmp_path / f'{i}.json').write_bytes(team)
        cls = TrainingTeams
    source = cls(directory=tmp_path, seed=12, mirror_teams=True)
    first, second = source.for_pair(0), source.for_pair(0)
    results = [first() for _ in range(30)]
    assert results == [second() for _ in range(30)]
    assert all(a == b for a, b in results)
    assert len({a for a, _ in results}) > 1
    assert source.describe()['mirror_teams'] is True
    assert 'mirrored' in source.label


def test_profile_mirrors_keep_category_pool_and_size():
    root = Path(__file__).resolve().parents[1]
    source = TrainingTeams(profile='mixed', team_size=3,
                           profile_root=root / 'config/teams/profiles', mirror_teams=True)
    choose = source.for_pair(1)
    results = [choose() for _ in range(100)]
    assert all(a == b and a.parent.name == '3' for a, b in results)
    assert {a.parent.parent.name for a, _ in results} == {'competitive', 'casual', 'random'}
    assert source.describe()['category_sampling'] == 'uniform_per_battle'


def test_cli_and_fixed_input_validation(tmp_path):
    parser = argparse.ArgumentParser()
    add_state_arguments(parser, include_teams=True)
    args = parser.parse_args(['--team-profile', 'mixed', '--mirror-teams'])
    assert args.mirror_teams
    with pytest.raises(ValueError, match='requires a random'):
        TrainingStates(states=[tmp_path/'a.ss1'] * 2, mirror_teams=True)
    with pytest.raises(ValueError, match='requires a random'):
        TrainingTeams(teams=[tmp_path/'a.json'] * 2, mirror_teams=True)
