import json

import pytest

from firered_rl.training_log import TrainingLog


def record(log, winner):
    outcomes = {'player1': ['won', 'lost'], 'player2': ['lost', 'won'],
                'draw': ['draw', 'draw'], None: ['ongoing', 'ongoing']}[winner]
    log.episode(0, states=['a', 'b'], episode=log.attempts + 1,
                terminated=winner is not None, truncated=winner is None,
                infos=[{'outcome': value} for value in outcomes], returns=[-600, -600],
                decisions=30, reset_seconds=0, battle_seconds=1)


def test_win_rates_include_draws_exclude_truncations_and_report_frozen_side(tmp_path):
    lines = []
    log = TrainingLog(tmp_path, announce=lines.append, learning_players=(0,))
    assert log.reward_metrics()['player1_win_rate'] is None
    record(log, None)
    assert 'p1_win=n/a' in lines[-1]
    for winner in ('player1', 'player1', 'player2', 'draw'):
        record(log, winner)
    data = json.loads((tmp_path / 'summary.json').read_text())
    assert data['player1_win_rate'] == .5
    assert data['player2_win_rate'] == .25
    assert data['draw_rate'] == .25
    assert data['win_rate_window_completed'] == 4
    assert data['win_rate_window_truncated'] == 1
    assert data['ep_rew_mean'] == -600
    assert 'p1_win=50.0% p2_win=25.0% draw=25.0% win_n=4 trunc_n=1' in lines[-1]
    event = json.loads((tmp_path / 'episodes.jsonl').read_text().splitlines()[-1])
    assert event['metrics']['player2_win_rate'] == .25


def test_rolling_window_expires_old_results_but_retains_overall_rate(tmp_path):
    log = TrainingLog(tmp_path, announce=lambda _: None)
    record(log, 'player1')
    for _ in range(100):
        record(log, 'player2')
    data = log.finish('completed')
    assert data['reward_window_battles'] == 100
    assert data['player1_win_rate'] == 0
    assert data['player2_win_rate'] == 1
    assert data['player1_win_rate_overall'] == pytest.approx(1 / 101)
    assert data['player2_win_rate_overall'] == pytest.approx(100 / 101)
