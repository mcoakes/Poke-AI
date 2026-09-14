import json
from pathlib import Path
import subprocess
import sys

from firered_rl.battle_blocks import BattleBlocks
from firered_rl.training_log import TrainingLog


def event(winner):
    return dict(terminated=winner is not None, truncated=winner is None,
                winner=winner, returns=[10, -20], decisions=5)


def test_disjoint_blocks_with_draw_and_truncation():
    blocks = BattleBlocks(4)
    for winner in ('player1', 'player2', 'draw'):
        assert blocks.add(event(winner)) is None
    first = blocks.add(event(None))
    assert first['completed'] == 3
    assert first['truncated'] == 1
    assert first['p1_win_rate'] == 1 / 3
    assert first['p2_win_rate'] == 1 / 3
    assert first['draw_rate'] == 1 / 3
    assert first['p1_reward_mean'] == 10
    blocks.add(event('player2'))
    partial = blocks.snapshot()
    assert (partial['first_episode'], partial['last_episode']) == (5, 5)
    assert partial['p2_win_rate'] == 1
    assert not partial['full']


def test_live_report_after_500_ended_battles(tmp_path):
    messages = []
    log = TrainingLog(tmp_path, announce=messages.append)
    for i in range(500):
        log.episode(0, states=['a', 'b'], episode=i + 1,
                    terminated=True, truncated=False,
                    infos=[{'outcome': 'won'}, {'outcome': 'lost'}],
                    returns=[10, -20], decisions=5, reset_seconds=0, battle_seconds=1)
    rows = (tmp_path / 'battle-blocks.jsonl').read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])['p1_win_rate'] == 1
    assert messages[-1].startswith('BLOCK 1 episodes=1–500')
    summary = log.finish('completed')
    assert summary['last_complete_battle_block']['episodes'] == 500
    assert summary['battle_block_progress']['episodes'] == 0


def test_compare_saved_run_includes_partial_and_ignores_unfinished_write(tmp_path):
    (tmp_path / 'episodes.jsonl').write_text(
        '\n'.join(json.dumps(event(w)) for w in ('player1', 'player2', None)) + '\n{"part')
    script = Path(__file__).resolve().parents[1] / 'mgba/compare_blocks.py'
    result = subprocess.run([sys.executable, str(script), '--run-dir', str(tmp_path),
                             '--block-size', '2', '--json'], capture_output=True, text=True, check=True)
    rows = json.loads(result.stdout)
    assert len(rows) == 2
    assert rows[0]['p1_win_rate'] == .5
    assert rows[1]['p1_win_rate'] is None
    assert rows[1]['truncated'] == 1
    assert rows[1]['first_episode'] == 3
