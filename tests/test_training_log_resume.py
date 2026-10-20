import json

import pytest

from firered_rl.training_log import TrainingLog


def event(index, winner="player1", **extra):
    row = dict(pair=1, episode=index, states=["a", "b"],
               terminated=winner is not None, truncated=winner is None,
               winner=winner, returns=[index, -index], decisions=index % 10 + 1,
               reset_seconds=0, battle_seconds=1)
    row.update(extra)
    return row


def write_events(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def record(log, episode=1, winner="player1"):
    outcomes = {"player1": ["won", "lost"], "player2": ["lost", "won"],
                "draw": ["draw", "draw"], None: ["ongoing", "ongoing"]}[winner]
    log.episode(0, states=["a", "b"], episode=episode,
                terminated=winner is not None, truncated=winner is None,
                infos=[{"outcome": outcome} for outcome in outcomes],
                returns=[5, -5], decisions=10, reset_seconds=0, battle_seconds=1)


def test_continue_keeps_cumulative_metrics_and_excludes_downtime(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("firered_rl.training_log.time.monotonic", lambda: clock[0])
    original = TrainingLog(tmp_path, announce=lambda _: None, learning_players=(0,), session_id="000001")
    clock[0] += 10
    record(original)
    record(original, episode=2, winner=None)
    original.update(128, 4, {"loss": 0.5})
    original.finish("completed")
    episode_bytes = (tmp_path / "episodes.jsonl").read_bytes()
    update_bytes = (tmp_path / "updates.jsonl").read_bytes()

    clock[0] += 50000
    continued = TrainingLog(tmp_path, announce=lambda _: None, learning_players=(0,),
                            resume=True, session_id="000002")
    assert continued.starting_attempts == 2
    assert continued.starting_updates == 4
    assert continued.starting_player_timesteps == 128
    assert continued.completed == 1
    assert continued.truncated == 1
    assert continued.status == "running"
    assert continued.reward_metrics()["win_rate_window_truncated"] == 1
    clock[0] += 5
    record(continued, winner="player2")
    continued.update(160, continued.starting_updates + 1, {})
    result = continued.finish("completed")
    assert result["elapsed_seconds"] == 15
    assert result["session_elapsed_seconds"] == 5
    assert result["session_count"] == 2
    assert result["finished_attempts"] == 3
    assert result["completed_battles"] == 2
    assert result["winners_by_player"] == {"player1": 1, "player2": 1, "draw": 0}
    assert result["episodes_per_hour"] == 480
    assert result["updates"] == 5
    assert (tmp_path / "episodes.jsonl").read_bytes().startswith(episode_bytes)
    assert (tmp_path / "updates.jsonl").read_bytes().startswith(update_bytes)
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [row["episode"] for row in rows] == [1, 2, 1]
    assert [row["session_id"] for row in rows] == ["000001", "000001", "000002"]
    assert json.loads((tmp_path / "updates.jsonl").read_text().splitlines()[-1])["session_id"] == "000002"


def test_replay_restores_recent_window_and_blocks_without_reemitting(tmp_path):
    rows = [event(index, "player1" if index < 10 else "player2") for index in range(1, 1000)]
    write_events(tmp_path / "episodes.jsonl", rows)
    old_blocks = b'{"block": 1, "preserved": true}\n'
    (tmp_path / "battle-blocks.jsonl").write_bytes(old_blocks)
    continued = TrainingLog(tmp_path, announce=lambda _: None, resume=True)
    assert continued.starting_attempts == 999
    assert continued.blocks.finished == 1
    assert continued.blocks.last_complete["last_episode"] == 500
    assert continued.blocks.snapshot()["episodes"] == 499
    assert continued.reward_metrics()["reward_window_battles"] == 100
    assert continued.reward_metrics()["player1_ep_rew_mean"] == pytest.approx(sum(range(900, 1000)) / 100)
    assert continued.reward_metrics()["player2_win_rate"] == 1
    assert (tmp_path / "battle-blocks.jsonl").read_bytes() == old_blocks
    record(continued)
    blocks = (tmp_path / "battle-blocks.jsonl").read_text().splitlines()
    assert len(blocks) == 2
    assert json.loads(blocks[1])["block"] == 2
    assert json.loads(blocks[1])["last_episode"] == 1000


def test_stale_summary_does_not_drop_new_complete_episode_or_update(tmp_path):
    write_events(tmp_path / "episodes.jsonl", [event(1), event(2, "draw", metrics={"elapsed_seconds": 20})])
    write_events(tmp_path / "updates.jsonl", [dict(timesteps=32, updates=1, elapsed_seconds=10),
                                            dict(timesteps=64, updates=2, elapsed_seconds=30)])
    (tmp_path / "summary.json").write_text(json.dumps(dict(finished_attempts=1, updates=1,
                                                         player_timesteps=32, elapsed_seconds=10)))
    continued = TrainingLog(tmp_path, announce=lambda _: None, resume=True)
    assert continued.attempts == 2
    assert continued.winners["draw"] == 1
    assert continued.updates == 2
    assert continued.player_steps == 64
    assert continued.elapsed_before_session == 30


def test_incomplete_tail_is_preserved_and_new_records_remain_readable(tmp_path):
    write_events(tmp_path / "episodes.jsonl", [event(1)])
    fragment = b'{"pair": 2, "returns": ['
    with (tmp_path / "episodes.jsonl").open("ab") as stream:
        stream.write(fragment)
    messages = []
    continued = TrainingLog(tmp_path, announce=messages.append, resume=True)
    assert continued.attempts == 1
    assert "incomplete final record" in messages[-1]
    backups = list(tmp_path.glob("episodes.jsonl.partial-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == fragment
    record(continued)
    assert len([json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]) == 2


def test_valid_unterminated_last_json_line_is_retained(tmp_path):
    (tmp_path / "episodes.jsonl").write_text(json.dumps(event(1)))
    continued = TrainingLog(tmp_path, announce=lambda _: None, resume=True)
    record(continued)
    rows = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert not list(tmp_path.glob("*.partial-*"))


@pytest.mark.parametrize("broken", [b'{"pair":\n', b'not json\n', b'[]\n'])
def test_malformed_complete_or_interior_line_fails_without_modification(tmp_path, broken):
    original = json.dumps(event(1)).encode() + b"\n" + broken + json.dumps(event(2)).encode() + b"\n"
    (tmp_path / "episodes.jsonl").write_bytes(original)
    with pytest.raises(ValueError, match=r"malformed .*episodes.jsonl:2"):
        TrainingLog(tmp_path, announce=lambda _: None, resume=True)
    assert (tmp_path / "episodes.jsonl").read_bytes() == original
    assert not list(tmp_path.glob("*.partial-*"))


def test_summary_ahead_warns_and_uses_complete_events(tmp_path):
    write_events(tmp_path / "episodes.jsonl", [event(1)])
    (tmp_path / "summary.json").write_text(json.dumps(dict(finished_attempts=2, updates=2,
                                                         player_timesteps=64, elapsed_seconds=10)))
    messages = []
    continued = TrainingLog(tmp_path, announce=messages.append, resume=True)
    assert continued.attempts == 1
    assert continued.updates == continued.player_steps == 0
    assert any("1 summary-only events" in message for message in messages)
    result = continued.finish("completed")
    assert result["finished_attempts"] == 1
    assert result["metrics_recovery_notes"]


def test_different_learning_players_are_rejected(tmp_path):
    original = TrainingLog(tmp_path, announce=lambda _: None, learning_players=(0,))
    record(original)
    with pytest.raises(ValueError, match="different learning players"):
        TrainingLog(tmp_path, announce=lambda _: None, learning_players=(0, 1), resume=True)


def test_complete_json_with_invalid_episode_is_not_silently_dropped(tmp_path):
    write_events(tmp_path / "episodes.jsonl", [event(1, returns=[float("nan"), 1])])
    with pytest.raises(ValueError, match="invalid paired rewards"):
        TrainingLog(tmp_path, announce=lambda _: None, resume=True)


def test_valid_json_of_wrong_type_at_unterminated_tail_is_not_discarded(tmp_path):
    (tmp_path / "episodes.jsonl").write_text("[]")
    with pytest.raises(ValueError, match="expected a JSON object"):
        TrainingLog(tmp_path, announce=lambda _: None, resume=True)
    assert (tmp_path / "episodes.jsonl").read_text() == "[]"
