from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import pytest

from firered_rl.benchmark import command_cycles, CommandPolicy, Throughput
from firered_rl.linked_env import PairedCalls


def test_commands_and_forced_replacements_do_not_consume_script():
    assert command_cycles(["MOVE1", "SWITCH2"], 4) == [[0, 5]]*4
    assert command_cycles([str(Path(__file__).parents[1]/"config/commands/first_move.json")], 8) == [[0]]*8
    policy = CommandPolicy([0], repeat=False)
    obs = {"action_mask":np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), "context":np.array([0, 1, 0, 0])}
    assert policy.choose(obs) == 5 and policy.index == 0
    obs["context"][1] = 0
    assert policy.choose(obs) == 0
    assert policy.choose(obs) is None
    with pytest.raises(ValueError):
        command_cycles(["MOVE5"], 2)


def test_wall_clock_rate_counts_matches_once_and_excludes_truncations(tmp_path):
    now = [0.0]
    meter = Throughput(tmp_path, 3, clock=lambda: now[0], announce=lambda _: None)
    now[0] = 180.0
    for number in (1, 2):
        meter.record(number, number, True, False, "player2", 150, 30, 1, "battle_complete")
    summary = meter.snapshot()
    assert summary["episodes_per_hour"] == 40
    assert summary["player_episodes_per_hour"] == 80
    now[0] = 360.0
    meter.record(3, 1, False, True, None, 150, 30, 1, "commands_exhausted")
    summary = meter.finish("completed")
    assert summary["episodes_per_hour"] == 20 and summary["completed_battles"] == 2
    assert summary["truncated"] == 1
    assert json.loads((tmp_path/"summary.json").read_text())["winners"]["player2"] == 2
    assert len((tmp_path/"episodes.jsonl").read_text().splitlines()) == 3


def test_gym_barrier_runs_one_joint_operation_for_two_players():
    calls = []
    def reset(values):
        calls.append(("reset", values))
        return ["r0", "r1"]
    def step(values):
        calls.append(("step", values))
        return values
    pair = PairedCalls(reset, step, timeout=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for op, expected in (("reset", ["r0", "r1"]), ("step", [0, 1]), ("step", [0, 1])):
            futures = [executor.submit(pair.call, i, op, i) for i in range(2)]
            assert [f.result(timeout=2) for f in futures] == expected
    assert len(calls) == 3


def test_gym_barrier_propagates_failure_to_both_players():
    def fail(_):
        raise ValueError("bad reset")
    pair = PairedCalls(fail, fail, timeout=1)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pair.call, i, "reset") for i in range(2)]
        for future in futures:
            with pytest.raises((ValueError, RuntimeError), match="bad reset"):
                future.result(timeout=2)
