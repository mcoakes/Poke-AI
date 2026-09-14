"""Command policies and wall-clock accounting for repeated paired Gym episodes."""
import json
from pathlib import Path
import re
from threading import Lock
import time


def command_action(value):
    if isinstance(value, dict) and len(value) == 1:
        kind, slot = next(iter(value.items()))
        if type(slot) is int and kind in ("move", "switch"):
            value = f"{kind}{slot}"
    if not isinstance(value, str):
        raise ValueError("commands must be MOVE1..MOVE4 or SWITCH1..SWITCH6")
    match = re.fullmatch(r"(MOVE|SWITCH)([1-6])", value.upper())
    if not match or (match[1] == "MOVE" and int(match[2]) > 4):
        raise ValueError(f"invalid command: {value}")
    return int(match[2])-1 if match[1] == "MOVE" else int(match[2])+3


def command_cycles(values, environments):
    if len(values) == 1 and values[0].lower().endswith(".json"):
        source = json.loads(Path(values[0]).read_text())
        if not isinstance(source, dict) or set(source)-{"name", "players"}:
            raise ValueError("benchmark JSON accepts name and players; repetition is a CLI option")
        players = source.get("players")
        if not isinstance(players, list) or len(players) not in (2, environments):
            raise ValueError("command JSON needs two alternating players or one per environment")
        cycles = []
        for player in players:
            cycle = player.get("cycle") if isinstance(player, dict) else None
            if not isinstance(cycle, list) or not cycle:
                raise ValueError("each player needs a nonempty cycle")
            cycles.append([command_action(value) for value in cycle])
        return [cycles[i % len(cycles)] for i in range(environments)]
    actions = [command_action(value) for value in values]
    if not actions:
        raise ValueError("at least one command is required")
    return [actions.copy() for _ in range(environments)]


class CommandPolicy:
    def __init__(self, cycle, repeat=True):
        self.cycle, self.repeat, self.index = cycle, repeat, 0

    def choose(self, observation):
        mask = observation["action_mask"]
        if observation["context"][1]:
            choices = [a for a in range(4, 10) if mask[a]]
            if not choices:
                raise ValueError("forced replacement has no legal party target")
            return choices[0]  # does not consume a scripted battle command
        if mask[10]:
            return 10  # partner is choosing a replacement
        if not self.repeat and self.index >= len(self.cycle):
            return None
        action = self.cycle[self.index % len(self.cycle)]
        if not mask[action]:
            raise ValueError(f"scripted action {action} is masked; choose another command sequence")
        self.index += 1
        return action


class Throughput:
    def __init__(self, directory, planned, clock=time.monotonic, announce=print):
        self.directory, self.planned, self.clock, self.announce = Path(directory), planned, clock, announce
        self.started = clock()
        self.lock = Lock()
        self.finished = self.completed = self.truncated = self.failed = 0
        self.wins = {"player1":0, "player2":0, "draw":0}
        self.reset_seconds = self.battle_seconds = 0.0

    def snapshot(self):
        elapsed = max(0.0, self.clock()-self.started)
        rate = self.completed*3600/elapsed if elapsed else 0.0
        return dict(planned_episodes=self.planned, finished_attempts=self.finished,
                    completed_battles=self.completed, player_episodes=self.completed*2,
                    truncated=self.truncated, failed=self.failed, winners=dict(self.wins),
                    elapsed_seconds=elapsed, episodes_per_hour=rate, player_episodes_per_hour=rate*2,
                    mean_reset_seconds=self.reset_seconds/self.finished if self.finished else 0,
                    mean_battle_seconds=self.battle_seconds/self.finished if self.finished else 0)

    def write(self, status):
        summary = dict(status=status, **self.snapshot())
        temporary = self.directory / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2)+"\n")
        temporary.replace(self.directory / "summary.json")
        return summary

    def record(self, episode, pair, terminated, truncated, winner, reset_seconds, battle_seconds,
               decisions, reason):
        with self.lock:
            if terminated and (truncated or winner not in self.wins):
                raise ValueError("completed battle needs an actual win/loss/draw result")
            self.finished += 1
            self.completed += int(terminated)
            self.truncated += int(truncated)
            if terminated:
                self.wins[winner] += 1
            self.reset_seconds += reset_seconds
            self.battle_seconds += battle_seconds
            metrics = self.snapshot()
            event = dict(event="episode", episode=episode, pair=pair, terminated=terminated,
                         truncated=truncated, winner=winner, reset_seconds=reset_seconds,
                         battle_seconds=battle_seconds, decisions=decisions, reason=reason, metrics=metrics)
            with (self.directory / "episodes.jsonl").open("a") as log:
                log.write(json.dumps(event)+"\n")
            self.write("running")
            self.announce(f"episode {self.finished}/{self.planned} pair={pair} winner={winner or 'none'} "
                          f"decisions={decisions} truncated={truncated} | "
                          f"{metrics['episodes_per_hour']:.1f} episodes/hour "
                          f"({metrics['player_episodes_per_hour']:.1f} player-episodes/hour)")

    def failure(self, episode, pair, error):
        with self.lock:
            self.failed += 1
            with (self.directory / "episodes.jsonl").open("a") as log:
                log.write(json.dumps(dict(event="failed", episode=episode, pair=pair, error=str(error)))+"\n")
            self.write("failed")

    def finish(self, status):
        with self.lock:
            return self.write(status)
