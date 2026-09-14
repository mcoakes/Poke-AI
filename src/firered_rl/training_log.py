"""Training metrics: count a linked battle once, and its player episodes twice."""
import json
from collections import deque
from pathlib import Path
from threading import Lock
import time

from .battle_blocks import BattleBlocks, format_block


class TrainingLog:
    def __init__(self, directory, announce=print, learning_players=(0, 1)):
        self.learning_players = tuple(learning_players)
        self.directory = Path(directory)
        self.announce = announce
        self.started = time.monotonic()
        self.lock = Lock()
        self.completed = self.truncated = self.attempts = 0
        self.player_steps = self.updates = 0
        self.winners = dict(player1=0, player2=0, draw=0)
        self.status = "running"
        self.recent = deque(maxlen=100)
        self.recent_outcomes = deque(maxlen=100)
        self.blocks = BattleBlocks()

    def _reward_metrics(self):
        count = len(self.recent)
        return dict(
            ep_rew_mean=sum(row[i] for row in self.recent for i in self.learning_players)
                /(len(self.learning_players)*count) if count else None,
            player1_ep_rew_mean=sum(a for a, _, _ in self.recent)/count if count else None,
            player2_ep_rew_mean=sum(b for _, b, _ in self.recent)/count if count else None,
            ep_len_mean=sum(length for _, _, length in self.recent)/count if count else None,
            reward_window_battles=count,
            **self._win_metrics())

    def _win_metrics(self):
        completed = sum(winner is not None for winner in self.recent_outcomes)
        return dict(
            player1_win_rate=self.recent_outcomes.count("player1") / completed if completed else None,
            player2_win_rate=self.recent_outcomes.count("player2") / completed if completed else None,
            draw_rate=self.recent_outcomes.count("draw") / completed if completed else None,
            win_rate_window_completed=completed,
            win_rate_window_truncated=self.recent_outcomes.count(None))

    def reward_metrics(self):
        with self.lock:
            return self._reward_metrics()

    def _write(self):
        elapsed = time.monotonic()-self.started
        rate = self.completed*3600/elapsed if elapsed else 0
        summary = dict(status=self.status, completed_battles=self.completed,
                       truncated_battles=self.truncated, finished_attempts=self.attempts,
                       player_episodes=self.completed*2, episodes_per_hour=rate,
                       player_episodes_per_hour=rate*2, elapsed_seconds=elapsed,
                       player_timesteps=self.player_steps, updates=self.updates,
                       winners_by_player=dict(self.winners))
        summary.update(learning_players=[i+1 for i in self.learning_players],
                       learner_episodes=self.completed*len(self.learning_players),
                       learner_episodes_per_hour=rate*len(self.learning_players))
        summary.update(self._reward_metrics())
        summary.update(battle_block_size=self.blocks.size,
                       battle_block_progress=self.blocks.snapshot(),
                       last_complete_battle_block=self.blocks.last_complete)
        summary.update(
            player1_win_rate_overall=self.winners["player1"] / self.completed if self.completed else None,
            player2_win_rate_overall=self.winners["player2"] / self.completed if self.completed else None,
            draw_rate_overall=self.winners["draw"] / self.completed if self.completed else None)
        temporary = self.directory / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2)+"\n")
        temporary.replace(self.directory / "summary.json")
        return summary

    def episode(self, pair, *, states, episode, terminated, truncated, infos, returns,
                decisions, reset_seconds, battle_seconds):
        outcomes = [info.get("outcome") for info in infos]
        winner = None
        if terminated:
            if outcomes == ["won", "lost"]:
                winner = "player1"
            elif outcomes == ["lost", "won"]:
                winner = "player2"
            elif outcomes == ["draw", "draw"]:
                winner = "draw"
            else:
                raise ValueError(f"invalid paired terminal outcomes: {outcomes}")
        with self.lock:
            self.attempts += 1
            self.completed += int(terminated)
            self.truncated += int(truncated)
            self.recent.append((float(returns[0]), float(returns[1]), int(decisions)))
            self.recent_outcomes.append(winner)
            if winner:
                self.winners[winner] += 1
            block = self.blocks.add(dict(terminated=terminated, truncated=truncated,
                                         winner=winner, returns=returns, decisions=decisions))
            if block is not None:
                with (self.directory / "battle-blocks.jsonl").open("a") as stream:
                    stream.write(json.dumps(block) + "\n")
            summary = self._write()
            event = dict(pair=pair+1, episode=episode, states=[str(path) for path in states],
                         terminated=terminated, truncated=truncated, winner=winner,
                         returns=returns, decisions=decisions, reset_seconds=reset_seconds,
                         battle_seconds=battle_seconds, reason=infos[0].get("reason"),
                         metrics=summary)
            with (self.directory / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(event)+"\n")
            def percent(key):
                value = summary[key]
                return f"{value:.1%}" if value is not None else "n/a"

            self.announce(f"battle={self.attempts} pair={pair+1} winner={winner or 'none'} "
                          f"decisions={decisions} truncated={truncated} | "
                          f"ep_rew_mean={summary['ep_rew_mean']:.3f} "
                          f"p1_mean={summary['player1_ep_rew_mean']:.3f} "
                          f"p2_mean={summary['player2_ep_rew_mean']:.3f} "
                          f"ep_len_mean={summary['ep_len_mean']:.1f} | "
                          f"p1_win={percent('player1_win_rate')} "
                          f"p2_win={percent('player2_win_rate')} "
                          f"draw={percent('draw_rate')} "
                          f"win_n={summary['win_rate_window_completed']} "
                          f"trunc_n={summary['win_rate_window_truncated']} | "
                          f"{summary['episodes_per_hour']:.1f} battles/hour "
                          f"({summary['player_episodes_per_hour']:.1f} player-episodes/hour)")
            if block is not None:
                self.announce(format_block(block))

    def update(self, player_steps, updates, metrics):
        with self.lock:
            self.player_steps, self.updates = int(player_steps), int(updates)
            summary = self._write()
            with (self.directory / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(timesteps=self.player_steps, updates=self.updates,
                                             metrics=metrics, elapsed_seconds=summary["elapsed_seconds"]))+"\n")

    def finish(self, status):
        with self.lock:
            self.status = status
            return self._write()
