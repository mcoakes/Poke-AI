"""Training metrics: count a linked battle once, and its player episodes twice."""
import json
import math
from collections import deque
from pathlib import Path
from threading import Lock
import time
import uuid

from .battle_blocks import BattleBlocks, format_block


class TrainingLog:
    def __init__(self, directory, announce=print, learning_players=(0, 1), *,
                 resume=False, session_id=None):
        self.learning_players = tuple(learning_players)
        self.directory = Path(directory)
        self.announce = announce
        self.started = time.monotonic()
        self.elapsed_before_session = 0.0
        self.session_id = session_id
        self.session_count = 1
        self.lock = Lock()
        self.completed = self.truncated = self.attempts = 0
        self.player_steps = self.updates = 0
        self.winners = dict(player1=0, player2=0, draw=0)
        self.status = "running"
        self.recent = deque(maxlen=100)
        self.recent_outcomes = deque(maxlen=100)
        self.blocks = BattleBlocks()
        self.recovery_notes = []
        if resume:
            self._restore()
        self.starting_attempts = self.attempts
        self.starting_updates = self.updates
        self.starting_player_timesteps = self.player_steps

    def _restore(self):
        """Replay finished events; downtime is never included in throughput."""
        summary_path = self.directory / "summary.json"
        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                if not isinstance(summary, dict):
                    raise ValueError("expected a JSON object")
            except (OSError, ValueError) as exc:
                raise ValueError(f"cannot continue metrics: invalid {summary_path}: {exc}") from exc
        recorded_learners = summary.get("learning_players")
        if recorded_learners is not None and recorded_learners != [i+1 for i in self.learning_players]:
            raise ValueError("cannot continue metrics with different learning players")
        size = summary.get("battle_block_size", 500)
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("cannot continue metrics: invalid saved battle block size")
        self.blocks = BattleBlocks(size)
        tails = []
        sessions = set()
        has_events = False

        def restore_clock(row):
            if not isinstance(row, dict):
                raise ValueError("cannot continue metrics: saved metric record must be an object")
            elapsed = row.get("elapsed_seconds", 0)
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("cannot continue metrics: invalid saved active elapsed time")
            self.elapsed_before_session = max(self.elapsed_before_session, float(elapsed))

        restore_clock(summary)
        for event in self._saved_events("episodes.jsonl", tails):
            self._validate_episode(event)
            has_events = True
            sessions.add(event.get("session_id") or "legacy")
            self._record_episode(event)
            restore_clock(event.get("metrics", {}))
        for event in self._saved_events("updates.jsonl", tails):
            has_events = True
            sessions.add(event.get("session_id") or "legacy")
            for key in ("timesteps", "updates"):
                if not isinstance(event.get(key), int) or isinstance(event[key], bool) or event[key] < 0:
                    raise ValueError(f"cannot continue metrics: invalid saved update {key}")
            self.updates = max(self.updates, event["updates"])
            self.player_steps = max(self.player_steps, event["timesteps"])
            restore_clock(event)
        for key, attribute in (("updates", "updates"), ("player_timesteps", "player_steps")):
            value = summary.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"cannot continue metrics: invalid saved {key}")
            if value > getattr(self, attribute):
                self.recovery_notes.append(
                    f"Saved summary {key}={value} is ahead of complete update events; "
                    f"restored {getattr(self, attribute)} from the log")
        saved_attempts = summary.get("finished_attempts", 0)
        if not isinstance(saved_attempts, int) or isinstance(saved_attempts, bool) or saved_attempts < 0:
            raise ValueError("cannot continue metrics: invalid saved finished_attempts")
        if saved_attempts > self.attempts:
            self.recovery_notes.append(
                f"Saved summary records {saved_attempts} finished battles, but episodes.jsonl "
                f"contains {self.attempts} complete events; ignored "
                f"{saved_attempts-self.attempts} summary-only events and restored the complete log")
        previous_sessions = summary.get("session_count", 1 if summary or has_events else 0)
        if not isinstance(previous_sessions, int) or isinstance(previous_sessions, bool) or previous_sessions < 0:
            raise ValueError("cannot continue metrics: invalid saved session count")
        self.session_count = max(previous_sessions, len(sessions)) + 1
        for note in self.recovery_notes:
            self.announce(f"Metrics recovery: {note}")
        # Do not mutate logs until all complete records have passed validation.
        for path, offset, tail, newline_only in tails:
            if newline_only:
                with path.open("ab") as stream:
                    stream.write(b"\n")
                continue
            backup = path.with_name(f"{path.name}.partial-{uuid.uuid4().hex[:12]}")
            backup.write_bytes(tail)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            self.announce(f"Recovered incomplete final record in {path.name}; preserved in {backup.name}")

    def _saved_events(self, name, tails):
        path = self.directory / name
        if not path.exists():
            return
        with path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                line_number += 1
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeDecodeError) as exc:
                    if not raw.endswith(b"\n") and not stream.read(1):
                        tails.append((path, offset, raw, False))
                        break
                    raise ValueError(f"cannot continue metrics: malformed {path}:{line_number}: {exc}") from exc
                if not isinstance(event, dict):
                    raise ValueError(f"cannot continue metrics: malformed {path}:{line_number}: expected a JSON object")
                yield event
                if not raw.endswith(b"\n"):
                    tails.append((path, offset, b"", True))

    @staticmethod
    def _validate_episode(event):
        try:
            if not isinstance(event["terminated"], bool) or not isinstance(event["truncated"], bool):
                raise ValueError("terminated/truncated must be boolean")
            if event["terminated"] == event["truncated"]:
                raise ValueError("a finished battle must be terminal or truncated")
            if event["winner"] not in ("player1", "player2", "draw", None):
                raise ValueError("invalid winner")
            if (event["winner"] is not None) != event["terminated"]:
                raise ValueError("winner disagrees with terminal state")
            if len(event["returns"]) != 2 or any(not math.isfinite(float(v)) for v in event["returns"]):
                raise ValueError("invalid paired rewards")
            if not isinstance(event["decisions"], int) or isinstance(event["decisions"], bool) or event["decisions"] < 0:
                raise ValueError("invalid decisions")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot continue metrics: invalid saved episode: {exc}") from exc

    def _record_episode(self, event):
        self.attempts += 1
        self.completed += int(event["terminated"])
        self.truncated += int(event["truncated"])
        self.recent.append((float(event["returns"][0]), float(event["returns"][1]), int(event["decisions"])))
        self.recent_outcomes.append(event["winner"])
        if event["winner"]:
            self.winners[event["winner"]] += 1
        return self.blocks.add(event)

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

    def _summary(self):
        elapsed = self.elapsed_before_session + time.monotonic()-self.started
        rate = self.completed*3600/elapsed if elapsed else 0
        summary = dict(status=self.status, completed_battles=self.completed,
                       truncated_battles=self.truncated, finished_attempts=self.attempts,
                       player_episodes=self.completed*2, episodes_per_hour=rate,
                       player_episodes_per_hour=rate*2, elapsed_seconds=elapsed,
                       player_timesteps=self.player_steps, updates=self.updates,
                       winners_by_player=dict(self.winners))
        summary.update(session_count=self.session_count, session_id=self.session_id,
                       session_elapsed_seconds=elapsed-self.elapsed_before_session)
        if self.recovery_notes:
            summary["metrics_recovery_notes"] = list(self.recovery_notes)
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
        return summary

    def _write(self, summary=None):
        if summary is None:
            summary = self._summary()
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
            event = dict(pair=pair+1, episode=episode, states=[str(path) for path in states],
                         terminated=terminated, truncated=truncated, winner=winner,
                         returns=returns, decisions=decisions, reset_seconds=reset_seconds,
                         battle_seconds=battle_seconds, reason=infos[0].get("reason"),
                         session_id=self.session_id)
            block = self._record_episode(event)
            summary = self._summary()
            event["metrics"] = summary
            with (self.directory / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(event)+"\n")
            if block is not None:
                with (self.directory / "battle-blocks.jsonl").open("a") as stream:
                    stream.write(json.dumps(block) + "\n")
            self._write(summary)
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
            summary = self._summary()
            with (self.directory / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(timesteps=self.player_steps, updates=self.updates,
                                             metrics=metrics, elapsed_seconds=summary["elapsed_seconds"],
                                             session_id=self.session_id))+"\n")
            self._write(summary)

    def finish(self, status):
        with self.lock:
            self.status = status
            return self._write()
