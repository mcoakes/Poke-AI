"""Two Gymnasium player views sharing one native cable session and decision barrier."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from threading import Condition, Event
import time

import gymnasium as gym

from firered_rl.backends.mgba_bridge import MGBABridgeBackend
from firered_rl.battle_rules import required_switch, restrictions
from firered_rl.link_inspection import pair_inspections, terminal_inspections
from firered_rl.link_test import connection_errors
from firered_rl.linked_reward import EpisodeRewardState, LinkedRewardConfig, reward_points
from firered_rl.readable_log import format_event
from firered_rl.battle_journal import BattleJournal
from firered_rl.turn_env import encode, observation_space


def local_mon(report):
    return next(m for m in report["battle"]["battlers"] if m["side"] == "player")


def local_menu(report):
    return next(iter(report["battle"].get("menus", [])), {})


def actions_submitted(states, reports, payloads):
    """Do not cancel a peer that is still navigating to its chosen action."""
    finished = {"replacement_ready", "choice_ready", "turn_complete", "battle_complete"}
    return all(s.get("phase") in finished or p["kind"] == "human" or
               (s.get("phase") == "resolving" and local_menu(r).get("state") == "busy_or_unknown")
               for s, r, p in zip(states, reports, payloads))


def linked_observation(report, terminal=False):
    source = deepcopy(report)
    # A fainted local battler still has a replacement decision in singles.
    for mon in source["battle"].get("battlers", []):
        if mon.get("valid") and mon["side"] == "player" and local_menu(source).get("state") == "party":
            mon["absent"] = False
    obs = encode(source, terminal=terminal)
    obs["action_mask"][10] = 0  # linked action 10 is WAIT, never escape
    if not terminal and local_menu(source).get("state") == "busy_or_unknown":
        obs["action_mask"][:] = 0
        obs["action_mask"][10] = 1
    return obs


class PairedCalls:
    """Invoke one joint operation after both player threads submit their request."""
    def __init__(self, reset, step, timeout, cancel=None):
        self.functions = {"reset":reset, "step":step}
        self.timeout = timeout
        self.cancel = cancel or Event()
        self.condition = Condition()
        self.pending, self.results = {}, {}
        self.operation = None
        self.failure = None

    def abort(self, error):
        with self.condition:
            self.failure = error
            self.condition.notify_all()

    def call(self, player, operation, value=None):
        deadline = time.monotonic()+self.timeout
        def wait():
            if self.failure is not None:
                raise RuntimeError(str(self.failure)) from self.failure
            if self.cancel.is_set():
                raise RuntimeError("paired Gymnasium operation cancelled")
            if time.monotonic() >= deadline:
                self.failure = TimeoutError("both Gymnasium players must submit matching operations concurrently")
                self.condition.notify_all()
                raise self.failure
            self.condition.wait(0.05)
        with self.condition:
            while self.results or player in self.pending:
                wait()
            if self.failure is not None or self.cancel.is_set():
                raise RuntimeError("paired Gymnasium session is closed or failed")
            if self.operation is not None and self.operation != operation:
                self.failure = ValueError("both players must call reset together or step together")
                self.condition.notify_all()
                raise self.failure
            self.operation = operation
            self.pending[player] = value
            compute = len(self.pending) == 2
            values = [self.pending.get(i) for i in range(2)]
        if compute:
            try:
                results = self.functions[operation](values)
            except BaseException as error:
                self.abort(error)
                raise
            with self.condition:
                self.results = dict(enumerate(results))
                self.pending, self.operation = {}, None
                self.condition.notify_all()
        with self.condition:
            while player not in self.results:
                wait()
            result = self.results.pop(player)
            self.condition.notify_all()
            return result


class LinkedBattleSession:
    """Reset both cores through a supplied lifecycle factory; submit joint actions."""
    def __init__(self, launch_episode, *, timeout=120, reset_timeout=420, max_steps=200,
                 cancel=None, release_episode=None, human_player=None, reward_config=None, readable_log=None, battle_log=None):
        self.battle_journal = BattleJournal(battle_log) if battle_log is not None else None
        self.readable_log = readable_log
        self.reward_config = reward_config or LinkedRewardConfig()
        if human_player is not None and human_player not in (0, 1):
            raise ValueError("human_player must be 0, 1, or None")
        self.human_player = human_player
        self.launch_episode = launch_episode
        self.release_episode = release_episode or (lambda runtime: runtime.close())
        self.timeout, self.max_steps = timeout, max_steps
        self.cancel = cancel or Event()
        self.runtime = None
        self.bridges = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.calls = PairedCalls(self._reset, self._step, max(timeout, reset_timeout)+10, self.cancel)
        self.reports, self.steps, self.done = None, 0, True
        self.closed = False

    def players(self):
        return [LinkedPlayerEnv(self, i) for i in range(2)]

    def reset_pair(self, seeds=None):
        """Joint caller API: reset once when both players are scheduled together."""
        if self.closed:
            raise RuntimeError("linked session is closed")
        if seeds is not None and len(seeds) != 2:
            raise ValueError("a paired reset needs two seeds")
        return self._reset([None, None] if seeds is None else seeds)

    def step_pair(self, actions):
        """Joint caller API: submit both choices without a per-player thread barrier."""
        if self.closed:
            raise RuntimeError("linked session is closed")
        if len(actions) != 2:
            raise ValueError("a paired step needs two actions")
        return self._step(actions)

    def request(self, method, payloads=None):
        futures = [self.executor.submit(getattr(b, method), **p)
                   for b, p in zip(self.bridges, payloads or [{}, {}])]
        results, errors = [], []
        for port, future in zip(self.runtime.ports, futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(f"port {port} {method}: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))
        return results

    def record(self, event, **fields):
        with (self.runtime.run / "gym-steps.jsonl").open("a") as log:
            log.write(json.dumps(dict(event=event, **fields))+"\n")
        if self.battle_journal is not None:
            self.battle_journal.record(event, self.runtime.run.name, fields.get("reports", self.reports), fields)
        if self.readable_log is not None and event in ("reset", "submitted", "completed", "failed"):
            reports = fields.get("reports", self.reports)
            observations = ([] if event == "failed" else
                            [linked_observation(r, fields.get("terminated", False)) for r in reports])
            text = format_event(event, self.runtime.run.name, reports, observations, fields)
            self.readable_log.parent.mkdir(parents=True, exist_ok=True)
            with self.readable_log.open("a", encoding="utf-8") as log:
                log.write(text)

    def read_reports(self, allow_absent=False):
        # A human can finish a switch while the two socket inspections run.
        # Retry only that known transient; all identity/legality checks remain.
        deadline = time.monotonic()+2
        while True:
            try:
                return pair_inspections(self.request("inspect"), self.runtime.ports, allow_absent=allow_absent)
            except ValueError as error:
                if (self.human_player is None or "local party index disagrees" not in str(error)
                    or self.cancel.is_set() or time.monotonic() >= deadline):
                    raise
                self.cancel.wait(0.02)

    def _release(self):
        for bridge in self.bridges:
            try:
                bridge.link_turn_cancel()
            except (OSError, RuntimeError, ValueError):
                pass
            bridge.close()
        self.bridges = []
        if self.runtime is not None:
            self.release_episode(self.runtime)
            self.runtime = None

    def _reset(self, seeds):
        self._release()
        self.runtime = self.launch_episode()
        self.bridges = [MGBABridgeBackend("127.0.0.1", port) for port in self.runtime.ports]
        statuses = self.request("status")
        if self.battle_journal is not None and any(not s.get("battle_events_supported") for s in statuses):
            raise RuntimeError("compact battle logs require adapter 0.16.0; restart with the updated scripts")
        if any(not s.get("link_move_supported") or not s.get("link_peer_simulation_supported")
               or not s.get("native_link_session") or s.get("link_session_id") != self.runtime.session_id
               for s in statuses):
            raise RuntimeError("linked Gymnasium episodes require adapter 0.12.0 in the assigned session")
        if [s.get("link_player") for s in statuses] != [1, 2]:
            raise RuntimeError("unexpected linked player assignment")
        if self.human_player is not None and any(not s.get("link_human_supported") for s in statuses):
            raise RuntimeError("human play requires adapter 0.14.0; extract the update and restart")
        errors = connection_errors(self.request("link_status"), require_battle=True)
        if errors:
            raise RuntimeError("; ".join(errors))
        self.reports = self.read_reports()
        if any(i != self.human_player and local_menu(r).get("state") != "command"
               for i, r in enumerate(self.reports)):
            raise ValueError("reset did not reach both command menus")
        self.steps, self.done = 0, False
        self.reward_state = EpisodeRewardState(self.reports)
        self.record("reset", reports=self.reports, seeds=seeds, seed_applied=False)
        return [(linked_observation(r), dict(player=i+1, seed_applied=False))
                for i, r in enumerate(self.reports)]

    def _step(self, actions):
        if self.done:
            raise RuntimeError("reset both players before stepping")
        previous_reports = self.reports
        payloads = []
        start = self.reports[0]["battle"]["turn"]
        if start >= 255:
            raise ValueError("native turn counter is saturated; reset the pair")
        for i, (report, action) in enumerate(zip(self.reports, actions)):
            if i == self.human_player:
                if action is not None:
                    raise ValueError("human actions come from mGBA controls; submit None for that player")
                payloads.append(dict(request_id=self.steps+1, kind="human", slot=1,
                                     expected_turn=start))
                continue
            obs = linked_observation(report)
            if not gym.spaces.Discrete(11).contains(action) or not obs["action_mask"][action]:
                raise ValueError(f"action {action} is unavailable for this player")
            mon = local_mon(report)
            forced = required_switch(report)
            kind = "move" if action < 4 else "switch" if action < 10 else "wait"
            payloads.append(dict(request_id=self.steps+1, kind=kind,
                slot=int(action)+1 if kind == "move" else int(action)-3 if kind == "switch" else 1,
                expected_turn=start, replacement=forced,
                skip_move_menu=kind == "move" and restrictions(report["battle"], mon)["skip_move_menu"]))
            if self.human_player is not None:
                payloads[-1]["human_opponent"] = True
        if all(p["kind"] == "wait" for p in payloads):
            raise ValueError("both players cannot wait without a pending replacement")
        actions = [None if a is None else int(a) for a in actions]
        if self.battle_journal is not None:
            for payload in payloads:
                payload["battle_log"] = True
        self.record("submitted", decision=self.steps+1, actions=actions, payloads=payloads)
        try:
            prepared = self.request("link_turn_prepare", payloads)
            if self.battle_journal is not None:
                self.battle_journal.collect(prepared)
            if any(s.get("phase") != "prepared" or s.get("request_id") != self.steps+1 for s in prepared):
                raise RuntimeError("both actions must be prepared before commit")
            committed = self.request("link_turn_commit", [dict(request_id=self.steps+1)]*2)
            if self.battle_journal is not None:
                self.battle_journal.collect(committed)
            deadline = time.monotonic()+self.timeout
            phases = None
            terminal, reason = False, "turn_complete"
            while True:
                if self.cancel.is_set():
                    raise RuntimeError("benchmark cancelled")
                states = self.request("link_turn_status", [dict(events_after=self.battle_journal.seen)]*2
                                      if self.battle_journal is not None else None)
                if self.battle_journal is not None:
                    self.battle_journal.collect(states)
                if [s.get("phase") for s in states] != phases:
                    phases = [s.get("phase") for s in states]
                    self.record("progress", decision=self.steps+1, reports=states)
                if any(s.get("request_id") != self.steps+1 for s in states):
                    raise RuntimeError(f"paired action failed: {states}")
                master = next(i for i, r in enumerate(self.reports) if r["battle"]["type_flags"] & 4)
                snapshot = states[master].get("terminal_inspection")
                if snapshot is not None:
                    self.reports = terminal_inspections(snapshot, self.reports, master, self.runtime.ports)
                    terminal, reason = True, "battle_complete"
                    break
                if any(s.get("error") for s in states):
                    raise RuntimeError(f"paired action failed: {states}")
                if states[master].get("phase") == "capturing_terminal":
                    # A peer's menu may finish while the master retries a party
                    # checksum. Wait for the terminal snapshot before reading
                    # ordinary turn boundaries or calculating rewards.
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"terminal capture timed out: {states}")
                    self.cancel.wait(0.02)
                    continue
                if any(s.get("phase") in ("replacement_ready", "choice_ready") for s in states):
                    if all(s.get("phase") in ("resolving", "replacement_ready", "choice_ready",
                           "turn_complete", "battle_complete", "observing") for s in states):
                        reports = self.read_reports(allow_absent=True)
                        if actions_submitted(states, reports, payloads):
                            replacement = any(i != self.human_player and required_switch(r)
                                              for i, r in enumerate(reports))
                            if not replacement and not any(local_menu(r).get("state") == "command" for r in reports):
                                raise ValueError("waiting boundary did not reach a selection menu")
                            self.reports = reports
                            fainted = any(i != self.human_player and required_switch(r)
                                          and local_mon(r)["current_hp"] == 0 for i, r in enumerate(reports))
                            reason = "forced_replacement" if fainted else "required_switch" if replacement else "waiting_boundary"
                            break
                if all(s.get("phase") == "turn_complete" for s in states):
                    self.reports = self.read_reports()
                    if self.reports[master]["battle"]["turn"] <= start:
                        raise ValueError("master did not advance the battle turn")
                    break
                if any(s.get("phase") == "turn_complete" for s in states):
                    reports = self.read_reports(allow_absent=True)
                    # One player may need a new choice while the other is locked
                    # into Fly, recharge, etc. Yield WAIT for the automatic side.
                    if reports[master]["battle"]["turn"] > start and all(
                        i == self.human_player or local_menu(r).get("state") == "command" or
                        (local_menu(r).get("state") == "busy_or_unknown" and
                         restrictions(r["battle"], local_mon(r))["automatic"]) for i, r in enumerate(reports)):
                        self.reports = reports
                        reason = "automatic_continuation"
                        break
                if any(not s.get("active") and s.get("phase") not in
                       ("turn_complete", "battle_complete", "replacement_ready", "choice_ready") for s in states):
                    raise RuntimeError(f"paired action was cancelled: {states}")
                if self.human_player is None and time.monotonic() >= deadline:
                    raise TimeoutError(f"paired action timed out: {states}")
                self.cancel.wait(0.02)
        except BaseException as error:
            self.done = True
            self.record("failed", error=str(error))
            raise
        finally:
            for bridge in self.bridges:
                try:
                    bridge.link_turn_cancel()
                except (OSError, RuntimeError, ValueError):
                    pass
        self.steps += 1
        truncated = not terminal and (self.steps >= self.max_steps or self.reports[0]["battle"]["turn"] >= 255)
        self.done = terminal or truncated
        rewards, components = reward_points(previous_reports, self.reports, self.reward_config, terminal,
                                            state=self.reward_state)
        results = []
        for i, report in enumerate(self.reports):
            outcome = report["battle"].get("outcome", "ongoing")
            reward = rewards[i]
            info = dict(player=i+1, outcome=outcome, reason="max_steps_or_counter" if truncated else reason,
                        decisions=self.steps, turn_start=start, turn_end=report["battle"]["turn"],
                        battle_turns=self.reward_state.turns,
                        reward_components=components[i])
            # A time limit is not a game terminal. Keep the real final observation
            # and mask so PPO can bootstrap the time-limit transition's value.
            results.append((linked_observation(report, terminal), reward, terminal, truncated, info))
        self.record("completed", decision=self.steps, terminated=terminal, truncated=truncated,
                    rewards=rewards, reward_components=components, reports=self.reports,
                    battle_turns=self.reward_state.turns)
        return results

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.calls.abort(RuntimeError("linked session closed"))
        self._release()
        self.executor.shutdown(wait=True, cancel_futures=True)


class LinkedPlayerEnv(gym.Env):
    """Moves 0..3, switches 4..9, partner-replacement WAIT 10; no item actions."""
    def __init__(self, session, player):
        super().__init__()
        self.session, self.player = session, player
        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = observation_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.session.calls.call(self.player, "reset", seed)

    def step(self, action):
        return self.session.calls.call(self.player, "step", action)

    def close(self):
        self.session.close()
