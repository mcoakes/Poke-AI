"""Local Gen III simulator using the existing paired Gymnasium/PPO contract."""
import json
import os
from pathlib import Path
import random
import selectors
import shutil
import subprocess
from threading import Event, Lock, Thread
import time

from .linked_env import LinkedPlayerEnv, PairedCalls
from .linked_reward import EpisodeRewardState, LinkedRewardConfig, reward_points
from .readable_log import format_event, names
from .runtime_logs import EpisodeLogs, RotatingLog, log_settings
from .turn_env import encode


def validate_showdown(project, node="node"):
    if not shutil.which(node):
        raise ValueError("Node.js 18+ is required for Showdown; see docs/showdown-training.md")
    package = Path(project) / "sim/showdown/node_modules/pokemon-showdown/package.json"
    if not package.is_file() or json.loads(package.read_text()).get("version") != "0.11.11":
        raise ValueError("Install the pinned simulator: npm --prefix sim/showdown ci --omit=optional")


def showdown_observation(report, terminal=False, version="v1"):
    from .observation_v2 import encode_v2, validate_version
    version = validate_version(version)
    obs = encode(report, terminal=terminal)
    # Legality comes from Gen III Showdown requests, never guessed from UI state.
    obs["action_mask"][:] = report["action_mask"]
    return encode_v2(report, obs) if version == "v2" else obs


class ShowdownFactory:
    """One persistent local Node worker per pair; reset battles in memory."""
    def __init__(self, root, run, ports, choose, args, cancel, *, runner=None):
        self.root, self.run, self.choose = Path(root), Path(run), choose
        self.args, self.cancel = args, cancel
        self.runner = Path(runner) if runner is not None else self.root / "sim/showdown/runner.cjs"
        self.episode, self.states = 0, None
        self.process = None
        self.stderr = None
        self.stderr_reader = None
        self.stderr_error = None
        self.lock = Lock()
        self.buffer = b""
        self.rng = random.Random(args.seed + ports[0] * 1000003)
        self.log_options, keep_episodes = log_settings(args)
        self.episode_logs = EpisodeLogs(self.run, enabled=getattr(args, "step_logs", False),
                                        keep=keep_episodes)
        self.logs = {}

    def log(self, path, text):
        path = Path(path)
        if path not in self.logs:
            self.logs[path] = RotatingLog(path, **self.log_options)
        self.logs[path].append(text)

    def _read_stderr(self):
        try:
            while True:
                chunk = os.read(self.stderr.fileno(), 65536)
                if not chunk:
                    return
                self.log(self.run / "showdown-stderr.log", chunk.decode("utf-8", errors="replace"))
        except (OSError, ValueError) as error:
            self.stderr_error = error

    def complete_episode(self):
        for removed in self.episode_logs.complete(self.episode_run):
            for path in list(self.logs):
                if path.parent == removed:
                    del self.logs[path]

    def request(self, command, **fields):
        if self.cancel.is_set():
            raise RuntimeError("training cancelled")
        with self.lock:
            if self.process is None:
                self.run.mkdir(parents=True, exist_ok=True)
                # Drain stderr continuously so neither its pipe nor the on-disk
                # diagnostic stream can grow without the configured limit.
                self.log(self.run / "showdown-stderr.log", "Showdown worker starting\n")
                self.process = subprocess.Popen([getattr(self.args, "node", "node"),
                    str(self.runner)], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.stderr = self.process.stderr
                self.stderr_reader = Thread(target=self._read_stderr, name="showdown-stderr", daemon=True)
                self.stderr_reader.start()
                (self.run / "workers.json").write_text(json.dumps(dict(python_pid=os.getpid(),
                    node_pid=self.process.pid, mode="thread"), indent=2) + "\n")
        process = self.process
        try:
            if self.stderr_error is not None:
                raise RuntimeError(f"Showdown diagnostic log failed: {self.stderr_error}")
            process.stdin.write((json.dumps(dict(command=command, **fields))+"\n").encode())
            process.stdin.flush()
            deadline = time.monotonic() + self.args.command_timeout
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while b"\n" not in self.buffer:
                    if self.stderr_error is not None:
                        raise RuntimeError(f"Showdown diagnostic log failed: {self.stderr_error}")
                    if self.cancel.is_set():
                        raise RuntimeError("training cancelled")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Showdown {command} timed out")
                    if selector.select(min(.2, max(0, deadline-time.monotonic()))):
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            raise RuntimeError(f"Showdown worker exited; see {self.run / 'showdown-stderr.log'}")
                        self.buffer += chunk
            line, self.buffer = self.buffer.split(b"\n", 1)
            response = json.loads(line)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Showdown request failed"))
            return response
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"Showdown worker disconnected: {error}; see {self.run / 'showdown-stderr.log'}") from error

    def reset(self, seeds=None):
        self.episode += 1
        self.states = self.choose()
        self.episode_run = self.episode_logs.start(self.episode)
        rng = random.Random(str(seeds)) if seeds and any(s is not None for s in seeds) else self.rng
        seed = [rng.randrange(65536) for _ in range(4)]
        teams = [json.loads(p.read_text()) for p in self.states]
        if self.episode_logs.enabled:
            (self.episode_run / "teams.json").write_text(json.dumps(dict(teams=teams, seed=seed), indent=2)+"\n")
        return self.request("reset", teams=teams, seed=seed)

    def stop(self):
        with self.lock:
            process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def close(self):
        self.stop()
        if self.process is not None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        if self.stderr_reader is not None:
            self.stderr_reader.join(timeout=2)
        if self.stderr is not None:
            self.stderr.close()


def spectator_lines(protocol, hp=None):
    """Render actual simulator messages, retaining exact damage/HP protocol separately."""
    lines, skip_public = [], False
    hp = {} if hp is None else hp
    for line in protocol:
        fields = line.split("|")
        event = fields[1] if len(fields) > 1 else ""
        if event == "split":
            skip_public = True
            continue
        # A split contains a private exact-HP line followed by a public percentage line.
        if skip_public == "next":
            skip_public = False
            continue
        if skip_public is True:
            skip_public = "next"
        values = fields[2:]
        label = lambda value: value.replace("p1a:", "P1").replace("p2a:", "P2")
        if event == "move":
            lines.append(f"  {label(values[0])} used {values[1]}!")
        elif event in ("-damage", "-heal"):
            current = int(values[1].split("/")[0].split()[0])
            old = hp.get(values[0])
            delta = abs(current-old) if old is not None else None
            verb = (f"took {delta} damage" if delta is not None else "took damage") if event == "-damage" else (
                f"recovered {delta} HP" if delta is not None else "recovered HP")
            hp[values[0]] = current
            lines.append(f"  {label(values[0])} {verb}; HP {values[1]}" +
                         (f" ({'; '.join(values[2:])})" if len(values) > 2 else ""))
        elif event in ("switch", "drag"):
            hp[values[0]] = int(values[2].split("/")[0].split()[0])
            lines.append(f"  {label(values[0])} entered; HP {values[2]}")
        elif event in ("-crit", "-supereffective", "-resisted", "-immune"):
            lines.append("  " + {"-crit":"A critical hit!", "-supereffective":"It's super effective!",
                "-resisted":"It's not very effective.", "-immune":"It has no effect."}[event])
        elif event == "faint":
            lines.append(f"  {label(values[0])} fainted!")
        elif event == "turn":
            lines.append(f"\nTurn {values[0]}")
        elif event in ("win", "tie"):
            lines.append(f"  {values[0]} wins!" if event == "win" else "  Draw.")
        elif event in ("-status", "-curestatus", "-start", "-end", "cant", "-fail", "-miss", "-weather"):
            lines.append(f"  {event.lstrip('-')}: " + " | ".join(label(v) for v in values))
    return lines


class ShowdownBattleSession:
    def __init__(self, factory, *, max_steps=200, reward_config=None, readable_log=None, battle_log=None):
        self.factory, self.max_steps = factory, max_steps
        from .observation_v2 import versioned_raw_space, validate_version
        self.observation_version = validate_version(
            getattr(factory.args, "session_observation_version", None)
            or getattr(factory.args, "observation_version", None) or "v1")
        self.observation_space = versioned_raw_space(self.observation_version)
        self.reward_config = reward_config or LinkedRewardConfig()
        self.readable_log, self.battle_log = readable_log, battle_log
        self.cancel = factory.cancel
        self.calls = PairedCalls(self.reset_pair, self.step_pair, factory.args.command_timeout+10, self.cancel)
        self.reports, self.steps, self.done, self.closed = None, 0, True, False
        self.spectator_hp = {}

    def players(self):
        return [LinkedPlayerEnv(self, i) for i in range(2)]

    def record(self, event, response=None, **fields):
        reports = self.reports
        if self.factory.episode_logs.enabled:
            self.factory.log(self.factory.episode_run / "gym-steps.jsonl",
                             json.dumps(dict(event=event, reports=reports, **fields))+"\n")
        if self.readable_log is not None:
            observations = [showdown_observation(r, fields.get("terminated", False),
                                                self.observation_version) for r in reports]
            self.factory.log(self.readable_log,
                             format_event(event, self.factory.episode_run.name, reports, observations, fields))
        if response is not None:
            protocol = response.get("protocol", [])
            if self.factory.episode_logs.enabled:
                self.factory.log(self.factory.episode_run / "showdown-protocol.log", "\n".join(protocol)+"\n")
            if self.battle_log is not None:
                text = f"\n=== {self.factory.episode_run.name} | Pokémon Showdown Gen III ===\n" if event == "reset" else ""
                text += "\n".join(spectator_lines(protocol, self.spectator_hp))+"\n"
                hp = [next(m for m in r["battle"]["battlers"] if m["side"] == "player") for r in reports]
                species = names("firered_us_v1_0").get("species", {})
                text += "  HP | " + " | ".join(f"P{i+1} {species.get(str(m['species_id']), '?')}: {m['current_hp']}/{m['max_hp']}"
                          for i, m in enumerate(hp)) + "\n"
                self.factory.log(self.battle_log, text)

    def reset_pair(self, seeds=None):
        if self.closed:
            raise RuntimeError("Showdown session is closed")
        response = self.factory.reset(seeds)
        self.reports, self.steps, self.done = response["reports"], 0, False
        self.spectator_hp = {}
        self.reward_state = EpisodeRewardState(self.reports)
        self.record("reset", response)
        return [(showdown_observation(r, version=self.observation_version),
                 dict(player=i+1, backend="showdown", observation_version=self.observation_version))
                for i, r in enumerate(self.reports)]

    def step_pair(self, actions):
        if self.done or self.closed:
            raise RuntimeError("reset both players before stepping")
        if len(actions) != 2 or any(int(a) != a or not 0 <= int(a) < 11 or
            not r["action_mask"][int(a)] for a, r in zip(actions, self.reports)):
            raise ValueError(f"illegal paired actions: {actions}")
        actions = [int(a) for a in actions]
        previous = self.reports
        self.record("submitted", decision=self.steps+1, actions=actions)
        audit_execution = getattr(self.factory.args, "move_order_audit", False)
        response = self.factory.request("step", actions=actions,
                                        **({"audit_execution": True} if audit_execution else {}))
        self.reports = response["reports"]
        self.steps += 1
        terminal = response["ended"]
        truncated = not terminal and (self.steps >= self.max_steps or self.reports[0]["battle"]["turn"] >= 254)
        rewards, components = reward_points(previous, self.reports, self.reward_config, terminal, self.reward_state)
        self.done = terminal or truncated
        self.record("completed", response, decision=self.steps, rewards=rewards, reward_components=components,
                    terminated=terminal, truncated=truncated, battle_turns=self.reward_state.turns)
        if self.done:
            self.factory.complete_episode()
        return [(showdown_observation(r, terminal, self.observation_version), rewards[i], terminal, truncated,
                 dict(player=i+1, outcome=r["battle"]["outcome"], backend="showdown", decisions=self.steps,
                      reason="max_steps_or_counter" if truncated else r["battle"]["outcome"],
                      turn_start=previous[i]["battle"]["turn"], turn_end=r["battle"]["turn"],
                      battle_turns=self.reward_state.turns, reward_components=components[i],
                      **({'audit_protocol': response.get('protocol', []),
                          'audit_choice': response.get('audit_execution', [None, None])[i]}
                         if audit_execution else {})))
                for i, r in enumerate(self.reports)]

    def close(self):
        if not self.closed:
            self.closed = True
            self.calls.abort(RuntimeError("Showdown session closed"))
            self.factory.close()
