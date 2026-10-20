"""Optional persistent Python + Node worker for each simulated battle pair."""
import multiprocessing as mp
import os
import signal
import time
import traceback

from .linked_env import LinkedPlayerEnv, PairedCalls


def worker(connection, cancelled, root, run, ports, args, reward_config):
    # Keep the Node child in this worker's process group for bounded cleanup.
    os.setsid()
    from .showdown_env import ShowdownFactory, ShowdownBattleSession
    selected = []
    factory = ShowdownFactory(root, run, ports, lambda: tuple(selected), args, cancelled)
    session = ShowdownBattleSession(factory, max_steps=args.max_steps, reward_config=reward_config,
        readable_log=run / "battle-readable.log" if args.readable_log else None,
        battle_log=run / "battle.log" if args.battle_log else None)
    try:
        while not cancelled.is_set():
            if not connection.poll(.2):
                continue
            request = connection.recv()
            if request[0] == "close":
                break
            try:
                if request[0] == "reset":
                    selected[:] = request[1]
                    result = session.reset_pair(request[2])
                elif request[0] == "step":
                    result = session.step_pair(request[1])
                else:
                    raise ValueError("unknown worker request")
                connection.send(dict(ok=True, result=result, episode=factory.episode,
                                     pid=os.getpid(), node_pid=factory.process.pid))
            except Exception:
                connection.send(dict(ok=False, error=traceback.format_exc()))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        session.close()
        connection.close()


class ProcessShowdownSession:
    def __init__(self, root, run, ports, choose, args, cancel, reward_config):
        self.root, self.run, self.ports = root, run, ports
        self.choose, self.args, self.cancel = choose, args, cancel
        from .observation_v2 import versioned_raw_space, validate_version
        self.observation_version = validate_version(
            getattr(args, "session_observation_version", None)
            or getattr(args, "observation_version", None) or "v1")
        self.observation_space = versioned_raw_space(self.observation_version)
        self.states, self.episode = None, 0
        self.calls = PairedCalls(self.reset_pair, self.step_pair, args.command_timeout+40, cancel)
        self.context = mp.get_context("spawn")
        self.cancelled = self.context.Event()
        self.connection, self.child_connection = self.context.Pipe()
        self.process = self.context.Process(target=worker, args=(self.child_connection, self.cancelled,
            root, run, ports, args, reward_config), name=f"showdown-{ports[0]}")
        self.started, self.closed = False, False

    def players(self):
        return [LinkedPlayerEnv(self, i) for i in range(2)]

    def request(self, message):
        if self.closed or self.cancel.is_set() or self.cancelled.is_set():
            raise RuntimeError("Showdown worker cancelled")
        if not self.started:
            self.process.start()
            self.started = True
            self.child_connection.close()
        self.connection.send(message)
        deadline = time.monotonic() + self.args.command_timeout + 30
        while not self.connection.poll(.2):
            if self.cancel.is_set() or self.cancelled.is_set():
                raise RuntimeError("Showdown worker cancelled")
            if not self.process.is_alive():
                raise RuntimeError(f"Python Showdown worker exited with code {self.process.exitcode}; see {self.run}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Python Showdown worker timed out; see {self.run}")
        response = self.connection.recv()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        self.episode = response["episode"]
        if message[0] == "reset":
            import json
            (self.run / "workers.json").write_text(json.dumps(dict(python_pid=response["pid"],
                node_pid=response["node_pid"], mode="process"), indent=2) + "\n")
        return response["result"]

    def reset_pair(self, seeds=None):
        self.states = self.choose()
        return self.request(("reset", self.states, seeds))

    def step_pair(self, actions):
        return self.request(("step", list(map(int, actions))))

    def stop(self):
        self.cancelled.set()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.calls.abort(RuntimeError("Showdown process session closed"))
        self.stop()
        if self.started:
            self.process.join(timeout=5)
            if self.process.is_alive():
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    self.process.kill()
                self.process.join()
        self.connection.close()
        self.child_connection.close()
