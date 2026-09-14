#!/usr/bin/env python3
"""Two cores in ONE mGBA process, using its native local-link controller."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import uuid

from launch_emulators import EXPECTED_SHA1, adapter_status, sha1, stop_processes, write_bootstrap
from firered_rl.backends.mgba_bridge import MGBABridgeBackend
from firered_rl.launch_options import add_speed_argument, speed_options
from firered_rl.link_connect import connect_pair


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive and finite")
    return number


def add_launch_arguments(parser, root, *, require_rom=True):
    parser.add_argument("--rom", required=require_rom, type=Path)
    parser.add_argument("--emulator", type=Path, default=root / ".local/mgba/bin/mgba-qt")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--startup-timeout", type=positive_seconds, default=30)
    parser.add_argument("--connect-timeout", type=positive_seconds, default=180,
                        help="Wall-clock seconds allowed for each connection/battle-start stage")
    parser.add_argument("--lobby-only", action="store_true",
                        help="Stop automatic setup in the Colosseum before walking to battle pads")
    add_speed_argument(parser)


def validate_launch(rom, emulator, states, ports, headless):
    if len(ports) != len(states) or len(set(ports)) != len(ports) or not all(1 <= p <= 65535 for p in ports):
        raise ValueError("one distinct valid port is required per checkpoint")
    for path in (rom, emulator, *states):
        if not path.expanduser().is_file():
            raise ValueError(f"required file does not exist: {path}")
    if any(p.suffix.lower() == ".sav" for p in states):
        raise ValueError("--states requires emulator checkpoints (.ss1), not battery .sav files")
    if headless and not shutil.which("xvfb-run"):
        raise ValueError("--headless requires xvfb-run")
    if sha1(rom.expanduser().resolve()) != EXPECTED_SHA1:
        raise ValueError("unsupported ROM SHA-1")
    for port in ports:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as error:
                raise ValueError(f"port {port} is unavailable: {error}") from error


class LinkProcess:
    """Lifecycle shared by the pair launcher and pool; caller always closes it."""
    def __init__(self, root, run, ports, states):
        if len(ports) != 2 or len(states) != 2:
            raise ValueError("a native session has exactly two players")
        self.root, self.run, self.ports, self.states = root, run, tuple(ports), tuple(states)
        self.session_id = uuid.uuid4().hex
        self.process, self.log = None, None
        self.headless = False

    def start(self, rom, emulator, speed=1.0, headless=False, visible_player=None):
        if visible_player not in (None, 1, 2):
            raise ValueError("visible_player must be 1 or 2")
        self.headless = headless
        self.run.mkdir(parents=True)
        players = []
        for i, (port, state) in enumerate(zip(self.ports, self.states), 1):
            directory = self.run / f"player{i}"
            directory.mkdir()
            local_rom = directory / "FireRed.gba"
            shutil.copyfile(rom.expanduser().resolve(), local_rom)
            bootstrap = directory / "bootstrap.lua"
            write_bootstrap(bootstrap, self.root / "mgba/adapter.lua",
                            self.root / "mgba/profiles/firered_us_v1_0.lua", "firered_us_v1_0", port)
            body = bootstrap.read_text()
            bootstrap.write_text("assert(FIRERED_RL_NATIVE_LINK_SESSION, 'mGBA link patch is missing')\n"
                f"FIRERED_RL_LINK_SESSION_ID = {json.dumps(self.session_id)}\nFIRERED_RL_LINK_PLAYER = {i}\n"
                f"FIRERED_RL_LOG_PATH = {json.dumps(str(directory / 'adapter.log'))}\n" + body)
            players.append(dict(rom=str(local_rom), state=str(state.expanduser().resolve()),
                                script=str(bootstrap), port=port))
        manifest = self.run / "manifest.json"
        manifest.write_text(json.dumps(players, indent=2)+"\n")
        environment = os.environ.copy()
        environment.pop("MGBA_LUA_SCRIPT", None)
        environment.pop("MGBA_VISIBLE_LINK_ROM", None)
        if visible_player is not None:
            environment["MGBA_VISIBLE_LINK_ROM"] = players[visible_player-1]["rom"]
        environment["MGBA_LINK_MANIFEST"] = str(manifest)
        environment["MGBA_SPEED_CONTROL"] = str(self.run / "speed.json")
        command = [str(emulator.expanduser().resolve()), *speed_options(speed),
                   "-C", "pauseOnFocusLost=0", "-C", "pauseOnMinimize=0",
                   "-C", "loadStateExtdata=11", "-C", "savegamePath="]
        if headless:
            command = ["xvfb-run", "-a", *command]
        self.log = (self.run / "mgba.log").open("wb")
        self.process = subprocess.Popen(command, env=environment, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        print(f"session={self.session_id} pid={self.process.pid} ports={self.ports} speed={speed}\n"
              f"Files/log: {self.run}", flush=True)

    def set_speed(self, speed, timeout=10):
        """Change both cores through the opt-in Qt patch; require both acknowledgements."""
        request = dict(id=uuid.uuid4().hex, speed=0 if speed == "unlimited" else float(speed))
        if not math.isfinite(request["speed"]) or request["speed"] < 0 or request["speed"] > 1e30:
            raise ValueError("runtime speed must be between 0 and 1e30")
        temporary = self.run / "speed.tmp"
        temporary.write_text(json.dumps(request))
        temporary.replace(self.run / "speed.json")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("mGBA exited while changing speed")
            ready = []
            for player in (1, 2):
                try:
                    ack = json.loads((self.run / f"player{player}/speed-applied.json").read_text())
                    ready.append(ack == request)
                except (OSError, ValueError):
                    ready.append(False)
            if all(ready):
                print(f"Both players now at speed={speed}", flush=True)
                return
            time.sleep(.05)
        raise RuntimeError("runtime speed patch missing or unresponsive; run python tools/setup_mgba_link.py, "
                           "then cmake --build build/mgba --parallel and cmake --install build/mgba")

    def wait_ready(self, timeout):
        pending = set(self.ports)
        deadline = time.monotonic()+timeout
        while pending:
            if self.process.poll() is not None:
                raise RuntimeError(f"mGBA exited; see {self.run / 'mgba.log'}")
            for port in tuple(pending):
                try:
                    status = adapter_status(port)
                except (OSError, RuntimeError, ValueError):
                    continue
                if not status.get("native_link_session") or status.get("link_session_id") != self.session_id:
                    raise RuntimeError(f"port {port} is not part of the assigned native link session")
                if not status.get("supported"):
                    raise RuntimeError(f"unsupported ROM on port {port}")
                if status.get("link_player") != self.ports.index(port)+1:
                    raise RuntimeError(f"port {port} has the wrong player assignment")
                pending.remove(port)
                print(f"player={status['link_player']} port={port} adapter={status['adapter_version']} ready", flush=True)
            if pending and time.monotonic() >= deadline:
                raise RuntimeError(f"adapter startup timed out; apply setup_mgba_link.py and rebuild/install mGBA; see {self.run / 'mgba.log'}")
            if pending:
                time.sleep(0.1)
        for port in self.ports:
            bridge = MGBABridgeBackend("127.0.0.1", port)
            try:
                initial = bridge.link_status()
            finally:
                bridge.close()
            if initial["battle_active"] or initial["connection_established"] or initial["received_remote_players"] or initial["wireless"]:
                raise RuntimeError(f"port {port}: checkpoint must be in the overworld before starting cable/wireless communication")

    def process_status(self):
        code = self.process.poll() if self.process is not None else None
        state = "not_started" if self.process is None else ("running" if code is None else "exited")
        result = dict(session_id=self.session_id, ports=list(self.ports), state=state,
                      pid=self.process.pid if self.process is not None else None,
                      process_role="xvfb_wrapper" if self.headless else "mgba",
                      returncode=code, log=str(self.run / "mgba.log"))
        if code is not None and code < 0:
            try:
                result["signal"] = signal.Signals(-code).name
            except ValueError:
                result["signal"] = f"signal {-code}"
        return result

    def capture_failure(self, error):
        # Capture before stop_processes sends SIGTERM, which would hide the cause.
        report = dict(error=str(error), process=self.process_status())
        if self.run.is_dir():
            (self.run / "failure.json").write_text(json.dumps(report, indent=2)+"\n")
        print("Session failure before cleanup: "+json.dumps(report), flush=True)
        return report

    def close(self):
        if self.process is not None:
            stop_processes([self.process])
        if self.log is not None:
            self.log.close()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser, root)
    parser.add_argument("--states", required=True, nargs=2, type=Path,
                        help="Two pre-connection overworld/Cable Club checkpoints")
    parser.add_argument("--ports", nargs=2, type=int, default=[55355, 55356])
    parser.add_argument("--auto-connect", action="store_true",
                        help="Navigate the NPC automatically from generated counter checkpoints")
    args = parser.parse_args()
    try:
        validate_launch(args.rom, args.emulator, args.states, args.ports, args.headless)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    run = root / ".local/run" / datetime.now().strftime("link-%Y%m%d-%H%M%S-%f")
    session = LinkProcess(root, run, args.ports, args.states)
    try:
        session.start(args.rom, args.emulator, args.speed, args.headless)
        session.wait_ready(args.startup_timeout)
        if args.auto_connect:
            result = connect_pair(session.ports, session.session_id, run / "connection.jsonl",
                                  timeout=args.connect_timeout, announce=lambda s: print(s, flush=True),
                                  process_state=session.process_status, start_battle=not args.lobby_only)
            (run / "connected.json").write_text(json.dumps(result, indent=2)+"\n")
            ready = "connected in the Colosseum" if args.lobby_only else "at the first link-battle command menu"
            print(f"PASS: both players {ready}; automatic input released.", flush=True)
        else:
            print("Both adapters ready. Enter Cable Club -> Colosseum -> Single Battle in both games.\n"
                  "Run test_link.py to verify the connection, or relaunch with --auto-connect.", flush=True)
        while session.process.poll() is None:
            time.sleep(0.25)
        return 1
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        session.capture_failure(error)
        print(f"Link launcher: {error}", flush=True)
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
