#!/usr/bin/env python3
"""Launch visible or headless mGBA instances with adapter scripts preloaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from firered_rl.launch_options import add_speed_argument, speed_options, speed_value

EXPECTED_SHA1 = "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc"
DEFAULT_PORTS = (55355, 55356)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Launch one patched mGBA process and Lua adapter per port."
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument(
        "--emulator",
        type=Path,
        default=project / ".local/mgba/bin/mgba-qt",
    )
    parser.add_argument("--ports", type=int, nargs="+", default=DEFAULT_PORTS)
    parser.add_argument("--states", type=Path, nargs="+",
                        help="One emulator save state per port, in matching order")
    parser.add_argument("--profile", default="firered_us_v1_0")
    add_speed_argument(parser)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use xvfb-run instead of displaying emulator windows.",
    )
    parser.add_argument("--startup-timeout", type=float, default=20.0)
    parser.add_argument(
        "--allow-unsupported-rom",
        action="store_true",
        help="Launch despite a SHA-1 mismatch (adapter commands will reject it).",
    )
    return parser.parse_args()


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lua_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_bootstrap(path: Path, adapter: Path, profile_path: Path, profile: str, port: int,
                    state: Path | None = None) -> None:
    content = (
        f"FIRERED_RL_PORT = {port}\n"
        f"FIRERED_RL_PROFILE = {lua_string(profile)}\n"
        f"FIRERED_RL_PROFILE_PATH = {lua_string(str(profile_path))}\n"
        f"FIRERED_RL_STATE_PATH = {lua_string(str(state)) if state else 'nil'}\n"
        f"dofile({lua_string(str(adapter))})\n"
    )
    path.write_text(content, encoding="utf-8")


def adapter_status(port: int, timeout: float = 0.5) -> dict[str, object]:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(b'{"command":"status"}\n')
        response = sock.makefile("rb").readline()
    if not response:
        raise RuntimeError("adapter closed without a response")
    return json.loads(response)


def stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parents[1]
    rom = args.rom.expanduser().resolve()
    emulator = args.emulator.expanduser().resolve()
    adapter = (project / "mgba/adapter.lua").resolve()
    profile_path = (project / "mgba/profiles" / f"{args.profile}.lua").resolve()

    for required in (rom, emulator, adapter, profile_path):
        if not required.is_file():
            raise SystemExit(f"required file does not exist: {required}")
    if len(set(args.ports)) != len(args.ports):
        raise SystemExit("ports must be unique")
    states = [p.expanduser().resolve() for p in args.states] if args.states else []
    if states and len(states) != len(args.ports):
        raise SystemExit("--states must have exactly one state per port")
    for state in states:
        if not state.is_file() or state.suffix.lower() == ".sav":
            raise SystemExit(f"expected an emulator save state, not a battery .sav: {state}")
    for port in args.ports:
        with socket.socket() as probe:
            # Match a reusable server bind: closed connections in TIME_WAIT
            # must not make the preflight reject a port with no listener.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except (OSError, OverflowError) as error:
                raise SystemExit(f"port {port} is unavailable: {error}") from error

    actual_sha1 = sha1(rom)
    if actual_sha1 != EXPECTED_SHA1 and not args.allow_unsupported_rom:
        raise SystemExit(
            f"unsupported ROM SHA-1: {actual_sha1}\nexpected: {EXPECTED_SHA1}"
        )

    run_dir = project / ".local/run"
    run_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    logs = []
    try:
        for player, port in enumerate(args.ports, start=1):
            bootstrap = run_dir / f"player{player}-{port}.lua"
            log_path = run_dir / f"player{player}-{port}.log"
            state = states[player - 1] if states else None
            write_bootstrap(bootstrap, adapter, profile_path, args.profile, port, state)
            instance_rom = rom
            if state:
                # Independent battery paths; never let the two teams share FireRed.sav.
                instance_dir = run_dir / f"player{player}-{port}"
                instance_dir.mkdir(exist_ok=True)
                instance_rom = instance_dir / "FireRed.gba"
                shutil.copyfile(rom, instance_rom)
            command = [str(emulator), *speed_options(args.speed)]
            if state:
                command.extend(["--savestate", str(state), "-C", "savegamePath=" + str(instance_rom.parent)])
            command.append(str(instance_rom))
            if args.headless:
                command = ["xvfb-run", "-a", *command]
            environment = os.environ.copy()
            environment["MGBA_LUA_SCRIPT"] = str(bootstrap)
            log = log_path.open("wb")
            logs.append(log)
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            mode = "headless" if args.headless else "visible"
            print(f"player {player}: pid={process.pid} port={port} mode={mode}")

        pending = set(args.ports)
        deadline = time.monotonic() + args.startup_timeout
        while pending and time.monotonic() < deadline:
            for port in tuple(pending):
                try:
                    status = adapter_status(port)
                except (OSError, ValueError, RuntimeError):
                    continue
                if not status.get("ok") or not status.get("supported"):
                    raise RuntimeError(f"adapter on port {port} rejected the ROM: {status}")
                if states and not status.get("reset_configured"):
                    raise RuntimeError(f"port {port} has no reset state; restart with adapter 0.3.0")
                pending.remove(port)
                print(
                    f"adapter ready: port={port} version={status.get('adapter_version')} "
                    f"profile={status.get('profile')}"
                )
            if pending:
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("an emulator exited during startup; check .local/run/*.log")
                time.sleep(0.1)
        if pending:
            raise RuntimeError(
                f"timed out waiting for adapter ports {sorted(pending)}; "
                "confirm the mGBA autoload patch is installed"
            )

        print("all adapters ready; press Ctrl+C to stop the emulators")
        while all(process.poll() is None for process in processes):
            time.sleep(0.25)
        return 1
    except KeyboardInterrupt:
        print("\nstopping emulators")
        return 0
    except RuntimeError as error:
        print(f"launcher error: {error}", file=sys.stderr)
        return 1
    finally:
        stop_processes(processes)
        for log in logs:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
