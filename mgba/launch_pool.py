#!/usr/bin/env python3
"""Launch linked environment pairs and optionally run a JSON command sequence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
from threading import Event
import time

from launch_link import LinkProcess, add_launch_arguments, validate_launch, positive_seconds
from inspect_state import decorate
from firered_rl.link_connect import connect_pair
from firered_rl.link_commands import load_commands, run_pair_commands
from firered_rl.match_pool import MatchPool, PoolMember


def make_members(count, states, ports=None, base_port=55355):
    if count < 2 or count % 2:
        raise ValueError("--environments must be even and at least 2 (e.g. 4 or 8)")
    if len(states) not in (2, count):
        raise ValueError("provide two alternating checkpoints, or one checkpoint per environment")
    assigned_ports = list(ports) if ports is not None else list(range(base_port, base_port+count))
    if len(assigned_ports) != count:
        raise ValueError("--ports must contain one port per environment")
    members = [PoolMember(f"agent-{i+1}", assigned_ports[i], states[i % len(states)]) for i in range(count)]
    queue = MatchPool()
    for member in members:
        queue.join(member)
    return members, queue


def write_status(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2)+"\n")
    temporary.replace(path)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser, root)
    parser.add_argument("--environments", type=int, default=4,
                        help="Even number of players; 4 creates two linked pairs (default: 4)")
    parser.add_argument("--states", required=True, type=Path, nargs="+",
                        help="Two alternating counter checkpoints, or one per environment")
    port_group = parser.add_mutually_exclusive_group()
    port_group.add_argument("--base-port", type=int, default=55355)
    port_group.add_argument("--ports", nargs="+", type=int)
    parser.add_argument("--commands", type=Path,
                        help="JSON command cycles to run after every pair reaches battle readiness")
    parser.add_argument("--command-timeout", type=positive_seconds, default=120,
                        help="Wall-clock seconds allowed for each paired battle turn")
    args = parser.parse_args()
    try:
        members, queue = make_members(args.environments, args.states, args.ports, args.base_port)
        if args.commands and args.lobby_only:
            raise ValueError("--commands requires battle startup; remove --lobby-only")
        command_plan = load_commands(args.commands, args.environments) if args.commands else None
        validate_launch(args.rom, args.emulator, [m.state for m in members],
                        [m.port for m in members], args.headless)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    run = root / ".local/run" / datetime.now().strftime("pool-%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True)
    status_file = run / "pool.json"
    status = {"status":"starting", "speed":args.speed, "environments":args.environments,
              "goal":"commands" if command_plan else "lobby" if args.lobby_only else "battle",
              "queue": [m.agent_id for m in queue.waiting], "matches":[]}
    sessions = []
    cancel = Event()
    executor = None
    print(f"Pool: {run}", flush=True)
    write_status(status_file, status)
    if command_plan:
        (run / "command-plan.json").write_text(json.dumps(command_plan, indent=2)+"\n")
    try:
        while (pair := queue.take_pair()) is not None:
            number = len(sessions)+1
            session = LinkProcess(root, run / f"match-{number}", [m.port for m in pair], [m.state for m in pair])
            sessions.append(session)
            match = {"match":number, "session_id":session.session_id, "status":"starting",
                     "players":[{"agent_id":m.agent_id, "port":m.port, "state":str(m.state.resolve()),
                                 "link_player":i} for i, m in enumerate(pair, 1)],
                     "directory":str(session.run)}
            status["matches"].append(match)
            status["queue"] = [m.agent_id for m in queue.waiting]
            print(f"match {number}: {pair[0].agent_id} <-> {pair[1].agent_id} "
                  f"ports {pair[0].port}/{pair[1].port}", flush=True)
            session.start(args.rom, args.emulator, args.speed, args.headless)
            match["pid"] = session.process.pid
            write_status(status_file, status)
        for session in sessions:
            session.wait_ready(args.startup_timeout)
        status["status"] = "connecting"
        for match in status["matches"]:
            match["status"] = "connecting"
        write_status(status_file, status)
        executor = ThreadPoolExecutor(max_workers=len(sessions))
        futures = {executor.submit(connect_pair, s.ports, s.session_id, s.run / "connection.jsonl",
                                   args.connect_timeout, cancel, lambda msg: print(msg, flush=True),
                                   process_state=s.process_status, start_battle=not args.lobby_only): i
                   for i, s in enumerate(sessions)}
        for future in as_completed(futures):
            index = futures[future]
            status["matches"][index]["connection"] = future.result()
            status["matches"][index]["status"] = status["matches"][index]["connection"]["status"]
            write_status(status_file, status)
            ready = "connected in the Colosseum" if args.lobby_only else "at the first link-battle command menu"
            print(f"match {index+1}: PASS, both players {ready}", flush=True)
        executor.shutdown(wait=True)
        executor = None
        status["status"] = "connected" if args.lobby_only else "battle_ready"
        write_status(status_file, status)
        print(f"POOL READY: {len(sessions)} linked pairs / {args.environments} environments.\n"
              f"Assignments: {status_file}\nAutomatic input released. Ctrl+C stops all sessions.", flush=True)
        if command_plan:
            status["status"] = "running_commands"
            write_status(status_file, status)
            print(f"Commands: {command_plan['repeats']} cycles x {command_plan['turns_per_repeat']} turns "
                  f"= {command_plan['total_turns']} paired turns per match.", flush=True)
            executor = ThreadPoolExecutor(max_workers=len(sessions))
            futures = {executor.submit(run_pair_commands, s.ports, s.session_id,
                       command_plan["players"][i*2:i*2+2], command_plan, s.run,
                       timeout=args.command_timeout, cancel=cancel,
                       announce=lambda msg: print(msg, flush=True), decorate=decorate): i
                       for i, s in enumerate(sessions)}
            for future in as_completed(futures):
                index = futures[future]
                result = future.result()
                status["matches"][index]["commands"] = result
                status["matches"][index]["status"] = "commands_passed"
                write_status(status_file, status)
                for check in result["checks"]:
                    print(f"port {check['port']}: PASS, expected slot {check['expected_slot']}, "
                          f"inspector slot {check['actual_slot']} ({check['species']}); "
                          f"{result['turns']} turns verified", flush=True)
            executor.shutdown(wait=True)
            executor = None
            status["status"] = "commands_passed"
            write_status(status_file, status)
            print("COMMAND TEST PASS. Final inspector reports are in each match directory.\n"
                  "All inputs released; windows remain open for inspection. Ctrl+C stops the pool.", flush=True)
        while all(s.process.poll() is None for s in sessions):
            time.sleep(0.25)
        raise RuntimeError("a paired emulator process exited; stopping the pool")
    except KeyboardInterrupt:
        status["status"] = "stopped"
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        status["status"], status["error"] = "failed", str(error)
        for match, session in zip(status["matches"], sessions):
            match["failure_snapshot"] = session.capture_failure(error)
        print(f"Pool failed: {error}\nDiagnostics: {run}", flush=True)
        return 1
    finally:
        cancel.set()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for session in sessions:
            session.close()
        for match in status["matches"]:
            match["status_at_shutdown"] = match["status"]
            match["status"] = "stopped"
        write_status(status_file, status)


if __name__ == "__main__":
    raise SystemExit(main())
