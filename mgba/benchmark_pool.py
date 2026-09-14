#!/usr/bin/env python3
"""Benchmark repeated native link battles using concurrent Gymnasium player environments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
from queue import Empty, Queue
from threading import Event
import time

from launch_link import LinkProcess, add_launch_arguments, validate_launch, positive_seconds
from launch_pool import make_members
from firered_rl.benchmark import command_cycles, CommandPolicy, Throughput
from firered_rl.link_connect import connect_pair
from firered_rl.linked_env import LinkedBattleSession


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_arguments(parser, root)
    parser.add_argument("--environments", type=int, default=8, help="Even number of Gymnasium players (8 = 4 battles)")
    parser.add_argument("--states", type=Path, nargs="+", required=True, help="Two alternating counter states, or one per player")
    parser.add_argument("--base-port", type=int, default=55355)
    parser.add_argument("--commands", nargs="+", default=["MOVE1"], help="MOVE1..4, SWITCH1..6, or one benchmark JSON path")
    parser.add_argument("--repeat-commands", action=argparse.BooleanOptionalAction, default=True,
                        help="Cycle commands until battle ends (default: on)")
    parser.add_argument("--episodes", type=positive_int, default=100, help="Total battle attempts across all pairs (default: 100)")
    parser.add_argument("--max-steps", type=positive_int, default=200, help="Decision limit per battle attempt")
    parser.add_argument("--command-timeout", type=positive_seconds, default=120)
    parser.add_argument("--seed", type=int, default=1, help="Gymnasium seed; emulator RNG comes from the saved checkpoints")
    parser.add_argument("--verbose", action="store_true", help="Print connection phases as well as episode rates")
    args = parser.parse_args()
    try:
        if args.lobby_only:
            raise ValueError("benchmark requires battles; remove --lobby-only")
        members, _ = make_members(args.environments, args.states, base_port=args.base_port)
        if args.episodes < args.environments//2:
            raise ValueError("--episodes must be at least --environments / 2 to exercise every pair")
        cycles = command_cycles(args.commands, args.environments)
        validate_launch(args.rom, args.emulator, [m.state for m in members], [m.port for m in members], args.headless)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    run = root / ".local/run" / datetime.now().strftime("benchmark-%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True)
    config = dict(environments=args.environments, pairs=args.environments//2, speed=args.speed,
                  headless=args.headless, episodes=args.episodes, cycles=cycles,
                  repeat_commands=args.repeat_commands, max_steps=args.max_steps,
                  command_timeout=args.command_timeout, seed=args.seed,
                  reset_mode="restart_native_pair_from_counter", timing="wall time including launch, connect, reset and cleanup",
                  python=platform.python_version(), platform=platform.platform(), cpus=os.cpu_count(),
                  rom=str(args.rom.resolve()), emulator=str(args.emulator.resolve()),
                  checkpoints=[dict(path=str(m.state.resolve()), port=m.port,
                    sha256=hashlib.sha256(m.state.read_bytes()).hexdigest()) for m in members])
    (run / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    print(f"Benchmark: {run}\n{args.environments} Gymnasium players / {args.environments//2} simultaneous battles; "
          f"speed={args.speed}, episodes={args.episodes}\nRates count actual completed battles and include setup/reset time.", flush=True)
    cancel = Event()
    ledger = Throughput(run, args.episodes, announce=lambda msg: print(msg, flush=True))
    jobs = Queue()
    # Reserve one first episode per pair before faster pairs claim further work.
    for number in range(args.environments//2+1, args.episodes+1):
        jobs.put(number)

    def worker(index):
        pair = members[index*2:index*2+2]
        episode = index+1
        def launch():
            session = LinkProcess(root, run / f"pair-{index+1}" / f"episode-{episode}",
                                  [m.port for m in pair], [m.state for m in pair])
            try:
                session.start(args.rom, args.emulator, args.speed, args.headless)
                session.wait_ready(args.startup_timeout)
                connect_pair(session.ports, session.session_id, session.run / "connection.jsonl",
                             timeout=args.connect_timeout, cancel=cancel,
                             announce=(lambda msg: print(msg, flush=True)) if args.verbose else (lambda _: None),
                             process_state=session.process_status, start_battle=True)
                return session
            except BaseException as error:
                session.capture_failure(error)
                release(session)
                raise
        def release(session):
            session.close()
            # Remove only these generated ROM copies; preserve all logs and saves.
            for player in (1, 2):
                (session.run / f"player{player}" / "FireRed.gba").unlink(missing_ok=True)
        session = LinkedBattleSession(launch, release_episode=release, timeout=args.command_timeout,
                    reset_timeout=args.startup_timeout+2*args.connect_timeout+30,
                    max_steps=args.max_steps, cancel=cancel)
        envs = session.players()
        try:
            with ThreadPoolExecutor(max_workers=2) as players:
                while not cancel.is_set():
                    started = time.monotonic()
                    futures = [players.submit(env.reset, seed=args.seed+i) for i, env in enumerate(envs)]
                    observations = [f.result()[0] for f in futures]
                    ready = time.monotonic()
                    policies = [CommandPolicy(cycles[index*2+i], args.repeat_commands) for i in range(2)]
                    decisions, terminated, truncated, winner, reason = 0, False, False, None, ""
                    while not cancel.is_set():
                        actions = [policy.choose(obs) for policy, obs in zip(policies, observations)]
                        if any(a is None for a in actions):
                            truncated, reason = True, "commands_exhausted"
                            break
                        futures = [players.submit(env.step, action) for env, action in zip(envs, actions)]
                        results = [f.result() for f in futures]
                        decisions += 1
                        observations = [r[0] for r in results]
                        if results[0][2:4] != results[1][2:4]:
                            raise ValueError("linked Gymnasium players disagree on episode completion")
                        terminated, truncated = results[0][2:4]
                        reason = results[0][4]["reason"]
                        if terminated:
                            outcomes = [r[4]["outcome"] for r in results]
                            winner = {("won","lost"):"player1", ("lost","won"):"player2",
                                      ("draw","draw"):"draw"}.get(tuple(outcomes))
                            if winner is None:
                                raise ValueError(f"contradictory outcomes: {outcomes}")
                        if terminated or truncated:
                            break
                    if cancel.is_set():
                        break
                    ledger.record(episode, index+1, terminated, truncated, winner, ready-started,
                                  time.monotonic()-ready, decisions, reason)
                    try:
                        episode = jobs.get_nowait()
                    except Empty:
                        break
        except BaseException as error:
            if not cancel.is_set():
                ledger.failure(episode, index+1, error)
                cancel.set()
            raise
        finally:
            session.close()

    executor = ThreadPoolExecutor(max_workers=args.environments//2)
    status = "completed"
    try:
        futures = [executor.submit(worker, i) for i in range(args.environments//2)]
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        status = "stopped"
        cancel.set()
    except Exception as error:
        status = "failed"
        cancel.set()
        print(f"Benchmark failed: {error}\nDiagnostics: {run}", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        summary = ledger.finish(status)
    print(f"Final: {summary['completed_battles']} completed battles, {summary['truncated']} truncated, "
          f"{summary['failed']} failed | {summary['episodes_per_hour']:.1f} episodes/hour\n"
          f"Winners: {summary['winners']}\nSummary: {run / 'summary.json'}", flush=True)
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
