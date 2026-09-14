#!/usr/bin/env python3
"""Observe both FireRed games while the user connects through Cable Club."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time

from firered_rl.backends.mgba_bridge import MGBABridgeBackend
from firered_rl.link_test import connection_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", nargs=2, type=int, default=[55355, 55356])
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--require-battle", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(".local/run") / datetime.now().strftime("link-test-%Y%m%d-%H%M%S-%f.jsonl"))
    args = parser.parse_args()
    if args.ports[0] == args.ports[1] or args.timeout <= 0:
        parser.error("use two distinct ports and a positive timeout")
    bridges = [MGBABridgeBackend("127.0.0.1", p) for p in args.ports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool, args.output.open("x") as log:
            statuses = list(pool.map(lambda b: b.status(), bridges))
            if not all(s.get("native_link_session") for s in statuses):
                raise RuntimeError("use launch_link.py; separate emulator processes do not share a cable")
            sessions = {s.get("link_session_id") for s in statuses}
            if len(sessions) != 1 or not next(iter(sessions)) or {s.get("link_player") for s in statuses} != {1, 2}:
                raise RuntimeError("ports do not identify opposite players of one session")
            log.write(json.dumps(dict(event="startup", statuses=statuses))+"\n")
            print("Watching cable connection; interact with BOTH emulator windows. Log:", args.output, flush=True)
            deadline, stable, previous = time.monotonic()+args.timeout, 0, None
            while time.monotonic() < deadline:
                reports = list(pool.map(lambda b: b.link_status(), bridges))
                errors = connection_errors(reports, args.require_battle)
                log.write(json.dumps(dict(reports=reports, pending=errors))+"\n")
                log.flush()
                summary = [(r["local_id"], r["player_count"], r["received_remote_players"],
                            r["link_battle"], r["error_bits"]) for r in reports]
                if summary != previous:
                    for port, r in zip(args.ports, reports):
                        print(f"port={port} id={r['local_id']} players={r['player_count']} "
                              f"exchanged={r['received_remote_players']} link_battle={r['link_battle']} "
                              f"errors=0x{r['error_bits']:x}", flush=True)
                    previous = summary
                stable = stable+1 if not errors else 0
                if stable >= 3:
                    print("PASS: both games report a two-player cable connection and matching exchanged player records."
                          + (" Both are in a link battle." if args.require_battle else " Battle entry is a separate check."))
                    return 0
                time.sleep(0.25)
            print("NOT VERIFIED:", "; ".join(errors))
            return 1
    except (OSError, RuntimeError, ValueError) as error:
        print("Link test:", error)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        for bridge in bridges:
            bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
