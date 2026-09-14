#!/usr/bin/env python3
"""Check repeatable state restoration independently for one or more emulators."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from firered_rl.backends.mgba_bridge import MGBABridgeBackend


def check_one(port, repeats):
    bridge = MGBABridgeBackend("127.0.0.1", port, timeout=15)
    try:
        status = bridge.status()
        if not status.get("supported") or not status.get("reset_configured"):
            raise RuntimeError(f"port {port}: launch adapter 0.3.0 with --states first")
        baseline = None
        for _ in range(repeats):
            result = bridge.reset_response()
            signature = (result["frame"], result["ram_fnv1a"])
            if baseline is not None and signature != baseline:
                raise RuntimeError(f"port {port}: reset differs: {signature} != {baseline}")
            baseline = signature
            bridge.step_response(0, 30)
        return {"port": port, "resets": repeats, "frame": baseline[0],
                "ram_fnv1a": baseline[1], "ok": True}
    finally:
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ports", type=int, nargs="+", default=[55355, 55356])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2 or len(set(args.ports)) != len(args.ports):
        parser.error("use at least two repeats and unique ports")
    with ThreadPoolExecutor(max_workers=len(args.ports)) as pool:
        results = list(pool.map(lambda port: check_one(port, args.repeats), args.ports))
    print(json.dumps(results, indent=2))
    print("PASS: repeated reset frame and RAM fingerprints match within each emulator")


if __name__ == "__main__":
    main()
