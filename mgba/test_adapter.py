from __future__ import annotations

import argparse
import json
import socket

from firered_rl.backends.mgba_bridge import MGBABridgeBackend, MGBAProtocolError


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the mGBA Lua adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55355)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    bridge = MGBABridgeBackend(args.host, args.port, args.timeout)
    try:
        response = bridge.status()
        print(json.dumps(response, indent=2, sort_keys=True))
        if not response.get("supported", False):
            raise SystemExit("Adapter is reachable, but the loaded ROM is unsupported")
    except ConnectionRefusedError as exc:
        raise SystemExit(
            f"No adapter is listening on {args.host}:{args.port}. "
            "Start mGBA, load FireRed and mgba/adapter.lua, then retry."
        ) from exc
    except (OSError, MGBAProtocolError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Adapter smoke test failed: {exc}") from exc
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
