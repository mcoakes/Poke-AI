from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from firered_rl.actions import GBAAction
from firered_rl.backends.mgba_bridge import MGBABridgeBackend


def validate_target_counts(
    parser: argparse.ArgumentParser,
    ports: list[int] | tuple[int, ...],
    actions: list[GBAAction] | tuple[GBAAction, ...],
) -> None:
    if len(ports) != len(actions):
        parser.error(
            "--ports and --actions must contain the same number of values "
            f"(received {len(ports)} ports and {len(actions)} actions)"
        )


def parse_action(value: str) -> GBAAction:
    try:
        return GBAAction(int(value))
    except (ValueError, KeyError):
        try:
            return GBAAction[value.upper()]
        except KeyError as exc:
            choices = ", ".join(action.name for action in GBAAction)
            raise argparse.ArgumentTypeError(f"action must be 0-10 or one of: {choices}") from exc


def step_one(host: str, port: int, action: GBAAction, frames: int) -> dict:
    bridge = MGBABridgeBackend(host, port)
    try:
        status = bridge.status()
        if not status.get("supported"):
            raise RuntimeError(f"port {port}: unsupported ROM: {status}")
        try:
            response = bridge.step_response(action, frames)
        except TimeoutError as exc:
            raise RuntimeError(
                f"port {port}: adapter timed out; confirm adapter 0.2.1 or newer is loaded"
            ) from exc
        advanced = int(response["frame_end"]) - int(response["frame_start"])
        if advanced != frames:
            raise RuntimeError(
                f"port {port}: requested {frames} frames but advanced {advanced}"
            )
        return {"port": port, "action_name": action.name, **response}
    finally:
        bridge.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test input on one or more mGBA adapters")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=(55355, 55356),
        metavar="PORT",
    )
    parser.add_argument(
        "--actions",
        type=parse_action,
        nargs="+",
        default=(GBAAction.A, GBAAction.A),
        metavar="ACTION",
    )
    parser.add_argument("--frames", type=int, default=1)
    args = parser.parse_args()

    validate_target_counts(parser, args.ports, args.actions)

    with ThreadPoolExecutor(max_workers=len(args.ports)) as executor:
        futures = [
            executor.submit(step_one, args.host, port, action, args.frames)
            for port, action in zip(args.ports, args.actions, strict=True)
        ]
        results = [future.result() for future in futures]

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
