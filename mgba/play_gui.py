#!/usr/bin/env python3
"""Open a local browser team builder and launch human-versus-AI FireRed matches."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import webbrowser

from firered_rl.launch_options import add_speed_argument, speed_value
from firered_rl.checkpoints import is_latest, resolve_checkpoint
from firered_rl.play_web import MatchJobs, PlayServer, PlaySettings, TeamCatalog
from launch_link import positive_seconds
from prepare_template import find_dotnet


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, type=Path, help="Supported FireRed v1.0 ROM")
    parser.add_argument("--source", required=True, type=Path, help="Existing battery .sav with Oak's Parcel delivered")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--checkpoint", type=Path, help="Preselect a checkpoint path or latest; otherwise prefer Smart v10 from models/")
    models.add_argument("--model-list", type=Path,
                        help="Classroom model-list JSON; replaces model autodiscovery with its choices and default")
    parser.add_argument("--checkpoint-dir", type=Path, default=root / ".local/run",
                        help="List up to 200 recent checkpoint ZIPs here, plus all models/ ZIPs (default: .local/run); ignored with --model-list")
    parser.add_argument("--emulator", type=Path, default=root / ".local/mgba/bin/mgba-qt")
    parser.add_argument("--port", type=int, default=8080, help="Local web port (default: 8080)")
    parser.add_argument("--ports", type=int, nargs=2, default=[55365, 55366], help="mGBA adapter ports (default: 55365 55366)")
    parser.add_argument("--workers", type=int, default=2, help="Maximum capture workers; capped at the two teams (default: 2)")
    add_speed_argument(parser)
    parser.add_argument("--capture-speed", type=speed_value, default="unlimited", help="Speed while generating states (default: unlimited)")
    parser.add_argument("--setup-speed", type=speed_value, default="unlimited", help="Link setup speed; changes to battle speed at readiness (default: unlimited)")
    parser.add_argument("--build-timeout", type=positive_seconds, default=180, help="Seconds per state capture (default: 180)")
    parser.add_argument("--startup-timeout", type=positive_seconds, default=30, help="Seconds to wait for mGBA adapters (default: 30)")
    parser.add_argument("--connect-timeout", type=positive_seconds, default=180, help="Seconds per cable setup stage (default: 180)")
    parser.add_argument("--max-steps", type=int, default=200, help="Battle decision limit (default: 200)")
    parser.add_argument("--device", default="cpu", help="PPO inference device (default: cpu)")
    parser.add_argument("--no-browser", action="store_true", help="Print the local URL without opening the browser")
    args = parser.parse_args()
    try:
        for name in ("rom", "source", "emulator", "checkpoint", "checkpoint_dir", "model_list"):
            value = getattr(args, name)
            if value is not None and not (name == "checkpoint" and is_latest(value)):
                setattr(args, name, value.expanduser().resolve())
        for name in ("rom", "source", "emulator"):
            if not getattr(args, name).is_file():
                raise ValueError(f"--{name} file not found: {getattr(args, name)}")
        if args.checkpoint:
            resolve_checkpoint(args.checkpoint, root, search_root=args.checkpoint_dir)
        if args.workers < 1 or args.max_steps < 1 or args.build_timeout < 10:
            raise ValueError("workers/max-steps must be positive and build-timeout at least 10 seconds")
        if len(set([args.port, *args.ports])) != 3 or any(not 1 <= n <= 65535 for n in [args.port, *args.ports]):
            raise ValueError("web and adapter ports must be distinct integers from 1 to 65535")
        if hashlib.sha1(args.rom.read_bytes()).hexdigest() != "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc":
            raise ValueError("the GUI currently supports FireRed USA/Europe v1.0 only")
        if args.source.stat().st_size != 131072:
            raise ValueError("--source must be a 128 KiB battery .sav, not an emulator state")
        if not os.environ.get("DISPLAY"):
            raise ValueError("human play needs a desktop display; run this launcher in your graphical session")
        if not shutil.which("xvfb-run"):
            raise ValueError("install xvfb for background state generation")
        find_dotnet(root)
        import sb3_contrib  # noqa: F401 — fail before generating states if inference dependencies are missing
        settings = PlaySettings(project=root, **{key: getattr(args, key) for key in PlaySettings.__dataclass_fields__ if key != "project"})
        catalog = TeamCatalog(root)
        jobs = MatchJobs(settings, catalog)
        server = PlayServer(args.port, jobs, catalog)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"play_gui: {error}\nSetup: docs/play-gui.md\n")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"FireRed Battle Studio: {url}\nPlayer 1 is human; Player 2 is the saved policy.\n"
          f"Adapter ports: {args.ports[0]} / {args.ports[1]}. Ctrl+C stops the server and its match.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("Stopping the local match…", flush=True)
    finally:
        server.server_close()
        jobs.shutdown()


if __name__ == "__main__":
    main()
