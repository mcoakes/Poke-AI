#!/usr/bin/env python3
"""Generate team battery saves and verified counter snapshots without GUI input."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from launch_emulators import lua_string, stop_processes
from prepare_template import backup_file, find_dotnet, prepare

from firered_rl.teams import TeamSpec
from firered_rl.launch_options import add_speed_argument, speed_options
from firered_rl.team_profiles import add_profile_arguments, profile_files, profile_location, validate_profile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_teams(args, project):
    profile, size = getattr(args, "team_profile", None), getattr(args, "team_size", None)
    validate_profile(profile, size)
    directory = getattr(args, "team_dir", None)
    if profile is not None:
        if directory is not None or getattr(args, "teams", None) is not None:
            raise ValueError("choose --team-profile, --team-dir, or --teams")
        groups = profile_files(project / "config/teams/profiles", profile, size)
        teams = [p for group in groups.values() for p in group]
    elif directory is not None:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"team directory does not exist: {directory}")
        teams = sorted(directory.glob("*.json"))
        if not teams:
            raise ValueError(f"no team JSONs in {directory}")
    else:
        teams = args.teams or [project / "config/teams/fire_vs_water.json",
                               project / "config/teams/water_vs_fire.json"]
    return [path.expanduser().resolve() for path in teams]


def team_outputs(args, project, teams, specs):
    """Mirror the category/size tree only when profile selection is requested."""
    keys, outputs = [], []
    output_root = (getattr(args, "output_dir", None) or project / "states").expanduser().resolve()
    for team, spec in zip(teams, specs, strict=True):
        if getattr(args, "team_profile", None):
            root = (project / "config/teams/profiles").resolve()
            category, size = profile_location(team, root)
            if len(spec.party) != size or any(len(mon.moves) != 4 for mon in spec.party):
                raise ValueError(f"profile template must have {size} Pokemon with four moves each: {team}")
            key = Path("profiles") / category / str(size) / team.stem
        else:
            key = Path(team.stem)
        if any(not re.fullmatch(r"[\w-]+", part) for part in key.parts):
            raise ValueError("team filenames must contain only letters, digits, _ or -")
        keys.append(key)
        outputs.append((output_root / "generated" / key.with_suffix(".sav"),
                        output_root / "counter" / key.with_suffix(".ss1"),
                        output_root / "counter" / key.with_suffix(".ss1.json")))
    if len(set(keys)) != len(keys):
        raise ValueError("team filenames must have unique names within each output directory")
    return keys, outputs


def verified_existing(group, team, source_hash, party_size):
    """Resume a bulk build only when the complete output still matches its inputs."""
    if not any(p.exists() for p in group):
        return None
    if not all(p.is_file() for p in group):
        raise ValueError(f"incomplete existing outputs: {group[1]}; use --force to replace them")
    data = json.loads(group[2].read_text())
    checks = dict(source_sha256=source_hash, team_sha256=digest(team),
                  save_sha256=digest(group[0]), state_sha256=digest(group[1]))
    if not data.get("ok") or len(data.get("verified_party", [])) != party_size or any(data.get(k) != v for k, v in checks.items()):
        raise ValueError(f"existing outputs differ from the requested inputs: {group[1]}; use --force to rebuild")
    return data


def publish(files, force):
    """Publish only verified outputs; back up any existing outputs first."""
    for source, destination in files:
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError(f"missing staged output: {source}")
        if destination.exists() and not force:
            raise FileExistsError(f"output exists: {destination}; use --force for backups")
    backups = []
    for _, destination in files:
        if destination.exists():
            backups.append(str(backup_file(destination)))
    for source, destination in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Each replacement is atomic, though the whole group is not a transaction.
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            staged = Path(stream.name)
            try:
                stream.write(source.read_bytes())
                stream.flush()
                os.replace(staged, destination)
            finally:
                staged.unlink(missing_ok=True)
    return backups


def capture(project, emulator, rom, save, job, party_count, visible, timeout, speed=1.0, *, cancel=None):
    instance_rom = job / "FireRed.gba"
    shutil.copyfile(rom, instance_rom)
    shutil.copyfile(save, job / "FireRed.sav")
    state, report = job / "counter.ss1", job / "capture.json"
    bootstrap = job / "capture.lua"
    values = {
        "CAPTURE_PROFILE_PATH": str(project / "mgba/profiles/firered_us_v1_0.lua"),
        "CAPTURE_STATE_PATH": str(state), "CAPTURE_REPORT_PATH": str(report),
        "CAPTURE_PROGRESS_PATH": str(job / "progress.json"),
    }
    bootstrap.write_text(
        "".join(f"{key} = {lua_string(value)}\n" for key, value in values.items())
        + f"CAPTURE_PARTY_COUNT = {party_count}\nCAPTURE_MAX_FRAMES = {int(timeout * 60)}\n"
        + f"dofile({lua_string(str(project / 'mgba/capture_state.lua'))})\n",
        encoding="utf-8")
    command = [str(emulator), *speed_options(speed), "-C", f"savegamePath={job}",
               "-C", "autoload=0", "-C", "loadStateExtdata=31", str(instance_rom)]
    if not visible:
        command = ["xvfb-run", "-a", *command]
    environment = os.environ.copy()
    environment["MGBA_LUA_SCRIPT"] = str(bootstrap)
    log_path = job / "emulator.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=project, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                check_cancelled(cancel)
                if report.is_file():
                    try:
                        result = json.loads(report.read_text())
                    except json.JSONDecodeError:
                        time.sleep(0.05)
                        continue
                    if not result.get("ok"):
                        raise RuntimeError(f"capture failed: {result}; log: {log_path}")
                    if not state.is_file() or state.stat().st_size == 0:
                        raise RuntimeError(f"capture reported success without a state: {log_path}")
                    return state, result
                if process.poll() is not None:
                    raise RuntimeError(f"mGBA exited before capture; log: {log_path}")
                time.sleep(0.1)
            raise RuntimeError(f"capture timed out; see {job / 'progress.json'} and {log_path}; "
                               "if no progress file exists, check the mGBA autoload patch")
        finally:
            stop_processes([process])


def check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise RuntimeError("state generation cancelled")


def edit_team(command, cancel):
    """Keep parallel workers interruptible while PKHeX writes their separate saves."""
    if cancel is None:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    check_cancelled(cancel)
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            deadline = time.monotonic() + 120
            while process.poll() is None:
                check_cancelled(cancel)
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, 120)
                cancel.wait(0.1)
            stdout.seek(0)
            stderr.seek(0)
            return subprocess.CompletedProcess(command, process.returncode,
                                               stdout.read().decode("utf-8", "replace"),
                                               stderr.read().decode("utf-8", "replace"))
        finally:
            stop_processes([process])


def build(args, project):
    source, rom = args.source.expanduser().resolve(), args.rom.expanduser().resolve()
    emulator = args.emulator.expanduser().resolve()
    teams = select_teams(args, project)
    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise ValueError("--workers must be positive")
    if args.timeout < 10:
        raise ValueError("--timeout must be at least 10 seconds")
    if not source.is_file():
        raise FileNotFoundError(
            f"source battery save not found: {source}\n"
            "--source must point to an existing, parcel-complete battery .sav; "
            "the builder does not create this input.\n"
            "states/base/overworld.sav is the optional backup path from setup. "
            "If your working save is FireRed.sav in the project root, use --source FireRed.sav."
        )
    inputs = [("ROM", rom), ("emulator executable", emulator)]
    inputs.extend(("team JSON", path) for path in teams)
    for label, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not args.visible and shutil.which("xvfb-run") is None:
        raise RuntimeError("install xvfb or pass --visible")
    specs = [TeamSpec.load(path) for path in teams]
    names, outputs = team_outputs(args, project, teams, specs)
    skip = getattr(args, "skip_existing", False)
    if skip and args.force:
        raise ValueError("choose --skip-existing or --force")
    input_paths = {p.resolve() for p in (source, rom, *teams)}
    input_ids = {(p.stat().st_dev, p.stat().st_ino) for p in input_paths}
    for group in outputs:
        for path in group:
            if path.resolve() in input_paths or (path.exists() and (path.stat().st_dev, path.stat().st_ino) in input_ids):
                raise ValueError(f"output would replace an input: {path}")
            if path.exists() and not args.force and not skip:
                raise FileExistsError(f"output exists: {path}; use --force for backups")
    source_hash = digest(source)
    existing = [verified_existing(group, team, source_hash, len(spec.party)) if skip else None
                for group, team, spec in zip(outputs, teams, specs, strict=True)]
    if all(data is not None for data in existing):
        print(f"All {len(teams)} teams already built and verified.", flush=True)
        return {"ok": True, "teams": existing, "skipped": len(teams), "backups": []}
    run_root = project / ".local/capture"
    run_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="build-", dir=run_root))
    print(f"build directory: {run}", flush=True)
    template = run / "template.sav"
    print("Preparing source template (eight badges and link counter).", flush=True)
    prepared = prepare(project, source, template, rom, True, True)
    source_hash = json.loads(Path(prepared["metadata"]).read_text())["source_sha256"]
    dotnet = find_dotnet(project)
    dll = project / "tools/pkhex_team_editor/bin/Debug/net10.0/PKHeXTeamEditor.dll"
    pending = sum(data is None for data in existing)
    workers = min(workers, pending)
    print(f"Capturing {pending} teams with {workers} worker(s).", flush=True)
    cancel = threading.Event() if workers > 1 else None

    def build_one(index):
        team, name, spec, group = teams[index - 1], names[index - 1], specs[index - 1], outputs[index - 1]
        if existing[index - 1] is not None:
            print(f"[{index}/{len(teams)}] verified existing {name}", flush=True)
            return existing[index - 1], []
        check_cancelled(cancel)
        job = run / name
        job.mkdir(parents=True)
        team_snapshot = job / "team.json"
        shutil.copyfile(team, team_snapshot)
        spec = TeamSpec.load(team_snapshot)
        save = job / "generated.sav"
        print(f"[{index}/{len(teams)}] generating {name} ({len(spec.party)} Pokemon)", flush=True)
        edited = edit_team([dotnet, str(dll), "--template", str(template),
                            "--team", str(team_snapshot), "--output", str(save)], cancel)
        if edited.returncode:
            raise RuntimeError(f"team generation failed: {edited.stderr or edited.stdout}")
        editor_report = json.loads(edited.stdout)
        requested_trainer = json.loads(team_snapshot.read_text()).get("trainer_name")
        if requested_trainer is not None and editor_report.get("trainer") != requested_trainer.strip():
            raise RuntimeError("team editor did not verify the requested trainer name; rebuild the updated editor")
        if not editor_report.get("ok") or len(editor_report.get("verified_party", [])) != len(spec.party):
            raise RuntimeError("team editor did not verify the written party; rebuild the updated editor")
        print(f"loading and capturing {name}; waiting for ready counter", flush=True)
        check_cancelled(cancel)
        state, report = capture(project, emulator, rom, save, job, len(spec.party),
                                args.visible, args.timeout, args.speed,
                                **({"cancel": cancel} if cancel is not None else {}))
        report.update(team=team.stem, source_sha256=source_hash, team_sha256=digest(team_snapshot),
                      save_sha256=digest(save), state_sha256=digest(state),
                      save=str(group[0]), state=str(group[1]),
                      verified_party=editor_report["verified_party"])
        if getattr(args, "team_profile", None):
            category, size = profile_location(team, (project / "config/teams/profiles").resolve())
            report.update(team_profile=category, team_size=size)
        manifest = job / "verified.json"
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # Keep completed teams if a later emulator capture fails or the user stops the batch.
        check_cancelled(cancel)
        backups = publish(list(zip((save, state, manifest), group, strict=True)), args.force)
        # capture() has stopped the child. Retain diagnostics and saves, not N ROM copies.
        (job / "FireRed.gba").unlink(missing_ok=True)
        print(f"[{index}/{len(teams)}] complete: {group[1]}", flush=True)
        return report, backups

    indices = range(1, len(teams) + 1)
    if workers == 1:
        completed = [build_one(index) for index in indices]
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        first_error = []
        error_lock = threading.Lock()
        try:
            # A failed worker cancels its peers even if its result is later in input order.
            def guarded(index):
                try:
                    return build_one(index)
                except BaseException as error:
                    with error_lock:
                        if not first_error:
                            first_error.append(error)
                        cancel.set()
                    raise
            try:
                completed = list(executor.map(guarded, indices))
            except Exception:
                # Report the original failing capture, not a peer's cancellation.
                if first_error:
                    raise first_error[0]
                raise
        finally:
            cancel.set()
            executor.shutdown(wait=True, cancel_futures=True)
    results = [report for report, _ in completed]
    backups = [path for _, paths in completed for path in paths]
    return {"ok": True, "teams": results, "skipped": sum(data is not None for data in existing),
            "backups": backups, "build_directory": str(run), "workers": workers}


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Existing parcel-complete battery .sav, e.g. FireRed.sav; not a .ss1")
    parser.add_argument("--rom", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--teams", type=Path, nargs="+",
                           help="Team JSON files (default: fire_vs_water and water_vs_fire)")
    selection.add_argument("--team-dir", type=Path,
                           help="Build every *.json team in this directory, in filename order")
    add_profile_arguments(parser, selection)
    parser.add_argument("--emulator", type=Path, default=project / ".local/mgba/bin/mgba-qt")
    add_speed_argument(parser)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--visible", dest="visible", action="store_true",
                         help="Watch the automatic preparation")
    display.add_argument("--headless", dest="visible", action="store_false",
                         help="Run without windows using Xvfb (default)")
    parser.set_defaults(visible=False)
    parser.add_argument("--timeout", type=float, default=180, help="Seconds allowed per emulator")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent team generation/capture jobs; capped at teams still to build (default: 1)")
    parser.add_argument("--output-dir", type=Path,
                        help="Save root containing generated/ and counter/ (default: states/)")
    replacement = parser.add_mutually_exclusive_group()
    replacement.add_argument("--force", action="store_true", help="Back up and replace existing outputs")
    replacement.add_argument("--skip-existing", action="store_true",
                             help="Resume a batch, skipping outputs whose hashes still match their inputs")
    args = parser.parse_args()
    try:
        result = build(args, project)
    except KeyboardInterrupt:
        parser.exit(130, "capture interrupted; child emulator stopped\n")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"build_states: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
