#!/usr/bin/env python3
"""Make an edited training template, preserving the source battery save."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROM_SHA1 = "41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc"
PROFILE = "firered_us_v1_0"


def find_dotnet(project: Path) -> str:
    local = project / ".local/dotnet/dotnet"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    system = shutil.which("dotnet")
    if system:
        return system
    raise RuntimeError(".NET SDK not found; follow the .NET 10 installation steps in README.md")


def run_editor(project: Path, source: Path, output: Path, badges: bool, warp: bool) -> dict:
    dotnet = find_dotnet(project)
    editor = project / "tools/pkhex_team_editor/PKHeXTeamEditor.csproj"
    # Keep build output separate from JSON, including .NET's first-run messages.
    built = subprocess.run([dotnet, "build", str(editor), "--nologo"],
                           text=True, capture_output=True, timeout=180, check=False)
    if built.returncode:
        raise RuntimeError(f"editor build failed:\n{built.stdout}\n{built.stderr}")
    dll = editor.parent / "bin/Debug/net10.0/PKHeXTeamEditor.dll"
    completed = subprocess.run([
        dotnet, str(dll), "--prepare-template", str(source), "--output", str(output),
        "--profile", PROFILE, "--grant-eight-badges", str(badges).lower(),
        "--warp-link-counter", str(warp).lower(),
    ], text=True, capture_output=True, timeout=60, check=False)
    if completed.returncode:
        raise RuntimeError(f"save edit failed: {completed.stderr.strip() or completed.stdout.strip()}")
    report = json.loads(completed.stdout)
    if not report.get("ok") or not report.get("checksums_valid"):
        raise RuntimeError(f"editor verification failed: {report}")
    return report


def backup_file(path: Path) -> Path:
    """Never overwrite an older backup."""
    number = 0
    while True:
        backup = path.with_name(path.name + (f".bak.{number}" if number else ".bak"))
        try:
            stream = backup.open("xb")
        except FileExistsError:
            number += 1
            continue
        with stream, path.open("rb") as original:
            shutil.copyfileobj(original, stream)
        return backup


def prepare(project: Path, source: Path, output: Path, rom: Path,
            badges: bool, warp: bool, force: bool = False) -> dict:
    source, output, rom = (p.expanduser().resolve() for p in (source, output, rom))
    if not source.is_file() or not rom.is_file():
        raise ValueError("source save and ROM must both exist")
    if output.suffix.lower() != ".sav" or output == rom:
        raise ValueError("output must be a separate .sav file, never the ROM")
    if source == output or (output.exists() and source.samefile(output)):
        raise ValueError("source and output must differ; the source is never overwritten")
    if not badges and not warp:
        raise ValueError("select --grant-eight-badges and/or --warp-link-counter")
    if hashlib.sha1(rom.read_bytes()).hexdigest() != ROM_SHA1:
        raise ValueError(f"unsupported ROM: expected FireRed USA/Europe v1.0 SHA-1 {ROM_SHA1}")
    metadata_path = output.with_suffix(output.suffix + ".json")
    if metadata_path.exists() and source.samefile(metadata_path):
        raise ValueError("metadata path refers to the source save")
    if (output.exists() or metadata_path.exists()) and not force:
        raise FileExistsError("template/metadata already exists; --force backs up and replaces it")
    snapshot = source.read_bytes()
    if len(snapshot) != 131072:
        raise ValueError("expected a raw 131072-byte battery .sav, not an emulator save state")
    output.parent.mkdir(parents=True, exist_ok=True)
    backups = []
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as directory:
        staged_source = Path(directory) / "source.sav"
        staged_output = Path(directory) / "edited.sav"
        staged_source.write_bytes(snapshot)
        report = run_editor(project, staged_source, staged_output, badges, warp)
        if source.read_bytes() != snapshot:
            raise RuntimeError("source changed during preparation; close mGBA and retry")
        edited = staged_output.read_bytes()
        if len(edited) != 131072:
            raise RuntimeError("editor returned an unexpected save size")
        metadata = {
            "template": str(output), "profile": PROFILE, "rom_sha1": ROM_SHA1,
            "source_sha256": hashlib.sha256(snapshot).hexdigest(),
            "sha256": hashlib.sha256(edited).hexdigest(), "pkhex": report,
        }
        staged_metadata = Path(directory) / "metadata.json"
        staged_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for existing in (output, metadata_path):
            if existing.exists():
                if not force:
                    raise FileExistsError(f"output appeared during preparation: {existing}")
                backups.append(str(backup_file(existing)))
        os.replace(staged_output, output)
        os.replace(staged_metadata, metadata_path)
    return {"template": str(output), "metadata": str(metadata_path),
            "backups": backups, **report}


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Edit FireRed badges and link-counter spawn")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rom", type=Path, required=True, help="Verify the supported FireRed v1.0 ROM")
    parser.add_argument("--output", type=Path,
                        default=project / "states/templates/firered_link_counter.sav")
    parser.add_argument("--grant-eight-badges", action="store_true")
    parser.add_argument("--warp-link-counter", action="store_true")
    parser.add_argument("--force", action="store_true", help="Back up and replace existing output")
    args = parser.parse_args()
    try:
        result = prepare(project, args.source, args.output, args.rom,
                         args.grant_eight_badges, args.warp_link_counter, args.force)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"prepare_template: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
