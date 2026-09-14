#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from firered_rl.teams import PKHeXTeamEditor, TeamSpec


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate a FireRed save with a selected party")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--team", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = TeamSpec.load(args.team)
    dotnet = shutil.which("dotnet")
    if dotnet is None:
        local_dotnet = project / ".local/dotnet/dotnet"
        if not local_dotnet.is_file():
            parser.error(".NET SDK not found; follow the .NET 10 installation steps in README.md")
        dotnet = str(local_dotnet)
    editor = PKHeXTeamEditor(project / "tools/pkhex_team_editor", dotnet=dotnet)
    output = editor.build(args.template, args.team, args.output)
    print(f"generated team '{spec.name}' with {len(spec.party)} Pokemon: {output}")


if __name__ == "__main__":
    main()
