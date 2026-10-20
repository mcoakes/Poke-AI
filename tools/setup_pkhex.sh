#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkhex_dir="$project_root/third_party/PKHeX"
pkhex_commit="77dcd3a7895bceaafbbff12d25bdf77c1acd8ca5"

if command -v dotnet >/dev/null 2>&1; then
    dotnet_command="$(command -v dotnet)"
elif [[ -x "$project_root/.local/dotnet/dotnet" ]]; then
    dotnet_command="$project_root/.local/dotnet/dotnet"
else
    echo "error: .NET 10 SDK not found; follow the .NET installation steps in README.md" >&2
    exit 1
fi

dotnet_major="$($dotnet_command --version | cut -d. -f1)"
if [[ "$dotnet_major" -lt 10 ]]; then
    echo "error: .NET 10 or newer is required; found $($dotnet_command --version)" >&2
    exit 1
fi

if [[ ! -d "$pkhex_dir/.git" ]]; then
    git clone --filter=blob:none https://github.com/kwsch/PKHeX.git "$pkhex_dir"
fi

git -C "$pkhex_dir" fetch origin "$pkhex_commit"
git -C "$pkhex_dir" checkout --detach "$pkhex_commit"
"$dotnet_command" build "$project_root/tools/pkhex_team_editor/PKHeXTeamEditor.csproj"
