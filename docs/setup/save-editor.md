# Save editor setup

[Quickstart](../quickstart.md) · [Dependencies](README.md) · [Build counter states](../automatic_states.md)

The save pipeline uses the **.NET 10 SDK** and a pinned PKHeX.Core checkout.
Showdown training does not need either. You need the SDK to compile the editor,
not only the .NET runtime.

## New installation

Follow [installation step 4](README.md#4-install-the-net-10-sdk-and-save-editor).
It lists the Ubuntu/Mint libraries, downloads Microsoft's SDK into
`.local/dotnet`, selects it in your terminal and builds the editor. No separate
PKHeX desktop application is needed.

## Existing SDK

Check the installed SDKs:

```bash
dotnet --list-sdks
dotnet --version
```

If a working 10.0 SDK is selected, skip the SDK download and build the editor:

```bash
bash tools/setup_pkhex.sh
```

The helper pins PKHeX commit `77dcd3a7895bceaafbbff12d25bdf77c1acd8ca5`
and builds `tools/pkhex_team_editor/PKHeXTeamEditor.csproj`. Expect `Build succeeded`
and `tools/pkhex_team_editor/bin/Debug/net10.0/PKHeXTeamEditor.dll`.
The Python state builder rebuilds the small editor before processing a batch.

If you installed the project-local SDK, select it before running the setup
helper, especially when an older SDK is also on PATH:

```bash
export DOTNET_ROOT="$PWD/.local/dotnet"
export PATH="$DOTNET_ROOT:$PATH"
dotnet --version
bash tools/setup_pkhex.sh
```

Run the exports from the project root in each new terminal. The Python save
utilities prefer the project-local executable directly.

Next, [create your initial save](first-save.md), then use [automatic state generation](../automatic_states.md)
to apply badges, counter placement and parties together. The source must already
contain Oak's Parcel delivery. [Direct editor commands](../reference/developer-tools.md#pkhex-editor)
are available for inspection and maintenance.
