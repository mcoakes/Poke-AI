# Developer utilities

[Utility index](README.md) · [Development and tests](../development.md)

These tools prepare pinned sources, export data or test lower-level components.
Run them from the project root.

## Generate a generalization campaign

```bash
python tools/generate_generalization_campaign.py \
  --output-dir config/curricula/my_generalization --seed 42
```

Creates ten stage directories with training/validation teams and a separate final
test set. Requires Python training extras and the bundled catalog; no ROM or
network. Existing nonempty output directories are rejected. See the
[complete plan and every flag](../generalization-campaign.md#build-a-different-sized-campaign).

## Setup mGBA link

```bash
python tools/setup_mgba_link.py
```

[Source](../../tools/setup_mgba_link.py). No flags, including no `--help` mode.
Checks the pinned commit in `third_party/mgba`, applies the autoload, native-link,
socket-lifecycle and runtime-speed patches, and recognizes changes already applied. It refuses
an incompatible source state. It does not compile or install mGBA.
Follow the [complete CMake commands](../setup/mgba.md) afterward.

## Setup PKHeX

```bash
bash tools/setup_pkhex.sh
```

[Source](../../tools/setup_pkhex.sh). No flags. Finds the .NET SDK, fetches and
checks out the pinned revision under `third_party/PKHeX`, then builds the editor.
See [.NET installation](../setup/save-editor.md) first. Treat that checkout as the
pinned dependency directory; the helper selects its configured commit.

## Export team catalog

```bash
python tools/export_team_catalog.py --source third_party/pokefirered \
  --output config/generation/firered_us_v1_0.json
```

[Source](../../tools/export_team_catalog.py). Reads species, moves, items and
learnsets from the pret source tree. Use commit
`c75f352304d529f6ba92d4f74b9cf8b5c3810788`; the exporter labels that provenance but
does not verify the checkout hash for you. A ROM is not required.

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit |
| `--source` | Required | Root of the matching pret/pokefirered source checkout |
| `--output` | Required | Output JSON path; existing output is overwritten |

## Test mGBA socket lifecycle

```bash
python tools/test_mgba_socket_lifecycle.py --source third_party/mgba
```

[Source](../../tools/test_mgba_socket_lifecycle.py). Requires POSIX sockets and a C
compiler available as `cc` (provided by `build-essential`). Extracts the selected
source's socket close/receive functions and compiles a temporary harness. Checks
repeated close, EOF and retryable nonblocking reads. This tests source behavior,
not whether the installed Qt executable was rebuilt from that source.

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit |
| `--source` | `third_party/mgba` | mGBA source checkout containing `src/script/socket.c` |

## PKHeX editor

The Python [template](prepare-template.md), [team](build-team.md) and
[state](build-states.md) utilities provide the normal workflow. The underlying
[.NET editor](../../tools/pkhex_team_editor/PKHeXTeamEditor.csproj) also accepts
direct arguments after it has been built:

```bash
dotnet tools/pkhex_team_editor/bin/Debug/net10.0/PKHeXTeamEditor.dll \
  --inspect FireRed.sav
```

Choose one mode. Every argument is a `--key VALUE` pair. There is no help switch;
booleans here require explicit `true` or `false`, unlike the Python wrappers.

| Argument | Applies to | Meaning |
|---|---|---|
| `--inspect FILE` | Inspect | Required battery save; outputs trainer, party and template-state JSON |
| `--template FILE` | Build team | Required source battery save |
| `--team FILE` | Build team | Required team JSON |
| `--output FILE` | Build team, prepare | Required output save; must not replace an input |
| `--prepare-template FILE` | Prepare | Required source battery save |
| `--profile firered_us_v1_0` | Prepare | Required supported profile ID |
| `--grant-eight-badges true` | Prepare | Required boolean; choose whether to grant all eight badges |
| `--warp-link-counter true` | Prepare | Required boolean; choose whether to set the Continue warp |

Use both preparation booleans even when one is `false`. The direct team mode
can overwrite an existing output without a backup. Python wrappers add validation,
metadata and managed replacement where documented. Success writes JSON to stdout;
errors write JSON to stderr and return exit status 1.

## Test template editing

```bash
dotnet run --project tools/pkhex_template_tests/PKHeXTemplateTests.csproj
```

[Source](../../tools/pkhex_template_tests/Program.cs). No application flags.
Uses synthetic save data to check template and party editing. Requires the
pinned PKHeX source and .NET 10 setup. No ROM or personal save is required.

## Internal runtime files

[Lua components](../../mgba/README.md) and the
[Node runner](../../sim/showdown/README.md) are launched by Python and accept
protocol messages or generated bootstrap configuration, not CLI flags. Test files
under `tests/` run with pytest as described in [development](../development.md).
