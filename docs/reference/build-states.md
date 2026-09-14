# Build counter states

[Utility index](README.md) · [Workflow guide](../automatic_states.md) · [Source](../../mgba/build_states.py)

Prepare a source save, generate parties, boot Continue and capture verified mGBA counter snapshots.

```bash
python mgba/build_states.py --source states/base/overworld.sav --rom FireRed.gba \
  --team-profile mixed --team-size 3 --headless --speed unlimited --skip-existing
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--source` | Required | Existing parcel-complete battery .sav, e.g. FireRed.sav; not a .ss1 |
| `--rom` | Required | Path to the supported FireRed v1.0 ROM. |
| `--teams` | Not set | Team JSON files (default: fire_vs_water and water_vs_fire) |
| `--team-dir` | Not set | Build every *.json team in this directory, in filename order |
| `--team-profile` | Not set | Select competitive, casual, random, or a mixture of all three. Choices: `competitive`, `casual`, `random`, `mixed`. |
| `--team-size` | Not set | Build only this size (1–6); requires `--team-profile`. Omit to build all selected sizes. Choices: `1`, `2`, `3`, `4`, `5`, `6`. |
| `--emulator` | `.local/mgba/bin/mgba-qt` | Path to the patched project-local `mgba-qt` executable. |
| `--speed` | `1` | Positive emulator speed multiplier (for example `4`) or `unlimited`. |
| `--visible` | Off | Watch the automatic preparation |
| `--headless` | On | Run without windows using Xvfb (default) |
| `--timeout` | `180` | Seconds allowed per emulator |
| `--workers` | `1` | Concurrent team generation/capture jobs. Positive; capped at teams still to build. Template preparation runs once before workers start. |
| `--output-dir` | `states` | Root containing `generated/` battery saves and `counter/` states/metadata. |
| `--force` | Off | Back up and replace existing outputs |
| `--skip-existing` | Off | Resume a batch, skipping outputs whose hashes still match their inputs |

## Usage notes

Choose at most one of `--teams`, `--team-dir`, or `--team-profile`. With none, both fire/water examples are built. `--team-size` requires `--team-profile`. `--visible` and `--headless` are exclusive; headless is the default. `--force` and `--skip-existing` are exclusive. Badges and counter warp are included automatically. Outputs and metadata are described in the [state guide](../automatic_states.md).
