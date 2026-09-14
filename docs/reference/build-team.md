# Build one battery save

[Utility index](README.md) · [Workflow guide](../team-generation.md) · [Source](../../mgba/build_team.py)

Replace a prepared template's party from one team JSON, preserving its existing progression and location.

```bash
python mgba/build_team.py --template states/templates/firered_link_counter.sav \
  --team config/teams/fire_vs_water.json --output states/generated/fire_vs_water.sav
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--template` | Required | Prepared FireRed battery `.sav` to copy; never modified. |
| `--team` | Required | One JSON team configuration. |
| `--output` | Required | Destination battery `.sav`; must differ from both inputs. |

## Usage notes

This creates a `.sav`, not an emulator snapshot. It does not grant badges or set a warp; use a prepared template. An existing output can be replaced without a backup by this low-level utility. Use [build_states.py](build-states.md) for verified capture and managed replacement.
