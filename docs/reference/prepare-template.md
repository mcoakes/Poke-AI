# Prepare a template save

[Utility index](README.md) · [Workflow guide](../template_preparation.md) · [Source](../../mgba/prepare_template.py)

Edit eight badges and/or the Continue-game counter warp in a copy of a FireRed battery save.

```bash
python mgba/prepare_template.py --source states/base/overworld.sav --rom FireRed.gba \
  --grant-eight-badges --warp-link-counter
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--source` | Required | Existing FireRed battery `.sav`; preserve a copy with Oak's Parcel already delivered. |
| `--rom` | Required | Path to the supported FireRed v1.0 ROM. |
| `--output` | `states/templates/firered_link_counter.sav` | Destination `.sav`; a `.sav.json` manifest is written alongside it. |
| `--grant-eight-badges` | Off | Set all eight badge flags in the output save. |
| `--warp-link-counter` | Off | Set a Continue-game warp to the Viridian Pokémon Center Direct Corner counter. |
| `--force` | Off | Back up and replace existing output |

## Usage notes

Choose at least one of the two edit flags. Input must be a raw 131,072-byte `.sav`, and output must differ from input. Oak's Parcel delivery is inherited, not automated. Existing output/metadata require `--force`, which preserves numbered backups.
