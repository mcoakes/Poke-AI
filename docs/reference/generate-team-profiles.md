# Generate categorized templates

[Utility index](README.md) · [Workflow guide](../team-profiles.md) · [Source](../../mgba/generate_team_profiles.py)

Create all three team categories and all six party sizes in a new directory.

```bash
python mgba/generate_team_profiles.py --count 150 --seed 20260913 \
  --level 50 --output-dir config/teams/profiles_new
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--count` | `150` | Teams per category/size directory (default: 150) |
| `--seed` | `20260913` | Seed used to generate reproducible batches for every category/size. |
| `--level` | `50` | Level assigned to each generated Pokémon; integer from 1 to 100. |
| `--output-dir` | Required | New output directory; existing files are never replaced |

## Usage notes

Count is per category/size directory, so the default writes 3 × 6 × 150 = 2,700 team JSONs. There are no category or size filters on this generator. A new output directory is required; existing files are not replaced. Training profile selection reads `config/teams/profiles/`; use `--team-dir` for a leaf of an alternate output tree.
