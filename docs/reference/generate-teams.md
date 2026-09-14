# Generate a custom team batch

[Utility index](README.md) · [Workflow guide](../team-generation.md) · [Source](../../mgba/generate_teams.py)

Create a flat directory of teams with configurable species, move styles and stats.

```bash
python mgba/generate_teams.py --count 40 --team-size 6 --seed 123 \
  --competitive-rate 0.6 --chaos-rate 0.2 --competitive-item-rate 0.8 \
  --output-dir config/teams/training_a
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--count` | `40` | Number of team files (default: 40) |
| `--team-size` | `6` | Pokémon per team, 1–6. |
| `--level` | `50` | Level assigned to each generated Pokémon; integer from 1 to 100. |
| `--seed` | `42` | Seed for reproducible team generation. |
| `--competitive-rate` | `0.6` | Fraction of Pokémon slots across the batch filled with curated competitive sets; 0–1. |
| `--chaos-rate` | `0.2` | Fraction of slots with unrestricted Gen III moves; 0–1. Together with competitive rate, must not exceed 1. |
| `--competitive-item-rate` | `0.8` | Fraction of Pokémon slots given common competitive held items; 0–1, overridden by `--held-item`. |
| `--species-pool` | `all` | Species candidates for generated slots: `all`, `fully-evolved`, or `kanto`. |
| `--iv-mode` | `perfect` | `perfect` uses 31 IVs; `random` draws each IV from 0–31. Explicit `--ivs` overrides it. |
| `--ev-mode` | `trained` | `trained`, `zero`, or `random` EV allocation. Explicit `--evs` overrides it. |
| `--evs` | Not set | Override every Pokémon with a spread such as `hp=252,def=252,spd=6`; omitted stats are 0; total at most 510. |
| `--ivs` | Not set | Override every Pokémon with a spread such as `hp=31,atk=0`; omitted stats are 31; each value 0–31. |
| `--held-item` | Not set | Override every held item by its Gen III name, or `None`. |
| `--nature` | Not set | Override every nature with a named Pokémon nature, such as `Modest`. |
| `--prefix` | `team` | Filename prefix; use a new one for separate state batches |
| `--output-dir` | Required | New directory for generated JSONs |

## Usage notes

Rates apply to individual Pokémon across the whole batch, rounded to slots and then shuffled. The remaining slots after competitive and chaos allocation use species learnsets. This command's `--team-size` works independently; the profile-only size restriction belongs to training/build-state selection. Generation needs no emulator, ROM or .NET. Outputs include team JSONs and a generation summary.
