# Teams and profiles

[Documentation](README.md) · [Team JSON format](team-generation.md) · [All utility flags](reference/README.md)

The project includes 2,700 team JSONs: three categories × six party sizes ×
150 teams. Every Pokémon has four distinct moves. Files live in
`config/teams/profiles/CATEGORY/SIZE/`.

## Categories

| Category | Teams and moves | Stats and items |
|---|---|---|
| Competitive | Curated sets combined into balanced, offensive or bulky rosters; type variety and varied leads | Perfect IVs, role-specific EVs/natures and curated held items |
| Casual | Species learnsets, favoring a same-type attack, coverage and support when possible | Unoptimized IVs/natures and moderate EV allocation |
| Random | All 386 species and unrestricted Gen III moves, including otherwise illegal combinations | Random IVs/natures/EVs |
| Mixed | A selector over the three directories, not a fourth generated category | Equal category probability for each player |

Competitive teams draw from 60 curated species/set templates. They are
competitive-style training examples, not tournament-tested optimal teams.
The editor currently uses the first ability slot. In casual and random batches,
80% of Pokémon slots are allocated common competitive items; remaining slots use
other held-item candidates. Competitive sets use curated items throughout.

Casual excludes species without four eligible moves. Random excludes Struggle
and Sketch as assigned moves; the battle engine still supplies Struggle when
appropriate. Six IVs of 31 are permitted; EVs share a total limit of 510, rather
than 510 per stat. See [custom teams](team-generation.md) for field details.

## Select profiles during training

For Showdown, JSONs are immediately usable:

```bash
python mgba/train_ppo.py --backend showdown --environments 4 \
  --team-profile mixed --team-size 3 --total-timesteps 100000
```

For mGBA, first [generate matching counter states](automatic_states.md), then:

```bash
python mgba/train_ppo.py --rom FireRed.gba --environments 4 \
  --team-profile competitive --team-size 3 \
  --speed unlimited --headless --total-timesteps 100000
```

`--team-profile` accepts `competitive`, `casual`, `random`, or `mixed`.
`--team-size 1` through `6` is optional and requires the profile flag. Omitting
size permits all available sizes, including unequal-size opponents.

Profile selection is optional. Fixed pairs and flat-directory selection remain
available through `--teams`/`--team-dir` in Showdown or `--states`/`--state-dir` in
mGBA. The human-play launcher also accepts built mGBA profiles.

## Build profiles for mGBA

```bash
python mgba/build_states.py --source states/base/overworld.sav --rom FireRed.gba \
  --team-profile mixed --team-size 3 --headless --speed unlimited --skip-existing
```

That builds 450 states: 150 per category at size three. Omit `--team-size` to
build all 2,700. Completed outputs remain available if the batch is interrupted;
`--skip-existing` verifies their input/output hashes before skipping them.

| Input | Matching outputs |
|---|---|
| `config/teams/profiles/casual/3/casual_3_001.json` | `states/generated/profiles/casual/3/casual_3_001.sav` |
| Same input | `states/counter/profiles/casual/3/casual_3_001.ss1` and `.ss1.json` |

## Generate a new catalog

```bash
python mgba/generate_team_profiles.py --count 150 --seed 20260913 \
  --level 50 --output-dir config/teams/profiles_new
```

This generator writes all categories and sizes to a new tree. For a custom flat
batch with configurable rates and stat overrides, use
[generate_teams.py](reference/generate-teams.md). The reference explains every
flag for [catalog generation](reference/generate-team-profiles.md),
[state building](reference/build-states.md), and [training](reference/train-ppo.md).

## Random mirror matches

Add `--mirror-teams` to select one random team per battle and assign that same
team to both players. Every new battle draws again; different pairs can receive
different teams. Species, lead order, moves, stats and held items match because
both sides use the same input file.

```bash
python mgba/train_ppo.py --backend showdown --environments 8 \
  --team-profile mixed --team-size 3 --mirror-teams \
  --resume "$checkpoint" --opponent-checkpoint "$checkpoint" \
  --opponent-stochastic --total-timesteps 100000
```

Set `checkpoint` to your saved model path first. With `mixed`, categories are
chosen with equal probability per battle, then a team within that category.
You can instead use `--team-dir DIRECTORY` for Showdown. For mGBA, use
`--state-dir DIRECTORY` or built profiles, with `--rom`, `--speed` and `--headless`
as usual. Each core receives its own working copy of the selected counter state.
The human-play launcher also supports mirrored random state selection.

The flag defaults off. It requires a random source (`--team-dir`, `--state-dir`,
or `--team-profile`). For a fixed mirror, omit the flag and pass the same file
twice to `--teams` or `--states`.

The run configuration records `mirror_teams`; startup labels the source
`(mirrored)`, and episode records contain identical input paths for both players.
Seeded team selection is reproducible per pair, but mirror teams do not force
identical battle RNG outcomes, actions or learning behavior. This controls team
strength within a battle; it is not a separate fixed-checkpoint evaluation mode.
Use matching policy sampling settings when comparing equal checkpoints. Begin a
new block comparison after changing the matchup distribution.

## Controlled training profiles

`--training-profile NAME` selects a focused Showdown team directory at
`config/training_profiles/NAME/teams/`, independently of the four category
profiles above. The bundled `six_types` profile contains a six-member team of
Fire, Water, Grass, Electric, Ice and Fighting Pokémon; each has four attacks
matching its type. See [its roster and example](../config/training_profiles/six_types/README.md).

Use `--team-dir PATH` for any other flat JSON directory. `--team-size` is only a
filter for `--team-profile`; controlled profiles use the sizes in their JSONs.
