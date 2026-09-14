# Create and edit teams

[Documentation](README.md) · [Profiles](team-profiles.md) · [Generator flags](reference/generate-teams.md)

Team JSONs are shared by Showdown and the PKHeX save pipeline. Edit files directly
or generate a batch. A Pokémon may have one to four moves; generated profile
teams always have four.

## JSON format

For generated FireRed saves, an optional top-level `"trainer_name": "Matt"`
overrides the trainer name for that team only (1–7 letters, digits, or spaces).
Omitting it preserves the template's trainer name. This is separate from `name`,
which labels the team. Battle Studio sets this only for the human player.
It has no effect on Showdown battles.

```json
{
  "name": "rain_team",
  "party": [
    {
      "species": "Blastoise",
      "level": 50,
      "moves": ["Surf", "Ice Beam", "Rapid Spin", "Toxic"],
      "nature": "Bold",
      "held_item": "Leftovers",
      "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
      "evs": {"hp": 252, "def": 252, "spd": 6}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | Team label; use a filename that identifies the team as well |
| `party` | Ordered list of 1–6 Pokémon; the first member is the lead |
| `species` | Species name from the bundled Gen III catalog |
| `level` | Integer 1–100 |
| `moves` | 1–4 distinct Gen III move names, in action-slot order |
| `nature` | Optional named nature; otherwise derived from the editor's fixed PID convention |
| `held_item` | Gen III item name; default `None` |
| `ivs` | Optional stat values 0–31; omitted stats default to 31 |
| `evs` | Optional stat values 0–255; omitted stats default to 0; total at most 510 |

Stat keys are `hp`, `atk`, `def`, `spa`, `spd`, and `spe`. The generated sets use
up to 252 EVs per stat. A 252/252/6 spread spends 510 EVs, though only four of the
last six contribute a full stat point. Held items are distinct from the bag.
Ability and gender are not configurable JSON fields in the current schema.

The save editor gives Pokémon uppercase species nicknames, the template's
trainer identity, full HP/PP and friendship 255. It preserves progression and
inventory. [Template preparation](template_preparation.md) is responsible for
badges and location, and [build_states.py](automatic_states.md) combines both steps.

## Generate a flat batch

```bash
python mgba/generate_teams.py --count 40 --team-size 6 --level 50 --seed 123 \
  --competitive-rate 0.6 --chaos-rate 0.2 --competitive-item-rate 0.8 \
  --prefix training_a --output-dir config/teams/training_a
```

The rates allocate individual Pokémon slots across the batch before shuffling.
Competitive and chaos fractions must sum to at most one; remaining slots use
species learnsets. This differs from `--team-profile mixed`, which chooses a
whole team's category from separate directories.

To override stats or equipment for all generated Pokémon:

```bash
python mgba/generate_teams.py --count 10 --team-size 3 \
  --evs hp=252,def=252,spd=6 --ivs hp=31,atk=0 \
  --nature Bold --held-item Leftovers --output-dir config/teams/defensive
```

Generation is local and requires only the base Python environment. It does not
create saves. Each batch includes a summary describing the generated distribution.
All flags are in the [generator reference](reference/generate-teams.md).

## Use the teams

```bash
python mgba/train_ppo.py --backend showdown --environments 4 \
  --team-dir config/teams/training_a --total-timesteps 100000
```

For mGBA:

```bash
python mgba/build_states.py --source states/base/overworld.sav --rom FireRed.gba \
  --team-dir config/teams/training_a --headless --speed unlimited
```

The supplied examples include fire/water teams, two reordered three-starter
teams, a 40-team mixed-member batch and the categorized profile catalog.
See [automatic state generation](automatic_states.md) for output paths and reset
checks, or [build_team.py](reference/build-team.md) to create only a battery save.
