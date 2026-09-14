# Evaluation plan templates

[Walkthrough](../../docs/model-evaluation.md) · [All runner flags](../../docs/reference/evaluate-models.md)

Use these JSON files with `python mgba/evaluate_models.py --config PATH`.
They describe the battle tests, while `--models-dir` chooses the participating
checkpoints and `--run-dir` chooses where results are saved.

| Template | Configurations |
|---|---|
| [round_robin.json](round_robin.json) | Six-type profile, mixed three-Pokémon teams, competitive six-Pokémon teams |
| [starters.json](starters.json) | The bundled Charizard-lead and Venusaur-lead starter teams |

## Top-level keys

| Key | Default | Meaning |
|---|---|---|
| `version` | Required | Must be `1`. |
| `name` | `Model round robin` | Display title. |
| `episodes` | `1000` | Battles per model pair per configuration. Positive and even; includes both player seats. |
| `max_steps` | `200` | Maximum decisions per battle; unfinished battles truncate. |
| `seed` | `10001` | Base seed for team selection, battle RNG and Random-policy choices. |
| `include_random` | `true` | Add a uniform random legal-action participant. |
| `scenarios` | Required | Nonempty list of team configurations. All use the same battle count. |

## Scenario keys

| Key | Meaning |
|---|---|
| `id` | Unique lowercase ID using letters, digits, `_` or `-`; starts with a letter. `overall` is reserved. |
| `name` | Optional display name; defaults to the ID. |
| `teams` | Exactly two team JSON paths in game-player order. |
| `team_dir` | Draw teams from the directory's direct `.json` files. |
| `training_profile` | Named project profile under `config/training_profiles/NAME/teams`. |
| `team_profile` | Built-in `competitive`, `casual`, `random` or `mixed` catalog. |
| `team_size` | Optional integer 1–6; requires `team_profile`. Omit to allow all catalog sizes. |
| `mirror_teams` | Optional boolean, default `false`; draw one team for both players. Use with a directory or profile. |

Choose **exactly one** source per scenario: `teams`, `team_dir`, `training_profile`
or `team_profile`. Unknown keys are rejected so misspelled settings cannot silently
change the intended test.

`teams` and `team_dir` paths are relative to the configuration file, or may be
absolute. Profile names resolve within the project. Use the standard
[team JSON schema](../../docs/team-generation.md) for moves, levels, EVs, IVs and items.

Directory sampling chooses two distinct files when available; a one-file directory
uses that team for both players. `mixed` samples competitive/casual/random
categories equally per player, independently of category directory sizes.
`mirror_teams` gives both players the same sampled team. For a fixed mirror, put
the same path twice in `teams` and omit `mirror_teams`.

Each model pair receives the same sampled teams and seeds. A scenario with
`episodes: 1000` at the top level samples 500 matchups and plays each twice with
the model seats swapped. Teams stay in their game-player positions during the
swap, giving each model a turn with each team.

These plans are independent of training campaigns. They neither train models
nor promote stages. Keep a saved tournament directory to resume it; use a new
directory after changing the plan or adding participants.
