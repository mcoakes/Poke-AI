# Controlled training profiles

Use `--training-profile NAME` with Showdown training or evaluation. A profile
is a directory of ordinary team JSONs in `NAME/teams/`. Both players draw from
that directory; one file gives both players that team every battle.

| Profile | Purpose |
|---|---|
| [six_types](six_types/README.md) | Six distinct types, with four same-type attacks per Pokémon |

The [growing-team curriculum](../../docs/curriculum.md) starts from `six_types`
and adds nine more teams with evaluation checks. Its ordered inputs live under
`config/curricula/type_progression/`; the runner builds cumulative runtime pools.

To add a profile, create a new directory with a `teams/` subdirectory and put
team JSON files there. No Python changes are needed. Names use lowercase letters,
digits, underscores and hyphens. Only direct `.json` children are sampled.
Use `--mirror-teams` to give both players the same sampled team when a profile
contains multiple files. Ordinary `--team-dir PATH` remains available.

These small profiles are separate from the competitive/casual/random category
catalog. `--team-size` only filters that category catalog via `--team-profile`;
team sizes in controlled profiles come from their JSON files.

[Evaluation](../../docs/evaluation.md) · [Team format](../../docs/team-generation.md)
