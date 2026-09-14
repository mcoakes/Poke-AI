# Type progression curriculum

[Run the ten-stage pipeline](../../../docs/curriculum.md) · [Runner flags](../../../docs/reference/run-curriculum.md) · [Create your own campaign](../../../docs/creating-campaigns.md)

This preset defaults to a fixed opponent. The
[self-play variant](../type_progression_selfplay/README.md) uses the same teams
and advances the frozen opponent after each promoted stage.

[`curriculum.json`](curriculum.json) adds one team per stage and retains all
earlier teams. Stage 1 is an exact copy of `six_types`; stage 2 changes party and
move order. Later additions introduce species, coverage, items, status and party
sizes. These are deliberately designed learning tasks, not automatically ranked
competitive teams.

`teams/01_*.json` through `teams/10_*.json` are the ten additions.
[`teams/11_final_validation.json`](teams/11_final_validation.json) is only the
new-team probe after stage 10. Every Pokémon has four moves, level 50, IVs of
31 and an EV total within 510. Names, moves and items use the project's Gen III
catalog. Later role sets adapt the project's existing competitive-style templates.
Catalog learnset checks are not full event/breeding/tournament legality certification.

To make another curriculum, copy this directory, edit its ordered `stages`
and `next_team`, then start a new run with `--curriculum PATH/curriculum.json`.
Stage numbers must be consecutive from 1. Paths are relative to the manifest;
each stage and the final probe need distinct team files. The runner snapshots
the files at initialization; edits do not change an existing run.

[`promotion.json`](promotion.json) documents the built-in thresholds. To use
an edited copy, explicitly pass `--promotion-rules PATH` on a new run. Its five
values are fractions from 0 to 1; `max_regression: 0.03` means three percentage
points. This file is a template, not an automatically reloaded runtime setting.
