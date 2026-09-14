# Type progression with previous-stage opponents

[Opponent progression guide](../../../docs/curriculum-opponents.md) · [Team stages](../type_progression/README.md)

This is an alternative campaign preset. It uses the same ten cumulative team
additions as `type_progression`, with `opponent_mode: previous-stage` in
[`curriculum.json`](curriculum.json).

Stage 1 faces the initial `--opponent-checkpoint`. Each later stage faces a frozen
copy of the previous stage's promoted candidate. Both learner and opponent start
from that candidate, but only the learner updates. Retries keep the opponent
chosen when the stage started.

For a new run, select:

```bash
--curriculum config/curricula/type_progression_selfplay/curriculum.json
```

The full launch command and continuation instructions are in the
[opponent progression guide](../../../docs/curriculum-opponents.md).
An explicit `--opponent-mode fixed` overrides this preset for a new campaign.

Team paths reference `../type_progression/teams/` so both presets share the same
source teams. Each campaign snapshots its files at initialization; later edits
do not change an existing run. Promotion thresholds remain the runner's defaults
unless you supply `--promotion-rules`.
