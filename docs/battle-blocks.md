# Compare 500-battle blocks

[Training](ppo-training.md) · [Reports](battle-logs.md) · [Comparison utility flags](reference/compare-blocks.md)

Use non-overlapping blocks to judge progress against a frozen opponent. The live
100-battle window remains useful for immediate feedback, but adjacent readings
mostly contain the same battles.

## Live summaries

Training automatically prints an extra `BLOCK` line after every 500 ended battles,
with both win rates, draw rate, completed/truncated counts, mean rewards and mean
decision count. No new training flag is needed. It writes each full block to
`battle-blocks.jsonl`; `summary.json` retains the last full block and current progress.
Both mGBA and Showdown use this reporter.

Blocks are episodes 1–500, 501–1000, 1001–1500, and so on, across all pairs in
completion order. They use ended battles, matching the console's `battle=` count.
Win-rate denominators include completed draws but exclude truncations. Reward and
decision means include every ended episode in the block. Compare truncation counts
too: a high rate can make completed-battle win rates misleading.

## Compare an existing or running session

Set the run directory to the path printed when training started:

```bash
run_dir=".local/run/ppo-YOUR_RUN"
python mgba/compare_blocks.py --run-dir "$run_dir"
```

The utility reads `episodes.jsonl`, so it also works for runs created before live
block reporting was added. You can leave that training process running. To refresh
the comparison every 30 seconds:

```bash
watch -n 30 python mgba/compare_blocks.py --run-dir "$run_dir"
```

It labels the final incomplete block as `(partial)`. For JSON or larger blocks:

```bash
python mgba/compare_blocks.py --run-dir "$run_dir" --json
python mgba/compare_blocks.py --run-dir "$run_dir" --block-size 1000
```

Changing this report's block size does not change training's automatic 500-battle
summaries. Counts restart in each run directory, including after checkpoint resume;
checkpoint timesteps do not restore prior reporting windows. Existing running
processes need a restart/resume to gain automatic live block lines, but the
comparison utility works immediately.

## Decide whether to refresh player 2

Keep the opponent checkpoint, team category/size distribution, reward configuration
and opponent sampling mode consistent while comparing blocks. Prefer complete
blocks. Illustrative P1 results of 53%, 58%, 63% suggest improvement; 55%, 54%, 56%
suggest a possible plateau. They are not proof by themselves: teams, RNG and a
changing learner still introduce variation.

A practical starting target is around 65–75% before loading a newer frozen P2.
This is a heuristic, not a required threshold. Sustained 60% can still represent
useful progress. Before promotion, a fixed-checkpoint evaluation on the same
matchups provides stronger evidence than training averages alone. This reporting
utility summarizes training; it does not perform that evaluation or automatically
replace an opponent.

Refresh P2 by starting the next run with the newer model as both `--resume` and
`--opponent-checkpoint`, keeping P2 frozen. Keep older checkpoints for comparisons.
Do not confuse short-term success against one opponent with general playing strength.

Use [random mirror matches](team-profiles.md#random-mirror-matches) with
`--mirror-teams` to give both players the same randomly selected team each battle.
