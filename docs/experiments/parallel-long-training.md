# Longer shared-moves training across four seeds

After the modest gains in the long and strategy batches, use the
[fixed-team learnability lab](fixed-team-lab.md) before extending broad training.

To compare different strategies in parallel, see
[parallel strategy experiments](parallel-strategies.md). The four-seed defaults
on this page remain supported, including existing batches.

Continue four copies of the verified 400k shared-moves reference, each with
1,000,000 additional transitions. This tests longer learning and variation
between continuation seeds, not training from independent random initial weights.
All use the same teams, rules opponent, rewards, learning rate 0.0001, entropy
0.01, 128 rollout steps, 128 batch size and four epochs. Do not train on the
new-team evaluation pools. Retain the diversity and 400k models as references.

## Start from the project root

Activate the usual training venv. The default reference is
`models/observation-v2-shared-moves-400k-reference-01.zip`; use `--checkpoint`
with the exact evaluated candidate path if the archived copy is not present.

```bash
python run_parallel_training.py --jobs 2 --dry-run
python run_parallel_training.py --jobs 2
```

Default output: `.local/run/shared-long-seeds-01`. The four seed directories
are `seed-261020` through `seed-261023`. Each run has 64 player environments
(32 learning players against Rules), process workers, CPU inference and one
PyTorch thread. Validation uses 16 player environments per run and runs every
200,000 additional transitions, rounded to a PPO update, plus initial/final
validation. Each run saves best and final checkpoints. These are local
Showdown subprocesses using pipes; the numeric port labels do not bind network
ports in this backend. Do not use this launcher for mGBA.

Use `--jobs 2` to queue the same four experiments two at a time. More simultaneous
runs can increase aggregate throughput while slowing each individual run. A
Linux five-minute load average is not CPU utilization. Monitor `vmstat 1`,
`top`/`htop`, available RAM and aggregate training progress before assuming four
is faster than two. The launcher keeps environment counts fixed when concurrency
changes, so PPO rollout batch sizes stay fixed. Do not increase environment
counts and change hyperparameters simultaneously to chase utilization.

Console logs rotate at 8 MiB with one backup per run; battle logs retain the
existing 2 MiB/one-backup limits per worker. Runs have independent directories
and locks. The initial model and rewards are copied into batch inputs. Training
teams and the suite must remain available; the trainer pins validation inputs
inside each launched run. Keep dependencies and input files unchanged during
the batch.

## Monitor and pause

Always include these monitoring commands when providing a parallel-training
start or resume command. Use separate terminals so stopping a viewer does not
interrupt training. Follow console logs (including rotation):

```bash
tail -n 20 -F .local/run/shared-long-seeds-01/console/*.log
```

Only active jobs advance; queued jobs may not have a log yet. Restart the viewer
when queued jobs begin so the shell glob includes their new files.

Record memory throughout training and validation:

```bash
vmstat -w -t 2 | tee /tmp/poke-ai-two-jobs-memory.log
```

Start with two jobs after the observed four-job OOM. Measure validation peaks
before considering three. The CLI default remains four; specify `--jobs 2`.

```bash
watch -n 10 'for f in .local/run/shared-long-seeds-01/seed-*/evaluations.md; do
  [ -f "$f" ] || continue
  echo "$f"
  cat "$f"
done'
```

Ctrl+C in the watch terminal stops only the viewer. Ctrl+C once in the launcher
terminal requests graceful stops of all active trainers and cancels queued
starts. Wait for the launcher to exit. Work since the last saved checkpoint is
lost, as with standalone training. If a run stops before any checkpoint was
saved, inspect its diagnostics and start a new batch directory; it cannot be
continued from saved progress.

```bash
python run_parallel_training.py --continue-batch --jobs 2
```

This restores the recorded batch plan, skips seeds whose saved progress already
reached the target and requests only each unfinished seed's remaining budget.
It does NOT add another million transitions. Only concurrency changes on this
command; the original plan's other parameters are restored. As with standalone
continuation, battles restart and resumed random streams are not bit-for-bit
identical to an uninterrupted run. A second launcher for the same batch is
rejected. Direct trainer locks still protect individual runs.

## Collect and interpret

At completion or a graceful stop, the launcher writes `batch-report.md` and
`batch-results.zip` in the batch directory, containing validation histories,
configs, summaries and available best-checkpoint metadata, but no model weights.
Upload `batch-results.zip` for comparison. Review all four learning curves and
best AND final performance; selecting only the highest peak exaggerates gains.
Do not keep running unchanged solely to consume CPU. If all runs plateau or
regress across several validations, investigate representation or training data.
If improvements persist across seeds, compare candidates on held-out team pools
before promotion. Previously inspected team pools are development evidence now.

## Preceding results

Against Rules on two separately generated 300-team pools (4,000 battles each):

| Pool | 400k reference | Control best | Change |
|---|---:|---:|---:|
| 1 | 1,461/4,000 (36.525%) | 1,515/4,000 (37.875%) | +1.350 points |
| 2 | 1,448/4,000 (36.200%) | 1,495/4,000 (37.375%) | +1.175 points |
| Combined | 2,909/8,000 (36.3625%) | 3,010/8,000 (37.6250%) | +1.2625 points |

No truncations. A 20,000-resample bootstrap pairing both seats and stratifying
by pool gives an approximate combined 95% interval +0.36 to +2.19 points.
This conditions on the sampled team pools and two fixed checkpoints; it does
not measure training-seed uncertainty, nor certify no overlap with training.
The higher-entropy branch gave no advantage on the prior separate schedule.
These results motivate longer default-entropy runs with multiple seeds.

## Seed handling fix and validation scope

This update makes an explicit `--seed` on a new `--resume` branch reseed the
policy as well as battle sampling. Previously that branch could retain the
checkpoint's policy seed. Omitting an explicit seed preserves the former
checkpoint-seed behavior; same-directory continuation keeps its existing reseed
behavior. Model weights and optimizer parameters are not changed by reseeding.

Five standard-library regression tests exercise seed selection, dry-run
behavior, bounded concurrency, report packaging, refusing accidental reuse,
completed-run skipping, highest-counter recovery, and Ctrl+C/remaining-budget
continuation. Launcher integration uses subprocess fixtures, not real training.
A full PPO training run was not performed in this workspace.

For fixed-to-varied, gated two-job experiments with live logs, memory sampling, and
collapse diagnostics, see [Staged team lab](staged-team-lab.md).

For the expanded fixed-to-1,024-team progression, editable teams, rehearsal, and
team inspection commands, see [Expanding team curriculum](team-curriculum.md).

For a stage-4 transfer failure after successful mirror retention, see
[type-transfer-lab.md](type-transfer-lab.md) for the two-job specialist-versus-fresh
comparison, exact team inspection, live logs, memory monitoring, and continuation.
