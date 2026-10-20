# Compare training strategies in parallel

Current diagnostic direction: [fixed-team learnability lab](fixed-team-lab.md).
Use its calibrated action-choice and mirror tests before more small changes here.

[Long control batch](parallel-long-training.md) · [Opponent pools](../opponent-pools.md)

Use multiple seeds to measure variation within a setup; use different settings
to test interventions. Both are useful. A practical four-slot comparison is two
strategies with two seeds each. Two seeds provide an initial consistency check,
not enough precision for a broad claim about an algorithm.

The existing long batch is a control: four seeds of the original setup, all
starting from the same 400k shared-moves checkpoint. It estimates variation among
continuations of those weights, not independent from-scratch training. Retain
these runs. The new batch adds two interventions, each compared against control
seeds 261020 and 261021:

| Strategy | Seeds | Change from control | Hypothesis |
|---|---|---|---|
| Existing control | 261020, 261021 (plus 261022, 261023) | None | More learning at the current settings may improve performance. |
| Lower learning rate | 261020, 261021 | 0.0001 to 0.00005 | Smaller updates may retain useful behavior after earlier peaks. |
| Mixed opponents | 261020, 261021 | Rules only to 85% Rules / 15% frozen 400k reference | Exposure to a learned policy may reduce reliance on one opponent style. |

The percentages are episode-sampling probabilities, not guaranteed transition
shares. The learned opponent is frozen; this is not evolving self-play.
Reward settings, training teams, checkpoint, rollout, batch size, epochs, entropy,
step budgets and evaluation suite remain identical. Starting weights are the
preserved 400k reference for all branches; do not initialize alternatives from
a later control checkpoint when claiming this comparison. Matching seeds help
organize comparisons but do not force identical trajectories after choices diverge.

## Run the strategy batch

Use the training venv, from the project root. Let the existing batch finish first
so the requested concurrency remains two total. You can prepare and dry-run
this configuration while it runs. It does not modify the existing batch.

```bash
python run_parallel_training.py \
  --experiments config/experiments/shared-strategies-85.json \
  --output-dir .local/run/shared-strategies-85-01 \
  --jobs 2 --dry-run
```

After the control batch finishes:

```bash
python run_parallel_training.py \
  --experiments config/experiments/shared-strategies-85.json \
  --output-dir .local/run/shared-strategies-85-01 \
  --jobs 2
```

Each run requests one million additional transitions and validates every 200,000
by default. Use the same command-line budgets as the control if it used overrides.
The dry-run prints the effective hyperparameters and opponent weights for each
run, followed by its actual trainer command. --jobs limits concurrent experiments;
--environments still means the player environments inside each training process.

Always include console monitoring alongside every parallel-training start or
resume command. Run viewers in separate terminals. Queued jobs create logs when
they start; restart `tail` then to include newly created files in its glob.

```bash
vmstat -w -t 2 | tee /tmp/poke-ai-two-jobs-memory.log
```

Use two concurrent jobs and watch memory through validation before considering
three. Four simultaneous jobs previously exhausted memory on this machine.
The CLI still defaults to four, so specify `--jobs 2` explicitly.

```bash
tail -n 20 -F .local/run/shared-strategies-85-01/console/*.log
```

```bash
watch -n 10 'for f in .local/run/shared-strategies-85-01/*/evaluations.md; do
  [ -f "$f" ] || continue
  echo "$f"
  cat "$f"
done'
```

Ctrl+C in either viewer only stops that viewer. Ctrl+C once in the launcher
requests a graceful stop of active trainers and cancels queued starts. Wait for
cleanup; progress since each last saved checkpoint is lost. To finish remaining
budgets, restore the saved plan without supplying the experiment file again:

```bash
python run_parallel_training.py \
  --output-dir .local/run/shared-strategies-85-01 \
  --continue-batch --jobs 2
```

An older four-seed batch remains resumable with its original command. Existing
batch directories are never silently converted into different experiments.

## Explicit experiment file

See `config/experiments/shared-strategies-85.json`. Top-level fields are `version: 1`
and a nonempty `experiments` list. Each strategy has a unique run-name/seed pair.

| Field | Default / meaning |
|---|---|
| `name` | Required lowercase letters, digits and hyphens; used in directory names. |
| `seeds` | Required list of seeds; each creates a separate run. |
| `learning_rate` | 0.0001; positive finite number. |
| `ent_coef` | 0.01; finite nonnegative number. |
| `rules_weight` | 1.0; nonnegative relative episode weight. |
| `frozen_reference_weight` | 0.0; nonnegative relative episode weight for the starting model. |

Opponent weights must sum to a positive number. The mixed branch sets 0.85 and
0.15. To include a new control in a standalone comparison, add a strategy with
only `name` and `seeds`. To screen four different ideas with one seed each, use
four entries, then repeat promising variants on additional seeds. Unknown fields
and duplicate run names are rejected before any output is created; do not put
arbitrary train_ppo flags into this file.

Common settings (teams, rewards, architecture, environments, rollout settings,
evaluation suite, budgets) still come from the launcher, its flags and the saved
batch plan. With --experiments, the JSON seeds apply rather than --seeds.
Batch-plan JSON expands the strategies into every effective per-run setting.
Opponent-pool JSON files live alongside the batch's copied reference model and
refer to that copy. Trainers pin those inputs inside their individual runs too.
Continuation restores saved settings and validates the reference, reward and pool
snapshots; editing the original experiment file does not alter an existing batch.

## Compare the results

Upload both `shared-long-seeds-01/batch-results.zip` and
`shared-strategies-85-01/batch-results.zip` from `.local/run/` when available. Each
contains a table naming the exact configurations and per-run validation curves.
Compare matching budgets, the two seeds together, Rules and other opponent
categories, and best plus final checkpoints. Don't pick a strategy based only
on the highest single peak. Repeat the strongest candidate on further seeds and
fresh matched evaluation cases before treating it as a reliable improvement.

This addition changes launcher configuration and reporting, not the PPO trainer
or rewards. Eight standard-library tests cover old-batch recovery, process
orchestration, flags and pinned mixed pools, invalid-input rejection, reporting,
and earlier seed handling. The process tests use a fixture trainer; no full PPO
training job was run here.

For fixed-to-varied, gated two-job experiments with live logs, memory sampling, and
collapse diagnostics, see [Staged team lab](staged-team-lab.md).

For the expanded fixed-to-1,024-team progression, editable teams, rehearsal, and
team inspection commands, see [Expanding team curriculum](team-curriculum.md).

For a stage-4 transfer failure after successful mirror retention, see
[type-transfer-lab.md](type-transfer-lab.md) for the two-job specialist-versus-fresh
comparison, exact team inspection, live logs, memory monitoring, and continuation.
