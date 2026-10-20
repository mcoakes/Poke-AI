# Periodic evaluation and best checkpoints

To retain existing Random/Dumb/Smart cases and add rules, use the
[combined validation walkthrough](combined-validation.md). Training opponents
and validation opponents are selected independently.


[Training](ppo-training.md) · [Evaluation](evaluation.md) · [Checkpoint selection](checkpoints.md) · [All training flags](reference/train-ppo.md)

Use a fixed validation suite to measure learning during Showdown training. The
trainer evaluates the current model first, then at intervals, and at successful
completion if that checkpoint has not already been evaluated. Each evaluation
uses the suite's same teams, opponents, seeds, player sides and decision limit.

## Continue a run with validation

The suite saved by an earlier evaluation can be reused directly. This example
continues the observation-v2 run and its existing frozen training opponent:

```bash
python mgba/train_ppo.py \
  --output-dir .local/run/ppo-observation-v2-01 \
  --if-exists continue \
  --environments 64 --showdown-workers process \
  --total-timesteps 500000 \
  --eval-suite .local/run/eval-observation-v2-03/suite.json \
  --eval-every 100000 --eval-environments 48 \
  --keep-evaluations 2 \
  --no-readable-log --battle-log --no-step-logs \
  --log-max-mb 2 --log-backups 1
```

The budget adds 500,000 learning-player transitions, rounded to a complete PPO
rollout. An evaluation interval is also rounded up to an update: for example,
with 4,096 transitions per update, 100,000 means 102,400 transitions. The next
interval starts at the last successfully evaluated checkpoint. A final evaluation
can occur sooner than a full interval. Evaluation does not consume training
timesteps or add its battles to training rewards and win rates.

Training pauses while an independent evaluator process runs. Its policy loads,
random seeds and simulator workers do not change the training policy, optimizer,
frozen opponent or live battles. An error or interruption stops the invocation
with the latest complete update saved; failed or partial evaluations never
replace the best checkpoint.

The suite and its opponent weights are copied into `inputs/evaluation/`. Copies
are verified by SHA-256; the suite itself remains unchanged, so its hash still
matches the original evaluation. A suite without an opponent pool uses the
training run's frozen opponent, copied separately. Self-play training requires
a suite containing an explicit opponent pool. Periodic validation currently
supports Showdown; standalone `evaluate_ppo.py` still supports both backends.

## How the best model is selected

`checkpoints/best.zip` holds the candidate with the highest number of wins on
the fixed suite. Its selection rate is **wins / all requested battles**; draws
and truncations count as non-wins. Only a finished evaluation covering every
scheduled battle is eligible. Equal scores retain the earlier model. The usual
standalone summary's `candidate_win_rate` excludes truncations; when there are
no truncations the two rates agree.

`checkpoints/best.json` records its evaluation, timestep count, selection rate,
suite hash and exact checkpoint hash. The first successful validation establishes
the baseline best. Earlier reports cannot restore weights that were already
overwritten; they are not imported as a best model without the model itself.

| File | Purpose |
|---|---|
| `checkpoints/latest.zip` | Resume recent training progress; also saved before each validation. |
| `checkpoints/final.zip` | Model at the end of a successful training invocation. |
| `checkpoints/best.zip` | Strongest validated candidate; excluded from numbered-checkpoint rotation. |
| `evaluations.md` | Compact table of every validation, including Random/Dumb/Smart rates when present. |
| `evaluations.csv` | The same history with counts, rewards, timing and checkpoint hashes. |
| `evaluation-history.json` | Authoritative validation history, retained across continuations. |
| `evaluations/evaluation-NNNNNN/` | Exact candidate snapshot, suite, results and bounded spectator logs. |

Watch results with:

```bash
watch -n 5 cat .local/run/ppo-observation-v2-01/evaluations.md
```

These tables are replaced atomically, so use `watch` rather than `tail -f`.
Console validation messages identify new bests and their timestep counts.
Ctrl+C in this watch terminal stops only the viewer. To pause training, press
Ctrl+C once in the training terminal and wait for cleanup. Only previously
saved progress is recoverable; see [pause and continue](quickstart.md#monitor-pause-and-continue-training).

## Continue, retain, and compare

Subsequent continuations restore the validation settings automatically:

```bash
python mgba/train_ppo.py \
  --output-dir .local/run/ppo-observation-v2-01 \
  --if-exists continue --total-timesteps 100000
```

You can change the interval, validation worker count and retention limit.
`--eval-every 0` pauses validation while preserving its best model and history;
set a positive interval to reenable it. Changing the suite or reference requires
a new output directory so scores from different tasks are not ranked together.

By default, retain the latest two validation directories **plus the best one**
if it is older. The compact history and `best.zip` remain. This limit applies
across continued sessions. Validation spectator logs follow the training
`--battle-log`, size and backup settings; detailed observation/step logs are off.
The evaluator's console log is capped at 2 MiB plus one backup. Standalone
evaluations outside this run have their own directories and are not pruned here.

Reported training battles/hour includes validation pauses, but only training
battles count in its numerator. Final `summary.json` reports validation wall
time for the session separately. Keep the evaluation schedule fixed when
comparing throughput.

Repeated selection on one validation suite can overfit that suite. Use a separate
held-out suite for a final performance claim; keep this suite for model selection.
The [new-team comparison](generalization-evaluation.md) checks overlap and runs
both best checkpoints on one separate fixed schedule.

## Standalone evaluation snapshots

Every `evaluate_ppo.py` run now writes `candidate.zip`: a byte-for-byte copy of
the candidate loaded before evaluation. Its path is recorded as
`models.candidate.snapshot_path` in `config.json`; the original path and SHA-256
remain recorded too. Updating the original `latest.zip` or `final.zip` cannot
change these saved candidate weights. This applies to both Showdown and mGBA.
