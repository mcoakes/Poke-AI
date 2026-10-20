# Establish a training baseline

[Evaluation](evaluation.md) · [Campaigns](curriculum.md) · [Stage 2 audit](training-audit.md) · [All flags](reference/establish-baseline.md)

Before changing training, compare a campaign's starting, best retained and latest
models on the same teams, seeds and opponents. This is Stage 1 of the training
improvement workflow. It measures what learning achieved without training new
weights, changing rewards or promoting a campaign stage.

## Run the comparison

Use the project root and the existing [Showdown training dependencies](setup/showdown.md).
Stop the source campaign while the baseline copies its inputs; the snapshot uses
the campaign lock to avoid racing a checkpoint save or cleanup. The original
campaign is preserved. The new baseline directory must be outside the source
campaign directory.

```bash
curriculum_run=".local/run/curriculum-generalization-01"
baseline_run=".local/run/baseline-generalization-01"

python mgba/establish_baseline.py \
  --campaign-run "$curriculum_run" --stage 1 \
  --run-dir "$baseline_run" \
  --episodes-per-opponent 1000 \
  --environments 48 --showdown-workers process \
  --dry-run
```

Review the selected checkpoints, team splits and opponent sources, then run the
same command without `--dry-run`:

```bash
python mgba/establish_baseline.py \
  --campaign-run "$curriculum_run" --stage 1 \
  --run-dir "$baseline_run" \
  --episodes-per-opponent 1000 \
  --environments 48 --showdown-workers process
```

The default budget is **18,000 battles**: three candidate roles × two team splits
× three opponent categories × 1,000 battles. The count includes both player-side
assignments. Candidates with identical checkpoint bytes share their evaluation
results, reducing the actual work. The Smart category's 1,000 battles are shared
across its saved opponent pool; it is not 1,000 battles per Smart checkpoint.

All candidates play the same newly generated schedule within each split and
opponent category. Each matchup is played from both seats using the same teams,
seed and reference opponent. The models choose legal actions deterministically;
Random uses seeded legal-action sampling.

## What is selected

| Role or input | Default selection |
|---|---|
| Starting | The selected campaign stage's starting checkpoint. For stage 1, this is the campaign's original learner. |
| Best | Highest evaluated score among retained candidates in the latest completed evaluation-rule cohort. |
| Latest | Latest recoverable saved candidate in the selected stage, including an interrupted attempt when available. |
| Training teams | Cumulative training team pool through the selected stage. |
| Validation teams | The stage's cumulative saved validation pool. |
| Random | Uniformly sampled legal actions. |
| Dumb | Saved Dumb evaluation reference; falls back to the initial campaign reference with a warning if no separate Dumb reference exists. |
| Smart | Frozen Smart pool saved for the latest completed campaign evaluation; falls back to the campaign's original learner if no Smart pool exists. |

“Best” refers to available candidate weights, not necessarily the best model ever
produced. Cleanup cannot recover a checkpoint whose weights were deleted. The
historical score used to choose Best is also subject to selection bias; the new
shared evaluation is the comparison that matters here.

Use checkpoint override flags or `--validation-dir` when automatic choices do not
match the experiment you intend. These are initial snapshot choices; use a new
baseline run directory when changing them. All options are documented in the
[CLI reference](reference/establish-baseline.md).

Campaigns without separate validation groups must supply `--validation-dir`
containing distinct team JSONs. The directory is scanned recursively. Reordered
copies of a training party do not qualify as separate validation teams.

## Read the report

```bash
cat "$baseline_run/status.json"
cat "$baseline_run/baseline-report.md"
```

The report separates training and validation results, and separates Random,
Dumb and Smart opponents. Win rate is wins divided by all attempted battles;
draws and truncations count as non-wins. It includes draws, truncations, rewards,
decision counts and comparisons against Starting. `baseline-report.json` keeps
the corresponding machine-readable results.

For Best and Latest, each split/opponent comparison is classified as:

- **Improvement:** the entire estimated 95% interval for the win-rate difference is above zero.
- **Regression:** the entire interval is below zero.
- **No clear change:** the interval includes zero.
- **Insufficient data:** fewer than 20 matched two-seat pairs are available in that comparison.

The calculation resamples matched battles while keeping each two-seat matchup
together and comparing candidates on those same matchups. That accounts for the
shared schedule instead of treating every outcome as an independent new test.
These are pointwise intervals for individual comparisons, not simultaneous
guarantees across the whole report or automatic promotion thresholds.
Identical observed outcomes can give a zero-width interval; that is not proof of
certain performance on future battles.

A model that improves only on training teams may be specializing to familiar
teams. A latest model that loses ground relative to Starting while Best improves
suggests that additional training did not preserve the earlier gain. A report
with no clear changes calls for the [observation/action audit in Stage 2](training-audit.md)
before another long campaign. Keep this baseline and its frozen snapshots, and
write the audit to a separate new directory. Stage 1 has no arbitrary target win rate: its purpose is
to establish which of these situations the evidence supports.

Validation teams already used for campaign promotion are development data, not
a final untouched test set. New battle seeds do not turn familiar team templates
into unseen teams. Keep a separate final test pool for the end of development.

## Resume and keep useful logs

The baseline snapshots model weights, rewards and matchup suites into its own
run directory. After this snapshot is created, it no longer depends on ongoing
changes in the source campaign. Continue an interrupted run with:

```bash
python mgba/establish_baseline.py --run-dir "$baseline_run"
```

Completed evaluations are reused. Interrupted evaluation retries use a new
`runs/<candidate>/<split>/attempt-NNN/` directory with the original saved schedule.
Execution settings such as worker count can change without changing the test:

```bash
python mgba/establish_baseline.py --run-dir "$baseline_run" \
  --environments 24 --showdown-workers process
```

Resume verifies simulator/policy code, dependency versions and inference settings.
Use a new baseline directory if those change; worker count and logging limits can
change while reusing the saved test.

Readable and spectator logs remain enabled, bounded to **2 MiB per file with one
rotated backup**. Full per-decision traces are off. This retains some useful
battle context without recreating an unbounded archive. Change the log limits
with the documented flags; they apply per file and pair, not as one total quota.
See [storage and retention](storage.md) for campaign cleanup and longer runs.
