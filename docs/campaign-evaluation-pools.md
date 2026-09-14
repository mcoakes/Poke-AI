# Campaign evaluation pools

[Battle-count campaigns](battle-count-campaigns.md) · [Runner flags](reference/run-curriculum.md)

Use a mix of opponents to decide whether a candidate advances to the next stage.
Training continues against the stage's frozen opponent. The pool is used only for
evaluation; none of its models learn during a test.

## Enable the pool on an existing campaign

Stop the campaign with Ctrl+C before rerunning it. For a campaign created with
`--train-battles 1000 --eval-episodes 1000`:

```bash
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10 \
  --eval-opponents mixed \
  --promotion-basic-win-rate 0.75 \
  --promotion-smart-win-rate 0.60 \
  --promotion-require both
```

Omit `--resume` and `--opponent-checkpoint`: the campaign already has its starting
models and continues from its latest attempt. The new evaluation settings persist,
so later commands can omit them too.

Each 1,000-battle evaluation contains:

| Opponent | Battles | Selection |
|---|---:|---|
| Random policy | 300 | Uniformly samples legal actions, including legal switches |
| Dumb model | 300 | Campaign's original frozen reference |
| Smart model | 400 | Uniformly samples from the starting Smart model and previously promoted Smart models |

The order is shuffled, with exact 30%/30%/40% shares. Every matchup is played with
the candidate in both player seats against the same opponent, teams and battle
seed. Smart model selection happens per two-battle matchup. Duplicate Smart
checkpoint files with identical contents count once in the sampling pool.

Teams come from the current stage's cumulative team pool. Each new training
attempt gets a new evaluation schedule and snapshots the available opponent
models. A retry after an interrupted evaluation replays its saved schedule and
model copies; it does not retrain a candidate whose training already completed.

## Choose the promotion rule

**Basic** is the combined Random + Dumb result. With 1,000 evaluations, the
example thresholds require:

- **Basic:** at least 450 wins out of 600 battles (75%).
- **Smart:** at least 240 wins out of 400 battles (60%).

`--promotion-require both` requires both thresholds. Change it to `either` to
promote when at least one passes. Equality passes. Draws and truncated battles
are non-wins; an incomplete or failed evaluation cannot promote a candidate.
The overall win rate is reported, but it is not a separate promotion gate.

Random and Dumb win rates are also shown separately, so a strong Random result
does not obscure the Dumb result in the report. The Basic gate itself uses their
combined score.

Change a gate while continuing the same campaign:

```bash
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10 \
  --promotion-smart-win-rate 0.65 \
  --promotion-require either
```

Changes apply to the next evaluation decision. Completed attempts and promoted
stages retain their original results. After a completed `HOLD`, a rerun trains
the next attempt before evaluating again; it does not regrade the old attempt.

## Choose the saved opponents

By default, the campaign retains its initial Dumb reference across all stages,
even when `--opponent-mode previous-stage` changes the training opponent. To use
a different Dumb checkpoint for future evaluations:

```bash
fronzen_checkpoint="models/dumb.zip"
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10 \
  --eval-dumb-checkpoint "$fronzen_checkpoint"
```

`--eval-dumb-checkpoint latest` is also supported. It resolves and copies the file
once; later training cannot update that frozen copy. This flag changes the
evaluation reference, not the training opponent.

The Smart pool includes `smart_v1` and every earlier promoted campaign model.
To add named models saved outside this campaign, use `--eval-smart-dir models`.
It scans that directory's direct `smart_v*.zip` files at each new evaluation and
copies them into the evaluation plan. It does not automatically include held
attempts or a candidate that is still being trained. Keep external Smart files
as saved exports; new files are picked up on the next attempt.

## Start a new campaign with mixed evaluations

After [preserving your named models](battle-count-campaigns.md#preserve-the-starting-models):

```bash
curriculum_run=".local/run/curriculum-mixed-opponents-01"
fronzen_checkpoint="models/dumb.zip"
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --resume models/smart_v1.zip \
  --opponent-checkpoint "$fronzen_checkpoint" \
  --opponent-mode previous-stage \
  --train-battles 1000 --eval-episodes 1000 \
  --eval-opponents mixed \
  --promotion-basic-win-rate 0.75 \
  --promotion-smart-win-rate 0.60 \
  --promotion-require both \
  --through-stage 10 --attempts-per-stage 10 \
  --environments 48 --showdown-workers process \
  --rollout-steps 128 --batch-size 128
```

Mixed evaluations require a battle-budget campaign and an evaluation count
divisible by 20, which keeps both the proportions and player seats balanced.
The original four-evaluation curriculum mode retains its pinned regression rules;
use a new run directory to switch that mode to a battle-budget campaign.

## Read results and replay a test

```bash
cat "$curriculum_run/evaluations.md"
cat "$curriculum_run/rule-changes.jsonl"
```

The table includes Random, Dumb, Basic and Smart win rates, the rule revision,
and `HOLD`/`PROMOTE` for every attempt. `evaluations.csv` also includes wins and
attempt counts for each category. Each attempt's `report.md` lists its actual
thresholds; old reports are preserved when the rules change.

Within `stage-NN/attempt-NNN/`:

- `evaluation-pool/suite.json`: teams, battle seeds, opponent assignments and model hashes.
- `evaluation-pool/models/`: frozen copies of all pool checkpoints.
- `eval-current-NNN/summary.json`: overall, per-category and per-model results.
- `eval-current-NNN/episodes.jsonl`: individual results with opponent IDs and categories.

To replay a saved pool against a different candidate, use a new output directory:

```bash
python mgba/evaluate_ppo.py \
  --checkpoint latest \
  --suite "$curriculum_run/stage-01/attempt-001/evaluation-pool/suite.json" \
  --environments 48 --showdown-workers process \
  --output-dir .local/run/pool-replay-01
```

The suite already specifies opponents; omit `--opponent-checkpoint`. Random
policy choices are seeded per battle and replay independently of worker count.
Repeated campaign checks are validation: use a separate held-out evaluation for
an independent measurement of the final agent.
