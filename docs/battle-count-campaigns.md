# Campaigns measured in battles

[Curriculum guide](curriculum.md) · [Mixed evaluation opponents](campaign-evaluation-pools.md) · [All runner flags](reference/run-curriculum.md)

This mode trains toward a battle budget, evaluates once on the current stage's
team pool, and promotes immediately when at least 75% of evaluation attempts are
wins. It does not run the baseline or expanded-team evaluations used by the
original curriculum mode. Each stage still adds one team to the cumulative pool.

## Preserve the starting models

From the project root, copy these checkpoints once. `cp -n` preserves an existing
named copy; choose a different destination if you intend to replace one.

```bash
mkdir -p models
cp -n .local/run/ppo-20260913-184241-113975/checkpoints/latest.zip models/smart_v1.zip
cp -n .local/run/ppo-20260913-151530-703645/checkpoints/latest.zip models/dumb.zip
fronzen_checkpoint="models/dumb.zip"
```

These paths identify the models you selected; the campaign does not independently
verify the earlier 90% evaluation. It snapshots both files at initialization.

## Start the ten-stage campaign

Use a new run directory:

```bash
curriculum_run=".local/run/curriculum-smart-generations-01"
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --resume models/smart_v1.zip \
  --opponent-checkpoint "$fronzen_checkpoint" \
  --opponent-mode previous-stage \
  --train-battles 1000 --eval-episodes 1000 \
  --promotion-win-rate 0.75 \
  --through-stage 10 --attempts-per-stage 10 \
  --environments 48 --showdown-workers process \
  --rollout-steps 128 --batch-size 128
```

Each attempt trains toward **1,000 ended battles across the whole pool**, including
truncations. PPO finishes the update that crosses this threshold, so training can
exceed 1,000 battles; no partial rollout is discarded. The training summary records
the actual `finished_attempts`, completed and truncated counts. This budget replaces
the timestep budget. Evaluation runs **exactly 1,000 attempts**, with candidate
sides balanced and fixed matchups/seeds within the stage. At least 750 wins passes;
draws and truncations count as non-wins.

On a failed evaluation, the next attempt continues from the candidate just trained
against the same frozen opponent. This command allows up to ten attempts per stage
in this invocation, stopping earlier as soon as the stage passes. On promotion it
advances immediately; after ten unsuccessful attempts it returns `HOLD`. Errors
stop the campaign and never count as a passing evaluation.

| Stage | Starting learner | Frozen opponent | Promoted export |
|---:|---|---|---|
| 1 | Smart v1 | Dumb | Smart v2 |
| 2 | Smart v2 | Smart v2 | Smart v3 |
| 3 | Smart v3 | Smart v3 | Smart v4 |
| 10 | Smart v10 | Smart v10 | Smart v11 |

Exports live under the campaign's `models/`, including preserved `smart_v1.zip`
and `dumb.zip`. Only promoted candidates receive the next Smart version; unsuccessful
attempts retain their ordinary training checkpoints. The learner updates while its
opponent remains frozen throughout a stage.

## Read the complete track record

After every completed evaluation, these files include all stage attempts:

```bash
cat "$curriculum_run/evaluations.md"
```

`evaluations.md` is a compact table of stage, attempt, candidate version, opponent,
wins/evaluations, win rate and result. `evaluations.csv` provides the same information
plus the checkpoint path for analysis. Held attempts remain visible. Both are rebuilt
from saved reports, so resuming does not duplicate rows. A candidate version in a
held row describes the version being trained, not a promoted export.

Each stage also retains `report.md`, and each attempt retains its full evaluation
summary and per-battle records. Training reward means and battles/hour remain enabled.

## Continue after HOLD or interruption

```bash
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10
```

Omit the starting models and budgets when resuming; they are pinned. Completed
training is reused if evaluation was interrupted. A training
interruption resumes its latest completed PPO update in a new attempt with a fresh
battle budget. Use a separate held-out evaluation suite after the campaign: repeated
checks on the stage suite do not establish generalization.

## Change the promotion threshold

Stop an active campaign before rerunning it with a different threshold:

```bash
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10 \
  --promotion-win-rate 0.70
```

The new gate persists for future evaluations. Previously completed attempts and
promotions keep their original results; the change is recorded in
`rule-changes.jsonl`. After a completed `HOLD`, the next attempt trains before
evaluating again.

For a shuffled **30% Random / 30% Dumb / 40% Smart** evaluation pool with separate
promotion gates, follow [Campaign evaluation pools](campaign-evaluation-pools.md).
That mode uses `--promotion-basic-win-rate` and `--promotion-smart-win-rate`
instead of the single `--promotion-win-rate` flag.
