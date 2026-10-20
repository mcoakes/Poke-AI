# Validate against rules and learned opponents

[Opponent pools](opponent-pools.md) · [Periodic validation](periodic-evaluation.md) ·
[Utility flags](reference/combine-eval-suites.md) · [Experiment record](experiments/README.md)

Training and validation select opponents independently. `--rule-based-opponent`
chooses who supplies training experience; `--eval-suite` chooses who tests the
checkpoints. A dash in the Rules column means that suite did not test rules.

## Combine the existing schedules

From the project root, with the training environment activated:

```bash
diversity_run=".local/run/diversity-v2-01"
combined_suite=".local/run/validation-rules-and-agents-01/suite.json"

python mgba/combine_eval_suites.py \
  --suites "$diversity_run/suites/validation.json" \
           .local/run/eval-rules-v1-01/suite.json \
  --output-dir .local/run/validation-rules-and-agents-01
```

For these experiment inputs, the result is 4,000 battles:

| Category | Battles | Share of overall score |
|---|---:|---:|
| Random | 1,000 | 25% |
| Dumb checkpoint | 1,000 | 25% |
| Smart checkpoint | 1,000 | 25% |
| Rules | 1,000 | 25% |

The helper prints actual counts; it does not assume source sizes. Each matchup
retains both candidate seats, teams, battle seed and opponent. Sources remain
unchanged. Checkpoint copies are hash-verified and bundled beside the new suite;
the current rule implementation must match the saved identity. No battles or
training run during preparation. Existing output directories are rejected.

The helper concatenates schedules rather than generating new cases. Source
case ranges and hashes are recorded in `manifest.json`. Categories can use
different team matchups inherited from their sources; this is a fixed combined
benchmark, not a controlled comparison of opponent strength on identical teams.
These are development validation cases already used in experiments, not a new
reserved test. Keep reserved-test inputs separate from checkpoint selection.

## Compare the models already trained

Archive the useful reference and diagnostic endpoints without overwriting any
existing named models:

```bash
mkdir -p models
cp -n .local/run/diversity-v2-01/train/checkpoints/best.zip \
  models/observation-v2-diversity-best-01.zip
cp -n .local/run/ppo-mixed-opponents-01/checkpoints/final.zip \
  models/observation-v2-mixed-final-01.zip
cp -n .local/run/ppo-rules-only-01/checkpoints/final.zip \
  models/observation-v2-rules-only-final-01.zip
```

Evaluate the original run checkpoints explicitly, so an existing archive name
cannot accidentally select different weights:

```bash
combined_suite=".local/run/validation-rules-and-agents-01/suite.json"
reward_config=".local/run/diversity-v2-01/inputs/reward.json"

for label in diversity mixed rules-only; do
  case "$label" in
    diversity) checkpoint=".local/run/diversity-v2-01/train/checkpoints/best.zip" ;;
    mixed) checkpoint=".local/run/ppo-mixed-opponents-01/checkpoints/final.zip" ;;
    rules-only) checkpoint=".local/run/ppo-rules-only-01/checkpoints/final.zip" ;;
  esac
  python mgba/evaluate_ppo.py \
    --checkpoint "$checkpoint" --suite "$combined_suite" \
    --reward-config "$reward_config" \
    --environments 48 --showdown-workers process \
    --rule-log --battle-log --no-readable-log --no-step-logs \
    --log-max-mb 2 --log-backups 1 \
    --output-dir ".local/run/eval-combined-${label}-01" || break
done
```

Use new evaluation output names on a rerun. The same suite is reusable. Inspect
`by_opponent_group` in each `summary.json`; `episodes.jsonl` retains individual
outcomes for paired comparisons. Summaries alone do not establish uncertainty
for changes between models. The evaluator also records candidate hashes and
preserves exact loaded candidate bytes.

## Use it in the next training experiment

Compare the existing models first. Adding the suite is a measurement change,
not evidence that another training budget will improve performance. If repeating
the rules-only experiment, this is a complete example using a new run directory:

```bash
python mgba/train_ppo.py \
  --backend showdown \
  --resume .local/run/diversity-v2-01/train/checkpoints/best.zip \
  --rule-based-opponent \
  --team-dir .local/run/diversity-v2-01/teams/training \
  --reward-config .local/run/diversity-v2-01/inputs/reward.json \
  --learning-rate 0.0001 --rollout-steps 128 --batch-size 128 \
  --environments 64 --showdown-workers process \
  --total-timesteps 100000 \
  --eval-suite .local/run/validation-rules-and-agents-01/suite.json \
  --eval-every 50000 \
  --rule-log --battle-log --no-readable-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --checkpoint-every 25000 --keep-checkpoints 2 --keep-evaluations 2 \
  --output-dir .local/run/ppo-rules-combined-validation-01 --if-exists cancel
```

To train against a mixture instead, replace `--rule-based-opponent` with
`--opponent-pool config/opponents/rules_and_frozen.json` and use another output
name. That pool is 70% frozen / 30% rules; validation remains the four-category
schedule above. Do not pass both training selectors.

`evaluations.md` now has populated Random, Dumb, Smart and Rules columns.
Learned may still show a dash because the checkpoint opponents are specifically
categorized as Dumb and Smart. Best selection uses **total wins / requested
battles**, counting draws and truncations as non-wins; each category has equal
weight for these inputs. No minimum per-category gate is imposed. Inspect the
Rules and learned-agent columns for tradeoffs, even if Overall improves.

Use a new training run when changing the validation suite. Retain the earlier
run history and do not compare old three-category Overall directly with the
new four-category Overall.
