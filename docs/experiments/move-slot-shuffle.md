# Learn move identity across different slots

[Experiment record](README.md) · [Training flags](../reference/train-ppo.md) ·
[Move-order diagnostic](../move-order-audit.md)

The diversity checkpoint repeatedly selects the same numeric slot when move
contents change. The next experiment adds `--shuffle-move-slots` to training.
It asks whether removing predictable learner move order improves decisions.
It does not reward agreement with a heuristic or directly penalize Move 1.

## How the option works

- Available in standalone `train_ppo.py`, with Showdown and v1 or v2 observations.
  It is off by default; campaign runners do not enable it implicitly.
- Each learning player receives independent random permutations of all four
  slots for each of its six party positions. A mapping lasts for the whole
  battle, including switches and move changes. It refreshes at the next battle.
- Move ID, type, current/max PP, v2 details and the active move mask travel
  together. Empty, disabled and forced-action slots follow the same mapping.
  One-, two- and three-move Pokemon are supported.
- A learner action is translated back to the original simulator move slot.
  Switch slots, wait actions, team files, rewards and the simulator stay unchanged.
  The ending battle's mapping is used for truncation value bootstrapping before
  the next battle receives a new mapping.
- The wrapper sits outside frozen-opponent selection. Rules, Random and frozen
  checkpoints receive their original observations. In self-play, both learning
  players receive their own permutations. The learner's opponent-move features
  retain their original order.

The policy learns on the presented observations, actions and masks. Its saved
observation/action spaces are unchanged. Evaluation and human-play inference do
not need a wrapper; actions refer to the move slots actually presented there.
This training augmentation remains Showdown-only. For native v2 inference,
see [mGBA v2](../mgba-observation-v2.md).

Independent RNG streams avoid consuming team, battle or policy sampling draws.
`config.json` records the setting, mapping version and effective shuffle seed.
Same-directory continuation restores the flag and starts fresh battles with the
usual new session seed. A new run from `--resume` must explicitly enable the
flag again. Changing it within an existing run is rejected to keep experiments
comparable. This is data augmentation, not a guarantee of order invariance.

## Run the first comparison

Use the archived diversity reference, the existing training teams and rewards,
and the rules-only opponent used in E08. This changes learner move presentation
while leaving the opponent's behavior definition fixed. Do not add a new reward
or increase the learning rate for this comparison.

Run from the project root with the training environment and Showdown installed:

```bash
shuffle_run=".local/run/ppo-move-shuffle-01"
reference_model="models/observation-v2-diversity-best-01.zip"
diversity_run=".local/run/diversity-v2-01"
combined_suite=".local/run/validation-rules-and-agents-01/suite.json"

python mgba/train_ppo.py \
  --backend showdown --resume "$reference_model" \
  --rule-based-opponent --shuffle-move-slots \
  --team-dir "$diversity_run/teams/training" \
  --reward-config "$diversity_run/inputs/reward.json" \
  --learning-rate 0.0001 --rollout-steps 128 --batch-size 128 \
  --environments 64 --showdown-workers process \
  --total-timesteps 100000 \
  --eval-suite "$combined_suite" --eval-every 50000 \
  --battle-log --no-readable-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --checkpoint-every 25000 --keep-checkpoints 2 --keep-evaluations 2 \
  --output-dir "$shuffle_run" --if-exists cancel

cat "$shuffle_run/evaluations.md"
```

This is a new fork, not continuation of the reference run. With 32 learning
players and rollout length 128, the requested budget rounds to 102,400 transitions.
Validation runs initially, at rounded 50,000-transition intervals and at the
endpoint. It uses the original team move order, with all four opponent categories.
The initial reference should reproduce E09: 59.175% overall, 74.2% Random,
65.2% Dumb, 61.9% Smart and 35.4% Rules on that suite. Inspect any discrepancy
before interpreting the intervention.

Spectator logs are retained and bounded. Simulator/readable logs describe the
**executed original move slots**, not the learner's shuffled slot numbers.
Programmatic step results include `info["move_slot_shuffle"]` with zero-based
`learner_action` and `simulator_action`, plus the four-entry one-based
`new_to_old_slots` mapping used for that decision. No additional unbounded
per-step log is created. Ordinary switching/reward/throughput metrics remain.

## Evaluate the endpoint and check slot preference

`best.zip` may still contain the initial reference. The first run's final
validation was evaluation 3 at 1,593,344 timesteps, with 55.9% overall and 32.2%
Rules wins. Its best checkpoint remained the initial reference; see the
[recorded results](README.md#e12--learner-move-slot-shuffling-first-run-completed).
Audit the retained final evaluation directly:

```bash
python mgba/audit_move_order.py \
  --evaluation-dir .local/run/ppo-move-shuffle-01/evaluations/evaluation-000003 \
  --output-dir .local/run/audit-move-shuffle-final-01 \
  --matchups 32 --seed 261002 \
  --environments 2 --showdown-workers process \
  --log-max-mb 2 --log-backups 1

cat .local/run/audit-move-shuffle-final-01/report.md
```

That command reuses its saved candidate checkpoint, suite and outcomes. For a
different run, use the directory of its completed final evaluation, identified
in `evaluation-history.json`. If the final evaluation was not retained, run
this alternative instead to evaluate `final.zip` and audit those results:

```bash
python mgba/evaluate_ppo.py \
  --checkpoint .local/run/ppo-move-shuffle-01/checkpoints/final.zip \
  --suite .local/run/validation-rules-and-agents-01/suite.json \
  --reward-config .local/run/diversity-v2-01/inputs/reward.json \
  --environments 48 --showdown-workers process \
  --battle-log --no-readable-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --output-dir .local/run/eval-move-shuffle-final-01

python mgba/audit_move_order.py \
  --evaluation-dir .local/run/eval-move-shuffle-final-01 \
  --output-dir .local/run/audit-move-shuffle-final-01 \
  --matchups 32 --seed 261002 \
  --environments 2 --showdown-workers process \
  --log-max-mb 2 --log-backups 1
```

Use new output directory names for reruns. Compare the final category results
with E08/E09 and the audit with E11: semantic agreement, same-slot selection and
aligned probability change. The same seed samples the same suite matchups, but
the new policy can visit different battle positions. This is not a paired
same-position policy comparison. A uniform slot histogram alone is insufficient:
random presentation can cause that without useful move selection.

Look for lower order sensitivity **and** stronger Rules results while retaining
performance against learned agents. Treat 100,000 steps as a bounded diagnostic,
not a promise of recovery from an established shortcut. Assess results before
extending the run or changing architecture. A promising selected checkpoint
still needs fresh held-out evaluation; these development suites have informed
several decisions already.

The diversity reference is already archived. Preserve the endpoint for a clear
experimental record, without replacing it:

```bash
mkdir -p models
cp -n .local/run/ppo-move-shuffle-01/checkpoints/final.zip \
  models/observation-v2-move-shuffle-final-01.zip
```

If validation selects an improved checkpoint, archive it separately:

```bash
cp -n .local/run/ppo-move-shuffle-01/checkpoints/best.zip \
  models/observation-v2-move-shuffle-best-01.zip
```

Record its exact evaluation row and retain the named diversity reference for
future comparisons. `best.zip` selects total wins/requested battles, not a
minimum Rules win rate.

The first run and its audit are now recorded in E12. The next bounded comparison
is [shared move scoring](shared-move-policy.md), using fresh models to test an
architectural constraint. Keep this shuffle endpoint as a diagnostic result.
