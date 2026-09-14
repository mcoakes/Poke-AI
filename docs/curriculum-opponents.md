# Advance the opponent between stages

[Battle-count campaigns, named models and evaluation history](battle-count-campaigns.md)

For Random, Dumb and Smart **evaluation** opponents without changing the training
opponent schedule, see [Campaign evaluation pools](campaign-evaluation-pools.md).

[Curriculum guide](curriculum.md) · [Create a campaign](creating-campaigns.md) · [Every flag](reference/run-curriculum.md)

Use `--opponent-mode previous-stage` to freeze each stage's promoted candidate
as the next stage's opponent. The learner also starts from that candidate, then
updates independently. This provides a sequence of fixed opponents within one
campaign while the cumulative team pool continues to grow.

| Stage | Learner starts from | Frozen opponent |
|---:|---|---|
| 1 | Your `--resume` checkpoint | Your `--opponent-checkpoint` |
| 2 | Stage 1's promoted candidate | Separate copy of stage 1's promoted candidate |
| 3 | Stage 2's promoted candidate | Separate copy of stage 2's promoted candidate |
| N | Stage N−1's promoted candidate | Separate copy of stage N−1's promoted candidate |

“Promoted candidate” means the final checkpoint from the attempt that passed the
stage's checks. It does not mean the newest file discovered elsewhere in your
project. Opponent promotion happens only at a stage boundary, after `PROMOTE`.

## Start a new campaign

After installing the updated project, run from its root:

```bash
fronzen_checkpoint=".local/run/ppo-20260913-184241-113975/checkpoints/latest.zip"
curriculum_run=".local/run/curriculum-types-selfplay-01"

python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --opponent-mode previous-stage \
  --resume latest \
  --opponent-checkpoint "$fronzen_checkpoint" \
  --environments 48 \
  --showdown-workers process \
  --rollout-steps 128 \
  --batch-size 128 \
  --timesteps 100000 \
  --eval-episodes 1000 \
  --through-stage 10
```

This uses the existing ten-stage team progression and stops at the first `HOLD`.
To start just stage 1, replace `--through-stage 10` with `--stage 1`. To begin
with identical player policies, use `--resume "$fronzen_checkpoint"` instead of
`--resume latest`.

Continue later with:

```bash
python mgba/run_curriculum.py --run-dir "$curriculum_run" --through-stage 10
```

The mode is remembered. Omit initialization flags, including `--opponent-mode`,
on subsequent commands. Existing campaigns without a stored mode remain in
`fixed` mode. Changing modes requires a new run directory so past evaluations
retain their meaning.

## Retries and evaluations

If stage 2 returns `HOLD`, its next attempt resumes stage 2's last candidate but
still faces stage 1's promoted model. It does **not** replace the opponent with
the held attempt's model. The same rule applies after interruption or restarting
the runner. No opponent updates occur during PPO training or evaluation.

Each stage records fresh baseline evaluations against its selected opponent,
then evaluates the trained candidate against that same snapshot. Retries reuse
that stage's baseline. The normal counts remain 4,000 evaluation battles on a
first attempt and 2,000 on subsequent attempts when `--eval-episodes 1000`.

From stage 2 onward, the entering candidate and opponent have identical weights.
Their side-balanced baseline therefore provides a self-match reference rather
than repeating the previous stage's score. Draws and truncations still matter.
After training, the existing promotion thresholds apply; no rules are loosened
automatically when the opponent changes.

## Configure the mode in a curriculum

The supplied
[`type_progression_selfplay` preset](../config/curricula/type_progression_selfplay/README.md)
uses the same ten team additions and includes this top-level manifest field:

```json
"opponent_mode": "previous-stage"
```

Select it with
`--curriculum config/curricula/type_progression_selfplay/curriculum.json` on a
new run; the separate mode flag is then unnecessary. Add the same field to
your own manifest to use this behavior. `fixed` keeps the initial reference for
every stage. An explicit `--opponent-mode` overrides the manifest; a manifest
without the field defaults to `fixed`.

## Check which opponent was used

Each stage prints the opponent source and SHA-256 checksum. It also saves:

| File | Contents |
|---|---|
| `stage-NN/opponent.zip` | The actual frozen opponent for that stage |
| `stage-NN/stage.json` | Opponent mode, source stage, source path and checksum |
| `stage-NN/attempt-NNN/attempt.json` | Learner resume checkpoint and opponent identity for the attempt |
| `stage-NN/result.json` and `report.md` | Promotion result plus the opponent identity |
| `inputs/opponent.zip` | The original reference, retained unchanged for broad comparisons |

Compare before/after results **within a stage**. A lower win rate in a later
stage can reflect a different opponent and team pool. To measure longer-term
progress, replay the same separate suite against `inputs/opponent.zip`, using
the [broad evaluation command](curriculum.md#reports-interruption-and-broader-checks).
Winning against successive recent checkpoints alone does not demonstrate
retained performance against all earlier policies.
