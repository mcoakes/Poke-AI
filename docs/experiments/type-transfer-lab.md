# Stage-4 transfer experiment

The completed million-step run and next controlled experiment are documented
in [matchup-ablation.md](matchup-ablation.md). Its uploaded measurements are in
`transfer-01-analysis.json`. The text below records the original transfer-run
design and the earlier 200k results that motivated it.

The first expanded curriculum stopped correctly at stage 4. The fixed mirror
remained at 999/1000 wins before and after training, while the selected model
won only 339/1000 on seen-team tests and 307/1000 on heldout teams.

| Measurement | A: LR 0.0001 | B: LR 0.00002 |
|---|---:|---:|
| Starting validation | 17.0% | 17.0% |
| Best validation | 26.2% | 22.5% |
| Final validation | 26.2% | 21.7% |
| Best checkpoint, seen teams | 33.9% | 25.7% |
| Best checkpoint, heldout | 30.7% | 21.4% |
| Additional training transitions | 200,704 | 200,704 |
| Training time, including validation | 9.93 min | 9.97 min |

A's best checkpoint was its final checkpoint. Its validation score was still
improving near the budget limit, so this does not establish a learning plateau.
Both arms sampled the fixed mirror at approximately 9.8% of episode starts,
close to the intended 10%. Retention was measured successfully; there was no
no-rehearsal control to establish that rehearsal caused the preservation.

Memory monitoring recorded at least 22.36 GiB available out of 31.21 GiB,
with no swap use. Keep two concurrent jobs for this next experiment.

## What the decisions show

The first 16 heldout battles include concrete errors by A:

- Golem chose Earthquake into Flying-type Dodrio four consecutive times,
  although Double Edge was legal; the selected action had about 99.94% probability.
- Raichu chose Thunderbolt into Ground-type Golem twice while Strength was legal.
- Blastoise chose Waterfall into Water Absorb Lapras five times while Body Slam
  was legal. Lapras stayed at full HP as Blastoise lost HP.

These examples identify bad decisions, but do not quantify their full-run rate
or prove the cause of every loss. The observation already contains defender
types and abilities. The policy must learn how those features interact with each
move; this evidence does not establish a missing-observation bug.

The new audit adds current-target immunity measurements without changing PPO
choices, legal action masks, rewards, or the Rules opponent. They describe the
defender at decision time; an opponent can switch before the move resolves.
Check the metric's assessed/unknown counts and scope before interpreting it as
an error rate. This is diagnostic information, not a policy override.

## Question and controls

Compare the qualified fixed-mirror specialist's starting state with a genuinely
fresh policy on exactly the same frozen stage-4 train, validation and heldout
pools. Both arms use:

- The shared-moves architecture and v2 observations.
- Rules opponents, LR 0.0001, entropy coefficient 0.01, and the same seed.
- 90% stage-4 / 10% fixed-mirror episode sampling, with both teams drawn from
  the chosen group.
- Up to 1,000,000 new learner transitions (rounded to complete rollouts),
  validating every 50k steps and stopping after two consecutive validations at
  or above 65% wins with zero truncations.
- Two jobs, 16 player environments per job, CPU inference and one Torch thread.

The specialist arm starts again from the original qualified foundation, not
from the already-tested 200k stage-4 result. That keeps the new-task training
budget equal. The original blocked run and all its checkpoints stay unchanged.
The fresh arm has new weights and optimizer state; the specialist retains its
saved optimizer state. This comparison tests inherited training state as a
whole, not weights alone. One seed is a diagnostic, not a replication study.

Checkpoint selection uses validation only. Frozen checks compare both arms and
random/greedy/Rules baselines on the same heldout schedule. These are reused
stage-4 development checks, not an untouched final benchmark. The curriculum's
reserved stage-10 benchmark is not used.

Mirror performance is reported before and after for each arm, alongside the
common specialist reference. A fresh policy is not assumed to have learned the
mirror already. Any reported readiness criteria remain separate from selecting
the best validation checkpoint; the experiment does not automatically advance
or rewrite the blocked curriculum.

## Run and monitor

Extract the updated project archive over your source. From the project root:

```bash
source .venv/bin/activate
python run_transfer_lab.py \
  --source-run .local/run/team-curriculum-01 \
  --output-dir .local/run/type-transfer-01 \
  --jobs 2
```

Use the source directory that produced the uploaded curriculum report. No manual
checkpoint copying is needed. Initial frozen checks run before the two trainers
start; progress for those checks appears in the launcher terminal. Keep other training batches stopped so total
concurrency stays at two.

Watch both jobs in another terminal:

```bash
tail -n 20 -F .local/run/type-transfer-01/console/*.log
```

Watch evaluations:

```bash
watch -n 5 'for f in .local/run/type-transfer-01/runs/*/evaluations.md; do
  [ -f "$f" ] || continue
  echo "$f"
  cat "$f"
done'
```

Watch memory in a separate terminal:

```bash
watch -n 2 free -h
```

The runner also records memory.csv. This records available RAM and swap; it is
not a hard memory limit. Frozen-check progress appears in the launcher terminal.

Inspect the actual training team definitions:

```bash
python run_transfer_lab.py --output-dir .local/run/type-transfer-01 \
  --inspect-teams train --limit 3 --full-json
```

Use `mirror`, `validation`, or `heldout` to inspect another group. Inspection is
read-only. Generated source pools are preserved; the new run pins its own inputs.

Ctrl+C once in the launcher pauses gracefully. Wait for worker cleanup, then
resume the remaining budgets with:

```bash
python run_transfer_lab.py --output-dir .local/run/type-transfer-01 \
  --continue --jobs 2
```

The same viewers still apply:

```bash
tail -n 20 -F .local/run/type-transfer-01/console/*.log
```

```bash
watch -n 2 free -h
```

Upload `.local/run/type-transfer-01/transfer-results.zip` when it finishes.
The report should tell us whether more practice helps the specialist adapt,
whether fresh training avoids the inherited behavior, and whether either model
learns the broader matchup decisions while also mastering the earlier mirror.

## Verification

47 focused tests passed. A real simulator smoke ran both training jobs and
33 frozen checks (264 battles, zero truncations). It verified source preservation,
fresh initialization at timestep zero with an empty optimizer, complete episode
records, the immunity counters, team inspection, and a completed continuation
that started no new training or checks. These are operational checks using tiny
local smoke models, not learning results from your saved checkpoints.

See `type-transfer-verification.json` and `curriculum-01-analysis.json` for the
verification scope and measurements from the uploaded curriculum report.
