# Bench-order learning comparison

## Completed result: October 19, 2026

Both arms completed 500,736 added transitions. The uploaded results contain
35 complete frozen checks, 32,680 battles, no truncations, and no classified
immune move choices. The observed memory minimum was 22.1 GiB available with
no swap use; the full run took about 46 minutes, including checks.

| Heldout comparison, mean of both orders | Control | Bench shuffle | Difference | Paired 95% interval |
|---|---:|---:|---:|---:|
| Validation-selected checkpoints | 37.4% | 36.7% | -0.7 pp | -2.05 to +0.65 pp |
| Fixed final checkpoints | 40.6% | 42.1% | +1.5 pp | -0.30 to +3.30 pp |

The shuffled arm never exceeded its initial 42.1% balanced validation score;
its selected checkpoint therefore remains at the starting timestep. Control's
selected checkpoint reached 45.2% validation at 852,992 total steps. The final
shuffled model's higher heldout score does not override validation selection.

Intervals use 20,000 paired bootstrap resamples of 500 matchup/seed groups,
preserving both candidate seats and bench orders. They describe uncertainty on
the reused development suite, not repeatability across training seeds.

The shuffled final model chose the lowest legal replacement slot on 45.2% of
eligible seen-team decisions and 71.0% of heldout decisions. The control final
rates were 62.4% and 76.6%. This is a behavior change, not proof of improved
replacement quality. Both final models increased mirror wins from 45.9% to
about 61%; the selected-policy mirror warning is not evidence of forgetting.

**Next step, updated October 19:** continue targeted training with the
[replacement-feature comparison](switch-feature-ablation.md), while keeping the
existing models available for student play. This supersedes the earlier advice
to stop training for a formal presentation. Full verification and uncertainty
calculations are in `bench-order-01-analysis.json`.
Checkpoint hashes were checked against the supplied metadata; model ZIPs were
not included in the uploaded results and could not be rehashed locally.

![Bench-order validation and paired heldout differences](figures/bench-order-result.png)

## Original experiment design

This is one bounded training experiment for the presentation deadline. The
question is whether varying bench positions helps the same policy choose
replacements by matchup rather than by party slot.

## Why this experiment

The completed frozen replacement diagnostic contains 20,000 battles, with no
truncations. The validation-selected matchup model chose the lowest available
replacement slot in all 3,970 eligible decisions across both pools and orders.
Rules-assisted replacements raised its original-order heldout wins from 36.0%
to 40.8%; the later final checkpoint rose from 40.6% to 45.0%. Those assisted
scores are hybrid-policy results, not learned-agent scores.

That identifies a useful intervention to investigate. It does not show that
every choice disagreeing with Rules was wrong, or that replacements explain all
remaining losses. The next change is training order diversity, with the policy
architecture and reward held fixed.

## Fixed design

| Setting | Both jobs |
|---|---|
| Initialization | Same source validation-best matchup checkpoint and optimizer state |
| Source checkpoint | 501,760 steps in the completed normal run |
| New training budget | 500,000 learning transitions, rounded to a complete PPO rollout |
| Training seed | 261100 |
| Opponent | Rules |
| Teams | Same 16 current teams and fixed mirror rehearsal |
| Sampling | 90% current task; 10% fixed mirror |
| PPO | Learning rate 0.0001; entropy coefficient 0.01; 128 rollout steps; batch size 128; four epochs |
| Parallelism | Two jobs maximum; 16 environment players per job, eight of them learning |
| Validation | Equal original and reversed candidate-bench cases; every 50,000 added transitions |
| Selection | Highest balanced-validation wins/requested; earlier checkpoint retains ties |
| Endpoints | Validation best and fixed-budget final, reported separately |

The control keeps the existing party order. The treatment samples a canonical
pair of teams first, then independently randomizes only Player 1's occupied
bench slots. The initial lead, opponent's team order, species, moves, levels,
items and other team values stay fixed. In the current three-Pokémon task there
are two possible bench orders, each sampled with equal probability.

The fixed-pair mirror rehearsal keeps its original order in both jobs. This
preserves the earlier task while varying the current task. Team selection and
bench permutation use separate random-number streams. Corresponding per-pair
canonical draws follow the same sampling process; asynchronous battle lengths
can still make the global episode sequence differ between policies.

Do not implement this by adding reversed copies to the team directory: that
would change the canonical sampling pool and could introduce same-composition
pairings that were previously excluded.

Both jobs receive the same fixed budget. Neither stops early because it looks
better on heldout checks. Checkpoints are written throughout; Ctrl+C can pause
the experiment. There is no automatic curriculum progression or model promotion.

## Decision after this run

The primary comparison is the validation-selected treatment minus the
validation-selected control, averaged equally over original and reversed
heldout orders. A gain of five percentage points is a useful practical target,
not a guarantee of statistical significance. Report each order separately.

Also check fixed-mirror performance against this run's starting checkpoint,
the control, and the immunity audit. The source best model's earlier mirror
score was 45.9%; the older specialist's approximately 99.6% is a different
reference, not a skill this starting model had already mastered.

Lowest-slot frequency is descriptive. Fifty percent is not an optimal strategy,
and simply switching to the highest slot would not establish learning. Wins
and behavior need to improve together. The reused heldout pools are development
checks; one training-seed comparison does not establish repeatability.

For the deadline, finish this pair before considering another change. If the
gain is clear and time allows, use a second training seed for confirmation.
If it is small, retain the negative result; the current next experiment targets
replacement inputs directly rather than repeating bench randomization.
Reserve a genuinely fresh evaluation on the same task distribution for the
selected final candidate; do not keep choosing models on the reused heldout pool.

## Run and monitor

Extract the updated project archive over the existing source. From the project
root, with the simulator and training dependencies already installed:

```bash
source .venv/bin/activate
python run_bench_order_ablation.py \
  --source-run .local/run/matchup-ablation-01 \
  --output-dir .local/run/bench-order-ablation-01 \
  --jobs 2
```

Allow roughly 30–60 minutes based on the preceding machine measurements,
including validation and final checks. This is an estimate, not a runtime limit.
The default budget is fixed. `--timesteps 250000` can request a shorter comparison
when preparing a new run; do not change a prepared run's settings file.

Watch both job logs in another terminal:

```bash
tail -n 20 -F .local/run/bench-order-ablation-01/console/*.log
```

Watch validation history:

```bash
watch -n 5 'for f in .local/run/bench-order-ablation-01/runs/*/evaluations.md; do
  [ -f "$f" ] || continue
  echo "$f"
  cat "$f"
done'
```

Watch the current experiment phase:

```bash
watch -n 5 cat .local/run/bench-order-ablation-01/status.json
```

Watch memory separately:

```bash
watch -n 2 free -h
```

Each trainer also writes `updates.jsonl` and PPO CSV metrics. The launcher writes
`memory.csv`, including validation peaks. `inputs/settings.json`, `plan.json`
and each trainer's `config.json` record the effective settings. The treatment's
bench-order manifest records the canonical teams and generated variants.

Inspect the original team definitions and moves:

```bash
python run_bench_order_ablation.py \
  --output-dir .local/run/bench-order-ablation-01 \
  --inspect-teams train --limit 3 --full-json
```

Use `mirror`, `validation`, or `heldout` for another group. Prepared inputs are
immutable; the two jobs use the same copied team definitions.

Pause with Ctrl+C once and wait for cleanup. Continue with:

```bash
python run_bench_order_ablation.py \
  --output-dir .local/run/bench-order-ablation-01 --continue --jobs 2
```

Use the same viewers after continuing:

```bash
tail -n 20 -F .local/run/bench-order-ablation-01/console/*.log
```

```bash
watch -n 2 free -h
```

Continuation restores each job's saved condition and remaining budget. It starts
new battles; it does not promise an identical uninterrupted learning trajectory.
Complete frozen checks require matching saved identities, episode records,
replacement audits and bounded traces. Partial checks are preserved and retried.

## Results and presentation

The runner writes `report.md` with detailed results and
`presentation-summary.md` with a compact account of the question, comparison,
results and limits. Use `presentation-brief.md` in this documentation directory
for the project narrative and completed findings.

Upload `.local/run/bench-order-ablation-01/bench-order-results.zip` when complete.
It includes the evidence and reports; model weights stay on your machine.
The original 400k reference, source checkpoints and earlier failed gates remain
unchanged.

## Implementation verification

The verification record is in `bench-order-ablation-verification.json`: 132
focused tests passed, followed by real two-job training with tiny test models
and 35 accepted frozen checks (320 battles, no truncations). Saved episodes,
audits, traces, team variants and checkpoint provenance were checked. This is
an operational test, not evidence that bench shuffling improves the user model.

The local smoke encountered incomplete saved evaluation files. Strict gates
rejected them; normal continuation preserved and retried those attempts. The
final checks passed independently, and a further completed continuation left
inputs, training outputs, checks and console logs unchanged. If a real run
reports incomplete evidence, use the continuation command above; do not edit
the summaries to mark it successful.
