# Frozen replacement-choice diagnostic

This diagnostic has completed: all 20 checks and 20,000 battles, with no
truncations. The next experiment is the
[bench-order learning comparison](bench-order-ablation.md). Detailed results are
in `replacement-01-analysis.json`; the rest of this guide records the diagnostic
design and its runnable commands.

The completed matchup experiment improved learning and eliminated the recorded
immunity mistakes. It did not yet produce a reliably stronger validation-selected
checkpoint on the heldout pool.

All percentages below are wins against Rules. Best means the checkpoint selected
by validation; final means the checkpoint at the fixed training budget.

| Policy | Checkpoint | Validation | Seen teams | Heldout teams | Fixed mirror |
|---|---|---:|---:|---:|---:|
| Control: matchup inputs disabled | Best | 27.0% | 34.0% | 34.2% | 27.5% |
| Matchup: inputs enabled | Best | 42.4% | 43.0% | 36.0% | 45.9% |
| Control | Final | 25.6% | 36.3% | 32.2% | 49.6% |
| Matchup | Final | 40.2% | 50.4% | 40.6% | 61.8% |

Each arm received 1,000,448 steps and took about 42 minutes to train. Initial
evaluation behavior matched exactly. The recorded initialization proof reports
equal weights, empty optimizers and identical predictions; the uploaded results
did not contain checkpoint tensors for independent weight inspection.

The heldout improvement of the selected-best model was 1.8 percentage points.
A paired bootstrap interval was approximately -0.6 to +4.2 points. At the fixed
final budget the improvement was 8.4 points, with an interval of +5.8 to +11.0.
These intervals retain both candidate seats in each matchup/seed cluster. They
describe this fixed 16-team pool and trained pair, not variation across training
seeds or entirely new teams. Final is reported as a diagnostic endpoint; its
higher heldout score does not replace validation-based selection.

The strongest behavioral improvement is unambiguous within the audited scope:

| Heldout checkpoint | Immune choices / assessed moves |
|---|---:|
| Control best | 773 / 8,863 (8.7%) |
| Control final | 596 / 8,974 (6.6%) |
| Matchup best | 0 / 8,530 |
| Matchup final | 0 / 8,638 |

Every selected move in these checks was assessed. These counters concern the
current defender before the opponent acts; they do not guarantee what happens
after a switch. Zero observed mistakes is not a guarantee for all future teams.
The known-mechanics inputs helped the policy learn this behavior; no actions
were forced during that training experiment.

## Why test replacements next?

In the bounded traces, the best matchup model chose party slot 2 in all 48
multi-option forced replacements across the seen-team, validation and heldout
checks. Final selected the lowest legal replacement slot in 37 of 47 such
decisions. These traces cover the first 16 battles and at most 24 decisions per
battle in each suite. They suggest a positional preference, but cannot establish
its frequency across all battles or prove that any particular choice was wrong.

The current move features describe the active attacker against its current
defender. They do not supply equivalent matchup relationships for each bench
Pokémon. Also, the earlier rotated checks rotate move slots, not bench slots.

The greedy baseline's 43.8% heldout score is not a pure move-selection comparison:
greedy chooses attacks using estimated damage, but uses Rules for forced
replacements. Its advantage could involve either component.

The next experiment keeps the PPO weights frozen and crosses two factors:

| Condition | Candidate team order | Who chooses forced replacements? |
|---|---|---|
| Original / PPO | Original | The learned model |
| Reversed bench / PPO | Same lead, bench reversed | The learned model |
| Original / Rules-assisted | Original | Rules, only when multiple replacements are legal |
| Reversed bench / Rules-assisted | Same lead, bench reversed | Rules, only when multiple replacements are legal |

With three Pokémon, bench reversal simply swaps slots 2 and 3. Species, moves,
stats, items, first Pokémon, opponent team, battle seeds and candidate seats stay
the same. This changes the position of available replacements without giving
either side a different team composition or lead.

Rules-assisted conditions are explicitly diagnostic hybrid policies. They keep
PPO's attack choices and voluntary switches, but let Rules choose after a faint
when multiple replacements are available. Their win rates must never be reported
as the learned model's unassisted performance. Rules is a heuristic, not a source
of guaranteed optimal replacements.

Run both the validation-selected best and fixed-budget final matchup checkpoints
on the seen-team and heldout schedules. This produces 16 policy checks, plus four
greedy/Rules baseline checks to measure their own sensitivity to bench order:
20,000 battles with the current 1,000-battle schedules. All-battle audits measure multi-option
replacement decisions, lowest-slot choices, disagreement with Rules, and how often
assistance actually changes the model's proposed action. Bounded traces retain
the original proposal alongside the executed choice.

## What the result can tell us

| Observation | Implication |
|---|---|
| Bench reversal changes unassisted wins substantially | Party position affects performance; investigate order robustness. |
| Rules-assisted replacements improve wins | This replacement intervention helps; learned replacement decisions deserve targeted work. |
| Assistance reduces sensitivity to bench order | Further evidence linking order sensitivity to replacement choices. |
| Assistance gives little benefit or hurts | Rules replacements did not solve the remaining losses; examine move valuation and voluntary switching before assuming a replacement fix. |

Battles can diverge after any changed action. These are full-policy intervention
results, not exact counts of wins attributable to individual decisions. A later
training experiment should test the mechanism supported by these results. A new
seed replication of the matchup input comparison remains useful before adopting
it as a generally reliable improvement.

Rehearsal worked as configured in the completed training: about 9.9% of episode
starts used the mirror. Both agents began fresh, at 0.2% mirror wins, so their
final mirror scores measure partial acquisition of that task rather than loss of
a previously mastered 99.9% skill. The original specialist still supplies a
separate target. This frozen diagnostic performs no training and cannot cause
forgetting.

## Run and monitor

Extract the updated archive over your project source. From the project root:

```bash
source .venv/bin/activate
python run_replacement_diagnostic.py \
  --source-run .local/run/matchup-ablation-01 \
  --output-dir .local/run/replacement-diagnostic-01 \
  --jobs 2
```

Here `--jobs 2` means two concurrent evaluation jobs. There are no training jobs.
The runner copies and verifies the required checkpoints itself. It preserves the
source results, source checkpoints, and original 400k working reference. Allow
several minutes for the full set of battles; progress appears in the launcher.

Watch job logs in another terminal:

```bash
tail -n 20 -F .local/run/replacement-diagnostic-01/console/*.log
```

Watch completed checks and active jobs in another terminal:

```bash
watch -n 5 cat .local/run/replacement-diagnostic-01/progress.json
```

Watch memory separately:

```bash
watch -n 2 free -h
```

The runner also records `memory.csv`. The previous training pair retained at
least 21.80 GiB available RAM with zero swap use. Keep concurrency at two.

Inspect the original team definitions and moves:

```bash
python run_replacement_diagnostic.py \
  --output-dir .local/run/replacement-diagnostic-01 \
  --inspect-teams heldout --limit 3 --full-json
```

Use `train`, `mirror`, or `validation` for other original groups. The diagnostic's
frozen suite files record the actual original and reordered battle teams.

Ctrl+C stops the evaluation workers. Preserve the directory, then continue with:

```bash
python run_replacement_diagnostic.py \
  --output-dir .local/run/replacement-diagnostic-01 --continue --jobs 2
```

Complete checks are reused only when their saved identities and episode records
validate. Partial checks remain available and are rerun in separate retry folders.
Use the same viewers after continuing:

```bash
tail -n 20 -F .local/run/replacement-diagnostic-01/console/*.log
```

```bash
watch -n 2 free -h
```

Upload `.local/run/replacement-diagnostic-01/replacement-results.zip` when it
finishes. The report separates learned and Rules-assisted outcomes and includes
the replacement counts needed to test the current hypothesis.

The uploaded matchup analysis is recorded in `matchup-01-analysis.json`.
Operational verification is recorded in `replacement-diagnostic-verification.json`.

The update passed 70 targeted tests and a 160-battle simulator smoke using tiny
local models. Two checks had incomplete saved episode logs despite earlier
completion messages; independent verification caught them. Their original files
were preserved, and retrying those two checks with one worker produced complete
saved evidence. The cause of the saved-output inconsistency was not established.
The smoke does not measure your trained models' performance.

If the runner reports incomplete saved evidence, inspect the named console log,
then retry with one worker:

```bash
python run_replacement_diagnostic.py \
  --output-dir .local/run/replacement-diagnostic-01 --continue --jobs 1
```

Keep watching the logs and memory during that retry:

```bash
tail -n 20 -F .local/run/replacement-diagnostic-01/console/*.log
```

```bash
watch -n 2 free -h
```
