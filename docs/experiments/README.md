# Training experiment record

The current task is [switch practice plus longer PPO](switch-practice.md):
two matched one-million-step jobs from the selected teacher-trained model,
one with a new warmup that gives rare switch decisions more attention.
The guide includes launch, live logs, evaluations, memory, team inspection,
pause/resume, permanent model copies and play commands.

The completed [teacher warmup comparison](teacher-training.md) selected a model
with 51.37% validation, 46.50% on then-unseen compositions and 93.7% fixed-mirror
wins against Rules. Its +1.70-point unseen-team advantage over PPO alone remains
uncertain; its +3.65-point improvement over the parent is more encouraging.
The warmup's held-back ordinary-switch agreement was only 4/65 examples,
motivating the next targeted treatment. See `teacher-training-01-analysis.json`.

The completed [training-coverage comparison](training-coverage.md) did not
establish better general play. Expanded-pool best won the old validation score
(48.7% versus 48.0%), but scored 39.50% on reused heldout checks versus 42.55%
for small-pool best. See `training-coverage-01-analysis.json` for verified
results and paired uncertainty. Ordinary move/switch decisions diverged before
forced replacements in the sampled traces, motivating direct action supervision.

The completed [replacement-feature comparison](switch-feature-ablation.md)
produced a fixed-final heldout gain of 3.35 percentage points, but neither arm
beat its initial validation score. See `switch-feature-01-analysis.json` for
verified results and paired uncertainty. The measured training-pool imbalance
motivates the current coverage experiment; it is not a proven sole cause.

The [student play and Q&A guide](presentation-brief.md) explains the existing
results. The [fresh-team check](presentation-check.md) remains an optional
frozen evaluation; it is not a requirement to stop training before the event.

The [bench-order learning comparison](bench-order-ablation.md) completed two
matched 500k-step continuations. Shuffling changed replacement preferences but
did not demonstrate a useful win-rate improvement: selected-policy heldout
difference -0.7 points; fixed-final difference +1.5 points. Both paired intervals
include zero. The numerical verification is in `bench-order-01-analysis.json`.

The [frozen replacement diagnostic](replacement-diagnostic.md) has completed:
20,000 battles confirmed a lowest-slot shortcut and a net benefit from a
replacement-only Rules intervention. Its numerical record is
`replacement-01-analysis.json`; hybrid scores remain separate from learned scores.

The preceding experiment is the [matchup-feature comparison](matchup-ablation.md):
fresh and specialist training both plateaued on stage-4 teams after one million
new steps. That two-job experiment isolated explicit matchup inputs while
preserving the teams, rewards, rehearsal, and validation rules. Its guide includes
live job logs, memory monitoring, team inspection, and pause/resume commands.

For the current fixed-team learnability diagnosis, see
[fixed-team lab](fixed-team-lab.md). This replaces further small tuning runs
until the simple action-choice and mirror tasks have been checked.

For the four-seed, longer shared-moves experiment, see
[parallel longer training](parallel-long-training.md). To test explicit alternative
settings, see [parallel strategy experiments](parallel-strategies.md).

[Documentation](../README.md) · [Next experiment](shared-move-policy.md)

This is the consolidated record of the observation-v2 experiments. Results below
come from user-supplied terminal reports and uploaded diagnostic JSON. Referenced
run directories live on the user's machine; only supplied artifacts were inspected. Rounded percentages
are retained as reported. Unknown checkpoint timesteps are not reconstructed.
This is a development record, not a finalized methodology or evidence of wins
across independent training seeds. Implementation fixes belong elsewhere.

## Observation-v2 progression

The interface audit reported 15 v2 checks passed, zero correctness failures and
zero reported information limitations in that scope. Its separate v1 regression
scope reported 51 passes and 13 legacy information limitations. These are wiring
checks, not tests of tactical skill or complete mechanic coverage.

All three evaluations below used the same 3,000-case validation suite (SHA-256
`bf3604d97a671788c09423261732c3b8c994f126537adaba3379eef35db231a3`).

| Evaluation directory under `.local/run/` | Timesteps | Overall | Random | Dumb | Smart | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| `eval-observation-v2-01` | 101,376 | 60.6% | 72.6% | 57.8% | 51.4% | 0 |
| `eval-observation-v2-02` | Not supplied | 65.0% | 73.2% | 64.7% | 57.2% | 0 |
| `eval-observation-v2-03` | 1,081,344 | 62.9% | 73.0% | 61.0% | 54.7% | 0 |

Continued training did not monotonically improve validation. Earlier evaluation
weights were not always preserved, so the historical 65.0% result is not treated
as a recoverable checkpoint. Periodic validation was then enabled.

## E01 — Periodic validation, original learning rate

Run: `.local/run/ppo-observation-v2-01`. Learning rate was the original 0.0003
setting; the frozen opponent and validation pool were retained. A further
500,000 requested timesteps were rounded to complete PPO rollouts.

| Timesteps | Overall | Random | Dumb | Smart | Selected best at this point |
|---:|---:|---:|---:|---:|---|
| 1,081,344 | 62.9% | 73.0% | 61.0% | 54.7% | Yes |
| 1,183,744 | 65.1% | 73.7% | 62.3% | 59.3% | Yes |
| 1,286,144 | 63.4% | 71.6% | 61.4% | 57.2% | No |
| 1,388,544 | 59.7% | 70.1% | 56.6% | 52.3% | No |
| 1,490,944 | 63.0% | 72.9% | 61.7% | 54.3% | No |
| 1,585,152 | 65.1% | 73.3% | 64.6% | 57.3% | No |

All rows had zero truncations. Selection uses exact wins, not rounded display
percentages; the final 65.1% did not replace the earlier best.

## E02 — Lower learning rate

Run: `.local/run/ppo-observation-v2-lr0001-01`. Started from E01's best at
1,183,744 timesteps; learning rate 0.0001, with the same mixed three-Pokemon
training profile, frozen opponent, rewards and validation suite.

| Timesteps | Overall | Random | Dumb | Smart | Selected best at this point |
|---:|---:|---:|---:|---:|---|
| 1,183,744 | 65.1% | 73.7% | 62.3% | 59.3% | Yes |
| 1,286,144 | 69.6% | 77.5% | 69.3% | 62.0% | Yes |
| 1,388,544 | 68.3% | 76.2% | 67.5% | 61.2% | No |
| 1,490,944 | 64.3% | 74.1% | 62.8% | 56.0% | No |
| 1,593,344 | 65.8% | 77.5% | 63.5% | 56.5% | No |
| 1,687,552 | 65.0% | 74.9% | 63.3% | 56.8% | No |

All rows had zero truncations. The best improved, but continued training again
lost the peak. This single run does not establish a general learning-rate effect.

## E03 — Fresh battles, familiar team pool

Report: `.local/run/verify-v2-lr0001-01/comparison.md`. Compared E01 best
(1,183,744) against E02 best (1,286,144) on 3,000 identical new test cases per
model; same Random/Dumb/Smart references, new pairings/seeds, both player seats.

| Opponent | E01 best | E02 best | Change | Paired 95% interval |
|---|---:|---:|---:|---|
| Overall | 63.5% | 68.2% | +4.7 pp | +2.8 to +6.5 pp |
| Random | 73.5% | 75.9% | +2.4 pp | -0.8 to +5.6 pp |
| Dumb | 62.3% | 69.0% | +6.7 pp | +3.4 to +9.8 pp |
| Smart | 54.8% | 59.7% | +4.9 pp | +1.7 to +8.1 pp |

Zero truncations. Mean decisions: 16.3 versus 16.4. Evidence supports an
improvement on new battles drawn from the familiar team pool, not yet new teams.

## E04 — New team configurations

Report: `.local/run/generalization-v2-01/baseline-report.md`. Same two best
checkpoints, 3,000 cases each. Separate three-Pokemon pool generated with seed
26092702, 60% competitive sets, 20% unrestricted sets, 20% learnset-based sets,
80% common competitive items. Percentages apply to Pokemon slots, not teams.
The tool excludes exact normalized party overlap with recorded training and
validation inputs; the accepted count was not included in the supplied report.

| Opponent | E01 best | E02 best | Change | Paired 95% interval |
|---|---:|---:|---:|---|
| Overall | 57.0% | 57.3% | +0.2 pp | -1.4 to +1.9 pp |
| Random | 63.0% | 64.8% | +1.8 pp | -1.3 to +4.7 pp |
| Dumb | 55.3% | 54.1% | -1.2 pp | -3.9 to +1.5 pp |
| Smart | 52.8% | 52.9% | +0.1 pp | -2.6 to +2.9 pp |

Exact total wins: 1,711 versus 1,718 out of 3,000. E02 had one truncation;
E01 had none. Mean decisions: 17.9 versus 18.6. No clear improvement transferred
to this pool. Unfamiliar configurations and a changed generator distribution
both contribute to this test, so this does not isolate memorization or prove
the cause of weak transfer. Individual species and curated sets may be familiar.

## E05 — Broader training distribution (completed)

[Setup and commands](team-diversity.md). Start from E02 best, retain learning
rate 0.0001 and the frozen training opponent, and generate disjoint pools of
3,000 training, 500 validation and 500 reserved test teams with matched generator
settings. This changes both pool size and composition relative to E02; it tests
a broader-data intervention, not size alone.

User-supplied result: `.local/run/diversity-v2-01/final-test/baseline-report.md`.
Starting checkpoint: 1,286,144 timesteps; validation-selected candidate:
1,490,944 timesteps. Each received 3,000 reserved-test battles, with both
candidate seats represented. Checkpoint/suite hashes are in the run artifacts
and were not included in the pasted report.

| Opponent | Starting | Selected | Change | Paired 95% interval |
|---|---:|---:|---:|---|
| Overall | 55.3% | 67.2% | +11.9 pp | +10.3 to +13.6 pp |
| Random | 62.0% | 74.9% | +12.9 pp | +9.9 to +15.9 pp |
| Dumb | 56.3% | 66.3% | +10.0 pp | +7.4 to +12.9 pp |
| Smart | 47.6% | 60.4% | +12.8 pp | +9.8 to +15.7 pp |

Exact wins: 1,659 versus 2,016. Truncations: 1 versus 0. Mean decisions:
18.8 versus 16.0; mean rewards: 989.18 versus 2449.19. Starting seat win rates:
54.8%/55.8%; selected: 68.1%/66.3%. This supports improvement on disjoint team
configurations from this generator, across all three opponent groups. It does
not isolate diversity from additional training, establish performance on every
team distribution, or quantify variation across training seeds. Intervals
resample matched cases, not independent training runs. Retain the selected
checkpoint as a reference and preserve this reserved-test result.

Use the initial validation as this experiment's reference. Select checkpoints
only on validation; compare starting/best on the reserved test after selection.
E04 has now influenced our decisions and is a development benchmark, not an
untouched final test. Future results should append rows here with run paths,
checkpoint timesteps/hashes, suite hashes, counts, uncertainty and limitations.
Preparation writes `experiment.json`; training/evaluation supply the exact
artifacts. This narrative record is maintained explicitly, not automatically
inferred from every run on the user's machine.

## E06–E08 — Rules evaluation and training diagnostics

The standalone rule schedule is fixed at 1,000 battles, both seats, suite hash
`7696d8b26658d4163ca5e47dea3c2e522f44a138e4e744ca9311ff94a02e17bd`.
All results here are user-supplied. Checkpoint hashes were not included.

| Experiment / candidate | Rules wins / losses / draws | Rules win rate | Truncated | Mean decisions |
|---|---|---:|---:|---:|
| E06: E05 diversity best, 1,490,944 steps | 354 / 643 / 3 | 35.4% | 0 | 11.061 |
| E07: mixed-training final, 1,593,344 steps | 357 / 639 / 4 | 35.7% | 0 | 11.074 |
| E08: rules-only final, 1,593,344 steps | 368 / 628 / 4 | 36.8% | 0 | 11.003 |

Evaluation directories: `eval-rules-v1-01`, `eval-rules-v1-mixed-final-01`,
`eval-rules-only-final-01` under `.local/run/`. A separate
`eval-rules-v1-after-mixed-01` used the unchanged best checkpoint and repeated
354 wins; it is not evidence about the trained final weights.

Both training forks started at E05 best, retained training teams and rewards,
and requested 100,000 additional timesteps at learning rate 0.0001. E07 used
70% frozen / 30% rules per battle; recorded completions were 4,814 frozen and
2,066 rules (30.0% rules). These are battle proportions, not transition shares.
E08 used rules only. Neither diagnostic showed a large measured rules gain;
+0.3 and +1.4 percentage points do not establish a clear improvement without
paired-case uncertainty. No inference about independent training seeds is made.

Their periodic validation still used the old three-category suite, not rules:

| Training run | Timesteps | Overall | Random | Dumb | Smart | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| Shared start | 1,490,944 | 67.1% | 74.2% | 65.2% | 61.9% | 0 |
| `ppo-mixed-opponents-01` | 1,544,192 | 66.4% | 74.1% | 64.5% | 60.5% | 0 |
| `ppo-mixed-opponents-01` | 1,593,344 | 66.2% | 73.7% | 64.4% | 60.5% | 1 |
| `ppo-rules-only-01` | 1,544,192 | 67.1% | 74.6% | 65.0% | 61.7% | 1 |
| `ppo-rules-only-01` | 1,593,344 | 66.9% | 74.2% | 65.2% | 61.2% | 1 |

Both runs retained their initial best. The rules-only model broadly retained
three-category performance; the mixed endpoint was slightly lower. Keep E05
best as the reference and archive both diagnostic final checkpoints.

## E09 — Combined validation (completed for reference and rules-only)

[Setup and commands](../combined-validation.md). Combine the existing 3,000
Random/Dumb/Smart cases with the existing 1,000 rules cases; retain each source's
teams, seeds and frozen identities. Each category contributes 25% to selection
for these source counts. Compare the existing candidates before starting
another training experiment. Combined validation does not add training. This is
validation already used for development, not an untouched reserved test.

User-supplied combined-suite results, hash
`eb6f690242b5ab39c1db32e71049847f2f36fee4b84577c46e5e37395ddfc337`:

| Candidate | Wins / requested | Overall (all requests) | Random | Dumb | Smart | Rules | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Diversity reference | 2,367 / 4,000 | 59.175% | 74.2% | 65.2% | 61.9% | 35.4% | 0 |
| Rules-only final | 2,374 / 4,000 | 59.350% | 74.2% | 65.2% | 61.2% | 36.8% | 1 |

Runs: `.local/run/eval-combined-diversity-01` and
`.local/run/eval-combined-rules-only-01`. This reproduces the previous category
win counts. Rules-only gains 14 rules wins, loses 7 smart wins and gains 7 wins
overall; paired-case uncertainty has not been computed. The top-level evaluator
rate excludes truncations, whereas the table above uses all requested battles.
Keep the diversity reference and diagnostic final weights.

## E10 — Rules loss inspection (user result received)

[Commands and interpretation](../rules-loss-review.md). Replay 12 seeded
loss-containing matchups from E09's diversity reference, including both seats.
Verify outcomes before reviewing legal choices, heuristic disagreements and
damage estimates. No training, reward changes or optimal-action claims.

User report: reproduction passed for 12 of 473 eligible loss-containing
matchups, including 16 originally lost battles. Those losses had 179 decisions,
31 automatic decisions, 148 choices, 91 heuristic disagreements, 22 lower
direct-damage estimates and 30 unsupported/ineffective heuristic flags. These
are loss-conditioned signals, not error rates. All displayed attacking examples
used Move 1, including five Extrasensory selections against Sableye. The early
flagged examples are not representative evidence of overall slot frequency.

## E11 — Checkpoint move-order diagnostic (results received)

[Commands](../move-order-audit.md). Sample Rules matchups without selecting on
outcome. Count all choices/probabilities, query consistent move permutations,
and compare original actions with actual simulator move announcements. Preserve
the reference model. This tests slot sensitivity and observed action wiring
before changing training or rewards.

User report: `.local/run/audit-diversity-move-order-01/report.md`, plus uploaded
`replay/move-order-audit.json`. Replay verification passed for 64 battles.
There were 841 decisions, 219 automatic decisions and 622 choices. Move 1 was
chosen in 506 of 622 choices (81.4%). Across 538 eligible positions and 1,614
cyclic permutations, 1,470 selections stayed in their original numeric slot
(91.1%); only 57 followed the original semantic action (3.5%). Mean aligned
total variation was 0.8165; zero near ties or inference recheck mismatches were
reported. These are repeated, correlated positions within sampled Rules battles,
not independent trials or estimates of all battle distributions.

In one Walrein example the policy preferred slot 1 both when it contained Rock
Blast and when it contained Spider Web, Mimic or Swords Dance. This supports a
strong position shortcut. It does not prove that move features are ignored or
that this is the only cause of weak Rules performance. The generator explicitly
puts an attack first in casual/random sets; curated sets also have fixed order.
That provides a plausible training shortcut, not a demonstrated causal result.

All nine uploaded execution mismatches were `Hi Jump Kick` versus `High Jump Kick`.
The local Showdown alias table confirms these are the same move. Canonicalizing
that name classifies the supplied announcements as 455 matches, 107 without an
announcement, and zero unexplained mismatches. This is a review of the uploaded
evidence, not a rerun of the user's checkpoint or proof of every action mapping.

## E12 — Learner move-slot shuffling (first run completed)

[Commands and design](move-slot-shuffle.md). Start a separate run from the E05
diversity reference. Match E08's rules-only opponent, teams, rewards, learning
rate and requested 100,000-step budget; add random learner move presentation.
Frozen/rule opponents keep their original inputs. Use E09 combined validation
for checkpoint selection and a final-checkpoint move-order audit for diagnosis.
No architecture, reward, team-content or observation-schema changes. The new
validation schedule changes selection relative to E08, so compare final endpoints
as well as best checkpoints. Test the intervention before increasing the budget.

User-supplied validation history from `.local/run/ppo-move-shuffle-01`:

| Evaluation | Timesteps | Overall | Random | Dumb | Smart | Rules | Truncated | Selected |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1,490,944 | 59.2% | 74.2% | 65.2% | 61.9% | 35.4% | 0 | Initial best |
| 2 | 1,544,192 | 58.4% | 74.0% | 64.6% | 59.3% | 35.8% | 0 | No |
| 3 | 1,593,344 | 55.9% | 71.3% | 62.9% | 57.2% | 32.2% | 1 | No |

The endpoint is 102,400 additional learning transitions. Relative to the start,
displayed overall win rate fell 3.3 percentage points; Random fell 2.9, Dumb 2.3,
Smart 4.7 and Rules 3.2 points. These are descriptive changes on the fixed
development suite, not significance estimates or independent training replicates.
The initial row reproduces the reference's displayed category rates. No newly
trained checkpoint beat it, so this run's `best.zip` remains the starting model.

The observed endpoint is worse in all four groups. Keep the diversity reference
and retain this final model only as an experimental endpoint. Do not extend the
run automatically. Replay the final evaluation with the move-order audit to
separate persistent slot preference from reduced preference without improved
tactics. The supplied win rates alone cannot distinguish these explanations or
establish that randomized presentation cannot work. The retained final evaluation
at `evaluations/evaluation-000003` can be audited directly; no repeat of all
4,000 validation battles is needed.

The user then supplied `.local/run/audit-move-shuffle-final-01/report.md`.
Replay verification passed for all 64 battles, with 22 wins, 42 losses and no
truncations in this diagnostic sample. That sample's 34.4% win rate does not
replace the full Rules validation result of 32.2%.

| Diagnostic | Diversity reference | Shuffle endpoint |
|---|---:|---:|
| Decisions / nonautomatic choices | 841 / 622 | 782 / 586 |
| Move 1 among nonautomatic choices | 506/622 (81.4%) | 428/586 (73.0%) |
| Keeps same numeric move slot after permutation | 1470/1614 (91.1%) | 1130/1440 (78.5%) |
| Follows original semantic action | 57/1614 (3.5%) | 129/1446 (8.9%) |
| Mean aligned total variation | 0.8165 | 0.4232 |
| Mean Move 1 probability, all decisions | 0.600 | 0.366 |
| Near ties / inference recheck mismatches | 0 / 0 | 0 / 0 |

Slot sensitivity decreased in these sampled trajectories, but remains large;
the full evaluation showed worse play. Total variation can decrease simply
because probabilities become flatter, so the reduction is not sufficient
evidence of stronger move-identity reasoning. Semantic agreement includes
unchanged switch choices; its denominator differs from the same-slot rate.
Both audits used 32 matchups and seed 261002, but the policies visited different
positions, so these are not measurements on identical states or independent
training replicates.

The endpoint report has 430 matching move announcements, 87 without an
announcement and 5 execution differences. The user subsequently supplied all
five detail records:

| Replay episode | Decisions | Stored selected name | Announced |
|---:|---|---|---|
| 52 | 14, 15, 16 | Rock Slide | Struggle |
| 59 | 19, 20 | Earthquake | Struggle |

The adapter maps a native Struggle request to action 0 while the observation
retains the first stored move's name. The audit previously compared that stored
name against the announcement. This explains how a legitimate fallback can
produce these flags, especially when healthy bench members keep switching legal.
All five records are consistent with that case; the supplied records do not
include pre-action requests or PP/restriction evidence, so no specific cause is
claimed for these five turns and the historical counts are not rewritten.

Release 0.42.2 captures the native request and queued action only during this
audit, and classifies `execution_struggle_fallback` only when both specify
Struggle for the submitted action. Unexpected or unverified substitutions still
require review. Controlled simulator fixtures cover PP exhaustion and Disable,
healthy benches, both player seats, and unchanged protocol/state/RNG with
diagnostics enabled. No training, policy or reward behavior changed.

Confirmed Struggle fallbacks cannot enter the permutation probes because they
have fewer than two legal move actions. They do not account for the strong
same-slot result. Keep the diversity reference; this diagnostic clarification
does not reverse E12's worse validation or justify extending the run automatically.

## E13 — Shared move scoring (implementation ready; training not run)

E12 reduced measured slot sensitivity while validation deteriorated. The next
bounded experiment changes the architecture so that moving a move to a new slot
moves its score with it. Battle context and value predictions pool move features
without using move positions; party positions retain their original meaning.

Compare two **fresh** v2 policies: the existing `flat` architecture and the new
`shared-moves` architecture. Both use the existing diversity training teams,
rules-only opponents, saved rewards, learning rate 0.0001, rollout 128, batch 128,
four epochs, entropy 0.01 and seed 261003. With 64 environments, the requested
200,000-transition budget rounds to 200,704 per model. Neither resumes the
diversity checkpoint, which stays archived as a practical reference.

Use the same combined four-group validation suite, initially, every rounded
50,000 transitions and at completion. Compare both endpoints and validation-best
models. Parameter counts and architecture version are recorded in configuration;
this is a comparison of these implementations, not a parameter-count-matched
ablation. A single seed is a screening experiment. A promising result needs
replication and fresh held-out evaluation before a generalization claim.

No learning results are available yet. Controlled checks cover all 24 move
orders, one to four populated move slots, legal masks, both parties' moves,
value predictions, optimizer updates, checkpoint loading, frozen opponents and
short real-simulator training/validation/continuation. These verify the intended
structure and execution, not tactical skill.

Commands and interpretation: [shared move-scoring experiment](shared-move-policy.md).

For fixed-to-varied, gated two-job experiments with live logs, memory sampling, and
collapse diagnostics, see [Staged team lab](staged-team-lab.md).

For the expanded fixed-to-1,024-team progression, editable teams, rehearsal, and
team inspection commands, see [Expanding team curriculum](team-curriculum.md).

For a stage-4 transfer failure after successful mirror retention, see
[type-transfer-lab.md](type-transfer-lab.md) for the two-job specialist-versus-fresh
comparison, exact team inspection, live logs, memory monitoring, and continuation.
