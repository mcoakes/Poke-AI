# Inspect losses against Rules

[Rule opponent](rule-based-opponent.md) · [Combined validation](combined-validation.md) ·
[All flags](reference/review-rule-losses.md)

Use this diagnostic after evaluation shows weak performance against Rules. It
replays a seeded sample of loss-containing matchups and summarizes candidate
choices. It does not train, alter rewards, change actions or select a new best
checkpoint. The reference model remains archived under `models/`.

```bash
python mgba/review_rule_losses.py \
  --evaluation-dir .local/run/eval-combined-diversity-01 \
  --output-dir .local/run/review-diversity-rules-01 \
  --matchups 12 --seed 261001 \
  --environments 2 --showdown-workers process \
  --log-max-mb 2 --log-backups 1

cat .local/run/review-diversity-rules-01/review.md
```

Use the installed Python training environment and local Showdown dependencies.
The default is 12 seat pairs, at most 24 battles; a smaller eligible pool is
used in full. Two environments keep the battles in a single sequential pair,
which makes log reading easier. Higher even counts work too.

## Selection and replay

The source must contain `suite.json`, `config.json`, `episodes.jsonl` and its
preserved `candidate.zip` (or an accessible matching recorded checkpoint).
The candidate must use observation v2. The tool checks the source suite and
checkpoint hashes and requires deterministic, side-swapped Showdown evaluation.
Only Rules matchups with both original results available and at least one
nontruncated candidate loss are eligible. Both candidate seats are replayed,
including a winning, drawn or truncated partner case when present.

`--seed` changes which eligible matchups are sampled. Original battle seeds,
teams, opponent identity and candidate weights remain pinned. Episode and
matchup numbers are renumbered; `selection.json` maps back to source episodes.
Source files are unchanged. The output directory must be new.

The evaluator checks the current rule implementation against the saved hash.
The report verifies outcome, truncation and decision count against each source
case. A mismatch returns a failing exit status and is clearly flagged: inspect
that discrepancy before using the decision analysis. Matching those fields is
not proof of identical internal trajectories. Simulator/version or numerical
changes can affect replay; this is why the check is explicit.

## What to read

| File | Purpose |
|---|---|
| `review.md` | Short report: reproduction, counts and selected examples |
| `review.json` | Structured report for sharing or further analysis |
| `selection.json` | Source case mapping, sampling seed and candidate/suite hashes |
| `replay/decision-audit.json` | All per-case decision counters, rule identity and up to 40 early flagged examples |
| `replay/candidate-decisions.jsonl` and backups | Candidate choices, legal alternatives, heuristic scores/reasons, HP and status bits |
| `replay/rule-decisions.jsonl` and backups | Actual opposing rule choices |
| `replay/pair-*/battle.log` and `battle-readable.log` | Spectator text and detailed observation/action context |

Send `review.md` and `review.json` first. They are sufficient to identify which
cases merit examining in the detailed logs. Traces use the chosen per-file cap
and backup count. Counts in the JSON report include all decisions even when
old trace entries rotate out. Inputs include one checkpoint snapshot, and the
replay evaluator preserves its own copy; this is a small bounded diagnostic,
not something to run automatically after every PPO update.

## Interpretation

- **Heuristic disagreement:** the model chose a different legal action from
  the rules policy when given the model's own observation. This is a lead for
  review, not an error label or proof that the rules choice wins.
- **Lower direct damage estimate:** another legal move's approximate expected
  direct damage was at least 25% of enemy maximum HP higher. A status move,
  healing or setup can intentionally trade immediate damage for future value.
- **Unsupported or ineffective by heuristic:** the rule scorer could not give
  the selected move a supported useful effect in that position. These two cases
  are deliberately not reported as definitively bad actions.
- **Voluntary switches:** switches with more than one legal action and no
  forced-replacement flag. Replacement and single-legal-action decisions are
  counted separately; a high or low switch rate is not inherently good.

The heuristic is the same approximate `gen3-rules-v1`, not an optimal planner or
complete damage oracle. Estimated damage ignores some mechanics and does not
prove a guaranteed knockout. No alternative actions are executed, so the tool
cannot show what would have happened after a different decision. The sample is
conditioned on losses and cannot estimate overall error frequency or win rate.
Examples are early flagged decisions, not a ranking of the worst mistakes.

Use these signals to choose concrete battle cases to investigate. Do not add a
switch bonus, change rewards, or imitate every rule choice based only on the
aggregate disagreement count.
