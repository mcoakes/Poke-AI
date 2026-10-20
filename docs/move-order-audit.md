# Check a checkpoint for move-slot preference

[Loss review](rules-loss-review.md) · [Utility flags](reference/audit-move-order.md)

A loss report showing repeated Move 1 choices is a reason to test the policy,
not proof of broken action wiring. This diagnostic separates choices of move
identity from choices of slot position. It uses the exact saved checkpoint and
an outcome-unfiltered sample of Rules matchups: both seats, any wins, losses,
draws and truncations present in the selected cases.

```bash
python mgba/audit_move_order.py \
  --evaluation-dir .local/run/eval-combined-diversity-01 \
  --output-dir .local/run/audit-diversity-move-order-01 \
  --matchups 32 --seed 261002 \
  --environments 2 --showdown-workers process \
  --log-max-mb 2 --log-backups 1

cat .local/run/audit-diversity-move-order-01/report.md
```

Run from the project root with Python training dependencies and Showdown installed.
No training or model copying command is needed: the diversity reference already
exists in `models/observation-v2-diversity-best-01.zip`. Source evaluation inputs
and checkpoint hashes are checked. The new output contains its own immutable
input snapshot. Existing output directories are rejected.

## What it measures

Every candidate decision in the replay contributes action counts and masked
probabilities for all 11 actions. Single-legal-action decisions are separated
from meaningful choices. Switch counts in this report can include replacements;
use the loss review for voluntary-switch rates.

For nonreplacement positions with at least two legal moves and no duplicate
move IDs, three cyclic permutations move every move to every other slot. Move
IDs, types, current/max PP, v2 power/accuracy/category/priority/type details and
the first four legal-mask bits move together in the active party row. Other
party rows, opponent data and switch actions stay unchanged. Empty and disabled
slots move with their data and legality. The policy is queried on these alternate
observations; the alternate actions are **never executed**.

| Metric | Meaning |
|---|---|
| `follows_original_action / permutations` | Same semantic move after rearrangement, or same switch/wait action |
| `stays_in_original_slot / original_move_permutations` | Originally chose a move and keeps that numeric slot despite its contents moving |
| Mean aligned total variation | Probability change after translating alternate slots back to original moves; 0 means identical distribution, 1 means completely disjoint distributions |
| `near_tie_permutations` | Original or permuted top two action probabilities differ by less than 1e-7; argmax changes can reflect ties |
| `inference_recheck_mismatch` | Individual probability argmax differs from the original batched action; investigate numerical ties or inference before interpreting |

These are sensitivity measurements, not tactical accuracy scores. A slot bias
can be a learned policy property even when simulator wiring is correct. High
semantic agreement can still mean choosing a poor move consistently. Source
sample selection is independent of outcome, but limited to Rules opponents and
this suite's teams; it does not describe all opponents or training trajectories.
Lower aligned total variation can also reflect flatter probabilities. Compare
semantic choices and battle results as well; audits of different checkpoints
can visit different positions even when their matchup schedules match.

## Execution verification

Original multi-choice move selections are compared with Showdown's first move
announcement for that candidate player after the paired step:

The comparison treats Gen III's **Hi Jump Kick** and Showdown's **High Jump Kick**
as the same move. The nine apparent mismatches in the diversity reference audit
were all this naming difference; they did not identify incorrect execution.

- `execution_matched`: the announced move matches the selected move.
- `execution_struggle_fallback`: the submitted action's native request and
  queued move both specified Struggle before execution, and Struggle was
  announced. The adapter uses action 0 for this fallback even though the stored
  move row still names the original first move. Switching may also be legal,
  so this is not necessarily a single-action decision.
- `execution_not_observed`: no move announcement, possibly due to sleep,
  flinching, fainting before acting, or another interruption.
- `execution_review_mismatch`: a different move was announced; inspect the
  protocol/context for forced moves, an unverified Struggle substitution or
  called-move behavior before calling it a mapping bug.

Move-order audits capture each submitted action, simulator command, requested
move ID and queued move ID before execution in `submitted_choice`. This evidence
is only requested for this diagnostic; ordinary training/evaluation responses,
policy inputs, rewards and action mapping are unchanged. An announcement saying
Struggle alone is not enough to mark a verified fallback. Older saved records
without this evidence remain unverified, rather than being silently rewritten.

Action-frequency counts describe numeric buttons, so Move 1 counts can include
Struggle selections. A confirmed Struggle request has only one move action and
is excluded from permutation probes, which require at least two legal moves.

The report includes up to 12 retained mismatch examples with replay episode,
decision, selected move and announced move names. These remain review flags;
name differences and battle mechanics can explain them. For an existing report
that only shows the count, inspect its already-saved JSON without rerunning:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('.local/run/audit-move-shuffle-final-01/replay/move-order-audit.json')
data = json.loads(path.read_text())
rows = [r for r in data.get('examples', [])
        if r.get('execution') == 'review_mismatch']
print('Total flagged:', data['counts'].get('execution_review_mismatch', 0))
print('Retained examples:', len(rows))
print(json.dumps(rows, indent=2))
PY
```

Single-legal-action/continuation decisions and switch execution are outside
this check. Outcomes, truncations and decision counts must also match the
original evaluation. This confirms the reported replay properties, not every
internal state or every possible mechanic. Diagnostic inference adds overhead;
ignore this run's throughput for performance comparisons.

## Files to share

Send **`report.md`** and **`replay/move-order-audit.json`**. The JSON includes
all per-episode counters, mean action probabilities and bounded examples with
the original/permuted distributions. `selection.json` maps replay episodes to
source cases. `verification.json` records reproduction and hashes.

`replay/move-order-decisions.jsonl` plus rotated backups contains detailed
probabilities and execution checks. `replay/pair-*/battle.log` supplies battle
context. Logs are bounded per file; aggregate counters survive rotation. Model
weights, action masks, rewards and training data are not changed.

If the audit identifies a strong slot preference, the next controlled option is
[learner move-slot shuffling](experiments/move-slot-shuffle.md). A lower slot-bias
metric is useful evidence about the intervention, but battle evaluations must
still establish whether the policy improved.
