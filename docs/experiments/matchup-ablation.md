# Matchup-feature experiment

This experiment has completed. Its results and the next frozen-policy tests are
in [replacement-diagnostic.md](replacement-diagnostic.md), with the numerical
analysis in `matchup-01-analysis.json`. The remaining text records its design.

The million-step initialization comparison produced a useful negative result:
fresh training did not escape the broad-team learning problem. All scores below
are deterministic policy wins against Rules, with draws counting as nonwins.

| Validation-selected checkpoint | Validation | Seen teams | Heldout teams | Fixed mirror |
|---|---:|---:|---:|---:|
| Specialist best | 27.9% | 38.8% | 27.1% | 99.9% |
| Fresh best | 26.2% | 40.4% | 31.0% | 62.7% |

The same heldout schedule gave greedy 43.8% and Rules 49.7%. All checks had zero
truncations. Each training arm added 1,000,448 steps and took about 41 minutes.
Fresh improved from 133/512 validation wins at 200,704 steps to 134/512 at
1,000,448. This is a practical plateau, not proof of mathematical convergence.
Validation selected the specialist; the higher fresh heldout score does not
change that selection. These are reused development checks, not a new final
generalization estimate. The stage-10 benchmark remains untouched.

Specialist best selected currently immune attacks on 413/9,182 assessed moves
(4.5%); fresh best did so on 521/8,995 (5.8%). Each had a legal assessed nonimmune
alternative. These measurements concern the current defender before the opponent
acts. They are not estimates of wins recoverable by changing those moves.
Both policies also lost hundreds of battles without making an immunity selection.

## What changes

Two fresh policies start from the same saved, untrained initialization in the
transfer run. Both use the new `matchup-moves` architecture with the same parameter
count. In `control`, the new matchup inputs are zero. In `matchup`, those inputs
describe assessed type effectiveness, supported ability immunities, and STAB
for each move against the observed active defender. Unsupported interactions
remain explicitly unassessed.

New input weights start at zero. Initial action probabilities and value
predictions must match the original policy and each other. Both optimizers are
empty and both timestep counters start at zero. This isolates the feature inputs
without introducing a different random starting policy or larger control budget.
The source checkpoints and original 400k reference are preserved.

These features supply known game mechanics. If they help, that demonstrates the
benefit of this representation; it does not show that RL discovered the type
chart independently. Every legal action remains available and the policy still
learns which action to select.

The teams, Rules opponent, reward, PPO settings, and training seed are identical
between arms. Keep LR 0.0001, entropy coefficient 0.01, and the same maximum
one-million-step budget and validation schedule. The next question is whether
matchup inputs improve both decisions and battle wins. Reward changes or a new
training task should be separate experiments. One seed remains diagnostic;
replicate a substantial improvement before expanding the curriculum.

## Rehearsal and retention

Both arms retain 90% current stage-4 team sampling and 10% original fixed-mirror
sampling. Both sides of each battle come from the chosen group. The report
checks fixed-mirror performance separately at initialization, best checkpoint,
and final checkpoint, alongside seen-team and heldout-team performance.

The previous specialist kept 99.9% mirror wins while receiving 10.01% mirror
episodes. That establishes observed retention, not that rehearsal caused it:
there was no comparison with rehearsal disabled. Fresh mirror performance rose
from 0.2% to 62.7%; it was learning that task for the first time.

For later curriculum stages, the existing curriculum runner samples current,
previous-stage, and fixed-mirror tasks and runs separate retention checks. This
ablation does not advance that curriculum or overwrite its failed gates.

## Run and monitor

Extract the updated project archive over the source files. From the project root:

```bash
source .venv/bin/activate
python run_matchup_ablation.py \
  --source-run .local/run/type-transfer-01 \
  --output-dir .local/run/matchup-ablation-01 \
  --jobs 2
```

Use the transfer directory that produced the uploaded results. No manual model
copying is needed. Initial frozen checks run before training; their progress is
in the launcher terminal. Keep total training concurrency at two. The previous
run had at least 21.79 GiB RAM available and no swap use, but this is a measurement,
not a memory guarantee. `memory.csv` records this run's RAM and swap.

Watch both jobs in another terminal:

```bash
tail -n 20 -F .local/run/matchup-ablation-01/console/*.log
```

Watch validation histories:

```bash
watch -n 5 'for f in .local/run/matchup-ablation-01/runs/*/evaluations.md; do
  [ -f "$f" ] || continue
  echo "$f"
  cat "$f"
done'
```

Watch memory in a separate terminal:

```bash
watch -n 2 free -h
```

Inspect the actual teams and their moves:

```bash
python run_matchup_ablation.py \
  --output-dir .local/run/matchup-ablation-01 \
  --inspect-teams train --limit 3 --full-json
```

Use `mirror`, `validation`, or `heldout` for another group. The pinned team files
are under `inputs/teams/`, the episode weights are in `inputs/mixture.json`, and
the frozen schedules contain the evaluation teams. Inspection is read-only.

Ctrl+C once in the launcher pauses gracefully. Wait for trainers to save and
workers to stop. Resume remaining budgets with:

```bash
python run_matchup_ablation.py \
  --output-dir .local/run/matchup-ablation-01 --continue --jobs 2
```

Use the same viewers after resuming:

```bash
tail -n 20 -F .local/run/matchup-ablation-01/console/*.log
```

```bash
watch -n 2 free -h
```

Upload `.local/run/matchup-ablation-01/matchup-results.zip` when it finishes.
This contains the report, initial-equivalence evidence, frozen checks, histories,
and decision audits. It excludes model weights and bulk console logs.

`transfer-01-analysis.json` records the uploaded million-step results.
`matchup-ablation-verification.json` records implementation verification. Small
local smoke tests verify operation; they do not predict learning performance
for the user's saved models.

## Verification

97 focused tests passed; two CUDA-only tests were skipped because this test
environment has no GPU. The checks include 1,610 synthetic battle observations
compared with the independent decision audit, initial policy equivalence,
serialization, legal-action preservation, and move-order symmetry.

A real simulator smoke trained both arms for 256 steps and verified 24 accepted
frozen checks (192 battles, zero truncations). It also checked source preservation,
logs, memory recording, team inspection, and continuing a completed run without
additional training or evaluations. Saved checks with incomplete evidence were
preserved and retried through continuation. The launcher now verifies every
referenced summary and episode log before reporting completion; if that check
fails, use the continuation command above after inspecting the reported error.
