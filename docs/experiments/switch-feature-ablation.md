# Targeted replacement training for student play

## Completed result: October 19, 2026

Both arms completed 500,736 new steps in about 27 minutes. The whole run took
43.6 minutes and left at least 20.88 GiB available RAM with zero swap use.
All 25 saved checks are complete: 20,000 new battles plus 5,000 imported baseline
battles, zero truncations, zero Rules overrides, and no immune selections among
213,535 classified moves. This move total includes repeated policies and pools.

| Policy | Balanced validation | Seen mean | Heldout mean | Fixed mirror |
|---|---:|---:|---:|---:|
| Starting model / both selected bests | 45.21% | 44.15% | 37.40% | 62.6% |
| Control final | 37.79% | 49.30% | 40.30% | 62.6% |
| Replacement-feature final | 41.80% | 50.50% | 43.65% | 61.1% |

Seen and heldout means weight original and reversed bench orders equally. The
fixed-final heldout difference is +3.35 percentage points, with a paired 95%
bootstrap interval of +1.55 to +5.15 points (20,000 resamples, seed 261121).
Pairs match episode identity, not completion order; each of the 500 resampled
matchup/seed groups retains both seats and both bench orders. This measures
evaluation uncertainty on reused development teams, not training-seed variation.
The primary validation-selected comparison is a tie: neither arm improved on
its initial validation, so `models/selected.zip` retains the earlier control.

The mechanism is not established. Feature-final chose the lowest legal forced
replacement slot more often than control (76.7% versus 53.9% on heldout), while
voluntary switching was lower (2.79% versus 5.73%). The gain cannot be attributed
solely to better forced replacements. Retained traces show earlier ordinary
action differences as well.

The small training pool is imbalanced: Vaporeon appears in 2/16 teams, Venusaur
1/16 and Dodrio 1/16, versus 5/8, 3/8 and 4/8 validation teams. This motivates
the [next coverage comparison](training-coverage.md), using uniform coverage
rather than tailoring training frequencies to those eight validation teams.
The numerical record is `switch-feature-01-analysis.json`. Uploaded model bytes
were absent; the next runner verifies model hashes on the user's machine.

Save both useful models permanently from the project root:

```bash
mkdir -p models
cp -n .local/run/switch-feature-ablation-01/models/selected.zip \
  models/switch-feature-ablation-01-selected.zip
cp -n .local/run/switch-feature-ablation-01/models/switch-features-final.zip \
  models/switch-feature-ablation-01-feature-final.zip
```

The first is the validation-selected fallback; the second is the promising
development candidate, not a validation winner. `-n` protects existing copies.

## Original design

The goal is to improve the playable agent while preserving a working fallback
for October 20. Two matched jobs test whether a learned scorer benefits from
direct information about each replacement Pokémon. This follows the bench-order
result: changing slot preferences alone did not establish a useful win-rate gain.

## What changes

| Setting | Both jobs |
|---|---|
| Starting model | Validation-selected bench-order control best, 852,992 steps in the real source run |
| New architecture | `matchup-switches`; original parameters retained, plus 3,328 shared residual-scoring parameters |
| Initial behavior | Same logits, legal-action probabilities and values as the source model |
| Optimizer | Existing Adam state preserved; added parameters start with empty state |
| Budget | 500,000 added learning transitions per arm, rounded to complete rollouts |
| Training seed | 261120 |
| Opponent | Rules |
| Training teams | Same 16 current three-Pokémon teams; no bench shuffling |
| Rehearsal | 90% current task, 10% fixed mirror |
| PPO | Learning rate 0.0001, entropy 0.01, 128 rollout steps, batch 128, four epochs |
| Parallelism | Two jobs maximum; 16 environment players per job, eight learning |
| Validation | Existing equal original/reversed bench-order suite; every 50,000 added transitions |
| Selection | Highest validation wins/requested; earlier checkpoint retains ties; control retains cross-arm ties |

`control` supplies zeros to the new candidate-feature inputs.
`switch-features` supplies 18 observation-derived values per candidate. The
same shared scorer adds a residual to each original switch score only during
forced replacement with at least two legal choices. Its output layer starts at
zero. Neither arm asks Rules to choose its actions during inference.

The features are defined in `FEATURE_NAMES` in
`src/firered_rl/matchup_switch_policy.py`:

- Health fraction, maximum HP, level, five stats and status presence.
- Outgoing assessed-move fraction, maximum matchup multiplier, a power proxy
  and a stat-adjusted power proxy against the active opponent.
- Incoming assessed-move fraction, maximum matchup multiplier and stat-adjusted
  power proxy for the active opponent's observed moves against that candidate.
- Relative speed and the active opponent's health fraction.

The proxies use power, nominal accuracy, Same-Type Attack Bonus, type/known
ability immunities and relevant attack/defense ratios. They are comparisons,
not simulator damage or KO predictions. Weather, held-item effects, offensive
abilities, status stat penalties, accuracy/evasion stages, priority and special
move mechanics are outside those proxies. Unsupported moves have an assessed
indicator of zero. No future opponent action is supplied.

The added scorer has no party-slot index and its features follow the Pokémon
when slots move. The inherited dense switch pathway still sees ordered party
information, so the complete policy is not permutation invariant. PPO updates
the whole policy in both arms: ordinary move choices can also change during
training even though the new residual applies only to forced replacements.

## Run now

Extract the update over the existing project source. Keep the completed source
run, checkpoints, the original 400k reference and installed environment intact.
From the project root:

```bash
source .venv/bin/activate
python run_switch_feature_ablation.py \
  --source-run .local/run/bench-order-ablation-01 \
  --output-dir .local/run/switch-feature-ablation-01 \
  --jobs 2
```

This launches training and then automatically checks best/final checkpoints on
seen teams, reused heldout teams in both bench orders, and the fixed mirror.
Allow roughly an hour based on the prior full run, but use the live logs to
judge actual progress. There is no early stop based on a promising test score.
No existing model is overwritten or automatically promoted into another run.

In another terminal, watch the job output:

```bash
tail -n 20 -F .local/run/switch-feature-ablation-01/console/*.log
```

Watch validation after the job directories appear:

```bash
watch -n 5 'for f in .local/run/switch-feature-ablation-01/runs/*/evaluations.md; do [ -f "$f" ] && { echo "$f"; cat "$f"; }; done'
```

Watch memory and the current phase in separate terminals:

```bash
watch -n 2 free -h
watch -n 5 cat .local/run/switch-feature-ablation-01/status.json
```

`memory.csv` records machine-wide RAM and swap, including validation peaks.
Each training run also has `updates.jsonl`, PPO CSV metrics, `summary.json` and
its own checkpoints. Frozen-check progress prints in the launch terminal and
is saved under `checks/`.

## Pause and continue

Press Ctrl+C once in the launch terminal and allow both workers to save and
exit. Resume the existing bounded budget with:

```bash
python run_switch_feature_ablation.py \
  --output-dir .local/run/switch-feature-ablation-01 \
  --continue --jobs 2
```

Use the same log, validation and memory viewers above. Do not add a new
`--timesteps` on continuation, change saved inputs, or overwrite partial check
summaries. Strict gates preserve and retry incomplete checks. A completed
continuation reuses accepted training/check results.

## Inspect the actual teams

```bash
python run_switch_feature_ablation.py \
  --output-dir .local/run/switch-feature-ablation-01 \
  --inspect-teams train --limit 16 --full-json
```

Use `--inspect-teams mirror`, `validation`, `heldout` or `all` for other pools.
Training teams are copied under `inputs/teams/train/`; rehearsal is under
`inputs/teams/mirror/`. `inputs/mixture.json` gives their sampling weights.
`inputs/configuration.json` and `plan.json` record the frozen settings and hashes.
Both jobs use these same inputs. Stage expansion is deliberately deferred until
this specific replacement change has been measured.

## Read the result

```bash
cat .local/run/switch-feature-ablation-01/report.md
```

All scores are wins against Rules; draws and truncated battles count as nonwins.
The imported start copies five complete source-best checks after conversion
identity is verified. It is existing evidence, not a new evaluation.
The runner evaluates each arm's validation-best and fixed-final checkpoint in
five conditions, up to 20,000 new battles; identical checkpoint hashes share checks.

Validation alone selects `models/selected.zip`. Heldout scores describe the
result, never select the model. Heldout here means excluded from training but
reused during development, not an untouched final test. Compare wins and
replacement behavior together. Lower lowest-slot frequency or agreement with
Rules is not by itself evidence of better decisions. Check mirror retention
against this run's source score, not an older specialist's unrelated peak.

A clear increase in validation and consistent seen/heldout gains would support
the added representation. A small or inconsistent difference is inconclusive
from one seed pair; do not promise a 50% win rate or better human play.
Both final checkpoints remain available even if their validation was lower.

Upload `.local/run/switch-feature-ablation-01/switch-feature-results.zip` for
review. It contains metrics, configuration, teams and bounded decision evidence;
model weights remain on your machine. The source run stays untouched.

## Play while training, then try the selected result

You can rehearse now using the existing source model:

```bash
python mgba/play_gui.py \
  --rom "$PWD/FireRed.gba" --source "$PWD/FireRed.sav" \
  --checkpoint .local/run/bench-order-ablation-01/runs/control/checkpoints/best.zip \
  --workers 2 --speed 1
```

After the new run finishes, its `models/` contains `selected.zip`,
`control-best.zip`, `control-final.zip`, `switch-features-best.zip` and
`switch-features-final.zip`, with hashes in `models/manifest.json`.
Make the familiar teams visible in the GUI:

```bash
mkdir -p config/teams/student-play-20261019
cp -n .local/run/switch-feature-ablation-01/inputs/teams/train/*.json \
  config/teams/student-play-20261019/
python mgba/play_gui.py \
  --rom "$PWD/FireRed.gba" --source "$PWD/FireRed.sav" \
  --checkpoint .local/run/switch-feature-ablation-01/models/selected.zip \
  --workers 2 --speed 1
```

Open `http://127.0.0.1:8080`, select teams from
`teams/student-play-20261019`, and initially leave AI action sampling off.
Keep the three members and two moves to demonstrate the actual task family.
The human plays in the Player 1 mGBA window; the browser configures the match.

Play two complete games, including one after restarting the GUI. Ask students
to predict an action, play a counter, then discuss why the learned policy might
succeed or fail. Custom teams are useful challenges, but they test beyond the
restricted training setup. Record one working battle as a fallback.
Keep an eye on memory if training and the GUI run together.

The local verification covers policy conversion, training, frozen checks,
serialization and continuation with tiny Showdown fixtures. Actual learning
gains and the ROM/save/display workflow must be tested on the event machine.
See `switch-feature-verification.json` for the verified scope.
