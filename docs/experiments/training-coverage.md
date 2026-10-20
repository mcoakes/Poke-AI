# Broader, balanced training for student play

## Completed result: training-coverage-01

The expanded pool was used as configured, but did not establish stronger
general play. The eight-composition validation suite selected the expanded-pool
model while the reused seen and heldout checks favored the small-pool models.

| Model | Validation | Seen mean | Heldout mean | Fixed mirror |
|---|---:|---:|---:|---:|
| Starting feature-final | 41.8% | 50.5% | 43.65% | 61.1% |
| Small-pool best | 48.0% | 49.9% | 42.55% | 61.0% |
| Small-pool final | 34.8% | 50.2% | 43.35% | 62.7% |
| Expanded-pool best/final | 48.7% | 42.7% | 39.50% | 61.6% |

All wins are against Rules. Seen/heldout means weight the two bench orders
equally, with both player seats represented. Expanded minus small best heldout
difference is -3.05 percentage points; a paired 20,000-resample bootstrap gives
a 95% interval of -4.80 to -1.35 points. It resamples 500 matchup/seed groups,
keeping both seats and orders together. It measures evaluation variation on
these reused pools, not independent training-seed or new-team uncertainty.

The upload contains 20 unique complete frozen checks: 20,000 battles, no
truncations. Five starting checks are imported evidence; 15,000 battles were
new. All available 130 input pins and 18 implementation hashes matched. Weight
ZIPs were omitted, so the real policy weights were not inspected here. The
two jobs added 500,736 transitions each; the run took 39.0 minutes, retained at
least 21.85 GiB available RAM and used no swap.

The expanded model used voluntary switches less often on heldout teams
(1.18% versus 2.65% for small best) and chose the lowest eligible replacement
slot more often (90.59% versus 77.58%). In 33 of 64 bounded non-mirror traces,
its first divergence from the starting policy was an ordinary move or voluntary
switch; none first diverged at a forced replacement. These observations do not
prove a single cause of the regression. All 171,702 assessed move actions
avoided a currently immune target within the audit's supported scope.

The next step is the [teacher warmup comparison](teacher-training.md), which
adds a direct learning signal and replaces the narrow selector with a larger
predeclared validation pool. Existing candidates remain available as fallbacks.
Save these useful completed models now:

```bash
mkdir -p models
cp -n .local/run/training-coverage-01/models/selected.zip \
  models/training-coverage-01-selected.zip
cp -n .local/run/training-coverage-01/models/small-pool-best.zip \
  models/training-coverage-01-small-pool-best.zip
cp -n .local/run/training-coverage-01/inputs/start.zip \
  models/training-coverage-01-starting-feature-final.zip
```

These preserve distinct useful candidates. The selected and expanded-final
exports in this completed run have identical weights, so saving both adds no
new policy. The original experimental protocol follows.

The replacement-feature final model won 43.65% on reused heldout teams versus
40.30% for its matched control, but neither arm beat the source validation
score. The 16 training teams also have uneven species exposure. This experiment
tests a larger training pool before another architecture or reward change.

Both jobs start from the same feature-enabled fixed-final checkpoint at
1,353,728 total steps, preserving its optimizer state. This is an explicitly
developmental continuation of a promising final model. The previous
validation-selected checkpoint remains a separately preserved fallback.

## What the two jobs test

| Setting | `small-pool` | `balanced-pool` |
|---|---|---|
| Original 16 training teams | 90% of episode starts | 25% |
| New 96 training teams | 0% | 65% |
| Fixed mirror rehearsal | 10% | 10% |
| Added training | 500,000 learning transitions, rollout-rounded | Same |
| PPO settings, reward, seed | Unchanged settings; seed 261130 | Same |
| Architecture | Feature-enabled `matchup-switches` | Same |
| Ordering | Fixed saved team order; no runtime bench shuffle | Same |
| Validation and frozen checks | Original suites | Same |

The new pool has 96 unique three-species compositions from the same 12-species
family, using the existing level-50 simple builds and two damaging moves.
Each species appears **24 times overall and eight times in each party position**.
Teams keep their saved order during training. This changes training coverage
and its sampling mixture; it does not isolate species balancing from pool size.

The runner excludes the unordered species compositions and semantic team
signatures of all original train, mirror, validation, seen and heldout inputs.
Reversed bench variants are therefore excluded too. Generation fails if it
cannot produce the requested unique balanced pool. Validation and heldout
teams are never introduced into training. Known development results motivate
uniform coverage, not a hand-tuned match to the validation frequencies.

The treatment uses 112 non-mirror teams in total: 16 retained plus 96 new.
Sampling occurs by group, so original teams get more practice per team than
the new teams. Both players' current-task teams come from the sampled group;
cross-group pairings are not added in this experiment. Mirror remains a fixed
pair. Sampling percentages refer to episode starts, not a guaranteed fraction
of transitions, because battle lengths differ.

## Run and monitor

Extract the updated archive over the existing project source. Retain the
completed `switch-feature-ablation-01` directory and its checkpoints. From the
project root:

```bash
source .venv/bin/activate
python run_training_coverage.py \
  --source-run .local/run/switch-feature-ablation-01 \
  --output-dir .local/run/training-coverage-01 \
  --jobs 2
```

Allow roughly an hour based on the preceding run's 44 minutes, with additional
time possible for the broader pool. Both jobs use 16 environment players, eight
learning, and one PyTorch thread. Validation runs every 50,000 added transitions.
After training, the runner automatically checks best/final models on seen teams,
heldout teams in both bench orders, and fixed mirror retention. No additional
prompt or manual model copying is needed to start those checks.

In another terminal, watch job output:

```bash
tail -n 20 -F .local/run/training-coverage-01/console/*.log
```

Watch validation after the job directories appear:

```bash
watch -n 5 'for f in .local/run/training-coverage-01/runs/*/evaluations.md; do [ -f "$f" ] && { echo "$f"; cat "$f"; }; done'
```

Watch memory and the saved phase in separate terminals:

```bash
watch -n 2 free -h
watch -n 5 cat .local/run/training-coverage-01/status.json
```

`memory.csv` records machine-wide available RAM and swap, including validation
peaks. Each job has `updates.jsonl`, PPO CSV metrics and checkpoints. Frozen
check progress prints in the launch terminal and saved results live in `checks/`.
The previous two-job run used no swap and retained at least 20.88 GiB available.

## Inspect teams and sampling

```bash
python run_training_coverage.py \
  --output-dir .local/run/training-coverage-01 \
  --inspect-teams new --limit 3 --full-json
```

Use `old`, `mirror` or `all` instead of `new` for other training groups. Increase
`--limit` to 96 to print the whole new pool. The saved input files include the
team definitions, balance counts, generation recipe, exclusions and each job's
sampling mixture. `plan.json` pins their hashes. These are the actual inputs
used by the workers, not a separately generated illustration.

## Pause and continue

Press Ctrl+C once and let the launcher wait for both workers to save and exit.
Resume the remaining saved budget with:

```bash
python run_training_coverage.py \
  --output-dir .local/run/training-coverage-01 \
  --continue --jobs 2
```

Use the same log, validation and memory viewers above. Do not change saved
inputs or specify another budget on continuation. Incomplete frozen checks are
preserved and retried through the normal strict continuation path.

## Interpret the results

The primary comparison is each new arm's validation-best checkpoint, with the
fixed-final results reported separately. All wins are against Rules; draws and
truncations count as nonwins. Original/reversed scores receive equal weight.
The five starting checks are imported from the verified feature-final source;
they are existing evidence. Up to 20 new frozen checks follow training, with
identical checkpoint hashes sharing results.

The original eight validation compositions are unchanged. Their imbalance is
an acknowledged limitation; this run tests training coverage against the same
selection criterion. Heldout teams remain excluded from training but reused
as development checks. One matched seed pair cannot establish seed robustness
or performance on arbitrary six-Pokémon human teams.

`models/new-run-selected.zip` is the best of the two new arms by validation.
`models/selected.zip` compares that new winner against the previous selected
fallback on the same validation suite. **The fallback retains ties.** Reports
identify which model won and whether any new model exceeded the previous score.
Heldout results do not choose either export. Best/final models from both arms
and `models/previous-selected.zip` remain available with recorded hashes.

Read the completed report and upload the result archive:

```bash
cat .local/run/training-coverage-01/report.md
```

Upload `.local/run/training-coverage-01/training-coverage-results.zip`. Model
weights stay on your machine. The source run and original 400k reference remain
unchanged. Review mirror retention, original-team performance and both heldout
orders together before claiming that broader training helped.

## Save useful models permanently

After completion, from the project root:

```bash
mkdir -p models
cp -n .local/run/training-coverage-01/models/selected.zip \
  models/training-coverage-01-selected.zip
cp -n .local/run/training-coverage-01/models/balanced-pool-final.zip \
  models/training-coverage-01-balanced-final.zip
```

The first is the overall validation winner, possibly the prior fallback. The
second preserves the expanded-pool final model for comparison; it is not
automatically the preferred model. `-n` prevents overwriting existing copies.

To retain the preceding experiment's two useful candidates as well:

```bash
cp -n .local/run/switch-feature-ablation-01/models/selected.zip \
  models/switch-feature-ablation-01-selected.zip
cp -n .local/run/switch-feature-ablation-01/models/switch-features-final.zip \
  models/switch-feature-ablation-01-feature-final.zip
```

## Use the training time for student-play preparation

While these jobs run, try the feature-final candidate from the completed run:

```bash
python mgba/play_gui.py \
  --rom "$PWD/FireRed.gba" --source "$PWD/FireRed.sav" \
  --checkpoint models/switch-feature-ablation-01-feature-final.zip \
  --workers 2 --speed 1
```

Open `http://127.0.0.1:8080`, leave AI action sampling off initially and use
familiar three-Pokémon teams. Play two complete games, including one after a
GUI restart. Discuss an effective attack and a questionable replacement with
students. Keep the prior selected model ready as a fallback.

After the new run finishes and you copy its selected model, replace the
checkpoint argument with `models/training-coverage-01-selected.zip`. Human
games illustrate behavior; they do not estimate the Rules benchmark score.
Custom teams are useful challenges, but add task variation beyond the frozen
evaluation and potentially beyond this training recipe.

`training-coverage-verification.json` records operational verification with tiny
local Showdown models. No uploaded real model weights, ROM, battery save or
desktop were available locally, so learning gains and GUI operation require
verification on the user's machine.
