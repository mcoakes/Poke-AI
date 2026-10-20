# Rules teacher warmup, then PPO

This experiment compares **learning from Rules' decisions before PPO** against
**PPO alone**. It is a direct change to the learning signal after the expanded
training pool failed to improve the reused transfer checks. It also replaces
the narrow eight-composition selector with a broader, predeclared validation
pool. The current selected model and two earlier candidates remain available.

## Completed result — October 19, 2026

The selected teacher-plus-PPO checkpoint improved broader validation to **51.37%**,
with **46.50%** wins on the then-unseen composition pool and **93.7%** on the
fixed mirror. All scores are against Rules. Selection retained the best validation
checkpoint at 2,255,872 total PPO timesteps.

| Model | Validation | Seen teams, both orders | Fresh teams, both orders | Fixed mirror |
|---|---:|---:|---:|---:|
| Starting selected model | 46.00% | 42.70% | 42.85% | 61.6% |
| Teacher warmup only | 44.97% | 42.65% | 43.25% | 61.7% |
| PPO only, validation best | 47.41% | 45.35% | 44.80% | 62.7% |
| Teacher + PPO, validation best | **51.37%** | **50.00%** | **46.50%** | **93.7%** |
| Teacher + PPO, final | 51.22% | 47.85% | 47.50% | 98.8% |

The best-teacher versus best-PPO fresh-team difference is +1.70 percentage points;
a paired 95% bootstrap interval is -0.45 to +3.90 points. Compared with the starting
model, selected-teacher improves +3.65 points, interval +1.80 to +5.50. These
intervals keep both seats and bench orders together within 500 matchup/seed
clusters. They condition on this pool and shared training history, and do not
measure variation across training seeds. The final checkpoint is worth saving
for play comparisons, but its endpoint score does not replace validation selection.

Warmup alone barely changed battle results. Its useful effect emerged with PPO.
The selected imitation epoch was 6 of 8; holdout loss fell from 1.219 to 0.429.
However, ordinary decisions where Rules requested a switch had only **4/65**
correct holdout predictions, versus **130/218** in fitting examples. This is a
specific generalization weakness in matching the teacher, not proof that every
Rules recommendation was optimal. Those are correlated decision examples,
not independent battles. More training on the exact same examples is not the
obvious next step: epochs 7 and 8 had slightly worse holdout loss.

The next [switch-practice comparison](switch-practice.md) uses fresh training-only
trajectories and conservative decision weighting before longer PPO, against a
matched longer-PPO control. The 24 fresh compositions above become reused
development checks in that experiment; they are no longer an untouched test.

Verified: 44 distinct frozen checks, 48,192 requested/completed battles, zero
truncations and zero heuristic action overrides. The two jobs each added 500,736
PPO transitions. The full workflow took about 79 minutes; available RAM never
fell below 21.90 GiB and recorded swap use stayed zero. Uploaded model weights
and raw example arrays were omitted, so verification checked the supplied
provenance, saved episode records, 191 available input pins and 21 implementation
hashes rather than loading the real trained models here.

Numerical evidence: `teacher-training-01-analysis.json` and
`teacher-training-01-behavior.json`.

## What changes

| Setting | `ppo-only` | `teacher-ppo` |
|---|---|---|
| Starting model | Current coverage-selected checkpoint, 1,854,464 total steps | Same original weights |
| Warmup | None | Rules-labelled action examples plus source-policy mirror rehearsal |
| PPO optimizer | Fresh state | Fresh state after warmup |
| Added PPO training | 500,000 transitions, rounded to complete rollouts | Same |
| Training teams | 25% original 16 / 65% expanded 96 / 10% fixed mirror | Same |
| PPO settings | Learning rate 0.0001, entropy 0.01, matched seed 261140 | Same |
| Architecture, reward, ordering | Existing replacement features; fixed saved bench order | Same |
| Selection | New 36-composition validation suite | Same |

The common parent is fixed before scoring the new validation suite. In the
completed coverage run it is the expanded-pool best/final model. Resetting Adam
in both arms prevents old optimizer momentum from treating the warmup arm
differently. This experiment does not reproduce the preceding run's unchanged
optimizer continuation. Source checkpoint files are preserved.

The treatment receives extra supervised examples and computation. This compares
the added warmup strategy at the same PPO budget; it is not an equal-total-data
or equal-compute efficiency test.

The warmup collects decisions on training teams only, using frozen student
versus Rules and Rules versus Rules trajectories. Rules labels move choices,
voluntary switches and forced replacements. Single-legal-action observations
are skipped. If Rules assigns several actions the same best score, the loss
rewards their combined probability; it does not teach an arbitrary lowest-slot
tie break. Fixed-mirror examples instead preserve the starting model's action
probabilities as rehearsal. Percentages describe episode starts; varying battle
lengths mean transition proportions differ.

Collection is bounded to 2,000 battles, 50,000 examples and 24 decisions per
battle. Arrays stay on disk and training uses minibatches. Warmup is bounded
to eight epochs and retains the epoch with the best imitation holdout loss.
That holdout contains separate training-team pairs, with both seats and both
trajectory sources kept together. The single fixed-mirror pair stays in the
fit partition. These are not the fresh teams used for final battle checks.

Warmup trains the actor and shared observation encoder. Critic-only parameter
tensors are frozen during warmup, but the changed encoder can change critic
predictions. Whole-policy PPO then trains normally in both jobs. The saved
warmup-only model remains available even if subsequent PPO loses performance.

Rules is a heuristic teacher, not an optimal-action oracle. Better agreement
does not prove better battle results or guarantee a win rate above 50%.
This is **imitation plus reinforcement learning**. Rules supplies training
labels; the learned model chooses its own actions during evaluation and play.

## Selection and final checks

The new selector contains 36 unique compositions from the existing 12-species,
three-Pokémon, two-damaging-move recipe. Each species appears nine times overall
and three times in each party position. Its 2,048 battles balance original and
reversed bench order and both player seats. The old eight-team validation score
is not numerically comparable to this new score.

The final fresh pool has another 24 compositions, each species appearing six
times overall and twice per position. Its 1,000 original-order and 1,000
reversed-order battles are not used to select checkpoints. Both new pools
exclude every prior training, mirror, validation and development-check
composition, including reordered variants. Existing seen-team checks in both
orders and the fixed mirror check measure retention separately.

Before PPO training, the runner scores three declared fallback models on the
same new selector: current coverage-selected, coverage small-pool best and the
coverage starting feature-final model. Fallbacks retain ties. The final export
compares these with the warmup-only and new training candidates using this
selector only. Final-check results never choose the winner. The previous
eight-composition validation winner remains saved regardless of the outcome.

This is one matched training-seed comparison. Fresh composition checks test
transfer within the same restricted recipe, not unfamiliar mechanics or
arbitrary six-Pokémon human teams. Repeated validation is a model-selection
tool; the final unseen-team result is the stronger transfer measurement.

## Start the experiment

Extract the updated archive over the project source, keeping all `.local/run/`
directories and model files. From the project root:

```bash
source .venv/bin/activate
python run_teacher_training.py \
  --source-run .local/run/training-coverage-01 \
  --output-dir .local/run/teacher-training-01 \
  --jobs 2
```

The runner prepares suites, checks fallbacks, collects examples, performs the
warmup, runs the two PPO jobs, checks the saved models, and exports a selected
model and uploadable results. It needs no manual handoff between stages. Only
two PPO jobs run concurrently, each with 16 environment players and one Torch
thread. Allow longer than the preceding 39-minute run: warmup, broader
validation and additional baseline checks add work. The completed full workflow took about 79 minutes on the user's machine.

## Watch progress

In another terminal, follow the console logs throughout preparation and training:

```bash
tail -n 20 -F .local/run/teacher-training-01/console/*.log
```

Watch PPO validation after the job directories appear:

```bash
watch -n 5 'for f in .local/run/teacher-training-01/runs/*/evaluations.md; do [ -f "$f" ] && { echo "$f"; cat "$f"; }; done'
```

Watch the current phase and memory in separate terminals:

```bash
watch -n 5 cat .local/run/teacher-training-01/status.json
watch -n 2 free -h
```

`memory.csv` records machine-wide available RAM and swap, including validation
peaks. Each PPO job also saves `updates.jsonl`, PPO CSV metrics and checkpoints.
The warmup report at `warmstart/report.json` records example counts,
source/teacher identities, fit and holdout losses, teacher agreement by decision
type, and chosen epoch. Agreement is measured separately from battle wins.
Frozen-check progress appears in the launch
terminal; individual results are retained under `checks/`.

## Inspect the actual teams

After preparation, print saved team definitions:

```bash
python run_teacher_training.py \
  --output-dir .local/run/teacher-training-01 \
  --inspect-teams new --limit 3 --full-json
```

Use `old`, `mirror`, `validation`, `fresh-test` or `all` to inspect other pools.
Increase `--limit` to print more teams. JSON definitions are under
`inputs/teams/`; sampling weights and the suite inventories are also frozen in
the run inputs. These files are the inputs used by the jobs.

## Pause and resume

Press Ctrl+C once, then wait for the launcher and workers to exit. Continue
the saved budget with:

```bash
python run_teacher_training.py \
  --output-dir .local/run/teacher-training-01 \
  --continue --jobs 2
```

Keep the same saved inputs and budget. Completed preparation artifacts are
verified before reuse; interrupted preparation attempts are preserved. PPO
resumes from its saved checkpoint. An interrupted warmup may repeat its bounded
preparation work. Use the same log, evaluation and memory viewers above.

## Save and play useful models

After completion:

```bash
mkdir -p models
cp -n .local/run/teacher-training-01/models/selected.zip \
  models/teacher-training-01-selected.zip
cp -n .local/run/teacher-training-01/models/teacher-ppo-final.zip \
  models/teacher-training-01-final.zip
cp -n .local/run/teacher-training-01/models/teacher-only.zip \
  models/teacher-training-01-teacher-only.zip
cp -n .local/run/teacher-training-01/models/ppo-only-best.zip \
  models/teacher-training-01-ppo-only-best.zip
cp -n .local/run/teacher-training-01/models/teacher-ppo-best.zip \
  models/teacher-training-01-teacher-ppo-best.zip
```

`-n` prevents overwriting existing copies. The selected export may be a fallback
if new training fails to improve the new validation score. The other copies
preserve useful comparisons; a later checkpoint is not necessarily better.

Start visible human-versus-agent play:

```bash
python mgba/play_gui.py \
  --rom "$PWD/FireRed.gba" --source "$PWD/FireRed.sav" \
  --checkpoint models/teacher-training-01-selected.zip \
  --workers 2 --speed 1
```

Open `http://127.0.0.1:8080`. Keep AI action sampling off initially, as in the
frozen checks. Try the teacher-only and PPO-only copies on the same teams as
well. Familiar training teams are suitable for a student demonstration; those
human games are illustrations, not independent performance measurements.

Read and upload the completed results:

```bash
cat .local/run/teacher-training-01/report.md
```

Upload `.local/run/teacher-training-01/teacher-training-results.zip`. Weights
and the large example arrays remain on your machine. The archive includes the
configuration, provenance, warmup learning report, game results and memory
record. Original experiments, failed gates and the 400k working reference are
not overwritten.

## Workflow verification

The three new test modules passed 48 focused tests. A real Showdown smoke used
two tiny-model PPO jobs, 40 collection battles and 312 frozen-check battles,
with no truncations. It checked legal teacher targets, mirror rehearsal,
disjoint imitation pairs, serialized optimizer resets, unchanged critic heads,
all nine export hashes and validation-only selection. Completed continuation
left inputs, models, training, checks, logs and warmup files unchanged.

The smoke required normal continuations after some live summary/episode files
became earlier partial versions, even though the produced result archive held
complete evidence. The responsible process was not identified. Partial
directories were preserved, all final live and archived evidence was verified,
and no diagnostic counts were relaxed or reconstructed. The detailed record is
`teacher-training-verification.json`. This verifies workflow operation, not a
learning gain from the user's full-length models or the visible mGBA interface.
