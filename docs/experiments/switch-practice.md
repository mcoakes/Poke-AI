# Switch practice and longer PPO

The completed [teacher-training run](teacher-training.md) selected a model with
51.37% validation, 46.50% then-unseen-team wins and 93.7% fixed-mirror wins
against Rules. This experiment asks whether a fresh warmup that gives rare
switch decisions more attention helps it progress further than longer PPO alone.
Both jobs start from that selected model, at 2,255,872 PPO timesteps.

## Why this experiment

The previous warmup matched Rules on 85.5% of held-back decisions overall,
but only 4 of 65 ordinary decisions where Rules recommended switching. It
matched 130 of 218 such fitting examples. These are correlated examples from
training-team trajectories, not 65 independent battles, and Rules is a
heuristic teacher rather than an optimal-action oracle. Nevertheless, this is
a specific learning weakness worth testing before expanding the task again.

The treatment combines refreshed current-policy examples, stratified example
retention and decision weighting. Any gain belongs to that combined strategy;
this experiment cannot isolate weighting alone. It also receives more data and
supervised computation than the control, so this is not an equal-compute test.

## What each job does

| Setting | `ppo-only` | `switch-practice` |
|---|---|---|
| Starting weights | Previous validation-selected teacher-trained model | Same |
| Additional warmup | None | Fresh Rules-labelled training trajectories and mirror rehearsal |
| PPO optimizer | Fresh Adam state | Fresh Adam state after warmup |
| Added PPO transitions | 1,000,000 requested, rounded to full rollouts | Same |
| PPO seed | 261150 | Same |
| Learning rate / entropy | 0.0001 / 0.01 | Same |
| Episode-start mix | 25% original 16 teams / 65% expanded 96 / 10% fixed mirror | Same |
| Teams, moves, rewards, bench order | Existing saved configuration, fixed bench order | Same |
| Validation | Existing 36-composition selector, every 100k added transitions | Same |

Both jobs train the whole policy during PPO. Both reset optimizer history so
the warmup arm does not inherit a different Adam state. This is continuation
of learned weights with a fresh optimizer, not preservation of all optimizer
momentum. The budget is twice the preceding experiment's PPO budget per job.

Warmup collects 6,000 battles on training teams only: half current-student
trajectories and half Rules trajectories, all against Rules. It keeps at most
24 decisions per battle and skips observations with only one legal action.
Rules ties remain a set of equally accepted actions. A mixed move/switch tie
gets ordinary move weight, rather than being forced into a switch target.

| Teacher decision | Relative weight per retained example |
|---|---:|
| Ordinary move choice or mixed tied target | 1 |
| Forced replacement | 2 |
| Ordinary decision whose best targets are all switches | 4 |

A bounded, uniform reservoir within each category retains at most 30,000
ordinary move/mixed examples, 10,000 forced replacements, 5,000 ordinary-switch
examples and 5,000 mirror examples. It does not stop collecting when common
move examples fill their allocation. Rare examples are kept when encountered;
the collector does not fabricate switching situations or draw from evaluation
teams. Reports distinguish observed, retained, replaced and discarded examples.

Fixed-mirror examples preserve the starting model's action probabilities,
with 10% of the supervised objective assigned to this rehearsal. Rules-labelled
examples share the other 90% after applying the weights above. Training uses
disk-backed arrays and minibatches. The warmup runs up to eight epochs and
retains the lowest weighted teacher holdout loss, including the unchanged
epoch-zero model as a fallback. Both seats and both trajectory sources for an
unordered team pair stay in the same fitting/holdout partition. Mirror examples
stay in fitting. Ordinary-switch agreement is reported separately from forced
replacement agreement. The full run requires ordinary-switch examples in
both fitting and holdout partitions.

Warmup updates the actor and shared observation encoder while freezing
critic-only tensors. The changed encoder can still change value predictions.
Supervised updates do not advance the PPO timestep counter. Evaluation and
play use the learned model's own actions, with no Rules override.

## Retention, selection and interpretation

Earlier-task training remains active: 25% old-team episode starts and 10%
fixed-mirror episode starts during PPO, plus the mirror soft-target warmup
rehearsal. Separate seen-team and fixed-mirror checks measure retention.

The previous selected model remains a fallback. Checkpoint selection uses
the unchanged 2,048-battle validation suite only; earlier candidates retain
ties. Warmup-only, each job's validation-best and each final model are saved.
Final-check results do not choose the selected export.

The prior 24-composition fresh pool has now been inspected. In this run it is
a **reused development check**, still excluded from training and checkpoint
selection, but no longer an untouched final test. The frozen checks retain
both original and reversed bench order and both player seats. A later claim
about new-team generalization will need a new untouched pool and independent
training seeds. The existing task still uses 12 species, three Pokémon per
team and two damaging moves per Pokémon; these scores do not describe arbitrary
six-Pokémon human teams.

## Save the current model before starting

From the project root, retain the selected model and a useful final alternative:

```bash
mkdir -p models
cp -n .local/run/teacher-training-01/models/selected.zip \
  models/teacher-training-01-selected.zip
cp -n .local/run/teacher-training-01/models/teacher-ppo-final.zip \
  models/teacher-training-01-final.zip
```

`-n` prevents overwriting existing copies. The selected source has SHA-256
`528131b474ad60ed47dc424b60f2d0c383ff3b5596f674d3c8b570bd13fb8c4e`.
The runner verifies the source run and checkpoint rather than trusting a name.

## Launch the two jobs

Extract the updated project archive over the source files, retaining `.local/run/`,
models and the original 400k reference. Then:

```bash
source .venv/bin/activate
python run_switch_practice.py \
  --source-run .local/run/teacher-training-01 \
  --output-dir .local/run/switch-practice-01 \
  --jobs 2
```

Preparation, warmup, both PPO jobs, frozen checks and model export run in
sequence automatically. Only two PPO jobs run concurrently, with 16 environment
players per job and one Torch thread. The last full workflow took about
79 minutes, including about 35 minutes for simultaneous 500k-step PPO jobs.
This doubles the PPO budget and collects more examples, while reusing verified
parent checks. Allow roughly 1.5–2 hours on the same machine; this is an estimate,
not a measured runtime for the new experiment. Keep watching memory.

## Watch output and memory

In another terminal, follow preparation, warmup and worker logs:

```bash
tail -n 20 -F .local/run/switch-practice-01/console/*.log
```

Watch validation as the two job directories appear:

```bash
watch -n 5 'for f in .local/run/switch-practice-01/runs/*/evaluations.md; do [ -f "$f" ] && { echo "$f"; cat "$f"; }; done'
```

Current phase and memory, in separate terminals:

```bash
watch -n 5 cat .local/run/switch-practice-01/status.json
watch -n 2 free -h
```

Each job also saves `updates.jsonl`, PPO CSV metrics and checkpoints.
`memory.csv` records available RAM and swap through validation peaks. The
warmup report at `practice/report.json` records the actual retained categories, weighted and unweighted
holdout losses, switch agreement, selected epoch and model hashes. Frozen-check
progress appears in the launcher log; individual checks live under `checks/`.

## Inspect the teams being used

After preparation:

```bash
python run_switch_practice.py \
  --output-dir .local/run/switch-practice-01 \
  --inspect-teams new --limit 3 --full-json
```

Use `old`, `mirror`, `validation`, `fresh-test` or `all` for other pools.
`fresh-test` retains the previous file label for compatibility; in this
experiment it means the reused development pool. Increase `--limit` to inspect
more teams. The copied JSON definitions under `inputs/teams/` are the actual
saved inputs; the plan records the pools, weights and suite paths.

## Pause and resume

Press Ctrl+C once and wait for the launcher and workers to finish saving and
exit. Resume the remaining saved budget with:

```bash
python run_switch_practice.py \
  --output-dir .local/run/switch-practice-01 \
  --continue --jobs 2
```

Completed preparation and checks are verified before reuse. Interrupted
warmup attempts are retained and may repeat their bounded preparation work.
PPO resumes from saved checkpoints; continuation does not add another million
steps beyond the original budget. Use the same log and memory viewers above.

## Save the resulting models and play

After completion:

```bash
mkdir -p models
cp -n .local/run/switch-practice-01/models/selected.zip \
  models/switch-practice-01-selected.zip
cp -n .local/run/switch-practice-01/models/practice-only.zip \
  models/switch-practice-01-practice-only.zip
cp -n .local/run/switch-practice-01/models/ppo-only-best.zip \
  models/switch-practice-01-ppo-only-best.zip
cp -n .local/run/switch-practice-01/models/switch-practice-best.zip \
  models/switch-practice-01-best.zip
cp -n .local/run/switch-practice-01/models/switch-practice-final.zip \
  models/switch-practice-01-final.zip
```

The selected export can remain the previous model if neither treatment improves
validation. Start play with the already saved teacher-trained model while the
new jobs run, or substitute the new selected copy once available:

```bash
python mgba/play_gui.py \
  --rom "$PWD/FireRed.gba" --source "$PWD/FireRed.sav" \
  --checkpoint models/teacher-training-01-selected.zip \
  --workers 2 --speed 1
```

Open `http://127.0.0.1:8080`; leave AI action sampling off initially. Familiar
teams make a useful student demonstration. Human games are examples of behavior,
not replacements for measured win rates against Rules.

Read `report.md` and upload
`.local/run/switch-practice-01/switch-practice-results.zip` after the experiment.
Weights and large training arrays stay on the user's machine. The archive
contains the plan, learning diagnostics, decision audits, results and memory
record. Prior experiments and failed checks remain preserved.


## Workflow verification

All 85 focused tests passed across the prior teacher workflow and new weighted
practice/runner modules. A real Showdown smoke used two tiny-model PPO jobs
(256 added transitions each), 40 collection battles and 375 retained examples,
including 11 ordinary-switch examples. All 248 accepted frozen-check battles
completed with zero truncations and no Rules overrides. These are operational
checks, not evidence of a learning gain from the new full experiment.

The smoke verified legal/tied teacher targets, exact source-policy mirror
probabilities, pair-disjoint fitting and holdout, unchanged critic-only tensors
and PPO counters during practice, both optimizer resets, seven model hashes,
validation-only selection and the source run's preservation. Completed
continuation did no expensive work and left inputs, models, training results,
checks, console logs and practice artifacts unchanged. Team inspection and
live/archive content consistency also passed. The ROM/GUI play path was not
run here because the user's ROM and save were not provided.

The first smoke attempt stopped when strict saved-evidence validation detected
two incomplete/inconsistent frozen diagnostic artifacts. One normal
`--continue` retried those checks successfully and preserved their originals.
No code changes, relaxed gates or reconstructed counts were used. The cause
of that file inconsistency was not established. If the same explicit diagnostic
message appears in a full run, use the documented continuation command; do
not delete the failed checks. Exact evidence is in
`switch-practice-verification.json`.
