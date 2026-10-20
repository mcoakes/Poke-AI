# Expanding team curriculum

Use this after the staged fixed-team lab has produced a passed mirror-stability
checkpoint. It implements progression from the completed fixed tasks to hundreds
of distinct teams with varied moves and builds. It preserves the existing lab,
its failed replication evidence, and the broad 400k working reference.

The reported previous results support this starting point: both stability
continuations ended around 99.5–99.6% versus Rules, compared with 97.3% at the
start. They did not reproduce the earlier collapse. A one-win difference in
1,000 battles does not establish a learning-rate winner. Fresh replications at
97% and 61% show seed sensitivity at that budget, not inability to learn.

The original runner made success by both fresh seeds mandatory. This new run
records replication as a diagnostic and advances the separately qualified
stability model. It does not rewrite a failed result as passed or mutate the old
batch. New policy-specific qualification, held-out performance and retention
gates remain mandatory. The aggregate report establishes outcomes; PPO update
histories and decision traces are needed to diagnose the weaker seed or collapse.

## Stages

All training opponents remain Rules. Stage 10's final benchmark also checks the
final selected model against greedy damage and the frozen foundation policy.
Stage counts below are distinct species compositions, not renamed teams or
party/move-order permutations. Generated move order and party order also vary.

| Stage | Task | New training teams | Party size | Moves/builds | Max transitions per arm |
|---|---|---:|---:|---|---:|
| 1 | Requalify saved move-choice best | Fixed pair | 1 | Original Starmie/Golem | No training |
| 2 | Requalify passed stability model | Fixed mirror | 3 | Typhlosion/Blastoise/Sceptile | No training |
| 3 | Retain earlier replication results | Existing evidence | 3 | Diagnostic status, not a progression veto | No training |
| 4 | Type matchups | 16 | 3 | Two attacks: STAB and Normal | 200k |
| 5 | Move coverage | 48 | 3 | Three damaging moves, STAB plus coverage | 300k |
| 6 | Support decisions | 96 | 3 | Four moves, including support/status/recovery/setup choices | 400k |
| 7 | Varied builds | 160 | 3 | Varied natures, EV spreads and held items | 500k |
| 8 | Larger parties | 256 | 4 | Four moves, varied builds | 600k |
| 9 | Full battles | 384 | 6 | Four moves, varied builds | 800k |
| 10 | Broad teams | 1,024 | 6 | Mixed generated and curated sets | 1m |

Species pools grow from 12 to 64. Abilities vary through species' natural
simulator defaults; arbitrary ability selection is not supported. Moves are
individually checked against the bundled Gen III learnsets. This is not a
certificate of tournament legality or compatibility of every event/move
combination. Sets are constrained to useful STAB and coverage attacks, but legal
generated support combinations are not guaranteed to be competitively optimal.

The teams are generated once using the recorded seed and then frozen. Stage 10
samples a large finite pool, not an unlimited stream of new teams. You can change
counts, species, moves and generation seed in a new configuration/new batch.
The current runner deliberately extends the already-completed foundation; use
the fixed/staged lab to train that foundation on a new installation.

## Editable configuration and inspection

Curriculum definitions and generation settings are in
**config/experiments/team-curriculum.json**. Stage 10 also draws curated sets from
the referenced **config/generation/competitive_gen3.json** catalog:

- `fixed_stages`: introductory team JSON definitions.
- `species_pools`: which Pokemon can appear at each level.
- `move_pools`: damaging and support moves from which legal sets are assembled.
- `builds`: allowed natures, items, and EV spreads. IVs are perfect in this curriculum.
- `stages`: party sizes, train/validation/heldout/benchmark counts, move counts,
  stage budgets, win thresholds and maximum battle steps.
- `training`: learning-rate comparison, concurrency-related environment count,
  validation cadence, rehearsal proportions, retention and truncation limits.

Actual generated teams are stored in:

```text
.local/run/team-curriculum-01/inputs/curriculum-teams/STAGE/train/
.local/run/team-curriculum-01/inputs/curriculum-teams/STAGE/validation/
.local/run/team-curriculum-01/inputs/curriculum-teams/STAGE/heldout/
.local/run/team-curriculum-01/inputs/curriculum-teams/10-broad-teams/benchmark/
```

Inspect three actual stage-4 teams and the training mixture:

```bash
python run_team_curriculum.py --output-dir .local/run/team-curriculum-01 \
  --inspect-teams 4 --split train --limit 3
```

Inspect two stage-10 teams with their complete stat/item/move definitions:

```bash
python run_team_curriculum.py --output-dir .local/run/team-curriculum-01 \
  --inspect-teams 10 --split train --limit 2 --full-json
```

Omit the stage after `--inspect-teams` to list all pools. Use `--split validation`,
`--split heldout` or `--split benchmark` to inspect a different split. Inspection
is read-only and does not start training. The complete inventory, hashes and
diversity counts are in `inputs/curriculum-teams/index.json`.

Direct inspection of one file also works:

```bash
python -m json.tool \
  .local/run/team-curriculum-01/inputs/curriculum-teams/10-broad-teams/train/team-0001.json
```

To see the exact teams drawn in a completed training battle, inspect its episode
record. The historical field is named `states`; for Showdown it contains paths to
the two **team JSON files** actually used. Training copies are pinned under that
job's `inputs/team-mixture/` directory:

```bash
tail -n 1 .local/run/team-curriculum-01/runs/04-type-matchups-a/episodes.jsonl \
  | python -m json.tool
```

After pausing/resuming, new episode logs live in that job's `sessions/` tree.
`config.json` records the source and pinned team inventories; final
`summary.json → team_mixture_sampling` shows episode-start counts for the current
invocation. A planned pool and the files sampled in one battle are different
views of the same training inputs.

## Start the next experiment

Install the updated archive over the existing project source. Keep other training
batches stopped so total concurrency stays at two. From the project root:

```bash
source .venv/bin/activate
python run_team_curriculum.py \
  --source-run .local/run/staged-team-lab-01 \
  --config config/experiments/team-curriculum.json \
  --output-dir .local/run/team-curriculum-01 \
  --jobs 2
```

The source run may have overall status blocked due to replication: the importer
requires its move-choice and mirror-stability results to have passed, verifies
their selected checkpoint hashes, and copies those models. It then independently
requalifies move choice at 99% wins/first-turn Surf and the fixed mirror at 90%
wins, including move rotations. Source weights and files remain unchanged.

The command assumes the previous lab used `.local/run/staged-team-lab-01`.
Change `--source-run` if your report came from another directory.

If the conventional `models/observation-v2-shared-moves-400k-reference-01.zip`
exists, it is copied for a same-suite final benchmark against Rules. An optional
`--reference-checkpoint PATH` selects another explicit reference for that
comparison. It is never used to overwrite or replace your working reference.

Watch both jobs in a second terminal; all queued log files exist from startup:

```bash
tail -n 20 -F .local/run/team-curriculum-01/console/*.log
```

Watch stage status and validation:

```bash
watch -n 5 'cat .local/run/team-curriculum-01/status.json
for f in .local/run/team-curriculum-01/runs/*/evaluations.md; do
  [ -f "$f" ] || continue
  echo "$f"
  cat "$f"
done'
```

Watch RAM and swap:

```bash
watch -n 2 free -h
```

The runner also records machine-wide available memory and swap in `memory.csv`
every two seconds, including frozen evaluation. It is monitoring, not a hard
memory limit. Training uses 16 player environments per job (8 learners) and
four player environments for periodic validation. Calibration and frozen-check
progress appear in the launcher terminal, while training logs can be idle.

For inspection before training, add `--prepare-only` to the start command. All
pools/suites/model inputs are generated and pinned, then the runner exits. Inspect
them with the commands above and use the continuation command below to train.
Do not edit generated files in an existing batch: changed or additional team
files are rejected. Change the configuration before creating a new output folder.

## Advancement, rehearsal, and test meaning

Every broad stage compares two continuations from exactly the same selected
checkpoint and optimizer state: LR 0.0001 and 0.00002 with a shared training seed.
Other PPO settings are equal: entropy 0.01, rollout 128, batch 128, four epochs.
These are controlled configuration comparisons, not independent seed replications.

At stage 4, 90% of episode starts use the new pool and 10% use the original fixed
mirror. Later stages use 80% current, 10% distributed equally across earlier broad
stages, and 10% fixed mirror. The group is sampled once per episode; both players
receive teams from that group. This retains earlier tasks without randomly
pitting a three-member team against a six-member team.

Generated species compositions have one split assignment across the entire
curriculum. A composition reserved for validation, heldout or benchmark cannot
enter any stage's training pool or rehearsal. Stage validation therefore measures
excluded compositions, not just new RNG seeds on the training teams. All pools
have distinct team compositions internally; names and ordering cannot inflate
the counts. Historical inputs to the old 400k ancestor are not available here,
so exclusion is guaranteed **within this curriculum**, not over that model's
entire history.

Validation runs every 25k additional transitions (rounded to a rollout), with at
least 512 battles. It uses the validation pool. Two distinct consecutive successful
validations with zero truncations stop training early. Otherwise each arm runs
to its fixed budget. Budgets are maxima and can round up by one rollout.

| Stage | Validation threshold | Heldout threshold | Max battle decisions |
|---|---:|---:|---:|
| 4 | 65% | 60% | 100 |
| 5 | 62% | 58% | 100 |
| 6 | 60% | 57% | 150 |
| 7 | 58% | 55% | 150 |
| 8 | 58% | 55% | 200 |
| 9 | 56% | 55% | 250 |
| 10 | 55% | 55% | 300 |

The selected checkpoint is the highest best validation score across the two
arms; ties keep arm A. Selection is written before frozen test results are read.
No alternate arm is silently selected after a heldout failure. Both best and
final checkpoints are evaluated for diagnostics, deduplicating identical files.

Frozen checks include training-pool teams with new battle seeds, a separate
heldout pool, and the same heldout schedule with candidate move slots rotated.
Each suite runs at least 1,000 battles and enough paired cases to expose every
team in its pool before repeat cycles. Thus the stage 10 seen-team test expands
to 2,048 battles. Random, greedy and Rules baselines are measured on a separate
heldout calibration schedule; their counts also expand if coverage requires it.

Advancement requires the selected validation threshold, ordinary and rotated
heldout thresholds, no more than a 3-point rotation drop, and at most 1% truncation.
All rates count wins/requested; draws and truncations are non-wins. Retention checks
must remain within 10 points of the original imported mirror specialist and of
the stage-start model on the immediately previous stage's heldout distribution.
The mirror reference stays fixed across stages, preventing cumulative 10-point
losses from being treated as retained mastery.

These are explicit operational gates, not universal attainable ceilings or a
statistical certificate that a configuration is superior. Read the measured
baselines, side breakdowns, traces and optimization history when a stage blocks.
The runner stops on a failed gate, writes the diagnostic ZIP, and does not
silently extend budgets or choose a model using its test results.

Heldout progression checks are development gates. A separate 128-team stage 10
benchmark is reserved until all stages pass; the selected final model then plays
at least 2,000 battles against each of Rules, greedy damage and the frozen
foundation model. When available, the broad working reference also plays Rules
on that same benchmark. Benchmark results do not affect training, configuration
selection, or progression. Viewing a benchmark later for tuning makes it
historical development evidence for that subsequent experiment.

## Pause, resume, and results

Ctrl+C once in the launcher stops active trainers gracefully. Wait for cleanup.
Every completed PPO update saves latest.zip; an in-progress update can be lost.
Continuation starts new battles and is not bit-for-bit uninterrupted training.
Ctrl+C in a viewer only stops that viewer.

```bash
python run_team_curriculum.py --output-dir .local/run/team-curriculum-01 \
  --continue --jobs 2
```

Keep the same viewers running, or reopen:

```bash
tail -n 20 -F .local/run/team-curriculum-01/console/*.log
watch -n 2 free -h
```

Continuation restores pinned configuration and remaining budgets. Completed
stages/checks are reused; cached frozen checks require a complete episode log
with unique case IDs and outcome counts matching the summary. Partial or
inconsistent frozen checks retry in new directories. A blocked gate remains blocked. Use a new output directory and an edited config
for a new experiment. No action on the old staged run is needed to remove its
replication gate; this new run imports only its separately qualified policy.

Upload `.local/run/team-curriculum-01/curriculum-results.zip` after completion
or a blocked gate. It includes team/suite definitions, original gate evidence,
new checks/audits, training configuration, evaluation histories, update metrics,
sampling counts and memory records. Weights and bulk console/episode logs stay
local. `report.md` updates at stage boundaries and exit, while `status.json`
identifies the active stage.

`--smoke` uses tiny pools/budgets and explicitly bypasses learning gates to test
all stages. Its outputs are marked SMOKE and cannot supply a normal qualified
foundation. It verifies software operation, not learning success.

## Local verification

155 focused tests passed, covering team generation and split isolation,
weighted sampling, checkpoint imports, selection/gates, and native continuation.
A real simulator smoke exercised 14 training jobs across stages 4–10 and 135
frozen checks (1,080 requested battles, zero truncations), including the final
Rules/greedy/frozen-policy benchmark. Continuing the completed smoke started no
new training jobs. Team inspection was also checked.

These results verify operation, not learning performance. Tiny smoke budgets
explicitly bypass learning gates; the full experiment must run on your saved
qualified checkpoint. See `team-curriculum-verification.json` for scope and
fixture details.

For a stage-4 transfer failure after successful mirror retention, see
[type-transfer-lab.md](type-transfer-lab.md) for the two-job specialist-versus-fresh
comparison, exact team inspection, live logs, memory monitoring, and continuation.

That million-step comparison has now completed. See
[matchup-ablation.md](matchup-ablation.md) for the results and the next controlled
feature comparison. It retains the 90% stage-4 / 10% fixed-mirror mixture and
separate mirror checks; the curriculum's earlier failed gate remains unchanged.
