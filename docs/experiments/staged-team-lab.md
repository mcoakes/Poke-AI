# Staged Rules-only team lab

This workflow answers three different questions: can the policy learn a small
fixed problem, can continued training retain that skill, and can the skill
transfer when we introduce team variety? It preserves the broad 400k working
reference. Imported specialists and every new stage start are separate copies.

Before the stability continuations, the runner audits the imported mirror best
and final checkpoints on the same new-seed suite, and preserves the original
mirror update metrics in the new results archive.

The earlier fixed-team result motivates these tests: the fresh mirror best won
973/1000 new-seed battles at 153,600 transitions, while its final checkpoint won
21/1000 at 200,704. The resumed broad reference eventually reached 575/1000.
The collapse is observed; its cause is not yet established. This curriculum does
not assume that lowering the learning rate fixes it.

## Stages and advancement

All opponents are the unchanged Gen3 Rules policy. All stages use shared-moves
observations v2, level-50 Pokemon, Hardy natures, 31 IVs, zero EVs, and no held
items. Simulator randomness, natural abilities and move effects remain enabled.
The JSON configuration records all budgets, thresholds, seeds and learning rates.
Budgets below are additional learner transitions and may round up by one rollout.

| Stage | Training/problem | Configurations | Maximum per job | Advancement check |
|---|---|---|---:|---|
| Move choice | Starmie versus Golem, original fixed moves | Recheck imported move-choice best; train fresh only with `--fresh-start` | 50k if needed | At least 99% wins AND first-turn Surf, ordinary and rotated slots |
| Mirror stability | Original fixed Typhlosion/Blastoise/Sceptile team on both sides | Same saved best and optimizer; LR 0.0001 vs 0.00002, matched seed | 100k | Selected **final** checkpoint >=90% wins and loses at most 5 points versus frozen start |
| Mirror replication | Original fixed mirror | Two fresh models, seeds 261040/261041, LR 0.0001 | 200k each | **Both** best checkpoints >=90% on frozen ordinary and rotated checks |
| Party order | Same three Pokemon, all six party permutations | Same stage start and seed; LR 0.0001 vs 0.00002 | 300k | Selected best >=70% on ordinary and rotated checks |
| Team composition | Small fixed pool of three-Pokemon teams, always Fire/Water/Grass | Same stage start and seed; LR 0.0001 vs 0.00002 | 400k | Selected best >=60% on new-seed checks and >=55% on held-out compositions |

With a previous lab, stability runs before fresh replications because the known
collapse is the immediate diagnostic. With `--fresh-start`, replication comes
first to supply the stability starting checkpoint. A failed stage stops the
workflow and saves its reports; it does not launch harder training automatically.
A threshold is an explicit experimental gate, not a claim that every matchup has
an achievable universal win rate. A blocked stage calls for inspecting its
baselines and traces, not automatically adding millions of training steps.

Except for stability, training stops after two consecutive validation checks
meet that stage's win threshold, with zero truncations. Initial validation may
count once; the second must evaluate a later checkpoint. Validation runs every
5,000 transitions (rounded to a rollout), using 256 requested battles. The
stability branches intentionally use their full budget: stopping at their already
strong initial score would hide the failure we want to investigate.

Checkpoint/configuration selection uses validation only. Stability selects the
highest final validation score; other comparisons select the highest best
validation score, with earlier entries retaining ties. Frozen tests run after
selection and gate progression. A failed selected checkpoint does not cause the
runner to choose a different one using the same test results. Best AND final
checks are retained for every job; identical checkpoint files share a check.
A replication failure blocks progression even if the other seed succeeds.

For every stage, random, greedy damage, and Rules candidate baselines are measured
against Rules (200 battles each). Final checks use 1,000 requested battles with
balanced candidate seats. Truncations/incomplete results fail a gate. Rotated
checks use the same teams and battle seeds as the ordinary test and require no
more than a 3-point drop, so they are a paired invariance check, not another
independent seed sample. The successful mirror specialist's new seeds do not
establish arbitrary-team generalization.

Party-order sampling picks two distinct entries from the six-team directory.
The composition pool has two species choices per type: Typhlosion/Arcanine,
Blastoise/Vaporeon, and Sceptile/Venusaur. Moves remain Strength plus their STAB
move (Flamethrower, Surf, LeafBlade or RazorLeaf). Six compositions, with six
orders each, form 36 training entries. Two compositions are excluded from both
training and checkpoint selection: Typhlosion/Blastoise/Venusaur and
Arcanine/Vaporeon/Sceptile, with all six orders each. Held-out checks draw both
sides from these 12 entries. Species are seen in training; only their combinations
are held out. Directory pairs use the production sampler's two distinct entries
without replacement, not always mirrors. This adds composition, type and stat
variation, while leaving full movesets, items and 6v6 for later experiments.

The selected order/composition checkpoint is also checked on the original mirror
to measure forgetting. That retention score is reported separately and does not
select a checkpoint or block these later stages. After stability, fresh replication
checks reproducibility; it does not replace the selected stable continuation used
as the party-order starting checkpoint.

## Start and watch

Install this updated archive over the existing project source first. From the
project root with the training venv active:

```bash
python run_staged_team_lab.py \
  --source-lab .local/run/fixed-team-lab-01 \
  --config config/experiments/staged-team-lab.json \
  --output-dir .local/run/staged-team-lab-01 \
  --jobs 2
```

The runner verifies imported best checkpoint hashes against their metadata,
copies the mirror/move-choice specialists and reward configuration, and records
all team/suite hashes. It does not copy over the 400k reference. A fresh output
directory is required. Keep other training batches stopped so total concurrency
stays at two. Each trainer uses 16 player environments (8 learners); validation
uses four player environments. No new Python dependencies are required beyond
the existing training installation and pinned Showdown setup.

In a second terminal, **watch both current and queued job logs** (all files exist
from startup):

```bash
tail -n 20 -F .local/run/staged-team-lab-01/console/*.log
```

Watch training validation and overall stage status:

```bash
watch -n 5 'cat .local/run/staged-team-lab-01/status.json; for f in .local/run/staged-team-lab-01/runs/*/evaluations.md; do
  [ -f "$f" ] || continue
  echo "$f"
  cat "$f"
done'
```

Watch machine memory, including during validation:

```bash
watch -n 2 free -h
```

The runner also automatically records available RAM and free swap every two
seconds in `memory.csv`, including calibration and validation phases. These are
machine-wide samples, not per-process RSS or a memory limit. Inspect them with:

```bash
tail -n 10 -F .local/run/staged-team-lab-01/memory.csv
```

The launcher terminal displays baseline/frozen-check results; training job logs
can be idle during those phases. `report.md` updates at stage boundaries and on
exit. `status.json` identifies the active stage or the gate that blocked progress.
The `Overall` column in evaluations.md is Rules win rate; the separate Rules
column may show a dash for a single-opponent suite.

## Pause, resume, and change configuration

Ctrl+C once in the launcher stops its active trainers gracefully. Wait for cleanup.
Every update saves latest.zip; recovery can lose the update currently in progress
and begins new battles, so it is not bit-for-bit uninterrupted continuation.
Ctrl+C in a viewer only closes the viewer.

```bash
python run_staged_team_lab.py \
  --output-dir .local/run/staged-team-lab-01 --continue --jobs 2
```

Watch resumed jobs and memory with the same commands:

```bash
tail -n 20 -F .local/run/staged-team-lab-01/console/*.log
watch -n 2 free -h
```

Continuation restores pinned settings and budgets; only `--jobs` changes execution.
Completed stages/checks are reused, and partial checks retry in new directories.
A blocked gate stays blocked on continuation. To test a different configuration,
copy/edit the JSON configuration and start a new output directory; do not edit
saved inputs or erase failed diagnostics. There is no automatic extension of a
failed stage's budget and no automatic rollback/retry until the test happens to pass.

For a genuinely fresh workflow, replace `--source-lab ...` with `--fresh-start`
and optionally supply `--reward-config ...`. Without that override it uses the
bundled damage_knockout_win reward. Include the log/memory viewers above with
any start or resume command.

## Results and diagnosis

Upload `.local/run/staged-team-lab-01/staged-team-results.zip` when complete or
blocked. It contains settings, baseline/test episodes and summaries, training
histories, PPO updates/CSV metrics, memory samples, and bounded decision audits.
Weights and bulk console/battle logs stay on your machine.

For each trained job:

- `runs/NAME/evaluations.md`: deterministic validation win rates.
- `runs/NAME/updates.jsonl` (plus session directories after recovery): post-update
  PPO KL, clip fraction, entropy loss, value loss, explained variance, learning
  rate, policy loss, and rolling training outcomes.
- `runs/NAME/checkpoints/best.zip` and `final.zip`: selected best and ending policy.
- `checks/NAME-*/decision-audit.json`: all-battle action/switch counts.
- `checks/NAME-*/decision-traces.jsonl`: up to the first 16 cases, 24 decisions
  each; chosen move/switch target, Rules suggestion, legal actions, and frozen
  policy action probabilities/entropy. Rule suggestions are heuristic comparisons,
  not an oracle for optimal actions.

These diagnostics let us distinguish changes in switching/attack choices from
mere test noise, and relate the change to optimizer metrics. They do not on their
own prove why a policy collapsed. Source-lab weights are not included in the old
uploaded results archive, so the full stability study must run on the machine
that holds them.

`--smoke --fresh-start` is an installation/integration check: tiny budgets, eight
battles per suite, and explicit SMOKE status. It bypasses learning gates to exercise
all stages and must never be treated as a successful training curriculum.

## Verification for this update

54 focused tests passed, including real post-update validation/early stopping,
native continuation, remaining-budget recovery, stage gates, checkpoint selection,
team pool separation, and NumPy-safe audit serialization. A real nine-job smoke
run traversed all five stages and completed 48 frozen checks with zero truncations;
continuing the completed batch started no new training. Source-lab import was
verified with real local best/final checkpoints and pinned hashes. These tiny
integration runs do not demonstrate curriculum learning or explain the user's
collapse. Compact evidence is in [staged-team-verification.json](staged-team-verification.json).

For the expanded fixed-to-1,024-team progression, editable teams, rehearsal, and
team inspection commands, see [Expanding team curriculum](team-curriculum.md).
