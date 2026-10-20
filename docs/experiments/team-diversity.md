# E05: Broader training data

[Experiment record](README.md) · [All setup flags](../reference/prepare-diversity-experiment.md)

The lower-rate model improved on familiar teams but not clearly on the generated
new-team pool. Test broader training data while retaining the starting best
checkpoint, frozen opponent, rewards and PPO rollout/batch/epoch settings.
Learning rate remains 0.0001. Keep this separate from opponent-pool experiments.

## Prepare once

```bash
python mgba/prepare_diversity_experiment.py \
  --source-run .local/run/ppo-observation-v2-lr0001-01 \
  --exclude-team-dir .local/test-teams/generalization-01 \
  --output-dir .local/run/diversity-v2-01 \
  --train-teams 3000 --validation-teams 500 --test-teams 500 \
  --team-size 3 --environments 64
```

This does not launch training. It generates level-50 teams using the same
60% competitive / 20% unrestricted / 20% learnset-based slot mixture for all
three splits, with perfect IVs, trained EVs and 80% common competitive items.
Every Pokemon must have four moves; sparse learnsets and duplicate parties are
rejected and resampled. Realized style counts after filtering are recorded.
Pools are shuffled and split deterministically. No party is shared across
splits, including reordered copies. Source training/validation parties and
explicit exclusions are also removed. Missing/changed recorded inputs fail.

The old 300-team development pool is excluded via the flag above. Add other
known historical pools with repeatable `--exclude-team-dir` or `--exclude-suite`.
Checkpoint ancestors' unknown training inputs cannot be reconstructed. Separation
is by complete configuration, not unseen species or completely novel movesets.

The directory contains copied initial and frozen models, rewards, source
metadata, generated teams, fixed validation/test suites, `experiment.json` with
hashes/provenance, and two shell scripts. Existing directories are never erased.
The scripts use absolute paths and the Python interpreter used during setup;
keep that environment and project location available.

## Train and watch

```bash
bash .local/run/diversity-v2-01/train.sh
```

In another terminal:

```bash
watch -n 5 cat .local/run/diversity-v2-01/train/evaluations.md
```

Training uses 500,000 additional timesteps (rounded to a PPO rollout). The first
evaluation measures the starting model on the **new validation suite**. Compare
later results against that row, not against 69.6% on the old suite. Validation
runs every 100,000 additional steps at update boundaries and at completion.
Each evaluation has 1,000 battles per Random/Dumb/Smart category, balanced by
seat; Smart's quota is shared across saved Smart models. The frozen training
opponent is unchanged. Items, rewards and policy schema are not redesigned.

The generated command inherits the source base seed, episode cap, opponent
sampling/switch settings, mirror setting, device and PyTorch threads. PPO
rollout/batch/epoch settings come from the starting checkpoint. The prepared
command, source configuration and actual training configuration record these
choices. Process workers are used; setup's environment count applies to training
and evaluation. Spectator logs rotate at 2 MiB with one backup; detailed
observation/step logs are off. Retain two numbered checkpoints and two recent
evaluations plus best. Periodic checks also preserve `best.zip`.

If interrupted, continue the generated training run through the usual command:

```bash
python mgba/train_ppo.py \
  --output-dir .local/run/diversity-v2-01/train \
  --if-exists continue --total-timesteps 100000
```

That adds a new budget; it does not calculate the unspent portion of the original
500,000. Record any extension. `train.sh` is a new-run launcher and cancels if its
training directory already exists.

## Test only after checkpoint selection

Once training and checkpoint selection are finished:

```bash
bash .local/run/diversity-v2-01/test.sh
cat .local/run/diversity-v2-01/final-test/baseline-report.md
```

This compares the pinned starting model with the selected training `best.zip`
on the reserved test suite: 3,000 battles each, exact snapshots and a paired
comparison. It cannot overwrite existing test output. Stop training while doing
the final test. Do not use these results to repeatedly select checkpoints;
once used to guide more changes, this test becomes development data.

Record the result in the [experiment ledger](README.md). Improvement here would
support transfer on this generator distribution. The experiment changes both
pool size and composition; it cannot isolate size alone or establish superiority
across independent training seeds. Do not increase the budget automatically if
validation has plateaued; review the scheduled comparison first.
