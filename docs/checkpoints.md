# Selecting checkpoints

[Training](ppo-training.md) · [Evaluation](evaluation.md) · [Human play](human-vs-ai.md)

With [periodic evaluation](periodic-evaluation.md), `checkpoints/best.zip` preserves
the strongest validated model. `latest.zip` and `final.zip` keep their existing
training-progress meanings; same-directory continuation does not roll back to
`best.zip`. To start a separate experiment from the best, pass its explicit path
to `--resume` and choose a new output directory with the desired training flags.

Standalone evaluations preserve the exact loaded candidate in `candidate.zip`,
with its original path, hash and snapshot path in the evaluation's `config.json`.

## Observation compatibility

Checkpoint spaces identify v1 or v2 automatically. New models use v1 unless
`--observation-version v2` or `--policy-architecture shared-moves` is supplied to
the Showdown trainer. Resume preserves
the saved schema; an explicit conflicting version is rejected. The versions
have different feature networks, so v1 weights are not implicitly converted to
v2. See [observation v2](observation-v2.md) for training and evaluation examples.

Showdown frozen opponents and evaluations can mix versions, with each model
receiving its own inputs. Native linked mGBA training, evaluation and human play
also accept v2 with adapter 0.19.0 on both players. See [native v2](mgba-observation-v2.md).

## Policy architecture compatibility

The checkpoint's saved policy class selects `flat` or `shared-moves` when loading,
resuming, or evaluating. Existing checkpoints remain `flat`; they cannot be
converted by adding `--policy-architecture shared-moves` to `--resume`.
Start a new model without `--resume` to change architecture. An explicit
conflicting architecture is rejected, including during same-directory continuation.

`shared-moves` uses the same v2 observation/action contract and can face frozen
flat models in Showdown. Its weights, optimizer and architecture are saved together.
See the [experiment guide](experiments/shared-move-policy.md). Both architectures
can use the [native v2 adapter](mgba-observation-v2.md).

## Continue the same run

```bash
python mgba/train_ppo.py \
  --output-dir .local/run/ppo-observation-v2-01 \
  --if-exists continue --total-timesteps 400000
```

This restores the run's configuration and adds another 400,000 transitions.
No `--resume` argument is needed. The runner selects the recoverable checkpoint
with the highest saved `num_timesteps` **inside this run**, so an older
`final.zip` does not take precedence over newer training progress. It does not
use the global `latest` search described below. Weights and optimizer state
resume; battles start fresh.

Frozen opponents are pinned in `inputs/frozen-opponent.zip`, so continuing the
learner cannot silently replace its opponent with new weights. The run retains
earlier metrics and starts separate session logs. Matching flags may be repeated;
explicit PPO overrides still work. To change teams, rewards, backend, schema or
frozen opponent, use a new output directory with `--resume`.

When an existing output directory is encountered without an explicit choice,
the interactive prompt offers continue, erase or cancel. Cancel is the default;
scripts must set `--if-exists`. See [the training guide](ppo-training.md#continue-an-existing-run)
for recovery, logging and erase behavior.

## Select a file for a new run

Every PPO checkpoint argument accepts a file path or `latest`:

| Utility | Arguments |
|---|---|
| `mgba/train_ppo.py` | `--resume`, `--opponent-checkpoint` / `--player2-checkpoint` |
| `mgba/evaluate_ppo.py` | `--checkpoint`, `--opponent-checkpoint` / `--player2-checkpoint` |
| `mgba/play_agent.py` | `--checkpoint` |
| `mgba/play_gui.py` | `--checkpoint`; also the browser's Latest saved checkpoint choice |

`latest` means the most recently modified completed PPO save under the project's
`.local/run` directory. Candidates are `checkpoints/latest.zip`,
`checkpoints/final.zip`, and `checkpoints/ppo_<number>.zip`; modification time
determines recency, not the largest timestep number. Temporary saves and copied
human-play snapshots are excluded. For a custom GUI `--checkpoint-dir`, search
that tree, also accepting these filenames directly in its root. `later` is
accepted as an alias for `latest`.

Resolution happens once at launch. Both `latest` arguments in one command
resolve to the same path, and each unique checkpoint is read once into memory.
The frozen opponent does not follow subsequent training updates. A GUI Latest
selection resolves again for each new match and then freezes that match's model.
Resolved paths and checkpoint SHA-256 values are recorded in run configuration.

```bash
frozen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"

python mgba/train_ppo.py --backend showdown --environments 48 \
  --training-profile six_types --resume latest \
  --opponent-checkpoint "$frozen_checkpoint" --total-timesteps 100000
```

The shell variable above intentionally points at one particular training run.
An explicit numbered checkpoint is preferable if you need to preserve the
reference even when another process may replace that run's `latest.zip`.
Global `latest` can select a different experiment; check the printed path or use
an explicit path when comparing a specific candidate.

`--resume` restores policy weights, optimizer state and counters. Explicit
`--rollout-steps`, `--batch-size`, `--epochs`, `--learning-rate` and `--ent-coef`
values override the corresponding saved settings. Omitted flags retain the
checkpoint values; ordinary CLI defaults apply only to new models.

Changing rollout length rebuilds the rollout buffer for the new length. Changing
the learning rate updates its schedule and optimizer learning rate, while
retaining optimizer history. The frozen opponent is unaffected. The launcher
prints the overrides and effective settings; `config.json` records both
`resume_overrides` and `effective_ppo`.

```bash
python mgba/train_ppo.py --backend showdown --environments 48 \
  --training-profile six_types --resume latest \
  --opponent-checkpoint "$frozen_checkpoint" \
  --rollout-steps 128 --batch-size 128 --total-timesteps 100000
```
