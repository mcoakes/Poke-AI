# Selecting checkpoints

[Training](ppo-training.md) · [Evaluation](evaluation.md) · [Human play](human-vs-ai.md)

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
fronzen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"

python mgba/train_ppo.py --backend showdown --environments 48 \
  --training-profile six_types --resume latest \
  --opponent-checkpoint "$fronzen_checkpoint" --total-timesteps 100000
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
  --opponent-checkpoint "$fronzen_checkpoint" \
  --rollout-steps 128 --batch-size 128 --total-timesteps 100000
```
