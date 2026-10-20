# Shared move-scoring experiment

[Experiment record](README.md) · [Training flags](../reference/train-ppo.md) ·
[Policy source](../../src/firered_rl/shared_move_policy.py)

The move-shuffling experiment reduced measured slot preference but did not
improve validation. This experiment tests a stronger constraint: **the same
move must receive the same score when moved to another slot**, with its PP,
details and legality moved alongside it.

## What changes

`--policy-architecture shared-moves` uses one shared scorer for all four moves.
Each score combines that move's features with the current battle context.
Context pools moves within each party member without assigning weights to
particular move positions. Switch scores and the value estimate use this
order-independent context. Species, HP, stats, statuses, abilities, items,
weather and the other v2 inputs remain available.

The external v2 observation space and 11 actions stay the same. Action 0 still
means the move currently in slot 1. Masks still reject illegal actions. One to
four populated moves, empty slots, disabled moves, forced actions and transformed
move sets use the existing observation contract. Party positions remain ordered;
this is not a redesign of switch scoring or a new battle engine.

Permuting moves, their details and their legal bits permutes their probabilities
to floating-point tolerance, while preserving other action probabilities and
the value estimate. Exact ties can still choose the first legal slot under
deterministic inference. The constraint removes a positional shortcut; it does
not supply tactical rules or guarantee a better win rate.

## Checkpoints and compatibility

This architecture requires **fresh weights**. Do not add `--resume` to the
comparison below: existing flat networks have different parameter layouts and
are not converted. Keep `models/observation-v2-diversity-best-01.zip` as the
trained reference.

The flag is optional. Fresh runs default to `flat`; `shared-moves` selects v2
unless an incompatible schema is explicitly requested. The comparison below uses Showdown.
Checkpoints save their policy class and optimizer; evaluation and continuation
load the architecture automatically. Flat and shared models can face one another
as frozen opponents. For native play, [mGBA v2](../mgba-observation-v2.md)
requires adapter 0.19.0 on both players.

## Run the controlled comparison

Run from the project root with the training environment and Showdown installed.
These paths reuse the completed diversity experiment's training teams, rewards
and the combined Random/Dumb/Smart/Rules development suite. They refer to files
on your machine, not bundled checkpoints.

The two commands run sequentially. Both models start fresh, with the same seed,
training teams, rules opponent, rewards and training budget. Leave move-slot
shuffling off for this comparison. A shared seed does not imply identical
trajectories once the policies choose different actions.

```bash
experiment_root=".local/run/move-scorer-v2-01"
diversity_run=".local/run/diversity-v2-01"
combined_suite=".local/run/validation-rules-and-agents-01/suite.json"

for architecture in flat shared-moves; do
  python mgba/train_ppo.py \
    --backend showdown --observation-version v2 \
    --policy-architecture "$architecture" \
    --rule-based-opponent \
    --team-dir "$diversity_run/teams/training" \
    --reward-config "$diversity_run/inputs/reward.json" \
    --learning-rate 0.0001 --rollout-steps 128 --batch-size 128 \
    --epochs 4 --ent-coef 0.01 --seed 261003 \
    --environments 64 --showdown-workers process \
    --total-timesteps 200000 \
    --eval-suite "$combined_suite" --eval-every 50000 \
    --battle-log --no-readable-log --no-step-logs \
    --log-max-mb 2 --log-backups 1 \
    --latest-every-updates 5 --checkpoint-every 25000 \
    --keep-checkpoints 2 --keep-evaluations 2 \
    --output-dir "$experiment_root/$architecture" --if-exists cancel \
    || break
done
```

64 player environments give 32 learning players against 32 rules opponents.
Each update collects 4,096 learning transitions. The budget rounds up to
200,704 transitions per model. Validation runs at 0, 53,248, 106,496, 159,744 and
200,704 timesteps. Neither initial model should reproduce the trained diversity
reference: they start untrained.

Spectator logs are retained with rotation. Reward means, win rates, throughput,
checkpoint timings and evaluation history remain available. Parameter counts
and the architecture version are printed and saved in `config.json`; this first
comparison does not equalize parameter counts between architectures.

## Read the results

```bash
cat "$experiment_root/flat/evaluations.md"
cat "$experiment_root/shared-moves/evaluations.md"
```

Compare the final rows at the same training budget, and separately the best rows.
`best.zip` is selected by overall wins/requested battles, so also inspect Rules
and the three other groups. Better slot invariance alone is not an improvement
in battle skill. Check truncations and throughput alongside the win rates.

The archived diversity reference scored 59.175% overall and 35.4% against Rules
on this combined suite. It is a practical benchmark with much more training,
not the architecture control. The fresh flat run is that control.

Treat this as one screening experiment. If shared scoring improves battle
results, repeat with additional training seeds and fresh held-out teams before
making a generalization claim. If both fresh models remain weak, inspect their
learning curves before spending more time or changing rewards.

The move-order audit can also inspect a completed evaluation directory shown
in `evaluation-history.json`. Shared scoring should have near-zero aligned
probability change; the audit's same-slot count alone can include tied choices.
The automated checks already cover all 24 orders, populated/empty/disabled moves,
both parties, optimizer updates and serialization. An audit of actual battles
adds coverage of the positions this model encounters.

## Preserve the experiment

After both runs finish, archive their best and final models without replacing
the diversity reference:

```bash
mkdir -p models
for architecture in flat shared-moves; do
  cp -n "$experiment_root/$architecture/checkpoints/best.zip" \
    "models/observation-v2-${architecture}-best-01.zip"
  cp -n "$experiment_root/$architecture/checkpoints/final.zip" \
    "models/observation-v2-${architecture}-final-01.zip"
done
```

Retain each run's `config.json`, `evaluation-history.json` and `evaluations.md`
alongside these model paths. A best checkpoint can be an earlier or initial
model; use its evaluation row to identify the exact training timestep.

If a run is interrupted, continue that directory instead of repeating the loop:

```bash
python mgba/train_ppo.py \
  --output-dir "$experiment_root/shared-moves" --if-exists continue \
  --total-timesteps 100000
```

That example requests **100,000 additional** transitions, so adjust the remaining
budget to preserve a fair comparison. Architecture, schema, opponent and other
saved settings are restored. Assess the planned endpoint before extending it.
