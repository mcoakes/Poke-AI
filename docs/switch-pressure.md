# Train against an opponent that switches

[Campaign walkthrough](creating-campaigns.md) · [Training flags](reference/train-ppo.md) · [Campaign flags](reference/run-curriculum.md)

The `type_progression_switch_pressure` preset reuses the original ten team
stages and previous-stage frozen opponents. A probabilistic type-matchup rule
can replace Player 2's training action with a legal switch. Player 1 chooses
every action through PPO. Rewards and checkpoint schemas remain the same.

## Opponent behavior

The default is a **30% chance to check**, not a 30% switch rate. The replacement
must be a legal, non-active teammate with at least 35% HP and a damaging move
with PP. It must hit super effectively or resist every considered incoming
attack type, have no worse incoming type multiplier than the current Pokémon,
and improve its matchup score by at least one point.

The score is `log2(best outgoing multiplier) - log2(worst incoming multiplier)
+ log2(HP fraction)`. A zero multiplier uses 0.25 for scoring to keep it finite;
eligibility still recognizes immunity. Both defending types are applied,
without double-counting a single type. The Gen III chart includes Steel's old
Ghost/Dark resistances. Attack types come from damaging moves with remaining
PP, including coverage. If no opponent attack is known, its Pokémon types are
used as a fallback. Hidden Power and Weather Ball are excluded because their
effective types are not reliably available in this observation.

After any voluntary opponent switch, the rule skips **two normal decisions**.
The underlying policy can still switch during this cooldown. Forced replacements
and partner-wait intervals are left to the policy and do not consume cooldown;
each new battle resets it. The rule does not model abilities, hazards, boosts,
speed, accuracy or predicted knockouts.

There is no reward for switching itself. Start with 30% intervention checks:
the other 70% of eligible decisions retain the frozen policy's choice, and
staying in to attack can still be the learner's best response.

## Preserve the old campaign and start this one

Keep the successful run at its original path. These commands reuse its
**original starting candidate, initial frozen opponent, and rewards**, then
write new models and reports to a different directory. No old run files are
modified or copied over. This also avoids duplicating its large battle logs.

```bash
old_campaign=".local/run/curriculum-smart-generations-01"
curriculum_run=".local/run/curriculum-switch-pressure-01"
fronzen_checkpoint="$old_campaign/inputs/opponent.zip"

python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --curriculum config/curricula/type_progression_switch_pressure/curriculum.json \
  --resume "$old_campaign/inputs/start.zip" \
  --opponent-checkpoint "$fronzen_checkpoint" \
  --reward-config "$old_campaign/inputs/reward.json" \
  --opponent-switch-chance 0.30 --opponent-switch-cooldown 2 \
  --train-battles 1000 --eval-episodes 1000 \
  --eval-opponents mixed \
  --promotion-basic-win-rate 0.75 --promotion-smart-win-rate 0.52 \
  --promotion-require both \
  --through-stage 10 --attempts-per-stage 10 \
  --environments 48 --showdown-workers process \
  --rollout-steps 128 --batch-size 128
```

This uses the previously chosen 75% basic / 52% smart evaluation gates.
Choose a new run-directory name if this one already exists. To continue
improving the strongest old model instead of repeating the original starting
point, change `--resume` to that model's path, for example
`$old_campaign/models/smart_v11.zip`.

Continue with the saved settings:

```bash
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" --through-stage 10 --attempts-per-stage 10
```

Probability and cooldown are pinned at campaign creation; omit those flags on
resume. `0.10` means 10%, `0.30` means 30%, and `0` disables intervention.
The new preset defaults to 0.30/2; original presets default to 0/2. A manifest
can set `"opponent_switch": {"chance": 0.3, "cooldown": 2}`; CLI flags override it.

For standalone training, add the same two flags to `train_ppo.py` with an
`--opponent-checkpoint`. Both backends accept them; this campaign uses Showdown.
The rule is not embedded in a checkpoint, so include the flags when starting
another standalone training run, even with `--resume`.

## Measure progress

Each PPO update prints voluntary switch rates for P1 and P2. The denominator
is normal move-selection decisions; forced replacements and waiting are
excluded. P2 includes the policy and heuristic; P1 always reflects PPO choices.

Inside each `stage-NN/attempt-NNN/train/` directory:

| File | Contents |
|---|---|
| `switching.json` | Cumulative normal-decision, voluntary-switch and forced-switch counts, rates, and override count; updated per rollout and at exit. |
| `opponent-switches.jsonl` | Actual overrides: pair, episode, turn, original action, new slot and matchup scores. Created on the first override. |
| `ppo/progress.csv` | Switch counters and rates under `switching/`, alongside existing PPO metrics. |
| `config.json` | Probability, cooldown and heuristic version. |
| `summary.json` | Final statistics, including `switching`. |

Evaluation uses **unmodified policies**, including the mixed Random/Dumb/Smart
pool. Human play and exported checkpoints contain no forced-switch rule.
Watch the campaign's `evaluations.md` with `tail -F` as before. Compare its
checkpoints with the old models on the same held-out suite or round-robin plan.

More switches alone are not success. Look for improved evaluation results and
sensible switches out of disadvantaged matchups. If results stay flat, inspect
the new counters and representative battles before raising pressure or changing
rewards.

Implementation: [opponent rule](../src/firered_rl/switch_opponent.py),
[wrapper](../src/firered_rl/frozen_opponent.py), and
[type chart](../config/generation/gen3_type_chart.json), exported from the pinned
local Pokémon Showdown 0.11.11 `Dex.mod('gen3')` data.
