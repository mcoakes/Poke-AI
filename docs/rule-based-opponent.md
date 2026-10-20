# Rule-based opponent

To mix rules with one or more frozen agents, see [opponent pools](opponent-pools.md).

[Training](ppo-training.md) · [Evaluation](evaluation.md) · [Experiment record](experiments/README.md)

`--rule-based-opponent` selects **gen3-rules-v1**, an optional, deterministic
Showdown opponent. It uses hand-written scores, not learned weights. Existing
training defaults, campaigns, rewards and team-diversity scripts are unchanged.
Finish an experiment before replacing its opponent; use a new output directory
for this intervention.

The design is inspired by FireRed's [move-scoring scripts](https://github.com/pret/pokefirered/blob/master/data/battle_ai_scripts.s)
and [switching logic](https://github.com/pret/pokefirered/blob/master/src/battle_ai_switch_items.c).
Thanks to pret's contributors for documenting those systems. This is our own
small heuristic, **not a port or a cartridge-exact trainer AI**. Its strength
and usefulness for learning must be measured.

## What it sees and does

The opponent receives the existing full-information v2 observation, including
both parties' moves/stats/items. It cannot read the other player's submitted
action, future RNG, simulator objects or rewards. A v1 learner still receives
only v1; using the rules does not migrate a checkpoint. Showdown supplies v2 to
the rules internally. mGBA is not supported for this opponent.

It considers only actions permitted by Showdown's mask, including sparse move
slots, trapping, forced replacement, continuation and partner wait. Ties favor
the lowest action ID. Its scores consider:

- Approximate direct damage using level, stats, stages, type, STAB and accuracy;
  likely knockouts receive a bonus. Burn, sun/rain and several common abilities
  and Choice Band are included.
- Toxic/Poison Powder, paralysis, sleep and Will-O-Wisp, avoiding represented
  existing status, Substitute and supported immunities.
- Recovery when missing HP and estimated incoming damage make it useful; Rest
  receives a sleep cost. Swords Dance, Growth, Agility, Bulk Up, Calm Mind and
  Dragon Dance require an HP buffer and stop being preferred at +2.
- Switches based on incoming/outgoing damage estimates, remaining HP and a
  lost-turn cost. Forced replacements omit that cost. There is no random switch
  injection, switch reward or persistent cooldown.

These are estimates, not an exact damage calculator or a search over future
turns. Incoming damage uses the strongest estimated known attack, without the
enemy's action mask. Charge/recharge timing, multi-hit behavior, critical hits,
most items/abilities, hazards, screens, and many special effects are not fully
modeled. Unsupported variable-damage and status moves receive fallback scores;
the policy can still use them when the mask leaves no alternative. Conservative
switch costs reduce gratuitous switches but cannot guarantee absence of cycles.
Check truncations and spectator logs before using it as training pressure.

## Evaluate first

After the diversity experiment finishes, use its selected model and validation
teams. These commands create separate results and keep the same reported rewards:

```bash
diversity_run=".local/run/diversity-v2-01"
checkpoint="$diversity_run/train/checkpoints/best.zip"
rule_eval=".local/run/eval-rules-v1-01"

python mgba/evaluate_ppo.py \
  --backend showdown --checkpoint "$checkpoint" --rule-based-opponent \
  --team-dir "$diversity_run/teams/validation" \
  --reward-config "$diversity_run/inputs/reward.json" \
  --episodes 1000 --seed 26093001 --environments 48 \
  --showdown-workers process \
  --rule-log --battle-log --no-readable-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --output-dir "$rule_eval"

cat "$rule_eval/summary.json"
```

The default side swaps play 500 matchups twice. `candidate_win_rate` refers to
the PPO checkpoint; `opponent_win_rate` refers to the rules. Compare truncations,
battle length and representative decisions as well as wins. This establishes a
new benchmark; it is not directly comparable to a Random/Dumb/Smart pooled rate.

Replay the exact teams/seeds with another candidate:

```bash
python mgba/evaluate_ppo.py \
  --checkpoint path/to/another-checkpoint.zip \
  --suite "$rule_eval/suite.json" \
  --reward-config "$diversity_run/inputs/reward.json" \
  --environments 48 --showdown-workers process \
  --battle-log --no-step-logs --log-max-mb 2 --log-backups 1 \
  --output-dir .local/run/eval-rules-v1-02
```

The saved suite pins the rules' version and code/catalog fingerprint. Replay
automatically selects that opponent and rejects a changed implementation.
Choose a new directory for every evaluation. A saved Random/Dumb/Smart pool
cannot be overridden with this flag; it keeps its original opponents.

## Separate training experiment

Once the baseline evaluation has been reviewed, an optional short fork is:

```bash
python mgba/train_ppo.py \
  --backend showdown --resume "$checkpoint" --rule-based-opponent \
  --team-dir "$diversity_run/teams/training" \
  --reward-config "$diversity_run/inputs/reward.json" \
  --learning-rate 0.0001 --rollout-steps 128 --batch-size 128 \
  --environments 64 --showdown-workers process --total-timesteps 100000 \
  --eval-suite "$diversity_run/suites/validation.json" --eval-every 50000 \
  --rule-log --battle-log --no-readable-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --checkpoint-every 25000 --keep-checkpoints 2 --keep-evaluations 2 \
  --output-dir .local/run/ppo-rules-v1-01 --if-exists cancel
```

Only Player 1 learns: 64 environments means 32 learning players and 32 scripted
opponents. This first implementation uses the rules for **every training
battle**, at the teams' existing levels. For a configurable episode mixture,
use [opponent pools](opponent-pools.md) instead. Level handicaps are not
implemented. Keep the old heuristic switch flags off.

The example selects `best.zip` on the existing Random/Dumb/Smart validation
pool, so beating the heuristic alone is not the selection criterion. Re-evaluate
the selected model on the saved rule suite too. A rule suite also works as
`--eval-suite` if specifically testing progress against the rules. Changing
training opponents can cause forgetting; retain the starting checkpoint and
compare both benchmarks before promoting anything. Reserve fresh final-test
data for the eventual selected methodology.

Same-directory continuation restores the rule opponent and rejects changed
code/catalog fingerprints:

```bash
python mgba/train_ppo.py --output-dir .local/run/ppo-rules-v1-01 \
  --if-exists continue --total-timesteps 100000
```

## Flags, files and decision evidence

| Option | Default | Applies to |
|---|---|---|
| `--rule-based-opponent` | Off | Training/evaluation. Showdown only; mutually exclusive with a checkpoint opponent (and Random in evaluation). |
| `--rule-log`, `--no-rule-log` | Off | Training/evaluation with rules. Writes action scores and reasons to `rule-decisions.jsonl`; can be disabled on continuation. |
| `--log-max-mb`, `--log-backups` | 10 MiB, 2 backups | Also bound the rule-decision log; use 2 MiB and 1 backup in examples. |

Training logs live in the run root for the first session and its session
directory on continuation. Evaluation logs live in its output directory.
Records identify pair, episode, player, turn, species and candidate action
scores. Scores are policy preferences, not rewards. The unchanged spectator
`pair-N/battle.log` shows what actually happened. No decision logs are written
unless requested.

Implementation: [`rule_opponent.py`](../src/firered_rl/rule_opponent.py).
Static names/IDs: [`firered_us_v1_0.json`](../config/generation/firered_us_v1_0.json).
Effectiveness: [`gen3_type_chart.json`](../config/generation/gen3_type_chart.json).
Dynamic power/category/accuracy/stats come from the native v2 observation.
The rule identity appears in training `config.json`, evaluation `config.json`
and rule evaluation `suite.json`. Weights are never created for the rules.

Tests cover tactical fixtures, sparse/forced actions, bounded logs and real
Showdown training, continuation and replay. Passing these verifies those
fixtures and integration paths; it does not establish competitive strength.
