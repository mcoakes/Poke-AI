# Train PPO

[End-to-end quickstart](quickstart.md) · [Documentation](README.md) · [All training flags](reference/train-ppo.md) · [Reports](battle-logs.md)

The runner uses MaskablePPO with one shared policy by default. Each battle has two
Gymnasium player environments: four environments means two simultaneous battles.
A vector step collects one transition per learning player, not one emulator frame.

## Start a run

For local simulation, complete [Showdown setup](setup/showdown.md), then run:

```bash
python mgba/train_ppo.py --backend showdown --environments 4 \
  --teams config/teams/starters_charizard_lead.json config/teams/starters_venusaur_lead.json \
  --rollout-steps 128 --batch-size 128 --total-timesteps 100000
```

For FireRed, complete [mGBA setup](setup/mgba.md) and [build counter states](automatic_states.md):

```bash
python mgba/train_ppo.py --backend mgba --rom FireRed.gba --environments 4 \
  --states states/counter/starters_charizard_lead.ss1 states/counter/starters_venusaur_lead.ss1 \
  --speed unlimited --headless --total-timesteps 100000
```

The runner owns the battle lifecycle. You do not need a separate pool launcher,
emulator PID or manual opponent. Showdown resets in memory; mGBA restarts each
native pair from its counter states and reconnects through the NPC.

## Select teams

Choose exactly one source. By default, random sources draw a fresh pair each
episode, avoiding the same file for both players when alternatives are available.
`--mirror-teams` instead assigns the same randomly selected team to both sides.

| Source | Showdown | mGBA |
|---|---|---|
| Fixed pair | `--teams A.json B.json` | `--states A.ss1 B.ss1` |
| Random directory | `--team-dir DIR` | `--state-dir DIR` |
| Included categories | `--team-profile competitive`, `casual`, `random`, or `mixed` | Same flags, using built counter states |

`--team-size 1` through `6` is optional and only applies with `--team-profile`.
Without it, all available sizes participate. `mixed` selects the three categories
with equal probability for each player. Directory sources scan direct children,
not nested profile trees. See [team profiles](team-profiles.md).

```bash
python mgba/train_ppo.py --backend showdown --environments 8 \
  --team-profile mixed --team-size 3 \
  --rollout-steps 128 --batch-size 256 --total-timesteps 100000
```

## Resume training

```bash
checkpoint=".local/run/YOUR_RUN/checkpoints/latest.zip"

python mgba/train_ppo.py --backend showdown --environments 4 \
  --team-profile mixed --team-size 3 --resume "$checkpoint" \
  --total-timesteps 100000
```

Replace `YOUR_RUN` with the printed run directory, or use `--resume latest`
to select the newest completed PPO save. See [checkpoint selection](checkpoints.md). Resume restores policy,
optimizer and counters; battles start fresh. Explicit PPO hyperparameter flags
override saved values, while omitted flags retain them.
`--total-timesteps` adds that many learning-player transitions, rounded up to a
complete rollout. It is not the desired final counter value.

Checkpoint transfer between backends uses the same observation/action schema.
For mGBA, change the backend and supply the ROM and matching state source.
Team JSONs cannot replace `.ss1` inputs. Evaluate transferred policies in FireRed;
the two engines are not guaranteed equivalent for every mechanic.

## Freeze player 2

```bash
checkpoint=".local/run/YOUR_RUN/checkpoints/latest.zip"

python mgba/train_ppo.py --backend showdown --environments 4 \
  --team-profile mixed --team-size 3 \
  --resume "$checkpoint" --opponent-checkpoint "$checkpoint" \
  --opponent-stochastic --total-timesteps 100000
```

Only player 1 learns. The player-2 model loads once and remains unchanged even
while the learner writes new checkpoints. `--opponent-stochastic` samples the
frozen policy; omitting it selects its highest-probability legal action. PPO's
learner samples during collection, so omitting this flag creates a sampling
asymmetry even when both players began from the same weights.

To use different initial models, give `--resume` the learner checkpoint and
`--opponent-checkpoint` the opponent checkpoint. Omit the opponent flag for shared
self-play. The alias `--player2-checkpoint` has the same meaning.

## Rollouts, batches and time limits

| Setting | Meaning |
|---|---|
| Rollout steps | Transitions per learning player before an update |
| Batch size | Transitions in each optimization minibatch |
| Epochs | Passes over a collected rollout during an update |
| Total timesteps | Additional learning-player transitions for this run |
| Max steps | Paired decisions per battle before time-limit truncation |

Four learning players × 128 rollout steps = 512 transitions per update. Batch
size 128 divides that into four minibatches. With a frozen opponent there are
only two learning players, producing 256 transitions instead.

Batch size must be at least two and divide rollout steps × learning players.
On resume, explicitly supplied rollout, batch, epoch, learning-rate and entropy
flags override the saved values. Omitted flags retain their checkpoint values.
Use the effective rollout and batch values when choosing an environment count.
Rewards are selected afresh from `--reward-config`, including on resume.

The default decision limit is 200. Counter limits also truncate unfinished
battles. Truncation keeps the final observation for value bootstrapping; a genuine
win/loss/draw is terminal. Protocol errors stop the run rather than becoming
invented episode results.

## Progress and saving

A run prints its output directory, then reports episode reward means and
completed battles/hour after episodes end. Before the first ending, means are
unavailable. Each completed PPO update writes `checkpoints/latest.zip`;
`--checkpoint-every` controls numbered snapshots and successful completion writes
`final.zip`. Interrupting collection preserves the last completed update.

Read [rewards](training-rewards.md), [observations/actions](turn-environment.md)
and [logs and metrics](battle-logs.md) for the full contracts. See
[every training flag](reference/train-ppo.md) for defaults and accepted options.

## Non-overlapping comparisons

Training prints a separate summary every 500 ended battles. See
[500-battle block comparisons](battle-blocks.md) to review those results or analyze
an existing run without restarting it.

Use [random mirror matches](team-profiles.md#random-mirror-matches) with
`--mirror-teams` to give both players the same randomly selected team each battle.

## Controlled teams and evaluation

For a gradual expansion from one team, use the [ten-stage curriculum](curriculum.md).
It retains earlier teams, evaluates each checkpoint on the current and expanded
pools, and advances only when the configured checks pass.

Use `--training-profile six_types` instead of the category flags for six Pokémon
with different types and same-type attacks. The teams live in
`config/training_profiles/six_types/teams/`. New directories under
`config/training_profiles/NAME/teams/` work without code changes.

```bash
fronzen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"
python mgba/train_ppo.py --backend showdown --environments 48 \
  --training-profile six_types --resume latest \
  --opponent-checkpoint "$fronzen_checkpoint" --total-timesteps 100000
```

Resume preserves omitted hyperparameters and applies explicit overrides; the startup `Effective PPO` line shows
the values actually used. Measure improvement with the
[checkpoint evaluator](evaluation.md), which freezes both policies and reuses
the same matchup suite. Training win rates include exploration.
