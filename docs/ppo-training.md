# Train PPO

[End-to-end quickstart](quickstart.md) · [Documentation](README.md) · [All training flags](reference/train-ppo.md) · [Reports](battle-logs.md)

The runner uses MaskablePPO with one shared policy by default. Each battle has two
Gymnasium player environments: four environments means two simultaneous battles.
A vector step collects one transition per learning player, not one emulator frame.

An optional [rule-based opponent](rule-based-opponent.md) is available for
separate Showdown experiments; it does not change the default opponents.
Use [opponent pools](opponent-pools.md) to mix rules and frozen learned agents
with configurable per-battle probabilities.

New runs with the default `flat` architecture use observation v1. For richer Showdown inputs, use
`--observation-version v2`; see [the v2 guide](observation-v2.md) for a fresh
training run, mixed-version frozen opponents and matched evaluations. Resumed
checkpoints select their saved schema automatically.

The optional `--policy-architecture shared-moves` scores all four move slots
with one shared network and selects v2 inputs. It requires fresh weights;
resuming a checkpoint keeps its saved architecture. See the
[controlled architecture comparison](experiments/shared-move-policy.md) for
commands, compatibility and what to measure.

`--policy-architecture matchup-moves` adds explicit type-matchup and supported
ability-immunity inputs to that learned scorer, using v2 observations. Fresh
models enable these inputs by default; resumed models retain their saved toggle.
For a controlled feature-off/feature-on experiment with equal initial predictions,
use [the matchup ablation runner](experiments/matchup-ablation.md). It performs the
restricted conversion from an untrained shared-moves checkpoint and keeps every
legal action available.

`--policy-architecture matchup-switches` adds a shared candidate scorer on
forced replacements with at least two legal choices. It uses visible health,
stats and matchup features; it never invokes Rules to pick the action. The
[replacement-feature comparison](experiments/switch-feature-ablation.md) migrates
a trained matchup-moves model with identical initial predictions and preserved
existing Adam state. Normal `--resume` still rejects architecture changes.
Both feature-off and feature-on arms continue training the whole policy.

For the next comparison of longer PPO against fresh, decision-weighted teacher
practice followed by longer PPO, see [switch practice](experiments/switch-practice.md).
Both start from the completed [teacher training](experiments/teacher-training.md)
selected model and retain the expanded pool, rehearsal and broader validation suite. The guide includes
two-job launch, live logs, memory monitoring, team inspection, pause/resume and
permanent model-copy commands.

For automatic progress checks and a separately preserved `best.zip`, see
[periodic evaluation](periodic-evaluation.md). It reuses a fixed validation suite
and retains bounded evaluation logs across continued Showdown runs.

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

## Continue an existing run

To pause, press **Ctrl+C once in the training terminal** and wait for cleanup
and the shell prompt. The trainer preserves the last saved checkpoint; updates
and rollout progress since that save are lost. It does not save unfinished work
on interruption. The stop message reports the saved timestep count, or explains
that no trained checkpoint exists if stopped before the first save.

The continuation budget below is **additional**, not the remainder of the
interrupted budget. See the quickstart's
[monitor, pause, and continue workflow](quickstart.md#monitor-pause-and-continue-training)
for commands, including a separate `watch` terminal for validation results.

Use the same output directory to restore its saved training configuration and
latest recoverable checkpoint:

```bash
python mgba/train_ppo.py \
  --output-dir .local/run/ppo-observation-v2-01 \
  --if-exists continue \
  --total-timesteps 400000
```

This adds **400,000 more learning-player transitions**, rounded up to a complete
rollout. It restores the backend, teams, frozen opponent, rewards, observation
version, PPO settings and logging settings. You do not need to repeat those flags
or set `--resume`. The chosen checkpoint comes from this run's `checkpoints/`
directory, using its saved timestep counter rather than file modification time.

Without `--if-exists`, an existing `--output-dir` presents three choices in an
interactive terminal: **continue**, **erase and start over**, or **cancel**.
Pressing Enter cancels. Noninteractive commands must choose explicitly.

Continue restores weights, optimizer state and timestep counters, then starts
fresh battles. It does not restore live battles or reproduce the interrupted
random sequence exactly; the new session seed is recorded. Cumulative battle
counts, update counts and throughput continue across sessions; throughput excludes
time between invocations. Earlier logs remain available, and each continuation
writes its detailed pair logs and PPO CSV to `sessions/000002/` and onward.

You may repeat matching configuration flags or explicitly adjust PPO
hyperparameters, environment count, performance controls, logging and the new
training budget. Changing the team source, reward settings, backend, observation
schema or frozen opponent starts a different experiment: use a new directory
with `--resume` instead. Concurrent writers are rejected by a run lock. Continue
campaigns with `run_curriculum.py`, not by resuming an individual attempt in place.

Erasing requires a recognized standalone PPO run and a valid fresh training
configuration. Choosing erase from the prompt also requires typing the run name
or path. Explicit `--if-exists erase` authorizes deletion without another prompt:
old logs and checkpoints are removed. Cancel leaves the run unchanged. See
[storage](storage.md#standalone-run-sessions).

## Start a new run from a checkpoint

```bash
checkpoint=".local/run/YOUR_RUN/checkpoints/latest.zip"

python mgba/train_ppo.py --backend showdown --environments 4 \
  --team-profile mixed --team-size 3 --resume "$checkpoint" \
  --total-timesteps 100000
```

Replace `YOUR_RUN` with the printed run directory, or use `--resume latest`
to select the newest completed PPO save. This creates a new run directory; it
does not append to the checkpoint's original run. See [checkpoint selection](checkpoints.md). Resume restores policy,
optimizer and counters; battles start fresh. Explicit PPO hyperparameter flags
override saved values, while omitted flags retain them.
`--total-timesteps` adds that many learning-player transitions, rounded up to a
complete rollout. It is not the desired final counter value.

V1 checkpoint transfer between backends uses the same observation/action schema.
V2 checkpoints currently require Showdown and cannot run in mGBA human play.
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

For a policy that depends too much on move position, Showdown training supports
`--shuffle-move-slots`. Each learner sees a random order for each party member's
moves, fixed throughout that battle. The environment maps selected slots back to
the simulator and moves the matching features and legality mask together.
Frozen opponents retain their original inputs. Evaluation uses the original
team order and the observation schema stays compatible with existing checkpoints.
The option defaults off; see the [controlled experiment](experiments/move-slot-shuffle.md)
for commands, diagnostics and interpretation. Same-directory continuation restores
the setting; switching it on or off requires a new run directory.

For party-slot dependence, `--shuffle-bench-order` is a separate Showdown/Rules
option. It samples the original team pair first and then permutes only the
learner's occupied bench, retaining its lead and the opponent's order. Fixed-pair
rehearsal groups retain their original order. It defaults off and is restored on
continuation. The [bench-order comparison](experiments/bench-order-ablation.md)
runs matched control and treatment jobs with balanced validation, monitoring and
a fixed budget.

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
New runs, including forks with `--resume`, select rewards from `--reward-config`.
Same-directory continuation restores the saved rewards and rejects changes.

The default decision limit is 200. Counter limits also truncate unfinished
battles. Truncation keeps the final observation for value bootstrapping; a genuine
win/loss/draw is terminal. Protocol errors stop the run rather than becoming
invented episode results.

## Progress and saving

A run prints its output directory, then reports episode reward means and
completed battles/hour after episodes end. Before the first ending, means are
unavailable. By default, each completed PPO update writes `checkpoints/latest.zip`.
To save less often, add these flags to either `train_ppo.py` or `run_curriculum.py`:

```bash
--latest-every-updates 10 --checkpoint-every 0
```

This saves `latest.zip` after the first update, then every tenth update within
each training invocation. The displayed update counter remains cumulative when
continuing a run; the save schedule restarts for each invocation.
`--checkpoint-every 0` disables the independent numbered
backups; a positive value sets their interval in learning-player transitions,
rounded up to an update. Successful completion always leaves both `latest.zip`
and `final.zip` at the final trained update, even when it falls between saves.
Campaign evaluations and promoted model exports still occur as configured.

To apply this to an existing campaign, stop its running command with Ctrl+C,
then continue with the same run directory:

```bash
python mgba/run_curriculum.py --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10 \
  --latest-every-updates 10 --checkpoint-every 0
```

The campaign remembers these settings. They take effect in the next training
process; they do not change the running process. An interruption or failure
preserves the last saved update, so a longer interval can lose more training
progress. An interrupted campaign starts a new attempt from its last saved
`latest.zip` when continued.

Checkpoint cadence does not change rollout length, minibatches or PPO updates.
Avoid changing `--rollout-steps` solely to reduce saving. Save lines show seconds
spent writing checkpoints; `ppo/progress.csv` and `updates.jsonl` include
`time/checkpoint_seconds`, `time/checkpoint_seconds_total`, `time/update_seconds`
and `time/checkpoint_fraction`. The fraction compares save time with collection,
optimization and checkpoint time for that update. The final console summary and
`summary.json` include total save time, including final saves. If it is only a
small part of run time, saving less often will have little effect on battles/hour.

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
frozen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"
python mgba/train_ppo.py --backend showdown --environments 48 \
  --training-profile six_types --resume latest \
  --opponent-checkpoint "$frozen_checkpoint" --total-timesteps 100000
```

Resume preserves omitted hyperparameters and applies explicit overrides; the startup `Effective PPO` line shows
the values actually used. Measure improvement with the
[checkpoint evaluator](evaluation.md), which freezes both policies and reuses
the same matchup suite. Training win rates include exploration.
