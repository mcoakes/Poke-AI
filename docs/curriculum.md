# Growing-team training curriculum

[Battle-count campaigns, named models and evaluation history](battle-count-campaigns.md)

[Mixed evaluation opponents and editable gates](campaign-evaluation-pools.md)

[PPO training](ppo-training.md) · [Evaluation](evaluation.md) · [Every runner flag](reference/run-curriculum.md) · [Create a campaign](creating-campaigns.md)

Start with the existing six-types team, then add one team per promoted stage.
Each stage retains every earlier team. Both players independently sample the
current pool each battle, so later stages include different-team matchups as
well as mirrors. Only player 1 learns. By default, the reference policy stays
frozen for the whole curriculum. Alternatively, use
[`--opponent-mode previous-stage`](curriculum-opponents.md) to promote each
stage's candidate into the frozen opponent for the next stage. Both modes keep
the opponent fixed across all attempts of an individual stage.

The runner uses Showdown, which is headless and unthrottled. Install the
[simulator and training dependencies](setup/showdown.md) first. No save generation
or emulator launch is needed for this workflow.

## Start stage 1

Run from the project root. Choose a new curriculum directory:

```bash
fronzen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"
curriculum_run=".local/run/curriculum-types-01"

python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --stage 1 \
  --resume latest \
  --opponent-checkpoint "$fronzen_checkpoint" \
  --environments 48 \
  --showdown-workers process \
  --rollout-steps 128 \
  --batch-size 128 \
  --timesteps 100000 \
  --eval-episodes 1000
```

`latest` is resolved once at initialization. The candidate, opponent, teams and
reward settings are copied into this run. Another trainer updating its own
`latest.zip` cannot change these inputs. Use an explicit candidate path if you
want to start from a particular six-types result.

With 48 players, 24 learn. The settings above collect 3,072 transitions per PPO
update; each attempt adds 101,376 transitions after rounding 100,000 up to full
rollouts. Epochs, learning rate and entropy keep their checkpoint values unless
explicitly supplied. Existing reward defaults stay unchanged.

## What happens in a stage

1. Record the entering checkpoint's scores on both stage suites. This baseline
   runs once and remains fixed across retries of the stage.
2. Train for the requested additional transitions using only the current pool.
3. Evaluate the resulting checkpoint on the current pool.
4. Evaluate the same checkpoint on the current pool plus the next team.
5. Write a report and return `PROMOTE` or `HOLD`.

Every evaluation freezes both policies and swaps candidate sides on the same
teams and battle seeds. PPO continues to explore during training. This follows
the distinction between training and evaluation described in the
[Stable-Baselines3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#how-to-evaluate-an-rl-algorithm).

Each suite contains 1,000 battles by default:

| Suite | Matchups |
|---|---|
| Current | 1,000 battles using the stage's training pool |
| Expanded | 500 retained battles from the current suite, plus 500 involving the next team |

The expanded suite deliberately emphasizes the new team; it is **not** a uniform
sample of the enlarged pool. Reported incoming performance covers just those
500 battles. Candidate-side swaps ensure each policy gets both assignments,
including when the teams have different party sizes. The new team never enters
training until the following stage.

The first attempt therefore runs four evaluations (two before and two after
training). Later attempts reuse the two baseline results. Evaluation never
updates either policy. A stage's saved team/seed suites are reused across all
attempts, regardless of worker count.

## Why does evaluation start counting from 1 again?

`--eval-episodes 1000` means **1,000 battles per evaluation**, not per stage.
The first attempt has this sequence:

| Order | Operation | Battles / work | Output directory |
|---:|---|---|---|
| 1 | Baseline: current pool | 1,000 evaluation battles | `before-current-001/` |
| 2 | Baseline: current pool plus next team | 1,000 evaluation battles | `before-expanded-001/` |
| 3 | Train | `--timesteps` additional learning transitions | `attempt-001/train/` |
| 4 | Trained model: current pool | 1,000 evaluation battles | `attempt-001/eval-current-001/` |
| 5 | Trained model: current pool plus next team | 1,000 evaluation battles | `attempt-001/eval-expanded-001/` |
| 6 | Decide | `PROMOTE` or `HOLD` | `report.md` |

All paths above are inside `stage-NN/`. That is **4,000 evaluation battles plus
training battles** for a stage's first attempt. Seeing another `evaluation=1/1000`
is expected; check the printed `Evaluation:` output path to identify the phase.
The first 2,000 evaluation battles happen before this stage starts learning.

A stage can have multiple training attempts, but the runner does **not** launch
another attempt automatically after `HOLD`. Repeating the command creates
`attempt-002`, resumes the last candidate, trains again, and runs the two after
evaluations: **2,000 evaluation battles plus training**. The original baseline
is reused. `--through-stage 10` automatically advances through promoted stages;
it stops at the first held stage.

The `001` suffix on an evaluation directory counts executions of that evaluation
phase, not training attempts. If evaluation was interrupted, its retry can become
`eval-current-002` inside the same attempt. Completing interrupted evaluation does
not trigger another training chunk.

## Promotion checks

All checks must pass:

| Check | Initial threshold |
|---|---:|
| Current-pool win rate | At least 60% |
| Expanded-suite win rate | At least 55% |
| Incoming-team matchup win rate | At least 50% |
| Regression from the stage's entering checkpoint, current and expanded suites | At most 3 percentage points each |
| Truncated battle fraction, in every baseline/after suite and subgroup | At most 1% |

Win rate is wins divided by completed battles, including draws in the
denominator. These are practical starting thresholds, not statistical
significance tests or a promise that a stage will improve in 100,000 transitions.
A high average score cannot compensate for a failed incoming-team check.

The editable template is
[`promotion.json`](../config/curricula/type_progression/promotion.json).
To use different thresholds, copy that JSON, edit it, and pass
`--promotion-rules PATH` when creating a **new** run. A running curriculum keeps
its original rules and baseline. For a changed experiment, start a new run from
the checkpoint you want to retain.

## Ten stages

Stage N trains on teams 1 through N, not just team N. All Pokémon are level 50,
have four moves and maximum IVs, and use EV spreads within the 510-point limit.
The early stages isolate simple changes; later teams introduce more mechanics.
The sets are learning exercises, not a tournament strength ranking.

| Stage | Teams in pool | Team added | What changes |
|---:|---:|---|---|
| 1 | 1 | Arcanine, Blastoise, Sceptile, Jolteon, Glalie, Machamp | Your existing six-types team, unchanged |
| 2 | 2 | Same species, Blastoise lead | Different party positions and move slots |
| 3 | 3 | Ninetales, Venusaur, Lapras, Ampharos, Hitmonlee, Poliwrath | New species with familiar attack types |
| 4 | 4 | Original six species, Machamp lead | Off-type coverage attacks |
| 5 | 5 | Donphan, Alakazam, Aggron, Swellow, Umbreon, Scizor | More types and immunities |
| 6 | 6 | Venusaur, Blastoise, Jolteon, Ninetales, Machamp, Glalie | Status, setup, Substitute, hazards and items |
| 7 | 7 | Charizard, Starmie, Heracross | Three-member offense and recovery |
| 8 | 8 | Swampert, Metagross, Zapdos, Snorlax | Four-member balance, Choice Band and Rest |
| 9 | 9 | Tyranitar, Skarmory, Milotic, Breloom, Gengar | Five-member roles, sand and forced switches |
| 10 | 10 | Salamence, Forretress, Blissey, Suicune, Jolteon, Flygon | Six-member roles, recovery and Baton Pass |

Stage 10's expanded test adds an eleventh team: Aerodactyl, Slowbro, Ludicolo,
Weezing, Houndoom and Ursaring. It remains outside all ten training pools.

The [manifest and team files](../config/curricula/type_progression/README.md) are
ordinary JSON. Changing them affects new runs; existing runs use their saved
copies. Runtime stage pools are materialized in `stage-NN/teams/` and work with
the ordinary `--team-dir` flag if you want to run the trainer manually.

## Commands for each later stage

After stage 1 is initialized, its settings and frozen checkpoint are remembered.
Keep `curriculum_run` set as above, or set it again in a new terminal. Run the
next command only after the preceding stage reports `PROMOTE`:

```bash
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 2
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 3
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 4
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 5
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 6
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 7
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 8
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 9
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 10
```

Alternatively, let it advance through successful stages automatically:

```bash
python mgba/run_curriculum.py --run-dir "$curriculum_run" --through-stage 10
```

It stops at the first `HOLD`; it never retries indefinitely or skips a failed
stage. With no stage flags, it runs the first stage that has not been promoted.
Already promoted stages are not retrained accidentally.

To continue a held stage for another chunk, repeat its command. For example:

```bash
python mgba/run_curriculum.py --run-dir "$curriculum_run" --stage 3 --timesteps 50000
```

This resumes that stage's last candidate, keeps all earlier teams, and reuses
its fixed baseline/suites. Execution settings such as `--timesteps`, worker
count and explicit PPO overrides are remembered for subsequent invocations.
Omit the new-run flags (`--resume`, `--opponent-checkpoint`, `--opponent-mode`, suite settings and
input paths) when continuing an existing run.

If a stage stays held after another 100–200k transitions, inspect its report
before adding more compute. A weak incoming result can mean the next change is
too large to learn from the current pool alone. Reduce that change in a new
curriculum or investigate observations and decisions; endlessly rehearsing the
same pool is not guaranteed to solve it. Later stages exercise mechanics whose
details may not be fully represented in the current observation schema.

## Reports, interruption and broader checks

Read `stage-NN/report.md` for a compact table of before/after win rates and every
promotion check, together with the frozen opponent's identity. `result.json`
provides the candidate and opponent paths, hashes and evaluation
directories. Individual attempts retain their own reports under
`stage-NN/attempt-NNN/`. Training retains the usual episode means, 500-battle
blocks, battles/hour, checkpoints, diagnostic logs and spectator logs.

Ctrl+C preserves completed PPO updates. Repeating the stage resumes a saved
update if training stopped early. If training completed and evaluation stopped,
the runner finishes evaluation without another training chunk; completed
matching evaluations are reused. Failed/incomplete evaluation directories are
retained and a new one is used for the retry. One process at a time may operate
on a curriculum run directory.

Promotion scores are **validation** results: the curriculum uses them to decide
what to train next, so even the final new team is not an untouched final test.
Periodically run the separate mixed-three suite against the same frozen
reference to check broader transfer:

```bash
stage_checkpoint="$curriculum_run/stage-01/attempt-001/train/checkpoints/final.zip"

python mgba/evaluate_ppo.py \
  --environments 48 --showdown-workers process \
  --checkpoint "$stage_checkpoint" \
  --opponent-checkpoint "$curriculum_run/inputs/opponent.zip" \
  --suite .local/run/eval-mixed-three-before/suite.json \
  --output-dir .local/run/eval-curriculum-stage01-mixed
```

Use the actual checkpoint from the stage report if there were retries. Use a
new evaluation output directory each time. Repeated use of that broad suite
also makes it a validation set; reserve another untouched suite for your final
assessment. Success in ten pools is a useful milestone, not evidence of universal
battle skill. In previous-stage mode the original `inputs/opponent.zip` remains
unchanged, so this broad evaluation still has a stable reference.
