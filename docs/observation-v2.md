# Observation v2

[Training](ppo-training.md) · [Audit](training-audit.md) · [Checkpoint compatibility](checkpoints.md)

V2 adds explicit battle information to policy inputs. Showdown and native linked
mGBA sessions provide the same schema; see [native v2 setup](mgba-observation-v2.md).
Select it with `--observation-version v2` when starting a new model. V1 remains
the default for the `flat` architecture and supports both Showdown and mGBA.
The optional [shared move scorer](experiments/shared-move-policy.md) selects v2
automatically and requires fresh weights; it does not change the v2 schema.
Rewards, the 11 actions, team selection and episode boundaries are unchanged.

The extra inputs can help the policy distinguish situations that v1 represents
identically. Their effect on learning must be measured; more inputs do not
guarantee better decisions.

## What the policy receives

V2 retains the v1 arrays unchanged and adds the following fixed-size arrays.
The leading side dimension is **own side, opponent**. Party rows retain original
team slots, and empty slots are zero padded.

| Raw key | Shape | Values |
|---|---|---|
| `v2_party` | `(2, 6, 10)` | Valid slot, level, current HP, maximum HP, two base types, current item, current ability, fainted flag, major-status bits |
| `v2_stats` | `(2, 6, 5)` | Computed base party Attack, Defense, Special Attack, Special Defense and Speed, including level/IV/EV/nature effects; before battle boosts |
| `v2_active_stats` | `(2, 5)` | Current active stored stats in the same order, including temporary replacements such as Transform; before boosts |
| `v2_active` | `(2, 4)` | Exact active current HP, maximum HP, current item and current ability |
| `v2_boosts` | `(2, 7)` | Attack, Defense, Special Attack, Special Defense, Speed, accuracy and evasion stages, from −6 to +6 |
| `v2_move_details` | `(2, 6, 4, 6)` | Base power, accuracy, always-hit flag, Gen III category, priority and type ID for each move |
| `v2_weather` | `(3,)` | Weather identity, remaining duration, permanent-weather flag |
| `v2_volatiles` | `(2,)` | Both active Pokémon's supported volatile flags and counters |

Ability IDs use the simulator's Gen III numeric IDs. Species, moves, types and
items retain the existing project mappings. Move categories are `0` physical,
`1` special and `2` status. Always-hit moves use accuracy `0` plus a separate
flag. Weather IDs are `0` none, `1` sun, `2` rain, `3` sandstorm and `4` hail;
permanent weather uses duration `0` with its permanent flag set.

Active move details replace the entire party row, clearing unused slots after
Transform. Hidden Power uses its computed type/category and power. Other move
descriptors are base data, not damage predictions: conditional power, accuracy
modifiers, abilities, items, status and field effects still affect resolution.
The stat arrays likewise do not claim to be final damage or speed calculations
after every modifier.

Before float conversion, the policy expands party status into
`v2_status_bits (2, 6, 32)` and active volatile fields into
`v2_volatile_bits (2, 32)`. It zeros the corresponding raw packed fields to avoid
float32 losing low bits. Numeric values are scaled and categorical values use
embeddings. Missing party and move slots are masked.

This is intentionally a **full-information training profile**: opponent bench
HP, items, abilities and moves are available without being revealed in battle.
It is not a fair hidden-information competitive observation. Supported volatile
bits are not an exhaustive encoding of simulator state; entry hazards and every
temporary effect are not added here.

## Verify the interface

Use a new output directory; an audit never overwrites an earlier run:

```bash
python mgba/audit_training.py \
  --observation-version v2 \
  --output-dir .local/run/audit-observation-v2-01 \
  --log-max-mb 2 --log-backups 1
```

The v2 report separates new native-field and policy checks from v1 regression
evidence. Missing v1 fields are not reported as missing v2 inputs. Add
`--checkpoint PATH` to check a saved v2 policy; the audit never trains or
overwrites its weights.

## Train a new v2 model

Start fresh v2 learner weights. When continuing from the
[Stage 1 baseline](baseline.md), keep these roles separate:

| Role | Recommended selection | Purpose |
|---|---|---|
| Frozen training opponent | `opponents.dumb` from `baseline.json` | A stable initial reference while the fresh v2 learner trains |
| Existing v1 benchmark | `candidates.best` from `baseline.json` | Preserve its results and compare against v2 on the same saved suites |
| New learner | Fresh v2 model, without `--resume` | V1 weights cannot be loaded into the larger v2 schema |

The role named Dumb is the baseline's saved reference, not a guarantee of a
particular skill level. Keep it frozen during the first experiment. Do not use
global `latest` for either reference: it may select a newly trained v2 model.

Read the exact snapshot paths and reward configuration from the manifest.
Change `baseline_run` if your baseline uses a different directory:

```bash
baseline_run=".local/run/baseline-generalization-01"
frozen_checkpoint="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["opponents"]["dumb"]["path"])' "$baseline_run/baseline.json")"
benchmark_checkpoint="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["candidates"]["best"]["path"])' "$baseline_run/baseline.json")"
reward_config="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["reward"]["path"])' "$baseline_run/baseline.json")"
v2_run=".local/run/ppo-observation-v2-01"

printf 'Frozen opponent: %s\nV1 benchmark: %s\n' "$frozen_checkpoint" "$benchmark_checkpoint"
```

Without a saved baseline, set these variables to explicit checkpoint and reward
paths. The mixed three-Pokémon example below is an initial training run; it does
not automatically reproduce the source campaign's team pool. To measure the
observation change alone, use the same training teams in fresh v1 and v2 runs.

```bash

python mgba/train_ppo.py \
  --backend showdown --observation-version v2 \
  --environments 48 --showdown-workers process \
  --team-profile mixed --team-size 3 \
  --opponent-checkpoint "$frozen_checkpoint" \
  --reward-config "$reward_config" \
  --rollout-steps 128 --batch-size 128 \
  --total-timesteps 100000 \
  --readable-log --battle-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --keep-checkpoints 2 --latest-every-updates 5 \
  --output-dir "$v2_run"
```

This keeps bounded diagnostic and spectator logs. Forty-eight environments make
24 battles and, with a frozen opponent, 24 learning players. The first update
and successful completion save checkpoints even with the longer save interval.

Continue in the same run directory, retaining its frozen opponent, teams,
rewards, v2 schema and bounded logging settings:

```bash
python mgba/train_ppo.py \
  --output-dir "$v2_run" \
  --if-exists continue \
  --total-timesteps 400000
```

The budget adds another 400,000 transitions, rounded up to a rollout. You can
also repeat the original command: when its directory exists, the terminal asks
whether to continue, erase or cancel. Choose continue to keep prior results;
the next invocation starts fresh battles using the saved learner and opponent.
See [same-directory continuation](ppo-training.md#continue-an-existing-run).

Requesting a different schema from the resumed checkpoint is an error. V1
checkpoints are not silently resized or converted to v2. Frozen opponents can
use either version: each policy receives its own schema from the same battle,
and the frozen model's weights remain unchanged.

## Evaluate and compare

To run this comparison automatically during training and preserve the strongest
weights, follow [periodic evaluation](periodic-evaluation.md). The saved
validation suite and its v1/v2 opponent pool remain fixed across checks.

Replay the baseline's validation suite to preserve teams, seeds, player sides,
opponent pool and decision cap:

```bash
python mgba/evaluate_ppo.py \
  --checkpoint "$v2_run/checkpoints/final.zip" \
  --suite "$baseline_run/suites/validation.json" \
  --reward-config "$reward_config" \
  --environments 48 --showdown-workers process \
  --readable-log --battle-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --output-dir .local/run/eval-observation-v2-01
```

The baseline suite already pins Random, Dumb and Smart opponents, so omit
`--opponent-checkpoint`. Replay the same suite with
`--checkpoint "$benchmark_checkpoint"` and a different output directory for a
current-code v1 reference. This also refreshes the comparison if simulator or
encoding fixes landed after the original baseline was measured. Evaluation
detects each model's schema and records it in the model metadata.

Without a baseline suite, use another saved suite. A suite without an opponent
pool also needs `--opponent-checkpoint "$frozen_checkpoint"`.

Keep the earlier baseline intact. For a controlled comparison, train fresh v1
and v2 models with the same teams, frozen opponent, rewards, training-transition
budget and hyperparameters. Repeat across several training seeds and evaluate
on the same saved suites. Comparing fresh v2 against a much longer-trained v1
model is useful operationally, but does not isolate the benefit of the inputs.
Do not use validation suites to train either model.

## Backend and campaign compatibility

Native linked mGBA sessions accept v2 with adapter 0.19.0 on both players.
Human play selects the schema from the checkpoint automatically. Both flat and
shared-moves v2 checkpoints retain their weights and input dimensions.
[Native field mapping and live verification](mgba-observation-v2.md) explain the
read-only decoder and its validation limits. This update needs a script/server
restart, with no additional mGBA binary patch.

A new campaign can use a saved v2 checkpoint as its `--resume` model and either
version as its frozen reference. Training children infer the learner schema
from that checkpoint; evaluation does the same per model. Use a **new run
directory**. Existing campaigns retain their pinned models and settings; there
is no campaign option to upgrade an existing v1 learner in place.
