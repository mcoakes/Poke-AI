# Run a team curriculum

[Utility index](README.md) · [Workflow and stage commands](../curriculum.md) · [Create a campaign](../creating-campaigns.md) · [Source](../../mgba/run_curriculum.py)

Compose the existing Showdown trainer and frozen-policy evaluator into a growing
team-pool curriculum. This runner uses a POSIX file lock (Linux/macOS) to prevent
two commands from modifying the same run concurrently.

Stages accept a single `team` or recursive `team_dir`. Optional `validation_dir`
groups replace training teams in promotion evaluations. See the
[manifest schema](../creating-campaigns.md#directory-based-stages) and the
[ten-stage generalization plan](../generalization-campaign.md). All teams and group
membership are snapshotted for a run; old campaigns do not change.

```bash
python mgba/run_curriculum.py --run-dir .local/run/curriculum-types-01 \
  --resume latest --opponent-checkpoint "$frozen_checkpoint" \
  --environments 48 --rollout-steps 128 --batch-size 128 --timesteps 100000
```

## Navigation

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit. |
| `--run-dir` | Required | New or existing curriculum directory; stores inputs, settings, stages and reports. |
| `--stage` | First unpromoted stage | Run one stage. Requires previous stages to have passed; an already promoted stage is a no-op. |
| `--through-stage` | Selected stage only | Continue through this inclusive stage number; stop when an error occurs or a stage uses all allowed attempts without promotion. |

## New-run inputs

These are accepted only on the first invocation. Later invocations use the
snapshots in the run directory; omit these flags when continuing.

| Flag | Initial default | Meaning |
|---|---|---|
| `--curriculum` | `config/curricula/type_progression/curriculum.json` | Ordered manifest: `team` or `team_dir` per stage, optional validation groups, and final probe. Select `config/curricula/pokemon_generalization/curriculum.json` for the directory-based preset. |
| `--resume` | Required | Starting candidate path or `latest` (`later` alias accepted). Copies model and optimizer state once. |
| `--opponent-checkpoint`, `--player2-checkpoint` | Required | Initial frozen reference path or `latest`. Used for every stage in fixed mode, or stage 1 in previous-stage mode; its original copy is retained. |
| `--opponent-mode` | Manifest's `opponent_mode`, otherwise `fixed` | `fixed` keeps the initial reference; `previous-stage` freezes each promoted model as the next stage's opponent. Explicit flag overrides the manifest. Immutable within a campaign. |
| `--opponent-switch-chance` | Manifest's `opponent_switch.chance`, otherwise `0` | New run: training opponent switch-check probability, 0–1. Does not affect evaluation. Pinned for the campaign. |
| `--opponent-switch-cooldown` | Manifest's `opponent_switch.cooldown`, otherwise `2` | New run: nonnegative normal-decision cooldown after voluntary opponent switches. Pinned for the campaign. [Switching pressure](../switch-pressure.md). |
| `--reward-config` | `config/rewards/damage_knockout_win.json` | Reward settings to snapshot for training and evaluation. |
| `--promotion-rules` | Built-in 60%/55%/50%, 3-point regression, 1% truncation | JSON with all five threshold keys; [schema and example](../../config/curricula/type_progression/promotion.json). |
| `--eval-episodes` | `1000` | Battles per suite: divisible by 4, or by 20 for mixed evaluation opponents. Original expanded evaluation splits equally into retained and incoming matchups. |
| `--max-steps` | `200` | Decision cap per training/evaluation battle. Unfinished battles truncate. |
| `--seed` | `1` | Base seed. Stage evaluations stay fixed within the stage; mixed evaluations get a fresh, saved schedule per attempt. Each attempt also gets a new training seed. |

## Execution settings

Explicit values update remembered settings for the next attempt and later
commands. Omitted values reuse those settings. The simulator always runs
headlessly and unthrottled; there is no emulator speed setting here.

| Flag | Initial default | Meaning |
|---|---|---|
| `--environments` | `48` | Even count of concurrent players, at least 2. Half learn and half use the frozen reference. |
| `--showdown-workers` | `process` | `process` or `thread` Python workers; each pair also has a Node simulation process. |
| `--timesteps` | `100000` | Additional learning-player transitions per attempt, rounded up to a complete rollout. |
| `--rollout-steps` | `128` | Decisions per learner before an update; explicitly overrides the resumed checkpoint. |
| `--batch-size` | `128` | Minibatch size, at least 2; must divide rollout steps × learning players. |
| `--epochs` | Checkpoint value | Passes through each rollout; explicit override persists. |
| `--latest-every-updates` | `1` | Save training `latest.zip` every N updates, plus the first update and successful completion. Can change when continuing a campaign. |
| `--checkpoint-every` | `1024` | Learning-player transitions between numbered training backups, rounded up to an update; `0` disables them. Can change when continuing a campaign. |
| `--keep-checkpoints` | `2` | Retained numbered backups per training attempt; `0` keeps none, `-1` keeps all. Does not remove its `latest.zip` or `final.zip`. |
| `--learning-rate` | Checkpoint value | Positive finite optimizer learning rate; explicit override persists. |
| `--ent-coef` | Checkpoint value | Nonnegative finite entropy coefficient; explicit override persists. |
| `--device` | `cpu` | PyTorch training/inference device. Does not move the simulation to the GPU. |
| `--torch-threads` | `1` | PyTorch CPU thread count. |
| `--node` | `node` | Node executable used by both trainer and evaluator. |
| `--command-timeout` | `120` | Wall-clock seconds per paired battle decision. |
| `--readable-log`, `--no-readable-log` | On | Training diagnostic observation/action logs. |
| `--battle-log`, `--no-battle-log` | On | Training spectator battle logs. Evaluations use the evaluator's quiet logging defaults. |
| `--step-logs`, `--no-step-logs` | Off | Showdown per-decision simulation/protocol traces; forwarded to training and evaluation. |
| `--log-max-mb` | `10` | Showdown limit per rotating log in MiB; `0` means unlimited. Forwarded to training and evaluation. |
| `--log-backups` | `2` | Rotated copies retained per log; `0` keeps only the current file. |
| `--keep-episode-logs` | `2` | Recent completed Showdown episode debug directories per pair, plus active/failed episodes; `0` disables them, `-1` keeps all. Applies to training and evaluation. |

## Storage settings

Explicit values persist when continuing a campaign, like the execution settings
above. New campaigns enable compact retention. Existing campaigns without saved
retention settings retain their previous behavior until explicitly changed.

| Flag | Initial default | Meaning |
|---|---|---|
| `--retention` | New: `compact`; existing: `off` | `compact` prunes selected heavy data after completed attempts; `off` disables automatic cleanup. |
| `--keep-attempts` | `2` | Keep candidates from this many recent attempts per stage; `0` keeps only other selected/protected candidates, `-1` keeps all. |
| `--keep-best` | `2` | Keep best evaluated candidates per stage and evaluation-rule cohort; `0` disables score-based retention, `-1` keeps all. |
| `--keep-debug-attempts` | `0` | Keep pair debug data and detailed training traces for this many recent attempts per stage; `-1` keeps all. |

The candidate sets are combined; protected inputs, frozen opponents, promoted
candidates and required resume models are always retained. Reports, configurations,
suites, evaluation episode results and PPO progress CSVs also remain. These limits
do not impose a hard total disk quota, and pruning candidate weights prevents
reevaluating that candidate unless a separate copy exists.

The training process keeps `latest.zip` and `final.zip` outside its numbered-backup
limit. Subsequent compact campaign cleanup retains selected completed candidates
as `final.zip` and can remove their redundant `latest.zip` copies. The newest
unfinished attempt retains recovery checkpoints.

Use [the cleanup utility](prune-campaign.md) to preview and clean an existing
campaign while it is stopped. The [storage guide](../storage.md) provides exact
cleanup and restart commands.

## Outputs and exit status

`campaign.json` records frozen inputs and execution settings. `commands.jsonl`
records exact child command arguments. Each `stage-NN` contains the cumulative
team pool, saved suites, fixed baseline, numbered attempts, `result.json` and a
human-readable `report.md`. Each stage snapshots its opponent in `opponent.zip`
and records its source stage/checksum in `stage.json`, attempt metadata and
reports. See [opponent progression](../curriculum-opponents.md). Training defaults
to saving after every PPO update, with numbered backups every 1,024 learning
transitions, retaining two numbered backups by default. The save interval flags
above change those intervals without changing PPO
updates or evaluation/promotion saves. Settings persist for later attempts; each
training process starts a fresh update interval. See [checkpoint timing](../ppo-training.md#progress-and-saving).

| Exit code | Meaning |
|---:|---|
| 0 | Requested stages passed, or an already completed stage/run was selected. |
| 2 | A stage finished but promotion checks returned `HOLD`; checkpoint retained. Argparse also uses 2 for malformed CLI syntax. |
| 1 | Input validation, subprocess or other runtime failure. |
| 130 | Interrupted; completed artifacts retained. |

The runner stops on incomplete training/evaluation and does not classify an
error as a win, draw or promotion. See the [workflow guide](../curriculum.md) for
retry behavior and commands for all ten stages.

## Battle-budget mode and history

See the [complete walkthrough](../battle-count-campaigns.md). Existing campaigns
keep their saved settings unless an editable flag is explicitly supplied.

| Flag | Default | Meaning |
|---|---|---|
| `--train-battles` | Unset | New run: switches to one current-pool evaluation after training reaches N ended battles, finishing the PPO update. Replaces the timestep budget; includes truncations. |
| `--promotion-win-rate` | `0.75` | Stage-opponent evaluation gate in battle-budget mode. May change on resume; wins / all attempts, fraction in (0, 1]. Use the separate gates below for mixed opponents. |
| `--attempts-per-stage` | `1` | Maximum attempts per stage for this invocation. Retry HOLD using the last candidate; stop early on promotion. Can be changed when resuming. |

Named promoted models are exported as `models/smart_v2.zip`, `smart_v3.zip`, etc.;
initial snapshots are `models/smart_v1.zip` and `models/dumb.zip`. `evaluations.md`
and `evaluations.csv` collect the stage/attempt evaluation history. In legacy mode
the table reports current-pool results; the detailed reports retain every gate.

## Evaluation pools and editable gates

These flags apply to battle-budget campaigns and can be set on new runs or when
resuming. Explicit changes persist and are recorded in `rule-changes.jsonl`.
Completed reports and promotions are unchanged. See [examples and pool behavior](../campaign-evaluation-pools.md).

| Flag | Initial default | Meaning |
|---|---|---|
| `--eval-opponents` | `stage` | `stage` tests against the stage's training opponent. `mixed` uses 30% Random / 30% Dumb / 40% Smart with shuffled, side-balanced assignments. Does not change training opponents. |
| `--promotion-basic-win-rate` | `0.75` | Mixed gate: wins divided by all Random + Dumb battles combined. Fraction in (0, 1]. |
| `--promotion-smart-win-rate` | `0.60` | Mixed gate: wins divided by all Smart battles. Fraction in (0, 1]. |
| `--promotion-require` | `both` | `both` requires both category gates; `either` requires at least one. Equality passes. |
| `--eval-smart-dir` | Not set | Add direct `smart_v*.zip` files to the campaign's starting and previously promoted Smart models. Discover and snapshot on each new evaluation. Identical checkpoint contents count once. |
| `--eval-dumb-checkpoint` | Initial campaign reference | Optional path or `latest` (`later` alias accepted), resolved and copied once. Replaces only the Dumb evaluation reference. |

Omitted settings retain saved values. Draws/truncations count as non-wins; incomplete
evaluations cannot pass. Pool models, teams and seeds are pinned inside each
attempt's `evaluation-pool/`, so interrupted evaluations replay the same test.
The track record adds category rates and rule revisions. Set `--eval-opponents stage`
to return to the single-opponent gate for subsequent evaluations.
