# Train PPO

[Utility index](README.md) · [Workflow guide](../ppo-training.md) · [Source](../../mgba/train_ppo.py)

Train one shared masked PPO policy, or train player 1 against a frozen player-2 checkpoint.

```bash
python mgba/train_ppo.py --backend showdown --environments 4 \
  --team-profile mixed --team-size 3 \
  --rollout-steps 128 --batch-size 128 --total-timesteps 100000
```

## Options

`--showdown-workers thread|process` defaults to `thread` and applies to Showdown.
Both modes have a Node process per pair; `process` additionally gives each pair
its own Python environment process. PPO remains centralized. See
[CPU worker details](../showdown-training.md#cpu-cores-and-workers).

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--rom` | Not set | Required for mGBA; unused by Showdown. |
| `--emulator` | `.local/mgba/bin/mgba-qt` | Path to the patched project-local `mgba-qt` executable. |
| `--headless` | Off | mGBA: use Xvfb. Showdown is always headless. |
| `--startup-timeout` | `30` | Wall-clock seconds to wait for adapter readiness. |
| `--connect-timeout` | `180` | Wall-clock seconds allowed for each NPC connection or battle-start stage. |
| `--lobby-only` | Off | Inherited launcher option; rejected because PPO requires a battle. |
| `--speed` | `1` | mGBA multiplier or `unlimited`. Accepted but has no effect in unthrottled Showdown. |
| `--states` | Not set | Two fixed counter states, in player order |
| `--state-dir` | Not set | Draw two random .ssN counter states from this directory |
| `--teams` | Not set | Showdown: two fixed team JSON files, in player order |
| `--team-dir` | Not set | Showdown: draw random team JSON files from this directory |
| `--training-profile` | Not set | Showdown: sample teams in `config/training_profiles/NAME/teams`, such as `six_types`. |
| `--team-profile` | Not set | Select competitive, casual, random, or a mixture of all three. Choices: `competitive`, `casual`, `random`, `mixed`. |
| `--mirror-teams` | Off | Random sources only: sample one team/state per battle for both players; see [mirror matches](../team-profiles.md#random-mirror-matches). |
| `--team-size` | Not set | Restrict both players to this party size; requires --team-profile. Choices: `1`, `2`, `3`, `4`, `5`, `6`. |
| `--backend` | `mgba` | Battle engine (default: mgba); Showdown uses team JSON files directly. Choices: `mgba`, `showdown`. |
| `--observation-version` | New model: `v1`; resume: checkpoint schema | `v1` or `v2`. Native v2 requires adapter 0.19.0 on both linked players. Explicit conflicts with `--resume` are rejected; frozen opponents may use either schema. See [v2](../observation-v2.md). |
| `--shuffle-move-slots`, `--no-shuffle-move-slots` | Off | Showdown, v1/v2: independently permute each learner's own four move slots per party member per battle, with matching features, masks and action translation. Frozen opponents and evaluation retain original order. Restored on same-directory continuation; changing it requires a new run. [Experiment and examples](../experiments/move-slot-shuffle.md). |
| `--shuffle-bench-order`, `--no-shuffle-bench-order` | Off | Showdown with Rules only: sample the canonical team pair first, then randomly permute Player 1's occupied bench. Lead, moves and opponent order stay fixed; fixed-pair rehearsal groups are excluded. Separate permutation RNG; bounded, verified variant files under `inputs/bench-order`. Restored on continuation; changing the condition requires a new run. [Controlled comparison](../experiments/bench-order-ablation.md). |
| `--node` | `node` | Node.js executable for the Showdown backend |
| `--environments` | `4` | Even number of Gymnasium players (default: 4) |
| `--base-port` | `55355` | First mGBA adapter port. Showdown opens no ports; this value also contributes to its per-pair seed. |
| `--total-timesteps` | `10000` | Additional learning-player transitions, rounded up to a full rollout |
| `--train-battles` | Not set | Additional ended battles across the pool, including truncations; replaces the timestep budget and finishes the current PPO update when reached. |
| `--rollout-steps` | `64` | Steps per learning player before each PPO update |
| `--batch-size` | `64` | Transitions per optimization minibatch; at least 2 and must divide rollout steps × learning players. |
| `--epochs` | `4` | Optimization passes over each collected rollout. |
| `--learning-rate` | `0.0003` | Positive finite learning rate; explicit values override resume. |
| `--ent-coef` | `0.01` | Nonnegative entropy bonus; explicit values override resume. |
| `--max-steps` | `200` | Decision limit per battle; unfinished battles truncate |
| `--command-timeout` | `120` | Maximum wall-clock seconds for one paired battle decision. |
| `--seed` | `1` | Seed for team sampling and policy sampling; Showdown also seeds battle RNG. mGBA RNG comes from the counter states. |
| `--device` | `cpu` | PyTorch policy device, such as `cpu` or `cuda`; does not move the emulator or simulator to the GPU. |
| `--torch-threads` | `1` | Positive number of PyTorch CPU threads. |
| `--latest-every-updates` | `1` | Positive number of PPO updates between `latest.zip` saves. Also saves after the first update and at successful completion. |
| `--checkpoint-every` | `1024` | Learning-player transitions between numbered backups, rounded up to a completed update; `0` disables numbered backups. Independent of `latest.zip`. |
| `--keep-checkpoints` | `2` | Retain this many numbered PPO backups; `0` keeps none, `-1` keeps all. Does not delete `latest.zip` or `final.zip`. |
| `--resume` | Not set | Checkpoint path or `latest`; start a new run with saved model/optimizer state. Explicit PPO flags override saved values. Same-directory continuation selects its own checkpoint. |
| `--policy-architecture` | Fresh: `flat`; resume: saved class | `flat` keeps the existing network. `shared-moves` shares one scorer across the four moves. `matchup-moves` adds observed type-matchup and supported ability-immunity inputs. `matchup-switches` adds a shared residual scorer for multi-option forced replacements. All three require v2, selected automatically unless an incompatible schema is explicit. Resume preserves the saved architecture and feature toggles. See the [matchup comparison](../experiments/matchup-ablation.md) for untrained conversion, and the [replacement-feature comparison](../experiments/switch-feature-ablation.md) for trained-model migration with retained optimizer state. Normal resume rejects architecture changes. |
| `--opponent-checkpoint`, `--player2-checkpoint` | Not set | Checkpoint path or `latest`; freeze player 2 in every pair; only player 1 trains |
| `--opponent-stochastic` | Off | Sample frozen checkpoint actions, including checkpoints in a pool. Rules stay deterministic; weights stay fixed. |
| `--rule-based-opponent` | Off | Showdown: train only Player 1 against deterministic `gen3-rules-v1`; incompatible with `--opponent-checkpoint`. Rules receive v2; learner schema is unchanged. [Guide](../rule-based-opponent.md). |
| `--opponent-pool` | Off | Showdown: weighted [pool JSON](../../config/opponents/README.md), sampled once per battle. Only Player 1 learns. Exclusive with other opponent selectors. Models/weights are pinned for continuation. |
| `--rule-log`, `--no-rule-log` | Off | With rules or a pool: bounded `rule-decisions.jsonl` scores/reasons and pool `opponent-assignments.jsonl`; uses shared log limits. |
| `--opponent-switch-chance` | `0` | Fraction 0–1 for a type-based opponent switch check during training. Requires a frozen checkpoint when nonzero; changes only Player 2's actions. [Rule and examples](../switch-pressure.md). |
| `--opponent-switch-cooldown` | `2` | Nonnegative number of normal decisions to skip after a voluntary opponent switch. Applies to heuristic interventions, not the underlying checkpoint policy. |
| `--reward-config` | `config/rewards/damage_knockout_win.json` | JSON reward settings for new runs, including forks with `--resume`. Same-directory continuation restores saved rewards and rejects changes. |
| `--readable-log`, `--no-readable-log` | On | Write observations/actions to each pair's `battle-readable.log`; rotates in Showdown. |
| `--battle-log`, `--no-battle-log` | On | Write spectator battle text and HP events to each pair's `battle.log`; rotates in Showdown. |
| `--step-logs`, `--no-step-logs` | Showdown: off; mGBA: on | Write per-decision Gym/protocol traces and episode team copies. Explicit `--no-step-logs` is rejected for mGBA, which retains its own diagnostics. |
| `--log-max-mb` | `10` | Showdown size limit per rotating log in MiB; `0` means unlimited. |
| `--log-backups` | `2` | Rotated copies kept per log; `0` keeps only the current file. |
| `--keep-episode-logs` | `2` | Recent completed Showdown episode debug directories per pair, plus active/failed episodes; `0` disables them, `-1` keeps all. |
| `--output-dir` | Not set | Run directory. Default: a new `.local/run/ppo-<timestamp>/`. Existing directories use `--if-exists`. |
| `--if-exists` | `ask` | Existing output directory: `ask`, `continue`, `erase` or `cancel`. `ask` prompts only in an interactive terminal; otherwise choose explicitly. Continue restores saved run settings; erase deletes a recognized standalone PPO run after validating the new configuration. |
| `--verbose` | Off | Print NPC connection phases |

## Usage notes

| Periodic validation flag | Default | Meaning |
|---|---|---|
| `--eval-suite` | Off | Fixed Showdown validation suite. Copies its teams/opponent identities, evaluates between updates, and maintains `checkpoints/best.zip`. |
| `--eval-every` | `100000` | Additional learning-player transitions since the last validation, rounded up to a PPO update. Requires a suite; `0` pauses existing validation. Initial and final checkpoints are also evaluated when needed. |
| `--eval-environments` | Training environment count | Even number of validation players, at least two. Validation pauses training. |
| `--keep-evaluations` | `2` | Positive number of recent result directories to retain, plus the best directory if older. Compact history and `best.zip` are always retained. |

See [periodic evaluation](../periodic-evaluation.md) for continuation, fixed
opponents, selection rules and bounded logs. Latest saves are also forced before
validation, independently of `--latest-every-updates`.

Choose exactly one team/state source. Showdown accepts JSON `--teams`, `--team-dir`, or profiles; mGBA accepts `.ssN` `--states`, `--state-dir`, or built profiles and requires `--rom`. `--team-size` requires a profile. Environment count is even and at least 2. With a frozen opponent only half the players learn. Explicit rollout, batch, epoch, learning-rate and entropy flags override resume settings; omitted flags keep saved values. The effective batch size must divide effective rollout steps × learning players. `--lobby-only` is rejected. Outputs and checkpoint timing are described in [reports](../battle-logs.md).

See [checkpoint resolution](../checkpoints.md) for latest selection and the
[controlled profiles](../../config/training_profiles/README.md) for focused team pools.

For long runs, `--no-readable-log --no-battle-log --no-step-logs` keeps reward,
win-rate, throughput and PPO metrics while avoiding detailed battle traces.
Numbered checkpoint retention and `latest.zip` save frequency are independent.
See [storage and cleanup](../storage.md) for campaign-wide limits and existing-run
cleanup; per-file rotation alone is not a total run-size quota.

Hyperparameter defaults in the table apply to new models. On resume, omitted
flags use checkpoint values, even when those differ from these defaults.

### Existing output directories

```bash
python mgba/train_ppo.py \
  --output-dir .local/run/ppo-observation-v2-01 \
  --if-exists continue --total-timesteps 400000
```

Continue restores the saved backend, team/state source, reward values, frozen
opponent, observation schema, PPO settings and logging choices. It selects the
highest saved timestep count among recoverable checkpoints in that run. The
budget is additional; the command above does not set an absolute 400,000 target.
Explicit compatible flags are accepted; PPO overrides, execution settings and
logging controls may change. New teams, rewards, backend, observation schema or
frozen weights require a new output directory with `--resume`.

The terminal prompt uses `c` for continue, `e` for erase, and `q` or Enter for
cancel. Choosing erase there requires typing the run name or path. Explicit
`--if-exists erase` authorizes deletion without another prompt and requires a
complete fresh configuration. Unrecognized directories, active runs and campaign attempt
directories cannot be erased or continued through this interface.

Same-directory continuation retains cumulative metrics and separate session
logs, but starts fresh battles and random streams. See
[continuation details](../ppo-training.md#continue-an-existing-run) and
[session storage](../storage.md#standalone-run-sessions).

If both budget flags are omitted, continuation repeats the saved budget.
Explicit `--total-timesteps` switches a previous battle-count run to a timestep
budget. Supply one budget flag, not both, for a continuation.

### Battle-count budget

`--train-battles N` replaces the timestep budget: finish the PPO update in which
the whole pool reaches N ended battles (including truncations). Actual counts
can exceed N and appear in `summary.json`; evaluation budgets remain exact.
