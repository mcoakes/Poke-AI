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
| `--node` | `node` | Node.js executable for the Showdown backend |
| `--environments` | `4` | Even number of Gymnasium players (default: 4) |
| `--base-port` | `55355` | First mGBA adapter port. Showdown opens no ports; this value also contributes to its per-pair seed. |
| `--total-timesteps` | `10000` | Additional learning-player transitions, rounded up to a full rollout |
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
| `--checkpoint-every` | `1024` | Learning-player steps between numbered checkpoints; latest saves after every rollout |
| `--resume` | Not set | Checkpoint path or `latest`; resume model/optimizer; explicit PPO flags override saved values |
| `--opponent-checkpoint`, `--player2-checkpoint` | Not set | Checkpoint path or `latest`; freeze player 2 in every pair; only player 1 trains |
| `--opponent-stochastic` | Off | Sample the frozen opponent policy (weights stay fixed) |
| `--opponent-switch-chance` | `0` | Fraction 0–1 for a type-based opponent switch check during training. Requires a frozen checkpoint when nonzero; changes only Player 2's actions. [Rule and examples](../switch-pressure.md). |
| `--opponent-switch-cooldown` | `2` | Nonnegative number of normal decisions to skip after a voluntary opponent switch. Applies to heuristic interventions, not the underlying checkpoint policy. |
| `--reward-config` | `config/rewards/damage_knockout_win.json` | JSON reward settings, read afresh on every run, including resume. |
| `--readable-log`, `--no-readable-log` | On | Write live observations/actions to each pair's battle-readable.log (default: on) |
| `--battle-log`, `--no-battle-log` | On | Write compact actual battle text and HP events to battle.log (default: on) |
| `--output-dir` | Not set | New run directory. Default: `.local/run/ppo-<timestamp>/`; existing directories are rejected. |
| `--verbose` | Off | Print NPC connection phases |

## Usage notes

Choose exactly one team/state source. Showdown accepts JSON `--teams`, `--team-dir`, or profiles; mGBA accepts `.ssN` `--states`, `--state-dir`, or built profiles and requires `--rom`. `--team-size` requires a profile. Environment count is even and at least 2. With a frozen opponent only half the players learn. Explicit rollout, batch, epoch, learning-rate and entropy flags override resume settings; omitted flags keep saved values. The effective batch size must divide effective rollout steps × learning players. `--lobby-only` is rejected. Outputs and checkpoint timing are described in [reports](../battle-logs.md).

See [checkpoint resolution](../checkpoints.md) for latest selection and the
[controlled profiles](../../config/training_profiles/README.md) for focused team pools.

Hyperparameter defaults in the table apply to new models. On resume, omitted
flags use checkpoint values, even when those differ from these defaults.

### Battle-count budget

`--train-battles N` replaces the timestep budget: finish the PPO update in which
the whole pool reaches N ended battles (including truncations). Actual counts
can exceed N and appear in `summary.json`; evaluation budgets remain exact.
