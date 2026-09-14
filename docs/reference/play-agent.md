# Play against a trained agent

[Utility index](README.md) · [Workflow guide](../human-vs-ai.md) · [Source](../../mgba/play_agent.py)

Launch one visible FireRed link battle with a human on one side and a saved PPO policy on the other.

```bash
python mgba/play_agent.py --rom FireRed.gba --checkpoint checkpoints/latest.zip \
  --states states/counter/starters_charizard_lead.ss1 states/counter/starters_venusaur_lead.ss1 \
  --human-player 1 --speed 1
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--rom` | Required | Path to the supported FireRed v1.0 ROM. |
| `--emulator` | `.local/mgba/bin/mgba-qt` | Path to the patched project-local `mgba-qt` executable. |
| `--headless` | Off | Inherited launcher option; rejected because human play requires visible windows. |
| `--startup-timeout` | `30` | Wall-clock seconds to wait for adapter readiness. |
| `--connect-timeout` | `180` | Wall-clock seconds allowed for each NPC connection or battle-start stage. |
| `--lobby-only` | Off | Inherited launcher option; rejected because human play requires a battle. |
| `--speed` | `1` | Positive emulator speed multiplier (for example `4`) or `unlimited`. |
| `--checkpoint` | Required | Checkpoint path or `latest`; resolves once before launching |
| `--states` | Not set | Two fixed counter states, in player order |
| `--state-dir` | Not set | Draw two random .ssN counter states from this directory |
| `--team-profile` | Not set | Select competitive, casual, random, or a mixture of all three. Choices: `competitive`, `casual`, `random`, `mixed`. |
| `--mirror-teams` | Off | Random sources only: sample one team/state per battle for both players; see [mirror matches](../team-profiles.md#random-mirror-matches). |
| `--team-size` | Not set | Restrict both players to this party size; requires --team-profile. Choices: `1`, `2`, `3`, `4`, `5`, `6`. |
| `--human-player` | `1` | Which side you control: `1` or `2`; the other side is the saved policy. |
| `--ports` | `55355 55356` | Exactly two ports, in player order. |
| `--stochastic` | Off | Sample the trained policy; default: highest-probability legal action |
| `--seed` | `1` | Seed for reproducible selection or policy sampling. mGBA game RNG still comes from the save state. |
| `--device` | `cpu` | PyTorch policy device, such as `cpu` or `cuda`; does not move the emulator or simulator to the GPU. |
| `--max-steps` | `200` | Battle decision limit, as in training (default: 200) |
| `--output-dir` | New `.local/run/human-<timestamp>` folder | New directory for this match's configuration, runtime files and report. An existing path is rejected. |
| `--no-wait` | Off | Close both windows when the match ends without waiting for Enter. Used by the browser launcher. |
| `--setup-speed` | Same as `--speed` | Optional faster link setup, for example `unlimited`; changes both cores to `--speed` before READY. Requires the runtime-speed patch. |

## Usage notes

Choose exactly one of `--states`, `--state-dir`, or `--team-profile`; `--team-size` requires the profile. The model loads once and never trains or saves weights. `--headless` and `--lobby-only` are rejected. Wait for READY before controlling the human window. Results are in `.local/run/human-<timestamp>/`.
