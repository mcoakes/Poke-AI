# Launch one linked pair

[Utility index](README.md) · [Workflow guide](../link-test.md) · [Source](../../mgba/launch_link.py)

Start two mGBA cores in a shared native cable session, optionally navigating the NPC automatically.

```bash
python mgba/launch_link.py --rom FireRed.gba \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1 \
  --auto-connect --speed 2
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--rom` | Required | Path to the supported FireRed v1.0 ROM. |
| `--emulator` | `.local/mgba/bin/mgba-qt` | Path to the patched project-local `mgba-qt` executable. |
| `--headless` | Off | Run mGBA under Xvfb instead of showing windows. |
| `--startup-timeout` | `30` | Wall-clock seconds to wait for adapter readiness. |
| `--connect-timeout` | `180` | Wall-clock seconds allowed for each NPC connection or battle-start stage. |
| `--lobby-only` | Off | Stop automatic setup in the Colosseum before walking to battle pads |
| `--speed` | `1` | Positive emulator speed multiplier (for example `4`) or `unlimited`. |
| `--states` | Required | Two pre-connection overworld/Cable Club checkpoints |
| `--ports` | `55355 55356` | Exactly two ports, in player order. |
| `--auto-connect` | Off | Navigate the NPC automatically from generated counter checkpoints |

## Usage notes

Exactly two states and two ports are required. Without `--auto-connect`, you control NPC dialogue. With it, setup reaches the battle command menus; `--lobby-only` stops automated setup in the Colosseum. Ctrl+C closes the pair. Diagnostics are written to `.local/run/link-<timestamp>/`.
