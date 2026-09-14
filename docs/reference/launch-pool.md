# Launch a linked pool

[Utility index](README.md) · [Workflow guide](../link-pool.md) · [Source](../../mgba/launch_pool.py)

Pair an even number of players, connect each pair through the NPC, and start their link battles.

```bash
python mgba/launch_pool.py --rom FireRed.gba --environments 4 \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1 \
  --commands config/commands/switch_stress.json --speed unlimited
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
| `--environments` | `4` | Even number of players; 4 creates two linked pairs (default: 4) |
| `--states` | Required | Two alternating counter checkpoints, or one per environment |
| `--base-port` | `55355` | First adapter port; subsequent players use consecutive ports. |
| `--ports` | Not set | One unique port per environment; alternative to `--base-port`. |
| `--commands` | Not set | JSON command cycles to run after every pair reaches battle readiness |
| `--command-timeout` | `120` | Maximum wall-clock seconds for one paired battle decision. |

## Usage notes

Supply two alternating states or one per player. `--ports` and `--base-port` are exclusive. `--commands` cannot be combined with `--lobby-only`. Without commands, players remain at the ready battle menus for your controller. Outputs include `pool.json` and per-match diagnostics under `.local/run/pool-<timestamp>/`. Ctrl+C stops all pairs.
