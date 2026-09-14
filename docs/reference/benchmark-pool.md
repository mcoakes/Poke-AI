# Benchmark linked battles

[Utility index](README.md) · [Workflow guide](../benchmark.md) · [Source](../../mgba/benchmark_pool.py)

Run scripted mGBA matches and measure completed battles per hour across the pool.

```bash
python mgba/benchmark_pool.py --rom FireRed.gba --environments 4 \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1 \
  --commands MOVE1 --episodes 100 --speed unlimited --headless
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
| `--lobby-only` | Off | Inherited launcher option; rejected because benchmarking requires battles. |
| `--speed` | `1` | Positive emulator speed multiplier (for example `4`) or `unlimited`. |
| `--environments` | `8` | Even number of Gymnasium players (8 = 4 battles) |
| `--states` | Required | Two alternating counter states, or one per player |
| `--base-port` | `55355` | First adapter port; subsequent players use consecutive ports. |
| `--commands` | `MOVE1` | MOVE1..4, SWITCH1..6, or one benchmark JSON path |
| `--repeat-commands`, `--no-repeat-commands` | On | Cycle commands until battle ends (default: on) |
| `--episodes` | `100` | Total battle attempts across all pairs (default: 100) |
| `--max-steps` | `200` | Decision limit per battle attempt |
| `--command-timeout` | `120` | Maximum wall-clock seconds for one paired battle decision. |
| `--seed` | `1` | Gymnasium reset seed; the scripted policy is fixed and FireRed RNG comes from the save states. |
| `--verbose` | Off | Print connection phases as well as episode rates |

## Usage notes

Provide two alternating states or one per environment. Environment count must be even and at least 2; episodes must be at least the number of pairs. `--lobby-only` is rejected. Commands use the [benchmark format](../benchmark.md#command-sequences), which differs from the pool switch-test format. Rates include launch, NPC connection, reset and cleanup. Logs go to `.local/run/benchmark-<timestamp>/`.
