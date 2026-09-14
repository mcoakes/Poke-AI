# Launch independent emulators

[Utility index](README.md) · [Workflow guide](../inspection.md) · [Source](../../mgba/launch_emulators.py)

Start one independent mGBA process per port, with its adapter loaded automatically.

```bash
python mgba/launch_emulators.py --rom FireRed.gba --ports 55355 \
  --states states/counter/fire_vs_water.ss1 --speed 2
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--rom` | Required | Path to the supported FireRed v1.0 ROM. |
| `--emulator` | `.local/mgba/bin/mgba-qt` | Path to the patched project-local `mgba-qt` executable. |
| `--ports` | `55355 55356` | One or more ports; starts a separate emulator process for each. |
| `--states` | Not set | One emulator save state per port, in matching order |
| `--profile` | `firered_us_v1_0` | Lua ROM profile identifier. Currently supported: `firered_us_v1_0`. |
| `--speed` | `1` | Positive emulator speed multiplier (for example `4`) or `unlimited`. |
| `--headless` | Off | Run mGBA under Xvfb instead of showing windows. |
| `--startup-timeout` | `20.0` | Wall-clock seconds to wait for adapter readiness. |
| `--allow-unsupported-rom` | Off | Launch despite a SHA-1 mismatch (adapter commands will reject it). |

## Usage notes

One or more ports are supported. If states are supplied, provide exactly one per port. These processes are independent; use [launch_link.py](launch-link.md) or [launch_pool.py](launch-pool.md) to establish a native cable session. Ctrl+C stops the processes. Runtime logs are under `.local/run/`.
