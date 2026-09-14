# Test button inputs

[Utility index](README.md) · [Workflow guide](../inspection.md) · [Source](../../mgba/test_inputs.py)

Send one raw GBA action to each selected adapter and verify the requested frame duration.

```bash
python mgba/test_inputs.py --ports 55355 55356 --actions DOWN UP --frames 1
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--host` | `127.0.0.1` | Adapter host. The supplied Lua server binds to `127.0.0.1`; this is a local workflow. |
| `--ports` | `55355 55356` | Adapter ports in player order; use distinct available local ports. |
| `--actions` | `A A` | One GBA action per port: `NOOP`, `A`, `B`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `START`, `SELECT`, `L`, `R`, or integer 0–10. |
| `--frames` | `1` | Frames to hold each selected action; integer 1–600, enforced by the adapter. |

## Usage notes

Provide one action for each port. For a single emulator use `--ports 55355 --actions A`. `DOWN UP` means DOWN on the first port and UP on the second, not a two-button sequence. Buttons are raw inputs, not the PPO move/switch actions.
