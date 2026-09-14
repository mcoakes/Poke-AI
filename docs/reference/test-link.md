# Check a cable connection

[Utility index](README.md) · [Workflow guide](../link-test.md) · [Source](../../mgba/test_link.py)

Poll two adapters until their native two-player connection and data exchange pass validation.

```bash
python mgba/test_link.py --ports 55355 55356 --require-battle --timeout 180
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--ports` | `55355 55356` | Adapter ports in player order; use distinct available local ports. |
| `--timeout` | `180` | Maximum wall-clock seconds to wait for a valid cable connection. |
| `--require-battle` | Off | Also require both games to report an active link battle. |
| `--output` | Timestamped path under `.local/run/` | Destination JSONL file for connection reports and final success/failure. |

## Usage notes

Use two cores from the same `launch_link.py` session or pool match. Independent emulator processes do not form a native cable session. This checker does not navigate the NPC.
