# Monitor an NPC episode

[Utility index](README.md) · [Workflow guide](../turn-environment.md) · [Source](../../mgba/test_episode.py)

Reset one emulator, then watch frame-based Gymnasium rewards while you play the battle manually.

```bash
python mgba/test_episode.py --port 55355 --max-steps 20000
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--port` | `55355` | Port of the already-running adapter to control or inspect. |
| `--max-steps` | `20000` | Maximum NOOP monitoring steps before truncation. |

## Usage notes

The monitor sends NOOP and reports state changes. These are frame-level monitoring steps, not one-turn PPO decisions. Configure an NPC battle reset state before running it.
