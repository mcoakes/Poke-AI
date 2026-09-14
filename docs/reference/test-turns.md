# Try NPC turn actions

[Utility index](README.md) · [Workflow guide](../turn-environment.md) · [Source](../../mgba/test_turns.py)

Interactively submit a move or switch and inspect its reward at the next decision boundary.

```bash
python mgba/test_turns.py --port 55356 --emulator-pid 12345
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--port` | `55356` | Port of the already-running adapter to control or inspect. |
| `--emulator-pid` | Required | Actual `mgba-qt` process ID for this single NPC battle; used to pause it while waiting for decisions. |

## Usage notes

Replace `12345` with the actual mGBA PID. Actions 0–3 select moves, 4–9 select original party slots, and 10 attempts escape. This single-NPC utility pauses the process while waiting for input; do not use it on a native linked pair. Enter `q` to quit.
