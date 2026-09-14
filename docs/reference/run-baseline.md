# Run an NPC baseline

[Utility index](README.md) · [Workflow guide](../turn-environment.md) · [Source](../../mgba/run_baseline.py)

Run random legal turn actions, or replay recorded actions, against an NPC in one existing emulator.

```bash
python mgba/run_baseline.py --port 55356 --emulator-pid 12345 \
  --episodes 10 --seed 1 --moves-first
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--port` | `55356` | Port of the already-running adapter to control or inspect. |
| `--emulator-pid` | Required | Actual `mgba-qt` process ID for this single NPC battle; used to pause it while waiting for decisions. |
| `--episodes` | `10` | Number of NPC episodes to run; positive integer. |
| `--max-steps` | `200` | Maximum turn decisions per NPC episode. |
| `--decision-timeout` | `120` | Positive wall-clock seconds allowed to resolve one decision. |
| `--seed` | `1` | Seed for reproducible selection or policy sampling. mGBA game RNG still comes from the save state. |
| `--allow-escape` | Off | Allow action 10 (escape) in random action selection. |
| `--moves-first` | Off | Choose a legal move when one exists, rather than randomly selecting a voluntary switch. |
| `--output` | Timestamped path under `.local/run/` | Directory for the baseline run's config, episode summaries and steps. |
| `--replay-log` | Not set | Existing baseline `steps.jsonl` to replay instead of sampling actions. |
| `--replay-episode` | `1` | 1-based episode number to select from `--replay-log`. |

## Usage notes

Replace `12345` with the actual PID of your single-battle emulator. Its adapter must have an NPC battle state configured for reset. This utility does not launch an emulator or use the paired PPO reward configuration. Replay uses the selected episode's commands; see its JSONL output for observations, decisions and rewards.
