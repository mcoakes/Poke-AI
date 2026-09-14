# Compare battle blocks

[Utility index](README.md) · [Comparison guide](../battle-blocks.md) · [Source](../../mgba/compare_blocks.py)

Summarize disjoint blocks from a current or older PPO run's `episodes.jsonl`.
It reads the file without controlling training or modifying the run.

```bash
python mgba/compare_blocks.py --run-dir .local/run/ppo-YOUR_RUN
```

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit |
| `--run-dir` | Required | Run directory containing `episodes.jsonl` |
| `--block-size` | 500 | Positive count of ended battles per block; changes this report only |
| `--json` | Off | Output a JSON array instead of readable summaries |

Includes a labeled partial final block. A final record still being written is
ignored until a subsequent invocation. Missing files and malformed complete
records produce an error. JSON win rates are fractions; readable rates are percentages.
