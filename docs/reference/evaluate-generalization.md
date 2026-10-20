# evaluate_generalization.py

[Utility index](README.md) · [Walkthrough and example](../generalization-evaluation.md) · [Source](../../mgba/evaluate_generalization.py)

Compare two periodic-validation best checkpoints on new JSON teams with their
shared opponents and rewards. Showdown only; run from the project root.

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Print usage and exit. |
| `--previous-run` | Required | Reference run; uses `checkpoints/best.zip` and `best.json`. |
| `--candidate-run` | Required | New run; same saved validation suite and rewards. |
| `--team-dir` | Required | New team JSON directory; recursive. |
| `--exclude-team-dir` | None | Additional previously used team directory; recursive, repeatable. |
| `--exclude-suite` | None | Additional Showdown suite whose embedded teams are excluded; repeatable. |
| `--output-dir` | Required | New directory outside input runs/team directories; never overwritten. |
| `--episodes-per-opponent` | `1000` | Positive even battles per category per candidate, including both seats; total is six times this value. |
| `--seed` | `26092703` | Test-schedule and report-bootstrap seed. |
| `--environments` | `48` | Positive even number of player environments. |
| `--showdown-workers` | `process` | `process` or `thread` workers. |
| `--node` | `node` | Node executable. |
| `--device` | `cpu` | Policy inference device; simulation stays on CPU. |
| `--torch-threads` | `1` | Positive PyTorch thread count. |
| `--battle-log` / `--no-battle-log` | Enabled | Keep/disable bounded spectator logs. |
| `--log-max-mb` | `2` | Positive finite per-file limit in MiB; no unlimited setting here. |
| `--log-backups` | `1` | Nonnegative rotated backups per log. |
| `--dry-run` | Off | Validate inputs, print team counts, exit without writes or battles. |

Both runs need `inputs/evaluation/suite.json` containing Random, Dumb and Smart
opponents and `inputs/reward.json`. Missing/changed recorded teams, mismatched
checkpoint hashes, different suites or rewards, and zero accepted teams are
errors. Interrupted comparisons retain partial results; use a new directory to
retry. See the walkthrough for overlap-check scope and result interpretation.
