# review_rule_losses.py

[Walkthrough](../rules-loss-review.md) · [Utilities](README.md)

```bash
python mgba/review_rule_losses.py \
  --evaluation-dir .local/run/eval-combined-diversity-01 \
  --output-dir .local/run/review-diversity-rules-01
```

| Flag | Default | Meaning |
|---|---|---|
| `--evaluation-dir PATH` | Required | Existing deterministic, side-swapped Showdown evaluation with a v2 candidate and rules cases |
| `--output-dir PATH` | Required | New directory; never overwrites an existing review |
| `--matchups N` | `12` | Sample 1–100 loss-containing matchups; both candidate seats replayed; capped by eligible count |
| `--seed N` | `261001` | Sample selection seed; does not replace original battle seeds |
| `--environments N` | `2` | Even concurrent player count; two gives sequential logs in one pair |
| `--showdown-workers {thread,process}` | `process` | Simulator worker implementation |
| `--node PATH` | `node` | Node.js executable |
| `--device DEVICE` | `cpu` | PPO inference device |
| `--log-max-mb N` | `2` | Finite positive cap per diagnostic log file, in MiB |
| `--log-backups N` | `1` | Nonnegative rotated backups per diagnostic log |
| `-h`, `--help` | — | Show usage |

Readable, spectator and rule logs are enabled; full simulator step traces are
disabled. The replay subprocess uses the active Python interpreter. Source
checkpoint bytes and reward configuration are copied into the review's inputs.
It never trains. Exit status: 0 for verified replay, 1 for replay failure or
outcome/decision-count mismatch, 2 for rejected input. Failed replay diagnostics
are retained. Use a new output directory to retry.
