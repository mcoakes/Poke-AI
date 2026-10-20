# audit_move_order.py

[Walkthrough](../move-order-audit.md) · [Utilities](README.md)

```bash
python mgba/audit_move_order.py \
  --evaluation-dir .local/run/eval-combined-diversity-01 \
  --output-dir .local/run/audit-diversity-move-order-01
```

| Flag | Default | Meaning |
|---|---|---|
| `--evaluation-dir PATH` | Required | Saved deterministic side-swapped Showdown evaluation, v2 candidate, pinned Rules opponent, config, suite, episodes and checkpoint |
| `--output-dir PATH` | Required | New audit directory, never overwritten |
| `--matchups N` | `32` | 1–100 outcome-unfiltered Rules pairs; both seats replayed, capped by available completed pairs |
| `--seed N` | `261002` | Sample-selection seed; battle seeds stay unchanged |
| `--environments N` | `2` | Positive even player count; two makes sequential logs |
| `--showdown-workers {thread,process}` | `process` | Simulator execution mode |
| `--node PATH` | `node` | Node executable |
| `--device DEVICE` | `cpu` | Policy inference device |
| `--log-max-mb N` | `2` | Finite positive log-file cap in MiB |
| `--log-backups N` | `1` | Nonnegative backups per log |
| `-h`, `--help` | — | Usage |

The active Python interpreter runs the evaluation child process. Spectator logs
are enabled; readable observation logs and full step traces are off. Exit status
0 means outcome/truncation/decision-count reproduction passed; 1 indicates a
replay failure/mismatch; 2 indicates invalid inputs. Probability or execution
review flags are diagnostic findings, not automatic failing exit statuses.
The report includes normalized slot-sensitivity rates and up to 12 retained
execution differences with selected/announced move names. Detailed bounded
examples remain in `replay/move-order-audit.json`; rotated traces are unchanged.
Confirmed Struggle fallbacks have their own `execution_struggle_fallback` count,
supported by the pre-execution simulator request and queued move. Retained JSON
examples include that evidence in `submitted_choice`. Missing or conflicting
evidence remains a review flag. No new CLI flags or checkpoint changes are needed.
