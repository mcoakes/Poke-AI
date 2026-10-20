# Prune campaign data

[Utility index](README.md) · [Storage guide](../storage.md) · [Source](../../tools/prune_campaign.py)

Remove old diagnostic data and unselected candidate weights from an existing
curriculum while preserving its inputs, reports and required models. Stop the
campaign first. The utility uses the campaign lock to prevent concurrent cleanup
and training.

```bash
python tools/prune_campaign.py \
  --run-dir .local/run/curriculum-generalization-01 \
  --keep-attempts 2 --keep-best 2 --keep-debug-attempts 0
```

The default is a dry run. Add `--apply` after reviewing the proposed deletions and
estimated space savings.

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit. |
| `--run-dir` | Required | Existing campaign directory; run from the project root when using project-relative paths. |
| `--keep-attempts` | `2` | Retain candidates from this many recent attempts per stage; `0` keeps only other selected/protected candidates, `-1` removes this limit. |
| `--keep-best` | `2` | Retain this many best evaluated candidates per stage and evaluation-rule cohort; `0` disables score-based retention, `-1` keeps all evaluated candidates. |
| `--keep-debug-attempts` | `0` | Keep detailed pair/training traces for this many recent attempts per stage; `-1` keeps all. |
| `--apply` | Off | Actually delete the proposed files. Without this flag, only print the cleanup plan. |

Candidate sets are combined. Inputs, frozen opponents, promoted candidates and
models required to continue the campaign take precedence over numerical limits.
Best candidates are ranked separately for each evaluation mode and saved rule
revision, so multiple cohorts can retain more than `--keep-best` models per stage.
The newest unfinished attempt retains its debug and recovery data; older aborted
attempts may be pruned.
Results, summaries, evaluation per-battle records, saved suites and PPO progress
CSVs remain. Cleanup does not remove entire attempt directories.

Retained candidates from completed attempts use `train/checkpoints/final.zip`;
redundant `latest.zip` and numbered backups may be removed. The newest unfinished
attempt keeps its recovery checkpoints.

The retained history supports reviewing completed evaluations; it does not
guarantee that a pruned candidate can be evaluated again. Its weights must have
been retained or copied elsewhere for that. Pruned detailed logs and training
episode records cannot be reconstructed from aggregate reports.

Apply writes `retention-history.jsonl` with the cleanup counts and logical bytes.
It revalidates the plan before deletion and rejects changes to the files selected
for cleanup. Physical space recovered can differ on compressed, sparse or
snapshotted filesystems.

This command performs cleanup once. Configure
[`run_curriculum.py --retention compact`](run-curriculum.md#storage-settings) for
automatic cleanup after completed attempts. No deletion setting guarantees a
specific percentage reduction or a hard total disk-size cap.
