# Storage and retention

[Training](ppo-training.md) · [Campaigns](curriculum.md) · [Logs](battle-logs.md) · [Cleanup reference](reference/prune-campaign.md)

Long campaigns can create thousands of battle traces and many checkpoint copies.
Checkpoint save frequency controls how often weights are written; it does not
limit battle logs or the number of completed attempts kept on disk.

Use compact campaign retention for long runs. It preserves the experiment's
inputs, results and selected models while removing older detailed traces and
unselected candidate weights. Retention is configurable; it is not a fixed quota
for the entire run directory.

## Standalone run sessions

`train_ppo.py --output-dir RUN --if-exists continue` keeps the existing run and
restores its saved configuration. Its root `episodes.jsonl` and `updates.jsonl`
retain cumulative history, while the current root summary reflects the combined
run. Earlier pair logs and `ppo/progress.csv` remain intact. Continuation writes
new detailed pair logs and PPO CSV files under `sessions/000002/`,
`sessions/000003/`, and so on; `sessions/000001/` records the first session's
metadata. Checkpoints remain under the run's root `checkpoints/` directory and
use the configured numbered-checkpoint retention.

The frozen opponent snapshot in `inputs/frozen-opponent.zip` remains fixed.
Keep run configuration, session metadata, root `episodes.jsonl` and
`updates.jsonl`, that snapshot and a recoverable learner checkpoint together
when archiving a resumable run. The metric logs reconstruct cumulative counts
and rolling windows on continuation. The recorded
training time excludes downtime between sessions.

Log rotation applies within each session, so many continuations can accumulate
historical session logs. Rotation is not a total run-size cap. Preserve the
metrics and model snapshots when removing older detailed logs manually.

If you want a fresh experiment, use a new output directory. For an intentional
restart in place, `--if-exists erase` deletes the old recognized standalone PPO
run after the new configuration passes validation. It requires full fresh-run
arguments; it cannot restore settings that it is about to erase. Choosing erase
from the interactive menu adds a typed confirmation; explicit `--if-exists erase`
does not ask again. Cancel leaves existing data unchanged, and an
active run lock prevents concurrent continuation or erasure. Campaign attempts
must be managed through the campaign runner and cleanup utility below.

## Periodic validation retention

For standalone [periodic evaluation](periodic-evaluation.md),
`--keep-evaluations 2` retains the two latest result directories plus the best
one if older, across sessions. `checkpoints/best.zip`, its metadata, the compact
evaluation history and pinned suite/opponents are kept independently of numbered
checkpoint rotation. Each retained evaluation includes its exact candidate ZIP.
Standalone evaluation directories outside a training run are not automatically
removed by this option.

## Clean up an existing campaign

Stop the campaign with Ctrl+C and wait for its workers to exit. Run cleanup from
the project root, not from `.local`. The command takes the same campaign lock as
the runner and refuses to clean a campaign that is still running.

```bash
cd ~/Documents/Code/Projects/Poke-AI
source .venv/bin/activate
curriculum_run=".local/run/curriculum-generalization-01"

du -sh "$curriculum_run"
python tools/prune_campaign.py \
  --run-dir "$curriculum_run" \
  --keep-attempts 2 --keep-best 2 --keep-debug-attempts 0
```

The first invocation is a dry run. Review its proposed deletions and estimated
recoverable space, then apply the same plan:

```bash
python tools/prune_campaign.py \
  --run-dir "$curriculum_run" \
  --keep-attempts 2 --keep-best 2 --keep-debug-attempts 0 \
  --apply

du -sh "$curriculum_run"
```

Cleanup removes selected files, not entire attempt directories. In particular,
it keeps:

- Campaign inputs, team snapshots, frozen opponents and evaluation suites.
- Promoted models and original candidate files needed by promotions or resume.
- Recent candidates and the best evaluated candidates selected by the limits.
- Evaluation results, per-battle evaluation records, reports and configurations.
- Training summaries, PPO metrics in `ppo/progress.csv`, and campaign history.

For completed attempts, a retained candidate is its `train/checkpoints/final.zip`.
Cleanup can remove the redundant `latest.zip` and numbered backups even when it
keeps that final model. The newest unfinished attempt keeps the checkpoints
needed for recovery.

New standalone evaluator outputs include `candidate.zip`. Within completed
campaign attempts, cleanup can remove these extra weight copies along with
redundant checkpoints; the campaign's selected final models and any snapshot
explicitly used for continuation remain protected. Evaluation configurations,
hashes, summaries and per-battle results remain available.

The largest removable groups are pair debug directories, old training
`episodes.jsonl` / `updates.jsonl`, redundant checkpoint backups and unselected
candidate weights. Deleting a candidate's weights means you cannot run a new
evaluation of that candidate; its recorded results remain readable. Removing
old training episode records also removes the ability to recompute arbitrary
episode blocks from those records.

Do not delete all ZIP files: campaign inputs, frozen opponents and promoted
checkpoints are ZIP files too. The preview measures what this campaign can
release. A particular percentage, such as 99%, is not guaranteed; model and
report retention settings determine what remains.

## Continue with bounded storage

Existing campaigns keep their previous retention behavior until explicitly
changed. Enable compact retention while preserving recent readable and spectator
logs when continuing:

```bash
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10 \
  --retention compact \
  --keep-attempts 2 --keep-best 2 --keep-debug-attempts 1 \
  --readable-log --battle-log --no-step-logs \
  --log-max-mb 10 --log-backups 2 --keep-episode-logs 2 \
  --keep-checkpoints 0 --checkpoint-every 0 \
  --latest-every-updates 10
```

These execution settings are saved for later invocations. Automatic cleanup runs
after completed attempts. New campaigns start with compact retention; use
`--retention off` to disable automatic pruning. Explicitly running the cleanup
utility remains a separate action.

This keeps bounded live battle logs and the most recent attempt's diagnostic
history per stage. Full per-decision traces remain disabled. If diagnosing an
individual turn requires those traces, enable them explicitly as shown below.

`--keep-attempts` and `--keep-best` control candidate history. The retained sets
are combined, so keeping two recent and two best can retain up to four different
candidates, plus protected campaign dependencies. Best candidates are selected
separately for each evaluation mode and saved rule revision, so changing promotion
rules can retain additional models. Setting a keep control to `-1` means unlimited.
Promoted and required resume models remain protected even when a limit is zero.
The newest unfinished attempt and its recovery data are protected; older aborted
attempts may be cleaned.

Applied cleanup records its counts and reclaimed logical bytes in
`retention-history.jsonl`. Filesystem compression, sparse files or snapshots can
make the actual freed space differ from the preview; check `du` and free disk space
afterward.

## Keep a small amount of debugging data

For runs where live battle logs are useful, retain them with rotation:

```bash
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10 \
  --retention compact --keep-debug-attempts 1 \
  --readable-log --battle-log --step-logs \
  --log-max-mb 10 --log-backups 2 --keep-episode-logs 2 \
  --keep-checkpoints 2
```

Each enabled Showdown log keeps its current file and at most two backups by
default. This covers readable and spectator logs, Node stderr and enabled
Gym/protocol traces. File limits apply independently to each log and each pair;
they are not a combined run-wide limit. A large number of pairs therefore still
uses more space. `--keep-episode-logs` retains recent completed Showdown episode
debug directories per pair, plus active or failed episodes. It does not limit
evaluation result records or checkpoints. mGBA has separate diagnostics; these
runtime rotation and episode limits apply to Showdown.

Rotation keeps normal records intact. An individual record larger than the file
limit is reduced to a marked text tail or a valid JSON omission marker, so a
bounded trace is not guaranteed to contain every event. Use unbounded logging
only for a deliberate diagnostic run if complete raw traces are needed.

Reward means, win rates, battles/hour, training summaries and campaign evaluation
history remain available with all detailed battle logging disabled. Follow the
campaign history with:

```bash
tail -n 30 -F "$curriculum_run/evaluations.md"
```

## Controls at a glance

| Scope | Control | Default | Purpose |
|---|---|---|---|
| Campaign | `--retention compact\|off` | New: `compact`; existing without a saved setting: `off` | Prune completed attempts automatically. |
| Campaign/cleanup | `--keep-attempts` | `2` | Keep recent candidate attempts per stage. |
| Campaign/cleanup | `--keep-best` | `2` | Keep best evaluated candidates per stage and evaluation-rule cohort. |
| Campaign/cleanup | `--keep-debug-attempts` | `0` | Keep detailed data for this many recent attempts per stage. |
| Training/evaluation | `--step-logs` / `--no-step-logs` | Off for Showdown | Write per-decision simulation/protocol traces. |
| Training/evaluation | `--log-max-mb` | `10` | Size of each rotating Showdown log in MiB; `0` means unlimited. |
| Training/evaluation | `--log-backups` | `2` | Number of rotated copies retained per log; `0` keeps only the current file. |
| Training/evaluation | `--keep-episode-logs` | `2` | Recent completed Showdown episode debug directories per pair, plus active/failed episodes; `0` disables them, `-1` keeps all. |
| Training | `--keep-checkpoints` | `2` | Numbered PPO backups retained; `0` keeps none, `-1` keeps all. |
| Training | `--checkpoint-every` | `1024` | Transitions between numbered backups; `0` disables their creation. |
| Training | `--latest-every-updates` | `1` | Updates between saves of `latest.zip`; separate from retention. |

The campaign runner forwards the applicable training/evaluation controls and
remembers explicit changes. See the full references for
[training](reference/train-ppo.md), [evaluation](reference/evaluate-ppo.md),
[campaigns](reference/run-curriculum.md) and [cleanup](reference/prune-campaign.md).
