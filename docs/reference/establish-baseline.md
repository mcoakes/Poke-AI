# Establish a campaign baseline

[Utility index](README.md) · [Workflow](../baseline.md) · [Source](../../mgba/establish_baseline.py)

Compare frozen Starting, Best and Latest campaign candidates against shared
Random, Dumb and Smart schedules, separately for training and validation teams.
This runs local Showdown evaluations and does not train or promote models.

```bash
python mgba/establish_baseline.py \
  --campaign-run .local/run/curriculum-generalization-01 --stage 1 \
  --run-dir .local/run/baseline-generalization-01 \
  --episodes-per-opponent 1000 --environments 48 \
  --showdown-workers process
```

## Run and snapshot options

Stop the source campaign for initial snapshot creation or a dry run; the command
takes its campaign lock. Resume from the independent snapshot with `--run-dir`
only. Use a new run directory to change models, teams, opponents or the schedule.
The baseline run directory must be outside the source campaign. Checkpoint
override flags accept a path or `latest` (`later` is an alias). Explicit `latest`
uses the project's global checkpoint discovery, then freezes that result; automatic
candidate selection is scoped to the chosen campaign and stage.

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit. |
| `--run-dir` | Required | New baseline directory outside the source campaign, or an existing baseline to continue. |
| `--campaign-run` | Required for a new baseline | Source curriculum directory; preserved and used only to choose/snapshot initial inputs. |
| `--stage` | `1` | Campaign stage whose anchor, retained candidates and cumulative team pools are compared. |
| `--episodes-per-opponent` | `1000` | Positive even total battles per candidate, split and opponent category, including side swaps. The Smart pool shares this budget. |
| `--seed` | `26001` | Seed for generating the shared matchup schedules. Saved on initialization. |
| `--max-steps` | Campaign decision limit | Per-battle decision cap. Unfinished battles truncate. |
| `--starting-checkpoint` | Stage anchor | Checkpoint path or `latest` to use for Starting. |
| `--best-checkpoint` | Best retained in latest completed rules cohort | Checkpoint path or `latest` to use for Best. |
| `--latest-checkpoint` | Latest recoverable stage candidate | Checkpoint path or `latest` to use for Latest. |
| `--dumb-checkpoint` | Saved Dumb reference, otherwise initial campaign reference | Dumb reference checkpoint path or `latest`. |
| `--smart-checkpoint` | Saved Smart pool, otherwise campaign's original learner | One Smart checkpoint path or `latest` instead of the saved pool. |
| `--validation-dir` | Saved cumulative validation pool | Recursive validation team JSON directory. Required when the campaign lacks separate validation groups; duplicate training parties are rejected even if reordered. |
| `--dry-run` | Off | Inspect selection and budget without writing the snapshot or running battles. |

Byte-identical candidates share their evaluation work. Candidate and reference
snapshots, rewards, team contents and suites remain pinned when continuing.
Validation reused in campaign promotion is development data; a new schedule does
not make it an untouched final test.

## Execution and logging

Worker, logging and report settings can change when resuming; omitted flags reuse
the saved settings. Code, dependency versions, Node version, inference device and
PyTorch thread count must match the saved runtime identity. Use a new baseline
directory to compare a changed runtime. All simulation runs headlessly without
an emulator speed cap.

| Flag | Initial default | Meaning |
|---|---|---|
| `--environments` | `48` | Even number of players; two players per concurrent battle. |
| `--showdown-workers` | `process` | `process` or `thread` Python workers; each pair also uses a local Node process. |
| `--node` | `node` | Node.js executable. |
| `--device` | `cpu` | PyTorch policy inference device; fixed by the saved runtime identity. |
| `--torch-threads` | `1` | PyTorch CPU thread count; fixed by the saved runtime identity. |
| `--command-timeout` | `120` | Wall-clock seconds allowed for one paired decision. |
| `--bootstrap-samples` | `4000` | Resamples for the report's paired-matchup 95% intervals; at least 100. Can change when continuing without rerunning completed battles. |
| `--bootstrap-seed` | `1` | Seed for reproducible report resampling. Can change when continuing. |
| `--readable-log`, `--no-readable-log` | On | Diagnostic observation/action logs per pair. |
| `--battle-log`, `--no-battle-log` | On | Spectator battle text and HP logs per pair. |
| `--step-logs`, `--no-step-logs` | Off | Full per-decision Gym/protocol traces and episode team copies. |
| `--log-max-mb` | `2` | Size of each rotating Showdown log in MiB; `0` means unlimited. |
| `--log-backups` | `1` | Rotated copies per log; `0` keeps only the current file. |
| `--keep-episode-logs` | `2` | Completed debug episode directories per pair, plus active/failed episodes; `0` disables traces, `-1` keeps all. |

## Reports and continuation

`status.json` tracks work. `baseline-report.md` and `baseline-report.json` summarize
the comparisons. Evaluation artifacts live in
`runs/<canonical-candidate>/<split>/attempt-NNN/`; candidate roles with identical
weights point to the same evaluation. Completed evaluations are reused on resume;
failed/incomplete evaluations are retried under a new attempt directory.

The report compares Best and Latest with Starting using matched schedules and
95% bootstrap intervals that keep side-swapped matchup pairs together. Each
split/opponent result is `improvement`, `regression` or `no_clear_change` according to
whether the difference interval is above zero, below zero or includes zero.
Fewer than 20 matchup pairs yields `insufficient_data`. Win rates include draws
and truncations as non-wins; constant observed outcomes can produce zero-width
intervals without establishing certainty.
Intervals apply to individual comparisons; this utility defines no promotion
gate and makes no simultaneous confidence claim across the entire report.
