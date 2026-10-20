# Evaluate a model directory

[Utility index](README.md) · [Walkthrough](../model-evaluation.md) · [JSON schema](../../config/evaluations/README.md) · [Source](../../mgba/evaluate_models.py)

Run a frozen-policy Showdown round robin over several team configurations.

```bash
python mgba/evaluate_models.py \
  --models-dir models \
  --config config/evaluations/round_robin.json \
  --run-dir .local/run/model-evaluation-01 \
  --environments 48 --showdown-workers process
```

## Run and input selection

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit. |
| `--run-dir` | Required | New tournament directory, or an existing one to resume. Empty shell variables are rejected. |
| `--config` | Required for a new run | Evaluation-plan JSON. Paths inside it resolve relative to that file. |
| `--models-dir` | Required for a new run | Directory of saved PPO ZIPs. Models are copied once into the run. |
| `--model-glob` | `*.zip` | New run: discovery pattern relative to the model directory. Quote it. `smart_v*.zip` selects named Smart models; `**/final.zip` searches subdirectories. |
| `--episodes` | Config value, otherwise `1000` | New run: battles per unordered model pair per configuration. Must be positive and even. |
| `--max-steps` | Config value, otherwise `200` | New run: per-battle decision cap. |
| `--seed` | Config value, otherwise `10001` | New run: seed for the saved team/battle schedules. |
| `--include-random`, `--no-include-random` | Config value, otherwise on | New run: include/exclude one random legal-action participant. At least two total participants are required. |
| `--dry-run` | Off | List models and planned battle count without writing files or running battles. Checks files and teams; skips PPO loading. Also works on a saved tournament. |

`--config`, model selection and sampling flags are new-run inputs. Omit them when
resuming; they are stored in `tournament.json`. New models, changed teams or a
different battle count require a new run directory. Identical checkpoint files
are deduplicated and their aliases recorded. PPO files must use the existing
11-action battle schema with supported v1 or v2 observations. Each model receives
its own inputs; `model-schemas.json` records detected versions. See
[observation v2](../observation-v2.md).

## Execution settings

Explicit values are remembered for future invocations; omitted values retain
saved settings. These flags may change when resuming.

| Flag | Initial default | Meaning |
|---|---|---|
| `--environments` | `48` | Even number of concurrent players within the current pair/configuration test. Two players form one battle; separate model-pair tests run in sequence. |
| `--showdown-workers` | `process` | `process` or `thread` Python workers, with one local Node simulator per concurrent battle. |
| `--node` | `node` | Node executable. |
| `--device` | `cpu` | PyTorch inference device. Simulator execution remains on CPU. |
| `--torch-threads` | `1` | Central PyTorch CPU thread count. |
| `--command-timeout` | `120` | Positive integer seconds per paired decision. |
| `--keep-battle-logs`, `--no-keep-battle-logs` | Off | Enable/disable per-turn traces and readable/spectator logs. Compact per-battle results are always kept. |

Showdown already runs headlessly and without a speed limit; no `--speed` or
`--headless` flag is needed. This utility does not launch mGBA.

## Results and resume

Rank by `(wins + 0.5 × draws) / all battle attempts`. Truncations score zero for
both models. The final overall ranking weights every opponent and configuration
equally. Rankings are provisional until all tests finish; equal scores share a
rank. No Elo update or training reward is involved.

```bash
python mgba/evaluate_models.py --run-dir .local/run/model-evaluation-01
watch -n 5 cat .local/run/model-evaluation-01/leaderboard.md
```

Completed tests are validated and skipped. An interrupted test restarts its
complete saved schedule in a new attempt directory. `matches.jsonl` appends one
result per completed test; Markdown/CSV standings are rewritten. Keep model
snapshots, suites, evaluator `config.json`, `summary.json` and `episodes.jsonl`
for resume. Detailed logs are optional.

| Exit code | Meaning |
|---:|---|
| `0` | Finished, already complete, or successful dry run. |
| `1` | Input/model validation or evaluation failure. No unfinished test is ranked. |
| `2` | Invalid CLI syntax. |
| `130` | Interrupted; completed tests remain available for resume. |
