# Opponent pool configurations

[Walkthrough and commands](../../docs/opponent-pools.md)

Pass a JSON file to `--opponent-pool` in `train_ppo.py` or `evaluate_ppo.py`.
The examples reference the frozen model copied by the diversity experiment;
edit `path` if your model is elsewhere. Paths are relative to the JSON file.

- `rules_and_frozen.json`: 70% frozen checkpoint / 30% rules for training.
- `rules_and_frozen_eval.json`: equal weights for evaluation; 2,000 battles gives
  1,000 per opponent, with both candidate seats represented equally.

Schema:

```json
{
  "version": 1,
  "opponents": {
    "reference": {"kind": "checkpoint", "path": "../../models/reference.zip", "weight": 7},
    "rules": {"kind": "rule_based", "weight": 3},
    "random": {"kind": "random", "weight": 1}
  }
}
```

| Field | Meaning |
|---|---|
| `version` | Required integer `1`. |
| `opponents` | Required nonempty object, keyed by unique, nonempty opponent IDs. |
| `kind` | Required: `checkpoint`, `rule_based`, or `random`. |
| `path` | Required for checkpoints: absolute path, JSON-relative path, or `latest`. Resolved once and copied into the run. |
| `weight` | Positive finite relative weight, default `1`. Weights are normalized; omit opponents you do not want. |
| `group` | Checkpoints: `learned` by default, or `dumb`/`smart` for existing report categories. Rules must be `rules`; Random must be `random`. |
| `label` | Optional human-readable label; defaults to the opponent ID. |
| `sha256` | Optional checkpoint fingerprint; generated automatically in pinned copies. Mismatches are rejected. |
| `identity` | Optional rule identity; generated automatically with version and code/catalog fingerprint. Mismatches are rejected. |

Add multiple checkpoint entries for multiple learned opponents. One entry is
also valid for training/evaluation against only one kind. Duplicate checkpoint
bytes share one loaded model; entry weights still count independently.
Do not edit generated pinned pool files to change an existing run. Start a
new output directory with `--resume` and your new pool instead.
