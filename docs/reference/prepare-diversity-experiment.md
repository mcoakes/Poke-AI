# prepare_diversity_experiment.py

[Utility index](README.md) · [Walkthrough and command](../experiments/team-diversity.md) · [Source](../../mgba/prepare_diversity_experiment.py)

Prepare a Showdown experiment with disjoint JSON pools, immutable model/reward
copies, and `train.sh` / `test.sh`. Does not train or evaluate during preparation.

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Usage and exit. |
| `--source-run` | Required | Run with best.zip/best.json, frozen-opponent metadata, pinned inputs and pooled periodic validation. |
| `--output-dir` | Required | New experiment directory outside source/exclusion directories; never overwritten. |
| `--train-teams` | `3000` | Unique training team count. |
| `--validation-teams` | `500` | Unique validation team count. |
| `--test-teams` | `500` | Unique reserved test team count. |
| `--team-size` | `3` | Party size 1–6; all splits use the same size. |
| `--seed` | `26092801` | Generation, splitting and suite seeds; source base seed is retained for training. |
| `--episodes-per-opponent` | `1000` | Positive even battles per category per suite; both seats included. |
| `--total-timesteps` | `500000` | Additional training steps written into train.sh; rounded to a PPO rollout. |
| `--learning-rate` | `0.0001` | Positive finite rate written into train.sh; other PPO hyperparameters come from the model. |
| `--environments` | `64` | Positive even player count for training and evaluations, process workers. |
| `--exclude-team-dir` | None | Other known team directory, recursive and repeatable. |
| `--exclude-suite` | None | Other known Showdown suite whose embedded teams are excluded; repeatable. |

Counts must be positive and total at most 10,000. Generator settings are fixed
for this controlled workflow (see walkthrough); use the general team generator
for other distributions. Up to 20 deterministic generation batches fill quotas
after excluding duplicates and sparse movesets. Exhaustion fails explicitly.
Snapshots are verified against saved hashes. Scope, realized styles, rejected
counts, source/catalog hashes and script arguments appear in `experiment.json`.
