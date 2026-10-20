# Evaluate PPO

[Utility index](README.md) · [Workflow and examples](../evaluation.md) · [Source](../../mgba/evaluate_ppo.py)

Observation versions are detected from each checkpoint, with no extra flag.
Showdown supports mixed v1/v2 candidates and opponents, including saved pools;
mGBA requires v1. Model metadata records `observation_version`. See
[observation v2](../observation-v2.md).

PPO policies remain frozen and select legal actions deterministically. A random
reference instead samples uniformly from legal actions using a per-battle seed.

Each run saves the exact loaded candidate bytes as `candidate.zip`. The original
path, SHA-256 and new `snapshot_path` appear in `config.json` under
`models.candidate`. No model reserialization or training is needed to preserve it.
When replaying a bundled suite, hash-verified `opponents/SHA256.zip` copies beside
the suite take precedence over its original opponent paths. The suite hash does
not change. See [periodic evaluation](../periodic-evaluation.md).

```bash
frozen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"
python mgba/evaluate_ppo.py --checkpoint latest \
  --opponent-checkpoint "$frozen_checkpoint" \
  --training-profile six_types --episodes 1000 --environments 48
```

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit. |
| `--checkpoint` | Required | Candidate checkpoint path or `latest`. |
| `--opponent-checkpoint`, `--player2-checkpoint` | Required unless using Random, rules or a pool | Reference checkpoint path or `latest`. The reference changes seats when side swaps are enabled. |
| `--random-opponent` | Off | Uniform random legal-action reference instead of a checkpoint. Mutually exclusive with `--opponent-checkpoint`; omit both when replaying a pooled suite. |
| `--rule-based-opponent` | Off | Showdown: deterministic `gen3-rules-v1`. Mutually exclusive with checkpoint/Random; cannot override a saved pool. Rule suites pin its fingerprint and select it automatically on replay. [Guide](../rule-based-opponent.md). |
| `--opponent-pool` | Off | Showdown: [pool JSON](../../config/opponents/README.md) with one or more checkpoints, rules and/or Random. Fixed weighted quotas preserve seat pairs; copies checkpoint bytes into the output. Exclusive with other opponent selectors; cannot replace a replay suite's pinned opponents. |
| `--rule-log`, `--no-rule-log` | Off | With rules: write bounded `rule-decisions.jsonl` scores/reasons using the shared log limits. |
| `--backend` | `showdown`, or replay suite value | `showdown` or `mgba`. |
| `--suite` | Not set | Replay a previous `suite.json`; mutually exclusive with team/state sources. A pooled suite also pins each opponent and its model hash. |
| `--training-profile` | Not set | Named directory under `config/training_profiles/NAME/teams`; Showdown only. |
| `--teams` | Not set | Two fixed team JSON files in game-player order; Showdown only. |
| `--team-dir` | Not set | Sample direct JSON children from a directory; Showdown only. |
| `--states` | Not set | Two fixed counter `.ssN` states; mGBA only. |
| `--state-dir` | Not set | Sample direct `.ssN` children; mGBA only. |
| `--team-profile` | Not set | Competitive, casual, random or mixed category catalog for the chosen backend. |
| `--team-size` | Not set | 1–6; requires `--team-profile`. |
| `--mirror-teams` | Off | Give both players the same team sampled from a random source. For a fixed mirror, pass the same file twice. |
| `--episodes` | `1000` | Total battle attempts; must be even when swapping sides. Fixed by a replay suite. |
| `--seed` | `10001` | Team and battle schedule seed. Fixed by a replay suite. Cartridge RNG is not reseeded. |
| `--swap-sides`, `--no-swap-sides` | On | Play each matchup with the candidate in both seats. Fixed by a replay suite. |
| `--max-steps` | `200` | Decision limit; unfinished battles truncate. Fixed by a replay suite. |
| `--environments` | `8` | Even number of players; two players per concurrent pair. |
| `--showdown-workers` | `thread` | `thread` or `process`; Node simulations are separate processes in both modes. |
| `--node` | `node` | Node.js executable. |
| `--device` | `cpu` | PyTorch inference device. |
| `--torch-threads` | `1` | Central PyTorch CPU threads. |
| `--reward-config` | `config/rewards/damage_knockout_win.json` | Reward points for reporting; does not change policies. Keep fixed across comparisons. |
| `--readable-log`, `--no-readable-log` | Off | Detailed observation/action log per pair. |
| `--battle-log`, `--no-battle-log` | Off | Spectator battle log per pair. |
| `--step-logs`, `--no-step-logs` | Showdown: off; mGBA: on | Per-turn Gym/protocol traces and episode team copies. Per-battle results and shared suites remain when disabled. Explicit `--no-step-logs` is rejected for mGBA. |
| `--log-max-mb` | `10` | Showdown size limit per rotating log in MiB; `0` means unlimited. |
| `--log-backups` | `2` | Rotated copies kept per log; `0` keeps only the current file. |
| `--keep-episode-logs` | `2` | Recent completed Showdown episode debug directories per pair, plus active/failed episodes; `0` disables them, `-1` keeps all. |
| `--output-dir` | Timestamped `.local/run/evaluation-…` | Must be a new directory. |
| `--command-timeout` | `120` | Seconds allowed for a paired request. |
| `--rom` | Required for mGBA | Supported FireRed v1.0 ROM path. |
| `--emulator` | `.local/mgba/bin/mgba-qt` | mGBA executable. |
| `--headless` | Off | Wrap mGBA with Xvfb. Showdown is already headless. |
| `--speed` | `1` | mGBA multiplier or `unlimited`; no effect on Showdown. |
| `--base-port` | `55455` | First mGBA adapter port; consecutive ports per player. |
| `--startup-timeout` | `30` | Seconds for mGBA adapter startup. |
| `--connect-timeout` | `180` | Seconds per link connection/start stage. |
| `--verbose` | Off | Print mGBA connection phases. |
| `--lobby-only` | Off | Inherited launcher option; rejected because evaluation requires battles. |

Choose exactly one team/state source or `--suite`. Replay rejects seed, episode,
side, max-step, mirror and team-size overrides. Checkpoint `later` is accepted as
an alias for `latest`; see [checkpoint selection](../checkpoints.md).

Exit status is 0 for a finished evaluation, 1 for a runtime failure, 2 for invalid
arguments, and 130 when interrupted. A finished evaluation can include truncations;
always inspect `summary.json` before interpreting win rates.

Logging limits do not remove the evaluation's `episodes.jsonl`, summary or saved
suite. Those remain available for statistical comparisons and campaign reports.
See [storage and retention](../storage.md).

## Random and pooled references

```bash
python mgba/evaluate_ppo.py --checkpoint latest --random-opponent \
  --training-profile six_types --episodes 1000 --environments 48
```

For [campaign evaluation pools](../campaign-evaluation-pools.md), pass the saved
`evaluation-pool/suite.json` with no reference flag. A pooled suite adds an
`opponents` object containing each policy's `kind`, `group`, `label`, and checkpoint
`path`/`sha256` where applicable. Each case records `opponent_id`, `opponent_group`
and `opponent_seed`; side-swapped cases share those values. The runner generates
this format automatically and checks checkpoint hashes when replaying it.

Pool summaries add `by_opponent_group` (Random, Dumb, Smart and combined Basic)
and `by_opponent` statistics. Their `win_rate` is wins / all attempted battles,
including truncations as non-wins. The existing top-level `candidate_win_rate`
uses completed battles only. Campaign promotion always uses the group rates.

## Optional decision diagnostics

`--decision-audit` (default off) requires a Showdown observation-v2 candidate.
It records bounded `candidate-decisions.jsonl` and an unrotated compact
`decision-audit.json` with per-episode counters and at most 40 early examples.
Uses `--log-max-mb` and `--log-backups`. It compares legal candidate choices with
the rule heuristic without changing either player's action. It adds analysis
overhead; omit for throughput comparisons. Use
[review_rule_losses.py](review-rule-losses.md) to select and verify a small replay
sample and produce a readable report.

`--move-order-audit` (default off) also requires a Showdown v2 candidate and is
mutually exclusive with `--decision-audit`. It records masked probabilities,
three inference-only cyclic move permutations per eligible position, and actual
move-announcement comparisons. Writes `move-order-audit.md`,
`move-order-audit.json`, and rotated `move-order-decisions.jsonl` using the shared
log limits. For sampled replay with verification use
[audit_move_order.py](audit-move-order.md).
