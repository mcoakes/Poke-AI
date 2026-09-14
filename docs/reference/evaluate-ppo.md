# Evaluate PPO

[Utility index](README.md) · [Workflow and examples](../evaluation.md) · [Source](../../mgba/evaluate_ppo.py)

PPO policies remain frozen and select legal actions deterministically. A random
reference instead samples uniformly from legal actions using a per-battle seed.

```bash
fronzen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"
python mgba/evaluate_ppo.py --checkpoint latest \
  --opponent-checkpoint "$fronzen_checkpoint" \
  --training-profile six_types --episodes 1000 --environments 48
```

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit. |
| `--checkpoint` | Required | Candidate checkpoint path or `latest`. |
| `--opponent-checkpoint`, `--player2-checkpoint` | Required unless using Random or a saved pool | Reference checkpoint path or `latest`. The reference changes seats when side swaps are enabled. |
| `--random-opponent` | Off | Uniform random legal-action reference instead of a checkpoint. Mutually exclusive with `--opponent-checkpoint`; omit both when replaying a pooled suite. |
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
| `--step-logs`, `--no-step-logs` | On | Showdown per-turn Gym/protocol traces and per-episode team copies. Disable for compact evaluation; per-battle results and shared suites remain. Disabling is currently rejected for mGBA. |
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
