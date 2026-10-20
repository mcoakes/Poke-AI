# Reports and battle logs

[Training](ppo-training.md) · [Rewards](training-rewards.md) · [Storage and cleanup](storage.md) · [Troubleshooting](troubleshooting.md)

Each training run prints its output directory. Reports update as play progresses;
episode reward means appear after the first battle ends.

## Console metrics

| Metric | Meaning |
|---|---|
| `timesteps` | Learning-player transitions, including the checkpoint's prior count when resumed |
| `ep_rew_mean` | Mean unscaled episode return for learning players over the latest 100 ended battles |
| `p1_mean`, `p2_mean` | Separate player-side returns over that window, including a frozen opponent |
| `ep_len_mean` | Mean paired decision count per ended battle; replacements can add decisions within a turn |
| `battles/hour` | Completed battles divided by total run wall time, across the whole pool |
| `player-episodes/hour` | Twice battles/hour, since every completed battle has two participants |
| `p1_win`, `p2_win` | Each side's win percentage among completed battles in the latest 100 ended battles |
| `draw` | Draw percentage in the same completed-battle denominator |
| `win_n`, `trunc_n` | Completed and truncated counts in that window |
| `truncated` | Battle ended at a limit without a confirmed win, loss or draw |

Win rates count draws in the denominator and exclude truncations; before any
completed battle they display `n/a`. The two win percentages plus draw percentage
sum to 100% apart from rounding. They cover all pairs and both sides, even with a
frozen opponent. The window starts fresh when a training process starts, including
on resume. `summary.json` also contains cumulative `player1_win_rate_overall`,
`player2_win_rate_overall` and `draw_rate_overall`. JSON rates are fractions from
0 to 1; console rates are percentages. Rolling rates also appear in episode
records and PPO update metrics.

Episode means include truncated episodes. Throughput counts completed battles
only, while elapsed time includes startup, collection, resets, PPO updates and
waiting. Rates are cumulative, not the speed of just the latest battle. Divide
player-episodes/hour by the player count to estimate per-player throughput in a
balanced pool. With frozen player 2, only player 1 contributes to `ep_rew_mean`
and training timesteps; both participants still count toward throughput.

## Files

| Location within the run | Contents |
|---|---|
| `config.json` | Backend, teams/states, seed, rewards and model settings |
| `summary.json` | Current aggregate result and throughput |
| `episodes.jsonl` | One record per ended battle and its metrics |
| `updates.jsonl` | PPO update records |
| `ppo/progress.csv` | Optimizer metrics in CSV form |
| `checkpoints/latest.zip` | Most recently saved completed PPO update; may lag training when saves are spaced out |
| `checkpoints/ppo_*.zip` | Periodic numbered checkpoints; newest two retained by default |
| `checkpoints/final.zip` | Model at successful completion |
| `checkpoints/best.zip` | Strongest model from optional fixed-suite periodic validation |
| `evaluations.md`, `evaluations.csv` | Periodic validation scores, categories and best-model history |
| `pair-N/battle.log` | Rotating spectator transcript |
| `pair-N/battle-readable.log` | Rotating observations, action masks, choices and rewards |
| `pair-N/episode-NNNNNN/gym-steps.jsonl` | Structured per-decision reports; optional in Showdown |

Checkpoints save after updates; see [checkpoint timing](ppo-training.md#progress-and-saving).
When [periodic evaluation](periodic-evaluation.md) is enabled, training rates
include the validation pauses but count only training battles. Validation results
have their own history; final summaries record the session's validation wall time.
`time/checkpoint_seconds` and `time/checkpoint_fraction` in update records and
`ppo/progress.csv` show write overhead. The final `summary.json` includes a
`checkpointing` object with configured intervals, latest saved timesteps, total
write seconds and their fraction of total run time, including final saves.
The two text logs span episodes for their pair and are enabled by default during
training. In Showdown, each rotates at `--log-max-mb 10`, with
`--log-backups 2` retained copies.
Use `--no-battle-log` or `--no-readable-log` to disable either one. Evaluation
defaults to both off. Showdown per-decision traces are also off by default; enable
them with `--step-logs`. `--keep-episode-logs 2` retains recent completed episode
debug directories per pair, plus active or failed episodes. These runtime limits
apply to Showdown; mGBA keeps its separate diagnostics. Aggregate results and evaluation per-battle records remain
available with these detailed logs disabled.

After an attempt completes, compact campaign retention may remove its debug logs
and older training episode/update records. Its summaries, PPO progress CSV,
evaluation results and campaign history remain. See [storage and retention](storage.md)
before choosing how much history to preserve.

Compact campaign cleanup keeps selected completed candidates as `final.zip`; it
may remove the same attempt's redundant `latest.zip` and numbered backups. Recovery
checkpoints for the newest unfinished attempt remain protected.

Follow a live battle, substituting the run directory printed by training:

```bash
run_dir=".local/run/ppo-YOUR_RUN"
tail -n 80 -F "$run_dir/pair-1/battle.log"
```

Illustrative spectator output:

```text
Turn 1 · decision 1
  Foe BLASTOISE used HYDRO PUMP!
  P1 Charizard took 103 damage. (153 → 50/153 HP)
  HP  |  P1 Charizard 50/153  |  P2 Blastoise 154/154
  A critical hit!
  It's super effective!
```

The mGBA transcript reads game text and active HP records. The episode header
identifies whose view supplies “foe”; HP lines use launcher player numbers.
Poison, recoil, healing and HP costs also appear. Nearby messages supply context;
HP changes are not guessed from the policy's chosen action. Substitute damage
can produce text without reducing the Pokémon's HP. The sampler can miss messages
between frames, and reports gaps if its bounded event buffer overflows.

Showdown renders simulator protocol events into its spectator log. When step logs
are enabled, its raw `showdown-protocol.log` and `teams.json` are kept in retained
episode directories. Node diagnostics go to `pair-N/showdown-stderr.log`. These
logs use the same file rotation settings. Oversized records are replaced by a
marked text tail or JSON omission marker rather than exceeding the size limit;
the [storage guide](storage.md) explains the tradeoff.

For mGBA, each episode also has `connection.jsonl`, `mgba.log` and
`playerN/adapter.log` for connection, process and Lua diagnostics. Preserve these
alongside the failing `gym-steps.jsonl` when reporting an error. A failure is
recorded as a failure, without manufacturing a terminal reward.

## Non-overlapping comparisons

Training prints a separate summary every 500 ended battles. See
[500-battle block comparisons](battle-blocks.md) to review those results or analyze
an existing run without restarting it.
