# Throughput benchmarking

[Benchmark flags](reference/benchmark-pool.md) · [Pool guide](link-pool.md) · [Training reports](battle-logs.md)

`benchmark_pool.py` measures complete mGBA battles with fixed command policies.
It includes emulator launch, cable connection, battle play, cleanup and the next
episode's startup. It does not update a learning model.

```bash
python mgba/benchmark_pool.py --rom "$PWD/FireRed.gba" --environments 8 \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1 \
  --commands MOVE1 --episodes 100 --speed unlimited --headless
```

Eight environments are four simultaneous pairs. `--episodes 100` schedules 100
battle attempts across the pool, not 100 per player. Supply two alternating states
or one per environment. Episodes must be at least the number of pairs.

## Command sequences

`--commands MOVE1` selects the first move for every player at every normal decision.
Supply multiple commands for a cycle: `--commands MOVE1 MOVE2`. Valid names are
`MOVE1`–`MOVE4` and `SWITCH1`–`SWITCH6`, with slots in original party order.
Repeating is on by default; `--no-repeat-commands` ends an unfinished episode when
its finite sequence runs out.

To give each side a different cycle, save a JSON file and pass its path as the
single `--commands` argument:

```json
{
  "name": "first_moves",
  "players": [
    {"cycle": ["MOVE1"]},
    {"cycle": [{"move": 1}]}
  ]
}
```

`players` contains two alternating entries or one per environment. A cycle accepts
command strings or `{"move": N}` / `{"switch": N}` objects. Repetition is a CLI
option here; this is not the [pool switch-test schema](link-commands.md).

Required replacements use the first legal surviving slot without consuming the
scripted cycle. Waiting for the partner is automatic. Other masked commands fail
with diagnostics instead of being silently changed. Battle outcomes depend on
team stats, move accuracy, effects and RNG; the benchmark records the observed
winner rather than requiring water to win.

## Reading throughput

After each episode the console updates two rates:

| Rate | Meaning |
|---|---|
| Episodes/hour | Completed battles per wall-clock hour across the whole pool |
| Player-episodes/hour | Twice that rate, counting both sides of each completed battle |

Truncations count as ended attempts but not completed battles. A failed run
retains completed results and diagnostics. `summary.json` also records winners,
reset time, battle time and elapsed time; `episodes.jsonl` records individual results.

Compare configurations with the same teams, commands, episode count and display
mode. Change environment count or speed, then compare the resulting pool rates.
`unlimited` removes throttling; adding more environments can still reduce efficiency
through CPU contention. The emulator remains CPU-based even if PPO uses CUDA.
For more stable comparisons, use enough completed episodes to reduce startup and
RNG variation, and compare turn/decision counts alongside battle rate.
