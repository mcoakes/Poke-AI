# Scripted switch sequences

[Pool guide](link-pool.md) · [Launcher flags](reference/launch-pool.md) · [Benchmark commands](benchmark.md#command-sequences)

The pool's `--commands` file runs coordinated switches after all pairs reach the
battle menus. Use it to check slot selection and cable turn coordination. For
move actions and repeated episodes, use the separate benchmark utility.

## Run the supplied stress test

```bash
python mgba/launch_pool.py --rom "$PWD/FireRed.gba" --environments 4 \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1 \
  --commands config/commands/switch_stress.json --speed unlimited
```

[The example](../config/commands/switch_stress.json) runs 24 × 5 = 120 paired
turns per match. Fire alternates slots 2 and 1; water cycles through 2, 3, 4, 5, 6.
Cycles continue across repeat boundaries. With the supplied unmodified teams:

| Side | Expected final active slot | Pokémon |
|---|---:|---|
| Fire | 1 | Charizard |
| Water | 6 | Gyarados |

The test inspects each result, including requested active slots and unchanged HP
and PP, then leaves the games open for inspection. Passive damage/healing effects
would change those assumptions; use the supplied plain switch-test teams.

```bash
python mgba/inspect_state.py --port 55355 --peer-port 55356
python mgba/inspect_state.py --port 55356 --peer-port 55355
```

## JSON format

```json
{
  "name": "switch_check",
  "repeats": 2,
  "turns_per_repeat": 5,
  "players": [
    {"name": "fire", "cycle": [{"switch": 2}, {"switch": 1}]},
    {"name": "water", "cycle": [{"switch": 2}, {"switch": 3}, {"switch": 4}, {"switch": 5}, {"switch": 6}]}
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | Optional label |
| `repeats` | Positive repeat count |
| `turns_per_repeat` | Positive turn count per repeat |
| `players` | Two alternating entries or one per environment, in launcher order |
| `players[].name` | Optional label; does not select a state by name |
| `players[].cycle` | Nonempty list of `{"switch": N}` commands, with 1-based original party slots |

Targets must exist and differ from the current active Pokémon. A requested test
that would overflow the native turn counter is rejected. This schema accepts
switches only and cannot be combined with `--lobby-only`.

Each match keeps `commands.jsonl`, initial inspections and a final verification
report in its run directory. A failure retains the inspected state and requested
command so the mismatch can be diagnosed.
