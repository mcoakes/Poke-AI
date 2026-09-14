# Inspecting emulator state

[Documentation](README.md) · [Inspector flags](reference/inspect-state.md) · [Policy observation](turn-environment.md)

The inspector reads a running mGBA adapter without controlling the game. It shows
parties, species, HP, moves and PP, types, held items, bag contents, active slots,
menus and battle outcomes. Full opponent information is evaluator data; the
policy's exact subset is documented in the observation guide.

## Inspect one emulator or a linked pair

```bash
python mgba/inspect_state.py --port 55355
```

For a linked battle, supply the partner port. This combines authoritative battle
simulation with the local player's menu and party view:

```bash
python mgba/inspect_state.py --port 55356 --peer-port 55355
python mgba/inspect_state.py --port 55356 --peer-port 55355 --json
```

Do not assume battler 0 is always your player in a link battle. The paired view
resolves roles and original party slots. Party slots printed for people are
1-based; raw menu cursors can be 0-based.

| Output | Interpretation |
|---|---|
| Party types | Base species types; active battle types can differ |
| Active records | Current battler HP, moves, PP, types and status overlays |
| `counts_valid: false` | Count is unknown/unreliable; zero is not a verified knockout |
| `busy_or_unknown` or `?` | No recognized selection at that snapshot; common during dialogue and animation |
| `terminal: true` | Validated battle result rather than an assumption from an unreadable party |
| Bag pocket/selection | Item list and current menu details when applicable |

An in-game status such as poison differs from a fainted count: remaining Pokémon
are valid, non-Egg party members with positive HP. Party RAM can lag active HP
during animations. Capture both JSON inspections if a stable selection menu
appears wrong; a transcript of displayed text alone can omit the relevant state.

Showdown has no mGBA inspection port. Its corresponding snapshots are written
in [training logs](battle-logs.md).

## Connectivity and raw inputs

For one independent visual emulator:

```bash
python mgba/launch_emulators.py --rom FireRed.gba --ports 55355 --speed 1
```

In another terminal:

```bash
python mgba/test_adapter.py --port 55355
python mgba/test_inputs.py --ports 55355 --actions A --frames 1
```

For two adapters, each action maps to the port in the same position:

```bash
python mgba/test_inputs.py --ports 55355 55356 --actions DOWN UP --frames 1
```

That sends Down to 55355 and Up to 55356. It is not a two-press sequence. One
frame may be too short to see an animation; the script checks held-input duration.
Raw requests allow 1–600 frames and release the key afterward. They do not freeze
free-running emulation between requests.

| Raw ID | Button |
|---:|---|
| 0 | NOOP |
| 1, 2 | A, B |
| 3, 4, 5, 6 | Up, Down, Left, Right |
| 7, 8 | Start, Select |
| 9, 10 | L, R |

Use the [pool](link-pool.md) or [pair launcher](link-test.md) for native cable
battles. Independent processes launched by `launch_emulators.py` do not connect
through a cable simply because their adapter ports are adjacent.
