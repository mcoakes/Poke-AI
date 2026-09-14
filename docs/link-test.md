# Testing one cable pair

[mGBA setup](setup/mgba.md) · [Launch flags](reference/launch-link.md) · [Check flags](reference/test-link.md)

Start with two [generated counter states](automatic_states.md) and the patched
mGBA build. A native cable session contains both cores in one process.

## Automatic test

```bash
python mgba/launch_link.py --rom "$PWD/FireRed.gba" \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1 \
  --auto-connect --speed 2
```

The launcher handles dialogue and starts the battle. Add `--lobby-only` to stop
in the Colosseum, or `--headless` to hide windows through Xvfb.

From another terminal, check a battle that has started:

```bash
python mgba/test_link.py --ports 55355 55356 --require-battle --timeout 180
```

The checker polls connection evidence and writes JSONL results. It does not press
buttons or advance dialogue. A reachable socket alone is not evidence of a cable
connection: both players must report the corresponding link state and roles.

## Manual NPC test

Omit `--auto-connect` to control the two visible windows yourself. At each wired
Direct Corner counter, choose Colosseum and Single Battle, accept the save prompts,
and confirm the leader's connection when both players are present. After entering
the Colosseum, move both characters onto their battle pads. Use mGBA's configured
GBA button bindings to advance dialogue.

The source save must already have Oak's Parcel delivered and the service unlocked;
see [first save](setup/first-save.md). Select the wired counter, not a wireless
Union Room service.

`test_link.py` without `--require-battle` checks the connection stage. For an active
battle, use `--require-battle` as above. Ctrl+C in the launcher closes the session.
To retry, restart both cores together from the counter states.

The run directory contains `connection.jsonl`, `mgba.log` and each player's
`adapter.log`. If a stage stalls, collect those and the printed error before
changing inputs. See [troubleshooting](troubleshooting.md).
