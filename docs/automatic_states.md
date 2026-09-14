# Build saves and counter states

[Quickstart with example JSONs](quickstart.md#3-build-and-customize-teams) · [Documentation](README.md) · [All builder flags](reference/build-states.md) · [Initial save](setup/first-save.md)

`build_states.py` prepares a template, writes each requested party, boots its save
and captures a verified state at the wired-link counter. The source must be an
existing FireRed battery save with Oak's Parcel delivered and Direct Corner
service available. Complete [mGBA](setup/mgba.md) and [save-editor setup](setup/save-editor.md) first.

## Build the example teams

Close any emulator using the source battery save, then run:

```bash
python mgba/build_states.py --source states/base/overworld.sav --rom FireRed.gba \
  --teams config/teams/fire_vs_water.json config/teams/water_vs_fire.json \
  --headless --speed unlimited
```

For the three-starter teams:

```bash
python mgba/build_states.py --source states/base/overworld.sav --rom FireRed.gba \
  --teams config/teams/starters_charizard_lead.json config/teams/starters_venusaur_lead.json \
  --headless --speed unlimited
```

Replace the source path with your actual parcel-complete `.sav`. The builder does
not create that initial input. Showdown uses these JSONs directly and skips this
entire save-generation step.

## What the pipeline does

1. Verify the ROM identity and source battery-save format.
2. Prepare a copy with eight badges, a pending Direct Corner warp and adventure-recap suppression.
3. Create the party with PKHeX.Core and verify serialized stats, moves and items.
4. Start an isolated mGBA instance, select Continue and wait for the counter position.
5. Clear input, capture a state including battery data, reload it twice and verify readiness, frame and RAM fingerprints.
6. Publish each completed team's outputs as soon as verification succeeds.

Headless capture is the default. Use `--visible` to watch it. Captures run
sequentially by default; add `--workers 2` to generate and capture two teams at a
time. The source template and .NET editor are prepared once before the workers
start. Worker count is capped at the number of teams still to build. `--timeout`
is the wall-clock limit per emulator and an emulated-frame limit also applies.
`--speed` controls emulator pacing.

Use `--output-dir PATH` to put `generated/` and `counter/` under another root.
The [browser team builder](play-gui.md) uses this to isolate every match's saves.

## Outputs and batch resume

| File | Contents |
|---|---|
| `states/generated/TEAM.sav` | Generated battery save with a pending Continue warp |
| `states/counter/TEAM.ss1` | Snapshot already positioned and facing the counter |
| `states/counter/TEAM.ss1.json` | Input/output hashes, verified party and restoration results |
| `.local/capture/build-*/TEAM/` | Capture progress, emulator log and diagnostics |

Profile outputs preserve `profiles/CATEGORY/SIZE/` beneath the generated and
counter directories. The source save and ROM are not overwritten. Existing
outputs are refused by default. `--force` backs up and replaces them;
`--skip-existing` verifies complete matching outputs and skips them. These flags
are mutually exclusive. Successful teams remain available after an interruption.

```bash
python mgba/build_states.py --source states/base/overworld.sav --rom FireRed.gba \
  --team-profile mixed --team-size 3 --headless --speed unlimited --skip-existing
```

## Inspect and check resets

```bash
python mgba/launch_emulators.py --rom FireRed.gba --ports 55355 55356 \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1
```

In another activated terminal:

```bash
python mgba/inspect_state.py --port 55355
python mgba/test_reset.py --ports 55355 55356 --repeats 3
```

Fingerprints must match across repeated resets within each emulator. They do not
need to match between different teams. These checks apply to independent cores;
a live native cable pair must be restarted together rather than restoring one
core. Continue with [linked pools](link-pool.md) or [PPO training](ppo-training.md).
