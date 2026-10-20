# Battle Studio: build a team and play

Human play shows only the human player's mGBA window. The AI core runs with its
window hidden inside the same native link session. This also follows
`--human-player 2` when using the CLI. Apply the current patches and rebuild once:

```bash
python tools/setup_mgba_link.py
cmake --build build/mgba --parallel
cmake --install build/mgba
```

[Human vs AI](human-vs-ai.md) · [Every GUI flag](reference/play-gui.md) · [Dependencies](setup/README.md)

Battle Studio is a local browser interface for choosing both parties and a trained
PPO opponent. It generates the saves, verifies counter states, and starts the
link battle automatically. You play in the **Player 1 mGBA window**; the browser
handles team editing, progress, and the match result.

## Start the interface

Complete [mGBA setup](setup/mgba.md), [save-editor setup](setup/save-editor.md),
and the [first-save instructions](setup/first-save.md). Use a battery `.sav` with
Oak's Parcel delivered and the Direct Corner unlocked. Close any emulator using
that source file before starting. You do not need prebuilt team states.

Run from your Linux desktop with the project virtual environment active:

```bash
python -m pip install -e '.[train]'
python mgba/play_gui.py --rom "$PWD/FireRed.gba" \
  --source "$PWD/FireRed.sav" \
  --workers 2 --speed 1
```

Put trained checkpoint ZIPs in `models/`. The browser opens at
`http://127.0.0.1:8080`. If needed, open that address manually. The GUI uses the
existing Python dependencies and bundled images; no frontend build or Node.js
installation is needed for this workflow. Both Showdown-trained and mGBA-trained
models use the existing human-play inference interface.

V2 checkpoints are supported with adapter 0.19.0 on both linked cores. Restart
this server and its emulator sessions after updating; select the model normally.
See [native v2 verification](mgba-observation-v2.md). No additional CMake build
is required for the v2 Python/Lua update.

Without `--model-list`, the model picker includes every checkpoint ZIP in `models/` and its
subfolders. **Smart v10** is the default when available: `smart_v10.zip` and
`smartv10.zip` are recognized regardless of case, spaces, or hyphens. If several
copies exist, the closest to `models/` wins, then alphabetical path order. If it
is absent, the first available model is selected.

The picker also lists up to 200 recent ZIPs from `.local/run`; use
`--checkpoint-dir checkpoints` to change that additional search folder.
`--checkpoint models/smart_v11.zip` or `--checkpoint latest` overrides the
default. **Refresh** updates the model list while keeping your current selection
when it still exists. The selected model is copied at match confirmation and
stays fixed throughout the battle.

## Classroom: two opponents and a clear default

For the October 20 student session, use this small menu:

| Student-facing name | Saved model | What students can explore |
|---|---|---|
| **Coached AI (default)** | Teacher-training validation-selected model | Practised decisions from the Rules bot, then learned through battles |
| **Rewards-only AI** | Teacher-training PPO-only best | Continued learning through battle rewards without that teacher practice |

These two shared the starting model, architecture, teams and allotted PPO
training budget. Their selected checkpoints occur at different timesteps.
The coached model is the recommended default: it scored 51.37% validation,
46.50% then-unseen-team wins and 93.7% fixed-mirror wins against Rules.
Rewards-only scored 47.41%, 44.80% and 62.7%, respectively. The broader advantage
is modest and uncertain; these are two learning approaches, not guaranteed
hard/easy difficulty levels. Neither uses live Rules assistance during play.

Keep the 400k reference and fixed-team specialist as instructor examples outside
this menu. The 400k comparison changes several factors at once, and the specialist's
high score applies to its narrow fixed matchup. Do not list every seed, best/final
snapshot or intermediate checkpoint for students.

Save the two classroom copies from the completed experiment:

```bash
mkdir -p models/classroom
cp -n .local/run/teacher-training-01/models/selected.zip \
  models/classroom/coached-ai.zip
cp -n .local/run/teacher-training-01/models/ppo-only-best.zip \
  models/classroom/rewards-only-ai.zip
```

Launch the restricted menu:

```bash
python mgba/play_gui.py \
  --rom "$PWD/FireRed.gba" --source "$PWD/FireRed.sav" \
  --model-list config/play/classroom.json \
  --workers 2 --speed 1
```

Open `http://127.0.0.1:8080`. The only choices are the two above, with Coached AI
selected on page load. **Refresh** keeps a student's current choice when it
remains listed. This mode does not scan other model folders or offer `latest`,
so a background training checkpoint cannot unexpectedly enter the menu.

`config/play/classroom.json` sets the names, paths and default. Relative model
paths resolve from the project root. Each entry needs a unique `id`, a `label`
and an existing checkpoint ZIP; `default` must identify one of those entries.
Missing models or invalid lists produce an explicit error. Do not combine
`--model-list` with `--checkpoint`; `--checkpoint-dir` has no effect in list mode.
Without `--model-list`, the usual research-oriented discovery still works.

Start with familiar three-Pokémon teams and leave **Vary AI choices** off.
Play the same matchup against both models and ask what changed in move choices,
voluntary switches and replacements. Human games are illustrative: these models
were trained on a restricted 12-species, two-damaging-move task, not arbitrary
competitive six-Pokémon teams. The current training run can continue independently.

To use a later winner, first save it under a new permanent filename, update the
appropriate entry in this list, and refresh the menu. Keep the current copies
as fallbacks. In-progress training results do not automatically replace them.

The classroom menu passed 49 focused play-service tests, including the default,
restricted model list, invalid configuration, and rejecting unlisted models at
match launch. Frontend JavaScript syntax also passed. The new menu was not used
to run a ROM battle here; use the launch command above for the local play check.

## Choose each party

Enter **Your trainer name** above the parties to change your in-game name.
Use up to seven letters, digits, or spaces; leave it blank to keep the source
save's name. Only the human player's generated save is renamed. The AI retains
its trainer name, and the source save remains unchanged.

Each side has independent controls:

| Mode | How it works |
|---|---|
| **Saved teams** | Choose a directory category, optionally search by team name, filename, or Pokémon, then select a team and click **Use team**. |
| **Build a team** | Click an empty slot, search the pictured Pokédex, and select a Pokémon. |
| **Random team** | Select Competitive, Casual, Completely random, or Mixed categories; choose a size from 1–6 and click **Roll team**. |

**Saved teams** scans all of `config/` recursively. Each team's parent directory
becomes its category, relative to `config/`. Full paths distinguish nested
folders with the same name; files directly in `config/` go under **Misc**.

| Team file | Category |
|---|---|
| `config/my_team.json` | Misc |
| `config/teams/fire_vs_water.json` | teams |
| `config/teams/profiles/competitive/3/competitive_3_001.json` | teams/profiles/competitive/3 |
| `config/training_profiles/six_types/teams/six_types.json` | training_profiles/six_types/teams |

Only usable team JSONs with a `party` of one to six Pokémon appear. Reward,
campaign, command, and other settings files are excluded, as are malformed teams.
Generator metadata is accepted; older move spellings such as `IceBeam` are
normalized to `Ice Beam`. Existing one-to-three-move sets keep their empty slots.
After adding or editing a team file, click **Refresh teams** or reload the page.
No category registration or server restart is needed. Loading or editing a team
changes the preview; its source JSON remains unchanged.

Random teams are sampled from `config/teams/profiles/CATEGORY/SIZE/`. Mixed
chooses one of the three categories with equal probability, then a template in
that category. Changing category or size takes effect on the next roll. The
previewed party is the one used at confirmation; it is not rolled again.

Click a populated slot to edit it. The editor supports:

- Level 1–100, defaulting to **50** when adding a Pokémon.
- One to four distinct Gen III moves; new suggestions fill four slots. Leave
  trailing slots blank to preserve a smaller moveset. The default suggestion list uses the species'
  move pool; **Include all Gen III moves** exposes experimental combinations.
- Nature and held item, including **None**.
- Six IVs, each 0–31; new Pokémon start at 31. **Max IVs** restores them to 31.
- Six EVs, each 0–255, with a combined maximum of 510; new Pokémon start at zero.
- **Use suggested set** for species with a curated moveset, EV spread, nature,
  and held item. It preserves the level and IVs you entered.
- **Make lead**, or removal of a party member. The first slot leads the battle.

Editing a saved or random party switches that side to custom mode. All parties must have
one to six Pokémon with at least one move each. Competitive templates are curated
styles, not a guarantee of tournament legality. Experimental sets use the same
save editor as the existing JSON pipeline.

## Confirm and play

1. Select the AI opponent. Under **Match settings**, choose battle speed,
   save workers, and whether the AI samples its actions.
2. Click **Review matchup** to check both parties and the selected model.
3. Click **Generate saves & start battle**. The source template is prepared once,
   including eight badges and the counter warp. Workers generate and capture
   the two parties in parallel when enabled.
4. Wait for **Your move**. Focus the Player 1 mGBA window and use your normal
   keyboard/gamepad controls. Advance your dialogue and select replacements;
   the AI handles its own side.
5. Read the browser's result, then choose **Build another matchup** to play again.

Generation runs in Xvfb at `--capture-speed unlimited` by default. Battle speed
defaults to 1× for human play. NPC dialogue, cable connection and walking to the
pads run at `--setup-speed unlimited`; both cores acknowledge the switch to
battle speed before READY. These are separate from capture speed. The effective
worker count cannot exceed two because each match generates two teams. Use one
worker if you prefer serial generation while training runs elsewhere.

Install the runtime-speed patch once after extracting this update:

```bash
python tools/setup_mgba_link.py
cmake --build build/mgba --parallel
cmake --install build/mgba
```

Restart the GUI launcher after installing. A missing speed patch produces a clear
error before NPC automation starts. Full [CMake setup](setup/mgba.md) is documented
for first installations.

**Stop match** stops this job and its emulator processes. Ctrl+C in the launcher
terminal stops both the GUI server and its match. Closing the browser tab alone
does not stop the server; reopening the local URL restores the current progress.
One match can run through this GUI at a time.

## Files and troubleshooting

Each match gets a new `.local/run/play-ui-<timestamp>-<id>/` directory:

| Path inside the match | Contents |
|---|---|
| `player1.json`, `player2.json` | The exact selected party specifications |
| `agent.zip`, `config.json` | Frozen checkpoint and launch settings |
| `states/generated/playerN.sav` | Generated battery saves |
| `states/counter/playerN.ss1` and `.ss1.json` | Verified counter states and metadata |
| `generation.log` | Save editor and capture output |
| `battle-launch.log` | Link setup and human-play progress |
| `battle/result.json` | Outcome and decision count |
| `job.json` | Final GUI job status |

The browser's **Generation and battle details** expands the live log tails.
A generation or verification failure stops before battle launch. Fix the
reported cause, then build another matchup. Capture diagnostics also remain
under `.local/capture/build-*/`.

The GUI binds only to localhost. Its default adapter ports are **55365/55366**,
separate from the default training pair; if those ports are already in use,
choose another pair with `--ports`. `--port` changes the browser's HTTP port.
If no windows appear, check that the launcher is running in your desktop session,
the mGBA autoload/link patches are installed, and the source save has Direct
Corner access. See [troubleshooting](troubleshooting.md).

## Suggested sets

Every species has a suggested set. Edit `config/generation/competitive_gen3.json`
for the 48 curated sets, or `config/generation/suggested_gen3.json` for the
338 fallback suggestions. Curated entries take priority. The fallback sets are
automatic starting points, not competitive recommendations. Move and learnset
data lives in `config/generation/firered_us_v1_0.json`. Restart the GUI server
after editing these files.

Species with fewer than four natural moves have explicitly labeled experimental
sets with added moves. The editor allows you to customize all four slots.
