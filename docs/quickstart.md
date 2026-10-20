# Quickstart: from installation to a playable agent

[Project overview](../README.md) · [All utility flags](reference/README.md) · [Troubleshooting](troubleshooting.md)

Follow this walkthrough from the project root with your Python virtual environment
active. It uses two included three-Pokémon teams, saves a PPO checkpoint, plays it
in FireRed, then starts a campaign with a single evaluation history.

1. [Install dependencies](#1-install-dependencies).
2. [Set up the initial save](#2-set-up-the-initial-save-file).
3. [Build and customize teams](#3-build-and-customize-teams).
4. [Train with Showdown or mGBA](#4-train-an-agent), including [pause and continue](#monitor-pause-and-continue-training).
5. [Play against the agent](#5-play-against-the-agent).
6. [Run a campaign](#6-run-a-campaign).
7. [View campaign results](#7-view-campaign-results).

**Simulator-only route:** complete installation steps 1–2, skip the save-file
section and the mGBA-only parts below, then use Showdown training and campaigns.
**Full route:** complete all four installation steps and follow this page in order.

## 1. Install dependencies

Open [the installation guide](setup/README.md). It contains all copy-and-paste
commands in this order:

1. **Python:** create `.venv` and install `.[train]`. This installs Gymnasium,
   NumPy, PyTorch, Stable-Baselines3 and SB3-Contrib.
2. **Node.js + Showdown:** install Node 24 and run the pinned npm install. This
   provides local simulated battles for training and campaigns.
3. **mGBA:** install compiler/Qt/Lua/Xvfb packages, clone the pinned source,
   apply the patches, and run CMake configure/build/install.
4. **.NET + PKHeX.Core:** install the .NET 10 SDK and build the save editor.

For this full walkthrough you also need your own supported **FireRed USA/Europe
v1.0 ROM**, a Linux desktop and a browser. ROMs, saves and trained models are not
included. The two JSON team examples are included.

After installation, run these checks from the project root:

```bash
source .venv/bin/activate
python mgba/train_ppo.py --help
node --version
./.local/mgba/bin/mgba-qt --version
./.local/dotnet/dotnet --version
```

Expected: the trainer's help, Node `v24.x`, mGBA `0.10.5`, and .NET `10.0.xxx`.
If you chose a system SDK, use `dotnet --version` for the last check. A Showdown-only
installation only needs the first three lines. Run the activation line again in
every new terminal.

## 2. Set up the initial save file

This section is needed for mGBA training and human play. Put your own ROM in the
project root as `FireRed.gba`, then check it:

```bash
sha1sum FireRed.gba
```

Expected SHA-1: `41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc`.

Start the game:

```bash
mkdir -p states/base states/counter
./.local/mgba/bin/mgba-qt FireRed.gba
```

1. Receive your starter, finish the first rival battle and reach the overworld.
2. Use the **game's SAVE menu**. Also create an emulator snapshot using mGBA's
   save-state controls; keep it as `states/base/overworld.ss1` for recovery.
3. Go to the Viridian Poké Mart, collect **Oak's Parcel**, return to Professor
   Oak in Pallet Town and finish the delivery dialogue.
4. Visit the upstairs Pokémon Center **Direct Corner**: the far-right attendant
   for the wired link service. Confirm the service is offered, then leave the
   dialogue. If it is unavailable, finish the parcel/Pokédex conversation first.
5. Use the game's SAVE menu again, then **close mGBA** before copying its save.

With the default save location, preserve the game's battery save:

```bash
cp --backup=numbered FireRed.sav states/base/overworld.sav
stat -c '%n: %s bytes' states/base/overworld.sav
```

Expected size: **131072 bytes**. If `FireRed.sav` is missing, check mGBA's configured
save directory and copy that file instead. Do not rename an emulator snapshot
into a `.sav`.

- **`FireRed.sav`:** input to the save editor; must contain parcel
  delivery. It does not need to contain your final party or all badges.
- **`states/base/overworld.ss1`:** early recovery snapshot; it is not the editor input.

The generator will grant eight badges, create each party and place it at the
link counter. It does not complete Oak's Parcel for you. You only need to prepare
this base save once. More detail: [first-save guide](setup/first-save.md).

## 3. Build and customize teams

Two complete JSON files are ready to use:

- [quickstart_charizard.json](../config/teams/quickstart/quickstart_charizard.json):
  Charizard leads, followed by Blastoise and Venusaur.
- [quickstart_venusaur.json](../config/teams/quickstart/quickstart_venusaur.json):
  Venusaur leads, followed by Blastoise and Charizard.

Both have level-50 Pokémon, four moves each, held items, all IVs at 31 and a
510-total EV spread. They contain the same sets in different party orders.
Read or edit them in your preferred text editor:

```bash
cat config/teams/quickstart/quickstart_charizard.json
```

A Pokémon record looks like this; the complete files wrap their records in a
`party` list. The first record is the lead, and move order determines action slots.

```json
{
  "species": "Blastoise",
  "level": 50,
  "moves": ["Surf", "Ice Beam", "Rapid Spin", "Toxic"],
  "nature": "Bold",
  "held_item": "Leftovers",
  "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
  "evs": {"hp": 252, "def": 252, "spd": 6}
}
```

The [team format guide](team-generation.md#json-format) explains every field.
Each IV is 0–31; EVs total at most 510 across all six stats. Maxing every EV to
255 is not a valid spread.

### Generate FireRed saves and counter states

**For mGBA:** build both JSONs in one command. The two workers generate their
states headlessly at unlimited speed:

```bash
python mgba/build_states.py \
  --source FireRed.sav \
  --rom "$PWD/FireRed.gba" \
  --teams config/teams/quickstart/quickstart_charizard.json \
          config/teams/quickstart/quickstart_venusaur.json \
  --workers 2 --headless --speed unlimited --skip-existing
```

The command prepares the template, grants eight badges, edits the party, boots
Continue, positions the player at the wired counter and verifies the captured
snapshot. You do not need a separate `prepare_template.py` call or a manual
counter snapshot for each team.

For each filename, expect these outputs:

- `states/generated/quickstart_charizard.sav` — edited battery save.
- `states/counter/quickstart_charizard.ss1` — ready-to-launch counter state.
- `states/counter/quickstart_charizard.ss1.json` — verification metadata.

The Venusaur team produces the same three extensions with `quickstart_venusaur`.
Keep these files together. The launchers load the counter states directly, so
there is no need to overwrite `FireRed.sav`. `--skip-existing` resumes an unchanged
batch. After editing a JSON, replace it with `--force` to back up and rebuild outputs.

**For Showdown:** the JSON files are already usable; skip this build command.

### Try more teams

The project also includes 150 templates in each category/size directory:
`config/teams/profiles/{competitive,casual,random}/{1,2,3,4,5,6}/`.
Showdown can sample these immediately with `--team-profile mixed --team-size 3`.
For custom batches, see [team generation](team-generation.md); for building their
mGBA states, see [the state builder](automatic_states.md).

## 4. Train an agent

### Start with Showdown

Showdown handles battle decisions directly and avoids emulator frames, NPC
connection dialogue and save-state resets. It is the faster starting point for
PPO experiments, with separate local workers for concurrent battles. It uses no
public server or account. Checkpoints use the same observation/action format as
mGBA; cartridge behavior still needs validation in mGBA.

Start a small training run with the example pair:

```bash
showdown_run=".local/run/quickstart-showdown-01"
python mgba/train_ppo.py \
  --backend showdown \
  --environments 4 --showdown-workers process \
  --teams config/teams/quickstart/quickstart_charizard.json \
          config/teams/quickstart/quickstart_venusaur.json \
  --rollout-steps 128 --batch-size 128 \
  --checkpoint-every 1024 --total-timesteps 10000 \
  --output-dir "$showdown_run"
```

Four environments mean **two simultaneous battles**, with one shared learning
policy controlling all four players. `128` rollout steps per player collects
512 transitions per PPO update; minibatches contain 128 transitions. The timestep
budget rounds up to finish an update. Showdown is always headless and unthrottled;
it needs neither `--headless` nor `--speed`.

Expect episode rewards, player win rates and battles/hour after battles finish,
plus a message when each PPO update saves `latest.zip`. On successful completion,
`final.zip` is saved. This short run establishes the pipeline; it does not certify
a strong agent. Use a new output-directory name if `quickstart-showdown-01` exists.

After successful completion, preserve the final agent and the first numbered
checkpoint for the campaign example below:

```bash
mkdir -p models
cp --backup=numbered "$showdown_run/checkpoints/final.zip" models/quickstart_agent.zip
cp --backup=numbered "$showdown_run/checkpoints/ppo_000001024.zip" models/quickstart_reference.zip
```

To train on random three-Pokémon teams in a subsequent run:

```bash
python mgba/train_ppo.py \
  --backend showdown --environments 4 --showdown-workers process \
  --team-profile mixed --team-size 3 \
  --resume models/quickstart_agent.zip \
  --rollout-steps 128 --batch-size 128 --total-timesteps 100000
```

Each player independently samples a team for each battle. This creates a new
run directory printed at startup; it does not overwrite the named model copy.
See [training settings](ppo-training.md) and [Showdown mechanics](showdown-training.md).

### Train or fine-tune in mGBA

Once the two counter states exist, use the same trained policy in real FireRed:

```bash
mgba_run=".local/run/quickstart-mgba-01"
python mgba/train_ppo.py \
  --backend mgba --rom "$PWD/FireRed.gba" \
  --environments 2 \
  --states states/counter/quickstart_charizard.ss1 \
           states/counter/quickstart_venusaur.ss1 \
  --resume models/quickstart_agent.zip \
  --speed unlimited --headless \
  --rollout-steps 128 --batch-size 128 --total-timesteps 2048 \
  --output-dir "$mgba_run"
```

Two environments mean one native link battle. The launcher connects the games,
starts the battle and resets between episodes automatically. Remove `--headless`
to watch. Use a new output-directory name for a second run.

For mGBA-only training from scratch, omit `--resume`. After a successful mGBA run,
you can use its `checkpoints/final.zip` for play, or update your named agent:

```bash
cp --backup=numbered "$mgba_run/checkpoints/final.zip" models/quickstart_agent.zip
```

Both engines simulate battles on the CPU. Start with the small pool above and
measure throughput before adding environments. [All training flags](reference/train-ppo.md).

### Monitor, pause, and continue training

You can stop a standalone PPO run and continue it later. In the **training
terminal**, press **Ctrl+C once**, then wait for cleanup and the shell prompt
before closing the terminal or restarting the run. The interruption message
reports the last saved checkpoint. Progress since that save is not preserved;
Ctrl+C does not save an unfinished rollout or update. If you stop before the
first checkpoint is saved, there is no trained checkpoint to recover.

To continue the Showdown example, run this from the project root with the virtual
environment active (set the path again in a new terminal):

```bash
training_run=".local/run/quickstart-showdown-01"
python mgba/train_ppo.py \
  --output-dir "$training_run" \
  --if-exists continue \
  --total-timesteps 10000
```

This requests **10,000 additional transitions**, rounded up to a full rollout;
it does not automatically finish the original remaining budget. Choose the new
budget you want. Use your actual run directory, such as
`.local/run/quickstart-mgba-01`, to continue a different standalone run.

Continuation restores the saved settings and latest recoverable checkpoint,
including weights, optimizer state, counters, logging and periodic validation
settings. It starts fresh battles; it does not resume a battle mid-turn. No
`--resume` flag is needed. `best.zip` is for selecting a validated model and may
be older than the checkpoint used to continue training. Keep the run directory
and its saved inputs. This also works after a completed run when you want more
training. See [continuation details](ppo-training.md#continue-an-existing-run).

**For runs configured with periodic evaluation**, open another terminal and
monitor the validation history:

```bash
training_run=".local/run/YOUR_RUN"
watch -n 5 "cat '$training_run/evaluations.md'"
```

Replace `YOUR_RUN` with the actual directory. The table appears after a completed
evaluation and refreshes every five seconds. The basic quickstart training
commands above do not enable periodic evaluation, so they do not produce this
table; monitor their training terminal instead. See
[periodic evaluation setup](periodic-evaluation.md) to enable validation.
Use `watch` because the table is replaced atomically. **Ctrl+C in the watch
terminal only stops the viewer**; Ctrl+C in the training terminal stops training.

For campaigns, use the [campaign continuation command](#6-run-a-campaign)
instead of continuing an individual attempt with `train_ppo.py`.

## 5. Play against the agent

Both play methods run a real FireRed link battle. Only the **human's emulator
window is shown**; the AI window is hidden within the same linked process.
Use a terminal in your Linux desktop session. The model is fixed during play.

### Browser GUI: Battle Studio

The GUI builds saves from your selected parties, so it only needs the base save
and a trained checkpoint; the states built in step 3 are not required here.

```bash
python mgba/play_gui.py \
  --rom "$PWD/FireRed.gba" \
  --source FireRed.sav \
  --checkpoint models/quickstart_agent.zip \
  --workers 2 --capture-speed unlimited \
  --setup-speed unlimited --speed 1
```

Open [Battle Studio at localhost:8080](http://127.0.0.1:8080) if your browser does not open automatically.

1. For each side, choose **Saved teams**, select `teams/quickstart`, and click
   **Use team**. You can also choose **Random team** and a category/size, or **Build a team**.
2. Click a party member to edit it, or an empty slot to choose a Pokémon by picture/name. Edit moves, level,
   IVs, EVs and held item. Level defaults to 50.
3. Select the checkpoint and review both parties.
4. Click **Generate saves & start battle**. Generation and link setup run at
   unlimited speed; battle play switches to 1×.
5. Wait for **Your move**, focus the human mGBA window and use its normal controls.

Saved teams are discovered throughout `config/`, grouped by their parent directory;
root-level files appear under **Misc**. **Refresh teams** picks up new JSONs.
The model picker includes `models/` and its subfolders. Omit `--checkpoint` to
prefer `models/smart_v10.zip` (also `smartv10.zip`) when available; the explicit
quickstart checkpoint above takes priority. See [Battle Studio](play-gui.md)
for the full selection workflow.

The browser builds teams and manages the match; actual gameplay is in the mGBA
window. Set your keyboard/gamepad bindings in mGBA if needed. **Stop match** stops
the battle; Ctrl+C in the launcher terminal stops the server too. Close this
launcher before using the same ports for CLI play. [GUI guide](play-gui.md).

### CLI: launch with the existing counter states

```bash
python mgba/play_agent.py \
  --rom "$PWD/FireRed.gba" \
  --checkpoint models/quickstart_agent.zip \
  --states states/counter/quickstart_charizard.ss1 \
           states/counter/quickstart_venusaur.ss1 \
  --human-player 1 --ports 55365 55366 \
  --setup-speed unlimited --speed 1
```

Wait for `READY`, then play in the visible window. You control Charizard's team;
the agent controls Venusaur's team. `--human-player 2` reverses the roles while
keeping the state order unchanged. The terminal starts and reports the match;
it does not ask you to enter move numbers. At the end, press Enter to close the
match or use Ctrl+C. [CLI guide and flags](reference/play-agent.md).

## 6. Run a campaign

Campaigns use **Showdown** and cumulative team pools. This example selects the
[generalization campaign](generalization-campaign.md): stage 1 has 24 training
teams, and every later stage adds another directory of 24 teams. Promotion uses
separate validation teams. Switching overrides are disabled; the policy can
choose normal switch actions. These teams are independent of the starter pair above.

The example below starts from your final quickstart model and freezes the earlier
1,024-step model as the first opponent. Earlier does not guarantee weaker; replace
these two paths with your chosen candidate and evaluated reference when running
a serious campaign. For your saved Smart v1 and Dumb models, see
[the named-model walkthrough](battle-count-campaigns.md).

```bash
curriculum_run=".local/run/quickstart-generalization-01"
frozen_checkpoint="models/quickstart_reference.zip"

python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --curriculum config/curricula/pokemon_generalization/curriculum.json \
  --resume models/quickstart_agent.zip \
  --opponent-checkpoint "$frozen_checkpoint" \
  --opponent-mode previous-stage \
  --train-battles 1000 --eval-episodes 1000 \
  --promotion-win-rate 0.75 \
  --through-stage 10 --attempts-per-stage 10 \
  --environments 4 --showdown-workers process \
  --rollout-steps 128 --batch-size 128
```

Each attempt trains until at least **1,000 ended battles across the whole pool**,
finishes the current PPO update, then evaluates exactly **1,000 battles**. Training
can exceed the budget; truncations count toward the training count.

- **750 or more evaluation wins:** promote immediately and start the next stage.
- **Below 750 wins:** resume the candidate and try again against the same opponent.
- **Ten unsuccessful attempts:** stop on `HOLD`, keeping the checkpoints.

Draws and truncations count as non-wins for this gate. The previous stage's
promoted model becomes the next stage's frozen opponent. The run preserves its
starting inputs as `models/smart_v1.zip` and `models/dumb.zip`; stage 1 exports
`models/smart_v2.zip`, stage 2 exports `models/smart_v3.zip`, and so on **inside the
campaign directory**. Those are version labels, not independently verified skill ratings.

To continue after a hold or interruption, use the same run directory and omit
its pinned starting options:

```bash
python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10
```

The 75% gate is a configurable target; improvement is not guaranteed at each
stage. You can change it on resume with `--promotion-win-rate 0.70`; old results
retain their original rules. To evaluate against Random, Dumb and saved Smart
models with separate gates, use [Campaign evaluation pools](campaign-evaluation-pools.md).
To define different teams/stages, follow [Create a campaign](creating-campaigns.md).
[All campaign flags](reference/run-curriculum.md).

## 7. View campaign results

In another terminal, point to the campaign above:

```bash
curriculum_run=".local/run/quickstart-generalization-01"
cat "$curriculum_run/evaluations.md"
```

This compact table is updated after each completed evaluation. It contains every
stage and attempt, frozen opponent, wins out of 1,000, win rate and `HOLD`/`PROMOTE`.
Held attempts remain visible. To refresh it every five seconds:

```bash
watch -n 5 cat "$curriculum_run/evaluations.md"
```

Ctrl+C stops the viewer without stopping training. For the full details:

```bash
cat "$curriculum_run/stage-01/report.md"
cat "$curriculum_run/stage-01/attempt-001/train/summary.json"
cat "$curriculum_run/stage-01/attempt-001/eval-current-001/summary.json"
```

Use the corresponding stage/attempt number for later runs. Reports appear when
their phase has run; evaluation retry directories may use `002`, `003`, etc.
`evaluations.csv` contains the same campaign history with checkpoint paths for
spreadsheets. Training summaries include rewards and battles/hour; evaluation
summaries include wins, losses, draws and truncations.

Keep the promoted checkpoints for play or further training. For an independent
comparison outside the campaign's repeated evaluation suite, follow
[checkpoint evaluation](evaluation.md).
