# Play against a trained agent

Human play shows only the human player's mGBA window. The AI core runs with its
window hidden inside the same native link session. This also follows
`--human-player 2` when using the CLI. Apply the current patches and rebuild once:

```bash
python tools/setup_mgba_link.py
cmake --build build/mgba --parallel
cmake --install build/mgba
```

[mGBA setup](setup/mgba.md) · [Build states](automatic_states.md) · [All play flags](reference/play-agent.md)

For a visual party picker with Pokémon pictures, editable moves/stats and automatic
save generation, use [Battle Studio](play-gui.md). The command-line workflow below
starts from existing counter states.

`play_agent.py` opens a visible native FireRed link battle. You control one side
with mGBA's keyboard/gamepad bindings; a saved PPO policy controls the other.
The policy loads once and never learns or overwrites its checkpoint during play.
A model trained with either backend uses this same inference interface.

```bash
python mgba/play_agent.py --rom "$PWD/FireRed.gba" \
  --checkpoint checkpoints/latest.zip \
  --states states/counter/starters_charizard_lead.ss1 states/counter/starters_venusaur_lead.ss1 \
  --human-player 1 --speed 1
```

Replace the checkpoint path with your saved model. State order is always player 1,
then player 2, regardless of which side is human.

Add `--setup-speed unlimited --speed 1` for fast NPC/cable setup and normal-speed
play. Apply the [runtime speed patch and rebuild](setup/mgba.md) first. The launcher
waits for acknowledgements from both cores before announcing READY.

1. Wait for `READY` before touching the controls.
2. Focus the human player's window and select moves and switches normally.
3. Advance your own battle text and choose replacements after fainting.
4. At the end, read the result and press Enter in the terminal to close the windows.

There is no terminal action-number prompt. The AI advances its own dialogue and
does not control the human core. Keep the human window open; Ctrl+C stops the match.
To play again, rerun the launcher from the counter states.

## Random teams and other settings

Select random built profile states instead of a fixed pair:

```bash
python mgba/play_agent.py --rom "$PWD/FireRed.gba" \
  --checkpoint checkpoints/latest.zip --team-profile mixed --team-size 3 \
  --human-player 2 --speed 1 --stochastic
```

`--state-dir states/counter` is another alternative. Select exactly one state
source. [Build profile states](automatic_states.md) before using a profile here.
`--stochastic` samples legal policy actions; the default chooses the most probable
legal action. The AI observes all opposing moves according to the
[training schema](turn-environment.md), but not your pending menu selection.
Bag use is not an AI action; held items work.

If training is using the default ports, add `--ports 55365 55366`. The loaded model
stays fixed even if another run later updates its file. `--headless` is rejected
because this mode requires the human window.

## Match report

The printed `.local/run/human-<timestamp>/` directory contains `config.json`,
`result.json`, and a `match/` folder with structured decisions and connection logs.
The configuration records checkpoint identity, states and player roles. The result
contains outcomes, decisions and termination/truncation status. `failure.json`
records exceptions when present.

This match scoreboard uses +1 for a win, −1 for a loss and 0 for a draw. The
[weighted PPO rewards](training-rewards.md) are training settings and do not alter
this scoreboard. The decision limit truncates unfinished matches without assigning
a win. This utility does not write the PPO training transcript/report collection.
