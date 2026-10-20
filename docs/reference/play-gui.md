# Local human-vs-AI GUI

[Utility index](README.md) · [Workflow](../play-gui.md) · [Source](../../mgba/play_gui.py)

Choose parties in the browser, generate verified states, and play a visible mGBA
battle against a checkpoint. Requires the mGBA, save-editor and PPO dependencies.

```bash
python mgba/play_gui.py --rom FireRed.gba --source FireRed.sav \
  --workers 2 --speed 1
```

Select a saved team from any directory under `config/` on either side. Teams
directly in `config/` appear under **Misc**. Without `--model-list`, checkpoints in `models/` and its
subfolders are listed automatically; Smart v10 is preselected when available.
For the two-choice classroom menu, add `--model-list config/play/classroom.json`
after saving the models described in the workflow guide.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show options and exit. |
| `--rom` | Required | Supported FireRed USA/Europe v1.0 ROM. |
| `--source` | Required | Existing 128 KiB battery `.sav` with Oak's Parcel delivered and Direct Corner unlocked. |
| `--checkpoint` | Smart v10 if available | Preselect a checkpoint path or `latest`, overriding the automatic preference for Smart v10 in `models/`. Latest resolves when each match starts. Without Smart v10, select the first available model. |
| `--model-list` | Off | JSON shortlist with `version: 1`, a `default` ID and `models` entries (`id`, `label`, `checkpoint`). Lists only those models, default first; relative checkpoint paths use the project root. Mutually exclusive with `--checkpoint`. |
| `--checkpoint-dir` | `.local/run` | Additional folder to scan recursively for up to 200 recent checkpoint ZIPs. All ZIPs under `models/` are also listed, outside this limit. Ignored with `--model-list`. |
| `--emulator` | `.local/mgba/bin/mgba-qt` | Patched mGBA executable used for capture and human play. |
| `--port` | `8080` | HTTP port; always bound to `127.0.0.1`. |
| `--ports` | `55365 55366` | Exactly two adapter ports: human first, AI second. All three ports must differ. |
| `--workers` | `2` | Maximum available save workers in the page, capped at the two teams. Must be positive. |
| `--speed` | `1` | Initial battle speed shown in the page: positive multiplier or `unlimited`. |
| `--capture-speed` | `unlimited` | Separate speed used for background save-state capture. |
| `--setup-speed` | `unlimited` | Speed for NPC dialogue and walking into battle; changes to the selected battle speed before READY. Requires the runtime-speed mGBA patch. |
| `--build-timeout` | `180` | Wall-clock seconds per capture; at least 10. The capture's emulated-frame limit also applies. |
| `--startup-timeout` | `30` | Seconds allowed for mGBA adapter readiness. |
| `--connect-timeout` | `180` | Seconds allowed per automatic cable connection/battle-start stage. |
| `--max-steps` | `200` | Decision limit for the human/AI match; unfinished battles truncate. |
| `--device` | `cpu` | PPO inference device, such as `cpu` or `cuda`; emulation stays on the CPU. |
| `--no-browser` | Off | Print the local URL without opening a browser automatically. |

The page also accepts a human trainer name (1–7 letters, digits, or spaces).
Blank preserves the source name; the AI's name is retained. No CLI flag is needed.

The page controls saved-team directory categories, random-team categories and sizes,
custom parties, selected checkpoint,
battle speed, worker count, and stochastic AI sampling. Player 1 is always human.
The checkpoint is frozen per match. Save generation is headless through Xvfb;
the game itself opens visibly. There is no `--headless` GUI mode.
