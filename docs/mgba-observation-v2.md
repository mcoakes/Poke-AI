# Use v2 checkpoints in native FireRed

[Human play](human-vs-ai.md) · [Battle Studio](play-gui.md) ·
[Observation contract](observation-v2.md) · [Inspection flags](reference/inspect-state.md)

Adapter **0.19.0** supplies native fields for the existing v2 observation schema.
Linked mGBA training, evaluation and human play accept existing v2 checkpoints,
including `flat` and `shared-moves` policies. No weight conversion or retraining
is required to load them. This does not guarantee identical behavior or win rates
between the two battle engines.

## Update and play

Extract the project update, stop existing emulator sessions and restart the
Battle Studio Python server if it is running. Both linked players must load the
new Lua scripts. An older adapter is rejected before policy actions are submitted.
No additional CMake rebuild is needed for this change if your existing native
link/human-play build already works.

Use the same GUI command and choose a v2 checkpoint from `models/`. Alternatively:

```bash
python mgba/play_agent.py --rom "$PWD/FireRed.gba" \
  --checkpoint models/observation-v2-diversity-best-01.zip \
  --states states/counter/starters_charizard_lead.ss1 states/counter/starters_venusaur_lead.ss1 \
  --human-player 1 --setup-speed unlimited --speed 1
```

Schema selection is automatic; there is no extra human-play flag. Wait for
`READY`. The AI window stays hidden and the human controls the visible game.

## Live verification

While both games are at a battle selection menu, use their printed ports:

```bash
python mgba/inspect_state.py --port 55356 --peer-port 55355 \
  --observation-version v2
```

For Battle Studio's usual ports, use `55366` and `55365` instead. The output
includes exact active HP, item and ability IDs, stored stats, stat stages and
weather. `--json` includes all eight v2 arrays. Check both player perspectives,
a switch, a stat-changing move, weather and a knockout during your first live
test. Missing data produces an error rather than filled-in defaults.

Support is scoped to **FireRed US/Europe v1.0, two-player native singles link
battles**. A standalone NPC inspector without a peer is not a v2 inference path.

## Where the fields come from

| Field | Native source / normalization |
|---|---|
| Party HP, status, types, item, ability | Each player's decrypted local party; ability uses the saved ability bit and species table |
| Party stats | Stored party Attack, Defense, Sp. Attack, Sp. Defense, Speed; no species-stat estimates |
| Active HP, item, ability, stats, status | The simulation master's `gBattleMons`, mapped to each player's perspective; active values override lagging party copies |
| Stat stages | Native stages reordered to Atk/Def/SpA/SpD/Spe/accuracy/evasion and centered from 0–12 to −6–6 |
| Move descriptors | Bundled `gen3_move_details.json`, from pinned Showdown 0.11.11 Gen III Dex, in FireRed move-ID order |
| Hidden Power | Type and power computed from party or transformed active IVs |
| Weather | Master's `gBattleWeather` and `gWishFutureKnock.weatherDuration`; ability weather marked permanent |
| Volatiles | Both authoritative active status2 bitfields, kept as integers until bit expansion |

The move table matches the descriptors seen during training, including variable
damage moves whose cartridge power byte is only a sentinel. It is not a damage
prediction. Its data is bundled with Python: human play does not require Node.js.
Move details replace the entire active row after Transform/Mimic; empty trailing
slots stay empty. Original party slots and the existing action meanings remain.

The follower emulator does not own valid battle simulation records. Pairing
uses the master's active state and weather, with each player's local party and
menus. Terminal observations use the master's latched final parties/outcome.
Data is read only; no gameplay memory is rewritten for compatibility.

Layout references: [pret/pokefirered Pokemon structures](https://github.com/pret/pokefirered/blob/master/include/pokemon.h),
[battle structures](https://github.com/pret/pokefirered/blob/master/include/battle.h),
[weather constants](https://github.com/pret/pokefirered/blob/master/include/constants/battle.h),
and [FireRed symbols](https://github.com/pret/pokefirered/blob/symbols/pokefirered.sym).

## Validation limits

Automated checks exercise the Lua memory readers with constructed memory,
encrypted party layouts, paired master/follower snapshots, side reversal,
weather, sparse moves, status bits, terminal observations and actual checkpoint
inference. These checks are not a live ROM replay or proof of complete cartridge
and Showdown equivalence. Verify the first live battle before relying on native
v2 training results. This release does not alter rewards or the earlier Showdown
experiments.

For native PPO training or evaluation, use the ordinary `--backend mgba`, ROM
and counter-state arguments. Checkpoints select v2 automatically on resume;
fresh training can select `--observation-version v2`. Each native session checks
both adapter versions. Showdown-only tools such as rule-based opponent pools,
move-slot augmentation and periodic Showdown validation remain backend-specific.
