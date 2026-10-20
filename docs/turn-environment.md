# Observations and actions

[Training](ppo-training.md) · [Rewards](training-rewards.md) · [Inspection](inspection.md) · [Stage 2 audit](training-audit.md)

Both PPO backends support the v1 structured observation and 11-action schema.
Showdown can also use [observation v2](observation-v2.md), with explicit stats,
stat stages, move descriptors, opponent party state, abilities and weather.
The raw arrays below are v1, also retained unchanged within v2.
A normal decision submits both players' choices and resolves to the next selection
or terminal outcome. Fainting and Baton Pass can require a recipient selection
within that turn; forced continuations can advance several turns without a choice.
Rewards cover the interval between these decision boundaries.

## PPO actions

| ID | Choice |
|---:|---|
| 0–3 | Move slots 1–4 |
| 4–9 | Original party slots 1–6 |
| 10 | Wait or forced continuation when the action mask permits it |

Slots retain their original team order after switching. Bag item use is disabled;
held items function normally. Action 10 is not Run in a linked or simulated match.
When no move can be selected, the controller supports Struggle through the legal
fallback action. Missing slots in one-, two- or three-move sets stay masked.
The bundled generated templates all have four moves.

Masks exclude empty, fainted, Egg and current switch targets, exhausted/absent
moves, and battle restrictions such as Disable, Encore, Taunt, Torment, Choice
Band, Imprison and trapping. Forced recipient selections allow only valid
replacements. mGBA reads cartridge state; Showdown uses simulator requests and
maps them into the same action slots. Only single battles are supported.

## Raw observation

The Gymnasium `Dict` contains fixed-size integer arrays, zero padded for missing
entries. Species, move, type and item IDs are categorical FireRed IDs, including
FireRed's internal species numbering. They are not strength rankings.

| Key | Shape | Fields in each row |
|---|---|---|
| `party` | `(6, 10)` | occupied, species, level, HP, max HP, type 1, type 2, major-status bits, held item, Egg flag |
| `moves` | `(6, 4, 4)` | move ID, base type, current PP, maximum PP |
| `active` | `(2, 10)` | occupied, species, level, HP, max HP, type 1, type 2, major-status bits, own party slot, own volatile-status bits |
| `bag` | `(5, 58, 2)` | item ID, quantity |
| `context` | `(4,)` | game turn, required-switch flag, terminal flag, outcome code |
| `action_mask` | `(11,)` | 1 for available actions, 0 otherwise |
| `opponent_party` | `(6, 2)` | occupied, species ID |
| `opponent_moves` | `(6, 4, 4)` | move ID, base type, current PP, maximum PP |
| `opponent_active_index` | `(1,)` | Original opponent party slot, 1–6; 0 if unavailable |

Active rows are player then opponent. Own HP is absolute; opponent active HP is
expressed as a percentage with max HP 100. Opponent fields for own party slot
and volatile status are zero. Bag pockets are Items, Key Items, Poké Balls,
TM/HM and Berries; the Showdown bag is empty. The game-turn field is capped at 255.

**All opponent species and moves are available immediately**, including bench
Pokémon and their PP. This is intentionally a full-moveset training observation.
It does not include every hidden opponent value: bench HP, held items and full
stats are not in those opponent arrays. Active types come from battle state.
Active move and PP overlays account for temporary battle changes.

Major statuses such as poison, burn, sleep, paralysis and freeze are represented
by their status bits. Own volatile status includes conditions such as confusion.
The PPO encoder expands these into `status_bits (8, 32)` and
`volatile_bits (2, 32)` before converting model inputs to floats. Categorical IDs
use embeddings, and numerical fields are scaled by the feature extractor.

The schema does not yet encode all battle variables, such as complete stat
stages or weather. Simulator mechanics can use information the policy cannot
observe. Matching schemas makes checkpoint transfer possible; it does not prove
identical mechanics or identical performance between backends.

The [Stage 2 training audit](training-audit.md) inventories information available
to the policy and checks controlled Showdown cases against read-only native
engine state. It also exercises masks and episode boundaries through the real
PPO environment. Missing information is reported separately from incorrect
encoding; passing those cases does not establish mGBA equivalence or model skill.

Source: [raw observations](../src/firered_rl/turn_env.py),
[PPO features](../src/firered_rl/ppo_features.py),
[paired environment](../src/firered_rl/linked_env.py),
[Showdown environment](../src/firered_rl/showdown_env.py).

## NPC and raw-input utilities

The independent NPC turn tester uses a different meaning for action 10: escape.
Launch one emulator with an NPC battle checkpoint saved at a command menu:

```bash
python mgba/launch_emulators.py --rom FireRed.gba --ports 55356 \
  --states states/battles/npc.ss1 --speed 1
```

In another terminal, replace `12345` with that emulator's printed PID:

```bash
python mgba/test_turns.py --port 55356 --emulator-pid 12345
```

The process pauses while the terminal waits for a choice. Enter `q` to quit.
Use the actual mGBA PID, not an Xvfb wrapper or a shared link-pair process.
If the client is killed while paused, `kill -CONT 12345` resumes that process.
Use battle style SET for NPC trainer tests; move-learning and evolution prompts
are outside this controller's scope.

[run_baseline.py](reference/run-baseline.md) automates legal NPC choices.
[test_episode.py](reference/test-episode.md) instead sends frame-level NOOP while
you play manually. [test_inputs.py](reference/test-inputs.md) sends raw GBA buttons.
These are separate utilities, not the paired PPO training interface.
