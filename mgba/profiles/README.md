# ROM profiles

[mGBA adapter](../README.md) · [Sources](../../docs/sources.md) · [Observation contract](../../docs/turn-environment.md)

A profile describes one verified ROM revision. Keep identity, addresses, structure
sizes and decoding rules here; transport, process control and Gymnasium code use
the decoded interface.

## Supported revision

| Field | Value |
|---|---|
| Profile | `firered_us_v1_0` |
| Game | Pokémon FireRed USA/Europe v1.0 |
| Game code | `BPRE`, normalized from `AGB-BPRE` when needed |
| SHA-1 | `41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc` |
| CRC32 | `dd88761c` |

Python verifies the ROM bytes before launch. Lua validates game code and CRC32
against the selected profile. The current launcher registry and save pipeline
support this revision only; another Lua filename alone does not add end-to-end
support for another ROM. FireRed v1.1, LeafGreen, hacks and translations require
separate validation.

## Profile files

| File suffix | Responsibility |
|---|---|
| [firered_us_v1_0.lua](firered_us_v1_0.lua) | Identity and profile composition |
| [_inspect.lua](firered_us_v1_0_inspect.lua) | Party decryption, integrity checks, bag and base lookup data |
| [_battle.lua](firered_us_v1_0_battle.lua) | Active battlers, party order, menus, restrictions and outcome |
| [_link.lua](firered_us_v1_0_link.lua) | Cable status and player identity |
| [_lobby.lua](firered_us_v1_0_lobby.lua) | NPC menus, map and connection tasks |
| [_battle_start.lua](firered_us_v1_0_battle_start.lua) | Colosseum movement and battle pads |
| [_events.lua](firered_us_v1_0_events.lua) | Battle text and HP events |
| [_revealed.lua](firered_us_v1_0_revealed.lua) | Executed-move history for diagnostics |
| [_names.json](firered_us_v1_0_names.json) | Human-readable species, moves, types and item IDs |

Base species/move types use static data. Temporary active types, moves and PP
come from the battle overlay. Remaining/fainted counts exclude invalid and Egg
records and retain a validity flag. An unreadable count must not imply a win.

## Adding a revision

Add its exact identity to the launcher registry and provide compatible profile
functions, lookup data and save-pipeline support where required. The adapter can
select a profile with `FIRERED_RL_PROFILE` or `FIRERED_RL_PROFILE_PATH` before load.
Validate party decoding, all move/PP/HP fields, bag quantities, menu cursors,
logical party order, status restrictions, resets and terminal outcomes against the
actual revision. Native link support also needs both player roles and connection
stages checked in-game.

Preserve the shared policy schema when using existing checkpoints, including the
intentional full opponent species/moves/PP fields. The evaluator can contain more
data than the policy; follow the explicit observation contract rather than a
blanket assumption that all inspected values are policy inputs.

Fixture tests and source symbols support validation, but similar memory layouts
alone are insufficient to advertise a ROM as supported. Record data provenance
beside the profile and in the [source guide](../../docs/sources.md).
