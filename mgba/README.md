# mGBA adapter

[Project documentation](../docs/README.md) · [Build setup](../docs/setup/mgba.md) ·
[Utility reference](../docs/reference/README.md) · [ROM profiles](profiles/README.md)

The Lua adapter connects Python to a running mGBA core over newline-delimited JSON
on `127.0.0.1`. Launchers create per-player bootstrap scripts, select ports and
states, and load the adapter automatically. The supported build and native patches
are documented in the setup guide.

## Components

| File | Role |
|---|---|
| [adapter.lua](adapter.lua) | Connection framing, validation, inspection, frame input and reset |
| [link_connect.lua](link_connect.lua) | NPC dialogue, cable setup and battle startup |
| [link_turn.lua](link_turn.lua) | Owned move/switch/wait/human control through decision boundaries |
| [capture_state.lua](capture_state.lua) | Boot a generated save and capture a counter snapshot |
| [profiles](profiles/README.md) | FireRed-specific identity, addresses and decoding |

Independent emulators have one process each. Native pairs share one process and
cable session, with a separate Lua context and adapter port per core. Headless mode
uses the same Qt frontend under Xvfb. `--speed` is configured by launchers.

## Bootstrap configuration

Set globals before loading `adapter.lua` when integrating a custom launcher:

| Global | Meaning |
|---|---|
| `FIRERED_RL_PORT` | Adapter port; default 55355 |
| `FIRERED_RL_PROFILE` | Profile ID; default `firered_us_v1_0` |
| `FIRERED_RL_PROFILE_PATH` | Optional explicit profile file |
| `FIRERED_RL_STATE_PATH` | Configured independent reset state |
| `FIRERED_RL_LOG_PATH` | Optional adapter diagnostic file |
| `FIRERED_RL_NATIVE_LINK_SESSION` | Native-pair mode, set by the link launcher |
| `FIRERED_RL_LINK_SESSION_ID` | Shared session identity, set by the launcher |
| `FIRERED_RL_LINK_PLAYER` | Launcher player number for the native pair |

The launchers verify ROM SHA-1; Lua checks normalized game code and CRC32.
Unsupported profiles reject control requests. A client cannot provide an arbitrary
reset filename over the socket.

## Protocol

Send one JSON object per line. Failures return `{"ok":false,"error":"..."}`.
Use the Python [bridge](../src/firered_rl/backends/mgba_bridge.py) for request and
response handling; inspect capability fields in `status` before using a controller.

| Command | Purpose |
|---|---|
| `ping`, `status` | Identity, compatibility and controller capabilities |
| `inspect` | Full profile inspection, including diagnostic validity fields |
| `step` | Raw button action held for 1–600 frames; returns frame diagnostics |
| `reset` | Restore configured independent state and report its frame/RAM fingerprint |
| `close` | Close the client connection |
| `link_status` | Cable identity and connection evidence |
| `link_connect` | Begin owned NPC connection control |
| `link_connect_continue` | Release a staged connection barrier |
| `link_connect_status`, `link_connect_cancel` | Read or cancel the connection controller |
| `link_battle_start` | Continue from the Colosseum onto battle pads |
| `link_turn_prepare` | Validate and prepare one move, switch, wait or human observer request |
| `link_turn_commit` | Commit the owning client's prepared request ID |
| `link_turn_status`, `link_turn_cancel` | Read or cancel the turn controller |

Raw input example: `{"command":"step","action":1,"frames":1}` presses A.
Frame callbacks count the duration and release keys; normal emulation continues
between requests. Reset schedules the load at a frame callback and reports
`seed_applied:false`: its seed argument does not reseed FireRed RAM.

## Paired control

Python prepares both choices before committing either. Each request has an ID,
kind, target slot where applicable, and expected turn. The controllers read fresh
menus and cursor positions, then resolve to a command menu, required replacement
or terminal outcome. Disconnects, cancellation and completion release owned keys.
The human observer preserves human input and lets the AI wait for a human choice.

The simulation master provides active battle state and the shared turn counter;
the follower provides its local menu and party view. Python
[pairs inspections](../src/firered_rl/link_inspection.py) before building policy
observations and validating rewards. Raw follower battle RAM is not treated as a
second independent authority. Terminal inspection is captured before battle cleanup.

Do not call independent reset or suspend a core inside an active native pair.
Episode reset restarts both cores from their counter states. The paired environment
handles replacements, time limits and validated outcomes; the low-level socket is
not itself a Gymnasium environment.

See [observations/actions](../docs/turn-environment.md),
[inspection](../docs/inspection.md) and [development](../docs/development.md).
