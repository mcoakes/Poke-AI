# Inspect emulator state

[Utility index](README.md) · [Workflow guide](../inspection.md) · [Source](../../mgba/inspect_state.py)

Read named party, bag, active-battler, menu and outcome data from a running adapter.

```bash
python mgba/inspect_state.py --port 55356 --peer-port 55355 --json
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--host` | `127.0.0.1` | Adapter host. The supplied Lua server binds to `127.0.0.1`; this is a local workflow. |
| `--port` | `55355` | Port of the already-running adapter to control or inspect. |
| `--peer-port` | Not set | Partner's adapter port for a complete paired link-battle inspection |
| `--json` | Off | Print complete named JSON diagnostics |
| `--observation-version` | `v1` | `v2` validates and prints native v2 policy fields; requires `--peer-port`, a linked battle and adapter 0.19.0 on both players. |

## Usage notes

For linked games, supply the partner's port to combine authoritative battle data with local menus and party information. This is an emulator inspector, not a Showdown socket client. Simulator inspections are recorded in run logs.
