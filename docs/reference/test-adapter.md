# Test adapter connectivity

[Utility index](README.md) · [Workflow guide](../inspection.md) · [Source](../../mgba/test_adapter.py)

Check that a running adapter responds and recognizes the loaded ROM.

```bash
python mgba/test_adapter.py --port 55355
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--host` | `127.0.0.1` | Adapter host. The supplied Lua server binds to `127.0.0.1`; this is a local workflow. |
| `--port` | `55355` | Port of the already-running adapter to control or inspect. |
| `--timeout` | `5.0` | Socket timeout in seconds for the connectivity check. |

## Usage notes

Prints the status response and fails if the ROM is unsupported or the adapter cannot be reached. It does not start mGBA.
