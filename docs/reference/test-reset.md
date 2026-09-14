# Check repeatable resets

[Utility index](README.md) · [Workflow guide](../automatic_states.md) · [Source](../../mgba/test_reset.py)

Reload each adapter's configured independent save state and compare restored frame/RAM fingerprints.

```bash
python mgba/test_reset.py --ports 55355 55356 --repeats 3
```

## Options

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and available options, then exit. |
| `--ports` | `55355 55356` | Adapter ports in player order; use distinct available local ports. |
| `--repeats` | `3` | Number of restore/fingerprint checks per port. |

## Usage notes

Ports may contain one or more independent adapters with configured reset states. Fingerprints must repeat within each emulator; different teams may have different fingerprints. Do not reset a single core inside a live native link pair.
