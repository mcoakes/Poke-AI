# Local Showdown adapter

[Setup](../../docs/setup/showdown.md) · [Training](../../docs/showdown-training.md) ·
[Observations/actions](../../docs/turn-environment.md) · [Sources](../../docs/sources.md)

`runner.cjs` uses `pokemon-showdown` **0.11.11** as a local Gen III battle library.
Python manages one persistent Node process per pair. No server, browser, network
port, account, ROM or save state is needed. Node.js 18 or newer is required.

Install from the project root:

```bash
npm --prefix sim/showdown ci --omit=optional
```

The package and lockfile are pinned together. The adapter reads private simulator
objects for full party data and precise action masks; changing the dependency
requires integration validation. See [npm warning guidance](../../docs/troubleshooting.md#npm-installs-with-dependency-warnings).
Generated `node_modules/` is not shipped or tracked.

## Internal protocol

The runner has no command-line flags. It reads one JSON request per stdin line
and writes one JSON response per stdout line:

| Command | Input | Output |
|---|---|---|
| `ping` | `command` | Package version and generation |
| `reset` | Two `teams` objects and a four-word `seed` | Both initial inspections, masks and protocol messages |
| `step` | Two player `actions` | Next decision/terminal inspections and protocol messages |

The Python [Showdown environment](../../src/firered_rl/showdown_env.py) owns this
protocol. Use [train_ppo.py](../../docs/reference/train-ppo.md) for user-facing
training rather than sending requests manually.

For saved-model comparisons, use [single-policy evaluation](../../docs/evaluation.md)
or [model-directory round robins](../../docs/model-evaluation.md). Round robins
default to compact result files; full per-turn logging is optional.

## Mapping and scope

Battles use `gen3customgame`. Teams come from the same JSON schema as the save
editor. FireRed name/ID tables and the generation catalog define supported
species, moves and items. PP starts without PP Ups; initial ability, nature,
gender and friendship follow the editor's construction defaults.

The runner translates simulator requests to original party/move slots and the
shared 11-action mask. Rewards, observation encoding and PPO checkpoints are shared
with mGBA. Ruleset defaults, obscure cartridge behavior and RNG timing can differ;
see [fidelity and checkpoint transfer](../../docs/showdown-training.md).
