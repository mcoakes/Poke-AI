# Development

[Documentation](README.md) · [Developer utilities](reference/developer-tools.md) · [Sources](sources.md)

## Components

| Component | Responsibility |
|---|---|
| [mgba/train_ppo.py](../mgba/train_ppo.py) | Training entry point, backend selection and checkpoints |
| [ppo_vec_env.py](../src/firered_rl/ppo_vec_env.py) | Vector collection across paired players; frozen-opponent mode |
| [linked_env.py](../src/firered_rl/linked_env.py) | Joint mGBA decisions and episode lifecycle |
| [showdown_env.py](../src/firered_rl/showdown_env.py) | Persistent local Node worker and simulator episodes |
| [ppo_features.py](../src/firered_rl/ppo_features.py) | Shared categorical features and numeric preprocessing |
| [linked_reward.py](../src/firered_rl/linked_reward.py) | Shared reward points, turn costs and damage budgets |
| [mgba/adapter.lua](../mgba/adapter.lua) | Local JSON transport and input ownership |
| [mgba/profiles](../mgba/profiles/README.md) | Revision-specific memory and menu decoding |
| [sim/showdown/runner.cjs](../sim/showdown/runner.cjs) | Gen III mechanics and translation to FireRed IDs |
| [tools/pkhex_team_editor](../tools/pkhex_team_editor/PKHeXTeamEditor.csproj) | Party and progression edits in battery saves |

The paired controller prepares both players' choices before committing either.
It resolves to a shared decision boundary, retains local menus, and uses the
simulation master's battle data. Neither core is paused during native cable
play. A new mGBA episode restarts the whole pair from pre-connection states.
Showdown resets the two-player simulation inside its existing worker instead.

Keep both backends aligned with the [observation and action contract](turn-environment.md).
Map requests into original move/party slots, apply legal-action masks, and retain
terminal observations before reset for reward calculation and value bootstrapping.
Mechanics are provided by the cartridge or simulator, rather than approximated by
the mock backend.

## Public workflows and lower-level APIs

Use the documented CLI for training, benchmarking and human play. `FireRedTurnEnv`
is an independent NPC utility; it suspends one emulator process while awaiting
input and is unsuitable for a native linked pair. The older `FireRedBattleEnv`
and `LinkBattleSession` examples use frame/button actions and a smaller scalar
observation. They illustrate the scaffolding, not the paired PPO schema.

[config/example.toml](../config/example.toml) is a configuration sketch for the
scaffold. Launchers do not read it. Current command flags and the reward JSON
control training; editing the TOML does not change a run.

The Node runner and Lua files are internal protocol components, not independent
user-facing flag-based utilities. Their contracts are documented in the
[mGBA adapter](../mgba/README.md) and [Showdown adapter](../sim/showdown/README.md).

## Tests

Install the development dependencies in the active virtual environment:

```bash
python -m pip install -e '.[dev,lua-test,train]'
npm --prefix sim/showdown ci --omit=optional
```

Run the suite, or select the component relevant to a change:

```bash
python -m pytest -q
python -m pytest -q tests/test_showdown.py tests/test_showdown_cli.py
python -m pytest -q tests/test_link_turn.py tests/test_linked_env.py
python -m pytest -q tests/test_linked_reward.py
python -m pytest -q tests/test_tournament.py tests/test_evaluation.py
```

Lua tests use a fake emulator through Lupa; simulator tests require the pinned
Node dependency. Emulator-memory fixtures do not replace live ROM validation.
Use the visible [inspector](inspection.md), [cable test](link-test.md),
[switch test](link-commands.md) and [benchmark](benchmark.md) for the corresponding
live behavior. No ROM is included in the test suite.

The [developer utility reference](reference/developer-tools.md) covers the C
socket harness, .NET template tests, source patching and data export. Update
relevant workflow examples and flag tables when changing a CLI.

## Adding support

Keep ROM identity, offsets and decoders inside a [profile](../mgba/profiles/README.md).
Validate each supported revision against its actual game state. Additional
simulators must preserve categorical IDs, masks, slot ordering, terminal results
and reward timing before reusing a trained checkpoint. There is currently no
cross-machine worker or matchmaking service.
