# Scaffold examples

[Utility index](README.md) · [Development](../development.md)

These examples use the mock backend. They need only the base Python installation;
they do not launch a ROM, create a native link session or train PPO.

## Random agent

```bash
python examples/random_agent.py
```

[Source](../../examples/random_agent.py). No command-line flags. Runs random raw
button actions in one mock `FireRedBattleEnv` episode and prints its return.

## Two-player scaffold

```bash
python examples/link_battle.py
```

[Source](../../examples/link_battle.py). No command-line flags. Advances two mock
backends through `LinkBattleSession`; it demonstrates the Python coordination
scaffold, not actual cable emulation.

## Vector example

```bash
python examples/vector_agents.py --num-envs 4
```

[Source](../../examples/vector_agents.py).

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit |
| `--num-envs` | 4 | Number of mock vector environments; choices 1, 2, 3, 4 |

For the actual structured battle policy and repeated training, use the
[PPO guide](../ppo-training.md). The mock examples retain their smaller scalar
observation and raw button actions.
