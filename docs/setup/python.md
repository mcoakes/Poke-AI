# Python environment

[Complete installation sequence](README.md#1-install-python-and-the-training-packages) · [Training](../ppo-training.md)

Use Python 3.11 or newer. From the project root:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git curl ca-certificates unzip xz-utils
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train]'
```

Activate the environment again in each new terminal:

```bash
source .venv/bin/activate
```

| Installation | Includes |
|---|---|
| `python -m pip install -e .` | Gymnasium, NumPy, team generation and emulator utilities |
| `python -m pip install -e '.[train]'` | Base dependencies plus PyTorch, Stable-Baselines3 and SB3-Contrib |
| `python -m pip install -e '.[dev,train,lua-test]'` | Training plus pytest, Ruff and synthetic Lua tests |

Dependencies and version ranges are defined in [pyproject.toml](../../pyproject.toml).
The editable install uses the project files directly, so extracted source updates
are available without copying Python modules elsewhere.

For a ROM-free check:

```bash
python examples/random_agent.py
```

This is a mock battle example. Continue with [Showdown](showdown.md) or
[mGBA](mgba.md) to use a battle engine.
