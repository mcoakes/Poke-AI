# FireRed RL

Train Pokémon battle agents with Gymnasium and PPO, using either a local Gen III
simulator or real FireRed link battles in mGBA. Run self-play, train against a
frozen opponent, or play against a saved agent.

[Quickstart: install → train → play](docs/quickstart.md) ·
[Install dependencies](docs/setup/README.md) · [Start training](docs/ppo-training.md) ·
[Evaluate checkpoints](docs/evaluation.md) · [Rank a model directory](docs/model-evaluation.md) ·
[Measure training progress](docs/baseline.md) ·
[Preserve the best checkpoint](docs/periodic-evaluation.md) ·
[Test new team configurations](docs/generalization-evaluation.md) ·
[Experiment record](docs/experiments/README.md) ·
[Rule-based opponent](docs/rule-based-opponent.md) ·
[Mixed opponent pools](docs/opponent-pools.md) ·
[Audit observations and actions](docs/training-audit.md) ·
[Richer Showdown observations](docs/observation-v2.md) ·
[Training curriculum](docs/curriculum.md) · [Create a campaign](docs/creating-campaigns.md) ·
[Generalization campaign](docs/generalization-campaign.md) ·
[Storage and cleanup](docs/storage.md) ·
[Teams and saves](docs/team-profiles.md) · [Play in Battle Studio](docs/play-gui.md) ·
[All utilities and flags](docs/reference/README.md) · [Documentation](docs/README.md)

## Built on

| Project | Used for |
|---|---|
| [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) | Local Gen III battle simulation |
| [mGBA](https://github.com/mgba-emu/mgba) | FireRed emulation, Lua scripting and local link-cable battles |
| [PKHeX.Core](https://github.com/kwsch/PKHeX) | Editing parties and preparing battery saves |
| [pret/pokefirered](https://github.com/pret/pokefirered) | Memory layouts, symbols and Gen III species/move/item data |
| [PokeAPI sprites](https://github.com/PokeAPI/sprites) | Pokémon pictures in the browser team builder |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | Environment interface |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | PPO, action masking and checkpoints |
| [PyTorch](https://github.com/pytorch/pytorch) and [NumPy](https://github.com/numpy/numpy) | Policy networks and observations |

Thank you to these projects' maintainers and contributors. Their work makes
this project possible. See [dependencies and source credits](docs/sources.md)
for pinned versions, data provenance and upstream licenses.

## Choose a battle backend

| | Showdown | mGBA |
|---|---|---|
| Battles | Local Gen III simulation | FireRed USA/Europe v1.0 |
| Teams | JSON files directly | Generated counter save states |
| Display | Headless, unthrottled | Visible or headless through Xvfb |
| Use | Train and iterate | Train, evaluate in FireRed, and play against an agent |
| Setup | [Python + Node.js](docs/setup/showdown.md) | [Python + mGBA](docs/setup/mgba.md), [save editor](docs/setup/save-editor.md) |

Both support the v1 PPO observation/action schema, rewards and checkpoint format.
Showdown also offers opt-in [observation v2](docs/observation-v2.md); v2 checkpoints
currently require Showdown.
Mechanics can differ between simulator and cartridge; evaluate transferred
policies in mGBA. Emulator support currently targets one verified ROM revision.
ROMs and save files are supplied by the user and are not included.

## Start with simulator training

After [installing dependencies](docs/setup/showdown.md), run from the project root:

```bash
python mgba/train_ppo.py --backend showdown \
  --environments 4 --team-profile mixed --team-size 3 \
  --rollout-steps 128 --batch-size 128 --total-timesteps 100000
```

This trains one shared policy across two simultaneous battles. Each player gets
a random three-Pokémon team each episode. The project includes 2,700 templates
across competitive, casual and random categories, with party sizes from one to six.

The console reports episode rewards and battles/hour. Checkpoints and logs are
saved under the run directory printed at startup. Continue with the
[training guide](docs/ppo-training.md), [reward guide](docs/training-rewards.md),
or [utility reference](docs/reference/README.md).
