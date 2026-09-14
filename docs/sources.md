# Dependencies and source credits

[Project overview](../README.md) · [Setup](setup/README.md) · [Development](development.md)

This project combines open-source emulation, battle simulation, save editing and
reinforcement-learning tools. Thank you to their maintainers, contributors and
research communities for making these components available.

## Runtime components

| Project | Version used here | Role and upstream information |
|---|---|---|
| [mGBA](https://github.com/mgba-emu/mgba) | 0.10.5, commit `26b7884bc25a5933960f3cdcd98bac1ae14d42e2` | Emulator; see [license](https://github.com/mgba-emu/mgba/blob/0.10.5/LICENSE) and [Lua API](https://mgba.io/docs/scripting.html) |
| [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) | npm 0.11.11 | Gen III simulator; see [license](https://github.com/smogon/pokemon-showdown/blob/master/LICENSE) |
| [PKHeX](https://github.com/kwsch/PKHeX) | commit `77dcd3a7895bceaafbbff12d25bdf77c1acd8ca5` | PKHeX.Core save editing; see [license](https://github.com/kwsch/PKHeX/blob/master/LICENSE) |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | `>=1.1,<2.0` | Environment spaces and API |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | 2.7.1 | PPO training infrastructure |
| [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | 2.7.1 | MaskablePPO |
| [PyTorch](https://github.com/pytorch/pytorch) | `>=2.5,<3.0` | Neural networks and optimization |
| [NumPy](https://github.com/numpy/numpy) | `>=1.26,<3.0` | Observation arrays |

The authoritative dependency specifications are [pyproject.toml](../pyproject.toml),
[package.json](../sim/showdown/package.json) and the
[npm lockfile](../sim/showdown/package-lock.json). Each upstream retains its own
license and notices. Source checkout patches are in [patches](../patches);
retain upstream notices when redistributing builds or derived components.

## FireRed data and symbols

The browser party picker also includes 386 default sprites from
[PokeAPI/sprites](https://github.com/PokeAPI/sprites), snapshot
`2ecb4eeacd5a1718621fc30f12772e3f60d830b9`. See the bundled
[source notice](../web/play/sprites/README.md) and
[upstream licence](../web/play/sprites/LICENCE.txt). Thank you to the sprite
collection's maintainers. Images are local, so the GUI does not need an image CDN.

[pret/pokefirered](https://github.com/pret/pokefirered) supplies the source layouts,
constants, learnsets and symbols used to decode FireRed and build the team catalog.

| Material | Source |
|---|---|
| Species, moves, items and generation catalog | Source snapshot `c75f352304d529f6ba92d4f74b9cf8b5c3810788` |
| ROM address symbols | [`pokefirered.sym` at `3b401eaf14f193946899eb7c056d5ac02a470126`](https://github.com/pret/pokefirered/blob/3b401eaf14f193946899eb7c056d5ac02a470126/pokefirered.sym) |
| Symbol-file SHA-256 | `6f9d2929b78d0b723180653082c9a115b4b876657af8ab1c0493b4d14151f7b0` |
| Checked-in catalog | [firered_us_v1_0.json](../config/generation/firered_us_v1_0.json) |
| Human-readable ID mapping | [firered_us_v1_0_names.json](../mgba/profiles/firered_us_v1_0_names.json) |

The [catalog exporter](reference/developer-tools.md#export-team-catalog) reads the
source snapshot directly. Competitive-style templates are generated from this
project's curated sets; they are not a downloaded tournament-team database.

ROMs, battery saves and emulator snapshots are user-supplied and are not included.
Pokémon names and game content belong to their respective owners. This is an
independent project, not an official Pokémon, Nintendo, Game Freak or Creatures product.
