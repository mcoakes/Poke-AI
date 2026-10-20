# Showdown training

[Documentation](README.md) · [Install dependencies](setup/showdown.md) · [Training flags](reference/train-ppo.md)

Showdown provides local Gen III battles without emulator frames, NPC dialogue or
save-state generation. One persistent Node.js worker owns each paired battle.
The worker uses Pokémon Showdown 0.11.11 as a library; it starts no server or
network listener.

## CPU cores and workers

The default `--showdown-workers thread` has one Python PPO process, a Python
thread per pair, and **one separate Node.js simulation process per pair**.
Seeing one Python process does not mean battles use only one CPU core.

Use `--showdown-workers process` to also move each pair's Python observation,
reward and logging work into its own persistent process:

```bash
python mgba/train_ppo.py --backend showdown --environments 8 \
  --showdown-workers process --team-profile mixed --team-size 3 \
  --rollout-steps 128 --batch-size 128 --total-timesteps 100000
```

Eight environments mean four paired battles: four Python environment workers,
four Node workers, and one central PPO learner. Workers do not train separate
policy copies. Resume, frozen opponents, action masking and reports work in both
modes. Each pair's `workers.json` records its Python and Node PIDs.

Compare battles/hour with identical team sources and training settings. Process
mode removes contention for the Python interpreter between pairs, but adds IPC
and memory overhead, so more processes do not guarantee higher throughput.
`--torch-threads` controls central PyTorch CPU work, not the simulation count.

## Quick start

After [installation](setup/showdown.md):

```bash
python mgba/train_ppo.py --backend showdown --environments 4 \
  --team-profile mixed --team-size 3 \
  --rollout-steps 128 --batch-size 128 --total-timesteps 100000
```

For fixed teams, replace the profile flags with:

```text
--teams config/teams/starters_charizard_lead.json config/teams/starters_venusaur_lead.json
```

For a flat directory, use `--team-dir config/teams/mixed_example`.
The backend reads JSONs directly from [the team catalog](team-profiles.md).
`--headless` is unnecessary and `--speed` has no effect: simulation is always
headless and unthrottled. Environment count and policy optimization settings still
affect throughput.

## Shared training interface

Use the same [training commands](ppo-training.md) for checkpoint resume, frozen
opponents and self-play. The reward function, v1 observation/action shapes, policy
encoder, checkpoint lifecycle and metrics are shared with mGBA.
Opt-in [observation v2](observation-v2.md) adds richer native simulator inputs
and currently requires Showdown. Existing v1 checkpoints keep working.

Both players submit choices together. A normal step ends at the next turn's
selection or a battle result. Fainting and Baton Pass can introduce a recipient
choice during the same turn. Mandatory continuations use action 10; the simulator
supplies its exact legal-action mask. Original party slots remain stable after
switches. Bag item-use actions are disabled.

Simulator events feed `battle.log`; translated observations and actions feed
`battle-readable.log`. Per-episode `teams.json` records both inputs and the RNG
seed, and `showdown-protocol.log` preserves the original events. See
[logs](battle-logs.md) and [the adapter contract](../sim/showdown/README.md).

## Rules and compatibility

The ruleset is **Gen III Custom Game**, bounded by the project team parser:

| Setting | Behavior |
|---|---|
| Battles | Singles; one active Pokémon per side |
| Teams | 1–6 Gen III Pokémon, levels 1–100, 1–4 distinct moves |
| Learnsets | Not enforced, allowing the random-moves category |
| Ladder rules | No OU tier bans, species clause, sleep clause or team preview |
| Initial PP | Zero PP Ups, matching the current save editor |
| Initial ability | First species ability slot, matching the current save editor |
| Nature/gender | Explicit nature or the save editor's PID-derived defaults |
| Friendship | 255 |
| Deoxys | FireRed's Attack form |
| Bag | Empty; held items still function |

Species IDs map to FireRed's internal IDs, and item IDs map to Gen III values.
This matters when transferring an existing policy: a National Dex number is not
always the corresponding FireRed species ID. Move names that differ between
sources are resolved by move number.

RNG seeds reproduce battles within this pinned simulator, not FireRed's RNG
stream. Status flags are translated, but internal countdown bookkeeping can
still differ. The shared policy omits some information such as complete stat
stages and field effects; preserving checkpoint compatibility does not fill those
missing observation fields. See [the observation reference](turn-environment.md).

The engine runs Showdown's Gen III mechanics rather than a custom damage-only
approximation. Complete cartridge equivalence is not established. Use mGBA for
cartridge evaluation and fine-tuning, including human-versus-AI play.

[Upstream simulator documentation](https://github.com/smogon/pokemon-showdown/blob/master/sim/README.md)

Use [random mirror matches](team-profiles.md#random-mirror-matches) with
`--mirror-teams` to give both players the same randomly selected team each battle.

## Controlled profiles and evaluation

Use `--training-profile six_types` for a six-type team with matching attacks.
Each profile is a directory under `config/training_profiles/NAME/teams/`.
See [controlled profiles](../config/training_profiles/README.md).

[Evaluate checkpoints](evaluation.md) separately from training with frozen
policies, deterministic actions and a replayable team/seed schedule.
