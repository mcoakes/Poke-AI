# Documentation

[Battle-count campaigns, named models and evaluation history](battle-count-campaigns.md)

[Project overview](../README.md) · [Utility reference](reference/README.md)

Examples run from the project root with the Python virtual environment active.
Paths such as `FireRed.gba` and `checkpoints/latest.zip` refer to your own files.

## Getting started

| Guide | Purpose |
|---|---|
| [Quickstart](quickstart.md) | Seven steps: dependencies, saves, example teams, both trainers, human play, campaigns and results |
| [Dependencies](setup/README.md) | Ordered copy-and-paste installation commands with readiness checks |
| [Python environment](setup/python.md) | Virtual environment and optional extras |
| [Showdown setup](setup/showdown.md) | Local simulator dependencies |
| [mGBA setup](setup/mgba.md) | Pinned build, scripting, CMake and ROM compatibility |
| [Save editor setup](setup/save-editor.md) | .NET SDK and PKHeX.Core |
| [First FireRed save](setup/first-save.md) | Starter, Oak's Parcel and the Direct Corner |

## Working with agents

| Guide | Purpose |
|---|---|
| [PPO training](ppo-training.md) | Self-play, resume, frozen opponents and settings |
| [Generalization campaign](generalization-campaign.md) | Ten directory-based stages, separate validation teams, and final transfer tests |
| [Growing-team curriculum](curriculum.md) | Ten stages with cumulative pools, fixed evaluations and promotion checks |
| [Create a campaign](creating-campaigns.md) | Walkthrough: choose teams, write stages, set promotion rules and run |
| [Advance campaign opponents](curriculum-opponents.md) | Freeze each promoted model as the opponent for the next stage |
| [Campaign evaluation pools](campaign-evaluation-pools.md) | Test Random, Dumb and Smart opponents with editable promotion gates |
| [Checkpoint evaluation](evaluation.md) | Frozen-policy comparisons on repeatable, side-balanced matchups |
| [Rule-based opponent](rule-based-opponent.md) | Optional observation-only Gen III heuristics for separate Showdown evaluations and training |
| [Move-order audit](move-order-audit.md) | Test checkpoint slot preference, probabilities and observed move execution |
| [Move-slot training experiment](experiments/move-slot-shuffle.md) | Randomize learner move presentation while preserving simulator actions and frozen opponents |
| [Shared move-scoring experiment](experiments/shared-move-policy.md) | Compare fresh policies with shared move scoring and the existing architecture |
| [Rules loss review](rules-loss-review.md) | Replay sampled losses with bounded action and heuristic diagnostics |
| [Combined validation](combined-validation.md) | Preserve existing cases while testing rules and learned agents in one suite |
| [Opponent pools](opponent-pools.md) | Train/evaluate against rules, multiple frozen agents, Random, or a configurable mixture |
| [Establish a baseline](baseline.md) | Compare starting, best and latest models on shared schedules with uncertainty estimates |
| [Periodic validation](periodic-evaluation.md) | Evaluate during training, preserve `best.zip`, and retain bounded evaluation logs |
| [New-team evaluation](generalization-evaluation.md) | Check overlap and compare two best models on fresh team configurations |
| [Experiment record](experiments/README.md) | Results, limitations and the next controlled training experiment |
| [Audit training inputs](training-audit.md) | Check information visibility, simulator encoding, legal actions and PPO episode boundaries |
| [Observation v2](observation-v2.md) | Richer Showdown inputs, schema compatibility and a controlled training comparison |
| [Native mGBA v2](mgba-observation-v2.md) | Use v2 checkpoints in linked FireRed, with field mapping and live verification |
| [Model round robin](model-evaluation.md) | Rank a directory of models, including Random, across multiple team configurations |
| [Checkpoint selection](checkpoints.md) | Explicit paths, latest discovery and frozen references |
| [Showdown training](showdown-training.md) | Simulator team inputs and compatibility |
| [Teams and profiles](team-profiles.md) | Competitive, casual, random and mixed pools |
| [Custom team generation](team-generation.md) | JSON schema, moves, EVs, IVs, items and nature |
| [Automatic save-state generation](automatic_states.md) | Turn JSON teams into mGBA counter states |
| [Template preparation](template_preparation.md) | Badge and location editing on its own |
| [Rewards](training-rewards.md) | Damage, knockouts, wins and battle-length costs |
| [Observations and actions](turn-environment.md) | Policy inputs, legal moves and decision boundaries |
| [Reports and battle logs](battle-logs.md) | Checkpoints, metrics, readable logs and diagnostics |
| [Storage and cleanup](storage.md) | Rotate logs, retain selected models and prune completed campaign data |
| [500-battle comparisons](battle-blocks.md) | Compare disjoint blocks against a frozen opponent |
| [Human versus AI](human-vs-ai.md) | Play a visible FireRed match against a checkpoint |
| [Battle Studio GUI](play-gui.md) | Build both parties in the browser, generate saves, and play against an agent |

## Emulator testing

| Guide | Purpose |
|---|---|
| [Linked environment pools](link-pool.md) | Automatically connect multiple pairs |
| [Scripted switch sequences](link-commands.md) | Verify repeated switches and final active slots |
| [Throughput benchmarking](benchmark.md) | Compare environment counts and emulator speeds |
| [Manual link test](link-test.md) | Diagnose one native cable pair |
| [State inspection](inspection.md) | Read parties, moves, bag, menus and outcomes |
| [Troubleshooting](troubleshooting.md) | Find the relevant check or diagnostic file |

## Reference and development

[Every utility and flag](reference/README.md) · [Architecture and tests](development.md) ·
[mGBA adapter](../mgba/README.md) · [ROM profiles](../mgba/profiles/README.md) ·
[Showdown adapter](../sim/showdown/README.md) · [Sources and thanks](sources.md)
