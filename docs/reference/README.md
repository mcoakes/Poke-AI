# Utility reference

[Documentation](../README.md) · [Setup](../setup/README.md)

Each utility page has a working example, every flag with its default, and usage
constraints. Examples run from the project root. Python CLI utilities accept
`-h` or `--help`; the no-argument setup scripts are explicitly marked separately.

## Training and play

| Utility | Purpose |
|---|---|
| [train_ppo.py](train-ppo.md) | Train masked PPO through self-play, against a frozen checkpoint, or against optional Showdown rules. |
| [evaluate_ppo.py](evaluate-ppo.md) | Compare checkpoints against checkpoints, Random or Showdown rules on reproducible matchups. |
| [audit_move_order.py](audit-move-order.md) | Probe move-slot sensitivity without training or executing alternative actions. |
| [review_rule_losses.py](review-rule-losses.md) | Replay loss-containing Rules matchups and review candidate choices. |
| [combine_eval_suites.py](combine-eval-suites.md) | Combine pinned schedules and bundle opponents for periodic validation. |
| [evaluate_generalization.py](evaluate-generalization.md) | Check new-team overlap and compare two best models against a shared opponent pool. |
| [prepare_diversity_experiment.py](prepare-diversity-experiment.md) | Prepare disjoint pools and launch scripts for a controlled broader-data experiment. |
| [establish_baseline.py](establish-baseline.md) | Compare campaign starting, best and latest checkpoints across training/validation teams and opponent categories. |
| [audit_training.py](audit-training.md) | Audit information visibility, controlled Showdown observations/actions and the real PPO interface without training. |
| [evaluate_models.py](evaluate-models.md) | Round-robin rankings for a model directory across several saved team configurations. |
| [run_curriculum.py](run-curriculum.md) | Train growing team pools, validate transfer and promote successful stages. |
| [prune_campaign.py](prune-campaign.md) | Preview or remove old campaign debug data and unselected candidate checkpoints. |
| [play_agent.py](play-agent.md) | Launch one visible FireRed link battle with a human on one side and a saved PPO policy on the other. |
| [play_gui.py](play-gui.md) | Open the browser party editor, generate both states, and launch a human/AI battle. |
| [benchmark_pool.py](benchmark-pool.md) | Run scripted mGBA matches and measure completed battles per hour across the pool. |
| [run_baseline.py](run-baseline.md) | Run random legal turn actions, or replay recorded actions, against an NPC in one existing emulator. |
| [compare_blocks.py](compare-blocks.md) | Compare non-overlapping blocks from current or older PPO runs. |

## Teams and saves

| Utility | Purpose |
|---|---|
| [build_states.py](build-states.md) | Prepare a source save, generate parties, boot Continue and capture verified mGBA counter snapshots. |
| [prepare_template.py](prepare-template.md) | Edit eight badges and/or the Continue-game counter warp in a copy of a FireRed battery save. |
| [build_team.py](build-team.md) | Replace a prepared template's party from one team JSON, preserving its existing progression and location. |
| [generate_team_profiles.py](generate-team-profiles.md) | Create all three team categories and all six party sizes in a new directory. |
| [generate_teams.py](generate-teams.md) | Create a flat directory of teams with configurable species, move styles and stats. |

## Emulator launchers

| Utility | Purpose |
|---|---|
| [launch_emulators.py](launch-emulators.md) | Start one independent mGBA process per port, with its adapter loaded automatically. |
| [launch_link.py](launch-link.md) | Start two mGBA cores in a shared native cable session, optionally navigating the NPC automatically. |
| [launch_pool.py](launch-pool.md) | Pair an even number of players, connect each pair through the NPC, and start their link battles. |

## Inspection and live checks

| Utility | Purpose |
|---|---|
| [inspect_state.py](inspect-state.md) | Read named party, bag, active-battler, menu and outcome data from a running adapter. |
| [test_adapter.py](test-adapter.md) | Check that a running adapter responds and recognizes the loaded ROM. |
| [test_inputs.py](test-inputs.md) | Send one raw GBA action to each selected adapter and verify the requested frame duration. |
| [test_reset.py](test-reset.md) | Reload each adapter's configured independent save state and compare restored frame/RAM fingerprints. |
| [test_link.py](test-link.md) | Poll two adapters until their native two-player connection and data exchange pass validation. |
| [test_turns.py](test-turns.md) | Interactively submit a move or switch and inspect its reward at the next decision boundary. |
| [test_episode.py](test-episode.md) | Reset one emulator, then watch frame-based Gymnasium rewards while you play the battle manually. |

## Examples and developer tools

[Mock examples](examples.md) cover all three scripts under `examples/`.
[Developer utilities](developer-tools.md) cover mGBA patching, PKHeX setup,
source-data export, the socket harness, direct .NET editor arguments and template
tests. Internal protocols are described in the [mGBA](../../mgba/README.md) and
[Showdown](../../sim/showdown/README.md) adapter documentation.
