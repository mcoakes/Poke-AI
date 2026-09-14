# Type progression with switching pressure

The original ten self-play team stages, with a 30% chance for a type-based rule
to replace the training opponent's action with a better legal switch. A
two-decision cooldown limits repeated interventions.

Stage 1 uses the initial frozen checkpoint. Later stages freeze the previous
stage's promoted model. The rule modifies only Player 2's training actions;
evaluation, learner actions and rewards keep their normal behavior.

Use a separate run directory to preserve previous campaigns. See the
[workflow and commands](../../../docs/switch-pressure.md) for setup,
customization, resuming and switch-rate reports.
