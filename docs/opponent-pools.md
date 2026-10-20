# Rules and learned opponents together

To retain existing Random/Dumb/Smart cases and add rules, use the
[combined validation walkthrough](combined-validation.md). Training opponents
and validation opponents are selected independently.


[Pool JSON schema](../config/opponents/README.md) · [Rule policy](rule-based-opponent.md) ·
[Training flags](reference/train-ppo.md) · [Evaluation flags](reference/evaluate-ppo.md)

Use `--opponent-pool FILE` with Showdown to train or evaluate against a mixture
of rules, frozen learned agents and optionally Random. Existing single-opponent
flags still work. This does not modify an existing campaign or change defaults.
Finish the rule-only baseline evaluation before deciding on a new training run.

## Choose one opponent or a pool

| Desired opponents | Training | Evaluation |
|---|---|---|
| Rules only | `--rule-based-opponent` | `--rule-based-opponent` |
| One frozen learned agent | `--opponent-checkpoint PATH` | `--opponent-checkpoint PATH` |
| Rules and learned agents, or multiple learned agents | `--opponent-pool FILE` | `--opponent-pool FILE` |
| Replay saved evaluation opponents | — | `--suite PATH` |

Do not combine a pool with another opponent selector. Existing pooled suites
pin their opponents and cannot be overridden. To create a different evaluation
pool, supply the team source and pool in a new output directory. Then reuse
that new suite for future checkpoint comparisons.

The pool option currently belongs to the standalone training and evaluation
CLIs. The curriculum launcher and round-robin configuration are not extended by
this flag. A curriculum can continue using its existing evaluation machinery.

## Training selection

The included `config/opponents/rules_and_frozen.json` gives each battle a 70%
chance of using the diversity experiment's frozen checkpoint and 30% chance
of using rules. Each pair chooses independently **once when a battle begins**;
the opponent cannot change during a move, forced replacement or waiting step.
Another choice is made after the battle ends or truncates. Short runs need
not realize exactly 70/30, and a longer-lived opponent will consume more steps.

Only Player 1 contributes learning transitions. Checkpoints in the pool remain
frozen. `--opponent-stochastic` samples their actions if requested; default is
deterministic. Rules remain deterministic and Random samples legal actions.
The older `--opponent-switch-chance` intervention is not supported with pools.

Models receive their original observation schema. A pool containing rules or
a v2 model makes the Showdown session produce v2, while v1 policies still get
v1 projections. The learner's schema is preserved when resuming.

## Establish a mixed evaluation

The evaluation example uses equal weights to make each result easy to compare.
It points to the same frozen reference used in the completed diversity run:

```bash
diversity_run=".local/run/diversity-v2-01"
checkpoint="$diversity_run/train/checkpoints/best.zip"
mixed_eval=".local/run/eval-rules-and-frozen-01"

python mgba/evaluate_ppo.py \
  --backend showdown --checkpoint "$checkpoint" \
  --opponent-pool config/opponents/rules_and_frozen_eval.json \
  --team-dir "$diversity_run/teams/validation" \
  --reward-config "$diversity_run/inputs/reward.json" \
  --episodes 2000 --seed 26093002 \
  --environments 48 --showdown-workers process \
  --rule-log --battle-log --no-readable-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --output-dir "$mixed_eval"

cat "$mixed_eval/summary.json"
```

Evaluation assigns fixed quotas, rounded using largest remainders in whole
seat pairs. Each weight must receive at least one matchup; otherwise increase
the budget. This example creates **1,000 battles against rules and 1,000 against
the checkpoint**, half in each candidate seat. The assignment order is seeded
and does not depend on worker completion order. With `--no-swap-sides`, quotas
use individual battles. `suite.json` records the realized counts.

`summary.json` contains `by_opponent` and `by_opponent_group`. `rules` is separate
from `learned`; checkpoint entries may instead use the `dumb`/`smart` labels.
`basic` continues to mean Random + Dumb only. Group win rates count truncations
and draws as non-wins. Always inspect counts and truncations alongside rates;
changing weights changes the meaning of the aggregate result.

Replay both types against a different candidate without reselecting the pool:

```bash
python mgba/evaluate_ppo.py \
  --checkpoint path/to/new-candidate.zip \
  --suite "$mixed_eval/suite.json" \
  --reward-config "$diversity_run/inputs/reward.json" \
  --environments 48 --showdown-workers process \
  --battle-log --no-step-logs --log-max-mb 2 --log-backups 1 \
  --output-dir .local/run/eval-rules-and-frozen-02
```

## Optional mixed training fork

Review the rule-only and mixed evaluations before starting this experiment.
The example keeps the frozen reference, rewards and lower learning rate; the
intervention is replacing 30% of opponent battles with rules:

```bash
python mgba/train_ppo.py \
  --backend showdown --resume "$checkpoint" \
  --opponent-pool config/opponents/rules_and_frozen.json \
  --team-dir "$diversity_run/teams/training" \
  --reward-config "$diversity_run/inputs/reward.json" \
  --learning-rate 0.0001 --rollout-steps 128 --batch-size 128 \
  --environments 64 --showdown-workers process --total-timesteps 100000 \
  --eval-suite "$mixed_eval/suite.json" --eval-every 50000 \
  --rule-log --battle-log --no-readable-log --no-step-logs \
  --log-max-mb 2 --log-backups 1 \
  --checkpoint-every 25000 --keep-checkpoints 2 --keep-evaluations 2 \
  --output-dir .local/run/ppo-mixed-opponents-01 --if-exists cancel
```

Here training is 70/30 while validation is 50/50. `evaluations.md` and CSV show
separate Learned and Rules columns. Best-checkpoint selection still uses total
wins / requested battles; it does not impose per-group gates. Inspect both
columns before accepting an improvement. Keep the original Random/Dumb/Smart
suite as an additional benchmark for forgetting. Alternatively, supply it as
`--eval-suite` and run the mixed evaluation separately.

Continue without repeating pool/team flags:

```bash
python mgba/train_ppo.py --output-dir .local/run/ppo-mixed-opponents-01 \
  --if-exists continue --total-timesteps 100000
```

The run copies models and a normalized pool into `inputs/opponent-pool/`.
Continuation validates opponent identities and weights. Original checkpoint
files, including `latest`, can advance without changing this pool. Continuing
starts fresh battles and uses the existing session-seed convention; it does
not resume an in-progress battle or promise bit-identical uninterrupted training.

`opponent-pool-metrics.json` reports per-opponent starts/completions for the
current training session. Starts include freshly reset battles still in flight.
With `--rule-log`, bounded `opponent-assignments.jsonl` records selections and
`rule-decisions.jsonl` records rule choices. Training logs use the run root for
the first session and the new session directory on continuation. Evaluation
saves the exact schedule and bundled checkpoint copies under `opponents/`.
Replay validates model and rule fingerprints and works without original files.

No level handicap or new switch reward is introduced. These pools are another
experimental control, not evidence that mixing opponents improves learning.
