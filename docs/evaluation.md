# Checkpoint evaluation

To compare every model in a directory, use the [round-robin evaluator](model-evaluation.md).
It includes Random, supports multiple team configurations, and builds shared standings.

For training and evaluation across progressively larger team pools, see the
[growing-team curriculum](curriculum.md). It uses this evaluator and saves the
same replayable suites and per-battle reports.

[Training](ppo-training.md) · [Every evaluation flag](reference/evaluate-ppo.md) · [Checkpoint selection](checkpoints.md)

`mgba/evaluate_ppo.py` measures a candidate against a fixed reference policy.
Both policies choose their highest-probability legal action. It never calls
`learn` or saves model weights. Training win rates include the learner's
exploration; evaluation uses the same action-selection rule for both players.

## Establish a baseline

Run from the project root with the training dependencies installed:

```bash
fronzen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"

python mgba/evaluate_ppo.py --backend showdown --environments 48 \
  --checkpoint "$fronzen_checkpoint" \
  --opponent-checkpoint "$fronzen_checkpoint" \
  --training-profile six_types --episodes 1000 --seed 10001 \
  --output-dir .local/run/eval-six-types-baseline
```

The same policy on both sides provides a reference. The default side swap runs
500 matchups twice for **1,000 total battle attempts**. Each pair shares team
assignments and battle seed, with the candidate controlling Player 1 in one game
and Player 2 in the other. Candidate statistics track the model's role, not a
fixed player number. Draws can make win rates lower than 50% for identical models;
score counts a draw as half a win.

## Train, then evaluate the new checkpoint

```bash
python mgba/train_ppo.py --backend showdown --environments 48 \
  --training-profile six_types --resume latest \
  --opponent-checkpoint "$fronzen_checkpoint" --total-timesteps 100000

python mgba/evaluate_ppo.py --environments 48 \
  --checkpoint latest --opponent-checkpoint "$fronzen_checkpoint" \
  --suite .local/run/eval-six-types-baseline/suite.json \
  --output-dir .local/run/eval-six-types-after
```

Use a new output directory for each run. `--suite` replays the original teams,
seeds, side assignments, episode count and decision limit. Do not combine it
with new team, seed, episode, side or max-step arguments. Showdown team contents
are embedded in the suite, so later edits to source JSONs cannot change a replay.
The schedule does not depend on worker count or completion order. Reproducibility
is within the pinned backend; changing simulator versions or inference platforms
is a separate comparison.

To assess broader performance, make another suite with
`--team-profile mixed --team-size 3`, or `--team-dir PATH`. Use `--mirror-teams`
with a random source to give both players the same sampled team. Seed 10001 is
separate from the training default of 1, but different seeds do not make shared
team templates a held-out team set. Use a separate team directory to test team
generalization. Keep the reference policy, suite and reward configuration fixed
when comparing candidates.

## Reports

The console updates candidate wins/losses/draws, win rate, truncations, reward
mean and battles/hour after each finished battle. The run directory contains:

| File | Contents |
|---|---|
| `config.json` | Candidate/reference paths, SHA-256 hashes, training counters and evaluation settings |
| `suite.json` | Reusable matchup schedule and team snapshots (or state paths/hashes for mGBA) |
| `episodes.jsonl` | One record per finished battle, with case ID, candidate side, outcome, rewards and decisions |
| `summary.json` | Live and final totals, win rates, draws, truncations, reward means and outcomes by side |
| `pair-N/` | Simulator or emulator diagnostics; optional spectator/readable logs |
| `failure.json` | Details if evaluation failed; errors are never counted as draws |

For compact Showdown evaluations, add `--no-step-logs`. It skips large per-turn
traces and per-episode team copies while keeping suites, model identities and
`episodes.jsonl` for replay and reporting. Readable/spectator logs have their own
flags and are already off by default in this evaluator.

Win rates use completed battles, including draws, as their denominator;
truncations are counted separately. Candidate score is `(wins + 0.5 × draws) /
completed`. Reward means include finished truncated attempts. Check truncations
before comparing percentages. Side-swapped games share a matchup, so treat the
pair as one unit when estimating statistical uncertainty. These reports do not
claim a confidence interval or automatically declare one model better.

## Evaluate in mGBA

```bash
python mgba/evaluate_ppo.py --backend mgba --rom "$PWD/FireRed.gba" \
  --checkpoint latest --opponent-checkpoint "$fronzen_checkpoint" \
  --states states/counter/fire_vs_water.ss1 states/counter/water_vs_fire.ss1 \
  --episodes 20 --environments 4 --speed unlimited --headless
```

mGBA runs the same policy schema through native paired Gymnasium battles. Counter
states determine the initial cartridge RNG; suite seeds do not reseed FireRed.
State paths and hashes must still be available when replaying an mGBA suite.
An mGBA suite is distinct from a Showdown suite. The emulator and simulation do
not promise identical outcomes. Evaluation ports start at 55455 by default to
avoid the training launcher's default range; choose a free range if running more
sessions concurrently.
