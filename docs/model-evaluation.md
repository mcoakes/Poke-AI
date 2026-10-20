# Compare a directory of models

[Every flag](reference/evaluate-models.md) · [Single-checkpoint evaluation](evaluation.md) · [Evaluation plan templates](../config/evaluations/README.md)

`mgba/evaluate_models.py` runs a **round-robin tournament**: every saved model
plays every other model, plus a uniform random legal-action agent by default.
One JSON plan can specify several team configurations. The runner produces an
overall leaderboard, a leaderboard for each configuration, and head-to-head tables.

This is evaluation only. PPO policies stay frozen and choose their
highest-probability legal action. It uses the local Showdown simulator, with the
supported v1/v2 policy schemas and the same 11 actions as training. Each model's
schema is detected, so a directory can contain both versions. The preflight
`model-schemas.json` records versions. No ROM, generated saves, mGBA
windows or external server are needed.

## 1. Choose a model directory

Run from the project root with [Python training extras](setup/python.md) and
[Showdown](setup/showdown.md) installed. Your campaign already exports the models
you need:

```bash
curriculum_run=".local/run/curriculum-smart-generations-01"
ls "$curriculum_run/models"
```

By default, every direct `*.zip` file becomes a participant, including `dumb.zip`
and the exported `smart_v*.zip` models. Byte-identical copies are counted once;
their aliases remain in the saved plan. Distinct model files are checked for the
expected PPO observation/action schema before battles begin.

Use any directory of saved agents with `--models-dir PATH`. To select only Smart
exports, add `--model-glob 'smart_v*.zip'`. To select nested final checkpoints,
use `--model-glob '**/final.zip'`. Quote the pattern so Python handles it.

## 2. Preview the workload

The bundled plan evaluates six-type teams, mixed teams of three, and competitive
teams of six:

```bash
evaluation_run=".local/run/model-evaluation-01"
python mgba/evaluate_models.py \
  --models-dir "$curriculum_run/models" \
  --config config/evaluations/round_robin.json \
  --run-dir "$evaluation_run" \
  --episodes 1000 \
  --environments 48 \
  --showdown-workers process \
  --dry-run
```

The preview lists participants and the total battle count. It hashes model files
and validates team inputs but does not load PPO policies, create output files or
run battles.

With **three checkpoints plus Random**, there are six model pairs. At 1,000
battles across three team configurations, that is **18,000 total battles**.
The count already includes player-side swaps; it is not doubled afterward.

## 3. Run the comparison

Use the same command without `--dry-run`:

```bash
python mgba/evaluate_models.py \
  --models-dir "$curriculum_run/models" \
  --config config/evaluations/round_robin.json \
  --run-dir "$evaluation_run" \
  --episodes 1000 \
  --environments 48 \
  --showdown-workers process
```

For a shorter first pass, use `--episodes 100` when starting the run. Each model
pair is tested in turn, using up to 24 concurrent battles with 48 environments.
The console reports the active pair, progress within that test and its result.

Each team configuration gets one saved schedule. **Every model pair uses that
same schedule**, including teams, battle seeds and the decision limit. Each
matchup is repeated with the models swapping player seats. Random-policy choices
also use saved per-battle seeds. There are no self-pairs or Random-vs-Random games.

Models are copied once into the run directory. Training can continue elsewhere
without updating these evaluation copies. Newly exported models are included
only when you start a new tournament.

## 4. Read the standings

```bash
watch -n 5 cat "$evaluation_run/leaderboard.md"
```

The leaderboard updates after each complete model-pair/configuration test.
It is marked **Provisional** until all pairs finish, because unfinished opponents
and configurations give participants unequal coverage.

The ranking uses:

```text
score = (wins + 0.5 × draws) / all battle attempts
```

Wins earn one point, draws half a point, and losses zero. Truncations earn zero
for both participants and remain visible as a separate count. Win rate is
`wins / all attempts`. All models play the same number of battles against every
other participant in each configuration, so the final overall score weights
opponents and configurations equally. Equal scores share a rank.

The top row is the highest-scoring model **on these opponents and team tests**.
Use the per-configuration rankings and head-to-head table to see where it wins
or struggles. Reward means do not affect this ranking. For a separate test of
generalization, choose team JSONs that were not used in training.

| File | Contents |
|---|---|
| `leaderboard.md` | Overall and per-configuration standings |
| `matchups.md` | Score matrices: row model versus column model |
| `leaderboard.csv` | Standings with exact numeric rates and counts |
| `head-to-head.csv` | Results by model, opponent and configuration, plus overall |
| `matches.jsonl` | One appended record per completed model-pair/configuration test |
| `summary.json` | Progress, active pair and machine-readable leaderboards |
| `tournament.json` | Frozen models, aliases, configuration and input hashes |
| `suites/` | Saved team contents, battle seeds and player-side assignments |
| `inputs/models/` | One copy of each distinct model |
| `matches/SCENARIO/PAIR/attempt-NNN/` | Evaluator settings, summary and individual battle results |
| `matches/SCENARIO/PAIR/attempt-NNN.log` | Evaluator console output and errors |

You can use `tail -f "$evaluation_run/matches.jsonl"` for appended results.
Use `watch` for Markdown tables, which are rewritten after each test.

## 5. Resume an interrupted tournament

```bash
python mgba/evaluate_models.py --run-dir "$evaluation_run"
```

The runner uses the saved models and suites. Omit `--config`, `--models-dir` and
the initial sampling options. Completed tests are validated and skipped; an
interrupted model-pair test restarts from its saved schedule in a new attempt
directory. Partial tests do not contribute to the leaderboard. Previous complete
results are not counted twice.

You can change execution settings such as `--environments 16` when resuming.
Changing participants, teams, episode count or seeds requires a new run directory.

## 6. Customize the team configurations

Copy a template and edit its `scenarios` list:

```bash
cp config/evaluations/round_robin.json config/evaluations/my_comparison.json
```

Then pass `--config config/evaluations/my_comparison.json` with a new run directory.
For example:

```json
{
  "version": 1,
  "name": "My model comparison",
  "episodes": 1000,
  "max_steps": 200,
  "seed": 10001,
  "include_random": true,
  "scenarios": [
    {"id": "familiar", "training_profile": "six_types"},
    {"id": "casual_three", "team_profile": "casual", "team_size": 3},
    {"id": "random_mirrors", "team_profile": "random", "team_size": 3, "mirror_teams": true},
    {"id": "custom", "team_dir": "../teams/quickstart"}
  ]
}
```

The [template reference](../config/evaluations/README.md) documents every JSON key,
fixed team pairs, path resolution and category sampling. `starters.json` provides
a ready-to-run fixed pair. Each configuration is tested separately, not mixed
into an unlabeled result.

## Disk use

Compact results are the default. The runner disables per-turn Gym traces, raw
Showdown protocol logs, readable/spectator logs and duplicate per-episode team
files. It retains model snapshots, shared suites, small worker diagnostics and
one result per battle. Resume and ranking use these compact records.

Use `--keep-battle-logs` only when you want full traces for diagnosis; they can be
large. The flag can change on resume and does not delete previously written logs.
Existing training/campaign logging defaults are unchanged.
