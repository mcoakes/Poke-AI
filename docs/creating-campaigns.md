# Create a training campaign

[Battle-count campaigns, named models and evaluation history](battle-count-campaigns.md)

[Mixed evaluation opponents and editable gates](campaign-evaluation-pools.md)

[Documentation](README.md) · [Stage behavior and reports](curriculum.md) · [Every flag](reference/run-curriculum.md)

A **curriculum** defines the ordered teams and stages. A **campaign** is one run
of that curriculum, with its own starting checkpoint, opponent progression, settings
and results. You can run the same curriculum more than once in separate directories.

For a ready-made campaign with groups of teams, use the
[generalization campaign](generalization-campaign.md): ten stages, 240 training
teams, 120 validation teams, and a separate final test set.

## Directory-based stages

Each stage accepts either one `team` file or a `team_dir`. A directory adds all
of its `*.json` team files recursively, in sorted order. It must be nonempty and
contain valid team JSONs. Subdirectories organize teams within a stage; they do
not create additional stages. Keep manifests and other JSON metadata outside
the directories selected for teams.

For example, a stage in the generalization manifest is:

```json
{
  "stage": 1,
  "team_dir": "stages/01_basic_types/train",
  "validation_dir": "stages/01_basic_types/validation",
  "focus": "Learn varied mono-type matchups."
}
```

Paths are relative to the manifest. Earlier training groups remain in later
stages, and both players sample uniformly from the cumulative files. At the top
level, `next_team_dir` selects a final expanded-evaluation probe directory instead
of the single file specified by `next_team`.

`validation_dir` (or a single `validation_team`) is optional. Supply it for every
stage or none. When present, all promotion evaluations use cumulative validation
groups instead of training groups, including mixed-opponent and battle-budget
evaluations. These files never enter training. The loader rejects duplicate
training/validation parties even when their names or ordering differ. Without
validation groups, the original evaluation behavior is preserved.

Files and group membership are copied into the run at initialization. Source
changes only affect new campaign runs. The [generalization guide](generalization-campaign.md)
covers final holdouts, generation and flags. The walkthrough below shows the
smaller, still-supported one-file-per-stage form.

This walkthrough creates a three-stage curriculum using existing team templates.
It needs four team files: three additions and a final evaluation-only team.

## 1. Create a curriculum directory

Run these commands from the project root with your virtual environment active.
Use a new directory name if `my_campaign` already exists:

```bash
mkdir -p config/curricula/my_campaign/teams

cp config/curricula/type_progression/teams/01_six_types.json \
  config/curricula/my_campaign/teams/team_a.json
cp config/curricula/type_progression/teams/02_water_lead_reordered.json \
  config/curricula/my_campaign/teams/team_b.json
cp config/curricula/type_progression/teams/03_familiar_types_new_species.json \
  config/curricula/my_campaign/teams/team_c.json
cp config/curricula/type_progression/teams/04_familiar_species_coverage.json \
  config/curricula/my_campaign/teams/team_d.json
```

You may replace these with other team JSONs. Each file represents a party of
one to six Pokémon. The curriculum loader requires exactly four moves per Pokémon.
Use the [team JSON guide](team-generation.md) for species, moves, nature, held
items, EVs and IVs. Keep early additions modest if the goal is to measure transfer
from one stage to the next.

## 2. Write the stage manifest

Copy and paste:

```bash
cat > config/curricula/my_campaign/curriculum.json <<'JSON'
{
  "version": 1,
  "name": "my_campaign",
  "stages": [
    {
      "stage": 1,
      "team": "teams/team_a.json",
      "focus": "Learn the starting six-type team."
    },
    {
      "stage": 2,
      "team": "teams/team_b.json",
      "focus": "Handle different party positions and move slots."
    },
    {
      "stage": 3,
      "team": "teams/team_c.json",
      "focus": "Transfer to new species with familiar types."
    }
  ],
  "next_team": "teams/team_d.json"
}
JSON
```

The runner constructs these pools automatically:

| Stage | Training pool | Expanded evaluation pool |
|---:|---|---|
| 1 | A | A + B |
| 2 | A + B | A + B + C |
| 3 | A + B + C | A + B + C + D |

Both players independently draw from the training pool. The expanded evaluation
devotes half its battles to matchups involving the newly added team. See
[evaluation composition](curriculum.md#what-happens-in-a-stage) for details.

| Manifest field | Meaning |
|---|---|
| `version` | Use `1`. |
| `name` | A descriptive curriculum name. |
| `opponent_mode` | Optional: `fixed` (default) or `previous-stage`. Selects one reference for the run or the previous stage's promoted candidate for each later stage. |
| `opponent_switch` | Optional: `{"chance": 0.3, "cooldown": 2}` adds probabilistic switching pressure to the training opponent. Defaults to disabled; evaluation remains unmodified. See [switching pressure](switch-pressure.md). |
| `stages` | Ordered list of stage objects; any positive number of stages is supported. |
| `stage` | Consecutive stage number starting at 1. |
| `team` / `team_dir` | Exactly one: a team file or recursive directory introduced at this stage. Paths are relative to the manifest. |
| `validation_team` / `validation_dir` | Optional file or recursive directory used only for promotion evaluation. Specify one for every stage, or omit for all stages. |
| `focus` | Description printed when the stage starts. |
| `next_team` / `next_team_dir` | Exactly one: final probe file or directory. Used in the last expanded evaluation in the original mode; battle-budget mode needs a separate final evaluation command. |

For N stages, supply N stage additions plus a distinct final probe. Previous teams
stay in every later pool. With `team`, adding an unlisted file has no effect;
with `team_dir`, it joins that stage in the next new run. The manifest determines
stage order. The final probe never enters training in this campaign.

To vary opponents between stages, add `"opponent_mode": "previous-stage"` at
the top level, beside `name`, or pass `--opponent-mode previous-stage` when
starting the campaign. Stage 1 uses the supplied initial opponent. Later stages
use frozen copies of the previous stage's promoted model, including across
retries. See the [opponent progression walkthrough](curriculum-opponents.md) for
a ready-made ten-stage preset, commands and how to interpret the new baselines.

## 3. Choose promotion rules

Copy the existing template:

```bash
cp config/curricula/type_progression/promotion.json \
  config/curricula/my_campaign/promotion.json
```

It contains:

```json
{
  "current_win_rate": 0.60,
  "expanded_win_rate": 0.55,
  "incoming_win_rate": 0.50,
  "max_regression": 0.03,
  "max_truncation_rate": 0.01
}
```

All five fields are required. Values are fractions between 0 and 1. In order,
they require 60% current-pool wins, 55% expanded-suite wins, 50% wins in incoming
matchups, at most three percentage points of regression from the stage baseline,
and at most 1% truncations. See [promotion checks](curriculum.md#promotion-checks)
for denominators and scope.

The runner uses built-in thresholds unless you explicitly pass
`--promotion-rules`. Merely editing a `promotion.json` beside the manifest does
not load it. These thresholds guide progression; they are not confidence tests.

## 4. Check the files before training

This checks the manifest, party format, four-move requirement and rule fields
without starting a simulation:

```bash
python - <<'PY'
from pathlib import Path
from firered_rl.curriculum import load_curriculum, read_json, validate_rules

folder = Path("config/curricula/my_campaign")
curriculum, teams = load_curriculum(folder / "curriculum.json")
validate_rules(read_json(folder / "promotion.json"))
print(f"Ready: {len(curriculum['stages'])} stages, {len(teams)} team files")
PY
```

This structural check does not certify species learnsets or tournament legality.
The copied examples have already been checked against the project's catalog.

## 5. Start the campaign

Use a new output directory for a new campaign:

```bash
frozen_checkpoint=".local/run/ppo-20260913-151530-703645/checkpoints/latest.zip"
campaign_run=".local/run/my-campaign-01"

python mgba/run_curriculum.py \
  --run-dir "$campaign_run" \
  --curriculum config/curricula/my_campaign/curriculum.json \
  --promotion-rules config/curricula/my_campaign/promotion.json \
  --resume latest \
  --opponent-checkpoint "$frozen_checkpoint" \
  --environments 48 \
  --showdown-workers process \
  --rollout-steps 128 \
  --batch-size 128 \
  --timesteps 100000 \
  --eval-episodes 1000 \
  --stage 1
```

The candidate selected by `latest` and the reference are copied once. The
campaign also snapshots its teams and rewards. With the default `fixed` mode,
the opponent stays frozen across all stages. In `previous-stage` mode, it changes
only after a stage is promoted; it stays fixed during retries. Both evaluations after training use that attempt's
exact final checkpoint, even if another training process updates a different
`latest.zip`.

On the first attempt, expect two 1,000-battle baseline evaluations, training,
then two more 1,000-battle evaluations. That is **4,000 evaluation battles plus
training**, not 1,000 total. See [phase names and output paths](curriculum.md#why-does-evaluation-start-counting-from-1-again).

## 6. Continue or retry

Read the result:

```bash
cat "$campaign_run/stage-01/report.md"
```

After `PROMOTE`, run stage 2:

```bash
python mgba/run_curriculum.py --run-dir "$campaign_run" --stage 2
```

After stage 2 passes, run stage 3 with `--stage 3`. Or automatically continue
from the first unpromoted stage through stage 3:

```bash
python mgba/run_curriculum.py --run-dir "$campaign_run" --through-stage 3
```

Automatic advancement stops on the first `HOLD`. To retry a held stage, repeat
that stage's command. For example, if stage 2 is held:

```bash
python mgba/run_curriculum.py --run-dir "$campaign_run" --stage 2 --timesteps 50000
```

This creates another attempt, continues from its last candidate, and reuses
the original baseline. It runs two after-training evaluations, for 2,000
evaluation battles at the default setting. Attempts are not automatically
repeated until a stage passes.

Omit initialization flags when continuing: `--resume`, `--opponent-checkpoint`, `--opponent-mode`,
`--curriculum`, `--promotion-rules`, `--reward-config`, `--eval-episodes`,
`--max-steps` and `--seed`. They are pinned for this campaign. Execution settings
such as `--timesteps`, environment count and PPO overrides can be changed and
are remembered on subsequent invocations.

## 7. Change the curriculum or start another experiment

Edit your source JSONs, then choose a new `--run-dir` and run the initialization
command again. Use an explicit checkpoint from an earlier stage report as
`--resume` if you want to retain that model's progress. Existing campaigns use
their snapshots, so source edits do not alter them mid-run.

To extend to ten stages, add stage objects 4 through 10, each with its added
team, and point `next_team` at an eleventh file. You can also copy the provided
ten-stage manifest instead of writing one from scratch.

Retain a separate evaluation suite for broader comparisons. A team used in
promotion decisions is validation data, even before it enters training; passing
the campaign does not establish performance against every team or opponent.
