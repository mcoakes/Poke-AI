# Train across groups of Pokémon teams

[Documentation](README.md) · [Create your own campaign](creating-campaigns.md) ·
[Runner flags](reference/run-curriculum.md) · [Campaign manifest](../config/curricula/pokemon_generalization/curriculum.json)

The `pokemon_generalization` curriculum introduces a directory of teams at each
stage. It contains **240 training teams, 120 validation teams and 40 final test
teams**. All Pokémon are level 50 and have four distinct moves from their bundled
Gen III species learnsets. This is a battle-learning curriculum, not a tournament
tier or legality certification.

## Design and stage plan

Every stage adds 24 training teams and 12 validation teams. Earlier training teams
remain available: stage 1 trains on 24 teams, stage 2 on 48, and stage 10 on 240.
Both players independently sample the cumulative training pool, uniformly by file.
Each stage's directory contains separate `train/` and `validation/` subdirectories.
Party order and move slots vary across the supplied files; they are not shuffled
again during each battle. Training resets vary matchups and battle RNG.

| Stage | Parties in the new group | New concepts |
|---:|---|---|
| 1 | 3 Pokémon; mono-type species | Fire, Water, Grass, Electric, Ice and Fighting; same-type attacks and Normal coverage; no items |
| 2 | 3 Pokémon; broader mono-types | Immunities and more resistances, including Ground, Ghost and Normal |
| 3 | 3 Pokémon; dual types included | Combined weaknesses/resistances; an attack's type can matter more than its nominal power |
| 4 | 3–4 Pokémon | Off-type coverage, trained EV spreads and offensive natures |
| 5 | 4 Pokémon | Status moves, Lum Berry and Leftovers; every new team includes a status move |
| 6 | 4 Pokémon | Recovery, Protect, residual damage and preserving HP |
| 7 | 5 Pokémon | Setup, Substitute and disruption such as Encore, Taunt or Haze |
| 8 | 5 Pokémon | Weather, hazards, Rapid Spin, phazing and some Choice Band sets |
| 9 | 6 Pokémon | Combinations of the project's curated attacking and support sets |
| 10 | 1–6 Pokémon, four teams of each size | Broad species pool, curated and less conventional learnable moves, varied IVs and EVs |

Stages 1–9 use fully evolved species;
legendary/mythical species are excluded until stage 10. The stages emphasize the
listed concepts rather than guaranteeing a strict increase in battle difficulty.
Species' abilities and attack secondary effects still operate from stage 1; the
simulator's mechanics are never disabled to simplify a lesson.

The cumulative pool provides practice on earlier lessons. At later stages, a new
group is a smaller fraction of the pool, so a larger training budget may be needed.
The supplied 5,000 battles per attempt is an initial budget, not an estimate of how
many battles guarantee success. Check performance after each attempt.

## Switching and opponents

**The automatic switching override is disabled.** Both policies select their own
legal moves and switches. Switching receives no special reward. Existing damage,
knockout, win, loss and turn-cost rewards are unchanged.

The earlier experiment increased switching frequency without sustained improvement
against Smart opponents. That does not establish that the heuristic improves
battle skill. Diverse opposing teams and learned opponents provide the pressure
here. The optional override remains available for controlled experiments and old
campaigns, but is not used by this preset.

Start both policies from the same Smart checkpoint. Stage 1 therefore trains
against a competent frozen policy. Each later stage freezes the previous stage's
promoted learner as its opponent. The opponent remains fixed throughout retries
within a stage. Keep the original Dumb model separate for evaluation.

## Validation and final tests

Promotion uses the cumulative **validation** directories, never training teams.
The loader rejects an exact training-party copy in validation even if its team
name, party order or move order changed. It does not require every Pokémon set or
species to be new: stage validation measures transfer to different combinations
from the same lesson. All team contents are copied and hashed when the campaign
starts, so changing source directories cannot alter a running campaign.

In the command below, each attempt evaluates 2,000 battles: 600 against Random,
600 against Dumb and 800 against Smart checkpoints. Candidate seats are balanced.
Basic must win at least 75%, and Smart at least 52%; both gates must pass. These
practical gates are not statistical confidence tests. Mixed evaluation chooses a
new saved schedule per attempt. It tests historical Smart checkpoints as the
campaign advances, so raw scores across different stages are not directly comparable.

The final `holdout/` directory contains 20 remixed teams and 20 teams built only
from 15 species reserved from this campaign's training and stage-validation sets.
Reserved species may already be familiar to a resumed checkpoint from its earlier
training. This is therefore a test of this campaign's data separation, not proof
that the starting model has never seen those Pokémon.

Repeated promotion checks make validation a model-selection set. Keep the final
test results out of stage tuning. The benefit of distinct training and test
environments is supported by [Quantifying Generalization in Reinforcement Learning](https://proceedings.mlr.press/v97/cobbe19a.html);
this Pokémon curriculum is an engineering design, not a validated optimal curriculum.

## Start a separate campaign

Install [Python training extras and local Showdown](setup/showdown.md). No ROM,
save generation, .NET or mGBA rebuild is needed. Run from the project root. These
paths use the previously exported Smart v1 and the original campaign's Dumb input;
replace them with your own checkpoint paths when necessary.

```bash
initial_checkpoint="models/curriculum-smart-generations-01/smart_v1.zip"
frozen_checkpoint="$initial_checkpoint"
dumb_checkpoint=".local/run/curriculum-smart-generations-01/inputs/opponent.zip"
curriculum_run=".local/run/curriculum-generalization-01"

python mgba/run_curriculum.py \
  --run-dir "$curriculum_run" \
  --curriculum config/curricula/pokemon_generalization/curriculum.json \
  --resume "$initial_checkpoint" \
  --opponent-checkpoint "$frozen_checkpoint" \
  --opponent-switch-chance 0 \
  --train-battles 5000 --eval-episodes 2000 \
  --eval-opponents mixed --eval-dumb-checkpoint "$dumb_checkpoint" \
  --promotion-basic-win-rate 0.75 \
  --promotion-smart-win-rate 0.52 --promotion-require both \
  --through-stage 10 --attempts-per-stage 10 \
  --environments 48 --showdown-workers process \
  --rollout-steps 128 --batch-size 128 \
  --no-readable-log --no-battle-log
```

`--resume` loads the learner's starting checkpoint; the new `--run-dir` creates a
separate campaign. No old campaign is edited. This command leaves PPO learning rate,
entropy and epochs at their checkpoint values. Diagnostic battle logs are disabled
to save space; rewards, win rates, throughput, checkpoints and switch counts remain.
The final PPO update completes after reaching the battle budget, so an attempt can
finish slightly over 5,000 battles.

The original input is still exported under the historical filename `models/dumb.zip`
inside a campaign, even when a Smart checkpoint was selected as the training
reference. The explicit `--eval-dumb-checkpoint` above is what selects the true Dumb
evaluation model; its path/hash are recorded separately in `campaign.json`.

Continue a stopped or held campaign without repeating the initialization flags:

```bash
curriculum_run=".local/run/curriculum-generalization-01"
python mgba/run_curriculum.py --run-dir "$curriculum_run" \
  --through-stage 10 --attempts-per-stage 10
```

Each invocation allows up to ten additional attempts per pending stage and stops
early on promotion. A HOLD retains the checkpoint; it is not a simulator crash.

To reduce checkpoint writes, add `--latest-every-updates 10 --checkpoint-every 0`
when starting or continuing the campaign. This changes saving, not PPO learning:
the first update and successful completion still save, and evaluation/promotion
checkpoints are retained. Interrupted work resumes from the last saved update.
See [save intervals and timing](ppo-training.md#progress-and-saving).

## Watch progress and judge improvement

```bash
tail -F "$curriculum_run/evaluations.md"
cat "$curriculum_run/stage-01/report.md"
cat "$curriculum_run/stage-01/attempt-001/train/switching.json"
```

Look for stable validation wins, low truncation rates and useful switches rather
than maximizing switch frequency. Forced replacements do not count as voluntary
switches. Later-stage cumulative scores can hide weaknesses on the newest group;
use `evaluate_ppo.py --team-dir` on that stage's validation directory to inspect it
separately. Freeze the opponent and saved suite for before/after comparisons. Do
not lower a gate merely because one attempt narrowly missed it.

After stage 10 is promoted, run the final tests once:

```bash
for subset in recombinations unseen_species; do
  python mgba/evaluate_ppo.py \
    --checkpoint "$curriculum_run/models/smart_v11.zip" \
    --opponent-checkpoint "$frozen_checkpoint" \
    --team-dir "config/curricula/pokemon_generalization/holdout/$subset" \
    --episodes 2000 --seed 271828 \
    --environments 48 --showdown-workers process \
    --output-dir "$curriculum_run/final-$subset"
done
```

For an exact comparison, evaluate the entering model using the saved final suite:

```bash
python mgba/evaluate_ppo.py \
  --checkpoint "$curriculum_run/inputs/start.zip" \
  --opponent-checkpoint "$frozen_checkpoint" \
  --suite "$curriculum_run/final-recombinations/suite.json" \
  --environments 48 --showdown-workers process \
  --output-dir "$curriculum_run/initial-recombinations"
```

Repeat for `final-unseen_species/suite.json`. Inspect both summary files and both
candidate seats. A single curriculum run cannot establish robustness across RNG
seeds or all Pokémon; use independent runs and a broader [round robin](model-evaluation.md)
before selecting a model for play.

## Build a different-sized campaign

The supplied teams are ready to use. To generate another deterministic dataset,
choose a new output directory; the generator refuses to overwrite existing work:

```bash
python tools/generate_generalization_campaign.py \
  --output-dir config/curricula/generalization_more_teams \
  --seed 42 --train-count 100 --validation-count 30 --holdout-count 100
```

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir` | Required | New or empty directory for the manifest and team folders |
| `--seed` | `20260918` | Reproducible generation seed |
| `--train-count` | `24` | Training JSONs introduced at each of the ten stages; positive integer |
| `--validation-count` | `12` | Validation JSONs introduced at each stage; positive integer |
| `--holdout-count` | `40` | Final test JSONs, at least 2; split between remixed and reserved-species teams |
| `-h`, `--help` | — | Show help and exit |

The generator requires the Python training dependencies, reads the existing
catalog/curated sets and needs no network or ROM. `generation.json` records its seed,
counts and species coverage. Use its `curriculum.json` with a new run directory.

Existing one-file-per-stage manifests and saved campaigns retain their original
behavior. The final probe enters automatic evaluation only in the original
four-evaluation mode; the recommended battle-budget command above keeps it out of
promotion and uses the explicit final tests instead.
