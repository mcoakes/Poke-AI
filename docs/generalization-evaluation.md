# Evaluate new team configurations

[Evaluation](evaluation.md) · [Periodic validation](periodic-evaluation.md) · [All flags](reference/evaluate-generalization.md)

`evaluate_generalization.py` checks new teams for overlap, snapshots two runs'
best models and their frozen opponents, evaluates both on one fixed schedule,
and writes a paired comparison with 95% uncertainty intervals. Neither model
trains. This requires Showdown and two periodic-validation runs with the same
saved Random/Dumb/Smart opponent suite and reward settings.

## Create a separate test pool

Skip this if you already generated the directory. No save files, emulator or
.NET are needed.

```bash
python mgba/generate_teams.py \
  --count 300 --team-size 3 --level 50 --seed 26092702 \
  --competitive-rate 0.6 --chaos-rate 0.2 --competitive-item-rate 0.8 \
  --output-dir .local/test-teams/generalization-01
```

Style percentages apply to individual Pokemon sets. The remaining 20% use
learnset-based sets. This is not the same distribution as equally sampling
competitive/casual/random training categories; record it as a transfer test.

## Compare the models

```bash
python mgba/evaluate_generalization.py \
  --previous-run .local/run/ppo-observation-v2-01 \
  --candidate-run .local/run/ppo-observation-v2-lr0001-01 \
  --team-dir .local/test-teams/generalization-01 \
  --output-dir .local/run/generalization-v2-01 \
  --environments 48 --showdown-workers process
```

Add `--dry-run` to validate inputs and count accepted/excluded teams without
writing files or starting battles. Defaults run **6,000 battles total**: 1,000
per opponent category for each model, including both player seats. Smart battles
are shared across saved Smart models, not 1,000 per Smart model. Team pairs are
sampled uniformly from accepted teams, including possible mirror matches. Battle
limits come from the existing validation suite.

The command reads each run's `checkpoints/best.zip`, verifies `best.json`, and
copies the exact weights before evaluation. Missing or changed recorded training
files stop the overlap check. Existing output directories are never overwritten
or resumed; after interruption, inspect partial results and retry with a new
output directory. Neither source run is modified.

## Overlap checks and scope

Teams matching either run's recorded training inputs or embedded validation
teams are excluded. Labels, party order and move order are ignored; defaults,
name case and spaces are normalized. Duplicate test teams are also excluded.
No remaining teams is an error. Input team files are never edited.

Saved checkpoints do not contain their ancestors' complete training history.
Use repeatable `--exclude-team-dir DIRECTORY` and `--exclude-suite FILE` options
for other known training/evaluation pools. Team directories are recursive and
every JSON must be a team; generation-summary.txt is ignored. `team-check.json`
records accepted teams, exclusions and checked sources. Distinct EVs, items or
moves count as distinct configurations; this does not measure team similarity.
Individual species and curated sets may remain familiar.

Keep these teams and the resulting suite out of training and periodic model
selection. Repeatedly tuning against this report makes it development data;
reserve another untouched pool for a final generalization claim.

## Read the result

```bash
cat .local/run/generalization-v2-01/baseline-report.md
```

`previous` is the reference; `candidate` is the new model. The report includes
overall/per-opponent win rates, paired changes, battle lengths and seat results.
An improvement interval entirely above zero supports a gain on the sampled test
configurations; an interval crossing zero is inconclusive. These pointwise
intervals do not establish that a method wins across independent training runs.

Other outputs are `baseline-report.json`, `team-check.json`,
`comparison-config.json`, `suite.json`, pinned `inputs/`, and the two evaluation
directories (`previous/`, `candidate/`), each retaining its tested `candidate.zip`.
Spectator logs rotate at 2 MiB with one backup per file by default. Verbose
observation and per-step traces are disabled.
