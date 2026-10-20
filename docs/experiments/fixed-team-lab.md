# Fixed-team learnability lab

Use this diagnostic before more small learning-rate/opponent-mix experiments.
It asks whether the production PPO pipeline can learn a small, clearly defined
problem. It does not establish generalization to arbitrary teams.

## What runs

| Task | Candidate team | Rules team | Additional transitions per run |
|---|---|---|---:|
| Move choice | Starmie: Thunderbolt, Swift, Surf, Harden | Golem: Earthquake, RockThrow, Tackle, DefenseCurl | 50,000 |
| Mirror | Typhlosion, Blastoise, Sceptile | The identical team in the identical order | 200,000 |

The mirror team uses Strength plus Flamethrower/Surf/LeafBlade respectively.
All Pokemon are level 50, Hardy, have 31 IVs and zero EVs, and hold no items.
The mirror task has mutual Fire/Water/Grass advantages, forced replacements,
and voluntary switching. It retains normal simulator damage rolls, speed ties,
critical hits, abilities, and move effects. It is deliberately smaller than 6v6.

Each task runs once with fresh shared-moves/v2 weights and once from the preserved
400k checkpoint (including its optimizer state, as in normal --resume training).
The seed is 261030 by default; these four branches are diagnostic cases, not four
independent estimates of a strategy. They run two at a time, with 16 player
environments per trainer (8 learning players). Both use learning rate 0.0001,
entropy 0.01, rollout 128, batch 128, four epochs, and the supplied existing reward
configuration. Rules is unchanged. The trainer now supports optional validation-based early stopping. The previous random-team
experiments used more environments, so this is not a matched hyperparameter study.

Before training, the runner measures random legal actions, a greedy damage policy,
and Rules versus Rules through the real production simulator and observations.
Greedy attacks with the largest estimated damage and uses Rules for forced
replacements; it never voluntarily switches. This is a measured achievable
baseline, not proof of optimal play. Move-choice calibration must reach >=95%
and exceed random by >=20 points, otherwise training stops for inspection.

On a local 200-battle calibration, random/greedy/Rules win rates were respectively
52%/100%/100% for move choice and 2%/51%/50% for the mirror. These numbers describe
that schedule; the script recalibrates on the installed project.

## Early stopping

New batches stop a task after two consecutive, distinct validation checkpoints
reach 99% wins for move choice or 95% for mirror, with no truncations. Initial
validation counts once. Listed transition budgets are maxima. Final frozen
checks still run. Use `--no-early-stop` on a new batch to retain full budgets.
Existing saved plans retain their original behavior when continued.

For the next experiment after the observed mirror collapse, use the
[staged team lab](staged-team-lab.md), which deliberately disables success
stopping for its controlled stability comparison.

## Run and watch

Activate the normal training venv from the project root. Install the updated
package into that existing project first: it adds `run_fixed_team_lab.py` and
`src/firered_rl/fixed_team_lab.py`. Editable installations pick up the new module;
the runner also adds the project source directory to its own import path.

```bash
python run_fixed_team_lab.py \
  --checkpoint models/observation-v2-shared-moves-400k-reference-01.zip \
  --reward-config .local/run/diversity-v2-01/inputs/reward.json \
  --output-dir .local/run/fixed-team-lab-01 \
  --jobs 2
```

If another training batch is running, stop it gracefully first so concurrency
stays at two. The runner copies the reference and reward into its own inputs;
it preserves the working reference. It refuses an existing output directory.

Always include these viewers when sharing a start/resume command. In a separate
terminal, watch job output (all queued logs exist from the beginning):

```bash
tail -n 20 -F .local/run/fixed-team-lab-01/console/*.log
```

Watch validation tables:

```bash
watch -n 5 'for f in .local/run/fixed-team-lab-01/*/evaluations.md; do
  [ -f "$f" ] || continue
  echo "$f"
  cat "$f"
done'
```

Record memory through training and validation:

```bash
vmstat -w -t 2 | tee .local/run/fixed-team-lab-01/memory.log
```

Calibration and final-check progress prints in the launcher terminal. During
these phases, training console logs need not advance. The launcher automatically
evaluates best and final checkpoints on 1,000 new battle seeds/seat assignments,
then on the same schedule with candidate move slots rotated. No weights update
during these checks. Validation uses 256 battles every 25,000 transitions and
on initial/final checkpoints. A rollout can overshoot the requested budget.

## Pause and recover

Ctrl+C in a viewer only stops that viewer. Ctrl+C once in the launcher stops
active trainers gracefully and cancels queued launches; wait for cleanup. Progress
after the latest saved checkpoint can be lost. Resume saved budgets and inputs:

```bash
python run_fixed_team_lab.py --output-dir .local/run/fixed-team-lab-01 --continue --jobs 2
```

Use the same `tail`, `watch`, and `vmstat` commands above when resuming. Completed
trainers are skipped; final checks resume where completed. If interrupted before
the first checkpoint, that trainer cannot recover. Incomplete calibrations and
final checks are retained; continuation reruns them in new retry directories.
An interrupted final validation at the timestep target is
retried with one extra rollout, so a run is not silently marked successful.

`--calibrate-only` prepares inputs and measures baselines; `--continue` then trains.
`--fresh-only` omits reference branches. Continuation restores saved settings;
only the concurrency flag affects training. No change to the saved reward/teams
is accepted during continuation.

## Read the result

Upload `.local/run/fixed-team-lab-01/fixed-team-results.zip`; read `report.md`
locally. The archive contains inputs, calibration, training histories, and test
summaries/episode rows, PPO update diagnostics, battle blocks, switching metrics,
and PPO CSVs, but no model weights or bulky console logs. Initial
validation scores appear in each run's evaluations.md; compare them with its final
score to distinguish existing competence from new learning.
Every battle is against Rules. The existing evaluator places the rate in
`Overall`; the separate `Rules` column can show a dash because this suite has
one fixed opponent rather than a labeled opponent pool.

* Move choice: aim for >=95% wins AND >=95% first-turn Surf, including after move
  rotation. A 50% win rate fails this test: the random baseline already reached
  52%, whereas the simple correct action reached 100% in calibration.
* Mirror: first approach the measured Rules/greedy level (roughly 50% here).
  Exceeding 50% would demonstrate an exploitable weakness, but is not guaranteed
  merely because the opponent uses rules. Inspect both seats and truncations.
* A frozen policy failing after move rotation indicates representation/action
  mapping trouble, not inadequate Pokemon knowledge. This is a narrow invariance
  test; it is not testing unseen team generalization.
* If both initializations fail move choice, inspect observations, masks, action
  mapping, rewards, and optimizer updates before any more broad training.
* If move choice succeeds but mirror fails, investigate switching/replacement
  decisions and multi-turn credit assignment. Passing mirror permits adding one
  source of variety at a time; it does not prove broad battle competence.

For move choice the candidate always owns Starmie, including when seated as P2.
The script deliberately reverses the team assets for that seat. A conventional
side swap which makes the candidate play Golem half the time would erase the
designed advantage. Mirror evaluation uses ordinary side swaps. All win rates
count requested battles, including truncations as non-wins; ties keep earlier
best checkpoints. Final tests do not influence checkpoint selection.

## Verification performed for this update

Six focused experiment tests and 17 existing Rules/periodic-evaluation tests
passed. Two real fresh MaskablePPO/Showdown runs each reached 20,224 transitions
(seed 261030, 4 player environments, the bundled damage_knockout_win reward).
Move-choice validation went from 0% to 100%; mirror validation from 2% to 15%.
On 200 new evaluation battles, move choice achieved 100% wins and 100% first-turn
Surf, including rotated move slots. The mirror scored 16% with either move order.
All completed final checks had zero truncations. This is a short diagnostic,
with a smaller environment count and possibly different reward file from the
user's normal run; it is not the proposed full-budget experiment or evidence of
broad generalization. No user 400k model was available for local training.

The runner's reference-loading path was checked with a real local shared-moves
checkpoint. Continuation skipped completed trainers and recovered incomplete
calibration/final evaluation directories. Compact measured results are saved in
[fixed-team-local-verification.json](fixed-team-local-verification.json).

For the expanded fixed-to-1,024-team progression, editable teams, rehearsal, and
team inspection commands, see [Expanding team curriculum](team-curriculum.md).
