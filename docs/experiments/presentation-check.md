# Optional fresh-team check and demonstration

This optional runner performs frozen evaluation only. The active plan is now
[broader training coverage and student play](training-coverage.md);
training does not need to stop for the event. The bench-order experiment did not
meet the practical improvement target. Its selected model is the control checkpoint
at 852,992 total steps, chosen by 45.2% balanced validation. The final shuffled
model's 42.1% heldout mean remains a separately labeled secondary result.

## One final evaluation

This runner selects and copies the already chosen model before generating any
new teams or seeing their results. It then checks that model, a greedy damage
baseline, and Rules against Rules on the same frozen schedules. There is no
training or model reselection.

The new pool has 64 three-Pokémon compositions. It uses the same stage-4 species
pool and move/build recipe as the preceding experiment, while excluding every
composition and semantic team signature present in the source training,
validation, heldout, seen and mirror inputs. Species remain familiar: this tests
new combinations, not unseen species or unrestricted competitive teams.

There are 1,000 battles in each order, with both candidate seats, for each of
three policies: 6,000 battles total. The reversed condition changes only the
candidate bench; the lead and opponent order stay fixed. The two orders are
correlated checks, not 2,000 fully independent samples.

From the project root after extracting the update:

```bash
source .venv/bin/activate
python run_presentation_check.py \
  --source-run .local/run/bench-order-ablation-01 \
  --output-dir .local/run/presentation-check-01 \
  --jobs 2
```

Allow around 5–15 minutes, based on preceding frozen-check throughput. Watch the
launcher for progress; the actual time depends on startup and machine load.

In another terminal, watch the saved phase:

```bash
watch -n 5 cat .local/run/presentation-check-01/status.json
```

Once check logs appear, watch them:

```bash
tail -n 20 -F .local/run/presentation-check-01/console/*.log
```

Watch memory separately:

```bash
watch -n 2 free -h
```

For completed-check counts and current workers:

```bash
watch -n 5 cat .local/run/presentation-check-01/progress.json
```

This is frozen evaluation, so there is no training `evaluations.md`. Use the
launcher, per-check logs, and the completed report. The runner records memory
and preserves partial checks if interrupted. Continue with:

```bash
python run_presentation_check.py \
  --output-dir .local/run/presentation-check-01 --continue --jobs 2
```

Use the same phase/log/memory viewers above after continuing. Do not edit a
partial summary to mark it successful.

## Saved models and teams

The run saves three model copies under `models/` inside its output directory:

| Filename | Meaning |
|---|---|
| `selected.zip` | The validation-selected result, chosen before the fresh check |
| `control-final.zip` | Ordinary continuation at the fixed final budget |
| `bench-shuffle-final.zip` | Shuffled continuation at the fixed final budget |

Recorded hashes are checked against the original checkpoint files when the
runner prepares them. The original 400k reference stays intact.

The familiar training teams are copied into `demo-teams/`. To make those teams
available in the GUI, copy them into a new named category:

```bash
mkdir -p config/teams/presentation-20261019
cp -n .local/run/presentation-check-01/demo-teams/*.json \
  config/teams/presentation-20261019/
```

View a complete team definition with:

```bash
python -m json.tool \
  .local/run/presentation-check-01/demo-teams/team-0001.json
```

Inspect new evaluation teams without changing them:

```bash
python run_presentation_check.py \
  --output-dir .local/run/presentation-check-01 \
  --inspect-teams fresh --limit 3 --full-json
```

Do not substitute generated fresh-check teams into the familiar-team demo while
describing it as the training task. The GUI also allows custom teams; label
those games as demonstrations, not quantitative evaluations.

## Rehearse the actual game today

Use the already working ROM, battery save and graphical environment:

```bash
python mgba/play_gui.py \
  --rom "$PWD/FireRed.gba" \
  --source "$PWD/FireRed.sav" \
  --checkpoint .local/run/presentation-check-01/models/selected.zip \
  --workers 2 --speed 1
```

Open `http://127.0.0.1:8080` if the browser does not open. Select two distinct
teams from `teams/presentation-20261019`, keep their three members and two moves,
and leave AI action sampling off. Review both parties and start the match. The
human plays in the Player 1 mGBA window; the browser is the setup interface.

Play at least two games, including one after restarting the GUI. Record a short
clip showing startup, a meaningful move decision and a result as a fallback
for display or emulator trouble during the presentation. A human game does not
measure the same win rate as the Rules benchmark.

If the graphical setup is not already working, follow `docs/play-gui.md` and
`docs/setup/mgba.md`; do not spend all remaining preparation time rebuilding the
environment. Save logs from a failed launch and use recorded evidence while
diagnosing it. This update was checked with Showdown locally; the actual ROM,
battery save and desktop need rehearsal on the presentation machine.

## Finish the presentation

Use `docs/experiments/presentation-brief.md` for the narrative and
`docs/experiments/figures/bench-order-result.png` or `.svg` for the latest plot.
Add the fresh check's generated results after it completes. Report the new
result honestly even if it is lower; do not choose a different saved model by
testing repeatedly on this pool.

Upload `.local/run/presentation-check-01/presentation-results.zip` after the
check finishes. Model weights stay in the run's `models/` directory; the upload
contains the results, provenance, teams, and decision evidence.

Keep the training result and the fresh check distinct. The current finding is
that the agent learned narrow tasks and improved with explicit matchup inputs,
but still has limited generalization. Bench shuffling changed behavior without
establishing a useful win-rate gain.

Reserve the remaining hour for the explanation, timing, and the fallback video.

## Verification scope

`presentation-check-verification.json` records 27 focused tests and a real
two-job Showdown smoke with six checks and 48 battles, plus verification of
64 fresh compositions against the actual uploaded source pools. The local
smoke encountered incomplete saved evaluation files; strict gates rejected
them and normal continuation preserved and retried them. A subsequent completed
continuation left all frozen inputs, models, checks and logs unchanged.

These are operational checks with tiny local models. The user model's new
scores and the live graphical demonstration must be checked on the user's
machine; no fresh-check score is claimed in this package.
