# Audit observations, actions and PPO wiring

[Baseline](baseline.md) · [Observations and actions](turn-environment.md) · [All flags](reference/audit-training.md)

Keep the [Stage 1 baseline](baseline.md), including its reports and frozen model
snapshots, before changing training. Stage 2 checks whether simulator information
reaches the policy correctly and whether the real training interface handles
legal actions and episode boundaries. It does not train a model, change rewards,
or decide how intelligent a saved policy is.

Choose `--observation-version v2` to audit the richer
[Showdown schema](observation-v2.md). With a checkpoint, the default is its
saved schema; without one, the default remains v1. Explicit conflicts between
the flag and checkpoint are rejected.

## Run the audit

Use the project root with the Python environment active and the local
[Showdown dependencies](setup/showdown.md) installed. Complete checks require
the Python training extras (`python -m pip install -e '.[train]'`). No ROM,
emulator, save files or GPU are needed.

```bash
audit_run=".local/run/audit-generalization-01"

python mgba/audit_training.py --output-dir "$audit_run"
```

Choose a new, empty output directory outside the baseline and campaign
directories. An audit never resumes or overwrites an earlier audit; use a new
directory for each run. Without `--checkpoint`, inference checks use a fresh,
untrained, disposable policy. They test the interface only; the policy is never
trained or saved.

To check a saved model's compatibility with the same interface:

```bash
python mgba/audit_training.py \
  --output-dir .local/run/audit-latest-01 \
  --checkpoint latest
```

`latest` uses the project's **global** checkpoint discovery; `later` is an alias.
It is resolved once at launch. Check the printed resolved path and SHA-256 to
confirm the selected model. The report records checkpoint provenance. Global
`latest` is not necessarily the Latest candidate from your Stage 1 baseline.

To audit that exact baseline candidate, read its saved snapshot path from
`baseline.json`, rather than guessing a snapshot filename:

```bash
baseline_run=".local/run/baseline-generalization-01"
baseline_checkpoint="$(python - "$baseline_run/baseline.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    baseline = json.load(handle)
print(baseline["candidates"]["latest"]["path"])
PY
)"

python mgba/audit_training.py \
  --output-dir .local/run/audit-baseline-latest-01 \
  --checkpoint "$baseline_checkpoint"
```

Use `starting` or `best` in place of `latest` in the JSON lookup to select those
baseline roles. The checkpoint is loaded read-only: the audit checks the
observation/action contract, finite probabilities on valid actions and unchanged
weights. It does not call learning or save a replacement checkpoint.

## Read the findings

```bash
cat "$audit_run/audit-report.md"
```

`audit-report.md` starts with the result and correctness failures, then the
information inventory and check table. `audit-report.json` holds the full
machine-readable evidence, checkpoint provenance and inference probabilities.
Read its three parts separately:

| Part | What it establishes |
|---|---|
| Information inventory | Whether relevant battle information is **present**, **partial** or **missing** in current policy inputs. Missing information is a documented limitation, not automatically a correctness failure. |
| Controlled simulator cases | Whether production observations and legal-action masks agree with independent expectations derived from native Showdown state at the tested decision boundaries. |
| PPO interface checks | Whether the actual Showdown-backed `LinkedVecEnv` and policy encoding preserve masks, handle forced replacements, terminal results, truncation and reset, and produce valid masked policy probabilities. |

The controlled cases run local Showdown through an audit companion process. Its
inspection reads native engine state without changing it; expected values are
checked against the actual production observation encoder and action mapping.
The audit uses JSON team fixtures, not ROM or emulator fixtures.

For example, opponent species and moves are available in the current observation,
while opponent bench HP and held items are not. Complete stat stages and weather
are not encoded either. A missing field can explain why a policy cannot respond
to some battle situations even when all encoding checks pass. See the inventory
for the distinction between absent information and information encoded
incorrectly.

Version **0.32.1** fixes the two correctness defects found by the initial audit.
When a four-move Mew uses Transform on a Magikarp that has only Splash, the
own-active observation now clears the removed move slots. Both the synthetic
encoding probe and the live Transform case are expected to pass. The adapter
also reads Showdown's remaining confusion turns from `time`, so the native-state
comparisons on both player sides report the expected `2` turns in the default
fixtures. These are four checks covering two defects; they do not establish the
cause of weak generalization or any particular checkpoint's performance.

The default v1 audit is expected to report **51 passes, zero failed checks and
13 information limitations**, across the same 64 checks, and exit with code `0`.
The fixes do not expand the observation schema or change the action contract.
Existing checkpoint shapes and saved weights remain compatible, and no mGBA
rebuild is needed.

For v2, the headline summarizes the v2-specific checks. The unchanged v1 suite
also runs as separately labeled regression evidence: its 13 legacy information
limitations are not counted as missing v2 inputs. A failure in either scope
makes the overall audit fail. V2 checks include native stats/stages, both parties'
state, items and abilities, timed and permanent weather, move details, supported
volatile bits, feature sensitivity and actual vector/PPO episode boundaries.

```bash
python mgba/audit_training.py \
  --observation-version v2 \
  --output-dir .local/run/audit-observation-v2-01 \
  --log-max-mb 2 --log-backups 1
```

V2 evidence is under `controlled-v2/` and `pipeline-v2/`; the legacy regression
logs are under `legacy-v1/`. The report explains remaining representation limits,
including contextual move effects and unsupported field/volatile state. It does
not claim every battle mechanic is represented.

Rerun the audit in a new directory to compare with the original findings:

```bash
python mgba/audit_training.py \
  --output-dir .local/run/audit-generalization-02
```

Choose another unused name if that directory already exists; the audit refuses
to reuse existing output directories.

Passing the checks provides evidence for the covered cases. It does not prove
every Pokémon mechanic, mGBA equivalence, learning quality or checkpoint skill.
Use Stage 1 comparisons to measure performance; use this audit to identify
interface defects and observation limitations before another long campaign.
The audit leaves the observation schema, rewards and training defaults intact.

## Keep useful diagnostics

Diagnostics remain enabled with rotating logs bounded to **2 MiB per file and
one rotated backup** by default. Full per-step logs are off. Failure details are
retained with the audit outputs so a failed check can be investigated. These are
per-file bounds, not a total size cap on the report and its evidence. Controlled
case logs are `controlled/audit-diagnostics.jsonl` and `controlled/battle.log`.
Real environment logs are under `pipeline/terminal/`, `pipeline/replacement/`
and `pipeline/truncation/`. The exclusive `audit-owner.json` file reserves the
output directory against concurrent audit writers. No diagnostic checkpoint
artifact is created.

For another run with a larger diagnostic window:

```bash
python mgba/audit_training.py \
  --output-dir .local/run/audit-generalization-03 \
  --log-max-mb 4 --log-backups 2
```

Keep the report and failure evidence when discussing a mismatch. Fix a confirmed
defect, then run a new audit directory and compare its findings before changing
the training experiment. See the [CLI reference](reference/audit-training.md)
for all options and exit codes. A completed run with correctness failures exits
with code `1`; information limitations alone do not make the audit fail.
