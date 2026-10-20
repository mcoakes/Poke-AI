# Audit training observations and actions

[Utility index](README.md) · [Workflow](../training-audit.md) · [Source](../../mgba/audit_training.py)

Run the Stage 2 information inventory, controlled Showdown comparisons and real
PPO interface checks. Preserve the [Stage 1 baseline](../baseline.md) first.
This utility checks correctness and visibility of policy inputs; it does not
train, score model skill or change the observation, reward or training settings.

```bash
python mgba/audit_training.py \
  --output-dir .local/run/audit-training-01
```

Run from the project root with local [Showdown dependencies](../setup/showdown.md)
and Python training extras installed. The audit uses CPU inference and JSON
fixtures. It does not require a ROM, emulator, save files or GPU. Missing training
dependencies prevent a complete PPO check; an incomplete audit is not evidence
that the whole interface passes.

## Every flag

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show usage and exit. |
| `--output-dir` | Required | New, empty audit directory. Existing results are never resumed or overwritten. |
| `--checkpoint` | None | Optional PPO checkpoint path, `latest` or its alias `later`. Read-only compatibility and masked-inference checks; no learning or checkpoint saving. |
| `--observation-version` | Checkpoint schema, otherwise `v1` | Audit `v1` or `v2`. Explicit conflicts with a checkpoint are rejected. V2 runs its own checks plus separately labeled v1 regressions. |
| `--node` | `node` | Node.js executable for the local Showdown process. |
| `--seed` | `26002` | Integer seed in `0..2**32-1` for controlled battles and disposable-policy initialization. |
| `--command-timeout` | `30` | Positive integer wall-clock seconds allowed for a simulator command. |
| `--torch-threads` | `1` | Positive integer PyTorch CPU thread count for policy checks. |
| `--log-max-mb` | `2` | Positive, finite size limit in MiB for each rotating diagnostic log. `0` and unlimited logging are not accepted. |
| `--log-backups` | `1` | Nonnegative integer rotated copies retained per diagnostic log; `0` keeps only the current file. |

Diagnostics are on and full per-step logs are off. Failure details remain in the
audit outputs. Logging limits apply per file, not as a total output quota.

## Check a saved policy

```bash
python mgba/audit_training.py \
  --output-dir .local/run/audit-saved-policy-01 \
  --checkpoint latest \
  --log-max-mb 2 --log-backups 1
```

`latest` searches globally using the project's [checkpoint selection
rules](../checkpoints.md), then resolves once. The selected path and SHA-256 are
printed and checkpoint provenance is recorded in the report. Use an explicit
path for a specific experiment. To audit an actual baseline candidate, use its
`candidates.<role>.path` from that baseline's `baseline.json`; the
[workflow example](../training-audit.md#run-the-audit) reads that snapshot path.

With no checkpoint, the audit creates a fresh untrained policy solely to check
inference wiring. With a checkpoint, it verifies the saved observation/action
contract, valid masked probabilities and unchanged weights. Neither mode learns
or saves policy weights, and neither produces a skill score.

## Reports and scope

`audit-report.md` presents the result, correctness failures, inventory and check
table. `audit-report.json` retains full evidence, provenance and inference
probabilities. Inventory labels are `present`, `partial`
and `missing`; missing information is a limitation, not automatically a failed
correctness assertion. Controlled cases compare independent expectations from
read-only native engine inspection with production encoding and actions. PPO
checks exercise the real Showdown-backed `LinkedVecEnv`, including masks, forced
replacements, terminal observations, truncation and automatic reset.

| Output | Contents |
|---|---|
| `audit-report.md`, `audit-report.json` | Readable findings and full machine-readable evidence. |
| `controlled/audit-diagnostics.jsonl`, `controlled/battle.log` | Rotating controlled-case diagnostics and spectator battle text. |
| `pipeline/terminal/`, `pipeline/replacement/`, `pipeline/truncation/` | Rotating diagnostics from real environment boundary checks. |
| `audit-owner.json` | Exclusive ownership marker that prevents concurrent writers from sharing the audit directory. |

For v2, evidence uses `controlled-v2/` and `pipeline-v2/`; the separate v1
regression output is under `legacy-v1/`. Headline counts cover v2 checks only,
with `regression_counts` recorded separately in JSON. Failures in either scope
make the audit fail. See [observation v2](../observation-v2.md) for its inputs and
remaining scope limitations.

No policy checkpoint artifact is written. Version **0.32.1** fixes stale
own-active move slots after Transform and the remaining confusion-turn counter
on both player sides. The default v1 audit is expected to report **51 passes, zero
failed checks and 13 information limitations** across the same 64 checks, with
exit code `0`. See the [workflow guide's findings](../training-audit.md#read-the-findings)
for the covered cases and a rerun example. The observation schema and action
contract are unchanged: existing checkpoint shapes and saved weights remain
compatible, and no mGBA rebuild is required.

Use a fresh output directory after fixing a failure or changing dependencies.
There is no resume mode. Keep each report with its diagnostics. Passing covered
cases does not prove all simulator mechanics, mGBA equivalence, successful
learning or an intelligent policy.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | Completed with all correctness checks passing; information limitations may remain. |
| `1` | Completed with one or more correctness failures. Read the report and evidence. |
| `2` | Argument or startup error, such as unavailable dependencies or a nonempty output directory. |
| `130` | Interrupted. Partial evidence is retained if the audit had started; use a fresh directory for the next run. |
