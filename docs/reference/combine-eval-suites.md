# combine_eval_suites.py

[Walkthrough and commands](../combined-validation.md) · [All utilities](README.md)

Combine existing pinned evaluation schedules for `evaluate_ppo.py --suite` and
`train_ppo.py --eval-suite`. No candidate checkpoint or simulator is needed.
Use the Python environment with training dependencies installed.

```bash
python mgba/combine_eval_suites.py \
  --suites .local/run/diversity-v2-01/suites/validation.json \
           .local/run/eval-rules-v1-01/suite.json \
  --output-dir .local/run/validation-rules-and-agents-01
```

| Flag | Default | Meaning |
|---|---|---|
| `--suites PATH [PATH ...]` | Required | At least two distinct source files, in output order. Each must be a side-swapped Showdown suite with pinned opponents or a pinned standalone rule opponent. |
| `--output-dir PATH` | Required | New directory; existing paths are rejected. Contains `suite.json`, `manifest.json`, and content-addressed `opponents/*.zip` copies. |
| `-h`, `--help` | — | Show usage and exit. |

Sources must share `max_steps`. Team data is embedded. Opponent hashes and rule
implementation are verified before creating output. Standalone checkpoint or
Random suites without pinned pool metadata are rejected: the helper cannot
infer which original opponent was used. Checkpoint ZIP structure is checked
when evaluation loads it; preparation verifies its bytes against the saved hash.

Case order, teams, battle seeds, seats and opponent seeds are retained. Episode
and matchup numbers become unique in the combined schedule; opponent IDs are
prefixed by source number. Opponent groups retain their original names. A
standalone deterministic rule suite becomes a Rules pool entry. Its otherwise
unused opponent RNG seed matches the evaluator's default. Original suite bytes
are never changed. Bundled checkpoints allow replay after source files move.

Counts are added, not rebalanced. Overlapping cases across different sources
remain duplicated; use disjoint schedules if independent cases are required.
The manifest records each source hash and its new episode range. The combined
suite has its own hash. Exit status: 0 on success, 2 for invalid inputs.
