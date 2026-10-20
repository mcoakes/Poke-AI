# Pokémon generalization curriculum

[Stage plan, training commands, validation and final tests](../../../docs/generalization-campaign.md)

Ten stages each introduce 24 training teams and 12 held-out validation teams.
`stages/<stage>/train/` files enter the cumulative training pool;
`stages/<stage>/validation/` files enter the cumulative promotion-evaluation pool.
`holdout/` has 40 separate final test teams. Each JSON contains one party.

The preset uses the previous promoted policy as the next training opponent and
disables switching overrides. The agent retains its normal legal switch actions.

All groups are generated reproducibly by
[`tools/generate_generalization_campaign.py`](../../../tools/generate_generalization_campaign.py).
`generation.json` records the seed and species coverage. Paths in `curriculum.json`
are relative to this directory. A running campaign uses copies taken at startup;
editing these directories only affects new campaign runs.
