# Quickstart teams

These complete JSONs are used by the [end-to-end quickstart](../../../docs/quickstart.md).
They reproduce the existing three-starter sets with distinct filenames so their
counter states are easy to identify.

- `quickstart_charizard.json`: Charizard / Blastoise / Venusaur.
- `quickstart_venusaur.json`: Venusaur / Blastoise / Charizard.

All are level 50 with four moves, Leftovers, 31 IVs and a 510-total EV spread.
The first party member leads; array order defines move slots. Edit a copy to
experiment. The [team schema](../../../docs/team-generation.md#json-format)
explains every field. Use JSONs directly in Showdown, or build them into `.sav`
and `.ss1` files with the quickstart's `build_states.py` command.
