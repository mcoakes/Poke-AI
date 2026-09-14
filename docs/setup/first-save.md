# Create the initial FireRed save

[Quickstart](../quickstart.md#2-set-up-the-initial-save-file) · [Dependencies](README.md) · [Build counter states](../automatic_states.md)

This is the one manual preparation step for the mGBA pipeline. Use your verified
FireRed v1.0 ROM and the [project-local mGBA build](mgba.md).

```bash
mkdir -p states/base states/counter
./.local/mgba/bin/mgba-qt FireRed.gba
```

1. Start a game, receive a starter and finish the first rival battle.
2. Once you can walk in the overworld with a healthy Pokémon, use the game's
   **SAVE** menu. Also save an emulator state to `states/base/overworld.ss1` as
   an early recovery checkpoint.
3. Collect Oak's Parcel from the Viridian Poké Mart and deliver it to Professor
   Oak in Pallet Town. Finish the dialogue.
4. Visit the upstairs Pokémon Center **Direct Corner**, the far-right wired-link
   attendant. Verify that the link service is offered, then leave the dialogue.
5. Save in-game again and close mGBA. Preserve the battery save:

```bash
cp --backup=numbered FireRed.sav states/base/overworld.sav
stat -c '%n: %s bytes' states/base/overworld.sav
```

Expected size: **131072 bytes**.

If mGBA stores battery saves elsewhere, use its configured save directory for
the source of that copy.

| File | Meaning | Used by |
|---|---|---|
| `.sav` | Raw 131,072-byte battery save from the game's SAVE menu | PKHeX and the team builder |
| `.ss1` | Emulator snapshot, including running game state | mGBA startup and reset |

Changing a filename does not convert between these formats. The `.sav` used for
team generation must include the parcel delivery; the early recovery `.ss1` does
not. The editor grants badges and sets the counter warp, but does not perform
Oak's Parcel quest.

Next, run [automatic state generation](../automatic_states.md). It creates each
party, boots Continue and saves a verified snapshot at the wired-link counter.
You do not need to manually position and save every generated team.
