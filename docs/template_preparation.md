# Prepare a template save

[Documentation](README.md) · [All flags](reference/prepare-template.md) · [Automatic pipeline](automatic_states.md)

Use this utility when you want to edit badges/location separately. The automatic
state builder already performs these edits. Preparation copies a FireRed battery
save and preserves its party; it does not create a team or emulator state.

```bash
python mgba/prepare_template.py --source states/base/overworld.sav --rom FireRed.gba \
  --grant-eight-badges --warp-link-counter \
  --output states/templates/firered_link_counter.sav
```

Select one or both edit flags. The default output is the path shown above. A
`.sav.json` manifest is written beside it. Existing outputs require `--force`,
which creates numbered backups. Input and output must be different files.

## Source requirements

Use a raw 131,072-byte FireRed `.sav` with valid checksums, ordinary overworld
progress and a healthy non-Egg party member. Close mGBA while editing it.
Complete [Oak's Parcel and the initial save](setup/first-save.md) first; setting
badge/Pokédex flags does not complete the parcel quest. FR/LG share a save format,
so a PKHeX `FRLG` label alone does not identify the ROM revision.

## How the location edit works

The editor sets FireRed's Continue-game warp instead of replacing map coordinates
alone. On Continue, the game rebuilds the map and object state for Viridian
Pokémon Center 2F, group 5/map 5, walkable tile `(10,4)` in front of the far-right
Direct Corner counter. The middle attendant belongs to the wireless Union Room.

Badges and warp can be selected independently. Preparation also suppresses
adventure-recap playback; warp preparation sets the Pokémon/Pokédex acquisition
and attendant tutorial flags it needs. It leaves other campaign progression in
the source. PKHeX handles save-sector ordering and checksums.

## Load the generated battery save manually

Close mGBA, then copy the output to the ROM's battery-save location:

```bash
cp --backup=numbered states/templates/firered_link_counter.sav FireRed.sav
./.local/mgba/bin/mgba-qt FireRed.gba
```

Choose **Continue**. If a quick-state autoload is enabled, disable it so it does
not override the battery save. Face north with UP. You can save an emulator
snapshot here, or use [build_states.py](automatic_states.md) to do this for every
team automatically. Check mGBA's configured save directory if your ROM's battery
file lives elsewhere.

The [developer source reference](sources.md) links the save structures and map
information behind these edits.
