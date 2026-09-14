using PKHeX.Core;
using static System.Buffers.Binary.BinaryPrimitives;

// SaveBlock-relative offsets, NOT fixed file offsets. PKHeX handles rotated sectors.
// Sources and the tested profile are recorded in docs/template_preparation.md.
public static class FireRedTemplate
{
    public const string Profile = "firered_us_v1_0";
    private const int ContinueWarpOffset = 0x0C;
    private const int SpecialWarpFlagsOffset = 0x09; // SaveBlock2
    private const byte ContinueWarpFlag = 1;
    private const int CableClubWork = 0x6F; // VAR_CABLE_CLUB_STATE - 0x4000
    private const int TealaWork = 0x7C; // VAR_MAP_SCENE_POKEMON_CENTER_TEALA - 0x4000
    private const int PokedexFlag = 0x829;
    private const int PokemonFlag = 0x828;
    private const int QuestLogOffset = 0x1300;
    private const int QuestLogSceneSize = 0x668;
    private const int QuestLogSceneCount = 4;

    public static SAV3FRLG Load(byte[] data)
    {
        if (data.Length != 131072)
            throw new ArgumentException("expected a raw 131072-byte battery .sav, not a save state");
        // FR and LG share a save format; Python also verifies the ROM SHA-1.
        if (SaveUtil.GetSaveFile(data) is not SAV3FRLG save)
            throw new ArgumentException("source is not a recognized FR/LG-format battery save");
        if (!save.ChecksumsValid)
            throw new ArgumentException("source save checksum validation failed; use a fresh in-game save");
        return save;
    }

    public static object Inspect(SAV3FRLG save) => new {
        badge_mask = save.Badges,
        badge_count = System.Numerics.BitOperations.PopCount((uint)save.Badges),
        party_size = save.PartyCount,
        pokedex_enabled = save.GetEventFlag(PokedexFlag),
        saved_position = new {
            map_group = save.Large[4], map_number = save.Large[5],
            x = ReadInt16LittleEndian(save.Large), y = ReadInt16LittleEndian(save.Large[2..])
        },
        continue_warp_enabled = (save.Small[SpecialWarpFlagsOffset] & ContinueWarpFlag) != 0,
        continue_warp = new {
            map_group = save.Large[ContinueWarpOffset],
            map_number = save.Large[ContinueWarpOffset + 1],
            warp_id = (sbyte)save.Large[ContinueWarpOffset + 2],
            x = ReadInt16LittleEndian(save.Large[(ContinueWarpOffset + 4)..]),
            y = ReadInt16LittleEndian(save.Large[(ContinueWarpOffset + 6)..])
        },
        cable_club_state = save.GetWork(CableClubWork),
        teala_tutorial_state = save.GetWork(TealaWork),
        quest_log_scenes = Enumerable.Range(0, QuestLogSceneCount)
            .Count(i => save.Large[QuestLogOffset + i * QuestLogSceneSize] != 0)
    };

    public static void Apply(SAV3FRLG save, bool grantBadges, bool warp)
    {
        if (!grantBadges && !warp)
            throw new ArgumentException("select at least one template edit");
        if (warp)
        {
            if (save.PartyCount is < 1 or > 6 ||
                !save.PartyData.Any(p => p.Species != 0 && !p.IsEgg && p.Stat_HPCurrent > 0))
                throw new ArgumentException("save in the overworld with at least one healthy Pokemon first");
            if (save.Large[4] == 0 || save.GetWork(CableClubWork) != 0)
                throw new ArgumentException("leave the active link room/session and save in the overworld first");

            // Queue the normal continue-game warp. Preserve the ORIGINAL saved
            // map header/layout/objects: they load briefly before CB2_LoadMap
            // rebuilds all objects for the destination. Changing only map/XY
            // would reload stale objects and the old tileset instead.
            var destination = save.Large.Slice(ContinueWarpOffset, 8);
            destination.Clear();
            destination[0] = 5; // group_order index: IndoorViridian
            destination[1] = 5; // ViridianCity_PokemonCenter_2F
            destination[2] = 0xFF; // WARP_ID_NONE, explicit coordinates
            WriteInt16LittleEndian(destination[4..], 10);
            WriteInt16LittleEndian(destination[6..], 4); // y=3 is the solid counter
            save.Small[SpecialWarpFlagsOffset] |= ContinueWarpFlag;
            save.SetEventFlag(PokedexFlag, true); // Direct Corner dialogue check
            save.SetEventFlag(PokemonFlag, true);
            save.SetWork(CableClubWork, 0);
            save.SetWork(TealaWork, 2); // skip first-visit movement tutorial
        }
        if (grantBadges)
            save.Badges = 0xFF; // bit mask, NOT the number 8

        // Skip the adventure recap in templates; retain other history bytes.
        for (var i = 0; i < QuestLogSceneCount; i++)
            save.Large[QuestLogOffset + i * QuestLogSceneSize] = 0;
    }

    public static object Prepare(string source, string output, bool grantBadges, bool warp)
    {
        if (Path.GetFullPath(source) == Path.GetFullPath(output))
            throw new ArgumentException("source and output must differ");
        var save = Load(File.ReadAllBytes(source));
        var before = Inspect(save);
        Apply(save, grantBadges, warp);
        var bytes = save.Write().ToArray();
        var reopened = Load(bytes);
        if (grantBadges && reopened.Badges != 0xFF)
            throw new InvalidOperationException("badge round-trip verification failed");
        if (warp && ((reopened.Small[SpecialWarpFlagsOffset] & ContinueWarpFlag) == 0 ||
                     !reopened.Large.Slice(ContinueWarpOffset, 8)
                         .SequenceEqual(new byte[] { 5, 5, 255, 0, 10, 0, 4, 0 })))
            throw new InvalidOperationException("destination round-trip verification failed");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        using (var stream = new FileStream(output, FileMode.CreateNew, FileAccess.Write))
            stream.Write(bytes);
        return new {
            ok = true, profile = Profile, before, after = Inspect(reopened),
            checksums_valid = reopened.ChecksumsValid,
            spawn = warp ? "Viridian Pokemon Center 2F: Direct Corner, (10,4); tap UP then A" : null,
            quest_log_recap_disabled = true, source_unchanged = true
        };
    }
}
