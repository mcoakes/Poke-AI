using System.Text.Json;
using System.Text.Json.Serialization;
using PKHeX.Core;

// IDs in this catalog are national species IDs, move IDs and GEN III item IDs.
// Modern item IDs must never be written directly to PK3.HeldItem.
static class TeamPokemonBuilder
{
    private static readonly string[] Stats = ["hp", "atk", "def", "spa", "spd", "spe"];
    private static readonly JsonDocument Catalog = JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "firered_us_v1_0.json")));

    private static string Key(string value) => new(value.Replace("♀", "f").Replace("♂", "m")
        .Where(char.IsLetterOrDigit).Select(char.ToLowerInvariant).ToArray());

    private static ushort Lookup(string group, string value)
    {
        if (string.IsNullOrWhiteSpace(value)) throw new ArgumentException($"missing {group} name");
        var key = Key(value);
        foreach (var entry in Catalog.RootElement.GetProperty(group).EnumerateObject())
            if (Key(entry.Name) == key) return entry.Value.GetProperty("id").GetUInt16();
        throw new ArgumentException($"unknown FireRed {group} name: {value}");
    }

    private static int[] StatValues(Dictionary<string, int>? input, int fallback, int max, string label)
    {
        input ??= new();
        if (input.Keys.Any(key => !Stats.Contains(key)))
            throw new ArgumentException($"{label}: use hp, atk, def, spa, spd, spe");
        var result = Stats.Select(key => input.GetValueOrDefault(key, fallback)).ToArray();
        if (result.Any(value => value < 0 || value > max))
            throw new ArgumentException($"{label}: each value must be between 0 and {max}");
        if (label == "evs" && result.Sum() > 510)
            throw new ArgumentException("evs: total cannot exceed 510");
        return result;
    }

    public static PKM Create(SaveFile save, PokemonInput input, uint index)
    {
        if (input.Level is < 1 or > 100)
            throw new ArgumentException($"{input.Species}: level must be between 1 and 100");
        if (input.Moves is not { Length: >= 1 and <= 4 })
            throw new ArgumentException($"{input.Species}: provide between one and four moves");
        var species = Lookup("species", input.Species);
        var moves = input.Moves.Select(move => Lookup("moves", move)).ToArray();
        if (moves.Distinct().Count() != moves.Length)
            throw new ArgumentException($"{input.Species}: duplicate moves");
        var item = Lookup("items", input.HeldItem);
        var evs = StatValues(input.Evs, 0, 255, "evs");
        var ivs = StatValues(input.Ivs, 31, 31, "ivs");
        if (!save.Personal.IsPresentInGame(species, 0))
            throw new ArgumentException($"{input.Species} is not available in FireRed");
        var pokemon = save.BlankPKM;
        pokemon.Species = species;
        pokemon.PID = 0x13579BDFu + index * 0x10201u;
        if (input.Nature is not null)
        {
            if (!Enum.TryParse<Nature>(input.Nature, true, out var nature) || (byte)nature > 24)
                throw new ArgumentException($"unknown nature: {input.Nature}");
            // Nature is PID % 25 in Gen III. Keep the original PID parity.
            while (pokemon.PID % 25 != (uint)nature) pokemon.PID += 2;
        }
        pokemon.TID16 = save.TID16;
        pokemon.SID16 = save.SID16;
        pokemon.Language = (int)LanguageID.English;
        pokemon.OriginalTrainerName = save.OT;
        pokemon.OriginalTrainerGender = save.Gender;
        pokemon.Version = GameVersion.FR;
        pokemon.CurrentLevel = (byte)input.Level;
        pokemon.Nickname = SpeciesName.GetSpeciesNameGeneration(
            species, (int)LanguageID.English, 3).ToUpperInvariant();
        pokemon.IsNicknamed = false;
        (pokemon.EV_HP, pokemon.EV_ATK, pokemon.EV_DEF, pokemon.EV_SPA, pokemon.EV_SPD, pokemon.EV_SPE) =
            (evs[0], evs[1], evs[2], evs[3], evs[4], evs[5]);
        (pokemon.IV_HP, pokemon.IV_ATK, pokemon.IV_DEF, pokemon.IV_SPA, pokemon.IV_SPD, pokemon.IV_SPE) =
            (ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5]);
        pokemon.HeldItem = item;
        pokemon.CurrentFriendship = 255;
        pokemon.Moves = moves;
        pokemon.Heal(); // Recalculate party stats after IVs, EVs, nature and level.
        pokemon.RefreshChecksum();
        return pokemon;
    }

    public static object Inspect(PKM p) => new {
        species_id = p.Species, level = p.CurrentLevel, nickname = p.Nickname,
        held_item_id = p.HeldItem, nature = p.Nature.ToString(),
        evs = new { hp = p.EV_HP, atk = p.EV_ATK, def = p.EV_DEF, spa = p.EV_SPA, spd = p.EV_SPD, spe = p.EV_SPE },
        ivs = new { hp = p.IV_HP, atk = p.IV_ATK, def = p.IV_DEF, spa = p.IV_SPA, spd = p.IV_SPD, spe = p.IV_SPE },
        moves = p.Moves, hp = p.Stat_HPCurrent, max_hp = p.Stat_HPMax
    };
}

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
record PokemonInput(string Species, int Level, string[] Moves,
    Dictionary<string, int>? Evs = null, Dictionary<string, int>? Ivs = null,
    [property: JsonPropertyName("held_item")] string HeldItem = "None", string? Nature = null);
record TeamInput(string Name, PokemonInput[] Party,
    [property: System.Text.Json.Serialization.JsonPropertyName("trainer_name")] string? TrainerName = null);
