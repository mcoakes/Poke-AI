using System.Text.Json;
using PKHeX.Core;

static string Required(IReadOnlyDictionary<string, string> values, string name) =>
    values.TryGetValue(name, out var value) ? value : throw new ArgumentException($"missing {name}");


try
{
    var options = new Dictionary<string, string>();
    for (var i = 0; i < args.Length; i += 2)
    {
        if (i + 1 >= args.Length || !args[i].StartsWith("--", StringComparison.Ordinal))
            throw new ArgumentException("arguments must be --template FILE --team FILE --output FILE");
        options[args[i]] = args[i + 1];
    }

    if (options.TryGetValue("--prepare-template", out var sourcePath))
    {
        if (Required(options, "--profile") != FireRedTemplate.Profile)
            throw new ArgumentException("unsupported template profile");
        var report = FireRedTemplate.Prepare(sourcePath, Required(options, "--output"),
            bool.Parse(Required(options, "--grant-eight-badges")),
            bool.Parse(Required(options, "--warp-link-counter")));
        Console.WriteLine(JsonSerializer.Serialize(report));
        return 0;
    }

    if (options.TryGetValue("--inspect", out var inspectPath))
    {
        var inspected = FireRedTemplate.Load(File.ReadAllBytes(Path.GetFullPath(inspectPath)));
        Console.WriteLine(JsonSerializer.Serialize(new {
            ok = true,
            game = inspected.Version.ToString(),
            trainer = inspected.OT,
            party_size = inspected.PartyCount,
            party = inspected.PartyData.Select(TeamPokemonBuilder.Inspect),
            template_state = FireRedTemplate.Inspect(inspected)
        }));
        return 0;
    }

    var templatePath = Path.GetFullPath(Required(options, "--template"));
    var teamPath = Path.GetFullPath(Required(options, "--team"));
    var outputPath = Path.GetFullPath(Required(options, "--output"));
    var team = JsonSerializer.Deserialize<TeamInput>(
        File.ReadAllText(teamPath),
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
        ?? throw new ArgumentException("team JSON is empty");

    if (team.Party is not { Length: >= 1 and <= 6 })
        throw new ArgumentException("party must contain between one and six Pokemon");

    // FR/LG battery saves share a format. PKHeX can correctly report FRLG;
    // the Python preparation pipeline verifies the exact FireRed ROM separately.
    var save = FireRedTemplate.Load(File.ReadAllBytes(templatePath));
    var trainerName = save.OT;
    if (team.TrainerName is not null)
    {
        trainerName = team.TrainerName.Trim();
        if (trainerName.Length is < 1 or > 7 ||
            trainerName.Any(c => !"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ".Contains(c)))
            throw new ArgumentException("trainer_name must be 1–7 letters, digits, or spaces");
        save.OT = trainerName;
        if (save.OT != trainerName)
            throw new ArgumentException("trainer_name cannot be encoded in this save");
    }

    var party = team.Party.Select((input, index) =>
        TeamPokemonBuilder.Create(save, input, (uint)index)).ToList();
    if (outputPath == templatePath || outputPath == teamPath)
        throw new ArgumentException("output must not replace an input");

    save.PartyData = party;
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    var bytes = save.Write().ToArray();
    var verified = FireRedTemplate.Load(bytes);
    if (verified.OT != trainerName || verified.PartyCount != party.Count || !verified.ChecksumsValid ||
        !verified.PartyData.Select(p => JsonSerializer.Serialize(TeamPokemonBuilder.Inspect(p)))
            .SequenceEqual(party.Select(p => JsonSerializer.Serialize(TeamPokemonBuilder.Inspect(p)))))
        throw new InvalidOperationException("party stats failed save round-trip verification");
    File.WriteAllBytes(outputPath, bytes);
    Console.WriteLine(JsonSerializer.Serialize(new {
        ok = true,
        team = team.Name,
        trainer = verified.OT,
        party_size = party.Count,
        output = outputPath,
        preserved_progression = true,
        verified_party = verified.PartyData.Select(TeamPokemonBuilder.Inspect)
    }));
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine(JsonSerializer.Serialize(new { ok = false, error = error.Message }));
    return 1;
}
