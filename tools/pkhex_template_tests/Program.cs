using System.Security.Cryptography;
using System.Diagnostics;
using PKHeX.Core;
using static System.Buffers.Binary.BinaryPrimitives;

static void Check(bool condition, string description)
{
    if (!condition) throw new Exception(description);
}

// Synthetic, ROM-free save sectors. Test both active slots and rotated sector IDs.
static byte[] Fixture(int activeSlot)
{
    var raw = Enumerable.Repeat((byte)0xFF, 131072).ToArray();
    for (int slot = 0; slot < 2; slot++)
    for (int sector = 0; sector < 14; sector++)
    {
        var span = raw.AsSpan(slot * 0xE000 + sector * 0x1000, 0x1000);
        span[..0xF80].Clear();
        var id = (ushort)((sector + slot * 3) % 14);
        WriteUInt16LittleEndian(span[0xFF4..], id);
        WriteUInt32LittleEndian(span[0xFF8..], 0x08012025);
        WriteUInt32LittleEndian(span[0xFFC..], (uint)(slot == activeSlot ? 10 : 9));
        if (id == 0)
        {
            span[..8].Fill(0xFF);
            span[0] = 0xCE; // English trainer name character
            WriteUInt32LittleEndian(span[0xAC..], 1); // FR/LG discriminator
        }
        if (id == 1) span[4] = 3; // ordinary overworld, not a link room
        WriteUInt16LittleEndian(span[0xFF6..], Checksums.CheckSum32(span[..0xF80]));
    }
    var save = FireRedTemplate.Load(raw);
    save.OT = "TEST";
    save.Language = 2;
    save.Money = 12345;
    save.Badges = 3;
    save.Small[9] = 0xA0; // unrelated special-warp bits must survive
    save.SetWork(0x7C, 1);
    for (int i = 0; i < 4; i++) save.Large[0x1300 + 0x668 * i] = 1;
    var pokemon = new PK3 { Species = 25, Language = 2, PID = 12345, Version = GameVersion.FR };
    pokemon.CurrentLevel = 20;
    pokemon.Move1 = 84;
    pokemon.Heal();
    pokemon.RefreshChecksum();
    save.PartyData = new List<PKM> { pokemon };
    return save.Write().ToArray();
}

foreach (var activeSlot in new[] { 0, 1 })
{
    foreach (var edits in new[] { (true, true), (true, false), (false, true) })
    {
        var original = Fixture(activeSlot);
        var save = FireRedTemplate.Load(original.ToArray());
        var beforeLarge = save.Large.ToArray();
        var beforeSmall = save.Small.ToArray();
        var beforeStorage = save.Storage.ToArray();
        FireRedTemplate.Apply(save, edits.Item1, edits.Item2);
        var result = save.Write().ToArray();
        var after = FireRedTemplate.Load(result);
        Check(after.Badges == (edits.Item1 ? 255 : 3), "badge mask");
        Check(after.ChecksumsValid, "round-trip checksums");
        Check(after.Money == 12345 && after.OT == "TEST", "trainer and money preserved");
        Check(after.Storage.SequenceEqual(beforeStorage), "PC storage preserved");
        Check(after.Large.Slice(0x34, 0x290 - 0x34).SequenceEqual(
            beforeLarge.AsSpan(0x34, 0x290 - 0x34)), "party preserved");
        Check(after.Large[..12].SequenceEqual(beforeLarge.AsSpan(0, 12)), "original map preserved");
        if (edits.Item2)
        {
            Check(after.Large.Slice(0x0C, 8).SequenceEqual(
                new byte[] { 5, 5, 255, 0, 10, 0, 4, 0 }), "warp destination");
            Check((after.Small[9] & 1) == 1, "continue-game warp enabled");
            Check(after.GetWork(0x7C) == 2 && after.GetEventFlag(0x829), "link requirements");
        }
        Check(after.Small[9] == (edits.Item2 ? 0xA1 : 0xA0), "other warp flags preserved");
        var allowed = new HashSet<int>();
        for (int i = 0; i < 4; i++) allowed.Add(0x1300 + i * 0x668);
        if (edits.Item1) allowed.Add(0xEE0 + (0x820 >> 3));
        if (edits.Item2)
        {
            for (int i = 0xC; i < 0x14; i++) allowed.Add(i);
            allowed.Add(0xEE0 + (0x828 >> 3));
            foreach (int offset in new[] { 0x1000 + 2 * 0x6F, 0x1000 + 2 * 0x7C })
            { allowed.Add(offset); allowed.Add(offset + 1); }
        }
        for (int i = 0; i < beforeLarge.Length; i++)
            Check(beforeLarge[i] == after.Large[i] || allowed.Contains(i), $"unrelated Large byte {i:X}");
        for (int i = 0; i < beforeSmall.Length; i++)
            Check(beforeSmall[i] == after.Small[i] || (edits.Item2 && i == 9), $"unrelated Small byte {i:X}");
        var inactiveStart = (1 - activeSlot) * 0xE000;
        Check(original.AsSpan(inactiveStart, 0xE000).SequenceEqual(
            result.AsSpan(inactiveStart, 0xE000)), "inactive save slot preserved");
    }
}

var invalid = Fixture(0);
invalid[100] ^= 1;
try { FireRedTemplate.Load(invalid); throw new Exception("corrupt checksum accepted"); }
catch (ArgumentException) { }

var busy = FireRedTemplate.Load(Fixture(0));
busy.SetWork(0x6F, 1);
try { FireRedTemplate.Apply(busy, true, true); throw new Exception("busy link accepted"); }
catch (ArgumentException) { }

var empty = FireRedTemplate.Load(Fixture(0));
empty.PartyData = new List<PKM>();
try { FireRedTemplate.Apply(empty, true, true); throw new Exception("empty party accepted"); }
catch (ArgumentException) { }

using var directory = new TemporaryTestDirectory();
var source = Path.Combine(directory.Path, "source.sav");
var output = Path.Combine(directory.Path, "edited.sav");
File.WriteAllBytes(source, Fixture(1));
var hash = SHA256.HashData(File.ReadAllBytes(source));
FireRedTemplate.Prepare(source, output, true, true);
Check(hash.SequenceEqual(SHA256.HashData(File.ReadAllBytes(source))), "source unchanged");
var teamSave = FireRedTemplate.Load(File.ReadAllBytes(output));
var replacement = new PK3 { Species = 7, Language = 2, Version = GameVersion.FR };
replacement.CurrentLevel = 50;
replacement.Heal();
replacement.RefreshChecksum();
teamSave.PartyData = new List<PKM> { replacement };
teamSave = FireRedTemplate.Load(teamSave.Write().ToArray());
Check(teamSave.Badges == 255 && (teamSave.Small[9] & 1) == 1 &&
      teamSave.PartyData[0].Species == 7 && teamSave.Large.Slice(0xC, 8)
          .SequenceEqual(new byte[] { 5, 5, 255, 0, 10, 0, 4, 0 }),
      "party replacement preserves prepared badge and warp settings");
try { FireRedTemplate.Prepare(source, output, true, true); throw new Exception("overwrite accepted"); }
catch (IOException) { }

// Exercise the real file-based CLI, not only in-memory PartyData assignment.
var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../"));
var editor = Path.Combine(root, "tools/pkhex_team_editor/bin/Debug/net10.0/PKHeXTeamEditor.dll");
void RunEditor(params string[] arguments)
{
    var start = new ProcessStartInfo(Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") ?? "dotnet")
    { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
    start.ArgumentList.Add(editor);
    foreach (var argument in arguments) start.ArgumentList.Add(argument);
    using var process = Process.Start(start)!;
    var stdout = process.StandardOutput.ReadToEndAsync();
    var stderr = process.StandardError.ReadToEndAsync();
    if (!process.WaitForExit(30000)) { process.Kill(entireProcessTree: true); throw new Exception("CLI timeout"); }
    Check(process.ExitCode == 0, $"editor CLI failed: {stdout.Result} {stderr.Result}");
}
RunEditor("--inspect", output);
foreach (var name in new[] { "fire_vs_water", "water_vs_fire" })
{
    var generated = Path.Combine(directory.Path, name + ".sav");
    RunEditor("--template", output, "--team", Path.Combine(root, "config/teams", name + ".json"),
              "--output", generated);
    var loaded = FireRedTemplate.Load(File.ReadAllBytes(generated));
    Check(loaded.PartyCount == (name == "water_vs_fire" ? 6 : 2) &&
          loaded.Badges == 255 && (loaded.Small[9] & 1) == 1,
          "CLI generated team preserves template settings");
    Check(loaded.PartyData[0].Species == (name == "water_vs_fire" ? 9 : 6), "CLI generated species");
}
Check(hash.SequenceEqual(SHA256.HashData(File.ReadAllBytes(source))), "CLI source unchanged");

var statTeam = Path.Combine(directory.Path, "custom_stats.json");
var statSave = Path.Combine(directory.Path, "custom_stats.sav");
File.WriteAllText(statTeam, """
{"name":"stats","party":[
  {"species":"Starmie","level":50,"moves":["Surf","Thunderbolt","Ice Beam","Recover"],
   "held_item":"Leftovers","nature":"Timid",
   "evs":{"hp":4,"spa":252,"spe":252},"ivs":{"atk":0}},
  {"species":"Marowak","level":50,"moves":["Earthquake","Rock Slide"],
   "held_item":"Thick Club","nature":"Adamant",
   "evs":{"hp":252,"atk":252,"spd":4},"ivs":{"spe":0}}]}
""");
RunEditor("--template", output, "--team", statTeam, "--output", statSave);
var statsLoaded = FireRedTemplate.Load(File.ReadAllBytes(statSave));
var starmie = statsLoaded.PartyData[0];
var marowak = statsLoaded.PartyData[1];
Check(starmie.HeldItem == 200 && marowak.HeldItem == 224, "Gen III held item IDs");
Check(starmie.EV_HP == 4 && starmie.EV_SPA == 252 && starmie.EV_SPE == 252 &&
      starmie.EV_ATK == 0 && starmie.IV_ATK == 0 && starmie.IV_SPE == 31, "custom and omitted stats");
Check(marowak.EV_HP == 252 && marowak.EV_ATK == 252 && marowak.EV_SPD == 4 &&
      marowak.IV_SPE == 0 && marowak.IV_HP == 31, "stat order");
Check(starmie.Nature == Nature.Timid && marowak.Nature == Nature.Adamant, "PID natures");
Check(starmie.Stat_HPMax == 136 && starmie.Stat_HPCurrent == 136 &&
      marowak.Stat_HPMax == 167 && marowak.Stat_HPCurrent == 167, "HP recalculated and healed");
Check(starmie.Nickname == "STARMIE" && marowak.Nickname == "MAROWAK", "uppercase names retained");
Check(statsLoaded.Badges == 255 && statsLoaded.ChecksumsValid, "progression and checksums retained");
Console.WriteLine("PASS: both active slots, rotated sectors, badge/warp modes, preservation, checksums, guards");
Console.WriteLine("PASS: file-based inspect and both JSON team-generation commands");
Console.WriteLine("PASS: custom EVs, IVs, Gen III held item IDs, natures, recalculated HP and names");

sealed class TemporaryTestDirectory : IDisposable
{
    public string Path { get; } = Directory.CreateTempSubdirectory("pkhex-template-tests-").FullName;
    public void Dispose() => Directory.Delete(Path, recursive: true);
}
