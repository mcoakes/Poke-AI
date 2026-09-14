import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def builder(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "mgba"))
    spec = importlib.util.spec_from_file_location("build_states", root / "mgba/build_states.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_partial_publish_when_one_staged_file_missing(builder, tmp_path):
    valid = tmp_path / "valid.ss1"
    valid.write_bytes(b"state")
    out = tmp_path / "out.ss1"
    with pytest.raises(RuntimeError):
        builder.publish([(valid, out), (tmp_path / "missing", tmp_path / "other")], False)
    assert not out.exists()


def test_force_keeps_old_state_backup(builder, tmp_path):
    staged, out = tmp_path / "new.ss1", tmp_path / "out.ss1"
    staged.write_bytes(b"new")
    out.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        builder.publish([(staged, out)], False)
    backups = builder.publish([(staged, out)], True)
    assert Path(backups[0]).read_bytes() == b"old"
    assert out.read_bytes() == b"new"


def test_batch_directory_selects_only_json_in_filename_order(builder, tmp_path):
    from types import SimpleNamespace
    for name in ("team_002.json", "team_001.json", "generation-summary.txt"):
        (tmp_path / name).write_text("{}")
    args = SimpleNamespace(team_dir=tmp_path, teams=None)
    assert [path.name for path in builder.select_teams(args, tmp_path)] == ["team_001.json", "team_002.json"]
    with pytest.raises(FileNotFoundError):
        builder.select_teams(SimpleNamespace(team_dir=tmp_path / "missing", teams=None), tmp_path)


def test_profile_build_mirrors_directories_and_resumes_completed_teams(builder, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from firered_rl.team_profiles import CATEGORIES
    root = Path(__file__).parents[1]
    for category in CATEGORIES:
        folder = tmp_path / f"config/teams/profiles/{category}/2"
        folder.mkdir(parents=True)
        # Equal basenames across categories must not collide in jobs or outputs.
        data = json.loads(next((root / f"config/teams/profiles/{category}/2").glob("*.json")).read_text())
        (folder / "team.json").write_text(json.dumps(data))
    source, rom, emulator = (tmp_path / name for name in ("base.sav", "FireRed.gba", "mgba-qt"))
    for path in (source, rom, emulator):
        path.write_bytes(path.name.encode())
    args = SimpleNamespace(source=source, rom=rom, emulator=emulator, timeout=10, visible=True,
                           team_profile="mixed", team_size=2, teams=None, team_dir=None,
                           force=False, skip_existing=False, speed="unlimited")
    prepared = []

    def prepare(project, src, template, *_):
        prepared.append(src)
        template.write_bytes(b"prepared template")
        metadata = template.with_suffix(".json")
        metadata.write_text(json.dumps({"source_sha256": builder.digest(src)}))
        return {"metadata": str(metadata)}

    monkeypatch.setattr(builder, "prepare", prepare)
    monkeypatch.setattr(builder, "find_dotnet", lambda _: "dotnet")

    def edit(command, **_):
        save = Path(command[command.index("--output") + 1])
        team = json.loads(Path(command[command.index("--team") + 1]).read_text())
        save.write_bytes(json.dumps(team).encode())
        return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "verified_party": team["party"]}))

    monkeypatch.setattr(builder.subprocess, "run", edit)
    captures = []

    def capture(project, exe, rom, save, job, count, visible, timeout, speed):
        captures.append(job)
        if len(captures) == 2:
            raise RuntimeError("simulated capture failure")
        state = job / "counter.ss1"
        state.write_bytes(save.read_bytes() + b"snapshot")
        return state, {"ok": True}

    monkeypatch.setattr(builder, "capture", capture)
    with pytest.raises(RuntimeError, match="simulated capture failure"):
        builder.build(args, tmp_path)
    assert (tmp_path / "states/counter/profiles/competitive/2/team.ss1").is_file()
    assert not (tmp_path / "states/counter/profiles/casual/2/team.ss1").exists()
    args.skip_existing = True
    result = builder.build(args, tmp_path)
    assert result["skipped"] == 1 and len(result["teams"]) == 3
    for category in CATEGORIES:
        manifest = tmp_path / f"states/counter/profiles/{category}/2/team.ss1.json"
        data = json.loads(manifest.read_text())
        assert data["team_profile"] == category and data["team_size"] == 2
        assert builder.digest(Path(data["state"])) == data["state_sha256"]
    assert len(captures) == 4 and len(prepared) == 2
    assert builder.build(args, tmp_path)["skipped"] == 3
    assert len(captures) == 4 and len(prepared) == 2
    (tmp_path / "states/counter/profiles/random/2/team.ss1").write_bytes(b"changed")
    with pytest.raises(ValueError, match="existing outputs differ"):
        builder.build(args, tmp_path)


@pytest.mark.parametrize("count", [1, 2])
def test_profile_builder_rejects_wrong_size_or_fewer_than_four_moves(builder, tmp_path, count):
    from types import SimpleNamespace
    from firered_rl.teams import TeamSpec
    directory = tmp_path / "config/teams/profiles/casual/2"
    directory.mkdir(parents=True)
    team = directory / "team.json"
    team.write_text(json.dumps({"party": [{"species": "Pikachu", "level": 50, "moves": ["Thunderbolt"]}] * count}))
    with pytest.raises(ValueError, match="2 Pokemon with four moves"):
        builder.team_outputs(SimpleNamespace(team_profile="casual"), tmp_path, [team], [TeamSpec.load(team)])


def test_parallel_workers_share_preparation_and_publish_in_custom_root(builder, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import threading
    source, rom, emulator = [tmp_path / name for name in ("input.sav", "rom.gba", "mgba")]
    for path in (source, rom, emulator):
        path.write_bytes(b"input")
    teams = []
    for i in range(2):
        path = tmp_path / f"party{i}.json"
        path.write_text(json.dumps(dict(party=[dict(species="Pikachu", level=50, moves=["Tackle"])])))
        teams.append(path)
    args = SimpleNamespace(source=source, rom=rom, emulator=emulator, teams=teams,
                           timeout=30, visible=True, speed="unlimited", force=False,
                           workers=8, output_dir=tmp_path / "match/states")
    preparations = []
    def prepare(project, source, template, rom, badges, warp):
        preparations.append(template)
        template.write_bytes(b"prepared")
        metadata = template.with_suffix(".json")
        metadata.write_text(json.dumps(dict(source_sha256=builder.digest(source))))
        return dict(metadata=str(metadata))
    def edit(command, cancel):
        Path(command[command.index("--output") + 1]).write_bytes(b"save")
        return SimpleNamespace(returncode=0, stdout=json.dumps(dict(ok=True, verified_party=[dict(species="Pikachu")])))
    barrier = threading.Barrier(2)
    def capture(project, exe, rom, save, job, count, visible, timeout, speed, *, cancel):
        assert not cancel.is_set()
        barrier.wait(timeout=3)  # Both captures must run together, not serially.
        state = job / "counter.ss1"
        state.write_bytes(b"state")
        return state, dict(ok=True)
    monkeypatch.setattr(builder, "prepare", prepare)
    monkeypatch.setattr(builder, "find_dotnet", lambda p: "dotnet")
    monkeypatch.setattr(builder, "edit_team", edit)
    monkeypatch.setattr(builder, "capture", capture)
    result = builder.build(args, tmp_path)
    assert result["workers"] == 2 and len(preparations) == 1
    assert [team["team"] for team in result["teams"]] == ["party0", "party1"]
    assert all((args.output_dir / "counter" / f"party{i}.ss1").is_file() for i in range(2))
    assert not (tmp_path / "states").exists()
    assert source.read_bytes() == b"input"
