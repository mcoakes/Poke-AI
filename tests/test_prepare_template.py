"""Python transaction tests; the actual binary editor has a separate .NET test runner."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_template", Path(__file__).parents[1] / "mgba/prepare_template.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "source.sav"
    source.write_bytes(b"a" * 131072)
    rom = tmp_path / "FireRed.gba"
    rom.write_bytes(b"unit-test-only-not-a-ROM")
    # Test transaction code with synthetic bytes, not ROM data.
    monkeypatch.setattr(module, "ROM_SHA1", hashlib.sha1(rom.read_bytes()).hexdigest())
    return tmp_path, source, tmp_path / "output.sav", rom


def fake_editor(project, source, output, badges, warp):
    output.write_bytes(b"b" * 131072)
    return {"ok": True, "checksums_valid": True, "after": {"badge_count": 8}}


def test_prepare_writes_edited_bytes_and_preserves_source(inputs, monkeypatch):
    monkeypatch.setattr(module, "run_editor", fake_editor)
    project, source, output, rom = inputs
    report = module.prepare(project, source, output, rom, True, True)
    assert output.read_bytes() == b"b" * 131072
    assert source.read_bytes() == b"a" * 131072
    metadata = json.loads(Path(report["metadata"]).read_text())
    assert metadata["sha256"] != metadata["source_sha256"]


def test_force_preserves_existing_template_and_metadata(inputs, monkeypatch):
    monkeypatch.setattr(module, "run_editor", fake_editor)
    project, source, output, rom = inputs
    output.write_bytes(b"old save")
    metadata = output.with_suffix(".sav.json")
    metadata.write_text("old metadata")
    with pytest.raises(FileExistsError):
        module.prepare(project, source, output, rom, True, True)
    report = module.prepare(project, source, output, rom, True, True, True)
    assert len(report["backups"]) == 2
    assert Path(report["backups"][0]).read_bytes() == b"old save"
    assert Path(report["backups"][1]).read_text() == "old metadata"


def test_failed_editor_leaves_existing_output_untouched(inputs, monkeypatch):
    def fail(*args):
        raise RuntimeError("editor failed")
    monkeypatch.setattr(module, "run_editor", fail)
    project, source, output, rom = inputs
    output.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="editor failed"):
        module.prepare(project, source, output, rom, True, True, True)
    assert output.read_bytes() == b"previous"
    assert source.read_bytes() == b"a" * 131072


def test_reject_in_place_even_with_force(inputs):
    project, source, _output, rom = inputs
    with pytest.raises(ValueError, match="source and output"):
        module.prepare(project, source, source, rom, True, True, True)


def test_reject_wrong_rom(inputs):
    project, source, output, rom = inputs
    rom.write_bytes(b"wrong ROM")
    with pytest.raises(ValueError, match="unsupported ROM"):
        module.prepare(project, source, output, rom, True, True)
    assert not output.exists()


def test_reject_missing_edits(inputs):
    with pytest.raises(ValueError, match="select"):
        module.prepare(*inputs, False, False)


def test_reject_save_state(inputs):
    project, source, output, rom = inputs
    source.write_bytes(b"not a raw battery save")
    with pytest.raises(ValueError, match="131072"):
        module.prepare(project, source, output, rom, True, True)


def test_reject_rom_as_output(inputs):
    project, source, _output, rom = inputs
    with pytest.raises(ValueError, match="separate .sav"):
        module.prepare(project, source, rom, rom, True, True, True)


def test_reject_hardlinked_source(inputs):
    project, source, output, rom = inputs
    output.hardlink_to(source)
    with pytest.raises(ValueError, match="source and output"):
        module.prepare(project, source, output, rom, True, True, True)


def test_backup_numbering(tmp_path):
    original = tmp_path / "a.sav"
    original.write_bytes(b"first")
    first = module.backup_file(original)
    original.write_bytes(b"second")
    second = module.backup_file(original)
    assert first != second
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"


def test_source_change_is_detected(inputs, monkeypatch):
    project, source, output, rom = inputs
    def concurrent_save(*args):
        fake_editor(*args)
        source.write_bytes(b"changed externally")
        return {"ok": True, "checksums_valid": True}
    monkeypatch.setattr(module, "run_editor", concurrent_save)
    with pytest.raises(RuntimeError, match="source changed"):
        module.prepare(project, source, output, rom, True, True)
    assert not output.exists()
