"""Audit reports verify battle observations and detect injected regressions."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

pytest.importorskip("sb3_contrib")
from firered_rl import audit_cases

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(), reason="Showdown not installed")


def cli_module():
    spec = importlib.util.spec_from_file_location("audit_training_cli", ROOT / "mgba/audit_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_oracle_confirms_transform_and_volatile_counters(tmp_path):
    checks = audit_cases.run_battle_checks(ROOT, tmp_path)
    failures = [row for row in checks if row["status"] == "fail"]
    assert failures == []
    rows = {row["name"]: row for row in checks}
    for name in ("transformed_sparse_moves", "own_volatile_flags_and_counters_p1",
                 "own_volatile_flags_and_counters_p2"):
        assert rows[name]["status"] == "pass"
    assert sum(row["details"]["snapshots"] for row in checks) >= 130
    assert (tmp_path / "battle.log").is_file()


def test_oracle_catches_valid_shaped_wrong_hp(tmp_path, monkeypatch):
    encode = audit_cases.showdown_observation

    def wrong_hp(report, terminal=False):
        result = encode(report, terminal)
        result["party"][0, 3] += 1
        return result

    monkeypatch.setattr(audit_cases, "showdown_observation", wrong_hp)
    with pytest.raises(AssertionError, match="P1 own party 1: expected"):
        auditor = audit_cases.BattleAudit(ROOT, tmp_path, node="node", timeout=10, seed=4,
                                         log_max_mb=.05, log_backups=1)
        try:
            auditor.reset([audit_cases.mon("Magikarp", ["Splash"])], [audit_cases.mon("Magikarp", ["Splash"])])
        finally:
            auditor.close()


def test_cli_reports_findings_without_training_and_refuses_overwrite(tmp_path, monkeypatch):
    cli = cli_module()
    output = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit_training", "--output-dir", str(output), "--log-max-mb", ".05"])
    assert cli.main() == 0
    report = json.loads((output / "audit-report.json").read_text())
    assert report["status"] == "completed" and report["result"] == "pass"
    assert report["counts"] == {"pass": 51, "limitation": 13}
    assert report["counts"].get("fail", 0) == 0
    assert {row["group"] for row in report["checks"]} == {"inventory", "controlled", "pipeline"}
    assert not list(output.rglob("*.zip"))
    assert not list(output.rglob("episode-*"))
    for path in output.rglob("*"):
        if path.is_file() and (".log" in path.name or path.name.startswith("audit-diagnostics.jsonl")):
            assert path.stat().st_size <= int(.05*1024*1024)
    content = (output / "audit-report.json").read_bytes()
    assert cli.main() == 2
    assert (output / "audit-report.json").read_bytes() == content


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_cli_rejects_unbounded_or_invalid_logs(tmp_path, monkeypatch, value):
    cli = cli_module()
    output = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["audit_training", "--output-dir", str(output), "--log-max-mb", value])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not output.exists()


def test_native_oracle_catches_dropped_substitute_flag(tmp_path, monkeypatch):
    encode = audit_cases.showdown_observation

    def dropped_volatile(report, terminal=False):
        result = encode(report, terminal)
        result["active"][0, 9] = 0
        return result

    monkeypatch.setattr(audit_cases, "showdown_observation", dropped_volatile)
    audit = audit_cases.BattleAudit(ROOT, tmp_path, node="node", timeout=10, seed=4,
                                   log_max_mb=.05, log_backups=1)
    try:
        audit.reset([audit_cases.mon("Mew", ["Substitute"])], [audit_cases.mon("Magikarp", ["Splash"])])
        with pytest.raises(AssertionError, match="P1 own Substitute flag: expected True; observed False"):
            audit.step([0, 0])
    finally:
        audit.close()


def test_both_player_move_checks_catch_wrong_p2_dispatch(tmp_path, monkeypatch):
    request = audit_cases.ShowdownFactory.request

    def wrong_p2(factory, command, **fields):
        if command == "step" and fields["actions"][1] in (1, 2, 3):
            fields = {**fields, "actions": [fields["actions"][0], 0]}
        return request(factory, command, **fields)

    monkeypatch.setattr(audit_cases.ShowdownFactory, "request", wrong_p2)
    checks = {row["name"]: row for row in audit_cases.run_battle_checks(ROOT, tmp_path)}
    assert checks["all_move_counts_and_slots"]["status"] == "fail"
    assert "P2 move slot 2 execution" in checks["all_move_counts_and_slots"]["details"]["error"]
    assert checks["move_order_permutation"]["status"] == "fail"
    assert "P2 permuted action" in checks["move_order_permutation"]["details"]["error"]
