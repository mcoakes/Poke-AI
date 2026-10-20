"""Confusion counters follow native Gen III state through natural clearance."""
from pathlib import Path
import shutil

import pytest

pytest.importorskip("sb3_contrib")
from firered_rl.audit_cases import BattleAudit, mon

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
    reason="Showdown not installed")


@pytest.mark.parametrize("side", [0, 1], ids=["p1", "p2"])
@pytest.mark.parametrize("initial,seed", [(2, 6), (3, 0), (4, 1), (5, 4)])
def test_confusion_counter_decrements_and_clears_on_both_sides(tmp_path, side, initial, seed):
    audit = BattleAudit(ROOT, tmp_path, node="node", timeout=10, seed=seed,
                        log_max_mb=.05, log_backups=1)
    # The faster target moves before Confuse Ray, preserving the full initial
    # native counter. Subsequent Splash turns allow it to expire naturally.
    teams = [[mon("Mew", ["Splash"])], [mon("Blissey", ["Confuse Ray", "Splash"])]]
    if side:
        teams.reverse()
    try:
        audit.reset(*teams)
        assert int(audit.observations[side]["active"][0, 9]) & 7 == 0
        audit.step([0, 0])
        for remaining in range(initial, -1, -1):
            confusion = audit.active(side)["volatiles"].get("confusion")
            assert (confusion["time"] if confusion else 0) == remaining
            assert int(audit.observations[side]["active"][0, 9]) & 7 == remaining
            assert int(audit.observations[1-side]["active"][0, 9]) & 7 == 0
            if remaining:
                actions = [1, 1]
                actions[side] = 0
                audit.step(actions)
        assert "confusion" not in audit.active(side)["volatiles"]
        assert any(line.startswith(f"|-end|p{side+1}a:") and "|confusion" in line
                   for line in audit.raw["protocol"])
    finally:
        audit.close()
