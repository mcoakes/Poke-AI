from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("sb3_contrib")
from firered_rl.evaluation import digest, validate_suite
from firered_rl.generalization import checksum, party_identity, prepare, publish

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def team(species, level=50):
    return {"name": species, "party": [{"species": species, "level": level, "moves": ["Tackle", "Growl"]}]}


@pytest.fixture
def comparison_inputs(tmp_path):
    from firered_rl.baseline_setup import _suite
    training = write(tmp_path / "old.json", team("Pikachu"))
    validation = write(tmp_path / "validation.json", team("Bulbasaur"))
    opponent = tmp_path / "opponent.zip"
    opponent.write_bytes(b"frozen model")
    pool = {"random": dict(kind="random", group="random")}
    for group in ("dumb", "smart"):
        pool[group] = dict(kind="checkpoint", group=group, source=str(opponent), sha256=checksum(opponent.read_bytes()))
    suite = _suite([validation], pool, episodes=2, seed=1, max_steps=2)
    runs = [tmp_path / "previous", tmp_path / "candidate"]
    for run in runs:
        write(run / "config.json", dict(backend="showdown", teams={"teams": [dict(
            path=str(training), sha256=checksum(training.read_bytes()))]}))
        write(run / "inputs/evaluation/suite.json", suite)
        write(run / "inputs/reward.json", {})
        write(run / "checkpoints/best.json", {"checkpoint_sha256": checksum(b"model")})
        (run / "checkpoints/best.zip").write_bytes(b"model")
    folder = tmp_path / "new-teams"
    write(folder / "new.json", team("Squirtle"))
    return runs, folder, training, opponent


def test_overlap_reordered_and_duplicate_teams(comparison_inputs, tmp_path):
    runs, folder, training, _ = comparison_inputs
    old = json.loads(training.read_text())
    old["name"] = "renamed"
    old["party"][0]["moves"].reverse()
    write(folder / "old.json", old)
    write(folder / "validation.json", team("Bulbasaur"))
    write(folder / "copy.json", team("Squirtle"))
    plan = prepare(*runs, folder)
    assert len(plan["teams"]) == 1 and len(plan["excluded"]) == 3
    output = tmp_path / "comparison"
    models = publish(plan, output, episodes_per_opponent=4, seed=2)
    suite = validate_suite(json.loads((output / "suite.json").read_text()))
    assert len(suite["cases"]) == 12
    assert {c["opponent_group"] for c in suite["cases"]} == {"random", "dumb", "smart"}
    assert len(suite["assets"]) == 1
    assert all(Path(row["path"]).read_bytes() == b"model" for row in models.values())
    with pytest.raises(FileExistsError):
        publish(plan, output, episodes_per_opponent=4, seed=2)
    a = team("Squirtle")
    a["party"].append(team("Pikachu")["party"][0])
    b = deepcopy(a)
    b["party"].reverse()
    assert party_identity(a) == party_identity(b)


@pytest.mark.parametrize("kind", ["team", "opponent", "best", "reward", "suite", "missing"])
def test_reject_changed_or_missing_sources(comparison_inputs, kind):
    runs, folder, training, opponent = comparison_inputs
    if kind == "team":
        write(training, team("Pikachu", 49))
    elif kind == "opponent":
        opponent.write_bytes(b"changed")
    elif kind == "best":
        (runs[0] / "checkpoints/best.zip").write_bytes(b"changed")
    elif kind == "reward":
        write(runs[0] / "inputs/reward.json", {"win": 999})
    elif kind == "suite":
        path = runs[0] / "inputs/evaluation/suite.json"
        data = json.loads(path.read_text())
        data["seed"] += 1
        write(path, data)
    else:
        training.unlink()
    with pytest.raises((ValueError, OSError)):
        prepare(*runs, folder)


def test_additional_exclusions_and_all_rejected(comparison_inputs, tmp_path):
    runs, folder, _, _ = comparison_inputs
    extra = tmp_path / "ancestor-teams"
    write(extra / "old.json", team("Squirtle"))
    with pytest.raises(ValueError, match="no unseen"):
        prepare(*runs, folder, exclude_dirs=[extra])
    extra_suite = json.loads((runs[0] / "inputs/evaluation/suite.json").read_text())
    asset = {"team": team("Squirtle")}
    extra_suite["assets"] = {digest(asset): asset}
    for case in extra_suite["cases"]:
        case["assets"] = [digest(asset), digest(asset)]
    extra_path = write(tmp_path / "extra-suite.json", extra_suite)
    with pytest.raises(ValueError, match="no unseen"):
        prepare(*runs, folder, exclude_suites=[extra_path])


def test_real_cli_with_frozen_models(training, comparison_inputs, tmp_path):
    runs, folder, _, opponent = comparison_inputs
    # The existing fixture supplies a real v2 learner and v1 frozen policy.
    opponent.write_bytes(training.frozen.read_bytes())
    for run in runs:
        path = run / "inputs/evaluation/suite.json"
        suite = json.loads(path.read_text())
        for entry in suite["opponents"].values():
            if entry["kind"] == "checkpoint":
                entry["sha256"] = checksum(opponent.read_bytes())
        write(path, suite)
        data = (training.run / "checkpoints/final.zip").read_bytes()
        (run / "checkpoints/best.zip").write_bytes(data)
        write(run / "checkpoints/best.json", {"checkpoint_sha256": checksum(data)})
    before = [(run / "checkpoints/best.zip").read_bytes() for run in runs]
    output = tmp_path / "live-comparison"
    command = [sys.executable, str(ROOT / "mgba/evaluate_generalization.py"),
               "--previous-run", str(runs[0]), "--candidate-run", str(runs[1]),
               "--team-dir", str(folder), "--output-dir", str(output),
               "--episodes-per-opponent", "2", "--environments", "2"]
    dry = subprocess.run([*command, "--dry-run"], capture_output=True, text=True, timeout=90)
    assert dry.returncode == 0, dry.stderr
    assert not output.exists()
    done = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (output / "baseline-report.md").is_file()
    for label in ("previous", "candidate"):
        summary = json.loads((output / label / "summary.json").read_text())
        assert summary["attempted_battles"] == 6
    assert [(run / "checkpoints/best.zip").read_bytes() for run in runs] == before


# Real simulator fixture; ordinary unit cases above only need training imports.
from test_training_continue_cli import training
