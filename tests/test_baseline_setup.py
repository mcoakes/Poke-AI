"""Campaign baseline selection is reproducible even after attempt retention."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from firered_rl.baseline_setup import build_plan, save_plan, verify_plan
from firered_rl.curriculum import write_json
from firered_rl.evaluation import digest, validate_suite
from firered_rl.evaluation_pool import prepare_pool


def blob(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def team(path, name, species, moves=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(name=name, party=[dict(species=species, level=50,
              moves=moves or ["Tackle", "Protect", "Rest", "Sleep Talk"])]))
    return path


def evaluated(stage, number, score, *, revision=1, data=None):
    directory = stage / f"attempt-{number:03d}"
    candidate = blob(directory / "train/checkpoints/final.zip", data or f"weights-{number}".encode())
    initial = stage.parent / "inputs/start.zip"
    write_json(directory / "attempt.json", dict(resume=str(initial), resume_sha256=sha(initial)))
    write_json(directory / "train/summary.json", dict(status="completed"))
    write_json(directory / "report.json", dict(stage=int(stage.name.split("-")[-1]), attempt=number,
        status="hold", candidate=str(candidate), candidate_sha256=sha(candidate), win_rate=score,
        current={"all": {"wins": round(score*1000), "attempts": 1000, "win_rate": score}},
        settings_revision=revision, mode="mixed-evaluation"))
    return directory


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    start = blob(root / "inputs/start.zip", b"initial-smart")
    blob(root / "inputs/opponent.zip", b"initial-smart")
    write_json(root / "inputs/reward.json", {"win": 5000, "loss": -5000})
    trains = [team(root / f"inputs/teams/t{i}.json", f"train-{i}", species)
              for i, species in enumerate(("Charmander", "Squirtle", "Pikachu"))]
    valid = [team(root / f"inputs/validation/v{i}.json", f"valid-{i}", species)
             for i, species in enumerate(("Charmeleon", "Wartortle", "Raichu"))]
    config = dict(version=1, curriculum={"stages": [{"stage": 1}, {"stage": 2}]}, max_steps=200,
        teams=[str(p.relative_to(root)) for p in trains],
        team_groups=[[str(p.relative_to(root))] for p in trains],
        validation_groups=[[str(p.relative_to(root))] for p in valid],
        input_hashes={str(p.relative_to(root)): sha(p) for p in (root / "inputs").rglob("*") if p.is_file()})
    write_json(root / "campaign.json", config)
    stage = root / "stage-01"
    anchor = blob(stage / "anchor.zip", start.read_bytes())
    write_json(stage / "stage.json", dict(stage=1, anchor_sha256=sha(anchor)))
    evaluated(stage, 1, .99, revision=0)
    evaluated(stage, 2, .60)
    evaluated(stage, 3, .70)
    evaluated(stage, 4, .65)
    return root


def test_dry_plan_selects_latest_cohort_and_cumulative_disjoint_pools(campaign):
    before = {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    plan = build_plan(campaign, episodes_per_opponent=4)
    assert {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()} == before
    assert plan["candidates"]["starting"]["source"].endswith("stage-01/anchor.zip")
    assert plan["candidates"]["best"]["attempt"] == "attempt-003"
    assert plan["candidates"]["best"]["cohort"] == ["mixed-evaluation", 1]
    assert plan["candidates"]["latest"]["attempt"] == "attempt-004"
    assert plan["reward"]["value"]["damage_exponent"] == 2
    assert plan["stage_anchor"]["sha256"] == sha(campaign / "stage-01/anchor.zip")
    for split in plan["suites"].values():
        assert split["team_count"] == 1 and split["battles"] == 12
        suite = validate_suite(split["suite"])
        for group in ("random", "dumb", "smart"):
            selected = [c for c in suite["cases"] if c["opponent_group"] == group]
            assert len(selected) == 4
            assert sum(c["candidate_player"] == 1 for c in selected) == 2
    assert any("Dumb and at least one Smart" in warning for warning in plan["warnings"])
    assert any("intentional self-play" in warning for warning in plan["warnings"])


def test_retained_best_and_latest_interrupted_numeric_order(campaign):
    (campaign / "stage-01/attempt-003/train/checkpoints/final.zip").unlink()
    interrupted = campaign / "stage-01/attempt-1001"
    blob(interrupted / "train/checkpoints/latest.zip", b"newest-checkpoint")
    write_json(interrupted / "train/summary.json", {"status": "interrupted"})
    evaluated(campaign / "stage-01", 999, .1)
    plan = build_plan(campaign, episodes_per_opponent=2)
    assert plan["candidates"]["best"]["attempt"] == "attempt-004"
    assert plan["candidates"]["latest"]["attempt"] == "attempt-1001"
    assert not plan["candidates"]["latest"]["training_completed"]
    assert any("pruned" in warning for warning in plan["warnings"])
    assert any("incomplete/interrupted" in warning for warning in plan["warnings"])


def test_selection_after_actual_campaign_pruning(campaign):
    from firered_rl.retention import apply_campaign_retention, plan_campaign_retention
    apply_campaign_retention(plan_campaign_retention(campaign, keep_attempts=0, keep_best=0))
    plan = build_plan(campaign, episodes_per_opponent=2)
    assert plan["candidates"]["best"]["attempt"] == "attempt-004"
    assert plan["candidates"]["latest"]["attempt"] == "attempt-004"
    assert any("pruned" in warning for warning in plan["warnings"])


def test_saved_pool_models_preferred_to_live_config(campaign, tmp_path):
    attempt = campaign / "stage-01/attempt-004"
    dumb = blob(tmp_path / "weak.zip", b"weak")
    smart = blob(tmp_path / "reference.zip", b"smart-reference")
    extra = blob(tmp_path / "other.zip", b"smart-other")
    groups = [[campaign / f"inputs/validation/v{i}.json"] for i in range(3)]
    suite_path, _ = prepare_pool(attempt / "evaluation-pool", groups, stage=1, episodes=20,
                                seed=17, max_steps=200, dumb=dumb,
                                smart=[("reference", smart), ("other", extra)])
    report = json.loads((attempt / "report.json").read_text())
    report["evaluation_plan"] = str(suite_path)
    write_json(attempt / "report.json", report)
    # Selection must not consult these source files or discover new directory contents.
    dumb.unlink(); smart.unlink(); extra.unlink()
    config = json.loads((campaign / "campaign.json").read_text())
    config["evaluation"] = dict(smart_dir="/missing/live/directory", dumb_checkpoint={"path": "/missing/model", "sha256": "f"*64})
    write_json(campaign / "campaign.json", config)
    plan = build_plan(campaign, episodes_per_opponent=4)
    assert len([r for r in plan["opponents"].values() if r["group"] == "smart"]) == 2
    assert Path(plan["opponents"]["dumb"]["source"]).is_relative_to(attempt)
    for split in plan["suites"].values():
        assert sum(c["opponent_id"] == "smart_1" for c in split["suite"]["cases"]) == 2
        assert sum(c["opponent_id"] == "smart_2" for c in split["suite"]["cases"]) == 2


def test_stage_two_uses_its_anchor_and_cumulative_teams(campaign):
    stage = campaign / "stage-02"
    anchor = blob(stage / "anchor.zip", b"stage-two-start")
    write_json(stage / "stage.json", {"stage": 2, "anchor_sha256": sha(anchor)})
    evaluated(stage, 1, .7)
    plan = build_plan(campaign, stage=2, episodes_per_opponent=2)
    assert plan["candidates"]["starting"]["sha256"] == sha(anchor)
    assert plan["suites"]["training"]["team_count"] == 2
    assert plan["suites"]["validation"]["team_count"] == 2


def test_missing_validation_requires_disjoint_override(campaign, tmp_path):
    config = json.loads((campaign / "campaign.json").read_text())
    config.pop("validation_groups")
    config.pop("team_groups")  # Legacy one-file stage schema.
    write_json(campaign / "campaign.json", config)
    with pytest.raises(ValueError, match="--validation-dir"):
        build_plan(campaign, episodes_per_opponent=2)
    directory = tmp_path / "validation"
    team(directory / "nested/valid.json", "new-name", "Charmeleon")
    assert build_plan(campaign, episodes_per_opponent=2, validation_dir=directory)["suites"]["validation"]["team_count"] == 1
    team(directory / "duplicate.json", "renamed", "Charmander", ["Sleep Talk", "Rest", "Protect", "Tackle"])
    with pytest.raises(ValueError, match="duplicate party"):
        build_plan(campaign, episodes_per_opponent=2, validation_dir=directory)


def test_snapshot_resume_independent_of_campaign_and_candidate_aliases(campaign, tmp_path):
    candidate = campaign / "stage-01/attempt-004/train/checkpoints/final.zip"
    plan = build_plan(campaign, best_checkpoint=candidate, episodes_per_opponent=4)
    output = tmp_path / "baseline"
    saved = save_plan(plan, output)
    assert saved["candidates"]["best"]["path"] == saved["candidates"]["latest"]["path"]
    assert len(list((output / "snapshots/models").glob("*.zip"))) == 2
    shutil.rmtree(campaign)
    assert verify_plan(output) == saved
    for row in saved["suites"].values():
        suite = json.loads(Path(row["path"]).read_text())
        assert digest(suite) == row["suite_sha256"]
        assert len(suite["cases"]) == row["battles"] == 12
        for opponent in suite["opponents"].values():
            if opponent["kind"] == "checkpoint":
                assert Path(opponent["path"]).is_relative_to(output)


@pytest.mark.parametrize("what", ["plan", "suite", "model", "reward"])
def test_resume_rejects_snapshot_tampering(campaign, tmp_path, what):
    output = tmp_path / "baseline"
    saved = save_plan(build_plan(campaign, episodes_per_opponent=2), output)
    if what == "plan":
        changed = deepcopy(saved)
        changed["stage"] = 2
        write_json(output / "baseline.json", changed)
    elif what == "suite":
        path = Path(saved["suites"]["training"]["path"])
        value = json.loads(path.read_text())
        value["cases"][0]["seed"] += 1
        write_json(path, value)
    else:
        path = Path(saved["reward"]["path"] if what == "reward" else saved["candidates"]["best"]["path"])
        path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_plan(output)


def test_source_change_between_preview_and_save_rejected_without_output(campaign, tmp_path):
    plan = build_plan(campaign, episodes_per_opponent=2)
    Path(plan["candidates"]["best"]["source"]).write_bytes(b"changed")
    output = tmp_path / "baseline"
    with pytest.raises(ValueError, match="changed since preview"):
        save_plan(plan, output)
    assert not output.exists()


def test_existing_output_lock_is_allowed_and_other_files_are_protected(campaign, tmp_path):
    plan = build_plan(campaign, episodes_per_opponent=2)
    output = tmp_path / "baseline"
    blob(output / ".lock", b"")
    saved = save_plan(plan, output)
    assert verify_plan(output) == saved
    with pytest.raises(ValueError, match="must be empty"):
        save_plan(plan, output)
    with pytest.raises(ValueError, match="outside the source campaign"):
        save_plan(plan, campaign / "baseline")


def test_atomic_publication_embeds_final_paths_and_survives_source_removal(campaign, tmp_path):
    plan = build_plan(campaign, episodes_per_opponent=2)
    staging = tmp_path / "staged-baseline"
    published = tmp_path / "published-baseline"
    saved = save_plan(plan, staging, published_dir=published)
    assert not published.exists()
    assert (staging / "baseline.json").is_file()
    assert all(Path(row["path"]).is_relative_to(published) for row in saved["candidates"].values())
    assert Path(saved["reward"]["path"]).is_relative_to(published)
    assert all(Path(row["path"]).is_relative_to(published) for row in saved["suites"].values())
    staging.rename(published)
    shutil.rmtree(campaign)
    assert verify_plan(published) == saved


def test_atomic_publication_refuses_logical_campaign_destination(campaign, tmp_path):
    staging = tmp_path / "staged-baseline"
    with pytest.raises(ValueError, match="outside the source campaign"):
        save_plan(build_plan(campaign, episodes_per_opponent=2), staging,
                  published_dir=campaign / "baseline")
    assert not staging.exists()


def test_suite_generation_is_reproducible_and_models_share_identical_cases(campaign, tmp_path):
    first = build_plan(campaign, episodes_per_opponent=20, seed=11)
    second = build_plan(campaign, episodes_per_opponent=20, seed=11)
    assert first == second
    other = build_plan(campaign, episodes_per_opponent=20, seed=12)
    assert first["suites"]["training"]["suite"]["cases"] != other["suites"]["training"]["suite"]["cases"]
    for split in first["suites"].values():
        cases = split["suite"]["cases"]
        assert len({row["episode"] for row in cases}) == len(cases)
        for a, b in zip(cases[::2], cases[1::2]):
            assert (a["candidate_player"], b["candidate_player"]) == (1, 2)
            assert all(a[key] == b[key] for key in ("matchup", "seed", "opponent_id", "opponent_seed", "assets"))


@pytest.mark.parametrize("target", ["inputs/teams/t0.json", "stage-01/anchor.zip",
                                    "stage-01/attempt-003/train/checkpoints/final.zip"])
def test_pinned_source_checksum_change_rejected(campaign, target):
    (campaign / target).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        build_plan(campaign, episodes_per_opponent=2)


def test_override_selectors_resolved_once(campaign, monkeypatch):
    import firered_rl.baseline_setup as module
    seen = []
    selected = campaign / "inputs/start.zip"
    original = module.resolve_checkpoints
    def resolve(values, project):
        seen.append(values)
        return original([selected if str(v) in ("latest", "later") else v for v in values], project)
    monkeypatch.setattr(module, "resolve_checkpoints", resolve)
    plan = build_plan(campaign, episodes_per_opponent=2, starting_checkpoint="latest", best_checkpoint="later",
                      latest_checkpoint=selected, dumb_checkpoint=selected, smart_checkpoint=selected)
    assert len(seen) == 1
    assert len({v["sha256"] for v in plan["candidates"].values()}) == 1


def test_relocated_campaign_uses_local_known_snapshots(campaign, tmp_path):
    moved = tmp_path / "moved-campaign"
    campaign.rename(moved)
    plan = build_plan(moved, episodes_per_opponent=2)
    assert Path(plan["candidates"]["best"]["source"]).is_relative_to(moved)
    assert Path(plan["candidates"]["latest"]["source"]).is_relative_to(moved)


@pytest.mark.parametrize("options", [dict(stage=0), dict(stage=3), dict(episodes_per_opponent=3),
                                    dict(episodes_per_opponent=0), dict(max_steps=0)])
def test_invalid_parameters(campaign, options):
    with pytest.raises(ValueError):
        build_plan(campaign, **options)
