"""Retention preserves continuation and compact evaluation contracts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from firered_rl.retention import (apply_campaign_retention, campaign_retention_lock,
                                  plan_campaign_retention)

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def blob(path, value=b"model"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attempt(stage, number, *, score=.5, revision=0, status="hold", unfinished=False):
    run = stage.parent
    directory = stage / f"attempt-{number:03d}"
    start = run / "inputs/start.zip"
    write(directory / "attempt.json", dict(resume=str(start), resume_sha256=sha(start)))
    for name in ("latest.zip", "final.zip", "ppo_000000100.zip"):
        blob(directory / "train/checkpoints" / name, f"weights-{number}".encode())
    write(directory / "train/config.json", {"seed": number})
    write(directory / "train/summary.json", {"status": "running" if unfinished else "completed"})
    write(directory / "train/switching.json", {"learner_voluntary_switch_rate": .1})
    blob(directory / "train/ppo/progress.csv", b"timesteps,loss\n100,0\n")
    for name in ("episodes.jsonl", "updates.jsonl", "opponent-switches.jsonl"):
        blob(directory / "train" / name, b'{"episode": 1}\n')
    debug = directory / "train/pair-1"
    for name in ("battle-readable.log", "battle.log", "showdown-stderr.log", "workers.json"):
        blob(debug / name, b"diagnostics"*5)
    for name in ("gym-steps.jsonl", "teams.json", "showdown-protocol.log"):
        blob(debug / "episode-000001" / name, b"large-details"*100)
    # Frozen evaluation pool paths must remain even when a candidate is pruned.
    model = blob(directory / "evaluation-pool/models/frozen.zip", b"frozen-opponent")
    write(directory / "evaluation-pool/plan.json", {"frozen": str(model)})
    write(directory / "evaluation-pool/suite.json", {"frozen": str(model), "sha256": sha(model)})
    evaluation = directory / "eval-current-001"
    for name in ("config.json", "suite.json", "summary.json"):
        write(evaluation / name, {"retained": True})
    blob(evaluation / "episodes.jsonl", b'{"episode": 1}\n')
    blob(evaluation / "pair-1/episode-000001/gym-steps.jsonl", b"evaluation-details"*100)
    if not unfinished:
        candidate = directory / "train/checkpoints/final.zip"
        write(directory / "report.json", dict(stage=int(stage.name.split("-")[1]),
              attempt=number, status=status, candidate=str(candidate), candidate_sha256=sha(candidate),
              win_rate=score, current={"all": {"wins": round(score*100), "attempts": 100,
                      "completed": 100, "win_rate": score}}, mode="mixed-evaluation",
              settings_revision=revision))
    return directory


@pytest.fixture
def campaign(tmp_path):
    run = tmp_path / "campaign"
    start = blob(run / "inputs/start.zip", b"start")
    opponent = blob(run / "inputs/opponent.zip", b"opponent")
    team = write(run / "inputs/teams/team.json", {"name": "fixture"})
    config = dict(version=1, curriculum={"stages": [{"stage": 1}]},
                  teams=["inputs/teams/team.json", "inputs/teams/team.json"],
                  input_hashes={str(p.relative_to(run)): sha(p) for p in (start, opponent, team)},
                  opponent_mode="fixed", evaluation={})
    write(run / "campaign.json", config)
    stage = run / "stage-01"
    anchor = blob(stage / "anchor.zip", start.read_bytes())
    fixed = blob(stage / "opponent.zip", opponent.read_bytes())
    write(stage / "stage.json", dict(stage=1, source=str(start), anchor_sha256=sha(anchor),
          opponent=dict(mode="fixed", path=str(fixed), sha256=sha(fixed), source_stage=None,
                        source=str(opponent))))
    for i, score in enumerate((.8, .4, .7, .6), 1):
        attempt(stage, i, score=score)
    write(stage / "result.json", json.loads((stage / "attempt-004/report.json").read_text()))
    return run


def test_plan_is_read_only_and_apply_keeps_latest_best_metadata_and_pools(campaign):
    stage = campaign / "stage-01"
    untouched = {p: p.read_bytes() for p in campaign.rglob("*") if p.is_file()}
    plan = plan_campaign_retention(campaign, keep_attempts=1, keep_best=1)
    assert all(p.read_bytes() == data for p, data in untouched.items())
    assert plan["files"] > 0 and plan["totals"]["battle_debug"]["bytes"] > 0
    result = apply_campaign_retention(plan)
    assert result["files"] == plan["files"] and result["bytes"] == plan["bytes"]
    for i in range(1, 5):
        directory = stage / f"attempt-{i:03d}"
        assert (directory / "train/checkpoints/final.zip").exists() == (i in (1, 4))
        assert not (directory / "train/checkpoints/latest.zip").exists()
        assert not (directory / "train/pair-1").exists()
        assert not (directory / "train/episodes.jsonl").exists()
        for relative in ("attempt.json", "report.json", "train/config.json", "train/summary.json",
                         "train/switching.json", "train/ppo/progress.csv", "eval-current-001/episodes.jsonl",
                         "evaluation-pool/models/frozen.zip", "evaluation-pool/plan.json"):
            path = directory / relative
            assert path.read_bytes() == untouched[path]
    # Repeated pruning is safe even though old report candidates were removed.
    assert plan_campaign_retention(campaign, keep_attempts=1, keep_best=1)["files"] == 0
    history = json.loads((campaign / "retention-history.jsonl").read_text())
    assert history["files"] == result["files"] and history["status"] == "completed"


def test_promoted_original_candidate_is_always_kept(campaign):
    stage = campaign / "stage-01"
    path = stage / "attempt-002/report.json"
    report = json.loads(path.read_text())
    report["status"] = "promote"
    write(path, report)
    write(stage / "result.json", report)
    plan = plan_campaign_retention(campaign, keep_attempts=0, keep_best=0)
    apply_campaign_retention(plan)
    assert Path(report["candidate"]).is_file()
    assert (stage / "attempt-004/train/checkpoints/final.zip").is_file()


def test_evaluation_weight_copies_obey_campaign_retention(campaign):
    stage = campaign / "stage-01"
    for number in range(1, 5):
        directory = stage / f"attempt-{number:03d}"
        final = directory / "train/checkpoints/final.zip"
        evaluation = directory / "eval-current-001"
        snapshot = blob(evaluation / "candidate.zip", final.read_bytes())
        write(evaluation / "config.json", dict(models={"candidate": dict(
            path=str(final), sha256=sha(final), snapshot_path=str(snapshot))}))
    # A candidate copy explicitly used as a future resume point is protected.
    resume = stage / "attempt-002/eval-current-001/candidate.zip"
    pending = attempt(stage, 5, unfinished=True)
    write(pending / "attempt.json", dict(resume=str(resume), resume_sha256=sha(resume)))
    plan = plan_campaign_retention(campaign, keep_attempts=1, keep_best=1)
    apply_campaign_retention(plan)
    assert resume.is_file()
    assert not (stage / "attempt-001/eval-current-001/candidate.zip").exists()
    assert (stage / "attempt-001/train/checkpoints/final.zip").exists()  # Best candidate remains.
    assert (stage / "attempt-001/eval-current-001/config.json").exists()


def test_best_retention_is_per_evaluation_rule_cohort(campaign):
    stage = campaign / "stage-01"
    path = stage / "attempt-003/report.json"
    value = json.loads(path.read_text())
    value["settings_revision"] = 1
    write(path, value)
    apply_campaign_retention(plan_campaign_retention(campaign, keep_attempts=0, keep_best=1))
    assert all((stage / f"attempt-{i:03d}/train/checkpoints/final.zip").is_file() for i in (1, 3, 4))
    assert not (stage / "attempt-002/train/checkpoints/final.zip").exists()


def test_latest_unfinished_checkpoint_fallback_and_debug_survive(campaign):
    stage = campaign / "stage-01"
    old = attempt(stage, 5, unfinished=True)
    newest = attempt(stage, 6, unfinished=True)
    fallback = stage / "attempt-002/train/checkpoints/final.zip"
    marker = json.loads((newest / "attempt.json").read_text())
    marker.update(resume=str(fallback), resume_sha256=sha(fallback))
    write(newest / "attempt.json", marker)
    plan = plan_campaign_retention(campaign, keep_attempts=0, keep_best=0)
    apply_campaign_retention(plan)
    assert not (old / "train/pair-1").exists()
    assert (newest / "train/pair-1/episode-000001/gym-steps.jsonl").is_file()
    assert (newest / "train/checkpoints/latest.zip").is_file()
    assert (newest / "train/checkpoints/final.zip").is_file()
    assert fallback.is_file()


def test_unexpected_files_symlinks_and_outside_files_are_untouched(campaign, tmp_path):
    pair = campaign / "stage-01/attempt-001/train/pair-1"
    custom = blob(pair / "my-notes.txt", b"keep me")
    user_model = blob(pair.parent / "checkpoints/personal-model.zip", b"personal model")
    nested = blob(pair / "episode-000001/custom/data.txt", b"keep nested")
    elsewhere = blob(tmp_path / "outside.txt", b"outside")
    (pair / "episode-000002").symlink_to(tmp_path, target_is_directory=True)
    (pair / "episode-000001/external.log").symlink_to(elsewhere)
    apply_campaign_retention(plan_campaign_retention(campaign))
    assert custom.read_bytes() == b"keep me" and nested.read_bytes() == b"keep nested"
    assert user_model.read_bytes() == b"personal model"
    assert elsewhere.read_bytes() == b"outside"
    assert (pair / "episode-000002").is_symlink()
    assert (pair / "episode-000001/external.log").is_symlink()


@pytest.mark.parametrize("change", ["file", "metadata", "input", "symlink"])
def test_changed_plan_or_bad_metadata_fails_closed(campaign, change, tmp_path):
    plan = plan_campaign_retention(campaign)
    target = campaign / "stage-01/attempt-001/train/pair-1/battle.log"
    if change == "file":
        target.write_text("changed data")
    elif change == "metadata":
        write(campaign / "stage-01/attempt-001/report.json", {"status": "broken"})
    elif change == "input":
        (campaign / "inputs/start.zip").write_bytes(b"changed input")
    else:
        target.unlink()
        target.symlink_to(blob(tmp_path / "outside.bin", b"external"))
    with pytest.raises(ValueError):
        apply_campaign_retention(plan)
    assert (campaign / "stage-01/attempt-002/train/checkpoints/final.zip").exists()
    assert not (campaign / "retention-history.jsonl").exists()


def test_active_campaign_lock_refuses_cleanup(campaign):
    with campaign_retention_lock(campaign):
        with pytest.raises(ValueError, match="active"):
            apply_campaign_retention(plan_campaign_retention(campaign))


def test_keep_all_candidates_and_debug(campaign):
    plan = plan_campaign_retention(campaign, keep_attempts=-1, keep_best=-1, keep_debug_attempts=-1)
    assert plan["totals"]["battle_debug"]["files"] == 0
    assert plan["totals"]["training_details"]["files"] == 0
    apply_campaign_retention(plan)
    assert len(list(campaign.glob("stage-*/attempt-*/train/checkpoints/final.zip"))) == 4


def test_cli_dry_run_then_apply(campaign):
    command = [sys.executable, str(ROOT / "tools/prune_campaign.py"), "--run-dir", str(campaign)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0 and "DRY RUN" in result.stdout
    assert (campaign / "stage-01/attempt-001/train/pair-1/battle.log").exists()
    result = subprocess.run(command + ["--apply"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Removed" in result.stdout


def test_campaign_continuation_track_record_and_evaluation_reuse(campaign):
    pytest.importorskip("gymnasium")
    spec = importlib.util.spec_from_file_location("retention_campaign", ROOT / "mgba/run_curriculum.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    from firered_rl.curriculum import evaluation_rates
    from firered_rl.evaluation import digest
    from firered_rl.evaluation_pool import verify_pool
    stage = campaign / "stage-01"
    attempt_dir = stage / "attempt-004"
    config = json.loads((campaign / "campaign.json").read_text())
    suite = dict(version=1, backend="showdown", seed=1, max_steps=10, swap_sides=False,
                 assets={"team": {"team": {"name": "Team", "party": [
                     {"species": "Pikachu", "level": 50, "moves": ["Thunderbolt", "Thunder", "Quick Attack", "Growl"]}]}}},
                 cases=[dict(episode=1, matchup=1, assets=["team", "team"], seed=1234,
                             candidate_player=1, opponent_id="dumb", opponent_group="dumb", opponent_seed=1)])
    # Asset IDs are content hashes in real suites.
    asset = suite["assets"].pop("team")
    key = digest(asset)
    suite["assets"][key] = asset
    suite["cases"][0]["assets"] = [key, key]
    frozen = attempt_dir / "evaluation-pool/models/frozen.zip"
    suite["opponents"] = {"dumb": dict(kind="checkpoint", group="dumb", label="Dumb",
                                       path=str(frozen), sha256=sha(frozen))}
    pool_path = write(attempt_dir / "evaluation-pool/suite.json", suite)
    evaluation = attempt_dir / "eval-current-001"
    write(evaluation / "config.json", dict(suite_sha256=digest(suite),
        models={"candidate": {"sha256": sha(attempt_dir / "train/checkpoints/final.zip")}}))
    write(evaluation / "summary.json", dict(status="completed", suite_sha256=digest(suite)))
    row = dict(suite["cases"][0], candidate_outcome="won", terminated=True, truncated=False)
    (evaluation / "episodes.jsonl").write_text(json.dumps(row)+"\n")
    before = evaluation_rates(evaluation, suite)
    apply_campaign_retention(plan_campaign_retention(campaign, keep_attempts=1, keep_best=1))
    obj = runner.Campaign(campaign, config, runner.EXECUTION)
    assert obj.next_stage() == 1
    obj.track_record()
    assert "attempt" in (campaign / "evaluations.csv").read_text()
    assert len((campaign / "evaluations.csv").read_text().splitlines()) == 5
    assert verify_pool(suite) == suite
    assert evaluation_rates(evaluation, suite) == before
    selected, rates = obj.evaluation(attempt_dir, "eval-current", pool_path,
                                     attempt_dir / "train/checkpoints/final.zip")
    assert selected == evaluation and rates == before
    # Promoted-stage resolution checks original candidate plus stage opponent.
    report = json.loads((attempt_dir / "report.json").read_text())
    report.update(status="promote", opponent=json.loads((stage / "stage.json").read_text())["opponent"])
    write(stage / "result.json", report)
    assert obj.next_stage() == 2


def test_rotated_known_logs_and_completion_marker_are_removed(campaign):
    pair = campaign / "stage-01/attempt-001/train/pair-1"
    for relative in ("battle.log.1", "battle-readable.log.2", "episode-000001/gym-steps.jsonl.3",
                     "episode-000001/.completed.json"):
        blob(pair / relative, b"known rotation")
    apply_campaign_retention(plan_campaign_retention(campaign))
    assert not pair.exists()


def test_repeated_hold_prune_resume_promote_and_next_stage(tmp_path):
    spec = importlib.util.spec_from_file_location("retention_resume", ROOT / "mgba/run_curriculum.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    from firered_rl.evaluation import digest, summary
    source = blob(tmp_path / "start.zip", b"starting model")
    opponent = blob(tmp_path / "frozen.zip", b"frozen model")
    run = tmp_path / "run"
    run.mkdir()
    config = cli.initialize(argparse.Namespace(resume=source, opponent_checkpoint=opponent,
        curriculum=ROOT / "config/curricula/type_progression/curriculum.json", eval_episodes=4,
        max_steps=100, promotion_rules=None, reward_config=None, seed=7,
        train_battles=10, promotion_win_rate=.75, opponent_mode="previous-stage"), run)
    starts = []
    evaluations = []

    def execute(command, root):
        def arg(name):
            return Path(command[command.index(name)+1])
        output = arg("--output-dir")
        if Path(command[1]).name == "train_ppo.py":
            entering = arg("--resume").read_bytes()
            starts.append(entering)
            for name in ("latest.zip", "final.zip", "ppo_000010000.zip"):
                blob(output / "checkpoints" / name, entering + b" trained")
            write(output / "summary.json", {"status": "completed"})
            blob(output / "pair-1/episode-000001/gym-steps.jsonl", b"large debug"*100)
        else:
            suite = json.loads(arg("--suite").read_text())
            win = len(starts) == 3
            rows = [dict(case, candidate_outcome="won" if win else "lost", terminated=True,
                    truncated=False, candidate_reward=0, opponent_reward=0, decisions=1)
                    for case in suite["cases"]]
            write(output / "config.json", dict(suite_sha256=digest(suite), models={
                "candidate": {"sha256": sha(arg("--checkpoint"))},
                "opponent": {"sha256": sha(arg("--opponent-checkpoint"))}}))
            write(output / "suite.json", suite)
            (output / "episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            write(output / "summary.json", summary(rows, requested=len(rows), elapsed=1,
                        status="completed", suite_hash=digest(suite)))
            evaluations.append(output)
    reports = []
    for expected in ("hold", "hold", "promote"):
        campaign = cli.Campaign(run, config, cli.EXECUTION, execute=execute)
        report = campaign.run_stage(1)
        assert report["status"] == expected
        reports.append(report)
        apply_campaign_retention(plan_campaign_retention(run, keep_attempts=0, keep_best=0))
        assert Path(report["candidate"]).is_file()
        assert (Path(report["evaluations"]["current"]) / "episodes.jsonl").is_file()
    assert starts == [b"starting model", b"starting model trained", b"starting model trained trained"]
    assert not Path(reports[0]["candidate"]).exists()
    assert not Path(reports[1]["candidate"]).exists()
    campaign = cli.Campaign(run, config, cli.EXECUTION, execute=execute)
    assert campaign.next_stage() == 2
    stage2 = campaign.run_stage(2)
    assert stage2["status"] == "hold"
    assert starts[-1] == Path(reports[2]["candidate"]).read_bytes()
    apply_campaign_retention(plan_campaign_retention(run, keep_attempts=0, keep_best=0))
    assert campaign.next_stage() == 2
    campaign.track_record()
    assert len((run / "evaluations.csv").read_text().splitlines()) == 5
    assert Path(reports[2]["candidate"]).is_file()
    assert (run / "stage-02/opponent.zip").read_bytes() == Path(reports[2]["candidate"]).read_bytes()
    assert len(evaluations) == 4
