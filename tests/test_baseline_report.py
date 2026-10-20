from copy import deepcopy
import hashlib
import json

import pytest

from firered_rl.baseline_report import (build_baseline_report, format_baseline_markdown,
    load_baseline_evaluation, write_baseline_report)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def make_suite(split, clusters=60):
    asset = {"team": {"name": split, "party": [{"species": "Pikachu", "level": 50,
                                               "moves": ["Thunderbolt"]}]}}
    key = digest(asset)
    opponents = {"random": {"kind": "random", "group": "random", "label": "Random"}}
    for group in ("dumb", "smart"):
        opponents[group] = dict(kind="checkpoint", group=group, label=group.title(),
                               path=f"/archived/{group}.zip", sha256=digest(group))
    cases = []
    for matchup in range(1, clusters+1):
        group = ("random", "dumb", "smart")[(matchup-1) % 3]
        for side in (1, 2):
            cases.append(dict(episode=len(cases)+1, matchup=matchup, assets=[key, key], seed=matchup*7,
                              candidate_player=side, opponent_id=group, opponent_group=group,
                              opponent_seed=matchup*11))
    return dict(version=1, backend="showdown", seed=123, swap_sides=True, max_steps=200,
                assets={key: asset}, opponents=opponents, cases=cases)


def evaluation(directory, suite, role, outcomes=None):
    outcomes = outcomes or (lambda case: "won" if case["matchup"] % 3 == 0 else "lost")
    rows = []
    for case in suite["cases"]:
        outcome = outcomes(case)
        rows.append(dict(case, candidate_outcome=outcome, truncated=outcome == "truncated",
                         terminated=outcome != "truncated", decisions=10,
                         candidate_reward=5000 if outcome == "won" else -5000,
                         opponent_reward=-5000 if outcome == "won" else 5000))
    counts = {kind: sum(row["candidate_outcome"] == kind for row in rows)
              for kind in ("won", "lost", "draw", "truncated")}
    summary = dict(status="completed", suite_sha256=digest(suite), elapsed_seconds=12,
        requested_battles=len(rows), attempted_battles=len(rows),
        completed_battles=len(rows)-counts["truncated"], candidate_wins=counts["won"],
        opponent_wins=counts["lost"], draws=counts["draw"], truncated=counts["truncated"])
    config = dict(suite_sha256=digest(suite), backend="showdown", deterministic=True,
                  reward={"win": 1, "loss": -1}, models={"candidate": dict(path=f"/archived/{role}.zip",
                      sha256=digest(role), training_timesteps=100 if role == "starting" else 200)},
                  opponents=deepcopy(suite["opponents"]))
    write(directory / "suite.json", suite)
    write(directory / "config.json", config)
    write(directory / "summary.json", summary)
    # Results normally arrive out of order from parallel workers.
    (directory / "episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in reversed(rows)))
    return directory


@pytest.fixture
def runs(tmp_path):
    data = {label: {} for label in ("starting", "best", "latest")}
    for split in ("training", "validation"):
        suite = make_suite(split)
        for label in data:
            data[label][split] = evaluation(tmp_path / label / split, suite, label)
    return data


def report(runs, **options):
    return build_baseline_report(runs, bootstrap_samples=200, seed=321, **options)


def test_reproducible_paired_report_counts_reward_throughput_and_files(runs, tmp_path):
    first = report(runs)
    assert report(runs) == first
    all_stats = first["splits"]["validation"]["candidates"]["starting"]["groups"]["all"]
    assert all_stats["attempts"] == 120 and all_stats["matchup_clusters"] == 60
    assert all_stats["wins"] == 40 and all_stats["win_rate"] == pytest.approx(1/3)
    assert all_stats["candidate_reward_mean"] == pytest.approx(-5000/3)
    assert all_stats["decisions_mean"] == 10
    candidate = first["splits"]["training"]["candidates"]["best"]
    assert candidate["battles_per_hour"] == 36000
    assert candidate["by_candidate_side"]["1"]["attempts"] == 60
    assert candidate["by_opponent"]["random"]["attempts"] == 40
    comparison = first["comparisons"]["best"]["validation"]["groups"]["all"]
    assert comparison["conclusion"] == "no_clear_change" and comparison["delta_win_rate"] == 0
    json_path, md_path = write_baseline_report(first, tmp_path / "report")
    assert json.loads(json_path.read_text()) == first
    assert "Random" in md_path.read_text() and "selection bias" in md_path.read_text()
    assert "Paired interval" in format_baseline_markdown(first)


def test_bootstrap_keeps_two_seats_together(tmp_path):
    suite = make_suite("training")
    runs = {
        "starting": {"training": evaluation(tmp_path / "a", suite, "starting",
                                            lambda c: "won" if c["candidate_player"] == 1 else "lost")},
        "latest": {"training": evaluation(tmp_path / "b", suite, "latest",
                                          lambda c: "won" if c["candidate_player"] == 2 else "lost")}}
    result = report(runs)
    for candidate in ("starting", "latest"):
        interval = result["splits"]["training"]["candidates"][candidate]["groups"]["all"]["win_rate_ci"]
        # Every cluster has exactly one win; independent-row resampling would widen this.
        assert interval["low"] == interval["high"] == .5
        assert interval["status"] == "degenerate_observed_sample"
    delta = result["comparisons"]["latest"]["training"]["groups"]["all"]
    assert delta["delta_win_rate_ci"]["low"] == delta["delta_win_rate_ci"]["high"] == 0
    assert delta["conclusion"] == "no_clear_change"


@pytest.mark.parametrize("wins,expected", [(True, "improvement"), (False, "regression")])
def test_paired_delta_direction(tmp_path, wins, expected):
    suite = make_suite("validation")
    starting = "lost" if wins else "won"
    latest = "won" if wins else "lost"
    runs = {"starting": {"validation": evaluation(tmp_path / "a", suite, "starting", lambda _: starting)},
            "latest": {"validation": evaluation(tmp_path / "b", suite, "latest", lambda _: latest)}}
    result = report(runs)
    comparison = result["comparisons"]["latest"]["validation"]["groups"]["smart"]
    assert comparison["conclusion"] == expected
    assert comparison["delta_win_rate"] == (1 if wins else -1)


def test_small_groups_are_not_given_confident_verdicts(tmp_path):
    suite = make_suite("validation", clusters=6)
    runs = {"starting": {"validation": evaluation(tmp_path / "a", suite, "starting", lambda _: "lost")},
            "latest": {"validation": evaluation(tmp_path / "b", suite, "latest", lambda _: "won")}}
    result = report(runs)
    value = result["comparisons"]["latest"]["validation"]["groups"]["all"]
    assert value["delta_win_rate"] == 1 and value["conclusion"] == "insufficient_data"
    assert value["delta_win_rate_ci"]["low"] is None
    assert "insufficient matchups" in format_baseline_markdown(result)


def test_truncations_and_draws_are_nonwins(tmp_path):
    suite = make_suite("training")
    result_dirs = {}
    for label in ("starting", "latest"):
        result_dirs[label] = {"training": evaluation(tmp_path / label, suite, label,
            lambda case: "truncated" if case["candidate_player"] == 2 else
                         "draw" if case["matchup"] % 2 else "won")}
    result = report(result_dirs)
    value = result["splits"]["training"]["candidates"]["latest"]["groups"]["all"]
    assert value["truncated"] == 60 and value["draws"] == 30 and value["wins"] == 30
    assert value["win_rate"] == .25 and value["score"] == .375
    assert result["splits"]["training"]["candidates"]["latest"]["battles_per_hour"] == 18000


def test_identical_weights_roles_are_marked_and_reused(runs):
    runs["best"] = dict(runs["starting"])
    result = report(runs)
    assert result["candidates"]["best"]["identical_weights_to"] == ["starting"]
    assert result["comparisons"]["best"]["validation"]["groups"]["all"]["delta_win_rate"] == 0


@pytest.mark.parametrize("problem", ["missing", "duplicate", "incomplete", "summary", "reward", "opponent",
                                     "candidate", "flags", "nonfinite", "swaps", "cluster"])
def test_incoherent_results_fail_closed(runs, problem):
    path = runs["latest"]["validation"]
    if problem in ("missing", "duplicate", "flags", "nonfinite"):
        rows = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
        if problem == "missing":
            rows.pop()
        elif problem == "duplicate":
            rows[-1] = rows[0]
        elif problem == "flags":
            rows[0]["truncated"] = True
        else:
            rows[0]["candidate_reward"] = float("nan")
        (path / "episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    elif problem in ("incomplete", "summary"):
        value = json.loads((path / "summary.json").read_text())
        value["status" if problem == "incomplete" else "candidate_wins"] = "running" if problem == "incomplete" else 99
        write(path / "summary.json", value)
    elif problem in ("swaps", "cluster"):
        value = json.loads((path / "suite.json").read_text())
        if problem == "swaps":
            value["swap_sides"] = False
        else:
            value["cases"][0]["matchup"] = value["cases"][2]["matchup"]
        write(path / "suite.json", value)
    else:
        value = json.loads((path / "config.json").read_text())
        if problem == "reward":
            value["reward"]["win"] = 2
        elif problem == "opponent":
            value["opponents"]["smart"]["sha256"] = digest("other opponent")
        else:
            value["models"]["candidate"]["sha256"] = digest("different weights")
        write(path / "config.json", value)
    with pytest.raises(ValueError):
        report(runs)


def test_different_but_internally_valid_suites_are_rejected(runs):
    path = runs["latest"]["validation"]
    suite = json.loads((path / "suite.json").read_text())
    for case in suite["cases"][:2]:
        case["seed"] += 1
    evaluation(path, suite, "latest")
    with pytest.raises(ValueError, match="identical suite"):
        report(runs)


def test_identical_weights_with_inconsistent_outcomes_are_rejected(runs):
    for split in runs["latest"]:
        path = runs["latest"][split]
        suite = json.loads((path / "suite.json").read_text())
        evaluation(path, suite, "starting", lambda _: "won")
    with pytest.raises(ValueError, match="identical checkpoint"):
        report(runs)


def test_public_loader_does_not_need_live_models(runs):
    loaded = load_baseline_evaluation(runs["starting"]["training"])
    assert loaded["candidate"]["path"].startswith("/archived/")
    assert loaded["rows"][0]["episode"] == 1
    assert len(loaded["rows"]) == 120


def test_summary_mean_inconsistency_is_rejected(runs):
    path = runs["latest"]["validation"] / "summary.json"
    value = json.loads(path.read_text())
    value["candidate_reward_mean"] = 987654
    write(path, value)
    with pytest.raises(ValueError, match="candidate_reward_mean"):
        report(runs)


def test_selection_notes_are_rendered(runs):
    value = report(runs)
    value["selection_warnings"] = ["Best selected from retained candidates only."]
    assert "Best selected from retained candidates only." in format_baseline_markdown(value)
