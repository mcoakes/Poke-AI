"""Matched, seat-paired baseline statistics from completed evaluation artifacts.

Each candidate maps split names to evaluate_ppo output directories. Comparisons
must replay the same complete suite for every candidate. Bootstrap units are
matchup IDs (both candidate seats together), never individual battle rows.
Only NumPy and the standard library are needed to read and summarize artifacts.
"""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

_SHA = re.compile(r"[0-9a-f]{64}")
_GROUPS = ("random", "dumb", "smart")
_OUTCOMES = ("won", "lost", "draw", "truncated")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _read(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=_invalid_constant)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read baseline artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"baseline artifact must contain an object: {path}")
    return value


def _number(value, name, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be a finite {'nonnegative ' if nonnegative else ''}number")
    return value


def _checksum(value, name):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{name} needs a valid SHA-256")
    return value


def _suite(suite):
    if suite.get("version") != 1 or suite.get("backend") not in ("showdown", "mgba"):
        raise ValueError("unsupported baseline evaluation suite")
    if suite.get("swap_sides") is not True:
        raise ValueError("baseline evaluation requires side-swapped matchups")
    if type(suite.get("max_steps")) is not int or suite["max_steps"] <= 0:
        raise ValueError("invalid suite decision limit")
    if type(suite.get("seed")) is not int:
        raise ValueError("invalid suite seed")
    assets, opponents, cases = suite.get("assets"), suite.get("opponents"), suite.get("cases")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("baseline suite needs assets")
    for key, asset in assets.items():
        if not isinstance(asset, dict) or _digest(asset) != key:
            raise ValueError("baseline suite asset hash mismatch")
    if not isinstance(opponents, dict) or not opponents:
        raise ValueError("baseline suite needs a pinned opponent pool")
    for key, opponent in opponents.items():
        if not isinstance(key, str) or not isinstance(opponent, dict):
            raise ValueError("invalid suite opponent")
        if opponent.get("group") not in _GROUPS:
            raise ValueError("invalid opponent group")
        if opponent.get("kind") == "random":
            if opponent["group"] != "random":
                raise ValueError("random policy must belong to random group")
        elif opponent.get("kind") == "checkpoint":
            _checksum(opponent.get("sha256"), "opponent")
            if opponent["group"] == "random" or not isinstance(opponent.get("path"), str):
                raise ValueError("invalid frozen opponent")
        else:
            raise ValueError("unsupported opponent kind")
    if not isinstance(cases, list) or not cases or len(cases) % 2:
        raise ValueError("baseline suite needs complete pairs of cases")
    pairs = {}
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or type(case.get("episode")) is not int or case["episode"] != index:
            raise ValueError("suite episode IDs must be unique and consecutive")
        if type(case.get("matchup")) is not int or case["matchup"] < 1:
            raise ValueError("suite requires positive matchup cluster IDs")
        if type(case.get("candidate_player")) is not int or case["candidate_player"] not in (1, 2):
            raise ValueError("invalid candidate seat")
        if (not isinstance(case.get("assets"), list) or len(case["assets"]) != 2
                or any(key not in assets for key in case["assets"])):
            raise ValueError("invalid case assets")
        if type(case.get("seed")) is not int or type(case.get("opponent_seed")) is not int:
            raise ValueError("each case must pin battle and opponent seeds")
        entry = opponents.get(case.get("opponent_id"))
        if entry is None or entry["group"] != case.get("opponent_group"):
            raise ValueError("case disagrees with pinned opponent pool")
        pairs.setdefault(case["matchup"], []).append(case)
    for pair in pairs.values():
        if len(pair) != 2 or sorted(c["candidate_player"] for c in pair) != [1, 2]:
            raise ValueError("each matchup cluster must contain both seats exactly once")
        neutral = [{k: v for k, v in case.items() if k not in ("episode", "candidate_player")}
                   for case in pair]
        if neutral[0] != neutral[1]:
            raise ValueError("side-swapped cases disagree on teams, seeds or opponents")
    return pairs


def load_baseline_evaluation(directory):
    """Validate completed evaluate_ppo artifacts without reading live checkpoints."""
    directory = Path(directory).expanduser().resolve()
    suite, config, summary = [_read(directory / name) for name in ("suite.json", "config.json", "summary.json")]
    _suite(suite)
    fingerprint = _digest(suite)
    if config.get("suite_sha256") != fingerprint or summary.get("suite_sha256") != fingerprint:
        raise ValueError(f"evaluation suite/config/summary mismatch: {directory}")
    if summary.get("status") != "completed":
        raise ValueError(f"baseline requires a completed evaluation: {directory}")
    if config.get("backend") != suite["backend"] or config.get("deterministic") is not True:
        raise ValueError(f"evaluation backend or deterministic setting is invalid: {directory}")
    if not isinstance(config.get("reward"), dict):
        raise ValueError(f"evaluation reward configuration is missing: {directory}")
    _digest(config["reward"])
    models = config.get("models")
    candidate = models.get("candidate") if isinstance(models, dict) else None
    if not isinstance(candidate, dict):
        raise ValueError(f"candidate checkpoint identity is missing: {directory}")
    _checksum(candidate.get("sha256"), "candidate")
    if (not isinstance(candidate.get("path"), str) or type(candidate.get("training_timesteps")) is not int
            or candidate["training_timesteps"] < 0):
        raise ValueError(f"invalid candidate checkpoint metadata: {directory}")
    saved_opponents = config.get("opponents")
    if not isinstance(saved_opponents, dict) or set(saved_opponents) != set(suite["opponents"]):
        raise ValueError(f"evaluation opponent pool differs from suite: {directory}")
    for key, expected in suite["opponents"].items():
        actual = saved_opponents[key]
        if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError(f"evaluated opponent identity differs from suite: {directory}, {key}")
    rows = {}
    try:
        with (directory / "episodes.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line, parse_constant=_invalid_constant)
                if not isinstance(row, dict):
                    raise ValueError("episode row must be an object")
                episode = row.get("episode")
                if type(episode) is not int or episode in rows or not 1 <= episode <= len(suite["cases"]):
                    raise ValueError("duplicate or invalid episode ID")
                case = suite["cases"][episode-1]
                if any(row.get(key) != value for key, value in case.items()):
                    raise ValueError("episode row differs from its saved case")
                if row.get("candidate_outcome") not in _OUTCOMES:
                    raise ValueError("invalid candidate outcome")
                if (type(row.get("truncated")) is not bool or type(row.get("terminated")) is not bool
                        or row["truncated"] != (row["candidate_outcome"] == "truncated")
                        or row["terminated"] == row["truncated"]):
                    raise ValueError("inconsistent termination/truncation flags")
                if type(row.get("decisions")) is not int or row["decisions"] < 0:
                    raise ValueError("invalid decision count")
                for key in ("candidate_reward", "opponent_reward"):
                    _number(row.get(key), key)
                rows[episode] = row
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid baseline episode records {directory}: {error}") from error
    if len(rows) != len(suite["cases"]):
        raise ValueError(f"evaluation is incomplete or missing cases: {directory}")
    rows = [rows[i] for i in range(1, len(rows)+1)]
    counts = {name: sum(row["candidate_outcome"] == name for row in rows) for name in _OUTCOMES}
    expected_summary = dict(requested_battles=len(rows), attempted_battles=len(rows),
                            completed_battles=len(rows)-counts["truncated"],
                            candidate_wins=counts["won"], opponent_wins=counts["lost"],
                            draws=counts["draw"], truncated=counts["truncated"])
    if any(type(summary.get(key)) is not int or summary[key] != value
           for key, value in expected_summary.items()):
        raise ValueError(f"summary outcome counts disagree with episode records: {directory}")
    _number(summary.get("elapsed_seconds"), "elapsed_seconds", nonnegative=True)
    for key, row_key in (("candidate_reward_mean", "candidate_reward"),
                         ("opponent_reward_mean", "opponent_reward"), ("decisions_mean", "decisions")):
        if key in summary:
            value = _number(summary[key], key)
            expected = float(np.mean([row[row_key] for row in rows]))
            if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"summary {key} disagrees with episode records: {directory}")
    return dict(directory=str(directory), suite=suite, config=config, summary=summary, rows=rows,
                suite_sha256=fingerprint, candidate=candidate)


def _seed(seed, key):
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def _bootstrap(values, *, samples, confidence, seed, min_clusters):
    """Percentile CI over equal-weight matchup means; bounded resample batches."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    point = float(values.mean()) if n else None
    if n < min_clusters:
        return dict(estimate=point, low=None, high=None, confidence=confidence,
                    clusters=n, method="paired_matchup_bootstrap", status="insufficient_clusters")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    batch = max(1, min(256, 1_000_000 // n))
    for offset in range(0, samples, batch):
        size = min(batch, samples-offset)
        indexes = rng.integers(0, n, size=(size, n))
        draws[offset:offset+size] = values[indexes].mean(axis=1)
    low, high = np.quantile(draws, [(1-confidence)/2, (1+confidence)/2])
    return dict(estimate=point, low=float(low), high=float(high), confidence=confidence,
                clusters=n, method="paired_matchup_bootstrap",
                status="degenerate_observed_sample" if np.all(values == values[0]) else "estimated")


def _cluster_wins(rows):
    clusters = {}
    for row in rows:
        clusters.setdefault(row["matchup"], []).append(float(row["candidate_outcome"] == "won"))
    return {key: float(np.mean(values)) for key, values in sorted(clusters.items())}


def _stats(rows, *, options, seed):
    counts = {name: sum(row["candidate_outcome"] == name for row in rows) for name in _OUTCOMES}
    n = len(rows)
    cluster_means = _cluster_wins(rows)
    values = list(cluster_means.values())
    interval = _bootstrap(values, seed=seed, **options)
    return dict(attempts=n, completed=n-counts["truncated"], wins=counts["won"],
                losses=counts["lost"], draws=counts["draw"], truncated=counts["truncated"],
                matchup_clusters=len(values), win_rate=counts["won"]/n if n else None,
                score=(counts["won"]+.5*counts["draw"])/n if n else None,
                win_rate_ci=interval,
                candidate_reward_mean=float(np.mean([r["candidate_reward"] for r in rows])) if n else None,
                opponent_reward_mean=float(np.mean([r["opponent_reward"] for r in rows])) if n else None,
                decisions_mean=float(np.mean([r["decisions"] for r in rows])) if n else None)


def _compare(rows, starting, *, options, seed):
    left, right = _cluster_wins(rows), _cluster_wins(starting)
    if left.keys() != right.keys() or [r["episode"] for r in rows] != [r["episode"] for r in starting]:
        raise ValueError("paired comparison has mismatched evaluation cases")
    differences = [left[key]-right[key] for key in left]
    interval = _bootstrap(differences, seed=seed, **options)
    if interval["low"] is None:
        conclusion = "insufficient_data"
    elif interval["low"] > 0:
        conclusion = "improvement"
    elif interval["high"] < 0:
        conclusion = "regression"
    else:
        conclusion = "no_clear_change"
    return dict(delta_win_rate=interval["estimate"], delta_win_rate_ci=interval,
                conclusion=conclusion, matched_battles=len(rows), matchup_clusters=len(differences))


def build_baseline_report(evaluations, *, starting_label="starting", bootstrap_samples=4000,
                          seed=1, confidence=.95, min_clusters=20):
    """Summarize ``{candidate_label: {split: evaluation_directory}}``.

    Every candidate must replay the same suite within each split. Case alignment,
    opponent identities, reward settings, completion and checkpoint identities are
    verified before statistics are calculated. CIs are pointwise, not simultaneous
    tests, and do not correct selection bias in the best previously evaluated model.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be an integer >= 100")
    if type(min_clusters) is not int or min_clusters < 2:
        raise ValueError("min_clusters must be an integer >= 2")
    if type(seed) is not int or type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("seed must be an integer and confidence must be between zero and one")
    if not isinstance(evaluations, dict) or starting_label not in evaluations or len(evaluations) < 2:
        raise ValueError("baseline report needs starting and at least one comparison candidate")
    if not all(isinstance(k, str) and k and isinstance(v, dict) and v for k, v in evaluations.items()):
        raise ValueError("candidate labels must map to nonempty split dictionaries")
    splits = list(evaluations[starting_label])
    if not all(isinstance(split, str) and split for split in splits):
        raise ValueError("split names must be nonempty strings")
    if any(set(value) != set(splits) for value in evaluations.values()):
        raise ValueError("all candidates must have the same team splits")
    loaded, cache = {}, {}
    for label, directories in evaluations.items():
        loaded[label] = {}
        for split, directory in directories.items():
            key = str(Path(directory).expanduser().resolve())
            if key not in cache:
                cache[key] = load_baseline_evaluation(directory)
            loaded[label][split] = cache[key]
        identities = {run["candidate"]["sha256"] for run in loaded[label].values()}
        if len(identities) != 1:
            raise ValueError(f"candidate weights changed between team splits: {label}")
    anchor_config = loaded[starting_label][splits[0]]["config"]
    def pool_identity(run):
        return {key: {field: item.get(field) for field in ("kind", "group", "sha256")}
                for key, item in run["suite"]["opponents"].items()}
    anchor_pool = pool_identity(loaded[starting_label][splits[0]])
    for split in splits:
        reference = loaded[starting_label][split]
        if pool_identity(reference) != anchor_pool:
            raise ValueError("opponent identities differ between team splits")
        observed_by_hash = {}
        for label in loaded:
            candidate = loaded[label][split]
            if candidate["suite_sha256"] != reference["suite_sha256"]:
                raise ValueError(f"candidates did not evaluate the identical suite: {split}, {label}")
            for key in ("backend", "reward", "deterministic"):
                if candidate["config"][key] != anchor_config[key]:
                    raise ValueError(f"evaluation setting differs across candidates/splits: {key}")
            checksum = candidate["candidate"]["sha256"]
            signature = [{key: row[key] for key in ("episode", "candidate_outcome", "candidate_reward",
                                                   "opponent_reward", "decisions", "terminated", "truncated")}
                         for row in candidate["rows"]]
            if checksum in observed_by_hash and observed_by_hash[checksum] != signature:
                raise ValueError("identical checkpoint weights produced inconsistent deterministic results")
            observed_by_hash[checksum] = signature
    options = dict(samples=bootstrap_samples, confidence=confidence, min_clusters=min_clusters)
    report = dict(version=1, status="completed", starting_label=starting_label,
                  methodology=dict(win_rate_denominator="all attempted battles; draws and truncations are non-wins",
                    score="(wins + 0.5 * draws) / all attempted battles",
                    bootstrap="percentile bootstrap of matchup clusters, keeping both candidate seats together",
                    comparisons="paired candidate-minus-starting wins on identical saved cases",
                    confidence=confidence, bootstrap_samples=bootstrap_samples, seed=seed, min_clusters=min_clusters,
                    inference_scope="pointwise finite-suite sampling uncertainty; no multiple-comparison or model-selection correction",
                    degenerate_note="constant observed cluster outcomes produce zero-width bootstrap intervals; this is not proof of certainty",
                    holdout_note="validation teams may have been used for promotion; this is not an untouched final test",
                    promotion="diagnostic only; no automatic promotion or training changes"),
                  candidates={}, splits={}, comparisons={})
    for label, runs in loaded.items():
        identity = runs[splits[0]]["candidate"]
        same = [other for other in loaded if other != label
                and loaded[other][splits[0]]["candidate"]["sha256"] == identity["sha256"]]
        report["candidates"][label] = {**identity, "identical_weights_to": same,
                                         "evaluations": {split: run["directory"] for split, run in runs.items()}}
        if label != starting_label:
            report["comparisons"][label] = {}
    for split in splits:
        suite = loaded[starting_label][split]["suite"]
        split_report = dict(suite_sha256=loaded[starting_label][split]["suite_sha256"],
                            opponent_pool=suite["opponents"], candidates={})
        report["splits"][split] = split_report
        for label, runs in loaded.items():
            run = runs[split]
            rows = run["rows"]
            buckets = {"all": rows, **{group: [r for r in rows if r["opponent_group"] == group]
                                       for group in _GROUPS}}
            individual = {key: [r for r in rows if r["opponent_id"] == key] for key in suite["opponents"]}
            values = {group: _stats(subset, options=options, seed=_seed(seed, f"{split}:group:{group}"))
                      for group, subset in buckets.items()}
            seats = {str(side): _stats([r for r in rows if r["candidate_player"] == side],
                         options=options, seed=_seed(seed, f"{split}:seat:{side}")) for side in (1, 2)}
            elapsed = run["summary"]["elapsed_seconds"]
            split_report["candidates"][label] = dict(groups=values, by_candidate_side=seats,
                by_opponent={key: _stats(subset, options=options, seed=_seed(seed, f"{split}:opponent:{key}"))
                             for key, subset in individual.items()},
                elapsed_seconds=elapsed, battles_per_hour=values["all"]["completed"]*3600/elapsed if elapsed else None,
                attempts_per_hour=len(rows)*3600/elapsed if elapsed else None)
            if label != starting_label:
                baseline_rows = loaded[starting_label][split]["rows"]
                comparisons = {}
                for group, subset in buckets.items():
                    base = baseline_rows if group == "all" else [r for r in baseline_rows if r["opponent_group"] == group]
                    comparisons[group] = _compare(subset, base, options=options,
                                                    seed=_seed(seed, f"{split}:group:{group}"))
                report["comparisons"][label][split] = dict(groups=comparisons,
                    by_candidate_side={str(side): _compare([r for r in rows if r["candidate_player"] == side],
                        [r for r in baseline_rows if r["candidate_player"] == side], options=options,
                        seed=_seed(seed, f"{split}:seat:{side}")) for side in (1, 2)},
                    by_opponent={key: _compare(subset, [r for r in baseline_rows if r["opponent_id"] == key],
                        options=options, seed=_seed(seed, f"{split}:opponent:{key}"))
                                 for key, subset in individual.items()})
    return report


def _percent(value):
    return "—" if value is None else f"{100*value:.1f}%"


def _interval(interval, *, delta=False):
    if interval["low"] is None:
        return "insufficient matchups"
    return (f"{100*interval['low']:+.1f} to {100*interval['high']:+.1f} pp" if delta else
            f"{100*interval['low']:.1f}–{100*interval['high']:.1f}%")


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_baseline_markdown(report):
    """Render concise outcome/uncertainty tables; detailed opponent/seat data is JSON."""
    starting = report["starting_label"]
    confidence = report["methodology"]["confidence"]*100
    lines = ["# Reliable training baseline", "", "Completed evaluations; no promotion rules were changed.", "",
             f"Win rates count draws and truncations as non-wins. {confidence:g}% intervals resample whole "
             "matchups with both candidate seats together. Deltas compare identical cases against the starting checkpoint.", "",
             "| Candidate | Training timesteps | Identical weights |", "|---|---:|---|"]
    for label, candidate in report["candidates"].items():
        lines.append(f"| {_cell(label)} | {candidate['training_timesteps']:,} | "
                     f"{_cell(', '.join(candidate['identical_weights_to']) or '—')} |")
    if report.get("selection_warnings"):
        lines += ["", "Selection notes:", ""]
        lines += [f"- {_cell(note)}" for note in report["selection_warnings"]]
    lines += ["", "## Matched change from starting", "",
              f"| Split | Candidate | Win rate | {confidence:g}% interval | Change | Paired interval | Assessment |",
              "|---|---|---:|---|---:|---|---|"]
    for split, data in report["splits"].items():
        for label, stats in data["candidates"].items():
            overall = stats["groups"]["all"]
            if label == starting:
                delta, bounds, verdict = "—", "—", "Reference"
            else:
                comparison = report["comparisons"][label][split]["groups"]["all"]
                delta = f"{comparison['delta_win_rate']*100:+.1f} pp"
                bounds = _interval(comparison["delta_win_rate_ci"], delta=True)
                verdict = comparison["conclusion"].replace("_", " ")
            lines.append(f"| {_cell(split)} | {_cell(label)} | {_percent(overall['win_rate'])} | "
                         f"{_interval(overall['win_rate_ci'])} | {delta} | {bounds} | {verdict} |")
    lines += ["", "## Opponent groups", "",
              "| Split | Candidate | Opponent | W / L / D / truncated | Win rate | Interval | Change vs starting |",
              "|---|---|---|---:|---:|---|---|"]
    for split, data in report["splits"].items():
        for label, stats in data["candidates"].items():
            for group in _GROUPS:
                value = stats["groups"][group]
                change = "—"
                if label != starting:
                    comparison = report["comparisons"][label][split]["groups"][group]
                    change = (f"{comparison['delta_win_rate']*100:+.1f} pp "
                              f"({_interval(comparison['delta_win_rate_ci'], delta=True)})"
                              if comparison["delta_win_rate"] is not None else "—")
                lines.append(f"| {_cell(split)} | {_cell(label)} | {group.title()} | "
                    f"{value['wins']} / {value['losses']} / {value['draws']} / {value['truncated']} | "
                    f"{_percent(value['win_rate'])} | {_interval(value['win_rate_ci'])} | {change} |")
    lines += ["", "## Battle length, rewards and seats", "",
              "| Split | Candidate | Mean decisions | Mean reward | Battles/hour | Seat 1 win | Seat 2 win |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for split, data in report["splits"].items():
        for label, stats in data["candidates"].items():
            overall = stats["groups"]["all"]
            rate = f"{stats['battles_per_hour']:,.0f}" if stats["battles_per_hour"] is not None else "—"
            lines.append(f"| {_cell(split)} | {_cell(label)} | {overall['decisions_mean']:.1f} | "
                         f"{overall['candidate_reward_mean']:.2f} | {rate} | "
                         f"{_percent(stats['by_candidate_side']['1']['win_rate'])} | "
                         f"{_percent(stats['by_candidate_side']['2']['win_rate'])} |")
    lines += ["", "Intervals are pointwise and describe this suite's sampled matchups, not uncertainty across training runs. "
              "The best checkpoint was selected using earlier evaluations; this report does not correct that selection bias. "
              "Validation teams may already have influenced campaign promotion.", "",
              "Constant observed outcomes can produce zero-width bootstrap intervals; these do not prove certainty. "
              "Groups with too few matchup clusters are marked insufficient. "
              "Throughput is measured for each full evaluation; group-specific timing is not inferred.", "",
              "The JSON report includes per-opponent and per-seat counts, intervals, hashes and evaluation paths.", ""]
    return "\n".join(lines)


def write_baseline_report(report, output_dir):
    """Write baseline-report.json and baseline-report.md using atomic replacement."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / "baseline-report.json", directory / "baseline-report.md")
    texts = (json.dumps(report, indent=2, allow_nan=False)+"\n", format_baseline_markdown(report))
    for path, text in zip(paths, texts):
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return paths
