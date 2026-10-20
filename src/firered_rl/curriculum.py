"""Cumulative team pools, repeatable validation, and explicit promotion rules."""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

from .evaluation import digest, validate_suite
from .teams import TeamSpec
from .switch_opponent import switch_settings


DEFAULT_RULES = dict(current_win_rate=.60, expanded_win_rate=.55,
                     incoming_win_rate=.50, max_regression=.03, max_truncation_rate=.01)
OPPONENT_MODES = ("fixed", "previous-stage")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def team_groups(paths):
    """Accept legacy one-file stages as well as explicit groups of team files."""
    return [[Path(p)] if isinstance(p, (str, Path)) else [Path(q) for q in p] for p in paths]


def party_signature(path):
    """Ignore labels and ordering when detecting train/validation leakage."""
    members = []
    for mon in TeamSpec.load(path).party:
        row = asdict(mon)
        row["moves"] = sorted(m.lower().replace(" ", "") for m in mon.moves)
        row["species"] = mon.species.lower().replace(" ", "")
        row["held_item"] = mon.held_item.lower().replace(" ", "")
        members.append(json.dumps(row, sort_keys=True))
    return digest(sorted(members))


def load_curriculum(path, *, grouped=False):
    path = Path(path).expanduser().resolve()
    value = read_json(path)
    stages = value.get("stages", [])
    if value.get("version") != 1 or not stages:
        raise ValueError("curriculum needs version 1 and an ordered stages list")
    if value.get("opponent_mode", "fixed") not in OPPONENT_MODES:
        raise ValueError("curriculum opponent_mode must be fixed or previous-stage")
    switches = value.get("opponent_switch", {})
    if not isinstance(switches, dict) or set(switches) - {"chance", "cooldown"}:
        raise ValueError("curriculum opponent_switch accepts chance and cooldown")
    switch_settings(**switches)
    if [s.get("stage") for s in stages] != list(range(1, len(stages) + 1)):
        raise ValueError("curriculum stages must be numbered consecutively from 1")
    def resolve(entry, file_key, dir_key):
        if sum(key in entry for key in (file_key, dir_key)) != 1:
            raise ValueError(f"specify exactly one of {file_key} or {dir_key}")
        key = file_key if file_key in entry else dir_key
        source = entry[key]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{key} must be a nonempty path relative to the curriculum")
        source = (path.parent / source).resolve()
        if key == file_key:
            files = [source]
        else:
            if not source.is_dir():
                raise ValueError(f"team directory does not exist: {source}")
            files = sorted(p.resolve() for p in source.rglob("*.json") if p.is_file())
            if not files:
                raise ValueError(f"no team JSON files in {source}")
            if any(not p.is_relative_to(source) for p in files):
                raise ValueError(f"team directory contains a symlink outside its root: {source}")
        for team in files:
            try:
                spec = TeamSpec.load(team)
                if any(len(mon.moves) != 4 for mon in spec.party):
                    raise ValueError("curriculum Pokemon need four moves")
            except (ValueError, OSError, TypeError, AttributeError) as exc:
                raise ValueError(f"invalid curriculum team {team}: {exc}") from exc
        return files

    groups = [resolve(s, "team", "team_dir") for s in stages]
    groups.append(resolve(value, "next_team", "next_team_dir"))
    paths = [p for group in groups for p in group]
    if len(set(paths)) != len(paths):
        raise ValueError("each stage and the final validation group need distinct files")
    has_validation = ["validation_dir" in s or "validation_team" in s for s in stages]
    if any(has_validation) and not all(has_validation):
        raise ValueError("provide validation_dir or validation_team for every stage, or none")
    validation = None
    if all(has_validation):
        validation = [resolve(s, "validation_team", "validation_dir") for s in stages]
        validation.append(groups[-1])
        training = {p for group in groups[:-1] for p in group}
        held_out = [p for group in validation for p in group]
        if len(set(held_out)) != len(held_out) or training.intersection(held_out):
            raise ValueError("training and validation groups must use distinct files")
        signatures = {party_signature(p) for p in training}
        for team in held_out:
            if party_signature(team) in signatures:
                raise ValueError(f"validation duplicates a training party (ignoring names/order): {team}")
    if grouped:
        return value, groups, validation
    return value, paths


def validate_rules(value):
    if set(value) != set(DEFAULT_RULES):
        raise ValueError("promotion rules must specify: " + ", ".join(DEFAULT_RULES))
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in value.values()):
        raise ValueError("promotion thresholds must be finite fractions from 0 to 1")
    return value


def stage_suites(paths, *, stage, episodes, seed, max_steps):
    """Expanded suite: half retained matchups, half involving the next group.

    Retained cases are reused from the current suite. Both candidate seats use
    identical teams and RNG. The incoming team never enters the training pool.
    """
    if episodes < 4 or episodes % 4:
        raise ValueError("--eval-episodes must be a positive multiple of four")
    groups = team_groups(paths)
    if not 1 <= stage < len(groups) or any(not g for g in groups[:stage+1]):
        raise ValueError("stage needs its current teams and one next-team probe")
    current_count = sum(len(g) for g in groups[:stage])
    paths = [p for group in groups[:stage+1] for p in group]
    assets = {}
    keys = []
    for path in paths:
        asset = dict(team=read_json(path))
        key = digest(asset)
        keys.append(key)
        assets[key] = asset
    rng = random.Random(seed)

    def paired_cases(matchups, cohort):
        cases = []
        # Shuffle a balanced cycle of team pairs; extra pairs repeat the cycle.
        combinations = [(a, b) for a in range(current_count) for b in range(current_count)]
        if cohort == "incoming":
            incoming = range(current_count, len(paths))
            combinations = ([(i, j) for i in range(current_count) for j in incoming]
                            + [(j, i) for j in incoming for i in range(current_count)]
                            + [(a, b) for a in incoming for b in incoming])
        for i in range(matchups):
            if i % len(combinations) == 0:
                rng.shuffle(combinations)
            a, b = combinations[i % len(combinations)]
            battle_seed = rng.randrange(2**31 - 1)
            for side in (1, 2):
                cases.append(dict(assets=[keys[a], keys[b]], seed=battle_seed,
                                  candidate_player=side, cohort=cohort))
        return cases

    current_cases = paired_cases(episodes // 2, "retained")
    expanded_cases = copy.deepcopy(current_cases[:episodes // 2])
    expanded_cases += paired_cases(episodes // 4, "incoming")

    def suite(cases):
        used = {key for case in cases for key in case["assets"]}
        for i, case in enumerate(cases):
            case.update(episode=i+1, matchup=i//2+1)
        return validate_suite(dict(version=1, backend="showdown", seed=seed, swap_sides=True,
                                   max_steps=max_steps, assets={k: v for k, v in assets.items() if k in used},
                                   cases=cases))

    return suite(current_cases), suite(expanded_cases)


def evaluation_rates(directory, suite):
    """Require the entire expected schedule, including the new-team cases."""
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    if summary.get("status") != "completed" or summary.get("suite_sha256") != digest(suite):
        raise ValueError(f"evaluation is incomplete or uses another suite: {directory}")
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: r["episode"])
    if len(rows) != len(suite["cases"]):
        raise ValueError(f"evaluation is missing cases: {directory}")
    for row, case in zip(rows, suite["cases"]):
        if any(row.get(key) != value for key, value in case.items()):
            raise ValueError(f"evaluation case differs from the saved suite: {directory}")
        if row.get("candidate_outcome") not in ("won", "lost", "draw", "truncated"):
            raise ValueError(f"invalid battle outcome: {directory}")
        if bool(row.get("truncated")) != (row["candidate_outcome"] == "truncated"):
            raise ValueError(f"inconsistent truncation: {directory}")

    def rates(selected):
        ended = [r for r in selected if not r["truncated"]]
        wins = sum(r["candidate_outcome"] == "won" for r in ended)
        return dict(attempts=len(selected), completed=len(ended), wins=wins,
                    win_rate=wins/len(ended) if ended else None,
                    truncation_rate=(len(selected)-len(ended))/len(selected))

    result = dict(all=rates(rows))
    for cohort in ("retained", "incoming"):
        selected = [r for r in rows if r.get("cohort") == cohort]
        if selected:
            result[cohort] = rates(selected)
    if suite.get("opponents"):
        from .evaluation import opponent_rates
        for group in ("random", "dumb", "smart", "basic"):
            selected = [r for r in rows if r.get("opponent_group") == group or
                        (group == "basic" and r.get("opponent_group") in ("random", "dumb"))]
            result[group] = opponent_rates(selected)
    return result


def promotion_report(before_current, before_expanded, current, expanded, rules):
    checks = []

    def check(label, value, limit, direction="at_least"):
        passed = value is not None and (value + 1e-12 >= limit if direction == "at_least"
                                       else value <= limit + 1e-12)
        checks.append(dict(check=label, value=value, threshold=limit, direction=direction, passed=passed))

    for label, values, target in (("current", current["all"], rules["current_win_rate"]),
                                  ("expanded", expanded["all"], rules["expanded_win_rate"]),
                                  ("incoming", expanded["incoming"], rules["incoming_win_rate"])):
        check(label + " win rate", values["win_rate"], target)
    for label, before, after in (("current", before_current["all"], current["all"]),
                                 ("expanded", before_expanded["all"], expanded["all"])):
        old, new = before["win_rate"], after["win_rate"]
        check(label + " regression", old-new if old is not None and new is not None else None,
              rules["max_regression"], "at_most")
    for prefix, values in (("baseline current", before_current), ("baseline expanded", before_expanded),
                           ("current", current), ("expanded", expanded)):
        for cohort, rates in values.items():
            check(f"{prefix}/{cohort} truncation", rates["truncation_rate"],
                  rules["max_truncation_rate"], "at_most")
    return dict(status="promote" if all(c["passed"] for c in checks) else "hold", checks=checks,
                before_current=before_current, before_expanded=before_expanded,
                current=current, expanded=expanded)


def report_markdown(report):
    teams = ""
    if "training_teams" in report:
        kind = "held-out validation" if report.get("held_out_validation") else "evaluation"
        teams = f"Teams: {report['training_teams']} training; {report['evaluation_teams']} {kind}."
    if report.get("mode") == "mixed-evaluation":
        lines = [f"# Stage {report['stage']} — {report['status'].upper()}", "",
                 teams, "",
                 f"Rules revision: {report.get('settings_revision', 0)}; promotion requires **{report['promotion_require']}** groups.", "",
                 "| Opponent group | Wins / attempts | Win rate |", "|---|---:|---:|"]
        for group in ("random", "dumb", "basic", "smart"):
            row = report["current"][group]
            rate = f"{row['win_rate']:.1%}" if row['win_rate'] is not None else "n/a"
            lines.append(f"| {group} | {row['wins']} / {row['attempts']} | {rate} |")
        lines.extend(["", "Basic combines random and Dumb. Draws and truncations count as non-wins.", ""])
        for check in report["checks"]:
            lines.append(f"- {check['check']}: target {check['threshold']:.1%} — {'PASS' if check['passed'] else 'HOLD'}")
        lines.extend(["", f"Candidate: `{report['candidate']}`", "",
                      f"Evaluation plan: `{report['evaluation_plan']}`", ""])
        return "\n".join(lines)
    if report.get("mode") == "battle-budget":
        rates = report["current"]["all"]
        return (f"# Stage {report['stage']} — {report['status'].upper()}\n\n"
                f"{teams}\n\n"
                f"Wins: {rates['wins']} / {rates['attempts']} ({report['win_rate']:.1%}).\n\n"
                f"Candidate: `{report['candidate']}`\n\nFrozen opponent: `{report['opponent']['path']}`\n\n"
                "Draws and truncations count as non-wins. See campaign evaluations.md for all attempts.\n")
    lines = [f"# Stage {report['stage']} — {report['status'].upper()}", "", teams, "",
             f"Candidate: `{report['candidate']}`", ""]
    if opponent := report.get("opponent"):
        source = (f"promoted stage {opponent['source_stage']}" if opponent["source_stage"] is not None
                  else "initial reference")
        lines.extend([f"Frozen opponent: `{opponent['path']}` ({source}; mode `{opponent['mode']}`).", "",
                      f"Opponent SHA-256: `{opponent['sha256']}`", ""])
    lines.extend(["| Evaluation | Before | After |", "|---|---:|---:|"])
    for label, before, after in (("Current pool", report["before_current"]["all"], report["current"]["all"]),
                                ("Pool + next team", report["before_expanded"]["all"], report["expanded"]["all"]),
                                ("Next-team matchups", report["before_expanded"]["incoming"], report["expanded"]["incoming"])):
        fmt = lambda row: "n/a" if row["win_rate"] is None else f"{row['win_rate']:.1%}"
        lines.append(f"| {label} | {fmt(before)} | {fmt(after)} |")
    lines.extend(["", "Win rates exclude truncations and include draws in the denominator.", "",
                  "## Promotion checks", "", "| Check | Value | Requirement | Result |", "|---|---:|---:|---|"])
    for c in report["checks"]:
        value = "n/a" if c["value"] is None else f"{c['value']:.1%}"
        sign = ">=" if c["direction"] == "at_least" else "<="
        lines.append(f"| {c['check']} | {value} | {sign} {c['threshold']:.1%} | {'PASS' if c['passed'] else 'HOLD'} |")
    lines.extend(["", "These are practical promotion thresholds, not a statistical proof of generalization.", ""])
    return "\n".join(lines)
