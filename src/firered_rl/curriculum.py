"""Cumulative team pools, repeatable validation, and explicit promotion rules."""
import copy
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


def load_curriculum(path):
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
    paths = [(path.parent / s["team"]).resolve() for s in stages]
    paths.append((path.parent / value["next_team"]).resolve())
    if len(set(paths)) != len(paths):
        raise ValueError("each stage and the final validation team need distinct files")
    for team in paths:
        if any(len(mon.moves) != 4 for mon in TeamSpec.load(team).party):
            raise ValueError(f"curriculum Pokemon need four moves: {team}")
    return value, paths


def validate_rules(value):
    if set(value) != set(DEFAULT_RULES):
        raise ValueError("promotion rules must specify: " + ", ".join(DEFAULT_RULES))
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in value.values()):
        raise ValueError("promotion thresholds must be finite fractions from 0 to 1")
    return value


def stage_suites(paths, *, stage, episodes, seed, max_steps):
    """Expanded suite: half retained matchups, half explicitly involving the new team.

    Retained cases are reused from the current suite. Both candidate seats use
    identical teams and RNG. The incoming team never enters the training pool.
    """
    if episodes < 4 or episodes % 4:
        raise ValueError("--eval-episodes must be a positive multiple of four")
    if not 1 <= stage < len(paths):
        raise ValueError("stage needs its current teams and one next-team probe")
    assets = {}
    keys = []
    for path in paths[:stage+1]:
        asset = dict(team=read_json(path))
        key = digest(asset)
        keys.append(key)
        assets[key] = asset
    rng = random.Random(seed)

    def paired_cases(matchups, cohort):
        cases = []
        # Shuffle a balanced cycle of team pairs; extra pairs repeat the cycle.
        combinations = [(a, b) for a in range(stage) for b in range(stage)]
        if cohort == "incoming":
            combinations = [(i, stage) for i in range(stage)] + [(stage, i) for i in range(stage)]
            combinations.append((stage, stage))
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
    if report.get("mode") == "mixed-evaluation":
        lines = [f"# Stage {report['stage']} — {report['status'].upper()}", "",
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
                f"Wins: {rates['wins']} / {rates['attempts']} ({report['win_rate']:.1%}).\n\n"
                f"Candidate: `{report['candidate']}`\n\nFrozen opponent: `{report['opponent']['path']}`\n\n"
                "Draws and truncations count as non-wins. See campaign evaluations.md for all attempts.\n")
    lines = [f"# Stage {report['stage']} — {report['status'].upper()}", "",
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
