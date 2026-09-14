"""Saved round-robin evaluation plans and model comparison reports."""
import csv
from datetime import datetime, timezone
import hashlib
import itertools
from pathlib import Path
import re

from .curriculum import evaluation_rates, read_json, write_json
from .evaluation import create_suite, digest, validate_suite
from .training_teams import TrainingTeams
from .training_profiles import training_profile_directory


SOURCES = ("teams", "team_dir", "training_profile", "team_profile")
DEFAULTS = dict(episodes=1000, max_steps=200, seed=10001, include_random=True)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path, overrides=None):
    path = Path(path).expanduser().resolve()
    raw = read_json(path)
    allowed = {"version", "name", "scenarios", *DEFAULTS}
    if not isinstance(raw, dict) or set(raw) - allowed or raw.get("version") != 1:
        raise ValueError("evaluation config needs version 1; allowed keys: " + ", ".join(sorted(allowed)))
    value = {**DEFAULTS, **raw, **{k:v for k,v in (overrides or {}).items() if v is not None}}
    if "name" in value and (not isinstance(value["name"], str) or not value["name"].strip()):
        raise ValueError("config name must be a nonempty string")
    for key in ("episodes", "max_steps"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"config {key} must be a positive integer")
    if value["episodes"] % 2:
        raise ValueError("episodes must be even so every matchup gets both player sides")
    if type(value["seed"]) is not int or type(value["include_random"]) is not bool:
        raise ValueError("config seed must be an integer and include_random must be boolean")
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config needs a nonempty scenarios list")
    ids = set()
    for scenario in scenarios:
        allowed_scenario = {"id", "name", "mirror_teams", "team_size", *SOURCES}
        if not isinstance(scenario, dict) or set(scenario) - allowed_scenario:
            raise ValueError("unknown evaluation scenario keys")
        key = scenario.get("id")
        if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", key) or key == "overall" or key in ids:
            raise ValueError("scenario IDs must be unique lowercase names; 'overall' is reserved")
        ids.add(key)
        if sum(scenario.get(source) is not None for source in SOURCES) != 1:
            raise ValueError(f"{key}: choose exactly one team source: {', '.join(SOURCES)}")
        if type(scenario.get("mirror_teams", False)) is not bool:
            raise ValueError(f"{key}: mirror_teams must be boolean")
        if scenario.get("team_size") is not None and not scenario.get("team_profile"):
            raise ValueError(f"{key}: team_size requires team_profile")
        if "team_size" in scenario and (type(scenario["team_size"]) is not int or not 1 <= scenario["team_size"] <= 6):
            raise ValueError(f"{key}: team_size must be an integer from 1 to 6")
        if "teams" in scenario and (not isinstance(scenario["teams"], list) or len(scenario["teams"]) != 2):
            raise ValueError(f"{key}: teams must contain two JSON paths")
        if "teams" in scenario and any(not isinstance(p, str) or not p.strip() for p in scenario["teams"]):
            raise ValueError(f"{key}: teams must contain nonempty path strings")
        for field in ("name", "team_dir", "training_profile", "team_profile"):
            if field in scenario and (not isinstance(scenario[field], str) or not scenario[field].strip()):
                raise ValueError(f"{key}: {field} must be a nonempty string")
    return value, path.parent


def scenario_source(scenario, base, project, seed):
    def local(value):
        path = Path(value).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()
    directory = None
    if scenario.get("team_dir"):
        directory = local(scenario["team_dir"])
    elif scenario.get("training_profile"):
        directory = training_profile_directory(project, scenario["training_profile"])
    return TrainingTeams(teams=[local(p) for p in scenario["teams"]] if scenario.get("teams") else None,
        directory=directory, profile=scenario.get("team_profile"), team_size=scenario.get("team_size"),
        profile_root=project / "config/teams/profiles", seed=seed,
        mirror_teams=scenario.get("mirror_teams", False))


def discover_models(directory, pattern="*.zip"):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"model directory does not exist: {directory}")
    if not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("model-glob must stay inside the model directory")
    files = sorted(p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() == ".zip")
    if not files:
        raise ValueError(f"no model ZIPs match {pattern!r} in {directory}")
    models, hashes = [], {}
    for path in files:
        checksum = file_hash(path)
        alias = dict(name=path.relative_to(directory).with_suffix("").as_posix(), source=str(path.resolve()))
        if checksum in hashes:
            hashes[checksum]["aliases"].append(alias)
            continue
        model = dict(id=f"m{len(models)+1:03d}", kind="checkpoint", label=alias["name"],
            source=alias["source"], sha256=checksum, aliases=[], path=f"inputs/models/{checksum}.zip")
        models.append(model)
        hashes[checksum] = model
    return models


def make_plan(project, config_path, models_dir, *, model_glob="*.zip", overrides=None):
    project = Path(project)
    config, base = read_config(config_path, overrides)
    models = discover_models(models_dir, model_glob)
    if config["include_random"]:
        models.append(dict(id="random", label="Random", kind="random"))
    labels = set()
    for model in models:
        while model["label"] in labels:
            model["label"] += f" [{model['id']}]"
        labels.add(model["label"])
    if len(models) < 2:
        raise ValueError("round robin needs two distinct checkpoints, or one checkpoint plus Random")
    suites, scenarios = {}, []
    for index, scenario in enumerate(config["scenarios"]):
        seed = config["seed"] + index*1000003
        source = scenario_source(scenario, base, project, seed)
        suite = validate_suite(create_suite(source, backend="showdown", episodes=config["episodes"],
            seed=seed, swap_sides=True, max_steps=config["max_steps"]))
        suites[scenario["id"]] = suite
        scenarios.append(dict(id=scenario["id"], label=scenario.get("name", scenario["id"]),
            path=f"suites/{scenario['id']}.json", sha256=digest(suite)))
    matches = [dict(id=f"{scenario['id']}/{a['id']}-vs-{b['id']}", scenario=scenario["id"],
                    a=a["id"], b=b["id"])
               for scenario in scenarios for a,b in itertools.combinations(models, 2)]
    plan = dict(version=1, name=config.get("name", "Model round robin"), config=config,
        source_config=str(Path(config_path).resolve()), models_dir=str(Path(models_dir).resolve()),
        model_glob=model_glob, models=models, scenarios=scenarios, matches=matches,
        created_at=datetime.now(timezone.utc).isoformat())
    plan["plan_sha256"] = digest(plan)
    return plan, suites


def save_plan(run, plan, suites, *, validate_model):
    """Copy each model once, and embed only teams selected for the saved suites."""
    run = Path(run)
    # Validate all inputs before creating a tournament that cannot run.
    for model in plan["models"]:
        if model["kind"] == "checkpoint":
            validate_model(Path(model["source"]))
    for model in plan["models"]:
        if model["kind"] == "checkpoint":
            data = Path(model["source"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != model["sha256"]:
                raise ValueError(f"model changed during snapshot: {model['source']}")
            target = run / model["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for scenario in plan["scenarios"]:
        target = run / scenario["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, suites[scenario["id"]])
    write_json(run / "tournament.json", plan)


def verify_plan(run, plan):
    value = {key:v for key,v in plan.items() if key != "plan_sha256"}
    if plan.get("version") != 1 or digest(value) != plan.get("plan_sha256"):
        raise ValueError("saved tournament plan changed; create a new run for new inputs")
    for model in plan["models"]:
        if model["kind"] == "checkpoint" and file_hash(run / model["path"]) != model["sha256"]:
            raise ValueError(f"saved model changed: {model['label']}")
    for scenario in plan["scenarios"]:
        suite = validate_suite(read_json(run / scenario["path"]))
        if digest(suite) != scenario["sha256"]:
            raise ValueError(f"saved team suite changed: {scenario['id']}")


def read_match(run, plan, match, directory):
    models = {m["id"]:m for m in plan["models"]}
    scenario = next(s for s in plan["scenarios"] if s["id"] == match["scenario"])
    suite = read_json(run / scenario["path"])
    config = read_json(directory / "config.json")
    for role, key in (("candidate", "a"), ("opponent", "b")):
        expected = models[match[key]]
        actual = config.get("models", {}).get(role, {})
        valid = (actual.get("kind") == "random" if expected["kind"] == "random"
                 else actual.get("sha256") == expected["sha256"])
        if not valid:
            raise ValueError(f"{match['id']}: evaluation {role} does not match the saved model")
    if config.get("suite_sha256") != scenario["sha256"] or config.get("deterministic") is not True:
        raise ValueError(f"{match['id']}: evaluation settings disagree with the tournament")
    evaluation_rates(directory, suite)  # Requires all cases and checks every saved matchup/seed/side.
    import json
    rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    return dict(**match, evaluation=str(directory.relative_to(run)), games=len(rows),
        wins_a=sum(r["candidate_outcome"] == "won" for r in rows),
        wins_b=sum(r["candidate_outcome"] == "lost" for r in rows),
        draws=sum(r["candidate_outcome"] == "draw" for r in rows),
        truncated=sum(r["truncated"] for r in rows))


def completed_matches(run, plan):
    results = []
    for match in plan["matches"]:
        for path in sorted((run / "matches" / match["id"]).glob("attempt-*/summary.json")):
            try:
                status = read_json(path).get("status")
            except ValueError:
                continue  # A process may have stopped during a partial summary write.
            if status == "completed":
                results.append(read_match(run, plan, match, path.parent))
                break
    return results


def rates(matches, model):
    selected = [r for r in matches if model in (r["a"], r["b"])]
    wins = sum(r["wins_a"] if r["a"] == model else r["wins_b"] for r in selected)
    losses = sum(r["wins_b"] if r["a"] == model else r["wins_a"] for r in selected)
    draws = sum(r["draws"] for r in selected)
    truncated = sum(r["truncated"] for r in selected)
    games = sum(r["games"] for r in selected)
    return dict(fixtures=len(selected), games=games, wins=wins, losses=losses, draws=draws,
        truncated=truncated, win_rate=wins/games if games else None,
        score=(wins+.5*draws)/games if games else None)


def leaderboard(matches, models):
    rows = [dict(model=m["id"], label=m["label"], **rates(matches, m["id"])) for m in models]
    rows.sort(key=lambda r: (-(r["score"] if r["score"] is not None else -1), r["label"]))
    previous, rank = None, None
    for i, row in enumerate(rows, 1):
        # Equal scores share a rank; alphabetical ordering is only for display.
        if row["score"] is not None and row["score"] != previous:
            rank = i
        row["rank"] = rank if row["score"] is not None else None
        previous = row["score"]
    return rows


def atomic_text(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_csv(path, fields, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_reports(run, plan, results, *, status="running", active=None):
    complete = len(results) == len(plan["matches"])
    boards = {"overall": leaderboard(results, plan["models"])}
    boards.update({s["id"]:leaderboard([r for r in results if r["scenario"] == s["id"]], plan["models"])
                  for s in plan["scenarios"]})
    report = dict(status="completed" if complete else status, provisional=not complete,
        plan_sha256=plan["plan_sha256"], planned_matches=len(plan["matches"]),
        completed_matches=len(results), planned_battles=len(plan["matches"])*plan["config"]["episodes"],
        recorded_battles=sum(r["games"] for r in results), active_match=active,
        updated_at=datetime.now(timezone.utc).isoformat(), leaderboards=boards)
    write_json(run / "summary.json", report)
    percent = lambda v: "—" if v is None else f"{v:.1%}"
    escape = lambda v: str(v).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# Model evaluation leaderboard", "",
        f"**{'Complete' if complete else 'Provisional'}:** {len(results)}/{len(plan['matches'])} model-pair tests finished.", "",
        "Score = (wins + 0.5 × draws) / all battles. Truncations score zero for both models.",
        "Final rankings weight each opponent and team configuration equally. Partial rankings have incomplete coverage."]
    if active:
        lines += ["", f"Active: {escape(active)}"]
    csv_rows, head_rows = [], []
    for key, rows in boards.items():
        title = "Overall" if key == "overall" else next(s["label"] for s in plan["scenarios"] if s["id"] == key)
        lines += ["", f"## {escape(title)}", "",
                  "| Rank | Model | Games | Wins | Losses | Draws | Truncated | Win rate | Score |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['rank'] or '—'} | {escape(row['label'])} | {row['games']} | {row['wins']} | "
                f"{row['losses']} | {row['draws']} | {row['truncated']} | {percent(row['win_rate'])} | {percent(row['score'])} |")
            csv_rows.append(dict(scenario=key, **row))
        scope = results if key == "overall" else [r for r in results if r["scenario"] == key]
        for a,b in itertools.permutations(plan["models"], 2):
            selected = [r for r in scope if {r["a"], r["b"]} == {a["id"], b["id"]}]
            head_rows.append(dict(scenario=key, model=a["label"], opponent=b["label"], **rates(selected, a["id"])))
    atomic_text(run / "leaderboard.md", "\n".join(lines)+"\n")
    write_csv(run / "leaderboard.csv", list(csv_rows[0]), csv_rows)
    write_csv(run / "head-to-head.csv", list(head_rows[0]), head_rows)
    matrices = ["# Head-to-head scores", "", "Row model's score against column model. — means unplayed; self-pairs are excluded."]
    for key in boards:
        labels = [m["label"] for m in plan["models"]]
        scope = {(r["model"], r["opponent"]):r for r in head_rows if r["scenario"] == key}
        matrices += ["", f"## {escape(key)}", "", "| Model | " + " | ".join(map(escape, labels)) + " |",
                     "|---|" + "---:|"*len(labels)]
        for label in labels:
            cells = ["×" if label == other else percent(scope[(label,other)]["score"]) for other in labels]
            matrices.append("| " + escape(label) + " | " + " | ".join(cells) + " |")
    atomic_text(run / "matchups.md", "\n".join(matrices)+"\n")
    return report
