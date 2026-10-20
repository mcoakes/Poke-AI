"""Read-only campaign selection and independent, immutable baseline snapshots."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .curriculum import party_signature, read_json, write_json
from .checkpoints import resolve_checkpoints
from .evaluation import digest, validate_suite
from .linked_reward import LinkedRewardConfig
from .teams import TeamSpec


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked(path, expected=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"baseline input is missing: {path}")
    checksum = _hash(path)
    if expected is not None and checksum != expected:
        raise ValueError(f"baseline input checksum changed: {path}")
    return dict(source=str(path), sha256=checksum)


def _relative(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"campaign input path must be relative: {value}")
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"campaign input escapes its directory: {value}")
    return path


def _saved_path(value, fallback):
    """Known local snapshot paths also work after relocating a campaign directory."""
    fallback = Path(fallback)
    if fallback.is_file():
        return fallback
    if value:
        return Path(value).expanduser()
    return fallback


def _reports(directory):
    attempts = sorted((p for p in directory.glob("attempt-*") if p.is_dir()
                       and re.fullmatch(r"attempt-\d+", p.name)),
                      key=lambda p: int(p.name.split("-")[-1]))
    reports = []
    for attempt in attempts:
        marker = attempt / "report.json"
        if not marker.exists():
            continue
        report = read_json(marker)
        if (report.get("status") not in ("hold", "promote") or
                report.get("stage") != int(directory.name.split("-")[-1])):
            raise ValueError(f"invalid completed attempt report: {marker}")
        checksum = report.get("candidate_sha256")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"invalid candidate checksum: {marker}")
        current = report.get("current", {}).get("all", {})
        score = report.get("win_rate")
        if score is None and current.get("attempts"):
            score = current.get("wins", 0) / current["attempts"]
        if score is None:
            score = current.get("win_rate")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"invalid evaluation score: {marker}")
        cohort = (report.get("mode", "original"), report.get("settings_revision", 0))
        if not isinstance(cohort[0], str) or type(cohort[1]) is not int:
            raise ValueError(f"invalid evaluation cohort: {marker}")
        path = _saved_path(report.get("candidate"), attempt / "train/checkpoints/final.zip")
        reports.append(dict(directory=attempt, report=report, path=path,
                            score=score, cohort=cohort))
    return attempts, reports


def _candidate(path, selection, expected=None, **details):
    return {**_checked(path, expected), "selection": selection, **details}


def _latest(attempts, reports):
    by_attempt = {entry["directory"]: entry for entry in reports}
    for attempt in reversed(attempts):
        report = by_attempt.get(attempt)
        if report and report["path"].is_file():
            return _candidate(report["path"], "latest completed evaluation",
                              report["report"]["candidate_sha256"], attempt=attempt.name,
                              evaluated=True, training_completed=True)
        summary_path = attempt / "train/summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        complete = summary.get("status") == "completed"
        names = ("final.zip", "latest.zip") if complete else ("latest.zip", "final.zip")
        for name in names:
            path = attempt / "train/checkpoints" / name
            if path.is_file():
                return _candidate(path, "latest recoverable saved training checkpoint",
                                  attempt=attempt.name, evaluated=False,
                                  training_completed=complete)
    raise ValueError("no recoverable checkpoint in this stage; provide --latest-checkpoint")


def _team_paths(root, config, stage, validation_dir):
    groups = config.get("team_groups") or [[name] for name in config["teams"]]
    count = len(config["curriculum"]["stages"])
    validation = config.get("validation_groups")
    if (len(groups) != count + 1 or any(not isinstance(g, list) or not g for g in groups)
            or [name for group in groups for name in group] != config["teams"]):
        raise ValueError("campaign team groups disagree with its stages/teams")
    if validation and (len(validation) != count + 1 or any(not isinstance(g, list) or not g for g in validation)):
        raise ValueError("campaign validation groups disagree with its stages")
    for group in groups + (validation or []):
        for name in group:
            if name not in config["input_hashes"]:
                raise ValueError(f"team group references an unhashed campaign input: {name}")
            _relative(root, name)
    train = [_relative(root, name) for group in groups[:stage] for name in group]
    if validation_dir is not None:
        folder = Path(validation_dir).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"validation directory is missing: {folder}")
        valid = sorted(folder.rglob("*.json"))
        if any(not p.resolve().is_relative_to(folder) for p in valid):
            raise ValueError("validation directory contains a symlink outside that directory")
        source = "explicit validation directory"
    elif validation:
        valid = [_relative(root, name) for group in validation[:stage] for name in group]
        source = "campaign validation groups (previously used for promotion)"
    else:
        raise ValueError("campaign has no separate validation groups; provide --validation-dir with distinct teams")
    if not train or not valid:
        raise ValueError("training and validation pools must both contain teams")
    signatures = set()
    for path in train:
        TeamSpec.load(path)
        signatures.add(party_signature(path))
    for path in valid:
        TeamSpec.load(path)
        if party_signature(path) in signatures:
            raise ValueError(f"training/validation duplicate party (including reordered copies): {path}")
    return train, valid, source


def _opponents(root, config, completed, dumb_override, smart_override, warnings):
    saved_dumb, saved_smart = None, []
    if completed and completed[-1]["report"].get("evaluation_plan") and (dumb_override is None or smart_override is None):
        report = completed[-1]
        directory = report["directory"] / "evaluation-pool"
        suite_path = _saved_path(report["report"]["evaluation_plan"], directory / "suite.json")
        suite = validate_suite(read_json(suite_path))
        marker = read_json(suite_path.parent / "plan.json")
        if marker.get("suite_sha256") != digest(suite):
            raise ValueError("saved campaign opponent pool schedule changed")
        for key, entry in suite.get("opponents", {}).items():
            if entry["kind"] != "checkpoint":
                continue
            if ((entry["group"] == "dumb" and dumb_override is not None) or
                    (entry["group"] == "smart" and smart_override is not None)):
                continue
            path = _saved_path(entry["path"], suite_path.parent / "models" / f"{entry['sha256']}.zip")
            row = {**_checked(path, entry["sha256"]), "label": entry.get("label", key),
                   "selection": "latest completed evaluation's frozen opponent pool"}
            if entry["group"] == "dumb":
                if saved_dumb is not None:
                    raise ValueError("saved pool contains multiple Dumb models; provide a single explicit baseline opponent")
                saved_dumb = row
            elif entry["group"] == "smart":
                saved_smart.append(row)
    if dumb_override is not None:
        dumb = {**_checked(dumb_override), "label": "Dumb", "selection": "explicit override"}
    elif saved_dumb:
        dumb = saved_dumb
    elif config.get("evaluation", {}).get("dumb_checkpoint"):
        entry = config["evaluation"]["dumb_checkpoint"]
        path = _saved_path(entry["path"], root / "evaluation-models" / f"{entry['sha256']}.zip")
        dumb = {**_checked(path, entry["sha256"]), "label": "Dumb", "selection": "campaign's explicit Dumb reference"}
    else:
        dumb = {**_checked(root / "inputs/opponent.zip", config["input_hashes"].get("inputs/opponent.zip")),
                "label": "Initial reference (Dumb category)", "selection": "initial campaign opponent fallback"}
        warnings.append("Dumb falls back to the initial campaign opponent; that model may be strong. Use --dumb-checkpoint to select a known weak reference.")
    if smart_override is not None:
        smart = [{**_checked(smart_override), "label": "Smart", "selection": "explicit override"}]
    elif saved_smart:
        smart = saved_smart
    else:
        smart = [{**_checked(root / "inputs/start.zip", config["input_hashes"].get("inputs/start.zip")),
                  "label": "Campaign starting model", "selection": "campaign starting model fallback"}]
        warnings.append("No saved Smart evaluation pool was available; Smart uses the campaign starting model. Live model directories were not searched.")
    result = {"random": dict(kind="random", group="random", label="Random legal policy"),
              "dumb": dict(kind="checkpoint", group="dumb", **dumb)}
    seen = set()
    for row in smart:
        if row["sha256"] in seen:
            continue
        seen.add(row["sha256"])
        result[f"smart_{len(seen)}"] = dict(kind="checkpoint", group="smart", **row)
    if dumb["sha256"] in seen:
        warnings.append("Dumb and at least one Smart opponent have identical SHA-256 weights; their category labels do not indicate different strength.")
    return result


def _suite(paths, opponents, *, episodes, seed, max_steps):
    assets, keys = {}, []
    for path in paths:
        asset = dict(team=read_json(path))
        key = digest(asset)
        assets[key] = asset
        keys.append(key)
    rng = random.Random(seed)
    cases = []
    for group in ("random", "dumb", "smart"):
        options = [key for key, row in opponents.items() if row["group"] == group]
        pair_count = len(keys)**2
        pair_indexes = []
        # A partial shuffled cycle samples unique ordered matchups without
        # materializing the potentially enormous Cartesian team product.
        while len(pair_indexes) < episodes // 2:
            pair_indexes.extend(rng.sample(range(pair_count), min(pair_count, episodes//2-len(pair_indexes))))
        selected = [options[i % len(options)] for i in range(episodes // 2)]
        rng.shuffle(selected)
        for index, opponent_id in enumerate(selected):
            matchup = len(cases)//2 + 1
            first, second = divmod(pair_indexes[index], len(keys))
            team_pair = [keys[first], keys[second]]
            battle_seed, policy_seed = rng.randrange(2**31-1), rng.randrange(2**31-1)
            for side in (1, 2):
                cases.append(dict(episode=len(cases)+1, matchup=matchup, assets=list(team_pair),
                                  seed=battle_seed, candidate_player=side, opponent_id=opponent_id,
                                  opponent_group=group, opponent_seed=policy_seed))
    pool = {key: {**entry, **({"path": entry["source"]} if entry["kind"] == "checkpoint" else {})}
            for key, entry in opponents.items()}
    return validate_suite(dict(version=1, backend="showdown", seed=seed, swap_sides=True,
                               max_steps=max_steps, assets=assets, cases=cases, opponents=pool))


def build_plan(campaign_run, *, stage=1, episodes_per_opponent=1000, seed=26001, max_steps=None,
               starting_checkpoint=None, best_checkpoint=None, latest_checkpoint=None,
               dumb_checkpoint=None, smart_checkpoint=None, validation_dir=None):
    """Select retained sources and embed suites without creating or changing files."""
    root = Path(campaign_run).expanduser().resolve()
    config = read_json(root / "campaign.json")
    if (config.get("version") != 1 or not config.get("input_hashes") or
            not isinstance(config.get("curriculum", {}).get("stages"), list)):
        raise ValueError("unrecognized campaign.json")
    if type(stage) is not int or not 1 <= stage <= len(config["curriculum"]["stages"]):
        raise ValueError("stage is outside the campaign")
    if type(episodes_per_opponent) is not int or episodes_per_opponent < 2 or episodes_per_opponent % 2:
        raise ValueError("--episodes-per-opponent must be positive and even")
    max_steps = config.get("max_steps", 200) if max_steps is None else max_steps
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    required = {"inputs/start.zip", "inputs/opponent.zip", "inputs/reward.json"}
    if not required.issubset(config["input_hashes"]):
        raise ValueError("campaign is missing pinned start/opponent/reward input hashes")
    for name, expected in config["input_hashes"].items():
        _checked(_relative(root, name), expected)
    overrides = resolve_checkpoints([starting_checkpoint, best_checkpoint, latest_checkpoint,
                                     dumb_checkpoint, smart_checkpoint], Path(__file__).resolve().parents[2])
    starting_checkpoint, best_checkpoint, latest_checkpoint, dumb_checkpoint, smart_checkpoint = overrides
    warnings = []
    directory = root / f"stage-{stage:02d}"
    anchor, anchor_hash = directory / "anchor.zip", None
    if (directory / "stage.json").exists():
        metadata = read_json(directory / "stage.json")
        anchor_hash = metadata.get("anchor_sha256")
        if not isinstance(anchor_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", anchor_hash):
            raise ValueError("stage anchor is missing its recorded checksum")
        _checked(anchor, anchor_hash)
    elif stage == 1:
        anchor = root / "inputs/start.zip"
        anchor_hash = config["input_hashes"].get("inputs/start.zip")
    elif starting_checkpoint is None:
        raise ValueError("selected stage has no recorded anchor; provide --starting-checkpoint")
    attempts, reports = _reports(directory)
    candidates = {}
    candidates["starting"] = (_candidate(starting_checkpoint, "explicit starting override") if starting_checkpoint is not None
                              else _candidate(anchor, "selected stage's entering model", anchor_hash))
    if best_checkpoint is not None:
        candidates["best"] = _candidate(best_checkpoint, "explicit best override")
    else:
        if not reports:
            raise ValueError("no completed evaluations for this stage; provide --best-checkpoint")
        cohort = reports[-1]["cohort"]
        eligible = [r for r in reports if r["cohort"] == cohort and r["path"].is_file()]
        if not eligible:
            raise ValueError("no retained evaluated candidate in the latest evaluation cohort; provide --best-checkpoint")
        selected = max(eligible, key=lambda r: (r["score"], int(r["directory"].name.split("-")[-1])))
        candidates["best"] = _candidate(selected["path"], "best retained evaluated candidate in latest cohort",
            selected["report"]["candidate_sha256"], attempt=selected["directory"].name,
            score=selected["score"], cohort=list(cohort), evaluated=True, training_completed=True)
        removed = [r for r in reports if r["cohort"] == cohort and not r["path"].is_file()]
        if removed:
            warnings.append(f"{len(removed)} evaluated candidate(s) in this cohort were pruned; 'best' means best retained candidate, not necessarily the historical maximum.")
        warnings.append("Historical best selection uses noisy evaluation scores; this fresh fixed baseline is the comparison, not proof that the selected model is superior.")
    candidates["latest"] = (_candidate(latest_checkpoint, "explicit latest override") if latest_checkpoint is not None
                            else _latest(attempts, reports))
    if candidates["latest"].get("training_completed") is False:
        warnings.append("Latest is a saved checkpoint from an incomplete/interrupted attempt; it may precede that attempt's final in-memory updates.")
    if latest_checkpoint is None and attempts and candidates["latest"].get("attempt") != attempts[-1].name:
        warnings.append("The newest attempt has no recoverable saved model; Latest uses the newest earlier attempt with available weights.")
    train, valid, validation_source = _team_paths(root, config, stage, validation_dir)
    opponents = _opponents(root, config, reports, dumb_checkpoint, smart_checkpoint, warnings)
    for role, candidate in candidates.items():
        aliases = [key for key, row in opponents.items() if row.get("sha256") == candidate["sha256"]]
        if aliases:
            warnings.append(f"{role} has identical weights to opponent(s) {', '.join(aliases)}; those cells are intentional self-play controls.")
    reward_path = root / "inputs/reward.json"
    reward = _checked(reward_path, config["input_hashes"].get("inputs/reward.json"))
    reward["value"] = asdict(LinkedRewardConfig.load(reward_path))
    suites = {}
    for offset, (split, paths) in enumerate((("training", train), ("validation", valid))):
        suites[split] = dict(team_count=len(paths), team_sources=[_checked(p) for p in paths],
            battles=3*episodes_per_opponent,
            suite=_suite(paths, opponents, episodes=episodes_per_opponent,
                         seed=seed+offset*1000003, max_steps=max_steps))
    return dict(version=1, campaign_run=str(root), campaign_sha256=_hash(root / "campaign.json"),
                stage=stage, seed=seed, max_steps=max_steps, episodes_per_opponent=episodes_per_opponent,
                stage_anchor=dict(path=str(anchor), sha256=anchor_hash,
                                  metadata=str(directory / "stage.json") if (directory / "stage.json").exists() else None),
                candidates=candidates, opponents=opponents, reward=reward, suites=suites,
                validation_source=validation_source, warnings=warnings)


def save_plan(plan, output_dir, *, published_dir=None):
    """Snapshot inputs, optionally embedding paths for a later atomic publication.

    With a distinct ``published_dir``, files are written only to ``output_dir``.
    The caller must rename that staging directory and then call ``verify_plan``.
    """
    if plan.get("version") != 1 or "plan_sha256" in plan:
        raise ValueError("expected an unsaved baseline plan")
    output = Path(output_dir).expanduser().resolve()
    published = output if published_dir is None else Path(published_dir).expanduser().resolve()
    source_root = Path(plan["campaign_run"]).resolve()
    if output.is_relative_to(source_root) or published.is_relative_to(source_root):
        raise ValueError("baseline output must be outside the source campaign")
    if output.exists() and (not output.is_dir() or any(p.name != ".lock" for p in output.iterdir())):
        raise ValueError("new baseline output must be empty (apart from its lock)")
    saved = deepcopy(plan)
    data = {}
    entries = [*saved["candidates"].values(), *[r for r in saved["opponents"].values() if r["kind"] == "checkpoint"]]
    for row in entries + [saved["reward"]]:
        content = Path(row["source"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"baseline source changed since preview: {row['source']}")
        data[row["sha256"]] = content
    for entry in saved["suites"].values():
        for team in entry["team_sources"]:
            _checked(team["source"], team["sha256"])
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / "snapshots/models"
    snapshots.mkdir(parents=True)
    for row in entries:
        path = snapshots / f"{row['sha256']}.zip"
        if not path.exists():
            path.write_bytes(data[row["sha256"]])
        row["path"] = str(published / path.relative_to(output))
    reward_path = output / "snapshots/reward.json"
    reward_path.write_bytes(data[saved["reward"]["sha256"]])
    saved["reward"]["path"] = str(published / reward_path.relative_to(output))
    for split, entry in saved["suites"].items():
        suite = entry.pop("suite")
        for key, row in suite["opponents"].items():
            if row["kind"] == "checkpoint":
                row["path"] = saved["opponents"][key]["path"]
        validate_suite(suite)
        path = output / "suites" / f"{split}.json"
        path.parent.mkdir(exist_ok=True)
        write_json(path, suite)
        entry.update(path=str(published / path.relative_to(output)), suite_sha256=digest(suite))
    saved["plan_sha256"] = digest(saved)
    write_json(output / "baseline.json", saved)
    return verify_plan(output) if published == output else saved


def verify_plan(output_dir):
    """Verify a saved baseline without reading its original campaign or sources."""
    output = Path(output_dir).expanduser().resolve()
    saved = read_json(output / "baseline.json")
    checksum = saved.get("plan_sha256")
    payload = {key: value for key, value in saved.items() if key != "plan_sha256"}
    if saved.get("version") != 1 or digest(payload) != checksum:
        raise ValueError("saved baseline plan checksum changed")
    entries = [*saved["candidates"].values(), *[r for r in saved["opponents"].values() if r["kind"] == "checkpoint"], saved["reward"]]
    for row in entries:
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(output):
            raise ValueError("saved baseline snapshot points outside its directory")
        _checked(path, row["sha256"])
    LinkedRewardConfig.load(saved["reward"]["path"])
    for entry in saved["suites"].values():
        path = Path(entry["path"]).resolve()
        if not path.is_relative_to(output):
            raise ValueError("saved baseline suite points outside its directory")
        suite = validate_suite(read_json(path))
        if digest(suite) != entry["suite_sha256"]:
            raise ValueError("saved baseline evaluation suite changed")
        for key, row in suite["opponents"].items():
            pinned = saved["opponents"][key]
            if row["kind"] != pinned["kind"] or row["group"] != pinned["group"]:
                raise ValueError("saved baseline suite disagrees with opponent pool")
            if row["kind"] == "checkpoint" and any(row[field] != pinned[field] for field in ("path", "sha256")):
                raise ValueError("saved baseline suite opponent identity changed")
    return saved
