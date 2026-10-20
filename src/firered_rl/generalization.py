"""Build an auditable new-team comparison from two saved training runs."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .teams import PokemonSpec


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def party_identity(team):
    """Ignore labels, member order and move order; retain battle-relevant sets."""
    party = team.get("party")
    if not isinstance(party, list) or not 1 <= len(party) <= 6:
        raise ValueError("team must contain 1–6 Pokemon")
    normalize = lambda text: text.lower().replace(" ", "")
    members = []
    for member in party:
        row = asdict(PokemonSpec.from_dict(member))
        row["species"] = normalize(row["species"])
        row["held_item"] = normalize(row["held_item"])
        row["moves"] = sorted(normalize(move) for move in row["moves"])
        members.append(json.dumps(row, sort_keys=True))
    return checksum(json.dumps(sorted(members)).encode())


def team_files(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"team directory is missing: {directory}")
    paths = sorted(directory.rglob("*.json"))
    if not paths:
        raise ValueError(f"no team JSONs found: {directory}")
    return paths


def prepare(previous_run, candidate_run, team_dir, *, exclude_dirs=(), exclude_suites=()):
    """Read and validate every input before creating an output directory."""
    from .evaluation import digest, validate_suite
    from .linked_reward import LinkedRewardConfig
    from .periodic_evaluation import suite_opponent_path

    known, provenance, models, suites, rewards = {}, [], {}, [], []

    def remember(team, source):
        known.setdefault(party_identity(team), source)

    def remember_suite(path):
        suite = validate_suite(read_json(path))
        if suite["backend"] != "showdown":
            raise ValueError("generalization comparison requires Showdown suites")
        for asset in suite["assets"].values():
            remember(asset["team"], str(path))
        provenance.append(dict(path=str(path), suite_sha256=digest(suite)))
        return suite

    runs = [Path(p).expanduser().resolve() for p in (previous_run, candidate_run)]
    for label, run in zip(("previous", "candidate"), runs):
        config = read_json(run / "config.json")
        if config.get("backend") != "showdown":
            raise ValueError(f"not a Showdown training run: {run}")
        entries = config.get("teams", {}).get("teams", [])
        if not entries:
            raise ValueError(f"missing recorded training team inventory: {run}")
        for entry in entries:
            path = Path(entry["path"]).expanduser().resolve()
            data = path.read_bytes()
            if checksum(data) != entry["sha256"]:
                raise ValueError(f"recorded training team changed: {path}")
            remember(json.loads(data), str(path))
            provenance.append(dict(path=str(path), sha256=checksum(data)))
        path = run / "inputs/evaluation/suite.json"
        suites.append(remember_suite(path))
        reward = asdict(LinkedRewardConfig.load(run / "inputs/reward.json"))
        rewards.append(reward)
        data = (run / "checkpoints/best.zip").read_bytes()
        best = read_json(run / "checkpoints/best.json")
        # best.json is the published identity of the selected weights.
        if checksum(data) != best["checkpoint_sha256"]:
            raise ValueError(f"best checkpoint changed during read or disagrees with metadata: {run}")
        models[label] = dict(data=data, sha256=checksum(data), source=str(run / "checkpoints/best.zip"))

    if digest(suites[0]) != digest(suites[1]):
        raise ValueError("runs must use the same fixed validation suite and opponents")
    if rewards[0] != rewards[1]:
        raise ValueError("runs have different rewards; comparison must use the same reward settings")
    for folder in exclude_dirs:
        for path in team_files(folder):
            remember(read_json(path), str(path))
            provenance.append(dict(path=str(path), sha256=checksum(path.read_bytes())))
    for path in exclude_suites:
        remember_suite(Path(path).expanduser().resolve())

    accepted, rejected, seen = [], [], {}
    for path in team_files(team_dir):
        data = path.read_bytes()
        team = json.loads(data)
        signature = party_identity(team)
        duplicate = known.get(signature) or seen.get(signature)
        if duplicate:
            rejected.append(dict(path=str(path), overlaps=str(duplicate),
                                 reason="known_team" if signature in known else "duplicate_test_team"))
        else:
            seen[signature] = str(path)
            accepted.append(dict(path=str(path), sha256=checksum(data), signature=signature, data=data))
    if not accepted:
        raise ValueError(f"no unseen team configurations remain; excluded all {len(rejected)} teams")

    pool = suites[1].get("opponents", {})
    if {row["group"] for row in pool.values()} != {"random", "dumb", "smart"}:
        raise ValueError("validation suite must contain Random, Dumb and Smart opponents")
    opponents = {}
    for key, entry in pool.items():
        row = dict(entry)
        if row["kind"] == "checkpoint":
            path = suite_opponent_path(runs[1] / "inputs/evaluation/suite.json", row)
            row["data"] = path.read_bytes()
            if checksum(row["data"]) != row["sha256"]:
                raise ValueError(f"frozen opponent checksum mismatch: {path}")
        opponents[key] = row
    return dict(teams=accepted, excluded=rejected, exclusions=provenance, models=models,
                opponents=opponents, reward=rewards[1], max_steps=suites[1]["max_steps"],
                source_suite_sha256=digest(suites[1]))


def publish(plan, output, *, episodes_per_opponent, seed):
    """Pin all inputs and use the baseline's balanced matchup scheduler."""
    from .baseline_setup import _suite
    from .evaluation import digest

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    def write_json(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    def snapshot(name, data):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    paths = [Path(snapshot(f"inputs/teams/{index:04d}.json", row["data"]))
             for index, row in enumerate(plan["teams"], 1)]
    models = {label: {k: v for k, v in row.items() if k != "data"} |
              {"path": snapshot(f"inputs/models/{label}.zip", row["data"])}
              for label, row in plan["models"].items()}
    opponents = {}
    for key, row in plan["opponents"].items():
        entry = {k: v for k, v in row.items() if k != "data"}
        if "data" in row:
            entry["source"] = snapshot(f"inputs/opponents/{row['sha256']}.zip", row["data"])
        opponents[key] = entry
    suite = _suite(paths, opponents, episodes=episodes_per_opponent, seed=seed,
                   max_steps=plan["max_steps"])
    write_json("suite.json", suite)
    write_json("inputs/reward.json", plan["reward"])
    audit = dict(accepted=[{k: v for k, v in row.items() if k != "data"} for row in plan["teams"]],
                 excluded=plan["excluded"], checked_sources=plan["exclusions"],
                 scope="Recorded training teams and validation suites from both runs, plus explicit exclusions. "
                       "Checkpoint ancestors and unrecorded historical inputs are not automatically known.")
    write_json("team-check.json", audit)
    write_json("comparison-config.json", dict(models=models, source_suite_sha256=plan["source_suite_sha256"],
               suite_sha256=digest(suite), seed=seed, episodes_per_opponent=episodes_per_opponent))
    return models
