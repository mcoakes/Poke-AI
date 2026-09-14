"""Reproducible opponent pools for campaign evaluations."""
import hashlib
from pathlib import Path
import random

from .curriculum import read_json, stage_suites, write_json
from .evaluation import digest, validate_suite


DEFAULT_EVALUATION = dict(mode="stage", basic_win_rate=.75, smart_win_rate=.60,
                          require="both", smart_dir=None, dumb_checkpoint=None)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(config):
    return {**DEFAULT_EVALUATION, **config.get("evaluation", {})}


def copy_checkpoint(source, target):
    source, target = Path(source), Path(target)
    data = source.read_bytes()
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f"saved evaluation checkpoint changed: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    return hashlib.sha256(data).hexdigest()


def verify_pool(suite):
    validate_suite(suite)
    for entry in suite["opponents"].values():
        if entry["kind"] == "checkpoint" and file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError(f"evaluation opponent checkpoint changed: {entry['path']}")
    return suite


def prepare_pool(directory, paths, *, stage, episodes, seed, max_steps, dumb, smart):
    """Pin all opponents and the shuffled schedule once, including across retries."""
    directory = Path(directory)
    marker, suite_path = directory / "plan.json", directory / "suite.json"
    if marker.exists():
        plan, suite = read_json(marker), read_json(suite_path)
        if plan["suite_sha256"] != digest(suite):
            raise ValueError("saved evaluation pool schedule changed")
        verify_pool(suite)
        return suite_path, suite
    if episodes < 20 or episodes % 20:
        raise ValueError("mixed evaluations require --eval-episodes divisible by 20 (for 30/30/40 and side swaps)")
    if not smart:
        raise ValueError("mixed evaluation needs at least one saved Smart checkpoint")
    directory.mkdir(parents=True, exist_ok=True)
    opponents = {"random": dict(kind="random", group="random", label="Random legal policy")}

    def snapshot(key, group, label, source):
        checksum = file_hash(source)
        target = (directory / "models" / f"{checksum}.zip").resolve()
        copy_checkpoint(source, target)
        opponents[key] = dict(kind="checkpoint", group=group, label=label,
                              path=str(target), sha256=checksum)
        return checksum

    snapshot("dumb", "dumb", "Dumb", dumb)
    hashes, smart_keys = set(), []
    for label, source in smart:
        checksum = file_hash(source)
        if checksum in hashes:
            continue
        hashes.add(checksum)
        key = f"smart_{len(smart_keys)+1}"
        snapshot(key, "smart", label, source)
        smart_keys.append(key)
    suite, _ = stage_suites(paths, stage=stage, episodes=episodes, seed=seed, max_steps=max_steps)
    # Build paired quotas so each category has exactly equal candidate seats.
    units = episodes // 20
    rng = random.Random(seed ^ 0x723A5)
    chosen = ["random"]*(3*units) + ["dumb"]*(3*units)
    chosen += [rng.choice(smart_keys) for _ in range(4*units)]
    rng.shuffle(chosen)
    for i, key in enumerate(chosen):
        policy_seed = rng.randrange(2**31-1)
        for case in suite["cases"][2*i:2*i+2]:
            case.update(opponent_id=key, opponent_group=opponents[key]["group"], opponent_seed=policy_seed)
    suite["opponents"] = opponents
    verify_pool(suite)
    write_json(suite_path, suite)
    write_json(marker, dict(version=1, suite_sha256=digest(suite)))
    return suite_path, suite


def pool_promotion(rates, evaluation):
    checks = []
    for group, key in (("basic", "basic_win_rate"), ("smart", "smart_win_rate")):
        values = rates.get(group, {})
        n = values.get("attempts", 0)
        rate = values.get("wins", 0) / n if n else None
        checks.append(dict(check=group+" win rate", value=rate, threshold=evaluation[key],
                           direction="at_least", passed=rate is not None and rate+1e-12 >= evaluation[key]))
    # Missing a whole group is an incomplete evaluation, even under an either gate.
    complete = all(rates.get(group, {}).get("attempts", 0) > 0 for group in ("random", "dumb", "smart"))
    passed = (all(c["passed"] for c in checks) if evaluation["require"] == "both"
              else any(c["passed"] for c in checks))
    overall = rates["all"]
    return dict(mode="mixed-evaluation", status="promote" if complete and passed else "hold",
                checks=checks, current=rates, win_rate=overall["wins"]/overall["attempts"],
                promotion_require=evaluation["require"])
