"""Coverage-balanced frozen checks for the expanding team curriculum.

Validation, stage gates, and final benchmarks are separate saved suites. A
benchmark is informational: this module never selects a checkpoint or trains it.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import time

from .broad_team_curriculum import composition_signature, semantic_signature
from .evaluation import digest, validate_suite
from .fixed_team_lab import GreedyDamage
from .periodic_evaluation import atomic_write
from .rule_opponent import RuleBasedOpponent
from .staged_team_lab import score_stage


def _checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage_suite(project, paths, episodes, seed, *, max_steps=100,
                   permute_moves=False):
    """Cover every pool team per cycle, with paired candidate seats.

The requested budget is rounded up to an even count and at least twice the
pool size. Each cycle visits every team once as the first team, with a shuffled
cyclic partner schedule. Both candidates then play the same matchup/seed from
opposite seats. Rotated suites preserve that semantic schedule exactly.
"""
    paths = tuple(Path(path).resolve() for path in paths)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("coverage suite requires nonempty, distinct team paths")
    if type(episodes) is not int or episodes < 1 or type(seed) is not int:
        raise ValueError("episodes must be positive and seed must be an integer")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    rng, battle_rng = random.Random(seed), random.Random(seed ^ 0x5C017)
    assets, keys = {}, []
    for path in paths:
        asset = dict(team=json.loads(path.read_text()))
        key = digest(asset)
        assets[key] = asset
        keys.append(key)
    count = max(len(paths), (episodes + 1) // 2)
    cases, exposures = [], Counter()
    order, shift = [], 0
    for index in range(count):
        position = index % len(paths)
        if position == 0:
            order = list(range(len(paths)))
            rng.shuffle(order)
            shift = rng.randrange(1, len(paths)) if len(paths) > 1 else 0
        pair = [keys[order[position]], keys[order[(position + shift) % len(paths)]]]
        battle_seed = battle_rng.randrange(2**31 - 1)
        for seat in (1, 2):
            actual = list(pair)
            exposures[str(paths[order[position if seat == 1 else (position + shift) % len(paths)]])]+=1
            if permute_moves:
                asset = copy.deepcopy(assets[actual[seat - 1]])
                for member in asset["team"]["party"]:
                    offset = (index + 1) % len(member["moves"])
                    member["moves"] = member["moves"][offset:] + member["moves"][:offset]
                key = digest(asset)
                assets[key] = asset
                actual[seat - 1] = key
            cases.append(dict(episode=len(cases) + 1, matchup=index + 1,
                              assets=actual, seed=battle_seed, candidate_player=seat))
    suite = dict(version=1, backend="showdown", seed=seed,
        # Candidate-only rotations change asset lists across seat pairs.
        swap_sides=not permute_moves, max_steps=max_steps, assets=assets, cases=cases,
        rule_opponent=RuleBasedOpponent(project).identity,
        diagnostic=dict(schedule="full-pool-cycles-with-paired-seats", balanced_candidate_seats=True,
            permuted_moves=permute_moves, requested_budget=episodes,
            actual_battles=len(cases), pool_teams=len(paths),
            candidate_team_exposures=dict(exposures), full_cycles=count // len(paths)))
    return validate_suite(suite)


def pool_inventory(entry):
    """Validate exact pinned membership/hashes, then summarize all four splits."""
    root = Path(entry["root"]).resolve()
    result = dict(root=str(root), splits={})
    recorded = set()
    for split in ("train", "validation", "heldout", "benchmark"):
        relative = entry.get(f"{split}_dir")
        values = entry.get(f"{split}_paths", [])
        if relative is None:
            if values:
                raise ValueError(f"{split} has paths but no directory")
            result["splits"][split] = dict(teams=0, species={}, moves={}, compositions=0,
                                           semantic_teams=0, team_sizes={})
            continue
        directory = (root / relative).resolve()
        if not directory.is_relative_to(root):
            raise ValueError(f"pool directory escapes input root: {relative}")
        expected = {(root / value).resolve() for value in values}
        if len(expected) != len(values) or set(directory.glob("*.json")) != expected:
            raise ValueError(f"pinned pool membership changed: {relative}")
        species, moves, sizes = Counter(), Counter(), Counter()
        semantic, compositions = set(), set()
        for relative_path in values:
            path = (root / relative_path).resolve()
            if not path.is_relative_to(directory):
                raise ValueError(f"team path escapes its pool: {relative_path}")
            if entry["sha256"].get(relative_path) != _checksum(path):
                raise ValueError(f"pinned team changed: {relative_path}")
            recorded.add(relative_path)
            team = json.loads(path.read_text())
            sizes[str(len(team["party"]))] += 1
            semantic.add(semantic_signature(team))
            compositions.add(composition_signature(team))
            for mon in team["party"]:
                species[mon["species"]] += 1
                moves.update(mon["moves"])
        result["splits"][split] = dict(teams=len(values), species=dict(sorted(species.items())),
            moves=dict(sorted(moves.items())), team_sizes=dict(sorted(sizes.items())),
            compositions=len(compositions), semantic_teams=len(semantic))
    if recorded != set(entry["sha256"]):
        raise ValueError("pinned hash list differs from pool membership")
    return result


def _baseline_identity(project, name):
    if name == "random":
        return dict(kind="random", version="per-case-legal-v1")
    rules = RuleBasedOpponent(project).identity
    if name == "rules":
        return rules
    if name == "greedy":
        return dict(kind="greedy_damage", version="gen3-greedy-v1", rules=rules,
                    implementation_sha256=_checksum(Path(__file__).with_name("fixed_team_lab.py")))
    raise ValueError(f"unknown baseline: {name}")


def _load_model(checkpoint):
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(checkpoint, device="cpu")
    model.policy.set_training_mode(False)
    return model


def _complete_episode_log(directory, summary):
    """A completed summary alone cannot certify a complete saved check."""
    try:
        rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines() if line]
        requested = summary["requested_battles"]
        return (len(rows) == summary["attempted_battles"] == requested
            and {row["episode"] for row in rows} == set(range(1, requested + 1))
            and sum(row["candidate_outcome"] == "won" for row in rows) == summary["candidate_wins"]
            and sum(row["candidate_outcome"] == "lost" for row in rows) == summary["opponent_wins"]
            and sum(row["candidate_outcome"] == "draw" for row in rows) == summary["draws"]
            and sum(row["candidate_outcome"] in ("won", "lost", "draw") for row in rows) == summary["completed_battles"]
            and sum(row["candidate_outcome"] == "truncated" for row in rows) == summary["truncated"]
            and sum(bool(row["truncated"]) for row in rows) == summary["truncated"])
    except (ValueError, OSError, KeyError, TypeError):
        return False


def frozen_check(project, root, label, suite_path, checkpoint=None, baseline=None,
                 opponent_kind="rules", opponent_checkpoint=None):
    """Run/reuse a frozen check, keyed by candidate, opponent, suite, and reward.

    A missing candidate checkpoint means the named baseline, defaulting to
    random legal actions. Partial runs are preserved in place and retried in a
    new directory. Non-Rules opponents are explicitly labelled in saved suites
    and summaries; bounded Rules suggestions in audits remain only a reference.
    """
    root, project, suite_path = Path(root), Path(project), Path(suite_path)
    if not label or Path(label).name != label or label in (".", ".."):
        raise ValueError("check label must be one directory name")
    if checkpoint is not None and baseline is not None:
        raise ValueError("choose a candidate checkpoint or baseline")
    if opponent_kind not in ("rules", "greedy", "checkpoint", "frozen"):
        raise ValueError(f"unsupported benchmark opponent: {opponent_kind}")
    if (opponent_kind in ("checkpoint", "frozen")) != (opponent_checkpoint is not None):
        raise ValueError("frozen opponent requires exactly one opponent checkpoint")
    candidate_identity = (dict(kind="checkpoint", sha256=_checksum(checkpoint)) if checkpoint
                          else _baseline_identity(project, baseline or "random"))
    opponent_identity = (dict(kind="checkpoint", sha256=_checksum(opponent_checkpoint))
                         if opponent_checkpoint else _baseline_identity(project, opponent_kind))
    reward_path = root / "inputs/reward.json"
    suite_sha, reward_sha = _checksum(suite_path), _checksum(reward_path)
    saved_suite = validate_suite(json.loads(suite_path.read_text()))
    suite = copy.deepcopy(saved_suite)
    if opponent_kind == "rules":
        if suite.get("rule_opponent") not in (None, opponent_identity):
            raise ValueError("frozen suite Rules identity differs from current implementation")
        suite["rule_opponent"] = opponent_identity
    else:
        suite.pop("rule_opponent", None)
    # Coverage suites have one opponent. Never retain a misleading pooled label.
    if "opponents" in suite or any("opponent_id" in case for case in suite["cases"]):
        raise ValueError("curriculum frozen checks require a single-opponent suite")
    suite["check_opponent"] = opponent_identity
    suite["diagnostic"] = dict(suite.get("diagnostic", {}), opponent_kind=opponent_kind,
        audit_comparison="Rules suggestions are a reference, not necessarily the battle opponent")
    validate_suite(suite)
    identity = dict(candidate_identity=candidate_identity, opponent_identity=opponent_identity,
        input_suite_sha256=suite_sha, effective_suite_sha256=digest(suite), reward_sha256=reward_sha)
    base = root / "checks" / label
    for directory in [base, *sorted(base.parent.glob(base.name + "-retry-*"))]:
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            previous = json.loads(summary_path.read_text())
        except (ValueError, OSError):
            continue
        if (previous.get("status") == "completed"
                and all(previous.get(k) == v for k, v in identity.items())
                and _complete_episode_log(directory, previous)):
            # The requested denominator is authoritative even for truncated runs.
            previous["wins_per_requested"] = previous["candidate_wins"] / previous["requested_battles"]
            return previous
    run = base if not base.exists() else base.with_name(base.name + f"-retry-{time.time_ns()}")
    candidate = (_load_model(checkpoint) if checkpoint else
                 {"random": lambda: None, "rules": lambda: RuleBasedOpponent(project),
                  "greedy": lambda: GreedyDamage(project)}[baseline or "random"]())
    opponent = (_load_model(opponent_checkpoint) if opponent_checkpoint else
                RuleBasedOpponent(project) if opponent_kind == "rules" else GreedyDamage(project))
    result = score_stage(project, run, suite, candidate, reward_path,
        pairs=2, trace_cases=16, trace_decisions=24, opponent=opponent)
    result.update(identity, opponent_kind=opponent_kind,
        checkpoint=str(checkpoint) if checkpoint else None,
        opponent_checkpoint=str(opponent_checkpoint) if opponent_checkpoint else None,
        result_directory=str(run.relative_to(root)),
        wins_per_requested=result["candidate_wins"] / result["requested_battles"])
    atomic_write(run / "summary.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
