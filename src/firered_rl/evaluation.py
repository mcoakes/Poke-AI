"""Fixed-policy evaluation over an immutable, worker-independent matchup schedule."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .ppo_features import policy_observation
from .teams import PokemonSpec


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_suite(source, *, backend, episodes, seed, swap_sides, max_steps):
    if swap_sides and episodes % 2:
        raise ValueError("side-swapped evaluation requires an even --episodes count")
    choose, rng = source.for_pair(0), random.Random(seed)
    assets, cases = {}, []
    for matchup in range(episodes // (2 if swap_sides else 1)):
        paths = choose()
        keys = []
        for path in paths:
            if backend == "showdown":
                asset = dict(team=json.loads(path.read_text()))
            else:
                asset = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            key = digest(asset)
            assets[key] = asset
            keys.append(key)
        battle_seed = rng.randrange(2**31-1)
        for candidate in (0, 1) if swap_sides else (0,):
            cases.append(dict(episode=len(cases)+1, matchup=matchup+1, assets=keys,
                              seed=battle_seed, candidate_player=candidate+1))
    return dict(version=1, backend=backend, seed=seed, swap_sides=swap_sides,
                max_steps=max_steps, assets=assets, cases=cases)


def validate_suite(suite):
    if suite.get("version") != 1 or suite.get("backend") not in ("mgba", "showdown"):
        raise ValueError("unsupported evaluation suite")
    if type(suite.get("max_steps")) is not int or suite["max_steps"] < 1:
        raise ValueError("suite max_steps must be positive")
    if type(suite.get("swap_sides")) is not bool:
        raise ValueError("suite must specify swap_sides")
    assets, cases = suite.get("assets"), suite.get("cases")
    if not isinstance(assets, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("suite needs assets and cases")
    for key, asset in assets.items():
        if not isinstance(asset, dict) or digest(asset) != key:
            raise ValueError("evaluation asset hash mismatch")
        if suite["backend"] == "showdown":
            team = asset.get("team")
            if not isinstance(team, dict) or not isinstance(team.get("party"), list) or not 1 <= len(team["party"]) <= 6:
                raise ValueError("invalid team in evaluation suite")
            for mon in team["party"]:
                PokemonSpec.from_dict(mon)
        else:
            path = Path(asset["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"evaluation state changed: {path}")
    for i, case in enumerate(cases):
        if (case.get("episode") != i+1 or type(case.get("seed")) is not int
                or case.get("candidate_player") not in (1, 2)
                or not isinstance(case.get("assets"), list) or len(case["assets"]) != 2
                or any(key not in assets for key in case["assets"])):
            raise ValueError(f"invalid evaluation case {i+1}")
    opponents = suite.get("opponents")
    if opponents is not None:
        if not isinstance(opponents, dict) or not opponents:
            raise ValueError("suite opponent pool must be a nonempty object")
        for key, entry in opponents.items():
            if not isinstance(entry, dict) or entry.get("group") not in ("random", "dumb", "smart"):
                raise ValueError("invalid evaluation opponent group")
            if entry.get("kind") == "random":
                if entry["group"] != "random":
                    raise ValueError("random policy must belong to random group")
            elif entry.get("kind") == "checkpoint":
                checksum = entry.get("sha256", "")
                if (entry["group"] == "random" or not isinstance(entry.get("path"), str)
                        or not isinstance(checksum, str) or len(checksum) != 64
                        or any(c not in "0123456789abcdef" for c in checksum)):
                    raise ValueError("invalid opponent checkpoint identity")
            else:
                raise ValueError("unsupported opponent policy kind")
        for case in cases:
            if not isinstance(case.get("opponent_id"), str):
                raise ValueError("evaluation case needs a string opponent_id")
            entry = opponents.get(case.get("opponent_id"))
            if (entry is None or case.get("opponent_group") != entry["group"]
                    or type(case.get("opponent_seed")) is not int):
                raise ValueError("evaluation case disagrees with opponent pool")
    elif any("opponent_id" in case for case in cases):
        raise ValueError("evaluation cases require their saved opponent pool")
    if suite["swap_sides"]:
        if len(cases) % 2:
            raise ValueError("incomplete side-swapped suite")
        for a, b in zip(cases[::2], cases[1::2]):
            if (a["candidate_player"], b["candidate_player"]) != (1, 2) or any(
                    a.get(key) != b.get(key) for key in ("matchup", "assets", "seed", "opponent_id", "opponent_group", "opponent_seed")):
                raise ValueError("side-swapped cases must share matchup, assets and seed")
    return suite


def materialize_assets(suite, run):
    directory = run / "teams"
    directory.mkdir()
    paths = {}
    for key, asset in suite["assets"].items():
        if suite["backend"] == "showdown":
            path = directory / f"{key}.json"
            path.write_text(json.dumps(asset["team"], indent=2)+"\n")
        else:
            path = Path(asset["path"])
        paths[key] = path
    return paths


def summary(rows, *, requested, elapsed, status, suite_hash):
    wins = sum(r["candidate_outcome"] == "won" for r in rows)
    losses = sum(r["candidate_outcome"] == "lost" for r in rows)
    draws = sum(r["candidate_outcome"] == "draw" for r in rows)
    truncated = sum(r["truncated"] for r in rows)
    completed = wins+losses+draws
    mean = lambda key: float(np.mean([r[key] for r in rows])) if rows else None
    result = dict(status=status, suite_sha256=suite_hash, requested_battles=requested,
                attempted_battles=len(rows), completed_battles=completed, truncated=truncated,
                candidate_wins=wins, opponent_wins=losses, draws=draws,
                candidate_win_rate=wins/completed if completed else None,
                opponent_win_rate=losses/completed if completed else None,
                draw_rate=draws/completed if completed else None,
                candidate_score=(wins+.5*draws)/completed if completed else None,
                candidate_reward_mean=mean("candidate_reward"), opponent_reward_mean=mean("opponent_reward"),
                decisions_mean=mean("decisions"), elapsed_seconds=elapsed,
                battles_per_hour=completed*3600/elapsed if elapsed else 0,
                by_candidate_side={str(side): dict(
                    battles=sum(r["candidate_player"] == side for r in rows),
                    wins=sum(r["candidate_player"] == side and r["candidate_outcome"] == "won" for r in rows),
                    losses=sum(r["candidate_player"] == side and r["candidate_outcome"] == "lost" for r in rows))
                    for side in (1, 2)})

    if any("opponent_group" in row for row in rows):
        result["by_opponent_group"] = {}
        for group in ("random", "dumb", "smart", "basic"):
            subset = [row for row in rows if row.get("opponent_group") == group or
                      (group == "basic" and row.get("opponent_group") in ("random", "dumb"))]
            result["by_opponent_group"][group] = opponent_rates(subset)
        result["by_opponent"] = {key: opponent_rates([r for r in rows if r.get("opponent_id") == key])
                                  for key in sorted({r["opponent_id"] for r in rows})}
    return result


def opponent_rates(rows):
    n = len(rows)
    wins = sum(r["candidate_outcome"] == "won" for r in rows)
    truncated = sum(bool(r["truncated"]) for r in rows)
    return dict(attempts=n, completed=n-truncated, wins=wins, truncated=truncated,
                losses=sum(r["candidate_outcome"] == "lost" for r in rows),
                draws=sum(r["candidate_outcome"] == "draw" for r in rows),
                win_rate=wins/n if n else None)


def evaluate(sessions, selected, suite, paths, models, run, cancel, *, aborters=(), announce=print, opponents=None):
    """Central batched inference; each session advances its paired Gymnasium environment.

    Cases are assigned by precomputed episode index, never by a worker's RNG.
    No policy is trained, saved or modified here.
    """
    started = time.monotonic()
    suite_hash = digest(suite)
    rows, active = [], {}
    next_case = 0
    status = "completed"
    executor = ThreadPoolExecutor(max_workers=len(sessions))
    try:
        while next_case < len(suite["cases"]) or active:
            if cancel.is_set():
                raise KeyboardInterrupt
            resetting = {}
            for i, session in enumerate(sessions):
                if i not in active and next_case < len(suite["cases"]):
                    case = suite["cases"][next_case]
                    next_case += 1
                    selected[i][:] = [paths[key] for key in case["assets"]]
                    active[i] = dict(case=case, rewards=np.zeros(2), decisions=0,
                        opponent_rng=random.Random(case.get("opponent_seed", case["seed"] ^ 0x47B9)))
                    resetting[i] = executor.submit(session.reset_pair, [case["seed"], case["seed"]+1])
            for i, future in resetting.items():
                active[i]["observations"] = [policy_observation(r[0]) for r in future.result()]
            actions = {i: np.zeros(2, dtype=np.int64) for i in active}
            policies = [(True, None, models[0])]
            policies += ([(False, key, model) for key, model in opponents.items()] if opponents is not None
                         else [(False, None, models[1])])
            for is_candidate, key, model in policies:
                indexes, observations = [], []
                for i, job in active.items():
                    if not is_candidate and opponents is not None and job["case"]["opponent_id"] != key:
                        continue
                    side = job["case"]["candidate_player"]-1
                    if not is_candidate:
                        side = 1-side
                    indexes.append((i, side))
                    observations.append(job["observations"][side])
                if not indexes:
                    continue
                if model is None:
                    for (i, side), observation in zip(indexes, observations):
                        legal = np.flatnonzero(observation["action_mask"])
                        if not len(legal):
                            raise ValueError("random policy received an empty legal-action mask")
                        actions[i][side] = active[i]["opponent_rng"].choice(legal.tolist())
                else:
                    batch = {name: np.stack([obs[name] for obs in observations]) for name in observations[0]}
                    chosen, _ = model.predict(batch, action_masks=batch["action_mask"].astype(bool), deterministic=True)
                    for (i, side), action in zip(indexes, chosen):
                        actions[i][side] = action
            stepping = {i: executor.submit(sessions[i].step_pair, action) for i, action in actions.items()}
            for i, future in stepping.items():
                result = future.result()
                if result[0][2:4] != result[1][2:4]:
                    raise ValueError("evaluation players disagree about battle completion")
                job = active[i]
                job["rewards"] += [r[1] for r in result]
                job["decisions"] += 1
                terminated, truncated = result[0][2:4]
                if not terminated and not truncated:
                    job["observations"] = [policy_observation(r[0]) for r in result]
                    continue
                candidate = job["case"]["candidate_player"]-1
                outcomes = tuple(r[4]["outcome"] for r in result)
                if terminated and outcomes not in (("won", "lost"), ("lost", "won"), ("draw", "draw")):
                    raise ValueError(f"contradictory evaluation outcomes: {outcomes}")
                row = dict(job["case"], terminated=bool(terminated), truncated=bool(truncated),
                           candidate_outcome=outcomes[candidate] if terminated else "truncated",
                           candidate_reward=float(job["rewards"][candidate]),
                           opponent_reward=float(job["rewards"][1-candidate]), decisions=job["decisions"])
                rows.append(row)
                with (run / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
                stats = summary(rows, requested=len(suite["cases"]), elapsed=time.monotonic()-started,
                                status="running", suite_hash=suite_hash)
                (run / "summary.json").write_text(json.dumps(stats, indent=2)+"\n")
                rate = stats["candidate_win_rate"]
                rate_text = "n/a" if rate is None else f"{rate:.1%}"
                opponent_text = (f"opponent={suite['opponents'][row['opponent_id']].get('label', row['opponent_id'])} "
                                 if opponents is not None else "")
                announce(f"evaluation={len(rows)}/{len(suite['cases'])} case={row['episode']} "
                         f"candidate=P{candidate+1} {opponent_text}result={row['candidate_outcome']} | "
                         f"candidate_win={rate_text} wins/losses/draws={stats['candidate_wins']}/"
                         f"{stats['opponent_wins']}/{stats['draws']} truncated={stats['truncated']} | "
                         f"reward_mean={stats['candidate_reward_mean']:.2f} battles/hour={stats['battles_per_hour']:.1f}")
                del active[i]
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        cancel.set()
        # Unblock active requests before waiting for executor shutdown.
        for abort in aborters:
            abort()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        stats = summary(rows, requested=len(suite["cases"]), elapsed=time.monotonic()-started,
                        status=status, suite_hash=suite_hash)
        (run / "summary.json").write_text(json.dumps(stats, indent=2)+"\n")
    return stats
