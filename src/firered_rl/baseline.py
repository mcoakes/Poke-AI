"""Masked baseline episodes with flushed action logs and failure diagnostics."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import traceback


def choose_action(mask, rng, allow_escape=False, moves_first=False):
    choices = [i for i, allowed in enumerate(mask) if allowed and (allow_escape or i != 10)]
    if moves_first:
        choices = [i for i in choices if i < 4] or choices
    if not choices:
        raise RuntimeError("no permitted masked action; capture this decision for debugging")
    return rng.choice(choices)


def replay_actions(path, episode):
    actions = []
    with Path(path).open() as source:
        for line in source:
            event = json.loads(line)
            if event.get("event") == "action_requested" and event.get("episode") == episode:
                action = event["action"]
                if type(action) is not int or not 0 <= action <= 10:
                    raise ValueError("invalid action in replay log")
                actions.append(action)
    if not actions:
        raise ValueError("no requested actions for that replay episode")
    return actions


def run_baseline(env, output, episodes=10, seed=1, allow_escape=False,
                 moves_first=False, replay=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    summary = dict(episodes=[], outcomes={}, stopped_on_error=False)
    # Refuse to overwrite an earlier action log.
    with (output / "events.jsonl").open("x") as log:
        def emit(event, **fields):
            record = dict(event=event, utc=datetime.now(timezone.utc).isoformat(), **fields)
            log.write(json.dumps(record, default=lambda value: value.tolist())+"\n")
            log.flush()
        emit("run", seed=seed, episodes=episodes, allow_escape=allow_escape,
             policy="moves-first" if moves_first else "random-masked", replay=replay,
             port=getattr(getattr(env, "backend", None), "port", None),
             emulator_pid=getattr(env, "pid", None))
        try:
            for episode in range(1, episodes+1):
                rng = random.Random(seed+episode-1)
                total, steps = 0.0, 0
                emit("reset_requested", episode=episode, seed=seed+episode-1)
                try:
                    obs, info = env.reset(seed=seed+episode-1)
                    reset = getattr(getattr(env, "backend", None), "last_reset_response", {})
                    emit("reset_completed", episode=episode, observation=obs, info=info,
                         checkpoint={k: reset.get(k) for k in ("state_path", "frame", "ram_fnv1a", "seed_applied")})
                    while True:
                        if not env.observation_space.contains(obs):
                            raise RuntimeError("observation outside declared space")
                        if replay is not None:
                            if steps == len(replay):
                                outcome = "replay_exhausted"
                                break
                            action = replay[steps]
                            if not obs["action_mask"][action]:
                                raise RuntimeError(f"replay diverged: action {action} is masked at decision {steps+1}")
                        else:
                            action = choose_action(obs["action_mask"], rng, allow_escape, moves_first)
                        emit("action_requested", episode=episode, decision=steps+1,
                             action=action, mask=obs["action_mask"], observation=obs)
                        obs, reward, terminated, truncated, info = env.step(action)
                        steps += 1
                        total += reward
                        if not env.observation_space.contains(obs):
                            raise RuntimeError("returned observation outside declared space")
                        emit("action_completed", episode=episode, decision=steps,
                             reward=reward, total_reward=total, terminated=terminated,
                             truncated=truncated, info=info, observation=obs)
                        if terminated or truncated:
                            outcome = info.get("outcome_reason", "unknown") if terminated else "truncated"
                            break
                    result = dict(episode=episode, decisions=steps, total_reward=total, outcome=outcome)
                    results.append(result)
                    emit("episode_completed", **result)
                    print(f"episode={episode} decisions={steps} reward={total:g} outcome={outcome}", flush=True)
                except Exception as error:
                    failure_kind = "timeout" if isinstance(error, TimeoutError) or "timed out" in str(error).lower() else "error"
                    results.append(dict(episode=episode, decisions=steps, total_reward=total, outcome=failure_kind))
                    summary["stopped_on_error"] = True
                    emit("failure", episode=episode, decision=steps+1, kind=failure_kind, error=str(error),
                         traceback=traceback.format_exc(), last_inspection=getattr(env, "report", None))
                    print(f"episode={episode} ERROR: {error}\nDiagnostics: {output / 'events.jsonl'}", flush=True)
                    break
        except KeyboardInterrupt:
            summary["interrupted"] = True
            emit("interrupted")
        finally:
            summary["episodes"] = results
            summary["outcomes"] = dict(Counter(r["outcome"] for r in results))
            (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary
