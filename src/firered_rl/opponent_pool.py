"""Frozen opponent pools: sample once per training episode; quota evaluation cases."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random

import gymnasium as gym
import numpy as np

from .checkpoints import resolve_checkpoints
from .ppo_features import observation_version, policy_for_version, policy_space
from .runtime_logs import RotatingLog, log_settings


class OpponentPool:
    def __init__(self, root, path, *, count=1, seed=1, device="cpu"):
        from sb3_contrib import MaskablePPO
        from .rule_opponent import RuleBasedOpponent

        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("opponent pool must be a JSON object")
        entries = data.get("opponents")
        if type(data.get("version")) is not int or data["version"] != 1 or not isinstance(entries, dict) or not entries:
            raise ValueError("opponent pool needs version: 1 and a nonempty opponents object")
        self.models, self.entries, self.blobs = {}, {}, {}
        self.inputs = [path]
        cache = {}
        for key in sorted(entries):
            entry = entries[key]
            if not isinstance(key, str) or not key or not isinstance(entry, dict):
                raise ValueError("opponent pool IDs must be nonempty strings with object entries")
            kind = entry.get("kind")
            weight = entry.get("weight", 1)
            if type(weight) not in (float, int) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"{key}: weight must be finite and positive; omit unused opponents")
            group = entry.get("group", {"checkpoint": "learned", "random": "random", "rule_based": "rules"}.get(kind))
            if ((kind == "checkpoint" and group not in ("learned", "dumb", "smart")) or
                    (kind == "random" and group != "random") or (kind == "rule_based" and group != "rules")):
                raise ValueError(f"{key}: invalid opponent group")
            info = dict(kind=kind, group=group, label=str(entry.get("label", key)), weight=float(weight))
            if kind == "checkpoint":
                value = entry.get("path")
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{key}: checkpoint needs a path")
                source = Path(value).expanduser()
                if value.lower() not in ("latest", "later") and not source.is_absolute():
                    source = path.parent / source
                source = resolve_checkpoints([source], root)[0]
                blob = source.read_bytes()
                digest = hashlib.sha256(blob).hexdigest()
                if entry.get("sha256", digest) != digest:
                    raise ValueError(f"{key}: checkpoint fingerprint changed")
                if digest not in cache:
                    cache[digest] = MaskablePPO.load(BytesIO(blob), device=device)
                model = cache[digest]
                if model.action_space != gym.spaces.Discrete(11):
                    raise ValueError(f"{key}: checkpoint action space must have 11 actions")
                info.update(path=str(source), sha256=digest, training_timesteps=model.num_timesteps,
                            observation_version=observation_version(model.observation_space))
                self.blobs[digest] = blob
                self.inputs.append(source)
            elif kind == "rule_based":
                model = RuleBasedOpponent(root)
                if entry.get("identity", model.identity) != model.identity:
                    raise ValueError(f"{key}: rule implementation fingerprint changed")
                info["identity"] = model.identity
            elif kind == "random":
                model = None
            else:
                raise ValueError(f"{key}: kind must be checkpoint, rule_based or random")
            self.models[key], self.entries[key] = model, info
        self.identity = {key: {k: v for k, v in entry.items() if k != "path"}
                         for key, entry in self.entries.items()}
        self.keys = list(self.entries)
        weights = np.array([self.entries[k]["weight"] for k in self.keys])
        self.probabilities = weights / weights.max()
        self.probabilities /= self.probabilities.sum()
        version = "v2" if any(m is not None and observation_version(m.observation_space) == "v2"
                              for m in self.models.values()) else "v1"
        self.observation_space, self.action_space = policy_space(version), gym.spaces.Discrete(11)
        self.policy = self
        self.count = count
        self.selected = [None] * count
        self.started = dict.fromkeys(self.keys, 0)
        self.completed = dict.fromkeys(self.keys, 0)
        self.log_context, self.log = [], None
        self.set_random_seed(seed)
        self.set_training_mode(False)

    def set_training_mode(self, mode):
        if mode:
            raise ValueError("opponent pools are inference only")
        for model in self.models.values():
            if model is not None:
                model.policy.set_training_mode(False)

    def set_random_seed(self, seed):
        self.rngs = [np.random.default_rng(seed + i * 104729) for i in range(self.count)]
        self.action_rngs = [np.random.default_rng(seed + i * 104729 + 3571) for i in range(self.count)]
        for model in self.models.values():
            if model is not None:
                model.set_random_seed(seed)

    def configure_log(self, directory, args):
        if getattr(args, "rule_log", False):
            settings, _ = log_settings(args)
            self.log = RotatingLog(Path(directory) / "opponent-assignments.jsonl", **settings)
        for model in self.models.values():
            if hasattr(model, "configure_log"):
                model.configure_log(directory, args)

    def set_log_context(self, rows):
        self.log_context = rows

    def reset_episodes(self, indices):
        for i in indices:
            self.selected[i] = str(self.rngs[i].choice(self.keys, p=self.probabilities))
            self.started[self.selected[i]] += 1
            if self.log:
                self.log.append(json.dumps(dict(pair=int(i)+1, opponent=self.selected[i], event="episode_start")) + "\n")

    def finish_episodes(self, indices):
        for i in indices:
            self.completed[self.selected[i]] += 1

    def metrics(self):
        return dict(started=self.started.copy(), completed=self.completed.copy(), selected=self.selected.copy())

    def predict(self, observation, *, action_masks, deterministic=True, **kwargs):
        if len(action_masks) != self.count or any(key is None for key in self.selected):
            raise ValueError("reset opponent pool episodes before predicting a complete player batch")
        result = np.empty(self.count, dtype=np.int64)
        for key, model in self.models.items():
            indices = np.array([i for i, selected in enumerate(self.selected) if selected == key], dtype=int)
            if not len(indices):
                continue
            if model is None:
                for i in indices:
                    legal = np.flatnonzero(action_masks[i])
                    if not len(legal):
                        raise ValueError("random opponent received no legal action")
                    result[i] = self.action_rngs[i].choice(legal)
            else:
                batch = {k: v[indices] for k, v in observation.items()}
                batch = policy_for_version(batch, observation_version(model.observation_space))
                if hasattr(model, "set_log_context"):
                    model.set_log_context([{**self.log_context[i], "opponent_id": key} for i in indices]
                                          if self.log_context else [])
                actions, _ = model.predict(batch, action_masks=action_masks[indices], deterministic=deterministic)
                result[indices] = actions
        if any(not action_masks[i, action] for i, action in enumerate(result)):
            raise ValueError("opponent pool selected an illegal action")
        return result, None

    def install(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        entries = deepcopy(self.entries)
        for entry in entries.values():
            if entry["kind"] == "checkpoint":
                target = directory / f"{entry['sha256']}.zip"
                target.write_bytes(self.blobs[entry["sha256"]])
                entry["path"] = str(target.resolve())
        target = directory / "pool.json"
        target.write_text(json.dumps(dict(version=1, opponents=entries), indent=2) + "\n")
        return target

    def evaluation_suite(self, suite):
        """Largest-remainder quotas in whole seat pairs; seed fixes assignment order."""
        suite = deepcopy(suite)
        if suite.get("opponents") or suite.get("rule_opponent"):
            raise ValueError("a replay suite already pins opponents; cannot replace them with --opponent-pool")
        width = 2 if suite["swap_sides"] else 1
        count = len(suite["cases"]) // width
        desired = self.probabilities * count
        quotas = np.floor(desired).astype(int)
        for i in sorted(range(len(quotas)), key=lambda i: (-(desired[i]-quotas[i]), i))[:count-int(quotas.sum())]:
            quotas[i] += 1
        if np.any(quotas == 0):
            raise ValueError("evaluation budget too small to give every opponent a matchup; increase --episodes")
        assignments = [key for key, quota in zip(self.keys, quotas) for _ in range(quota)]
        rng = random.Random(suite["seed"] ^ 0x57A3)
        rng.shuffle(assignments)
        for match, key in enumerate(assignments):
            seed = rng.randrange(2**31)
            for case in suite["cases"][match*width:(match+1)*width]:
                case.update(opponent_id=key, opponent_group=self.entries[key]["group"], opponent_seed=seed)
        suite["opponents"] = deepcopy(self.entries)
        suite["opponent_quotas"] = {key: int(quota) * width for key, quota in zip(self.keys, quotas)}
        return suite
