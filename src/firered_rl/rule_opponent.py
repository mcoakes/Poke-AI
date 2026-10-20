"""Observation-only Gen III heuristics, inspired by trainer AI, not a retail port.

This is a stateless inference policy. No simulator object, selected enemy action,
reward, or checkpoint weights enter its decision. The legal mask is authoritative.
"""
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from .ppo_features import policy_space
from .runtime_logs import RotatingLog, log_settings


class RuleBasedOpponent:
    VERSION = "gen3-rules-v1"

    def __init__(self, project):
        root = Path(project)
        paths = [Path(__file__), root / "config/generation/firered_us_v1_0.json",
                 root / "config/generation/gen3_type_chart.json"]
        data = json.loads(paths[1].read_text())
        self.moves = {m["id"]: {**m, "name": name} for name, m in data["moves"].items()}
        self.chart = np.asarray(json.loads(paths[2].read_text())["multipliers"])
        self.identity = dict(kind="rule_based", version=self.VERSION, observation_version="v2",
                             sha256=hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest())
        self.observation_space = policy_space("v2")
        self.action_space = gym.spaces.Discrete(11)
        # Inference interface shared with frozen PPO opponents; no trainable weights.
        self.policy = self
        self.log = None
        self.log_context = []

    def set_log_context(self, rows):
        self.log_context = rows  # Identifiers for logging only; never read by choose().

    def set_training_mode(self, mode):
        if mode:
            raise ValueError("rule-based opponents are inference only")

    def set_random_seed(self, seed):
        pass  # Stable lowest-action tie breaking; no random state.

    def configure_log(self, directory, args):
        if getattr(args, "rule_log", False):
            settings, _ = log_settings(args)
            self.log = RotatingLog(Path(directory) / "rule-decisions.jsonl", **settings)

    @staticmethod
    def stage(value, accuracy=False):
        value = max(-6, min(6, value))
        base = 3 if accuracy else 2
        return (base + value) / base if value >= 0 else base / (base - value)

    @staticmethod
    def slot(obs, side):
        return int(obs["active"][0, 8] if side == 0 else obs["opponent_active_index"][0]) - 1

    def mon(self, obs, side, slot):
        row = obs["v2_party"][side, slot]
        active = slot == self.slot(obs, side)
        hp, maximum, item, ability = obs["v2_active"][side] if active else row[[2, 3, 6, 7]]
        types = obs["active"][side, 5:7] if active else row[4:6]
        stats = (obs["v2_active_stats"][side] if active else obs["v2_stats"][side, slot]).copy()
        boosts = obs["v2_boosts"][side] if active else np.zeros(7)
        stats *= np.array([self.stage(x) for x in boosts[:5]])
        status = (obs["status_bits"][6 + side] if active else obs["v2_status_bits"][side, slot])
        if status[6]:
            stats[4] *= .25  # Gen III paralysis speed.
        if int(ability) in (37, 74):  # Huge Power / Pure Power.
            stats[0] *= 2
        if int(item) == 186:  # Choice Band in FireRed's item table.
            stats[0] *= 1.5
        return dict(hp=float(hp), maximum=max(1., float(maximum)), types=set(map(int, types)),
                    stats=stats, boosts=boosts, status=status, level=float(row[1]),
                    ability=int(ability), item=int(item))

    def effectiveness(self, kind, defender):
        multiplier = math.prod(float(self.chart[kind, t]) for t in defender["types"])
        ability = defender["ability"]
        if ((ability == 26 and kind == 4) or (ability == 10 and kind == 13)
                or (ability == 11 and kind == 11) or (ability == 18 and kind == 10)):
            return 0.  # Levitate, Volt Absorb, Water Absorb, Flash Fire.
        if ability == 25 and multiplier <= 1:
            return 0.  # Wonder Guard (simplified; Struggle handled separately).
        return multiplier

    def damage(self, obs, side, slot, move_slot, target_slot):
        """Expected approximate direct damage and hit probability, not a damage oracle."""
        own, foe = self.mon(obs, side, slot), self.mon(obs, 1-side, target_slot)
        moves = obs["moves" if side == 0 else "opponent_moves"]
        move_id, _, pp, _ = moves[slot, move_slot]
        power, accuracy, always, category, priority, kind = obs["v2_move_details"][side, slot, move_slot]
        move_id, kind, category = int(move_id), int(kind), int(category)
        if not move_id or pp <= 0 or category == 2:
            return 0., 0.
        hit = 1. if always else min(1., accuracy / 100 * self.stage(own["boosts"][5] - foe["boosts"][6], True))
        multiplier = 1. if move_id == 165 else self.effectiveness(kind, foe)
        if multiplier == 0:
            return 0., hit
        fixed = {49: 20., 82: 40., 69: own["level"], 101: own["level"], 162: foe["hp"] / 2}
        if move_id in fixed:
            return fixed[move_id] * hit, hit
        if power <= 1:
            return 0., hit  # Unsupported variable/counter/OHKO damage: no invented power.
        attack, defense = (0, 1) if category == 0 else (2, 3)
        value = ((2 * own["level"] / 5 + 2) * power * own["stats"][attack]
                 / max(1., foe["stats"][defense]) / 50 + 2) * .925
        if kind in own["types"]:
            value *= 1.5
        if category == 0 and own["status"][4] and own["ability"] != 62:
            value *= .5
        if own["ability"] == 62 and own["status"].any() and category == 0:
            value *= 1.5
        if foe["ability"] == 47 and kind in (10, 15):  # Thick Fat.
            value *= .5
        weather = int(obs["v2_weather"][0])
        if kind in (10, 11) and weather in (1, 2) and own["ability"] not in (13, 76) and foe["ability"] not in (13, 76):
            value *= 1.5 if (weather, kind) in ((1, 10), (2, 11)) else .5
        return float(value * multiplier * hit), float(hit)

    def threat(self, obs, side, slot, target):
        return max(self.damage(obs, side, slot, move, target)[0] for move in range(4))

    def move_score(self, obs, slot, target, move):
        own, foe = self.mon(obs, 0, slot), self.mon(obs, 1, target)
        move_id = int(obs["moves"][slot, move, 0])
        name = self.moves.get(move_id, {}).get("name", f"move {move_id}")
        damage, hit = self.damage(obs, 0, slot, move, target)
        incoming = self.threat(obs, 1, target, slot)
        if damage:
            score = min(damage, foe["hp"]) / foe["maximum"]
            # Prefer a likely KO, especially before the expected incoming attack.
            if damage / max(hit, .001) >= foe["hp"]:
                priority = obs["v2_move_details"][0, slot, move, 4]
                faster = priority > 0 or (priority == 0 and own["stats"][4] >= foe["stats"][4])
                score += hit * (1. if faster else .65)
            return score, f"{name}: estimated damage {damage:.1f}, hit {hit:.0%}"
        power, accuracy, always, category, _, kind = obs["v2_move_details"][0, slot, move]
        hit = 1. if always else min(1., accuracy / 100 * self.stage(own["boosts"][5] - foe["boosts"][6], True))
        score, reason = -.2, "unsupported or ineffective effect"
        if move_id in (105, 135, 208, 234, 235, 236, 156):
            missing = own["maximum"] - own["hp"]
            heal = min(missing, own["maximum"] * (1. if move_id == 156 else .5))
            if missing > .3 * own["maximum"] and incoming < own["hp"] and heal > incoming:
                score = (heal - incoming) / own["maximum"] * .9
                if move_id == 156:
                    score -= .2  # Rest spends subsequent turns asleep.
                reason = "recover missing HP after estimated incoming damage"
        statuses = {92: "poison", 77: "poison", 78: "paralysis", 86: "paralysis",
                    147: "sleep", 79: "sleep", 95: "sleep", 261: "burn"}
        if move_id in statuses:
            status = statuses[move_id]
            immune = bool(foe["status"].any())
            immune |= bool(obs["v2_volatile_bits"][1, 24])  # Substitute.
            immune |= status == "poison" and (bool(foe["types"] & {3, 8}) or foe["ability"] == 17)
            immune |= status == "burn" and (10 in foe["types"] or foe["ability"] == 41)
            immune |= status == "sleep" and foe["ability"] in (15, 72)
            immune |= status == "paralysis" and (foe["ability"] == 7 or
                                                   (move_id == 86 and (4 in foe["types"] or foe["ability"] == 10)))
            # Safeguard/Substitute aren't fully represented: this remains a heuristic.
            if not immune and incoming < own["hp"]:
                score = hit * (.38 if status == "sleep" else .26) * min(1., foe["hp"] / foe["maximum"] * 2)
                reason = f"apply {status} to healthy target"
        setup = {14: (0,), 74: (2,), 97: (4,), 339: (0, 1), 347: (2, 3), 349: (0, 4)}
        if move_id in setup and own["hp"] > 2.5 * incoming and foe["hp"] > .5 * foe["maximum"]:
            stages = [own["boosts"][x] for x in setup[move_id]]
            usable = (obs["moves"][slot, :, 0] > 0) & (obs["moves"][slot, :, 2] > 0)
            categories = obs["v2_move_details"][0, slot, usable, 3]
            useful = (0 not in setup[move_id] or 0 in categories) and (2 not in setup[move_id] or 1 in categories)
            if useful and max(stages) < 2:
                score, reason = .3, "setup with HP buffer; stop at +2"
        return score, f"{name}: {reason}"

    def choose(self, obs, mask=None):
        mask = np.asarray(obs["action_mask"] if mask is None else mask, dtype=bool)
        if mask.shape != (11,) or not np.array_equal(mask, np.asarray(obs["action_mask"], dtype=bool)):
            raise ValueError("rule opponent requires the observation's authoritative action mask")
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise ValueError("rule opponent received no legal actions")
        slot, target = self.slot(obs, 0), self.slot(obs, 1)
        if len(legal) == 1:
            action = int(legal[0])
            return action, [dict(action=action, score=0., reason="only legal action (includes wait/forced continuation)")]
        if not 0 <= target < 6:
            raise ValueError("rule opponent needs a valid opponent active slot")
        forced = bool(obs["context"][1])
        if not forced and not 0 <= slot < 6:
            raise ValueError("rule opponent needs a valid local active slot")
        scores = []
        current_in = self.threat(obs, 1, target, slot) if not forced else 0.
        for action in legal:
            action = int(action)
            if action < 4:
                score, reason = self.move_score(obs, slot, target, action)
            elif action < 10:
                replacement = action - 4
                mon = self.mon(obs, 0, replacement)
                incoming = self.threat(obs, 1, target, replacement)
                outgoing = self.threat(obs, 0, replacement, target)
                foe = self.mon(obs, 1, target)
                score = min(outgoing, foe["hp"]) / foe["maximum"] - incoming / max(1., mon["hp"])
                if not forced:
                    # Pay the free incoming hit and require a meaningful defensive gain.
                    score -= .4 + incoming / mon["maximum"]
                    current = self.mon(obs, 0, slot)
                    score += min(1., current_in / max(1., current["hp"])) * .5
                reason = f"switch to slot {replacement+1}: estimated incoming {incoming:.1f}, outgoing {outgoing:.1f}; " + ("forced replacement" if forced else "includes lost-turn cost")
            else:
                score, reason = -10., "wait/escape fallback"
            scores.append(dict(action=action, score=float(score), reason=reason))
        # Stateless, deterministic tie breaking also makes batched evaluation reproducible.
        return max(scores, key=lambda row: (row["score"], -row["action"]))["action"], scores

    def predict(self, observation, *, action_masks, deterministic=True, **kwargs):
        batch = observation["action_mask"].ndim == 2
        rows = len(observation["action_mask"]) if batch else 1
        actions = []
        for i in range(rows):
            obs = {key: value[i] if batch else value for key, value in observation.items()}
            action, scores = self.choose(obs, action_masks[i] if batch else action_masks)
            actions.append(action)
            if self.log:
                context = self.log_context[i] if i < len(self.log_context) else {}
                self.log.append(json.dumps(dict(**context, version=self.VERSION, batch_row=i, turn=int(obs["context"][0]),
                    own_species=int(obs["active"][0, 1]), opponent_species=int(obs["active"][1, 1]),
                    action=action, candidates=scores)) + "\n")
        return np.asarray(actions, dtype=np.int64) if batch else np.int64(actions[0]), None
