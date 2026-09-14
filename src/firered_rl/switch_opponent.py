"""A probabilistic type-matchup opponent, using only the policy observation.

This changes player 2's training actions; it never modifies learner actions,
rewards, model weights, or evaluation behavior. Scores are a type heuristic,
not predicted damage: abilities, hazards and stat boosts are not modeled.
"""
import json
import math
from pathlib import Path
import random

import numpy as np


def switch_settings(chance=0.0, cooldown=2):
    if type(chance) not in (int, float) or not math.isfinite(chance) or not 0 <= chance <= 1:
        raise ValueError("opponent switch chance must be a fraction from 0 to 1")
    if type(cooldown) is not int or cooldown < 0:
        raise ValueError("opponent switch cooldown must be a nonnegative integer")
    return dict(chance=float(chance), cooldown=cooldown)


def normal_decision(obs):
    return (not obs["context"][1] and not obs["context"][2]
            and bool(np.any(obs["action_mask"][:4])))


class SwitchingOpponent:
    def __init__(self, project, count, *, chance, cooldown=2, seed=1):
        self.settings = switch_settings(chance, cooldown)
        data = json.loads((Path(project) / "config/generation/firered_us_v1_0.json").read_text())
        chart = json.loads((Path(project) / "config/generation/gen3_type_chart.json").read_text())
        self.chart = np.asarray(chart["multipliers"], dtype=float)
        if self.chart.shape != (18, 18) or chart["generation"] != 3:
            raise ValueError("expected the Gen III type chart in FireRed type-ID order")
        self.moves = {value["id"]: value for value in data["moves"].values()}
        names = json.loads((Path(project) / "mgba/profiles/firered_us_v1_0_names.json").read_text())
        self.names = {int(key): name for key, name in names["species"].items()}
        self.rngs = [random.Random(seed + 104729*i) for i in range(count)]
        self.remaining = np.zeros(count, dtype=int)

    def reset(self, indices):
        self.remaining[indices] = 0

    def effectiveness(self, attack, defenders):
        # A single type appears twice in the observation: multiply it only once.
        return math.prod(self.chart[int(attack), int(t)] for t in set(defenders))

    def attacks(self, moves):
        result = []
        for move, kind, pp, _ in moves:
            entry = self.moves.get(int(move))
            # Hidden Power and Weather Ball need state absent from this observation.
            if (entry and entry["power"] > 0 and entry["category"] != "status"
                    and pp > 0 and int(move) not in (237, 311)):
                result.append(int(kind))
        return result

    def matchup(self, mon, moves, foe, foe_attacks):
        own_attacks = self.attacks(moves)
        if not own_attacks:
            return None
        outgoing = max(self.effectiveness(t, foe[5:7]) for t in own_attacks)
        incoming = max(self.effectiveness(t, mon[5:7]) for t in foe_attacks)
        hp = min(1.0, float(mon[3]) / max(1.0, float(mon[4])))
        # One score point means a factor-of-two improvement, including HP fraction.
        score = math.log2(max(.25, outgoing)) - math.log2(max(.25, incoming)) + math.log2(max(.01, hp))
        return dict(score=score, outgoing=outgoing, incoming=incoming, hp=hp)

    def best_switch(self, obs):
        if not normal_decision(obs):
            return None
        active, foe = obs["active"]
        current_slot = int(active[8]) - 1
        if not active[0] or not foe[0] or not 0 <= current_slot < 6 or active[3] <= 0 or foe[3] <= 0:
            return None
        enemy_slot = int(obs["opponent_active_index"][0]) - 1
        enemy_moves = obs["opponent_moves"][enemy_slot] if 0 <= enemy_slot < 6 else []
        incoming_types = self.attacks(enemy_moves) or list(set(foe[5:7]))
        current = self.matchup(active, obs["moves"][current_slot], foe, incoming_types)
        if current is None:
            return None
        candidates = []
        for slot, mon in enumerate(obs["party"]):
            if slot == current_slot or not obs["action_mask"][4+slot] or not mon[0] or mon[9] or mon[3] <= 0:
                continue
            match = self.matchup(mon, obs["moves"][slot], foe, incoming_types)
            if (match is None or match["hp"] < .35 or match["incoming"] > current["incoming"]
                    or (match["outgoing"] <= 1 and match["incoming"] >= 1)
                    or match["score"] < current["score"] + 1):
                continue
            candidates.append((match["score"], slot, match))
        if not candidates:
            return None
        _, slot, match = max(candidates, key=lambda row: (row[0], -row[1]))
        return dict(action=4+slot, from_slot=current_slot+1, to_slot=slot+1,
                    species=self.names.get(int(obs["party"][slot, 1]), str(obs["party"][slot, 1])),
                    current=current, replacement=match)

    def choose(self, observations, actions):
        result = np.asarray(actions).copy()
        events = []
        for index, action in enumerate(actions):
            obs = {key: value[index] for key, value in observations.items()}
            if not normal_decision(obs):
                continue
            blocked = self.remaining[index] > 0
            if blocked:
                self.remaining[index] -= 1
            if (not blocked and self.settings["chance"] > 0
                    and self.rngs[index].random() < self.settings["chance"]):
                switch = self.best_switch(obs)
                if switch and switch["action"] != int(action):
                    result[index] = switch["action"]
                    events.append(dict(pair=index+1, turn=int(obs["context"][0]),
                                       policy_action=int(action), **switch))
            if 4 <= int(result[index]) <= 9 and obs["action_mask"][int(result[index])]:
                # A policy-selected voluntary switch also starts the cooldown.
                self.remaining[index] = self.settings["cooldown"]
        return result, events
