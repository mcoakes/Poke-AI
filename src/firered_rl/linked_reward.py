"""Reward points measured between paired battle decision boundaries."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class LinkedRewardConfig:
    damage: float = 0
    knockout: float = 0
    win: float = 1
    loss: float = -1
    damage_exponent: float = 2
    scale: float = 1
    turn_penalty: float = 0
    damage_cap_per_pokemon: float = 0  # zero keeps legacy uncapped damage rewards

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"reward {key} must be a finite number")
        if self.damage < 0 or self.knockout < 0 or self.win <= 0 or self.loss > 0:
            raise ValueError("damage/knockout must be nonnegative, win positive, and loss nonpositive")
        if self.damage_exponent < 1 or self.scale <= 0:
            raise ValueError("damage_exponent must be >= 1 and scale must be positive")
        if self.turn_penalty < 0 or self.damage_cap_per_pokemon < 0:
            raise ValueError("turn_penalty and damage_cap_per_pokemon must be nonnegative")

    @classmethod
    def load(cls, path):
        try:
            return cls(**json.loads(Path(path).expanduser().read_text()))
        except TypeError as error:
            raise ValueError(f"invalid reward configuration: {error}") from error


def party_health(report):
    party = {}
    for mon in report["party"]:
        if not mon.get("valid") or mon.get("is_egg"):
            continue
        party[mon["slot"]] = [mon["species_id"], mon["current_hp"], mon["max_hp"]]
    for mon in report["battle"]["battlers"]:
        if mon.get("valid") and mon["side"] == "player" and mon["party_slot"] in party:
            # Active HP can precede its stored party copy, especially on a KO.
            party[mon["party_slot"]][1:] = [mon["current_hp"], mon["max_hp"]]
    if any(not 0 <= hp <= maximum or maximum <= 0 for _, hp, maximum in party.values()):
        raise ValueError("invalid party HP in reward observation")
    return party


def battle_turn(reports):
    turns = [r["battle"].get("turn") for r in reports]
    if len(turns) != 2 or any(type(t) is not int or not 0 <= t <= 255 for t in turns) or turns[0] != turns[1]:
        raise ValueError("reward accounting requires matching validated battle turn counters")
    return turns[0]


class EpisodeRewardState:
    """One ledger per linked pair, reset with each new battle."""
    def __init__(self, reports):
        self.initial_turn = self.last_turn = battle_turn(reports)
        self.charged_through = self.initial_turn-1
        self.turns = 0
        self.damage_paid = [{}, {}]  # attacking player -> opposing logical slot -> points
        self.done = False


def reward_points(previous, current, config, terminal, state=None):
    if state is None and (config.turn_penalty or config.damage_cap_per_pokemon):
        raise ValueError("turn costs and damage caps require per-episode reward state")
    turns_charged = 0
    if state is not None:
        if state.done:
            raise ValueError("reward episode already ended; reset its accounting")
        before_turn, after_turn = battle_turn(previous), battle_turn(current)
        if before_turn != state.last_turn or after_turn < before_turn:
            raise ValueError("battle turn changed out of order during reward accounting")
        # The native counter increments before the NEXT command menu, but not
        # on a terminal turn. Charge the originating turn once, plus any turns
        # completed by automatic continuations. Same-turn recipient/WAIT
        # decisions do not charge it again.
        through = max(state.charged_through, before_turn, after_turn if terminal else after_turn-1)
        turns_charged = through-state.charged_through
        paid = [dict(p) for p in state.damage_paid]
    damage, knockouts = [0.0, 0.0], [0, 0]
    if config.damage or config.knockout:
        for victim in range(2):
            before, after = party_health(previous[victim]), party_health(current[victim])
            if before.keys() != after.keys() or any(before[s][0] != after[s][0] for s in before):
                raise ValueError("party identity changed during reward accounting")
            for slot, (_, hp, maximum) in before.items():
                lost = max(0, hp-after[slot][1]) / maximum
                attacker = 1-victim
                points = config.damage * lost**config.damage_exponent
                if config.damage_cap_per_pokemon:
                    spent = paid[attacker].get(slot, 0)
                    points = min(points, max(0, config.damage_cap_per_pokemon-spent))
                    paid[attacker][slot] = spent+points
                damage[attacker] += points
                knockouts[attacker] += int(hp > 0 and after[slot][1] == 0)
    values, components = [], []
    for i, report in enumerate(current):
        outcome = report["battle"].get("outcome", "ongoing")
        final = {"won":config.win, "lost":config.loss, "draw":0}.get(outcome, 0) if terminal else 0
        breakdown = dict(damage=damage[i], knockout=config.knockout*knockouts[i], outcome=final)
        if config.turn_penalty:
            breakdown["turn_penalty"] = -config.turn_penalty*turns_charged
        values.append(float(sum(breakdown.values())))
        components.append(breakdown)
    if state is not None:
        state.damage_paid = paid
        state.charged_through, state.last_turn = through, after_turn
        state.turns += turns_charged
        state.done = terminal
    return values, components
