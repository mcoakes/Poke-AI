"""Controlled simulator checks against native facts, independent of adapter reports."""
from copy import deepcopy
import json
from pathlib import Path
import re
from threading import Event
from types import SimpleNamespace

import numpy as np

from .ppo_features import policy_observation, policy_space
from .runtime_logs import RotatingLog, log_settings
from .showdown_env import ShowdownFactory, showdown_observation, spectator_lines
from .turn_env import observation_space


def mon(species, moves, level=50, **fields):
    return dict(species=species, level=level, moves=moves, **fields)


def same(actual, expected, field):
    actual = actual.tolist() if isinstance(actual, np.ndarray) else actual
    if actual != expected:
        raise AssertionError(f"{field}: expected {expected!r}; observed {actual!r}")


def key(value):
    return re.sub(r"[^a-z0-9]", "", value.lower().replace("♀", "f").replace("♂", "m"))


def major_status(pokemon):
    status = pokemon["status"]
    if status == "slp":
        return max(1, min(7, pokemon["status_state"].get("time", 1)))
    if status == "tox":
        return 128 | min(15, pokemon["status_state"].get("stage", 0)) << 8
    return {"": 0, "fnt": 0, "psn": 8, "brn": 16, "frz": 32, "par": 64}[status]


def reference_mask(side, ended):
    """Map native request choices to the public action contract without runner.choices()."""
    mask = [0] * 11
    if ended:
        return mask
    request = side["request"]
    if not request or request.get("wait"):
        mask[10] = 1
        return mask
    active = side["party"][side["active_slot"]-1]
    forced = bool(request.get("forceSwitch", [False])[0])
    if not forced:
        moves = request["active"][0]["moves"]
        forced_move = any(name in active["volatiles"] for name in
                          ("twoturnmove", "lockedmove", "mustrecharge"))
        if forced_move:
            mask[10] = 1
            return mask
        for move in moves:
            if move.get("disabled") or move.get("pp", 1) <= 0:
                continue
            if move["id"] == "struggle":
                mask[0] = 1
            else:
                for index, original in enumerate(active["moves"]):
                    if original["id"] == move["id"] and not original.get("disabled"):
                        mask[index] = 1
    if forced or not active["trapped"]:
        for pokemon in side["party"]:
            if pokemon["hp"] > 0 and not pokemon["fainted"] and pokemon["slot"] != side["active_slot"]:
                mask[pokemon["slot"]+3] = 1
    return mask


class BattleAudit:
    def __init__(self, root, output, *, node, timeout, seed, log_max_mb, log_backups):
        root, output = Path(root), Path(output)
        args = SimpleNamespace(seed=seed, node=node, command_timeout=timeout,
                               log_max_mb=log_max_mb, log_backups=log_backups,
                               step_logs=False, keep_episode_logs=0)
        self.factory = ShowdownFactory(root, output, [0, 1], lambda: [], args, Event(),
                                       runner=root / "sim/showdown/audit_runner.cjs")
        options, _ = log_settings(args)
        self.log = RotatingLog(output / "audit-diagnostics.jsonl", **options)
        self.battle_log = RotatingLog(output / "battle.log", **options)
        tables = json.loads((root / "mgba/profiles/firered_us_v1_0_names.json").read_text())
        self.ids = {group: {key(name): int(number) for number, name in rows.items()}
                    for group, rows in tables.items()}
        self.seed = [int(seed) % 65536, 11, 23, 37]
        self.checkpoints = 0
        self.case = "initializing"
        self.raw, self.oracle, self.observations = None, None, None
        self.hp = {}

    def reset(self, first, second):
        self.hp = {}
        self.raw = self.factory.request("reset", teams=[dict(party=first), dict(party=second)], seed=self.seed)
        return self.capture()

    def step(self, actions):
        self.raw = self.factory.request("step", actions=actions)
        return self.capture()

    def inspect(self):
        return self.factory.request("audit_inspect")

    def capture(self):
        self.oracle = self.inspect()
        self.observations = [showdown_observation(r, self.raw["ended"]) for r in self.raw["reports"]]
        self.checkpoints += 1
        self.log.append(json.dumps(dict(case=self.case, checkpoint=self.checkpoints,
            oracle=self.oracle, observations=[{k: v.tolist() for k, v in o.items()} for o in self.observations])) + "\n")
        self.battle_log.append(f"\n[{self.case}]\n" + "\n".join(spectator_lines(self.raw.get("protocol", []), self.hp)) + "\n")
        for i, observation in enumerate(self.observations):
            self.verify_observation(i, observation)
        return self.observations

    def types(self, values):
        result = [self.ids["types"][key(value)] for value in values]
        return (result + result[:1])[:2]

    def species(self, value):
        return self.ids["species"][key(value["base_species"])]

    def move_values(self, moves):
        result = [[0, 0, 0, 0] for _ in range(4)]
        for i, move in enumerate(moves):
            result[i] = [move["num"], self.ids["types"][key(move["type"])], move["pp"], move["maxpp"]]
        return result

    def verify_observation(self, i, obs):
        native = self.oracle
        own, foe = native["sides"][i], native["sides"][1-i]
        prefix = f"P{i+1}"
        if not observation_space().contains(obs):
            raise AssertionError(f"{prefix} raw observation outside declared space")
        converted = policy_observation(obs)
        if not policy_space().contains(converted):
            raise AssertionError(f"{prefix} PPO observation outside declared space")
        for slot in range(6):
            if slot >= len(own["party"]):
                same(obs["party"][slot], [0]*10, f"{prefix} empty own party {slot+1}")
                same(obs["moves"][slot], [[0]*4]*4, f"{prefix} empty own moves {slot+1}")
            else:
                p = own["party"][slot]
                held = self.ids["items"][key(p["item"]["name"])] if p["item"]["id"] else 0
                expected = [1, self.species(p["base_species"]), p["level"], p["hp"], p["maxhp"],
                            *self.types(p["base_types"]), major_status(p), held, 0]
                same(obs["party"][slot], expected, f"{prefix} own party {slot+1}")
                moves = p["moves"] if p["active"] else p["base_moves"]
                same(obs["moves"][slot], self.move_values(moves), f"{prefix} own move slots {slot+1}")
            if slot >= len(foe["party"]):
                same(obs["opponent_party"][slot], [0, 0], f"{prefix} empty opponent party {slot+1}")
                same(obs["opponent_moves"][slot], [[0]*4]*4, f"{prefix} empty opponent moves {slot+1}")
            else:
                p = foe["party"][slot]
                same(obs["opponent_party"][slot], [1, self.species(p["base_species"])], f"{prefix} opponent {slot+1}")
                same(obs["opponent_moves"][slot], self.move_values(p["moves"] if p["active"] else p["base_moves"]),
                     f"{prefix} opponent move slots {slot+1}")
        same(obs["opponent_active_index"], [foe["active_slot"]], f"{prefix} opponent active slot")
        for side_index, side in enumerate((own, foe)):
            p = side["party"][side["active_slot"]-1]
            hp, maximum = (p["hp"], p["maxhp"]) if side_index == 0 else (round(100*p["hp"]/p["maxhp"]), 100)
            expected = [1, self.species(p["species"]), p["level"], hp, maximum,
                        *self.types(p["types"]), major_status(p), p["slot"] if side_index == 0 else 0]
            same(obs["active"][side_index, :9], expected, f"{prefix} active row {side_index}")
            status_bits = [(major_status(p) >> n) & 1 for n in range(32)]
            same(converted["status_bits"][6+side_index], status_bits, f"{prefix} status bit expansion {side_index}")
        # Independently check two explicit cartridge fields against native
        # mechanics. Confusion's remaining counter is ``time`` in Showdown,
        # not the generic residual-effect ``duration`` field.
        own_active = own["party"][own["active_slot"]-1]
        own_volatile = int(obs["active"][0, 9])
        same(bool(own_volatile & (1 << 24)), "substitute" in own_active["volatiles"],
             f"{prefix} own Substitute flag")
        confusion = own_active["volatiles"].get("confusion")
        same(own_volatile & 7, confusion["time"] if confusion else 0,
             f"{prefix} own confusion remaining counter")
        same(int(obs["active"][1, 9]), 0, f"{prefix} opponent volatiles omitted by contract")
        same(converted["volatile_bits"][0], [(own_volatile >> n) & 1 for n in range(32)],
             f"{prefix} own volatile bit expansion")
        same(converted["volatile_bits"][1], [0]*32, f"{prefix} opponent volatile bits omitted")
        same(obs["action_mask"], reference_mask(own, native["ended"]), f"{prefix} native request legality")
        forced = bool((own["request"] or {}).get("forceSwitch", [False])[0])
        outcome = 0 if not native["ended"] else 3 if not native["winner"] else 1 if native["winner"] == own["name"] else 2
        same(obs["context"], [max(0, native["turn"]-1), int(forced), int(native["ended"]), outcome], f"{prefix} context")

    def active(self, side):
        record = self.oracle["sides"][side]
        return record["party"][record["active_slot"]-1]

    def legal(self, side):
        return np.flatnonzero(self.observations[side]["action_mask"]).tolist()

    def used(self, side, move):
        return any(line.startswith(f"|move|p{side+1}a:") and f"|{move}|" in line
                   for line in self.raw["protocol"])

    def close(self):
        self.factory.close()


def run_battle_checks(root, output_dir, *, node="node", timeout=30, seed=26002, log_max_mb=2, log_backups=1):
    audit = BattleAudit(root, output_dir, node=node, timeout=timeout, seed=seed,
                        log_max_mb=log_max_mb, log_backups=log_backups)
    rows = []

    def check(name, action):
        audit.case = name
        before = audit.checkpoints
        try:
            details = action() or {}
            rows.append(dict(name=name, status="pass", details=dict(snapshots=audit.checkpoints-before, **details)))
        except (AssertionError, ValueError, RuntimeError, KeyError, IndexError, TypeError, OSError) as error:
            rows.append(dict(name=name, status="fail", details=dict(error=str(error), snapshots=audit.checkpoints-before)))

    def basic():
        audit.reset([mon("Charizard", ["Flamethrower"]), mon("Treecko", ["Absorb"], held_item="Leftovers")],
                    [mon("Blastoise", ["Surf"]), mon("Lapras", ["Ice Beam", "Rest", "Sleep Talk"])])
        same(audit.observations[0]["moves"][0, 0], [53, 10, 15, 15], "FireRed Flamethrower IDs/PP")
        same(int(audit.observations[0]["party"][1, 1]), 277, "FireRed Treecko internal ID")
        same(int(audit.observations[0]["party"][1, 8]), 200, "FireRed Leftovers internal ID")
        same(audit.inspect(), audit.inspect(), "inspection must not advance native state/RNG")

    def move_slots():
        moves = ["Tackle", "Tail Whip", "Water Gun", "Withdraw"]
        for side in range(2):
            for count in range(1, 5):
                for slot in range(count):
                    teams = [[mon("Blastoise", moves[:count])], [mon("Blissey", ["Splash"])]]
                    if side:
                        teams.reverse()
                    audit.reset(*teams)
                    same(audit.legal(side), list(range(count)), f"P{side+1} legal slots for {count} moves")
                    actions = [0, 0]
                    actions[side] = slot
                    audit.step(actions)
                    same(audit.used(side, moves[slot]), True, f"P{side+1} move slot {slot+1} execution")
        return dict(move_counts=[1, 2, 3, 4], player_sides=[1, 2], executed_actions=20)

    def party_slots():
        species = ["Charizard", "Blastoise", "Venusaur", "Pikachu", "Lapras", "Gengar"]
        for order in (species, list(reversed(species))):
            team = [mon(name, ["Splash"]) for name in order]
            audit.reset(team, team)
            for slot in [2, 6, 3, 1, 5, 4, 2]:
                audit.step([slot+3, slot+3])
                for side in range(2):
                    same(audit.active(side)["species"]["name"], order[slot-1], "switch species")
                    same(audit.oracle["sides"][side]["active_slot"], slot, "original switch slot")
        return dict(team_orders=2, switches=28)

    def invalid():
        audit.reset([mon("Magikarp", ["Splash"])], [mon("Magikarp", ["Splash"])])
        before = audit.inspect()
        for action in (-1, 1, 4, 9, 10, 11):
            try:
                audit.factory.request("step", actions=[0, action])
            except RuntimeError as error:
                if "Illegal paired actions" not in str(error):
                    raise
            else:
                raise AssertionError(f"illegal action {action} was accepted")
            same(audit.inspect(), before, f"illegal partner action {action} changed state")

    def health_status():
        audit.reset([mon("Blissey", ["Rest", "Splash"])], [mon("Blastoise", ["Surf", "Splash"])])
        maximum = audit.active(0)["hp"]
        audit.step([1, 0])
        if audit.active(0)["hp"] >= maximum:
            raise AssertionError("damage fixture did not damage its target")
        audit.step([0, 1])
        same(audit.active(0)["hp"], maximum, "Rest healing")
        same(audit.active(0)["status"], "slp", "Rest sleep status")
        for move, status in (("Toxic", "tox"), ("Will-O-Wisp", "brn"), ("Thunder Wave", "par")):
            audit.reset([mon("Mew", [move])], [mon("Blissey", ["Splash"])])
            for _ in range(15):
                audit.step([0, 0])
                if audit.active(1)["status"] == status:
                    break
            same(audit.active(1)["status"], status, move + " status")

    def restrictions():
        audit.reset([mon("Magikarp", ["Splash"]), mon("Lapras", ["Splash"])], [mon("Wobbuffet", ["Splash"])])
        same(audit.legal(0), [0], "Shadow Tag blocks ordinary switches")
        for move in ("Encore", "Disable"):
            audit.reset([mon("Alakazam", [move, "Splash"])], [mon("Blastoise", ["Splash", "Surf"])])
            audit.step([1, 0])
            for _ in range(15):
                audit.step([0, 0])
                if move.lower() in audit.active(1)["volatiles"]:
                    break
            same(audit.legal(1), [0] if move == "Encore" else [1], move + " legal moves")

    def volatile_states(side):
        teams = [[mon("Mew", ["Substitute", "Splash"])], [mon("Blissey", ["Splash"])]]
        if side:
            teams.reverse()
        audit.reset(*teams)
        audit.step([0, 0])
        same("substitute" in audit.active(side)["volatiles"], True,
             f"P{side+1} Substitute fixture established")
        # The faster target has already moved when the slower user applies
        # Confuse Ray. The first snapshot therefore retains its initial native
        # time of 2..5, making a constant presence bit distinguishable from a
        # correctly represented counter for every seed.
        teams = [[mon("Mew", ["Splash"])], [mon("Blissey", ["Confuse Ray"])]]
        if side:
            teams.reverse()
        audit.reset(*teams)
        audit.step([0, 0])
        confusion = audit.active(side)["volatiles"].get("confusion")
        if confusion is None or not 2 <= confusion["time"] <= 5:
            raise AssertionError(f"P{side+1} Confuse Ray fixture did not establish its initial 2..5 counter")
        return dict(player_side=side+1, substitute_bit=24,
                    confusion_remaining=confusion["time"])

    def exhaustion():
        audit.reset([mon("Magikarp", ["Splash"])], [mon("Magikarp", ["Splash"])])
        for _ in range(40):
            audit.step([0, 0])
        same(audit.active(0)["moves"][0]["pp"], 0, "PP exhaustion")
        same(audit.legal(0), [0], "Struggle fallback")
        audit.step([0, 0])
        same(audit.used(0, "Struggle"), True, "Struggle execution")

    def continuation(move):
        audit.reset([mon("Charizard", [move, "Splash"])], [mon("Blissey", ["Splash"])])
        for _ in range(5):
            audit.step([0, 0])
            # A missed Hyper Beam correctly has no recharge in Gen III.
            # Establish a hit before asserting the forced continuation contract.
            if move != "Hyper Beam" or not any(line.startswith("|-miss|") for line in audit.raw["protocol"]):
                break
        else:
            raise AssertionError("Hyper Beam fixture missed all five attempts; forced-continuation setup was not established")
        same(audit.legal(0), [10], move + " forced continuation")
        audit.step([10, 0])

    def replacement():
        for move in ("Baton Pass", None):
            first = [mon("Ninjask", [move]), mon("Vaporeon", ["Splash"])] if move else [mon("Magikarp", ["Splash"], 1), mon("Lapras", ["Splash"])]
            second = [mon("Magikarp", ["Splash"])] if move else [mon("Blastoise", ["Surf"])]
            audit.reset(first, second)
            audit.step([0, 0])
            same(audit.legal(0), [5], "replacement choices")
            same(audit.legal(1), [10], "partner wait")
            audit.step([5, 10])
            same(audit.oracle["sides"][0]["active_slot"], 2, "replacement execution")

    def transform():
        audit.reset([mon("Mew", ["Transform", "Psychic", "Protect", "Rest"])], [mon("Magikarp", ["Splash"])])
        audit.step([0, 0])
        same(audit.active(0)["transformed"], True, "Transform fixture")

    def permutations():
        original = ["Tackle", "Tail Whip", "Water Gun", "Withdraw"]
        for side in range(2):
            for moves in (original, [original[i] for i in (2, 0, 3, 1)]):
                for index, move in enumerate(moves):
                    teams = [[mon("Blastoise", moves)], [mon("Blissey", ["Splash"])]]
                    if side:
                        teams.reverse()
                    audit.reset(*teams)
                    actions = [0, 0]
                    actions[side] = index
                    audit.step(actions)
                    same(audit.used(side, move), True, f"P{side+1} permuted action maps to visible move")
        return dict(player_sides=[1, 2], move_orders=2, executed_actions=16)

    def side_symmetry():
        first = [mon("Charizard", ["Splash"]), mon("Pikachu", ["Splash"])]
        second = [mon("Blastoise", ["Splash"]), mon("Lapras", ["Splash"])]
        audit.reset(first, second)
        before = deepcopy(audit.observations)
        audit.step([5, 5])
        after = deepcopy(audit.observations)
        audit.reset(second, first)
        for i in range(2):
            for field in before[i]:
                same(audit.observations[1-i][field], before[i][field].tolist(), f"side swap reset {field}")
        audit.step([5, 5])
        for i in range(2):
            for field in after[i]:
                same(audit.observations[1-i][field], after[i][field].tolist(), f"side swap switched {field}")
        # Unequal-speed, certain knockout avoids assuming mirrored random battles share trajectories.
        for loser in range(2):
            teams = [[mon("Magikarp", ["Splash"], 1)], [mon("Blastoise", ["Surf"])]]
            if loser:
                teams.reverse()
            audit.reset(*teams)
            audit.step([0, 0])
            same(audit.raw["ended"], True, "terminal fixture")
            same(int(audit.observations[loser]["context"][3]), 2, "losing player's terminal outcome")
            same(int(audit.observations[1-loser]["context"][3]), 1, "winning player's terminal outcome")

    try:
        for name, function in (("native_observation_oracle", basic), ("all_move_counts_and_slots", move_slots),
                ("all_original_party_slots", party_slots), ("invalid_action_atomicity", invalid),
                ("damage_healing_major_status", health_status), ("trapping_encore_disable", restrictions),
                ("pp_exhaustion_struggle", exhaustion), ("replacement_partner_wait", replacement),
                ("transformed_sparse_moves", transform), ("move_order_permutation", permutations),
                ("player_side_symmetry", side_symmetry)):
            check(name, function)
        for side in range(2):
            check(f"own_volatile_flags_and_counters_p{side+1}", lambda side=side: volatile_states(side))
        for move in ("Fly", "Outrage", "Hyper Beam"):
            check("forced_continuation_" + key(move), lambda move=move: continuation(move))
    finally:
        audit.close()
    return rows
