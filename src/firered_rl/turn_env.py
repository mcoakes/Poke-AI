"""Linux single-battle decision environment. Experimental live menu navigation."""
import os
from pathlib import Path
import signal
import time

import gymnasium as gym
import numpy as np

from firered_rl.backends.mgba_battle import MGBABattleBackend
from firered_rl.backends.mgba_bridge import MGBAProtocolError
from firered_rl.reward import RewardConfig, battle_reward
from firered_rl.battle_rules import required_switch, restrictions


POCKETS = ("items", "key_items", "poke_balls", "tm_hm", "berries")


def decision_reward(before, after, previous, current):
    def hp(report, side):
        party = report["party"] if side == "player" else report["battle"].get("opponent_party", [])
        values = {m["slot"]: (m["species_id"], m["current_hp"]) for m in party
                  if m.get("valid") and not m.get("is_egg")}
        for m in report["battle"].get("battlers", []):
            if m.get("valid") and m["side"] == side and not m.get("absent"):
                values[m["party_slot"]] = (m["species_id"], m["current_hp"])
        return values
    cfg = RewardConfig(damage_dealt=0, damage_received=0)
    reward = battle_reward(previous, current, cfg)
    for side, weight in (("player", -1), ("opponent", 1)):
        old, new = hp(before, side), hp(after, side)
        reward += weight * sum(max(0, value[1]-new[slot][1]) for slot, value in old.items()
                               if slot in new and value[0] == new[slot][0])
    return float(reward)


def observation_space():
    # Integers are raw IDs/values, not ordinal species or move rankings.
    return gym.spaces.Dict({
        "party": gym.spaces.Box(0, 2**32-1, (6, 10), dtype=np.int64),
        "moves": gym.spaces.Box(0, 65535, (6, 4, 4), dtype=np.int64),
        "active": gym.spaces.Box(0, 2**32-1, (2, 10), dtype=np.int64),
        "bag": gym.spaces.Box(0, 65535, (5, 58, 2), dtype=np.int64),
        "context": gym.spaces.Box(0, 65535, (4,), dtype=np.int64),
        "action_mask": gym.spaces.MultiBinary(11),
        "opponent_party": gym.spaces.Box(0, 65535, (6, 2), dtype=np.int64),
        "opponent_moves": gym.spaces.Box(0, 65535, (6, 4, 4), dtype=np.int64),
        "opponent_active_index": gym.spaces.Box(0, 6, (1,), dtype=np.int64),
    })


def menu(report):
    return next(iter(report["battle"].get("menus", [])), {})


def encode(report, terminal=False):
    result = {key: np.zeros(space.shape, dtype=space.dtype)
              for key, space in observation_space().spaces.items()}
    battle = report["battle"]
    for mon in battle.get("opponent_party", []):
        if not mon.get("valid"):
            continue
        i = mon["slot"]-1
        result["opponent_party"][i] = [1, mon["species_id"]]
        for move in mon["moves"]:
            result["opponent_moves"][i, move["slot"]-1] = [move["move_id"],
                move.get("type_id", 0), move["current_pp"], move.get("max_pp", 0)]
    for mon in report["party"]:
        if not mon.get("valid"):
            continue
        i = mon["slot"]-1
        result["party"][i] = [1, mon["species_id"], mon["level"], mon["current_hp"],
            mon["max_hp"], *mon["base_type_ids"], mon["status_raw"], mon["held_item_id"], int(mon["is_egg"])]
        for move in mon["moves"]:
            result["moves"][i, move["slot"]-1] = [move["move_id"], move.get("type_id", 0),
                move["current_pp"], move.get("max_pp", 0)]
    player = None
    for mon in battle.get("battlers", []):
        if not mon.get("valid") or mon.get("absent"):
            continue
        i = 0 if mon["side"] == "player" else 1
        # Opponent active HP stays percentage-based; full moves are exposed separately.
        hp = mon["current_hp"] if i == 0 else round(100*mon["current_hp"]/mon["max_hp"])
        maximum = mon["max_hp"] if i == 0 else 100
        result["active"][i] = [1, mon["species_id"], mon["level"], hp, maximum,
            *mon["type_ids"], mon["status_raw"], mon["party_slot"] if i == 0 else 0,
            mon["volatile_status_raw"] if i == 0 else 0]
        if i == 0:
            player = mon
            for move in mon["moves"]:
                result["moves"][mon["party_slot"]-1, move["slot"]-1] = [move["move_id"],
                    move.get("base_type_id", 0), move["current_pp"], move.get("max_pp", 0)]
        else:
            slot = mon["party_slot"]-1
            result["opponent_active_index"][0] = slot+1
            # Prefer active moves/PP (including temporary changes) over party copies.
            result["opponent_moves"][slot] = 0
            for move in mon["moves"]:
                result["opponent_moves"][slot, move["slot"]-1] = [move["move_id"],
                    move.get("base_type_id", 0), move["current_pp"], move.get("max_pp", 0)]
    for i, pocket in enumerate(POCKETS):
        for item in report["bag"].get(pocket, []):
            result["bag"][i, item["slot"]-1] = [item["item_id"], item["quantity"]]
    forced = required_switch(report)
    result["context"][:] = [battle.get("turn", 0), int(forced), int(terminal), battle.get("outcome_raw", 0) if terminal else 0]
    if player is not None and not terminal:
        rules = restrictions(battle, player)
        if not forced:
            result["action_mask"][:4] = rules["moves"]
            result["action_mask"][10] = int(rules["can_run"])
        for mon in report["party"]:
            result["action_mask"][4+mon["slot"]-1] = int(mon.get("valid", False)
                and not mon.get("is_egg", True) and mon.get("current_hp", 0) > 0
                and mon["slot"] != player["party_slot"] and (forced or rules["can_switch"]))
    return result


class FireRedTurnEnv(gym.Env):
    """Moves 0..3, switches 4..9, escape 10; one reward per decision interval."""
    def __init__(self, emulator_pid, port=55356, decision_timeout=120, max_steps=500):
        super().__init__()
        self.pid = int(emulator_pid)
        command = Path(f"/proc/{self.pid}/comm").read_text().strip()
        if "mgba" not in command.lower():
            raise ValueError("emulator_pid must be the actual mgba-qt PID, not xvfb-run or a shell")
        if Path(f"/proc/{self.pid}").stat().st_uid != os.getuid():
            raise ValueError("emulator process must belong to the current user")
        self.backend = MGBABattleBackend(port=port)
        self.action_space = gym.spaces.Discrete(11)
        self.observation_space = observation_space()
        self.timeout, self.max_steps = decision_timeout, max_steps
        self.done, self.state = True, None
        self.steps = 0

    def _pause(self):
        os.kill(self.pid, signal.SIGSTOP)

    def _resume(self):
        os.kill(self.pid, signal.SIGCONT)

    def _poll(self, key=0, frames=1, guard_menu=False):
        if time.monotonic() > self.deadline:
            raise MGBAProtocolError("decision timed out; inspect the menu before retrying (episode requires reset)")
        response = (self.backend._request("step", action=key, frames=frames, guard_menu=True)
                    if guard_menu else self.backend.step_response(key, frames))
        report = response.get("inspection")
        if report is None:
            raise MGBAProtocolError(response.get("inspection_error", "missing inspection"))
        self.report = report
        return report

    def _tap(self, key):
        self._poll(0, 2)
        return self._poll(key, 2)

    def _party_confirm(self, target, submenu):
        self._poll(0, 2)
        response = self.backend._request("step", action=1, frames=1,
            expected_party_slot=target, expected_party_submenu=submenu)
        if "inspection" not in response:
            raise MGBAProtocolError(response.get("inspection_error", "missing inspection"))
        self.report = response["inspection"]

    def _select_party(self, target):
        self._wait_menu("party")
        while menu(self.report).get("submenu") != "choose_pokemon":
            self._poll()
        # Observe cursor movement after releasing each direction, not during its hold.
        while True:
            self._poll(0, 2)
            m = menu(self.report)
            if m.get("submenu") != "choose_pokemon":
                raise MGBAProtocolError("party menu changed before selecting the requested Pokemon")
            if m.get("party_slot") == target+1:
                break
            self._tap(4)
        self._party_confirm(target, "choose_pokemon")
        while menu(self.report).get("submenu") != "pokemon_actions":
            self._poll()
        if menu(self.report).get("party_slot") != target+1:
            raise MGBAProtocolError("wrong Pokemon submenu opened; refusing to switch")
        while menu(self.report).get("action_id") not in (10, 11):
            self._tap(4)
            self._poll(0, 2)
        self._party_confirm(target, "pokemon_actions")

    def _advance_text(self):
        self._poll(0, 2)
        # Recheck after release frames: the turn may have ended during that wait.
        if (menu(self.report).get("state") == "busy_or_unknown"
                and not self.report["battle"].get("outcome_raw", 0)):
            self._poll(1, 1, guard_menu=True)

    def _wait_menu(self, wanted):
        while menu(self.report).get("state") != wanted:
            if self.report["battle"].get("outcome_raw", 0):
                raise MGBAProtocolError("battle ended during menu navigation")
            self._poll()

    def _grid(self, target):
        wanted = menu(self.report).get("state")
        if wanted not in ("command", "move"):
            raise MGBAProtocolError("grid navigation requires a command or move menu")
        current = menu(self.report).get("cursor_raw")
        count = 4 if wanted == "command" else menu(self.report).get("move_count")
        if type(count) is not int or not 1 <= count <= 4:
            raise MGBAProtocolError("visible move count unavailable")
        if current not in range(count) or target not in range(count):
            raise MGBAProtocolError("grid cursor or requested target is outside the visible menu")
        while current != target:
            horizontal = current % 2 != target % 2 and (current ^ 1) < count
            key = (6 if target % 2 else 5) if horizontal else (4 if target // 2 else 3)
            self._tap(key)
            # FireRed fades the move-type/PP display after directional input.
            # The profile deliberately hides the cursor while the palette fades.
            while menu(self.report).get("state") == "busy_or_unknown":
                if self.report["battle"].get("outcome_raw", 0):
                    raise MGBAProtocolError("battle ended during grid navigation")
                self._poll(0, 1)
            if menu(self.report).get("state") != wanted:
                raise MGBAProtocolError(f"expected {wanted} menu, observed {menu(self.report).get('state')}")
            current = menu(self.report).get("cursor_raw")
            if current not in range(count):
                raise MGBAProtocolError("left menu before selecting requested action")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._resume()
        self.done = True
        self.deadline = time.monotonic()+self.timeout
        try:
            status = self.backend.status()
            if status.get("native_link_session"):
                raise MGBAProtocolError("use the link-session tester; this controller cannot schedule two linked players")
            version = tuple(int(part) for part in status.get("adapter_version", "0.0.0").split("."))
            if version < (0, 7, 0):
                raise MGBAProtocolError("turn controller requires adapter 0.7.0 for revealed moves and restrictions")
            self.state = self.backend.reset(seed)
            self.report = self.backend.last_inspection
            if "opponent_party" not in self.report.get("battle", {}):
                raise MGBAProtocolError("opponent-party data missing; extract the complete archive")
            self._wait_menu("command")
            if "turn" not in self.report["battle"]:
                raise MGBAProtocolError("requires adapter 0.6.0")
            self.state = self.backend._convert({"inspection": self.report}, resetting=True)
            self.backend.previous = self.state
            self.steps, self.done = 0, False
            return encode(self.report), {"decision": "command"}
        finally:
            self._pause()

    def step(self, action):
        if self.done:
            raise RuntimeError("call reset before stepping")
        obs = encode(self.report)
        if not self.action_space.contains(action) or not obs["action_mask"][action]:
            raise ValueError("action unavailable at this decision")
        action = int(action)
        previous = self.state
        before = self.report
        start_turn = self.report["battle"]["turn"]
        start_frame = self.report["frame"]
        forced = bool(obs["context"][1])
        player = next(b for b in self.report["battle"]["battlers"] if b["side"] == "player")
        rules = restrictions(self.report["battle"], player)
        self.deadline = time.monotonic()+self.timeout
        self._resume()
        try:
            if not forced:
                self._grid(0 if action < 4 else 2 if action < 10 else 3)
                self._tap(1)
            if action < 4 and not rules["skip_move_menu"]:
                self._wait_menu("move")
                self._grid(action)
                self._tap(1)
            elif 4 <= action < 10:
                self._select_party(action-4)
            # Never return the same command menu before the committed action runs.
            left_menu = menu(self.report).get("state") != "command"
            while True:
                battle = self.report["battle"]
                m = menu(self.report)
                terminal = battle.get("outcome_raw", 0) != 0
                replacement = required_switch(self.report)
                progressed = battle.get("turn") != start_turn or start_turn == 255
                if terminal or (replacement and not forced) or (left_menu and m.get("state") == "command" and (progressed or forced or action == 10)):
                    break
                left_menu |= m.get("state") != "command"
                # Text advances only while not in a decoded selection menu.
                if m.get("state") in ("command", "move", "party", "bag", "target"):
                    self._poll()
                else:
                    self._advance_text()
            self.state = self.backend._convert({"inspection": self.report})
            self.backend.previous = self.state
            reward = decision_reward(before, self.report, previous, self.state)
            self.steps += 1
            terminated = self.state.terminal
            truncated = self.steps >= self.max_steps and not terminated
            self.done = terminated or truncated
            return encode(self.report, terminated), reward, terminated, truncated, {
                "outcome_reason": self.state.outcome_reason, "forced_replacement": replacement,
                "turn_start": start_turn, "turn_end": battle.get("turn"),
                "frames_elapsed": self.report["frame"]-start_frame}
        except Exception:
            self.done = True
            raise
        finally:
            self._pause()

    def close(self):
        self._resume()
        self.backend.close()
