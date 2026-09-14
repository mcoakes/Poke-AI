"""Initial single-battle Gymnasium integration using atomic inspection snapshots."""
from dataclasses import replace

from firered_rl.backends.mgba_bridge import MGBABridgeBackend, MGBAProtocolError
from firered_rl.state import BattleState


class MGBABattleBackend(MGBABridgeBackend):
    def __init__(self, host="127.0.0.1", port=55355, timeout=5.0):
        super().__init__(host, port, timeout)
        self.last_inspection = None
        self.previous = None

    def reset(self, seed=None):
        self.previous = None
        self.last_reset_response = self.reset_response(seed)
        state = self._convert(self.last_reset_response, resetting=True)
        self.previous = state
        return state

    def step(self, action, frames):
        if self.previous is None or self.previous.terminal:
            raise MGBAProtocolError("reset to an ongoing battle before stepping")
        self.previous = self._convert(self.step_response(action, frames))
        return self.previous

    def _convert(self, response, resetting=False):
        report = response.get("inspection")
        if not report:
            raise MGBAProtocolError(response.get("inspection_error", "requires adapter 0.5.0; restart launcher"))
        self.last_inspection = report
        battle = report.get("battle", {})
        reason = battle.get("outcome", "unknown")
        terminal = reason not in ("ongoing", "unknown")
        if resetting and (not battle.get("active") or terminal):
            raise MGBAProtocolError("reset checkpoint must be inside an ongoing battle, preferably at Fight/Bag/Pokemon/Run")
        if battle.get("type_flags", 0) & 2:
            raise MGBAProtocolError("link battles are not supported by this initial episode backend")
        outcome = {"won": 1, "lost": -1}.get(reason, 0)
        if not battle.get("active"):
            # Armed only by a successful in-battle reset. Never accept a stale reset result.
            if self.previous is not None and terminal:
                return replace(self.previous, terminal=True, outcome=outcome, outcome_reason=reason)
            raise MGBAProtocolError("battle ended without a recognized outcome")
        counts = [battle.get(f"{side}_counts", {}) for side in ("player", "opponent")]
        if not all(c.get("valid") for c in counts):
            raise MGBAProtocolError("invalid party counts; refusing to calculate a reward")
        battlers = {}
        for mon in battle.get("battlers", []):
            if mon.get("absent"):
                continue
            if not mon.get("valid") or mon["side"] in battlers:
                raise MGBAProtocolError("this backend requires valid single-battle records")
            battlers[mon["side"]] = mon
        if set(battlers) != {"player", "opponent"}:
            if terminal and self.previous is not None:
                return replace(self.previous, terminal=True, outcome=outcome, outcome_reason=reason)
            raise MGBAProtocolError("active battlers are not ready")
        player, opponent = battlers["player"], battlers["opponent"]
        menu = next(iter(battle.get("menus", [])), {}).get("state")
        return BattleState(
            player_hp=player["current_hp"], player_max_hp=player["max_hp"],
            opponent_hp=opponent["current_hp"], opponent_max_hp=opponent["max_hp"],
            player_active=player["party_slot"]-1, opponent_active=opponent["party_slot"]-1,
            player_fainted=counts[0]["fainted"], opponent_fainted=counts[1]["fainted"],
            menu_state={"command": 1, "move": 2, "party": 3, "bag": 4, "target": 5}.get(menu, 0),
            turn=battle.get("turn", 0), outcome=outcome, terminal=terminal, outcome_reason=reason)
