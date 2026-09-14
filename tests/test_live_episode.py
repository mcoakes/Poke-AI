from dataclasses import replace

import pytest

from firered_rl.backends.mgba_battle import MGBABattleBackend
from firered_rl.backends.mgba_bridge import MGBAProtocolError
from firered_rl.reward import RewardConfig, battle_reward
from firered_rl.state import BattleState


def snapshot(reason="ongoing", active=True):
    mon = dict(valid=True, absent=False, current_hp=20, max_hp=20, party_slot=1)
    return {"inspection": {"battle": dict(active=active, outcome=reason,
        player_counts=dict(valid=True, fainted=0), opponent_counts=dict(valid=True, fainted=0),
        battlers=[dict(mon, side="player"), dict(mon, side="opponent")], menus=[])}}


@pytest.mark.parametrize("reason,outcome", [("won", 1), ("lost", -1), ("ran", 0), ("caught", 0), ("draw", 0)])
def test_terminal_outcomes_and_stale_reset(reason, outcome):
    backend = MGBABattleBackend()
    backend.previous = backend._convert(snapshot(), resetting=True)
    state = backend._convert(snapshot(reason))
    assert state.terminal and state.outcome == outcome
    assert backend._convert(snapshot(reason, False)).terminal
    with pytest.raises(MGBAProtocolError):
        backend._convert(snapshot(reason, False), resetting=True)


def test_switch_does_not_reward_damage_and_terminal_bonus_once():
    before = BattleState(100, 100, 100, 100)
    switched = replace(before, opponent_active=1, opponent_hp=10)
    assert battle_reward(before, switched, RewardConfig()) == 0
    won = replace(switched, outcome=1, terminal=True)
    assert battle_reward(switched, won, RewardConfig()) == 100
    assert battle_reward(won, won, RewardConfig()) == 0
