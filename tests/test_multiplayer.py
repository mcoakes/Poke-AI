from firered_rl.actions import GBAAction
from firered_rl.backends import MockBattleBackend
from firered_rl.multiplayer import LinkBattleSession


def test_link_session_steps_both_players():
    session = LinkBattleSession(MockBattleBackend(), MockBattleBackend())
    states = session.reset(seed=1)
    result = session.step((GBAAction.A, GBAAction.NOOP))
    assert result.states[0].turn == states[0].turn + 1
    assert result.states[1].turn == states[1].turn + 1
    assert len(result.rewards) == 2
    session.close()

