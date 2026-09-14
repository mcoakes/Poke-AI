from firered_rl.actions import GBAAction
from firered_rl.backends import MockBattleBackend
from firered_rl.multiplayer import LinkBattleSession

session = LinkBattleSession(MockBattleBackend(), MockBattleBackend())
session.reset(seed=42)
result = session.run_turn(lambda state: GBAAction.A, lambda state: GBAAction.A)
while not result.terminated:
    result = session.run_turn(lambda state: GBAAction.A, lambda state: GBAAction.A)
print(f"final rewards: {result.rewards}")
session.close()

