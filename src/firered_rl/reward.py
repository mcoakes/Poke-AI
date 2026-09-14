from dataclasses import dataclass

from firered_rl.state import BattleState


@dataclass(frozen=True, slots=True)
class RewardConfig:
    damage_dealt: float = 1.0
    damage_received: float = -1.0
    knockout: float = 25.0
    fainted: float = -25.0
    win: float = 100.0
    loss: float = -100.0


def battle_reward(previous: BattleState, current: BattleState, cfg: RewardConfig) -> float:
    dealt = max(0, previous.opponent_hp - current.opponent_hp) if previous.opponent_active == current.opponent_active else 0
    received = max(0, previous.player_hp - current.player_hp) if previous.player_active == current.player_active else 0
    reward = cfg.damage_dealt * dealt + cfg.damage_received * received
    reward += cfg.knockout * max(0, current.opponent_fainted - previous.opponent_fainted)
    reward += cfg.fainted * max(0, current.player_fainted - previous.player_fainted)
    if current.outcome == 1 and previous.outcome != 1:
        reward += cfg.win
    elif current.outcome == -1 and previous.outcome != -1:
        reward += cfg.loss
    return float(reward)
