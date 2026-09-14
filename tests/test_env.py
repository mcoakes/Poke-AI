import numpy as np

from firered_rl.actions import GBAAction
from firered_rl.env import FireRedBattleEnv


def test_env_shapes_and_terminal_state():
    env = FireRedBattleEnv(max_episode_steps=100)
    observation, _info = env.reset(seed=7)
    assert observation.shape == (8,)
    assert observation.dtype == np.float32
    terminated = False
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(GBAAction.A)
        if terminated:
            break
    assert terminated
    assert not truncated
    env.close()


def test_time_limit_truncates():
    env = FireRedBattleEnv(max_episode_steps=1)
    env.reset(seed=7)
    _, _, terminated, truncated, _ = env.step(GBAAction.NOOP)
    assert not terminated
    assert truncated
