"""SB3 vector API with one concurrent worker per native linked pair."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
import time

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from firered_rl.ppo_features import policy_observation, policy_space


class LinkedVecEnv(VecEnv):
    def __init__(self, sessions, *, cancel=None, aborters=None, on_episode=None, reward_scale=1,
                 observation_version="v1"):
        if not sessions:
            raise ValueError("at least one linked pair is required")
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be positive and finite")
        self.reward_scale = reward_scale
        self.observation_version = observation_version
        space = policy_space(observation_version)
        self.sessions = sessions
        self.envs = [env for session in sessions for env in session.players()]
        self.cancel = cancel or Event()
        self.aborters = aborters or []
        self.on_episode = on_episode
        self.executor = ThreadPoolExecutor(max_workers=len(sessions))
        self.pending = None
        self.closed = False
        self.observations = None
        self.returns = np.zeros((len(sessions), 2), dtype=float)
        self.lengths = np.zeros(len(sessions), dtype=int)
        self.episode_started = np.zeros(len(sessions))
        self.ready = np.zeros(len(sessions))
        super().__init__(len(sessions)*2, space, gym.spaces.Discrete(11))

    @staticmethod
    def stack(observations):
        return {key: np.stack([obs[key] for obs in observations]) for key in observations[0]}

    def _collect(self, futures):
        results = [None] * len(futures)
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return results

    def _reset_pair(self, pair, seeds):
        self.episode_started[pair] = time.monotonic()
        result = self.sessions[pair].reset_pair(seeds)
        self.ready[pair] = time.monotonic()
        self.returns[pair] = 0
        self.lengths[pair] = 0
        for player in range(2):
            self.reset_infos[pair*2+player] = result[player][1]
        return [policy_observation(item[0], version=self.observation_version) for item in result]

    def reset(self):
        if self.closed or self.pending is not None:
            raise RuntimeError("cannot reset a closed vector or while a step is pending")
        if any(self._options):
            raise ValueError("linked PPO reset options are configured by --states/--state-dir")
        futures = {self.executor.submit(self._reset_pair, i, self._seeds[i*2:i*2+2]): i
                   for i in range(len(self.sessions))}
        results = self._collect(futures)
        self.observations = self.stack([obs for pair in results for obs in pair])
        self._reset_seeds()
        self._reset_options()
        return {key: value.copy() for key, value in self.observations.items()}

    def step_async(self, actions):
        if self.closed or self.pending is not None or self.observations is None:
            raise RuntimeError("reset first, and finish the previous vector step")
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("one integer action is required per Gymnasium player")
        mask = self.observations["action_mask"]
        if any(not 0 <= a < 11 or not mask[i, a] for i, a in enumerate(actions)):
            raise ValueError("PPO supplied an unavailable action; action masking must remain enabled")
        # Validate every action before sending any of them to any pair.
        self.pending = {self.executor.submit(self._step_pair, i, actions[i*2:i*2+2].copy()): i
                        for i in range(len(self.sessions))}

    def _step_pair(self, pair, actions):
        results = self.sessions[pair].step_pair(actions)
        if results[0][2:4] != results[1][2:4]:
            raise RuntimeError("linked players disagree about the end of the battle")
        terminated, truncated = results[0][2:4]
        done = bool(terminated or truncated)
        observations = [policy_observation(row[0], version=self.observation_version) for row in results]
        rewards = [float(row[1]) for row in results]
        infos = [dict(row[4], **{"TimeLimit.truncated": bool(truncated and not terminated)}) for row in results]
        self.returns[pair] += rewards
        self.lengths[pair] += 1
        if done:
            elapsed = time.monotonic() - self.episode_started[pair]
            for player in range(2):
                infos[player]["terminal_observation"] = observations[player]
                infos[player]["episode"] = dict(r=float(self.returns[pair, player]),
                                                 l=int(self.lengths[pair]), t=elapsed)
            if self.on_episode:
                self.on_episode(pair, terminated=bool(terminated), truncated=bool(truncated),
                    infos=infos, returns=self.returns[pair].tolist(), decisions=int(self.lengths[pair]),
                    reset_seconds=self.ready[pair]-self.episode_started[pair],
                    battle_seconds=time.monotonic()-self.ready[pair])
            # Both cores reset once, in this same worker. Other pairs keep running.
            observations = self._reset_pair(pair, [None, None])
        return observations, rewards, [done, done], infos

    def step_wait(self):
        if self.pending is None:
            raise RuntimeError("step_async must precede step_wait")
        try:
            results = self._collect(self.pending)
        finally:
            self.pending = None
        self.observations = self.stack([obs for result in results for obs in result[0]])
        return ({key: value.copy() for key, value in self.observations.items()},
                np.asarray([reward for result in results for reward in result[1]], dtype=np.float32)*self.reward_scale,
                np.asarray([done for result in results for done in result[2]], dtype=bool),
                [info for result in results for info in result[3]])

    def get_attr(self, attr_name, indices=None):
        indices = list(self._get_indices(indices))
        if attr_name == "action_masks":
            return [lambda i=i: self.env_method("action_masks", indices=[i])[0] for i in indices]
        if attr_name == "observation_space":
            return [self.observation_space for _ in indices]
        if attr_name == "observation_version":
            return [self.observation_version for _ in indices]
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        for i in self._get_indices(indices):
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name != "action_masks":
            raise AttributeError(f"unsupported linked vector method: {method_name}")
        if self.observations is None:
            raise RuntimeError("reset before requesting action masks")
        return [self.observations["action_mask"][i].astype(bool).copy() for i in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancel.set()
        # Stop native processes even if a worker is inside lobby startup.
        for abort in self.aborters:
            abort()
        self.executor.shutdown(wait=True, cancel_futures=True)
        for session in self.sessions:
            session.close()
