"""Train only player 1 of each pair; player 2 uses an inference-only checkpoint."""
import json
from pathlib import Path
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from .ppo_features import observation_version as space_version, policy_for_version, policy_space
from .switch_opponent import normal_decision


class FrozenOpponentVecEnv(VecEnv):
    def __init__(self, paired, opponent, stochastic=False, switching=None, switch_log=None,
                 observation_version=None):
        self.paired, self.opponent, self.stochastic = paired, opponent, stochastic
        self.switching, self.switch_log = switching, Path(switch_log) if switch_log else None
        self.switch_counts = {f"{side}_{field}": 0 for side in ("learner", "opponent")
                              for field in ("normal_decisions", "voluntary_switches", "forced_switches")}
        self.switch_counts["heuristic_overrides"] = 0
        self.episodes = np.zeros(paired.num_envs//2, dtype=int)
        self.full = None
        if paired.num_envs % 2:
            raise ValueError("frozen opponents require complete pairs")
        paired_version = space_version(paired.observation_space)
        self.observation_version = observation_version or paired_version
        learner_space = policy_space(self.observation_version)
        self.opponent_version = space_version(opponent.observation_space)
        if paired_version == "v1" and "v2" in (self.observation_version, self.opponent_version):
            raise ValueError("a v2 learner or opponent requires a paired v2 observation environment")
        if opponent.action_space != paired.action_space:
            raise ValueError("opponent checkpoint does not match the PPO action space")
        opponent.policy.set_training_mode(False)
        super().__init__(paired.num_envs//2, learner_space, paired.action_space)

    @staticmethod
    def select(observations, offset=0):
        return {key: value[offset::2].copy() for key, value in observations.items()}

    def reset(self):
        if any(self._options):
            raise ValueError("configure resets with --states/--state-dir")
        self.paired._seeds = [value for seed in self._seeds for value in (seed, None)]
        self.full = self.paired.reset()
        self.episodes += 1
        if hasattr(self.opponent, "reset_episodes"):
            self.opponent.reset_episodes(range(self.num_envs))
        if self.switching is not None:
            self.switching.reset(slice(None))
        self.reset_infos = self.paired.reset_infos[::2]
        self._reset_seeds()
        self._reset_options()
        return policy_for_version(self.select(self.full), self.observation_version)

    def step_async(self, actions):
        if self.full is None:
            raise RuntimeError("reset before stepping")
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("one integer action is required per learning player")
        other = policy_for_version(self.select(self.full, 1), self.opponent_version)
        if hasattr(self.opponent, "set_log_context"):
            self.opponent.set_log_context([dict(pair=i+1, episode=int(episode), player=2)
                                           for i, episode in enumerate(self.episodes)])
        chosen, _ = self.opponent.predict(other, action_masks=other["action_mask"].astype(bool),
                                          deterministic=not self.stochastic)
        chosen = np.asarray(chosen)
        if chosen.shape != (self.num_envs,) or not np.issubdtype(chosen.dtype, np.integer):
            raise ValueError("opponent returned invalid actions")
        if self.switching is not None:
            chosen, events = self.switching.choose(other, chosen)
            self.switch_counts["heuristic_overrides"] += len(events)
            if events and self.switch_log is not None:
                with self.switch_log.open("a") as stream:
                    for event in events:
                        event["episode"] = int(self.episodes[event["pair"]-1])
                        stream.write(json.dumps(event) + "\n")
        for label, observations, choices in (("learner", self.select(self.full), actions),
                                             ("opponent", other, chosen)):
            for index, action in enumerate(choices):
                obs = {key: value[index] for key, value in observations.items()}
                switch = 4 <= int(action) <= 9 and bool(obs["action_mask"][int(action)])
                if normal_decision(obs):
                    self.switch_counts[f"{label}_normal_decisions"] += 1
                    self.switch_counts[f"{label}_voluntary_switches"] += int(switch)
                elif obs["context"][1]:
                    self.switch_counts[f"{label}_forced_switches"] += int(switch)
        paired_actions = np.empty(self.paired.num_envs, dtype=np.int64)
        paired_actions[::2], paired_actions[1::2] = actions, chosen
        self.paired.step_async(paired_actions)

    def step_wait(self):
        self.full, rewards, dones, infos = self.paired.step_wait()
        self.episodes += np.asarray(dones[1::2], dtype=int)
        if hasattr(self.opponent, "reset_episodes"):
            ended = np.flatnonzero(dones[1::2])
            self.opponent.finish_episodes(ended)
            self.opponent.reset_episodes(ended)
        if self.switching is not None:
            self.switching.reset(np.flatnonzero(dones[1::2]))
        self.reset_infos = self.paired.reset_infos[::2]
        # Opponent actions/returns NEVER enter the learning policy's rollout.
        learner_infos = []
        for info in infos[::2]:
            info = dict(info)
            if "terminal_observation" in info:
                info["terminal_observation"] = policy_for_version(
                    info["terminal_observation"], self.observation_version)
            learner_infos.append(info)
        return (policy_for_version(self.select(self.full), self.observation_version),
                rewards[::2], dones[::2], learner_infos)

    def switching_metrics(self):
        result = self.switch_counts.copy()
        for side in ("learner", "opponent"):
            result[f"{side}_voluntary_switch_rate"] = (
                result[f"{side}_voluntary_switches"] / max(1, result[f"{side}_normal_decisions"]))
        return result

    def get_attr(self, attr_name, indices=None):
        if attr_name in ("observation_space", "observation_version"):
            return [getattr(self, attr_name) for _ in self._get_indices(indices)]
        return self.paired.get_attr(attr_name, [2*i for i in self._get_indices(indices)])

    def set_attr(self, attr_name, value, indices=None):
        return self.paired.set_attr(attr_name, value, [2*i for i in self._get_indices(indices)])

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return self.paired.env_method(method_name, *args, indices=[2*i for i in self._get_indices(indices)], **kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.paired.env_is_wrapped(wrapper_class, [2*i for i in self._get_indices(indices)])

    def close(self):
        self.paired.close()
