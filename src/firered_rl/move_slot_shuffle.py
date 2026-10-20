"""Training-only move presentation; the simulator and frozen opponents keep their order."""
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


def reorder_own_moves(observations, orders):
    """Copy a policy batch, applying new-slot -> simulator-slot permutations.

    ``orders`` is (players, 6 party members, 4 moves). Every move feature and
    the active move mask travel together. Other actions and opponent data do not.
    """
    result = {key: value.copy() for key, value in observations.items()}
    result["moves"] = np.take_along_axis(observations["moves"], orders[..., None], axis=2)
    if "v2_move_details" in result:
        result["v2_move_details"][:, 0] = np.take_along_axis(
            observations["v2_move_details"][:, 0], orders[..., None], axis=2)
    slots = observations["active"][:, 0, 8] - 1
    valid = (slots >= 0) & (slots < 6) & (slots == np.floor(slots))
    if np.any(~valid & observations["action_mask"][:, :4].any(axis=1)):
        raise ValueError("move-slot shuffle requires an active party slot when moves are legal")
    active_orders = orders[np.arange(len(orders)), np.where(valid, slots, 0).astype(int)]
    result["action_mask"][:, :4] = np.take_along_axis(
        observations["action_mask"][:, :4], active_orders, axis=1)
    return result, active_orders


class MoveSlotShuffle(VecEnvWrapper):
    """Give each learner independent, stable per-party permutations each episode.

    Wrap *outside* FrozenOpponentVecEnv so its policy, rules and switching logic
    still receive unmodified observations. SB3 stores presented observations,
    actions and masks in the rollout; only simulator-bound actions are mapped.
    """
    VERSION = 1

    def __init__(self, venv, seed=1):
        self.shuffle_seed = int(seed)
        self._pending_seed = self.shuffle_seed
        self._rngs = None
        self._orders = np.tile(np.arange(4), (venv.num_envs, 6, 1))
        self._action_orders = None
        self._masks = None
        self._pending_actions = None
        super().__init__(venv)

    def seed(self, seed=None):
        seeds = self.venv.seed(seed)
        self._pending_seed = int(seeds[0])
        self.shuffle_seed = self._pending_seed
        return seeds

    def _new_orders(self, indices):
        for index in indices:
            self._orders[index] = self._rngs[index].permuted(
                np.tile(np.arange(4), (6, 1)), axis=1)

    def _present(self, observations):
        result, self._action_orders = reorder_own_moves(observations, self._orders)
        self._masks = result["action_mask"].astype(bool, copy=True)
        self.reset_infos = self.venv.reset_infos
        return result

    def reset(self):
        observations = self.venv.reset()
        if self._pending_seed is not None:
            # Separate generators avoid consuming battle/team/policy RNG draws.
            self._rngs = [np.random.default_rng(np.random.SeedSequence(
                [self._pending_seed, 0x53485546, index])) for index in range(self.num_envs)]
            self._pending_seed = None
        self._pending_actions = None
        self._new_orders(range(self.num_envs))
        return self._present(observations)

    def step_async(self, actions):
        if self._masks is None:
            raise RuntimeError("reset before stepping")
        if self._pending_actions is not None:
            raise RuntimeError("wait for the pending move-slot step")
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("one integer action is required per learning player")
        if np.any((actions < 0) | (actions >= self.action_space.n)):
            raise ValueError("move-slot action is out of range")
        if not self._masks[np.arange(self.num_envs), actions].all():
            raise ValueError("move-slot action is unavailable in the presented mask")
        mapped = actions.copy()
        moving = np.flatnonzero(actions < 4)
        mapped[moving] = self._action_orders[moving, actions[moving]]
        details = [dict(learner_action=int(action), simulator_action=int(native),
                        new_to_old_slots=(order + 1).tolist())
                   for action, native, order in zip(actions, mapped, self._action_orders)]
        self.venv.step_async(mapped)
        self._pending_actions = details

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("no pending move-slot step")
        observations, rewards, dones, infos = self.venv.step_wait()
        result_infos = []
        for index, original in enumerate(infos):
            info = dict(original)
            info["move_slot_shuffle"] = self._pending_actions[index]
            if "terminal_observation" in info:
                # Time-limit value bootstrapping must use the ending episode's
                # mapping, before the autoreset observation gets a new one.
                batch = {key: value[None] for key, value in info["terminal_observation"].items()}
                terminal, _ = reorder_own_moves(batch, self._orders[index:index+1])
                info["terminal_observation"] = {key: value[0] for key, value in terminal.items()}
            result_infos.append(info)
        self._pending_actions = None
        self._new_orders(np.flatnonzero(dones))
        return self._present(observations), rewards, dones, result_infos

    def get_attr(self, attr_name, indices=None):
        if attr_name == "action_masks":
            return [lambda i=i: self.env_method("action_masks", indices=i)[0]
                    for i in self._get_indices(indices)]
        return self.venv.get_attr(attr_name, indices)

    def has_attr(self, attr_name):
        return attr_name == "action_masks" or self.venv.has_attr(attr_name)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name == "action_masks":
            if self._masks is None:
                raise RuntimeError("reset before querying action masks")
            return [self._masks[i].copy() for i in self._get_indices(indices)]
        return self.venv.env_method(method_name, *args, indices=indices, **kwargs)
