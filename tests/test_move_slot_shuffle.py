"""Move permutations must preserve executed moves, legality and PPO bootstrapping."""
from copy import deepcopy
from itertools import permutations
import json
import sys

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.vec_env import VecEnv

from firered_rl.move_slot_shuffle import MoveSlotShuffle
from firered_rl.ppo_features import policy_space
from firered_rl.frozen_opponent import FrozenOpponentVecEnv
from test_observation_v2_training import (
    cli, teams, command, real_vec, small_model, requires_showdown, MaskablePPO,
)
from test_periodic_evaluation import suite_file


class FixtureVec(VecEnv):
    def __init__(self, version="v2"):
        space = policy_space(version)
        self.raw = {k: np.zeros((2, *s.shape), dtype=s.dtype) for k, s in space.spaces.items()}
        self.raw["active"][:, 0, 8] = 1
        self.raw["moves"][:] = np.arange(1, 97).reshape(6, 4, 4)
        self.raw["opponent_moves"][:] = 23
        self.raw["action_mask"][:] = 1
        if version == "v2":
            self.raw["v2_move_details"][:, :, :, :, 0] = np.arange(1, 49).reshape(2, 6, 4)
        self.next = None
        self.actions = []
        self.closed = False
        super().__init__(2, space, gym.spaces.Discrete(11))

    def reset(self):
        self._reset_seeds()
        return deepcopy(self.raw)

    def step_async(self, actions):
        assert self.raw["action_mask"][np.arange(2), actions].all()
        self.actions.append(actions.copy())

    def step_wait(self):
        if self.next is not None:
            obs, dones, infos = self.next
            self.next = None
            self.raw = deepcopy(obs)
            return deepcopy(obs), np.array([7., -4.]), dones, infos
        return deepcopy(self.raw), np.zeros(2), np.zeros(2, dtype=bool), [{}, {}]

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in self._get_indices(indices)]
        if attr_name == "action_masks":
            return [lambda i=i: self.raw["action_mask"][i].astype(bool).copy()
                    for i in self._get_indices(indices)]
        raise AttributeError(attr_name)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [callback() for callback in self.get_attr(method_name, indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        self.closed = True


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_all_orders_sparse_masks_move_features_and_executed_identity(version):
    native = FixtureVec(version)
    vec = MoveSlotShuffle(native, seed=17)
    try:
        vec.reset()
        # Covers one/two/three/four moves, arbitrary empty/disabled positions,
        # as well as forced continuation/Struggle's single legal slot.
        for size in range(1, 5):
            for offset in range(4):
                present = np.roll(np.arange(4), offset)[:size]
                native.raw["moves"][:, :, :, 0] = 0
                native.raw["moves"][:, :, present, 0] = present + 1
                native.raw["action_mask"][:, :4] = 0
                native.raw["action_mask"][:, present] = 1
                for order in permutations(range(4)):
                    vec._orders[:] = order
                    obs = vec._present(native.raw)
                    assert is_masking_supported(vec)
                    masks = get_action_masks(vec)
                    assert np.array_equal(masks, obs["action_mask"])
                    assert np.array_equal(obs["moves"], native.raw["moves"][:, :, order, :])
                    if version == "v2":
                        assert np.array_equal(obs["v2_move_details"][:, 0],
                                              native.raw["v2_move_details"][:, 0][:, :, order, :])
                    unchanged = set(obs) - {"moves", "action_mask", "v2_move_details"}
                    assert all(np.array_equal(obs[k], native.raw[k]) for k in unchanged)
                    assert np.array_equal(obs["action_mask"][:, 4:], native.raw["action_mask"][:, 4:])
                    for chosen in np.flatnonzero(masks[0]):
                        vec.step(np.full(2, chosen))
                        assert native.actions[-1].tolist() == [order[chosen] if chosen < 4 else chosen] * 2
                    # Invalid actions are rejected before either player advances.
                    unavailable = np.flatnonzero(~masks[0])
                    if len(unavailable):
                        count = len(native.actions)
                        with pytest.raises(ValueError, match="unavailable"):
                            vec.step_async(np.array([int(np.flatnonzero(masks[0])[0]), unavailable[0]]))
                        assert len(native.actions) == count
        obs["action_mask"][:] = 0
        assert get_action_masks(vec).any()
    finally:
        vec.close()
    assert native.closed


def test_episode_stability_switching_terminal_mapping_and_independent_reset_streams():
    native = FixtureVec()
    vec = MoveSlotShuffle(native, seed=17)
    other = MoveSlotShuffle(FixtureVec(), seed=17)
    try:
        vec.reset()
        other.reset()
        assert np.array_equal(vec._orders, other._orders)
        old = vec._orders.copy()
        # Switching changes which party member supplies the action mapping.
        native.raw["active"][0, 0, 8] = 3
        obs, _, _, _ = vec.step(np.array([5, 10]))
        assert np.array_equal(old, vec._orders)
        chosen = np.array([0, 2])
        vec.step(chosen)
        assert native.actions[-1].tolist() == [old[0, 2, 0], old[1, 0, 2]]
        terminal = {k: v[0].copy() for k, v in native.raw.items()}
        terminal["context"][0] = 200
        reset = deepcopy(native.raw)
        reset["active"][0, 0, 8] = 1
        native.next = (reset, np.array([True, False]),
                       [{"terminal_observation": terminal, "TimeLimit.truncated": True}, {}])
        obs, rewards, dones, infos = vec.step(np.array([0, 0]))
        assert rewards.tolist() == [7., -4.] and dones.tolist() == [True, False]
        final = infos[0]["terminal_observation"]
        assert final["context"][0] == 200 and infos[0]["TimeLimit.truncated"]
        for party in range(6):
            assert np.array_equal(final["moves"][party], terminal["moves"][party, old[0, party]])
            assert np.array_equal(final["v2_move_details"][0, party],
                                  terminal["v2_move_details"][0, party, old[0, party]])
        assert not np.array_equal(old[0], vec._orders[0])
        assert np.array_equal(old[1], vec._orders[1])
        assert infos[0]["move_slot_shuffle"]["simulator_action"] == old[0, 2, 0]
        # Another player's reset cannot consume this player's random stream.
        other._new_orders([0])
        assert np.array_equal(vec._orders, other._orders)
        vec.seed(17)
        obs = vec.reset()
        assert np.array_equal(vec._orders, old)
        assert not np.array_equal(old[0], old[1])
    finally:
        vec.close()
        other.close()


@requires_showdown
def test_real_ppo_frozen_input_unchanged_truncation_bootstrap_and_reload(tmp_path, teams, monkeypatch):
    torch.set_num_threads(1)
    paired = real_vec(tmp_path / "pair", teams, "v2")
    opponent = small_model(paired)
    before = deepcopy(opponent.policy.state_dict())
    frozen = FrozenOpponentVecEnv(paired, opponent)
    vec = MoveSlotShuffle(frozen, seed=7)
    predicted = []
    original_predict = opponent.predict
    def predict(observations, **kwargs):
        expected = FrozenOpponentVecEnv.select(frozen.full, 1)
        assert all(np.array_equal(value, expected[key]) for key, value in observations.items())
        predicted.append(True)
        return original_predict(observations, **kwargs)
    monkeypatch.setattr(opponent, "predict", predict)
    try:
        learner = small_model(vec)
        # Verify PPO's *actual* truncation value path sees the old move mapping.
        finals = []
        original_wait = vec.step_wait
        def wait():
            result = original_wait()
            finals.append(deepcopy(result[3][0]["terminal_observation"]))
            return result
        monkeypatch.setattr(vec, "step_wait", wait)
        terminal_value_calls = []
        def values(obs):
            if finals and float(obs["context"][0, 0]) == float(finals[-1]["context"][0]):
                assert np.array_equal(obs["moves"][0].cpu().numpy(), finals[-1]["moves"])
                terminal_value_calls.append(True)
            return torch.full((len(obs["party"]), 1), .5)
        with monkeypatch.context() as patch:
            patch.setattr(learner.policy, "predict_values", values)
            learner.learn(4)
        assert predicted and len(terminal_value_calls) == 4
        assert all(torch.equal(value, opponent.policy.state_dict()[k]) for k, value in before.items())
        assert learner.rollout_buffer.observations["moves"].shape[1:] == (6, 4, 4)
        checkpoint = tmp_path / "model.zip"
        learner.save(checkpoint)
        loaded = MaskablePPO.load(checkpoint, device="cpu")
        # Checkpoints still act directly on unshuffled observations at evaluation.
        raw = paired.reset()
        actions, _ = loaded.predict(raw, action_masks=raw["action_mask"].astype(bool), deterministic=True)
        assert all(raw["action_mask"][i, a] for i, a in enumerate(actions))
    finally:
        vec.close()


@requires_showdown
def test_actual_move_announcements_follow_presented_slots(tmp_path, teams):
    moves = ["Splash", "Harden", "Growth", "Leer"]
    for path in teams:
        path.write_text(json.dumps({"party": [{"species": "Magikarp", "level": 50, "moves": moves}]}))
    paired = real_vec(tmp_path / "execution", teams, "v2")
    paired.sessions[0].max_steps = 20
    paired.sessions[0].factory.args.move_order_audit = True
    vec = MoveSlotShuffle(paired, seed=29)
    try:
        vec.reset()
        for slot in range(4):
            orders = vec._action_orders.copy()
            _, _, _, infos = vec.step(np.full(2, slot))
            for player in range(2):
                expected = moves[orders[player, slot]]
                emitted = [line.split("|")[3] for line in infos[player]["audit_protocol"]
                           if line.startswith(f"|move|p{player+1}a:")]
                assert emitted and emitted[0] == expected
    finally:
        vec.close()


@requires_showdown
def test_cli_continue_restores_option_and_preserves_switch_metrics(tmp_path, monkeypatch, cli, teams, capsys):
    run = tmp_path / "run"
    suite = suite_file(tmp_path, teams)
    monkeypatch.setattr(sys, "argv", command(run, teams, "--observation-version", "v2",
        "--shuffle-move-slots", "--rule-based-opponent", "--showdown-workers", "process",
        "--rollout-steps", "4", "--batch-size", "4", "--epochs", "1",
        "--eval-suite", str(suite), "--eval-every", "4"))
    assert cli.main() == 0
    config = json.loads((run / "config.json").read_text())
    assert config["shuffle_move_slots"] and config["move_slot_shuffle"]["scope"] == "learners_only"
    assert (run / "switching.json").is_file()
    assert (run / "checkpoints/best.zip").is_file()
    evaluations = json.loads((run / "evaluation-history.json").read_text())["records"]
    assert len(evaluations) == 3 and all(r["status"] == "completed" for r in evaluations)
    continuation = ["train_ppo", "--output-dir", str(run), "--if-exists", "continue", "--total-timesteps", "4"]
    monkeypatch.setattr(sys, "argv", continuation)
    assert cli.main() == 0
    config = json.loads((run / "config.json").read_text())
    assert config["shuffle_move_slots"] and config["session"]["id"] == 2
    assert config["effective_ppo"]["starting_timesteps"] == 8
    before = (run / "config.json").read_bytes()
    monkeypatch.setattr(sys, "argv", [*continuation, "--no-shuffle-move-slots"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and (run / "config.json").read_bytes() == before
    assert "differs from this run" in capsys.readouterr().err


def test_mgba_shuffle_rejected_before_launch(tmp_path, monkeypatch, cli, capsys):
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--shuffle-move-slots", "--output-dir", str(tmp_path / "run")])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not (tmp_path / "run").exists()
    assert "requires --backend showdown" in capsys.readouterr().err
