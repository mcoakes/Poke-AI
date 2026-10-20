"""Move identity must survive permutations before/after learning and serialization."""
from copy import deepcopy
from itertools import permutations
import json
import sys

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sb3_contrib')
from firered_rl.move_order_audit import permute_moves
from firered_rl.ppo_features import policy_space
from firered_rl.shared_move_policy import SharedMovePolicy, policy_architecture
from test_observation_v2_training import (cli, teams, command, requires_showdown,
                                         MaskablePPO, real_vec, small_model)
from firered_rl.frozen_opponent import FrozenOpponentVecEnv
from test_periodic_evaluation import suite_file


def example(count=4, active_slot=1):
    obs = {key: np.zeros(space.shape, dtype=space.dtype)
           for key, space in policy_space('v2').spaces.items()}
    for slot in range(6):
        obs['party'][slot] = [1, 6+slot, 50, 100, 150, 10, 2, 0, 0, 0]
        obs['opponent_party'][slot] = [1, 9+slot]
        for side, name in enumerate(('moves', 'opponent_moves')):
            obs['v2_party'][side, slot] = [1, 50, 100, 150, 10, 2, 0, 1, 0, 0]
            obs['v2_stats'][side, slot] = [100, 90, 80, 70, 60]
            for index in range(count):
                # Different numbers in every field catch partial permutation bugs.
                obs[name][slot, index] = [50 + index + slot * 4, 10 + index, 4 + index, 10 + index]
                obs['v2_move_details'][side, slot, index] = [50 + 10*index, 70 + index*10,
                                                           0, index % 3, index - 2, 10 + index]
    obs['active'][0] = [1, 6, 50, 100, 150, 10, 2, 0, active_slot, 0]
    obs['active'][1] = [1, 9, 50, 67, 100, 11, 11, 0, 0, 0]
    obs['v2_active'][:] = [100, 150, 0, 1]
    obs['v2_active_stats'][:] = [100, 90, 80, 70, 60]
    obs['opponent_active_index'][:] = 2
    obs['action_mask'][:count] = 1
    if count > 1:
        obs['action_mask'][count-1] = 0  # Present but disabled, not only empty padding.
    obs['action_mask'][4:10] = 1
    obs['action_mask'][active_slot+3] = 0
    return obs


def tensors(observations):
    return {key: torch.as_tensor(np.stack([obs[key] for obs in observations])) for key in observations[0]}


def outputs(policy, observations):
    batch = tensors(observations)
    with torch.no_grad():
        raw = policy.action_net(policy.extract_features(batch))
        probs = policy.get_distribution(batch, action_masks=batch['action_mask'].bool()).distribution.probs.clone()
        values = policy.predict_values(batch)
    return raw.numpy(), probs.numpy(), values.numpy()


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    return SharedMovePolicy(policy_space('v2'), gym.spaces.Discrete(11), lambda _: 1e-4)


@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_all_24_orders_preserve_move_scores_other_actions_and_values(policy, count):
    for active_slot in range(1, 7):
        obs = example(count, active_slot)
        before = deepcopy(obs)
        orders = list(permutations(range(4)))
        variants = [permute_moves(obs, order) for order in orders]
        raw, probs, values = outputs(policy, [obs, *variants])
        assert np.ptp(raw[0, :count]) > 1e-5 or count == 1
        for i, order in enumerate(orders, start=1):
            remap = list(order) + list(range(4, 11))
            np.testing.assert_allclose(raw[i], raw[0, remap], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(probs[i], probs[0, remap], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(values[i], values[0], atol=1e-6, rtol=1e-5)
            assert not probs[i][variants[i-1]['action_mask'] == 0].any()
        assert all(np.array_equal(obs[key], before[key]) for key in obs)


def test_all_party_moves_pool_symmetrically_including_duplicates(policy):
    obs = example(active_slot=3)
    obs['moves'][2, :, 0] = 53  # Same IDs with different PP/details must still commute.
    orders = np.random.default_rng(8).permuted(np.tile(np.arange(4), (2, 6, 1)), axis=2)
    changed = deepcopy(obs)
    for side, name in enumerate(('moves', 'opponent_moves')):
        changed[name] = np.take_along_axis(obs[name], orders[side, :, :, None], axis=1)
        changed['v2_move_details'][side] = np.take_along_axis(
            obs['v2_move_details'][side], orders[side, :, :, None], axis=1)
    changed['action_mask'][:4] = obs['action_mask'][orders[0, 2]]
    raw, probs, values = outputs(policy, [obs, changed])
    remap = orders[0, 2].tolist() + list(range(4, 11))
    np.testing.assert_allclose(raw[1], raw[0, remap], atol=1e-6)
    np.testing.assert_allclose(probs[1], probs[0, remap], atol=1e-6)
    np.testing.assert_allclose(values[1], values[0], atol=1e-6)


def test_forced_actions_terminal_values_and_policy_serialization(policy, tmp_path):
    observations = []
    for action in (0, 3, 5, 10):
        obs = example()
        obs['action_mask'][:] = 0
        obs['action_mask'][action] = 1
        observations.append(obs)
    _, probs, values = outputs(policy, observations)
    assert np.argmax(probs, axis=1).tolist() == [0, 3, 5, 10]
    assert np.isfinite(values).all()
    terminal = deepcopy(observations[0])
    terminal['action_mask'][:] = 0
    terminal['active'][:] = 0
    terminal['context'][2] = 1
    assert torch.isfinite(policy.predict_values(tensors([terminal]))).all()
    path = tmp_path/'policy.pt'
    policy.save(path)
    loaded = SharedMovePolicy.load(path, device='cpu')
    for a, b in zip(outputs(policy, observations), outputs(loaded, observations)):
        np.testing.assert_array_equal(a, b)


def test_rejects_incompatible_policy_contracts():
    with pytest.raises(ValueError, match='v2'):
        SharedMovePolicy(policy_space('v1'), gym.spaces.Discrete(11), lambda _: 1e-4)
    with pytest.raises(ValueError, match='Discrete'):
        SharedMovePolicy(policy_space('v2'), gym.spaces.Discrete(4), lambda _: 1e-4)


def test_policy_loss_updates_shared_scorer_and_preserves_permutation(policy):
    obs = example(active_slot=4)
    batch = tensors([obs])
    before = deepcopy(policy.state_dict())
    registered = {id(p) for group in policy.optimizer.param_groups for p in group['params']}
    assert registered == {id(p) for p in policy.parameters()}
    values, log_prob, _ = policy.evaluate_actions(batch, torch.tensor([0]),
                                                 action_masks=batch['action_mask'].bool())
    policy.optimizer.zero_grad()
    (-log_prob.mean() + .1 * values.square().mean()).backward()
    for parameter in (policy.action_net.moves[-1].weight,
                      policy.features_extractor.move_encoder[0].weight,
                      policy.features_extractor.context[0].weight):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    policy.optimizer.step()
    assert not torch.equal(before['action_net.moves.4.weight'], policy.action_net.moves[-1].weight)
    order = [3, 1, 0, 2]
    raw, probs, values = outputs(policy, [obs, permute_moves(obs, order)])
    remap = order + list(range(4, 11))
    np.testing.assert_allclose(raw[1], raw[0, remap], atol=1e-6)
    np.testing.assert_allclose(probs[1], probs[0, remap], atol=1e-6)
    np.testing.assert_allclose(values[1], values[0], atol=1e-6)


@requires_showdown
@pytest.mark.parametrize('learner_architecture', ['flat', 'shared-moves'])
def test_frozen_opponent_accepts_other_architecture_without_changing_weights(
        tmp_path, teams, learner_architecture):
    torch.set_num_threads(1)
    def model(vec, architecture):
        return (MaskablePPO(SharedMovePolicy, vec, n_steps=4, batch_size=4,
                           n_epochs=1, seed=7, device='cpu')
                if architecture == 'shared-moves' else small_model(vec))
    source = real_vec(tmp_path/'source', teams, 'v2')
    checkpoint = tmp_path/'opponent.zip'
    try:
        opponent_architecture = 'flat' if learner_architecture == 'shared-moves' else 'shared-moves'
        model(source, opponent_architecture).save(checkpoint)
    finally:
        source.close()
    opponent = MaskablePPO.load(checkpoint, device='cpu')
    before = deepcopy(opponent.policy.state_dict())
    original_bytes = checkpoint.read_bytes()
    paired = real_vec(tmp_path/'paired', teams, 'v2')
    vec = FrozenOpponentVecEnv(paired, opponent, observation_version='v2')
    try:
        learner = model(vec, learner_architecture)
        learner.learn(4)
        assert learner.num_timesteps == 4 and learner.policy.optimizer.state
        assert opponent.num_timesteps == 0
        assert all(torch.equal(value, opponent.policy.state_dict()[key]) for key, value in before.items())
        assert checkpoint.read_bytes() == original_bytes
    finally:
        vec.close()


@requires_showdown
def test_real_cli_training_validation_and_continuation(tmp_path, monkeypatch, cli, teams, capsys):
    suite = suite_file(tmp_path, teams)
    run = tmp_path/'shared'
    monkeypatch.setattr(sys, 'argv', command(run, teams, '--policy-architecture', 'shared-moves',
        '--rule-based-opponent', '--showdown-workers', 'process', '--rollout-steps', '4',
        '--batch-size', '4', '--epochs', '1', '--eval-suite', str(suite), '--eval-every', '4'))
    assert cli.main() == 0
    config = json.loads((run/'config.json').read_text())
    assert config['policy_architecture'] == 'shared-moves' and config['observation_version'] == 'v2'
    assert config['policy_architecture_version'] == 1 and config['policy_parameters'] > 0
    history = json.loads((run/'evaluation-history.json').read_text())['records']
    assert len(history) == 3 and all(row['status'] == 'completed' for row in history)
    saved = MaskablePPO.load(run/'checkpoints/final.zip', device='cpu')
    assert policy_architecture(saved) == 'shared-moves'
    assert saved.num_timesteps == 8 and saved.policy.optimizer.state
    weights = deepcopy(saved.policy.state_dict())

    monkeypatch.setattr(sys, 'argv', ['train_ppo', '--output-dir', str(run), '--if-exists', 'continue',
                                    '--total-timesteps', '4', '--eval-every', '0'])
    assert cli.main() == 0
    loaded = MaskablePPO.load(run/'checkpoints/final.zip', device='cpu')
    assert loaded.num_timesteps == 12 and loaded.policy.optimizer.state
    assert any(not torch.equal(weights[k], v) for k, v in loaded.policy.state_dict().items())
    assert all(torch.isfinite(v).all() for v in loaded.policy.state_dict().values())
    obs = example(active_slot=5)
    order = [2, 0, 3, 1]
    raw, probs, values = outputs(loaded.policy, [obs, permute_moves(obs, order)])
    np.testing.assert_allclose(probs[1], probs[0, order + list(range(4, 11))], atol=1e-6)
    np.testing.assert_allclose(values[1], values[0], atol=1e-6)

    before = (run/'config.json').read_bytes()
    monkeypatch.setattr(sys, 'argv', ['train_ppo', '--output-dir', str(run), '--if-exists', 'continue',
                                    '--policy-architecture', 'flat'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and (run/'config.json').read_bytes() == before
    assert 'differs from this run' in capsys.readouterr().err

    # New directories must not silently convert incompatible weights either.
    for arch in ('flat', 'shared-moves'):
        if arch == 'flat':
            checkpoint = run/'checkpoints/final.zip'
        else:
            old = tmp_path/'flat'
            monkeypatch.setattr(sys, 'argv', command(old, teams, '--observation-version', 'v2',
                '--rollout-steps', '4', '--batch-size', '4', '--epochs', '1'))
            assert cli.main() == 0
            checkpoint = old/'checkpoints/final.zip'
        rejected = tmp_path/f'rejected-{arch}'
        monkeypatch.setattr(sys, 'argv', command(rejected, teams, '--resume', str(checkpoint),
                                                '--policy-architecture', arch))
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2 and not rejected.exists()
        assert 'weights are not converted' in capsys.readouterr().err


def test_mgba_shared_moves_still_requires_rom(tmp_path, monkeypatch, cli, capsys):
    run = tmp_path/'unsupported'
    monkeypatch.setattr(sys, 'argv', ['train_ppo', '--policy-architecture', 'shared-moves',
                                    '--output-dir', str(run)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not run.exists()
    assert 'requires --rom' in capsys.readouterr().err
