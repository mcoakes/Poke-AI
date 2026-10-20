"""Feature ablation keeps initial policies and all game interfaces controlled."""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
from firered_rl.matchup_move_policy import (
    MatchupMovePolicy, MatchupRelations, create_ablation_checkpoint,
)
from firered_rl.move_order_audit import permute_moves
from firered_rl.ppo_features import policy_space
from firered_rl.shared_move_policy import SharedMovePolicy
from test_shared_move_policy import example, outputs, tensors


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def matchup(move_id=89, kind=4, target=(13, 13), ability=0, own=(4, 4)):
    obs = example()
    slot = int(obs['active'][0, 8]) - 1
    obs['active'][0, 5:7] = own
    obs['active'][1, 5:7] = target
    obs['v2_active'][1, 3] = ability
    obs['moves'][slot] = 0
    obs['v2_move_details'][0, slot] = 0
    for i, (ident, move_kind, power) in enumerate(((move_id, kind, 100), (33, 0, 35))):
        obs['moves'][slot, i] = [ident, move_kind, 10, 10]
        obs['v2_move_details'][0, slot, i] = [power, 100, 0, 0, 0, move_kind]
    obs['action_mask'][:4] = [1, 1, 0, 0]
    return obs


class ContractEnv(gym.Env):
    observation_space = policy_space('v2')
    action_space = gym.spaces.Discrete(11)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return matchup(), {}

    def step(self, action):
        return matchup(), float(action == 1), True, False, {}

    def action_masks(self):
        return matchup()['action_mask'].astype(bool)


@pytest.fixture
def template():
    return MaskablePPO(SharedMovePolicy, ContractEnv(), seed=37, device='cpu',
        n_steps=4, batch_size=4, n_epochs=1, learning_rate=1e-4)


@pytest.mark.parametrize('kind,target,ability,pure,effective', [
    (4, (13, 13), 0, 2., 2.),  # mono types count once
    (4, (5, 8), 0, 4., 4.),
    (4, (2, 0), 0, 0., 0.),
    (4, (13, 13), 26, 2., 0.),
    (13, (11, 11), 10, 2., 0.),
    (11, (10, 10), 11, 2., 0.),
    (10, (12, 12), 18, 2., 0.),
    (0, (0, 0), 25, 1., 0.),
    (10, (12, 12), 25, 2., 2.),
])
def test_matchup_values(kind, target, ability, pure, effective):
    obs = matchup(kind=kind, target=target, ability=ability, own=(kind, kind))
    row = MatchupRelations()(tensors([obs]))[0, 0].tolist()
    assert row == [1., pure / 4, effective / 4, float(effective == 0), 1.]


def test_sound_foresight_frozen_and_effective_hidden_power_type():
    sound = matchup(move_id=304, kind=0, target=(0, 0), ability=43)
    ghost = matchup(move_id=33, kind=0, target=(7, 3), own=(0, 0))
    ghost['v2_volatile_bits'][1, 29] = 1
    fire = matchup(move_id=53, kind=10, target=(12, 12), ability=18)
    fire['status_bits'][7, 5] = 1
    hidden = matchup(move_id=237, kind=11, target=(10, 10), own=(11, 11))
    hidden['moves'][0, 0, 1] = 0  # base Normal; effective v2 type is Water
    rows = MatchupRelations()(tensors([sound, ghost, fire, hidden]))[:, 0]
    assert rows[:, 2].tolist() == [0., .25, .5, .5]
    assert rows[1, 1].item() == 0  # pure chart remains Ghost-immune
    assert rows[3, 4].item() == 1


def test_unassessed_padding_dead_target_status_and_special_moves():
    examples = []
    for ident in (0, 165, 248, 251, 267, 311, 353, 500):
        examples.append(matchup(move_id=ident))
    status = matchup()
    status['v2_move_details'][0, 0, 0, 3] = 2
    examples.append(status)
    pp = matchup()
    pp['moves'][0, 0, 2] = 0
    examples.append(pp)
    no_power = matchup()
    no_power['v2_move_details'][0, 0, 0, 0] = 1
    examples.append(no_power)
    dead = matchup()
    dead['v2_active'][1, 0] = 0
    examples.append(dead)
    terminal = matchup()
    terminal['context'][2] = 1
    examples.append(terminal)
    result = MatchupRelations()(tensors(examples))
    assert torch.equal(result[:, 0], torch.zeros(len(examples), 5))
    assert torch.equal(result[:, 2:], torch.zeros(len(examples), 2, 5))


def test_ablation_starts_identical_preserves_template_and_settings(template, tmp_path):
    original = deepcopy(template.policy.state_dict())
    arms = [create_ablation_checkpoint(template, tmp_path / f'{enabled}.zip',
            enabled=enabled, seed=83, training_settings={'ent_coef': .01})
            for enabled in (False, True)]
    off, on = arms
    assert off.num_timesteps == on.num_timesteps == 0
    assert not off.policy.optimizer.state and not on.policy.optimizer.state
    assert off.ent_coef == on.ent_coef == .01
    assert all(torch.equal(v, on.policy.state_dict()[k]) for k, v in off.policy.state_dict().items())
    assert all(torch.equal(v, template.policy.state_dict()[k]) for k, v in original.items())
    assert sum(p.numel() for p in off.policy.parameters()) == sum(p.numel() for p in on.policy.parameters())
    observations = [matchup(), matchup(target=(2, 0)), matchup(ability=26)]
    source = outputs(template.policy, observations)
    for arm in arms:
        for old, changed in zip(source, outputs(arm.policy, observations)):
            np.testing.assert_allclose(old, changed, atol=1e-7, rtol=1e-6)
    for a, b in zip(outputs(off.policy, observations), outputs(on.policy, observations)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('enabled', [False, True])
def test_feature_gradients_and_legal_mask_permutation(template, tmp_path, enabled):
    model = create_ablation_checkpoint(template, tmp_path / 'model.zip', enabled=enabled, seed=83)
    policy = model.policy
    obs = matchup(target=(2, 0))
    before = deepcopy(obs)
    batch = tensors([obs])
    value, logp, _ = policy.evaluate_actions(batch, torch.tensor([1]),
                                            action_masks=batch['action_mask'].bool())
    policy.optimizer.zero_grad()
    (-logp.mean() + value.square().mean()).backward()
    features = policy.features_extractor
    new_gradient = policy.action_net.moves[0].weight.grad[:, features.context_dim + features.base_move_dim:]
    context_gradient = features.relation_context.weight.grad
    assert bool(new_gradient.abs().sum() > 0) is enabled
    assert bool(context_gradient.abs().sum() > 0) is enabled
    policy.optimizer.step()
    for order in ([1, 0, 3, 2], [3, 2, 1, 0]):
        raw, probs, values = outputs(policy, [obs, permute_moves(obs, order)])
        remap = order + list(range(4, 11))
        np.testing.assert_allclose(raw[1], raw[0, remap], atol=1e-6)
        np.testing.assert_allclose(probs[1], probs[0, remap], atol=1e-6)
        np.testing.assert_allclose(values[1], values[0], atol=1e-6)
    assert not outputs(policy, [obs])[1][0][obs['action_mask'] == 0].any()
    assert all(np.array_equal(obs[key], before[key]) for key in obs)


@pytest.mark.parametrize('enabled', [False, True])
def test_model_and_policy_roundtrip_preserve_feature_toggle(template, tmp_path, enabled):
    path = tmp_path / 'model.zip'
    model = create_ablation_checkpoint(template, path, enabled=enabled, seed=83)
    loaded = MaskablePPO.load(path, device='cpu')
    policy_path = tmp_path / 'policy.pt'
    model.policy.save(policy_path)
    policy = MatchupMovePolicy.load(policy_path, device='cpu')
    assert loaded.policy.features_extractor.relational_enabled is enabled
    assert policy.features_extractor.relational_enabled is enabled
    for roundtrip in (loaded.policy, policy):
        for old, changed in zip(outputs(model.policy, [matchup()]), outputs(roundtrip, [matchup()])):
            np.testing.assert_array_equal(old, changed)
    loaded.set_env(ContractEnv())
    loaded.learn(4)
    assert loaded.num_timesteps == 4 and loaded.policy.optimizer.state


def test_migration_refuses_trained_template_invalid_toggle_or_overwrite(template, tmp_path):
    with pytest.raises(ValueError, match='boolean'):
        create_ablation_checkpoint(template, tmp_path / 'bad.zip', enabled=1, seed=83)
    create_ablation_checkpoint(template, tmp_path / 'existing.zip', enabled=True, seed=83)
    with pytest.raises(ValueError, match='already exists'):
        create_ablation_checkpoint(template, tmp_path / 'existing.zip', enabled=True, seed=83)
    template.num_timesteps = 1
    with pytest.raises(ValueError, match='zero training'):
        create_ablation_checkpoint(template, tmp_path / 'trained.zip', enabled=True, seed=83)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
def test_relationship_buffers_follow_gpu():
    batch = tensors([matchup()])
    expected = MatchupRelations()(batch)
    actual = MatchupRelations().cuda()({k: v.cuda() for k, v in batch.items()}).cpu()
    torch.testing.assert_close(expected, actual)
