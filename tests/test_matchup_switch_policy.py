"""A forced-replacement residual must preserve the trained policy it extends."""
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
from firered_rl.matchup_move_policy import MatchupMovePolicy
from firered_rl.matchup_switch_policy import (
    BenchMatchupFeatures, MatchupSwitchPolicy, create_switch_checkpoint,
)
from firered_rl.ppo_features import policy_space
from firered_rl.shared_move_policy import policy_architecture
from test_matchup_move_policy import ContractEnv, matchup
from test_shared_move_policy import outputs, tensors


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def replacement():
    """A fainted lead and two healthy, meaningfully different replacements."""
    obs = matchup(target=(11, 11))
    obs['context'][1] = 1
    obs['action_mask'][:] = 0
    obs['action_mask'][5:7] = 1
    obs['party'][0, 3] = 0
    obs['v2_party'][0, 0, 2] = 0
    obs['v2_party'][0, 0, 8] = 1
    obs['v2_active'][0, 0] = 0
    for slot, kind, ident, hp, stats in (
        (1, 13, 85, 150, [110, 80, 120, 75, 90]),
        (2, 4, 89, 75, [140, 110, 50, 80, 40]),
    ):
        obs['party'][slot, 3:7] = [hp, 150, kind, kind]
        obs['v2_party'][0, slot, 2:6] = [hp, 150, kind, kind]
        obs['v2_party'][0, slot, 7:9] = [0, 0]
        obs['v2_stats'][0, slot] = stats
        obs['moves'][slot] = 0
        obs['v2_move_details'][0, slot] = 0
        obs['moves'][slot, 0] = [ident, kind, 10, 10]
        obs['v2_move_details'][0, slot, 0] = [100, 100, 0, int(kind >= 10), 0, kind]
    return obs


@pytest.fixture
def trained_template():
    model = MaskablePPO(MatchupMovePolicy, ContractEnv(), seed=37, device='cpu',
        n_steps=4, batch_size=4, n_epochs=1, learning_rate=1e-4)
    model.learn(4)
    assert model.policy.optimizer.state
    return model


def optimizer_by_name(policy):
    return {name: deepcopy(policy.optimizer.state[param])
            for name, param in policy.named_parameters()
            if param in policy.optimizer.state}


def assert_same_tree(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_tree(a, b)
    else:
        assert left == right


def test_migration_preserves_trained_weights_optimizer_and_initial_behavior(
        trained_template, tmp_path):
    template = trained_template
    weights = deepcopy(template.policy.state_dict())
    optimizer = optimizer_by_name(template.policy)
    original_optimizer = deepcopy(template.policy.optimizer.state_dict())
    arms = [create_switch_checkpoint(template, tmp_path / f'{enabled}.zip',
                enabled=enabled, seed=83, training_settings={'ent_coef': .01})
            for enabled in (False, True)]
    observations = [matchup(), replacement(), matchup(target=(2, 0)), matchup(ability=26)]
    expected = outputs(template.policy, observations)
    for enabled, model in zip((False, True), arms):
        assert type(model.policy) is MatchupSwitchPolicy
        assert policy_architecture(model) == 'matchup-switches'
        assert model.num_timesteps == template.num_timesteps == 4
        assert model.ent_coef == .01
        assert model.policy.features_extractor.bench_features_enabled is enabled
        for name, value in weights.items():
            assert torch.equal(value, model.policy.state_dict()[name]), name
        copied_optimizer = optimizer_by_name(model.policy)
        for name, state in optimizer.items():
            assert_same_tree(state, copied_optimizer[name])
        assert all(not name.startswith('action_net.switch_residual.')
                   for name in copied_optimizer)
        for before, after in zip(expected, outputs(model.policy, observations)):
            np.testing.assert_array_equal(before, after)
    assert_same_tree(arms[0].policy.state_dict(), arms[1].policy.state_dict())
    assert_same_tree(weights, template.policy.state_dict())
    assert_same_tree(original_optimizer, template.policy.optimizer.state_dict())


def test_residual_only_changes_multi_option_forced_switches(trained_template, tmp_path):
    model = create_switch_checkpoint(trained_template, tmp_path / 'switch.zip',
                                     enabled=True, seed=83)
    forced = replacement()
    voluntary = deepcopy(forced)
    voluntary['context'][1] = 0
    one = deepcopy(forced)
    one['action_mask'][6] = 0
    terminal = deepcopy(forced)
    terminal['context'][2] = 1
    mixed = deepcopy(forced)
    mixed['action_mask'][0] = 1
    observations = [forced, voluntary, one, terminal, mixed]
    before = outputs(model.policy, observations)
    final = model.policy.action_net.switch_residual[-1]
    with torch.no_grad():
        # Give the shared branch a constant nonzero correction; the final
        # layer deliberately has no redundant common-offset bias.
        hidden = model.policy.action_net.switch_residual[-3]
        hidden.weight.zero_()
        hidden.bias.fill_(1.)
        final.weight.fill_(1 / (final.in_features * np.tanh(1.)))
    after = outputs(model.policy, observations)
    np.testing.assert_allclose(after[0][0, 5:7] - before[0][0, 5:7], 1., atol=1e-6)
    np.testing.assert_array_equal(after[0][:, :4], before[0][:, :4])
    np.testing.assert_array_equal(after[0][:, 10:], before[0][:, 10:])
    np.testing.assert_array_equal(after[0][1:], before[0][1:])
    np.testing.assert_array_equal(after[2], before[2])
    for obs, probs in zip(observations, after[1]):
        assert not probs[obs['action_mask'] == 0].any()
        assert np.isclose(probs.sum(), 1)


@pytest.mark.parametrize('enabled', [False, True])
def test_feature_toggle_preserves_base_features_and_legal_gate(
        trained_template, tmp_path, enabled):
    model = create_switch_checkpoint(trained_template, tmp_path / 'switch.zip',
                                     enabled=enabled, seed=83)
    obs = replacement()
    batch = tensors([obs])
    original = trained_template.policy.extract_features(batch)
    augmented = model.policy.extract_features(batch)
    torch.testing.assert_close(augmented[:, :original.shape[1]], original, rtol=0, atol=0)
    tokens = augmented[:, original.shape[1]:].reshape(1, 6, 19)
    torch.testing.assert_close(tokens[0, :, -1], torch.tensor([0., 1., 1., 0., 0., 0.]))
    if enabled:
        torch.testing.assert_close(tokens[:, :, :-1], BenchMatchupFeatures()(batch))
    else:
        assert not tokens[:, :, :-1].any()
    assert not model.policy.action_net.residual_scores(augmented).any()


@pytest.mark.parametrize('enabled', [False, True])
def test_checkpoint_and_policy_roundtrip_keep_toggle_and_optimizer(
        trained_template, tmp_path, enabled):
    path = tmp_path / 'switch.zip'
    model = create_switch_checkpoint(trained_template, path, enabled=enabled, seed=83)
    loaded = MaskablePPO.load(path, device='cpu')
    policy_path = tmp_path / 'policy.pt'
    model.policy.save(policy_path)
    loaded_policy = MatchupSwitchPolicy.load(policy_path, device='cpu')
    for policy in (loaded.policy, loaded_policy):
        assert policy.features_extractor.bench_features_enabled is enabled
        for before, after in zip(outputs(model.policy, [replacement()]),
                                 outputs(policy, [replacement()])):
            np.testing.assert_array_equal(before, after)
    assert_same_tree(optimizer_by_name(model.policy), optimizer_by_name(loaded.policy))
    loaded.set_env(ContractEnv())
    loaded.learn(4, reset_num_timesteps=False)
    assert loaded.num_timesteps == 8
    assert all(torch.isfinite(value).all() for value in loaded.policy.state_dict().values())


def test_forced_training_reaches_new_scorer_with_legal_actions(
        trained_template, tmp_path):
    model = create_switch_checkpoint(trained_template, tmp_path / 'switch.zip',
                                     enabled=True, seed=83)
    policy = model.policy
    obs = replacement()
    batch = tensors([obs])
    final = policy.action_net.switch_residual[-1]
    old = final.weight.detach().clone()
    _, logp, _ = policy.evaluate_actions(batch, torch.tensor([5]),
                                        action_masks=batch['action_mask'].bool())
    policy.optimizer.zero_grad()
    (-logp.mean()).backward()
    assert final.weight.grad is not None and torch.isfinite(final.weight.grad).all()
    assert final.weight.grad.abs().sum() > 0
    policy.optimizer.step()
    assert not torch.equal(old, final.weight)
    assert not outputs(policy, [obs])[1][0][obs['action_mask'] == 0].any()


def permute_party(obs, order):
    changed = deepcopy(obs)
    for name in ('party', 'moves'):
        changed[name] = obs[name][order].copy()
    for name in ('v2_party', 'v2_stats', 'v2_status_bits', 'v2_move_details'):
        changed[name][0] = obs[name][0, order].copy()
    changed['status_bits'][:6] = obs['status_bits'][order].copy()
    changed['action_mask'][4:10] = obs['action_mask'][np.asarray(order) + 4]
    old_active = int(obs['active'][0, 8]) - 1
    changed['active'][0, 8] = order.index(old_active) + 1
    return changed


def test_bench_features_follow_whole_party_permutations_and_do_not_mutate():
    obs = replacement()
    before = deepcopy(obs)
    feature = BenchMatchupFeatures()
    expected = feature(tensors([obs]))[0]
    assert expected.shape == (6, len(feature.feature_names))
    assert len(feature.feature_names) == len(set(feature.feature_names)) == 18
    # Also move the fainted lead: validity follows identity, never a fixed index.
    for order in ([0, 2, 1, 5, 4, 3], [2, 4, 0, 1, 5, 3]):
        actual = feature(tensors([permute_party(obs, list(order))]))[0]
        torch.testing.assert_close(actual, expected[list(order)], rtol=0, atol=0)
    assert all(np.array_equal(obs[key], before[key]) for key in obs)


def test_shared_residual_scores_follow_party_permutation_after_output_is_nonzero(
        trained_template, tmp_path):
    model = create_switch_checkpoint(trained_template, tmp_path / 'switch.zip',
                                     enabled=True, seed=83)
    policy = model.policy
    with torch.no_grad():
        policy.action_net.switch_residual[-1].weight.fill_(.1)
    obs = replacement()
    expected = policy.action_net.residual_scores(policy.extract_features(tensors([obs])))[0]
    assert expected.abs().sum() > 0
    assert not expected[[0, 3, 4, 5]].any()
    for order in ([0, 2, 1, 5, 4, 3], [2, 4, 0, 1, 5, 3]):
        batch = tensors([permute_party(obs, list(order))])
        actual = policy.action_net.residual_scores(policy.extract_features(batch))[0]
        torch.testing.assert_close(actual, expected[list(order)], rtol=0, atol=0)


def with_opponent_attack(obs, ident=57, kind=11, category=1, power=100):
    obs = deepcopy(obs)
    obs['opponent_moves'][:] = 0
    obs['v2_move_details'][1] = 0
    for slot in range(6):
        obs['opponent_moves'][slot, 0] = [ident, kind, 10, 10]
        obs['v2_move_details'][1, slot, 0] = [power, 100, 0, category, 0, kind]
    return obs


def bench_rows(obs):
    feature = BenchMatchupFeatures()
    result = feature(tensors([obs]))[0]
    return [{name: value.item() for name, value in zip(feature.feature_names, row)}
            for row in result]


def test_bench_numeric_matchups_include_stab_category_stats_and_health():
    obs = with_opponent_attack(replacement())
    electric, ground = bench_rows(obs)[1:3]
    expected_electric = {
        'hp_fraction': 1., 'maximum_hp': 150/1024, 'level': .5,
        'attack': 110/1024, 'defense': 80/1024,
        'special_attack': 120/1024, 'special_defense': 75/1024,
        'speed': 90/1024, 'status_present': 0.,
        'outgoing_assessed_fraction': .25, 'outgoing_max_multiplier': .5,
        'outgoing_power_proxy': 100 * 1.5 * 2 / 1500,
        'outgoing_stat_proxy': 100 * 1.5 * 2 * (120/70) / 1500,
        'incoming_assessed_fraction': .25, 'incoming_max_multiplier': .25,
        'incoming_stat_proxy': 100 * 1.5 * (80/75) / 1500,
        'speed_fraction': 90/(90+60), 'opponent_hp_fraction': 100/150,
    }
    assert electric == pytest.approx(expected_electric)
    assert ground['hp_fraction'] == .5
    assert ground['outgoing_max_multiplier'] == .25
    assert ground['outgoing_power_proxy'] == pytest.approx(100 * 1.5 / 1500)
    assert ground['outgoing_stat_proxy'] == pytest.approx(100 * 1.5 * (140/90) / 1500)
    assert ground['incoming_max_multiplier'] == .5
    assert ground['incoming_stat_proxy'] == pytest.approx(100 * 1.5 * 2 * (80/80) / 1500)


def test_outgoing_target_ability_and_incoming_bench_ability_are_distinct():
    obs = with_opponent_attack(replacement(), ident=89, kind=4, category=0)
    base = bench_rows(obs)[1]
    assert base['outgoing_max_multiplier'] == .5
    assert base['incoming_max_multiplier'] == .5
    obs['v2_active'][1, 3] = 10  # Opponent's Volt Absorb blocks outgoing Electric.
    obs['v2_party'][0, 1, 7] = 26  # Candidate's Levitate blocks incoming Ground.
    immune = bench_rows(obs)[1]
    assert immune['outgoing_assessed_fraction'] == immune['incoming_assessed_fraction'] == .25
    assert immune['outgoing_max_multiplier'] == immune['incoming_max_multiplier'] == 0
    assert immune['outgoing_power_proxy'] == immune['outgoing_stat_proxy'] == 0
    assert immune['incoming_stat_proxy'] == 0


def test_current_foe_stages_affect_proxies_but_old_own_active_stages_do_not():
    obs = with_opponent_attack(replacement())
    baseline = bench_rows(obs)[1]
    obs['v2_boosts'][1, 2:5] = [2, -2, 2]  # SpA x2, SpD /2, Speed x2.
    boosted = bench_rows(obs)[1]
    assert boosted['outgoing_power_proxy'] == baseline['outgoing_power_proxy']
    assert boosted['outgoing_stat_proxy'] == pytest.approx(2 * baseline['outgoing_stat_proxy'])
    assert boosted['incoming_stat_proxy'] == pytest.approx(2 * baseline['incoming_stat_proxy'])
    assert boosted['speed_fraction'] == pytest.approx(90 / (90 + 120))
    obs['v2_boosts'][0] = 6  # A replacement does not inherit the fainted lead's stages.
    assert bench_rows(obs)[1] == boosted


@pytest.mark.parametrize('change', ['padding', 'fainted', 'empty_hp'])
def test_invalid_bench_members_are_zero_even_with_stale_moves(change):
    obs = with_opponent_attack(replacement())
    if change == 'padding':
        obs['v2_party'][0, 1, 0] = 0
    elif change == 'fainted':
        obs['v2_party'][0, 1, 8] = 1
    else:
        obs['v2_party'][0, 1, 2] = 0
    result = BenchMatchupFeatures()(tensors([obs]))[0]
    assert not result[1].any()
    assert result[2].any()


@pytest.mark.parametrize('change', ['terminal', 'missing_target', 'dead_target'])
def test_unusable_current_opponent_zeros_all_bench_features(change):
    obs = with_opponent_attack(replacement())
    if change == 'terminal':
        obs['context'][2] = 1
    elif change == 'missing_target':
        obs['active'][1, 0] = 0
    else:
        obs['v2_active'][1, 0] = 0
    assert not BenchMatchupFeatures()(tensors([obs])).any()


@pytest.mark.parametrize('change', ['no_pp', 'unsupported', 'status', 'no_power'])
def test_unassessed_attacks_do_not_become_damage_or_immunity_evidence(change):
    obs = with_opponent_attack(replacement())
    if change == 'no_pp':
        obs['moves'][1, 0, 2] = 0
        obs['opponent_moves'][:, 0, 2] = 0
    elif change == 'unsupported':
        obs['moves'][1, 0, 0] = 165
        obs['opponent_moves'][:, 0, 0] = 165
    elif change == 'status':
        obs['v2_move_details'][0, 1, 0, 3] = 2
        obs['v2_move_details'][1, :, 0, 3] = 2
    else:
        obs['v2_move_details'][0, 1, 0, 0] = 1
        obs['v2_move_details'][1, :, 0, 0] = 1
    row = bench_rows(obs)[1]
    for key in ('outgoing_assessed_fraction', 'outgoing_max_multiplier',
                'outgoing_power_proxy', 'outgoing_stat_proxy',
                'incoming_assessed_fraction', 'incoming_max_multiplier',
                'incoming_stat_proxy'):
        assert row[key] == 0
    assert row['hp_fraction'] == 1


def test_status_and_move_slot_permutation_leave_semantic_bench_summary_intact():
    obs = with_opponent_attack(replacement())
    obs['v2_status_bits'][0, 1, 4] = 1
    expected = BenchMatchupFeatures()(tensors([obs]))
    assert bench_rows(obs)[1]['status_present'] == 1
    order = [2, 0, 3, 1]
    for side, key in enumerate(('moves', 'opponent_moves')):
        obs[key] = obs[key][:, order].copy()
        obs['v2_move_details'][side] = obs['v2_move_details'][side][:, order].copy()
    torch.testing.assert_close(BenchMatchupFeatures()(tensors([obs])), expected, rtol=0, atol=0)


def test_converter_rejects_invalid_toggle_architecture_and_overwrite(trained_template, tmp_path):
    with pytest.raises(ValueError, match='boolean'):
        create_switch_checkpoint(trained_template, tmp_path / 'bad.zip', enabled=1, seed=83)
    path = tmp_path / 'existing.zip'
    model = create_switch_checkpoint(trained_template, path, enabled=True, seed=83)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        create_switch_checkpoint(trained_template, path, enabled=True, seed=83)
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        create_switch_checkpoint(model, tmp_path / 'twice.zip', enabled=True, seed=83)


def test_rejects_incompatible_policy_contracts():
    with pytest.raises(ValueError, match='v2'):
        MatchupSwitchPolicy(policy_space('v1'), gym.spaces.Discrete(11), lambda _: 1e-4)
    with pytest.raises(ValueError, match='Discrete'):
        MatchupSwitchPolicy(policy_space('v2'), gym.spaces.Discrete(4), lambda _: 1e-4)
