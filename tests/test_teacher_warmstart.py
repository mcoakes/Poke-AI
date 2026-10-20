"""Teacher warm-start controls: legal ties, rehearsal, freezing and group splits."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
from firered_rl.matchup_switch_policy import MatchupSwitchPolicy
from firered_rl import teacher_warmstart as warmstart
from firered_rl.teacher_warmstart import (
    critic_parameters, freeze_critic, imitation_loss, parameter_digest,
    reset_optimizer, split_pairs, tied_best_mask,
)
from test_matchup_move_policy import ContractEnv, matchup
from test_matchup_switch_policy import assert_same_tree, replacement
from test_shared_move_policy import outputs, tensors


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def padded(values, fill=0., *, dtype=np.float64):
    result = np.full(11, fill, dtype=dtype)
    result[:len(values)] = values
    return result


def test_teacher_ties_include_all_near_best_legal_actions_without_slot_preference():
    scores = padded([1., 1. - .5e-8, 1. - 2e-8, 999.], -999.)
    mask = padded([1, 1, 1], False, dtype=bool)
    expected = padded([1, 1], False, dtype=bool)
    actual = tied_best_mask(scores, mask)
    assert actual.dtype == np.bool_ and actual.shape == (11,)
    np.testing.assert_array_equal(actual, expected)
    order = [7, 2, 8, 1, 4, 5, 6, 3, 0, 9, 10]
    np.testing.assert_array_equal(tied_best_mask(scores[order], mask[order]), expected[order])
    assert tied_best_mask(scores, mask, tolerance=0.).sum() == 1


def test_equal_scores_make_every_legal_action_a_teacher_target():
    mask = padded([1, 0, 1, 0, 0, 1], False, dtype=bool)
    np.testing.assert_array_equal(tied_best_mask(np.zeros(11), mask), mask)
    only = np.zeros(11, dtype=bool)
    only[10] = True
    np.testing.assert_array_equal(tied_best_mask(np.arange(11), only), only)


def test_imitation_loss_is_tied_probability_mass_and_masks_illegal_logits():
    values = np.stack([padded(np.log([.2, .3, .5]), 999.),
                       padded(np.log([.25, .75]), 999.)])
    logits = torch.tensor(values, requires_grad=True)
    legal = torch.tensor(np.stack([padded([1, 1, 1], False, dtype=bool),
                                   padded([1, 1], False, dtype=bool)]))
    tied = torch.tensor(np.stack([padded([1, 1], False, dtype=bool),
                                  padded([0, 1], False, dtype=bool)]))
    loss = imitation_loss(logits, legal, tied, torch.zeros_like(logits),
                           torch.tensor([False, False]))
    assert loss.item() == pytest.approx((-math.log(.5) - math.log(.75)) / 2)
    loss.backward()
    expected = np.stack([padded([-.1, -.15, .25]), padded([.125, -.125])])
    np.testing.assert_allclose(logits.grad.numpy(), expected, atol=1e-12)
    assert not logits.grad[~legal].any()


def test_all_legal_teacher_ties_have_zero_loss_and_gradient():
    logits = torch.tensor(padded([4., -1., 2.])[None], requires_grad=True)
    legal = torch.tensor(padded([1, 1, 1], False, dtype=bool)[None])
    loss = imitation_loss(logits, legal, legal.clone(), torch.zeros_like(logits),
                           torch.tensor([False]))
    assert abs(loss.item()) < 1e-12
    loss.backward()
    assert logits.grad.abs().max() < 1e-12


def test_confidently_wrong_policy_has_finite_teacher_loss_and_nonzero_gradient():
    logits = torch.tensor(padded([1000., -1000.])[None], requires_grad=True)
    legal = torch.tensor(padded([1, 1], False, dtype=bool)[None])
    tied = torch.tensor(padded([0, 1], False, dtype=bool)[None])
    loss = imitation_loss(logits, legal, tied, torch.zeros_like(logits), torch.tensor([False]))
    assert torch.isfinite(loss) and loss.item() == pytest.approx(2000.)
    loss.backward()
    np.testing.assert_allclose(logits.grad.numpy(), padded([1., -1.])[None], atol=1e-12)


def test_rehearsal_uses_soft_source_distribution_and_ignores_teacher_tie_target():
    source = torch.tensor(padded([.25, .75])[None])
    legal = source > 0
    logits = torch.tensor(padded(np.log([.25, .75]), 500.)[None], requires_grad=True)
    tied = torch.tensor(padded([1], False, dtype=bool)[None])
    loss = imitation_loss(logits, legal, tied, source, torch.tensor([True]))
    assert loss.item() == pytest.approx(-.25 * math.log(.25) - .75 * math.log(.75))
    loss.backward()
    assert logits.grad.abs().max() < 1e-12
    changed = torch.tensor(padded(np.log([.75, .25]), 500.)[None], requires_grad=True)
    other_loss = imitation_loss(changed, legal, tied, source, torch.tensor([True]))
    assert other_loss > loss
    other_loss.backward()
    np.testing.assert_allclose(changed.grad.numpy(), [padded([.5, -.5])], atol=1e-12)


def test_mixed_teacher_and_rehearsal_rows_use_their_respective_targets():
    logits = torch.tensor(np.stack([padded(np.log([.2, .3, .5]), 999.),
                                    padded(np.log([.25, .75]), 999.)]))
    legal = torch.tensor(np.stack([padded([1, 1, 1], False, dtype=bool),
                                   padded([1, 1], False, dtype=bool)]))
    tied = torch.tensor(np.stack([padded([1, 1], False, dtype=bool),
                                  padded([1], False, dtype=bool)]))
    source = torch.tensor(np.stack([padded([.9, .1]), padded([.5, .5])]))
    loss = imitation_loss(logits, legal, tied, source, torch.tensor([False, True]))
    expected = (-math.log(.5) - .5 * math.log(.25) - .5 * math.log(.75)) / 2
    assert loss.item() == pytest.approx(expected)


@pytest.fixture
def trained_model():
    model = MaskablePPO(MatchupSwitchPolicy, ContractEnv(), seed=37, device='cpu',
        n_steps=4, batch_size=4, n_epochs=1, learning_rate=1e-4)
    model.learn(4)
    assert model.policy.optimizer.state
    return model


def test_reset_optimizer_gives_both_arms_empty_state_without_changing_source(
        trained_model, tmp_path):
    source_path = tmp_path / 'trained.zip'
    trained_model.save(source_path)
    source_bytes = source_path.read_bytes()
    weights = deepcopy(trained_model.policy.state_dict())
    optimizer = deepcopy(trained_model.policy.optimizer.state_dict())
    digest = parameter_digest(trained_model.policy)
    arms = [MaskablePPO.load(source_path, device='cpu') for _ in range(2)]
    for arm in arms:
        assert arm.policy.optimizer.state
        report = reset_optimizer(arm, learning_rate=2e-5)
        assert isinstance(report, dict)
        assert not arm.policy.optimizer.state
        assert arm.num_timesteps == trained_model.num_timesteps == 4
        assert parameter_digest(arm.policy) == digest
        assert_same_tree(weights, arm.policy.state_dict())
        assert {group['lr'] for group in arm.policy.optimizer.param_groups} == {2e-5}
        registered = {id(param) for group in arm.policy.optimizer.param_groups for param in group['params']}
        assert registered == {id(param) for param in arm.policy.parameters()}
        for before, after in zip(outputs(trained_model.policy, [matchup(), replacement()]),
                                 outputs(arm.policy, [matchup(), replacement()])):
            np.testing.assert_array_equal(before, after)
    assert_same_tree(arms[0].policy.state_dict(), arms[1].policy.state_dict())
    assert_same_tree(arms[0].policy.optimizer.state_dict(), arms[1].policy.optimizer.state_dict())
    assert_same_tree(weights, trained_model.policy.state_dict())
    assert_same_tree(optimizer, trained_model.policy.optimizer.state_dict())
    assert source_path.read_bytes() == source_bytes


def test_freeze_only_critic_parameters_leaves_shared_encoder_and_actor_trainable(trained_model):
    policy = trained_model.policy
    expected = {name: parameter for name, parameter in policy.named_parameters()
                if name.startswith(('mlp_extractor.critic.', 'value_net.'))}
    assert expected
    actual = critic_parameters(policy)
    assert actual.keys() == expected.keys()
    assert all(actual[name] is param for name, param in expected.items())
    # Freeze must restore actor/shared flags even if a caller previously froze everything.
    for param in policy.parameters():
        param.requires_grad_(False)
    digest = parameter_digest(policy)
    frozen = freeze_critic(policy)
    assert set(frozen) == set(expected)
    assert parameter_digest(policy) == digest
    for name, parameter in policy.named_parameters():
        assert parameter.requires_grad is (name not in expected), name
    batch = tensors([matchup(), replacement()])
    logits = policy.action_net(policy.extract_features(batch))
    legal = batch['action_mask'].bool()
    tied = torch.zeros_like(legal)
    tied[0, 1] = True
    tied[1, 6] = True
    policy.optimizer.zero_grad()
    loss = imitation_loss(logits, legal, tied, torch.zeros_like(logits),
                           torch.tensor([False, False]))
    loss.backward()
    assert all(param.grad is None for param in expected.values())
    for parameter in (policy.features_extractor.context[0].weight,
                      policy.action_net.moves[-1].weight,
                      policy.action_net.switch_residual[-1].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    old_critic = {name: parameter.detach().clone() for name, parameter in expected.items()}
    policy.optimizer.step()
    assert all(torch.equal(old_critic[name], parameter) for name, parameter in expected.items())
    assert parameter_digest(policy) != digest


def test_parameter_digest_changes_with_weights_but_not_optimizer_reset(trained_model):
    policy = trained_model.policy
    before = parameter_digest(policy)
    assert isinstance(before, str) and len(before) == 64
    reset_optimizer(trained_model)
    assert parameter_digest(policy) == before
    with torch.no_grad():
        policy.action_net.switch_residual[-1].weight[0, 0].add_(.01)
    assert parameter_digest(policy) != before


def test_pair_split_keeps_all_episodes_sides_and_trajectory_sources_together():
    pairs, groups = [], []
    for group in ('old', 'new'):
        for index in range(10):
            # Multiple decisions from both sides and both collectors share one canonical pair.
            pairs.extend([f'{group}-unordered-pair-{index}'] * 8)
            groups.extend([group] * 8)
    pairs.extend(['fixed-mirror-pair'] * 20)
    groups.extend(['mirror'] * 20)
    chosen = split_pairs(pairs, groups, seed=261140, fraction=.2)
    assert chosen.dtype == np.bool_ and chosen.shape == (len(pairs),)
    for pair in set(pairs):
        flags = {bool(chosen[index]) for index, value in enumerate(pairs) if value == pair}
        assert len(flags) == 1
    for group in ('old', 'new'):
        fit = {pair for pair, kind, heldout in zip(pairs, groups, chosen) if kind == group and not heldout}
        heldout = {pair for pair, kind, flag in zip(pairs, groups, chosen) if kind == group and flag}
        assert len(fit) == 8 and len(heldout) == 2 and not fit & heldout
    assert not chosen[np.asarray(groups) == 'mirror'].any()
    np.testing.assert_array_equal(split_pairs(pairs, groups, seed=261140, fraction=.2), chosen)
    order = np.random.default_rng(72).permutation(len(pairs))
    reordered = split_pairs([pairs[i] for i in order], [groups[i] for i in order], seed=261140, fraction=.2)
    np.testing.assert_array_equal(reordered, chosen[order])


def test_one_pair_group_and_mirror_alias_remain_fit():
    pairs = ['old-one'] * 3 + ['new-a', 'new-b'] * 2 + ['mirror-a'] * 2
    groups = ['old'] * 3 + ['new'] * 4 + ['fixed-mirror'] * 2
    chosen = split_pairs(pairs, groups, seed=261140, fraction=.2)
    assert not chosen[:3].any() and not chosen[-2:].any()
    assert chosen[3:7].sum() == 2


def test_split_rejects_one_canonical_pair_assigned_to_different_groups():
    with pytest.raises(ValueError):
        split_pairs(['same-pair', 'same-pair'], ['old', 'new'], seed=261140)


def test_prepare_fits_serializes_verifies_and_reuses_without_recollecting(
        trained_model, tmp_path, monkeypatch):
    root = tmp_path / 'warmstart-run'
    (root / 'inputs').mkdir(parents=True)
    source_path = root / 'inputs/start.zip'
    trained_model.save(source_path)
    source_bytes = source_path.read_bytes()
    source_digest = parameter_digest(trained_model.policy)
    source_weights = deepcopy(trained_model.policy.state_dict())
    source_optimizer = deepcopy(trained_model.policy.optimizer.state_dict())
    (root / 'inputs/reward.json').write_text('{}\n')
    teams = {}
    for group in ('old', 'new', 'mirror'):
        teams[group] = []
        for index in range(2):
            relative = f'inputs/{group}-{index}.json'
            team = dict(name=f'{group}-{index}', party=[dict(species='Blastoise',
                level=50, nature='Hardy', held_item='None', moves=['Surf', 'Strength'])])
            (root / relative).write_text(json.dumps(team))
            teams[group].append(relative)
    paths = ['inputs/start.zip', 'inputs/reward.json',
             *(relative for group in teams.values() for relative in group)]
    pins = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in paths}
    plan = dict(pins=pins, teams=teams,
        settings=dict(seed=261140, environments=2, max_steps=100, learning_rate=1e-4),
        warmstart=dict(battles=40, max_rows=16, decisions_per_battle=24, epochs=1,
            batch_size=4, learning_rate=1e-4, validation_fraction=.2, seed=261140))
    calls = []

    def collect(project, actual_root, attempt, request, source, cancel, announce):
        calls.append(attempt)
        data = warmstart._Dataset(attempt / 'dataset', request['config']['max_rows'])
        for group in ('old', 'new', 'mirror'):
            for decision in range(4):
                obs = replacement() if decision % 2 else matchup()
                if not data.arrays:
                    data.initialize(obs)
                row = data.rows
                legal = obs['action_mask'].astype(bool)
                tied = np.zeros(11, dtype=bool)
                soft = np.zeros(11, dtype=np.float32)
                rehearsal = group == 'mirror'
                if rehearsal:
                    with torch.no_grad():
                        batch, _ = source.policy.obs_to_tensor(obs)
                        soft = source.policy.get_distribution(batch,
                            action_masks=legal[None]).distribution.probs[0].cpu().numpy()
                else:
                    tied[6 if obs['context'][1] else 1] = True
                for name, value in obs.items():
                    data.arrays['obs__' + name][row] = value
                values = dict(legal=legal, tied=tied, soft=soft, rehearsal=rehearsal,
                    forced=bool(obs['context'][1]), episode=row // 2 + 1, decision=row % 2 + 1)
                for name, value in values.items():
                    data.arrays[name][row] = value
                data.pairs.append('mirror-pair' if rehearsal else f'{group}-pair-{decision // 2}')
                data.groups.append(group)
                data.counts['group/' + group] += 1
                data.rows += 1
        data.flush()
        return data, {'student': {'status': 'completed'}, 'rules': {'status': 'completed'}}

    monkeypatch.setattr(warmstart, '_collect', collect)
    report = warmstart.prepare(root, plan, announce=lambda message: None)
    assert report['status'] == 'completed' and report['dataset']['rows'] == 12
    assert len(calls) == 1
    assert report['fit']['gradient_updates'] == 2
    assert report['fit']['fit_rows'] == 8 and report['fit']['holdout_rows'] == 4
    assert report['fit']['critic_parameters_unchanged']
    assert report['fit']['initial_metrics']['holdout']['rules_tie_aware_agreement']['all']['rows'] == 4
    assert (root / 'inputs/teacher-ppo.zip').read_bytes() == (root / 'models/teacher-only.zip').read_bytes()
    for relative in warmstart.OUTPUTS:
        assert (root / relative).is_file()
        restored = MaskablePPO.load(root / relative, device='cpu')
        assert not restored.policy.optimizer.state
        assert restored.num_timesteps == trained_model.num_timesteps
        assert all(parameter.requires_grad for parameter in restored.policy.parameters())
        if relative == 'inputs/ppo-only.zip':
            assert parameter_digest(restored.policy) == source_digest
            assert_same_tree(restored.policy.state_dict(), source_weights)
    assert source_path.read_bytes() == source_bytes
    assert_same_tree(trained_model.policy.state_dict(), source_weights)
    assert_same_tree(trained_model.policy.optimizer.state_dict(), source_optimizer)
    json.dumps(report, allow_nan=False)
    assert warmstart.verify(root, plan) == report
    assert warmstart.prepare(root, plan, announce=lambda message: None) == report
    assert len(calls) == 1
    artifact = root / report['attempt'] / 'dataset/holdout.npy'
    artifact.write_bytes(artifact.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='artifact changed'):
        warmstart.verify(root, plan)
