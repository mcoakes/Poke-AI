"""Decision weighting must change the learning signal without changing its controls."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sb3_contrib')
from sb3_contrib import MaskablePPO
from firered_rl import decision_weighted_warmstart as practice
from firered_rl.decision_weighted_warmstart import (
    BUCKETS, StratifiedReservoir, bucket_for_targets, weighted_imitation_loss,
)
from test_matchup_move_policy import matchup
from test_matchup_switch_policy import assert_same_tree, replacement
from test_teacher_warmstart import trained_model


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def action_mask(*actions):
    result = np.zeros(11, dtype=bool)
    result[list(actions)] = True
    return result


@pytest.mark.parametrize('actions,forced,rehearsal,expected', [
    ((0,), False, False, 'move'),
    ((0, 1, 3), False, False, 'move'),
    ((4,), False, False, 'switch'),
    ((4, 6, 9), False, False, 'switch'),
    ((1, 4, 9), False, False, 'move'),
    ((4, 9), True, False, 'forced'),
    ((4, 9), True, True, 'mirror'),
    ((0,), False, True, 'mirror'),
])
def test_bucket_labels_use_all_tied_best_actions_and_mirror_precedence(
        actions, forced, rehearsal, expected):
    assert bucket_for_targets(action_mask(*actions), forced=forced, rehearsal=rehearsal) == expected


def test_reservoir_append_replacement_and_backing_metadata_stay_synchronized():
    capacities = dict(move=3, forced=2, switch=2, mirror=1)
    reservoir = StratifiedReservoir(capacities, seed=261150)
    backing_rows = {}
    offers = Counter()
    seen_trajectories = Counter()
    replaced, discarded = Counter(), Counter()
    last_size = 0
    for serial in range(160):
        bucket = BUCKETS[serial % len(BUCKETS)]
        trajectory = 'student' if serial % 3 else 'rules'
        metadata = dict(serial=serial, trajectory=trajectory,
            pair=f'pair-{serial // 2}', group='old' if serial % 2 else 'new',
            episode=serial // 2 + 1, decision=serial % 2 + 1)
        offers[bucket] += 1
        seen_trajectories[trajectory] += 1
        index = reservoir.offer(bucket, metadata)
        if index is None:
            discarded[bucket] += 1
            assert len(reservoir.metadata) == last_size
        else:
            assert isinstance(index, int) and 0 <= index < len(reservoir.metadata)
            if index in backing_rows:
                replaced[bucket] += 1
                assert reservoir.buckets[index] == bucket
            else:
                assert index == last_size
            backing_rows[index] = deepcopy(metadata)
        last_size = len(reservoir.metadata)
        assert len(reservoir.metadata) == len(reservoir.buckets) == len(backing_rows)
        assert len(backing_rows) <= sum(capacities.values())
        for row, expected in backing_rows.items():
            assert reservoir.metadata[row] == expected
    assert Counter(reservoir.buckets) == capacities
    summary = reservoir.summary()
    for bucket, capacity in capacities.items():
        assert summary[bucket] == dict(seen=offers[bucket], kept=capacity,
            replaced=replaced[bucket], discarded=discarded[bucket])
        assert summary[bucket]['seen'] == capacity + replaced[bucket] + discarded[bucket]
    assert sum(replaced.values()) > 0 and sum(discarded.values()) > 0
    assert reservoir.trajectory_counts() == {
        'seen': dict(seen_trajectories),
        'kept': dict(Counter(row['trajectory'] for row in reservoir.metadata)),
    }


def test_bucket_streams_are_independent_of_other_bucket_arrivals():
    capacities = dict.fromkeys(BUCKETS, 3)
    alone = StratifiedReservoir(capacities, seed=42)
    interleaved = StratifiedReservoir(capacities, seed=42)
    for index in range(80):
        row = dict(serial=index, trajectory='student')
        alone.offer('move', row)
        for other in ('forced', 'switch', 'mirror'):
            interleaved.offer(other, dict(serial=-index, trajectory='rules'))
        interleaved.offer('move', row)
    kept_alone = sorted(row['serial'] for row, bucket in zip(alone.metadata, alone.buckets) if bucket == 'move')
    kept_interleaved = sorted(row['serial'] for row, bucket in zip(interleaved.metadata, interleaved.buckets) if bucket == 'move')
    assert kept_alone == kept_interleaved


def test_uniform_reservoir_does_not_prefer_early_or_late_arrivals():
    # Deterministic seeds make this reproducible; broad limits detect biased
    # selection while leaving substantial room around the expected 50 per item.
    retained = Counter()
    for seed in range(1000):
        reservoir = StratifiedReservoir(dict.fromkeys(BUCKETS, 1), seed=seed)
        for serial in range(20):
            reservoir.offer('forced', dict(serial=serial, trajectory='student'))
        assert len(reservoir.metadata) == 1
        retained[reservoir.metadata[0]['serial']] += 1
    assert set(retained) == set(range(20))
    assert all(20 <= frequency <= 85 for frequency in retained.values())


def weighted_batch():
    # The three legal teacher targets have losses exactly 1, 2 and 3.
    probabilities = [math.exp(-value) for value in (1., 2., 3.)]
    logits = np.full((4, 11), 900., dtype=np.float64)
    pairs = [(0, 1), (5, 6), (0, 5), (0, 1)]
    for index, probability in enumerate(probabilities):
        logits[index, list(pairs[index])] = np.log([1-probability, probability])
    logits[3, :2] = np.log([.5, .5])
    legal = np.zeros((4, 11), dtype=bool)
    for index, pair in enumerate(pairs):
        legal[index, list(pair)] = True
    tied = np.zeros((4, 11), dtype=bool)
    for index, pair in enumerate(pairs[:3]):
        tied[index, pair[1]] = True
    soft = np.zeros((4, 11), dtype=np.float64)
    soft[3, :2] = [.25, .75]
    buckets = np.asarray([BUCKETS.index(name) for name in ('move', 'forced', 'switch', 'mirror')])
    return [torch.as_tensor(value) for value in (logits, legal, tied, soft, buckets)]


def test_weighted_teacher_mean_and_exact_ten_percent_mirror_objective():
    logits, legal, tied, soft, buckets = weighted_batch()
    loss = weighted_imitation_loss(logits, legal, tied, soft, buckets)
    assert loss.item() == pytest.approx(.9 * (1 + 2*2 + 4*3) / 7 + .1 * math.log(2))
    teacher_only = weighted_imitation_loss(logits[:3], legal[:3], tied[:3], soft[:3],
                                           buckets[:3], with_mirror=False)
    assert teacher_only.item() == pytest.approx((1 + 2*2 + 4*3) / 7)


def test_mirror_row_count_does_not_change_the_declared_objective_mass():
    batch = weighted_batch()
    expected = weighted_imitation_loss(*batch)
    indexes = [0, 1, 2] + [3] * 30
    actual = weighted_imitation_loss(*(value[indexes] for value in batch))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def test_teacher_weights_scale_gradients_and_illegal_logits_receive_no_gradient():
    logits, legal, tied, soft, buckets = weighted_batch()
    logits = logits.clone().requires_grad_(True)
    weighted_imitation_loss(logits, legal, tied, soft, buckets).backward()
    for row, (weight, pair) in enumerate(zip((1., 2., 4.), ((0, 1), (5, 6), (0, 5)))):
        probability = math.exp(-(row+1))
        expected = .9 * weight / 7 * (1 - probability)
        assert logits.grad[row, pair[0]].item() == pytest.approx(expected)
        assert logits.grad[row, pair[1]].item() == pytest.approx(-expected)
    assert logits.grad[3, 0].item() == pytest.approx(.1 * (.5 - .25))
    assert logits.grad[3, 1].item() == pytest.approx(.1 * (.5 - .75))
    assert not logits.grad[~legal].any()


def test_training_objective_requires_teacher_and_mirror_rows_separately():
    batch = weighted_batch()
    with pytest.raises(ValueError):
        weighted_imitation_loss(*(value[:3] for value in batch))
    with pytest.raises(ValueError):
        weighted_imitation_loss(*(value[3:] for value in batch))


def test_prepare_retains_epoch_zero_on_tie_resets_both_optimizers_and_verifies_reuse(
        trained_model, tmp_path, monkeypatch):
    root = tmp_path / 'practice-run'
    (root / 'inputs').mkdir(parents=True)
    source_path = root / 'inputs/start.zip'
    trained_model.save(source_path)
    source_bytes = source_path.read_bytes()
    source_digest = practice.base.parameter_digest(trained_model.policy)
    source_counters = practice.ppo_counters(trained_model)
    source_optimizer = deepcopy(trained_model.policy.optimizer.state_dict())
    assert trained_model.policy.optimizer.state
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
    plan = dict(pins=pins, teams=teams, smoke=False,
        settings=dict(seed=261150, environments=2, max_steps=100, learning_rate=1e-4),
        practice=dict(battles=40, max_rows=16, decisions_per_battle=24, epochs=1,
            batch_size=4, learning_rate=1e-4, validation_fraction=.2, seed=261150,
            capacities=dict.fromkeys(BUCKETS, 4),
            weights=dict(move=1., forced=2., switch=4.), mirror_mass=.1))
    calls = []

    def collect(project, actual_root, attempt, request, source, cancel, announce):
        calls.append(attempt)
        data = practice.base._Dataset(attempt / 'dataset', request['config']['max_rows'])
        reservoir = StratifiedReservoir(request['config']['capacities'], request['config']['seed'])
        target = {'action': 1}

        def choose(obs, legal):
            return target['action'], [dict(action=int(action), score=float(action == target['action']))
                                      for action in np.flatnonzero(legal)]

        rules = SimpleNamespace(choose=choose)
        for group in ('old', 'new'):
            trajectory = 'student' if group == 'old' else 'rules'
            collector = practice._Collector(data, reservoir, source, rules, request, trajectory, cancel)
            for pair in range(2):
                case = dict(training_group=group, imitation_pair=f'{group}-pair-{pair}', episode=pair+1)
                for decision, bucket in enumerate(BUCKETS[:3], start=1):
                    obs = replacement() if bucket == 'forced' else matchup()
                    target['action'] = {'move': 1, 'forced': 6, 'switch': 5}[bucket]
                    collector(case, decision, obs, target['action'])
        mirror = practice._Collector(data, reservoir, source, rules, request, 'student', cancel)
        for decision in range(1, 5):
            mirror(dict(training_group='mirror', imitation_pair='mirror-pair', episode=10),
                   decision, matchup(), 1)
        assert data.rows == 16 and Counter(reservoir.buckets) == dict.fromkeys(BUCKETS, 4)
        for index, metadata in enumerate(reservoir.metadata):
            assert data.pairs[index] == metadata['pair']
            assert data.groups[index] == metadata['group']
            assert int(data.arrays['episode'][index]) == metadata['episode']
            assert int(data.arrays['decision'][index]) == metadata['decision']
        data.flush()
        return data, reservoir, {
            'student': {'status': 'completed', 'attempted_battles': 20},
            'rules': {'status': 'completed', 'attempted_battles': 20},
        }

    monkeypatch.setattr(practice, '_collect', collect)
    # Exact ties exercise epoch-0 retention in the real fit loop, rather than
    # mocking the fit result. The source's populated Adam state predates this patch.
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda self, closure=None: None)
    report = practice.prepare(root, plan, announce=lambda message: None)
    assert report['status'] == 'completed' and report['dataset']['rows'] == 16
    assert len(calls) == 1 and report['fit']['gradient_updates'] == 2
    assert report['fit']['selected_epoch'] == 0
    assert report['fit']['history'][0]['holdout_loss'] == report['fit']['history'][1]['holdout_loss']
    assert report['fit']['fit_rows'] == 10 and report['fit']['holdout_rows'] == 6
    assert report['fit']['split_counts']['fit']['switch'] == 2
    assert report['fit']['split_counts']['holdout']['switch'] == 2
    assert report['fit']['mirror_objective_mass'] == .1
    assert report['fit']['critic_parameters_unchanged']
    assert report['ppo_counters_unchanged'] and report['ppo_counters'] == source_counters
    assert (root / 'inputs/switch-practice.zip').read_bytes() == (root / 'models/practice-only.zip').read_bytes()
    for relative in practice.OUTPUTS:
        restored = MaskablePPO.load(root / relative, device='cpu')
        assert not restored.policy.optimizer.state
        assert practice.ppo_counters(restored) == source_counters
        assert practice.base.parameter_digest(restored.policy) == source_digest
        assert all(parameter.requires_grad for parameter in restored.policy.parameters())
    assert source_path.read_bytes() == source_bytes
    assert practice.base.parameter_digest(trained_model.policy) == source_digest
    assert practice.ppo_counters(trained_model) == source_counters
    assert_same_tree(trained_model.policy.optimizer.state_dict(), source_optimizer)
    json.dumps(report, allow_nan=False)
    assert practice.verify(root, plan) == report
    assert practice.prepare(root, plan, announce=lambda message: None) == report
    assert len(calls) == 1
    changed = root / report['attempt'] / 'dataset/holdout.npy'
    changed.write_bytes(changed.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='artifact changed'):
        practice.verify(root, plan)
