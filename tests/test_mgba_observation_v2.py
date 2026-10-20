from copy import deepcopy

import numpy as np
import pytest

from firered_rl.link_inspection import pair_inspections
from firered_rl.linked_env import linked_observation, LinkedBattleSession
from firered_rl.mgba_observation_v2 import move_details, move_catalog
from firered_rl.observation_v2 import versioned_raw_space
from test_link_inspection import snapshots
from test_linked_env import runtime


def enrich(snapshots):
    for i, report in enumerate(snapshots):
        report['native_v2'] = 1
        report['battle'].update(weather_raw=0x20 if i == 0 else 0x80, weather_duration=4 if i == 0 else 1)
        for mon in report['party']:
            if mon.get('valid'):
                mon.update(ability_id=65+i, stats=[101+i, 91+i, 81+i, 71+i, 61+i], ivs_raw=0x3FFFFFFF)
        for mon in report['battle']['battlers']:
            mon.update(stats=[111, 92, 83, 74, 65], ivs_raw=0x3FFFFFFF,
                       stat_stages=[6, 7, 5, 8, 4, 9, 10, 3])
    for i, report in enumerate(snapshots):
        report['battle']['opponent_party'] = deepcopy(snapshots[1-i]['party'])
    return snapshots


@pytest.mark.parametrize('reverse', [False, True])
def test_paired_native_observations_use_master_and_both_local_parties(snapshots, reverse):
    enrich(snapshots)
    if reverse:
        snapshots.reverse()
    before = deepcopy(snapshots)
    reports = pair_inspections(snapshots, [55355, 55356])
    observations = [linked_observation(r, observation_version='v2') for r in reports]
    assert snapshots == before
    for obs in observations:
        assert versioned_raw_space('v2').contains(obs)
        np.testing.assert_array_equal(obs['v2_weather'], [1, 4, 0])
        np.testing.assert_array_equal(obs['v2_active_stats'][0], [111, 92, 83, 74, 65])
        np.testing.assert_array_equal(obs['v2_boosts'][0], [1, -1, -2, 3, 2, 4, -3])
    for name in ('v2_party', 'v2_stats', 'v2_active', 'v2_active_stats', 'v2_boosts', 'v2_move_details', 'v2_volatiles'):
        np.testing.assert_array_equal(observations[0][name], observations[1][name][::-1])


@pytest.mark.parametrize('flags,duration,expected', [(0, 8, [0,0,0]), (1,3,[2,3,0]),
    (5,5,[2,0,1]), (0x18,5,[3,0,1]), (0x60,2,[1,0,1]), (0x80,2,[4,2,0])])
def test_weather_contract(snapshots, flags, duration, expected):
    enrich(snapshots)[0]['battle'].update(weather_raw=flags, weather_duration=duration)
    report = pair_inspections(snapshots, [1,2])[1]
    assert linked_observation(report, observation_version='v2')['v2_weather'].tolist() == expected


def test_hidden_power_and_empty_slots():
    for ivs, expected in [(0, [30,100,0,0,0,1]), (0x3FFFFFFF, [70,100,0,1,0,17])]:
        rows = move_details(dict(ivs_raw=ivs, moves=[dict(slot=3, move_id=237)]))
        assert rows[2] == expected and rows[0] == rows[1] == rows[3] == [0]*6
    # Variable-damage cartridge power sentinels must not become a fake power 1.
    assert move_catalog()['68'][0] == 0  # Counter


def test_sparse_transform_and_status_bits_reach_policy(snapshots):
    pytest.importorskip('torch')
    from firered_rl.ppo_features import policy_observation
    enrich(snapshots)
    mon = snapshots[0]['battle']['battlers'][0]
    mon['moves'] = [dict(slot=2, move_id=237, current_pp=5, max_pp=5, base_type_id=0)]
    mon['status_raw'] = (1 << 30) | 1
    mon['volatile_status_raw'] = (1 << 31) | 1
    report = pair_inspections(snapshots, [1,2])[0]
    raw = linked_observation(report, observation_version='v2')
    assert raw['v2_move_details'][0,0,0].tolist() == [0]*6
    assert raw['v2_move_details'][0,0,1].tolist() == [70,100,0,1,0,17]
    encoded = policy_observation(raw)
    assert encoded['v2_status_bits'][0,0,[0,30]].tolist() == [1,1]
    assert encoded['v2_volatile_bits'][0,[0,31]].tolist() == [1,1]


def test_old_or_incomplete_data_fails_without_zero_filling(snapshots):
    report = pair_inspections(snapshots, [1,2])[0]
    with pytest.raises(ValueError):
        linked_observation(report, observation_version='v2')
    enrich(snapshots)
    del snapshots[0]['party'][0]['stats']
    report = pair_inspections(snapshots, [1,2])[1]
    with pytest.raises(ValueError, match='incomplete'):
        linked_observation(report, observation_version='v2')


def test_old_adapter_rejected_at_reset(runtime):
    launch, *_ = runtime
    session = LinkedBattleSession(launch, observation_version='v2')
    try:
        with pytest.raises(RuntimeError, match='adapter 0.19.0'):
            session.reset_pair()
    finally:
        session.close()


def test_native_session_terminal_observations_and_spaces(runtime, monkeypatch):
    launch, _, _, bridges, state = runtime
    import firered_rl.linked_env as module
    original = module.MGBABridgeBackend.status
    monkeypatch.setattr(module.MGBABridgeBackend, 'status',
                        lambda self: dict(original(self), native_observation_v2=1))
    def start():
        process = launch()
        enrich(state)
        return process
    session = LinkedBattleSession(start, observation_version='v2')
    try:
        reset = session.reset_pair()
        assert all(session.observation_space.contains(row[0]) for row in reset)
        results = session.step_pair([0,0])
        for obs, _, terminal, truncated, _ in results:
            assert terminal and not truncated and session.observation_space.contains(obs)
            assert not obs['action_mask'].any()
        assert results[0][0]['v2_party'][0,0,2] == results[1][0]['v2_party'][1,0,2] == 0
    finally:
        session.close()


@pytest.mark.parametrize('architecture', ['flat', 'shared-moves'])
def test_actual_checkpoint_predicts_both_mgba_seats(snapshots, tmp_path, architecture):
    torch = pytest.importorskip('torch')
    MaskablePPO = pytest.importorskip('sb3_contrib').MaskablePPO
    from firered_rl.ppo_features import policy_space, BattleFeatures
    from firered_rl.shared_move_policy import SharedMovePolicy
    from firered_rl.human_play import PPOOpponent
    import gymnasium as gym
    torch.set_num_threads(1)
    # No simulator is needed to construct this disposable inference checkpoint.
    class Env(gym.Env):
        observation_space = policy_space('v2')
        action_space = gym.spaces.Discrete(11)
    cls = SharedMovePolicy if architecture == 'shared-moves' else 'MultiInputPolicy'
    kwargs = {} if architecture == 'shared-moves' else dict(features_extractor_class=BattleFeatures)
    model = MaskablePPO(cls, Env(), n_steps=2, batch_size=2, policy_kwargs=kwargs)
    path = tmp_path/'model.zip'
    model.save(path)
    loaded = MaskablePPO.load(path)
    opponent = PPOOpponent(loaded)
    for report in pair_inspections(enrich(snapshots), [1,2]):
        action = opponent.choose(report)
        assert linked_observation(report)['action_mask'][action]
