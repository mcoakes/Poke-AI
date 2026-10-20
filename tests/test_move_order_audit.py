import json
from types import SimpleNamespace

import numpy as np
import pytest

from firered_rl.move_order_audit import permute_moves, MoveOrderAudit, execution_check
from firered_rl.rule_opponent import RuleBasedOpponent
from test_rule_opponent import observation, move, ROOT


def fixture():
    policy = RuleBasedOpponent(ROOT)
    obs = observation(policy)
    for i, name in enumerate(('Tackle','Surf','Toxic','Recover')):
        move(policy, obs, i, name, pp=i+1)
    return obs


def test_permutation_moves_every_related_field_and_keeps_input():
    obs = fixture()
    obs['action_mask'][1] = 0
    before = {k:v.copy() for k,v in obs.items()}
    order = [2,0,3,1]
    result = permute_moves(obs,order)
    assert np.array_equal(result['moves'][0],obs['moves'][0,order])
    assert np.array_equal(result['v2_move_details'][0,0],obs['v2_move_details'][0,0,order])
    assert np.array_equal(result['action_mask'][:4],obs['action_mask'][order])
    assert all(np.array_equal(v,obs[k]) for k,v in before.items())
    assert np.array_equal(result['opponent_moves'],obs['opponent_moves'])
    with pytest.raises(ValueError):
        permute_moves(obs,[0,0,1,2])


@pytest.mark.parametrize('follows_moves',[True,False])
def test_distinguishes_move_identity_from_slot_and_probability_alignment(tmp_path, monkeypatch, follows_moves):
    obs = fixture()
    wanted = int(obs['moves'][0,0,0])
    def probs(model, observations):
        result = np.zeros((len(observations),11))
        for i,o in enumerate(observations):
            action = np.flatnonzero(o['moves'][0,:,0] == wanted)[0] if follows_moves else 0
            result[i,action] = 1
        return result
    monkeypatch.setattr('firered_rl.move_order_audit.probabilities',probs)
    audit = MoveOrderAudit(None,tmp_path,SimpleNamespace(log_max_mb=2,log_backups=1))
    audit(dict(episode=1,candidate_player=1),1,obs,0)
    audit.after_step(dict(episode=1),1,dict(audit_protocol=['|move|p1a: PIKACHU|Tackle|p2a: MEW']))
    audit.finish()
    data=json.loads((tmp_path/'move-order-audit.json').read_text())
    counts=data['counts']
    assert counts.get('follows_original_action',0)==(3 if follows_moves else 0)
    assert counts.get('stays_in_original_slot',0)==(0 if follows_moves else 3)
    assert data['mean_aligned_total_variation']==(0 if follows_moves else 1)
    assert counts['execution_matched']==1


def test_execution_check_distinguishes_absence_and_mismatch():
    assert execution_check(['|move|p1a: Me|High Jump Kick|p2a: Foe'],1,'Hi Jump Kick')[0]=='matched'
    assert execution_check(['|move|p2a: Enemy|Surf|p1a: Me'],1,'Tackle')[0]=='not_observed'
    assert execution_check(['|move|p1a: Me|Metronome|p2a: Foe','|move|p1a: Me|Surf|p2a: Foe'],1,'Metronome')[0]=='matched'
    assert execution_check(['|move|p1a: Me|Surf|p2a: Foe'],1,'Tackle')[0]=='review_mismatch'


def test_struggle_requires_matching_native_request_and_queued_action():
    lines = ['|move|p1a: Me|Struggle|p2a: Foe']
    choice = dict(action=0, command='move 1', requested_move='struggle', resolved_move='struggle')
    assert execution_check(lines, 1, 'Rock Slide', action=0, choice=choice)[0] == 'struggle_fallback'
    assert execution_check(lines, 1, 'Rock Slide')[0] == 'review_mismatch'
    for changed in (dict(action=1), dict(requested_move='rockslide'),
                    dict(resolved_move='rockslide'), dict(command='switch 2')):
        assert execution_check(lines, 1, 'Rock Slide', action=0,
                               choice={**choice, **changed})[0] == 'review_mismatch'
    assert execution_check(lines, 2, 'Earthquake', action=0, choice=choice)[0] == 'not_observed'
    assert execution_check(['|move|p1a: Me|Surf|p2a: Foe'], 1, 'Rock Slide',
                           action=0, choice=choice)[0] == 'review_mismatch'
