"""Integration tests against the real pinned simulator (no ROM or server needed)."""
import json
from pathlib import Path
import random
import shutil
import subprocess
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from firered_rl.showdown_env import ShowdownBattleSession, ShowdownFactory, validate_showdown
from firered_rl.linked_reward import LinkedRewardConfig
from firered_rl.training_teams import TrainingTeams
from firered_rl.turn_env import observation_space
pytest.importorskip("stable_baselines3")
from firered_rl.ppo_features import policy_observation, policy_space

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
    reason="install simulator with npm --prefix sim/showdown ci --omit=optional")


def mon(species, moves, **kwargs):
    return dict(species=species, level=50, moves=moves, **kwargs)


@pytest.fixture
def session(tmp_path):
    args = SimpleNamespace(seed=3, node="node", command_timeout=10)
    paths = [tmp_path / f"team-{i}.json" for i in range(2)]
    factory = ShowdownFactory(ROOT, tmp_path / "pair", [55355, 55356], lambda: paths, args, Event())
    s = ShowdownBattleSession(factory, reward_config=LinkedRewardConfig.load(
        ROOT / "config/rewards/damage_knockout_win.json"), max_steps=100,
        readable_log=tmp_path / "pair/battle-readable.log", battle_log=tmp_path / "pair/battle.log")
    def reset(first, second):
        for p, party in zip(paths, [first, second]):
            p.write_text(json.dumps(dict(name=p.stem, party=party)))
        return s.reset_pair([123, 456])
    yield s, reset
    s.close()


def test_observation_ids_stats_status_and_original_switch_slots(session):
    s, reset = session
    a = [mon("Charizard", ["Flamethrower"]), mon("Treecko", ["Absorb", "Quick Attack"], held_item="Leftovers")]
    b = [mon("Blastoise", ["Surf"]), mon("Lapras", ["Ice Beam", "Rest", "Sleep Talk"])]
    obs = reset(a, b)[0][0]
    assert observation_space().contains(obs)
    assert policy_space().contains(policy_observation(obs))
    assert obs["party"][0, 3] == 153
    assert obs["party"][1, 1] == 277  # FireRed internal Treecko ID, not National Dex 252
    assert obs["party"][1, 8] == 200  # Gen III Leftovers ID, not modern ID 234
    assert obs["moves"][0, 0].tolist() == [53, 10, 15, 15]  # no PP Ups
    assert obs["moves"][1, 2:].sum() == 0
    assert obs["opponent_moves"][1, 2, 0] == 214  # all opponent bench moves
    for choices, slots in [([5,5],[2,2]), ([4,4],[1,1]), ([5,5],[2,2])]:
        result = s.step_pair(choices)
        assert [r[0]["active"][0, 8] for r in result] == slots
        assert result[0][0]["party"][:,1].tolist() == obs["party"][:,1].tolist()
    with pytest.raises(ValueError, match="illegal"):
        s.step_pair([5,5])  # currently active slots never legal switches


def test_baton_pass_replacement_does_not_charge_a_second_turn(session):
    s, reset = session
    reset([mon("Ninjask", ["Baton Pass"]), mon("Vaporeon", ["Splash"])], [mon("Magikarp", ["Splash"])])
    first = s.step_pair([0,0])
    assert first[0][0]["context"][:2].tolist() == [0,1]
    assert np.flatnonzero(first[0][0]["action_mask"]).tolist() == [5]
    assert np.flatnonzero(first[1][0]["action_mask"]).tolist() == [10]
    assert first[0][4]["reward_components"]["turn_penalty"] == -50
    second = s.step_pair([5,10])
    assert second[0][0]["active"][0,8] == 2
    assert second[0][4]["reward_components"]["turn_penalty"] == 0


@pytest.mark.parametrize("move", ["Fly", "Outrage", "Hyper Beam"])
def test_forced_move_continuations_and_recharge(session, move):
    s, reset = session
    reset([mon("Charizard", [move,"Splash"])], [mon("Blissey", ["Splash"])])
    after = s.step_pair([0,0])
    assert np.flatnonzero(after[0][0]["action_mask"]).tolist() == [10]
    s.step_pair([10,0])  # consume the mandatory action without a new move choice


def test_terminal_rewards_and_reproducible_resets(session):
    s, reset = session
    first = [mon("Magikarp", ["Splash"])]
    first[0]["level"] = 1
    second = [mon("Blastoise", ["Surf"])]
    def battle():
        reset(first, second)
        return s.step_pair([0,0])
    a, b = battle(), battle()
    assert a[0][2:4] == (True, False)
    assert a[0][4]["outcome"] == "lost" and a[1][4]["outcome"] == "won"
    assert a[1][1] == 5550  # 100 damage + 500 KO + 5000 win - 50 turn
    assert a[0][1] == -5050
    for x,y in zip(a,b):
        for field in x[0]:
            np.testing.assert_array_equal(x[0][field], y[0][field])
    text = s.battle_log.read_text()
    assert "used Surf!" in text and "damage; HP" in text and "P1 Magikarp" in text


def test_trapping_encore_disable_and_struggle(session):
    s, reset = session
    reset([mon("Magikarp", ["Splash"]), mon("Lapras", ["Splash"])], [mon("Wobbuffet", ["Splash"])])
    assert s.reports[0]["action_mask"][5] == 0  # Shadow Tag; no ordinary switch
    reset([mon("Alakazam", ["Encore","Splash"])], [mon("Blastoise", ["Splash","Surf"])])
    s.step_pair([1,0])
    after = s.step_pair([0,0])
    assert after[1][0]["action_mask"][:4].tolist() == [1,0,0,0]
    reset([mon("Alakazam", ["Disable","Splash"])], [mon("Blastoise", ["Splash","Surf"])])
    s.step_pair([1,0])
    for _ in range(10):  # Disable can miss in Gen III; stop once this seed lands it.
        after = s.step_pair([0,0])
        if after[1][0]["action_mask"][0] == 0:
            break
    assert after[1][0]["action_mask"][:4].tolist() == [0,1,0,0]
    reset([mon("Magikarp", ["Splash"])], [mon("Magikarp", ["Splash"])])
    for _ in range(40):
        after = s.step_pair([0,0])
    assert after[0][0]["moves"][0,0,2] == 0
    assert np.flatnonzero(after[0][0]["action_mask"]).tolist() == [0]
    s.step_pair([0,0])
    assert "Struggle" in s.battle_log.read_text()


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('cause', ['exhausted', 'disabled'])
def test_move_audit_checks_struggle_with_a_healthy_bench(session, tmp_path, monkeypatch, side, cause):
    from firered_rl.move_order_audit import MoveOrderAudit

    s, reset = session
    s.factory.args.move_order_audit = True
    s.observation_version = 'v2'
    own = [mon('Magikarp', ['Splash']), mon('Lapras', ['Splash'])]
    foe = [mon('Magikarp', ['Splash'])] if cause == 'exhausted' else [mon('Alakazam', ['Disable', 'Splash'])]
    reset(*(own, foe) if side == 0 else (foe, own))

    def step(own_action, foe_action):
        return s.step_pair([own_action, foe_action] if side == 0 else [foe_action, own_action])

    if cause == 'exhausted':
        for _ in range(40):
            result = step(0, 0)
        assert result[side][0]['moves'][0, 0, 2] == 0
    else:
        step(0, 1)  # Establish Splash as the last move before Disable.
        for _ in range(10):
            result = step(0, 0)
            active = next(m for m in s.reports[side]['battle']['battlers'] if m['side'] == 'player')
            # Request evidence on the *next* submitted turn establishes Disable;
            # the PP counter ceases to decrease once Splash cannot execute.
            if result[side][0]['moves'][0, 0, 2] == 39:
                break
        assert active['moves'][0]['current_pp'] > 0

    obs = result[side][0]
    assert np.flatnonzero(obs['action_mask']).tolist() == [0, 5]
    assert obs['moves'][0, 0, 0] == 150  # Stored move is still Splash, not Struggle.
    monkeypatch.setattr('firered_rl.move_order_audit.probabilities',
                        lambda model, observations: np.tile(np.eye(11)[0], (len(observations), 1)))
    audit = MoveOrderAudit(None, tmp_path, SimpleNamespace(log_max_mb=2, log_backups=1))
    case = dict(episode=1, candidate_player=side+1)
    audit(case, 1, obs, 0)
    result = step(0, 0 if cause == 'exhausted' else 1)
    evidence = result[side][4]['audit_choice']
    assert evidence == dict(action=0, command='move 1', requested_move='struggle', resolved_move='struggle')
    assert result[1-side][4]['audit_choice']['action'] == (0 if cause == 'exhausted' else 1)
    audit.after_step(case, 1, result[side][4])
    audit.finish()
    report = json.loads((tmp_path/'move-order-audit.json').read_text())
    assert report['counts']['execution_struggle_fallback'] == 1
    assert report['counts'].get('execution_review_mismatch', 0) == 0
    assert report['examples'][0]['submitted_choice'] == evidence


def test_random_profiles_and_all_templates(tmp_path):
    # Validates every template, including names that changed between Gen III and Showdown.
    code = """
      const fs=require('fs'), path=require('path'), {sets}=require('./sim/showdown/runner.cjs');
      let count=0;
      function visit(p){for(const f of fs.readdirSync(p,{withFileTypes:true})){
        const name=path.join(p,f.name); if(f.isDirectory())visit(name);
        else if(name.endsWith('.json')) {sets(JSON.parse(fs.readFileSync(name)));count++;}
      }}
      visit('config/teams/profiles'); if(count!==2700)throw Error('missing teams');
    """
    subprocess.run(["node", "-e", code], cwd=ROOT, check=True, capture_output=True, text=True)
    source = TrainingTeams(profile="mixed", team_size=3, profile_root=ROOT / "config/teams/profiles")
    assert len(source.files) == 450
    choose = source.for_pair(0)
    factory = ShowdownFactory(ROOT, tmp_path, [55355,55356], choose,
        SimpleNamespace(seed=1,node="node",command_timeout=10), Event())
    s = ShowdownBattleSession(factory, max_steps=40)
    rng = random.Random(13)
    try:
        for _ in range(12):
            observations = [row[0] for row in s.reset_pair()]
            pid = factory.process.pid
            for _ in range(40):
                choices = [rng.choice(np.flatnonzero(o["action_mask"]).tolist()) for o in observations]
                results = s.step_pair(choices)
                observations = [row[0] for row in results]
                assert all(observation_space().contains(o) for o in observations)
                if results[0][2] or results[0][3]:
                    break
            assert factory.process.pid == pid
    finally:
        s.close()


def test_profile_selection_and_restrictions():
    source = TrainingTeams(profile="mixed", profile_root=ROOT / "config/teams/profiles", seed=4)
    assert len(source.files) == 2700
    a, b = source.for_pair(3), source.for_pair(3)
    assert [a() for _ in range(10)] == [b() for _ in range(10)]
    with pytest.raises(ValueError, match="requires --team-profile"):
        TrainingTeams(teams=source.files[:2], team_size=3)
    validate_showdown(ROOT)
