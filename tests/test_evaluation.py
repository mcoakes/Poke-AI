import importlib.util
import json
from pathlib import Path
import shutil
import sys

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
from sb3_contrib import MaskablePPO
from firered_rl.evaluation import create_suite, validate_suite
from firered_rl.ppo_features import BattleFeatures, policy_space
from firered_rl.training_teams import TrainingTeams

ROOT = Path(__file__).resolve().parents[1]


def load_cli(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    spec = importlib.util.spec_from_file_location("evaluation_cli", ROOT / "mgba/evaluate_ppo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_six_type_profile_has_learnable_matching_moves_and_mirrors():
    from argparse import Namespace
    args = Namespace(states=None, state_dir=None, teams=None, team_dir=None, training_profile="six_types",
                     team_profile=None, team_size=None, seed=71, mirror_teams=True)
    source = TrainingTeams.from_args(args, ROOT)
    a, b = source.for_pair(0)()
    assert a == b
    party = json.loads(a.read_text())["party"]
    data = json.loads((ROOT / "config/generation/firered_us_v1_0.json").read_text())
    assert len({tuple(data['species'][p['species']]['types']) for p in party}) == 6
    for mon in party:
        species = data['species'][mon['species']]
        assert len(set(mon['moves'])) == 4
        assert set(mon['moves']) <= set(species['moves'])
        assert all(data['moves'][m]['type'] in species['types'] for m in mon['moves'])


def test_suite_is_seeded_and_detects_changed_assets(tmp_path):
    team = tmp_path / "team.json"
    team.write_text('{"party":[{"species":"Blastoise","level":50,"moves":["Surf"]}]}')
    source = TrainingTeams(teams=[team, team], seed=10001)
    suite = create_suite(source, backend="showdown", episodes=4, seed=9, swap_sides=True, max_steps=200)
    assert suite == create_suite(source, backend="showdown", episodes=4, seed=9, swap_sides=True, max_steps=200)
    assert [c['candidate_player'] for c in suite['cases']] == [1, 2, 1, 2]
    assert validate_suite(suite) is suite
    next(iter(suite['assets'].values()))['team']['party'][0]['level'] = 10
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_suite(suite)


@pytest.mark.skipif(not shutil.which("node") or not (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(),
                    reason="Showdown not installed")
@pytest.mark.parametrize("mode", ["thread", "process"])
def test_real_evaluation_side_balance_replay_and_no_model_updates(tmp_path, monkeypatch, mode):
    import torch
    torch.set_num_threads(1)
    class ShapeEnv(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)
    model = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2,
                        policy_kwargs=dict(features_extractor_class=BattleFeatures))
    checkpoint = tmp_path / "agent.zip"
    model.save(checkpoint)
    original = checkpoint.read_bytes()
    def no_training(*args, **kwargs):
        raise AssertionError("evaluation must not train or save a model")
    monkeypatch.setattr(MaskablePPO, "learn", no_training)
    monkeypatch.setattr(MaskablePPO, "save", no_training)
    teams = [tmp_path / f"team{i}.json" for i in range(2)]
    for path, species, move, level in zip(teams, ["Magikarp", "Blastoise"], ["Splash", "Surf"], [1, 50]):
        path.write_text(json.dumps(dict(party=[dict(species=species, moves=[move], level=level)])))
    cli = load_cli(monkeypatch)
    run = tmp_path / "evaluation"
    common = ["evaluate_ppo", "--checkpoint", str(checkpoint), "--opponent-checkpoint", str(checkpoint),
              "--showdown-workers", mode]
    monkeypatch.setattr(sys, "argv", [*common, "--teams", *map(str, teams), "--episodes", "6", "--max-steps", "4",
                                      "--environments", "4", "--output-dir", str(run)])
    assert cli.main() == 0
    summary = json.loads((run / "summary.json").read_text())
    assert summary['candidate_wins'] == summary['opponent_wins'] == 3
    assert summary['candidate_win_rate'] == .5 and summary['truncated'] == 0
    assert summary['completed_battles'] == 6
    assert checkpoint.read_bytes() == original
    # Different concurrency and source files gone: embedded teams/seeds still replay exactly.
    for path in teams:
        path.unlink()
    replay = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", [*common, "--suite", str(run / "suite.json"), "--environments", "2",
                                      "--output-dir", str(replay)])
    assert cli.main() == 0
    read_rows = lambda p: sorted([json.loads(s) for s in (p / 'episodes.jsonl').read_text().splitlines()], key=lambda r:r['episode'])
    assert read_rows(run) == read_rows(replay)
    assert checkpoint.read_bytes() == original


def test_truncations_are_separate_from_draws():
    from firered_rl.evaluation import summary
    row = dict(candidate_player=1, candidate_outcome='truncated', candidate_reward=0,
               opponent_reward=0, decisions=3, truncated=True)
    s = summary([row], requested=1, elapsed=1, status='completed', suite_hash='test')
    assert s['truncated'] == 1 and s['draws'] == s['completed_battles'] == 0
    assert s['candidate_win_rate'] is None
