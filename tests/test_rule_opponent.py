"""Tactical fixtures plus real Showdown CLI wiring. No claims about playing strength."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
from firered_rl.rule_opponent import RuleBasedOpponent

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy():
    return RuleBasedOpponent(ROOT)


def observation(policy):
    obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in policy.observation_space.spaces.items()}
    for side in range(2):
        obs["v2_party"][side, 0] = [1, 50, 150, 150, 0, 0, 0, 0, 0, 0]
        obs["v2_active"][side] = [150, 150, 0, 0]
        obs["v2_stats"][side, 0] = 100
        obs["v2_active_stats"][side] = 100
        obs["active"][side] = [1, 25, 50, 150, 150, 0, 0, 0, 1 if side == 0 else 0, 0]
    obs["opponent_active_index"][0] = 1
    return obs


def move(policy, obs, slot, name, *, side=0, party=0, accuracy=100, pp=10):
    entry = next(x for x in policy.moves.values() if x["name"] == name)
    types = {"Normal": 0, "Fighting": 1, "Flying": 2, "Poison": 3, "Ground": 4, "Rock": 5,
             "Bug": 6, "Ghost": 7, "Steel": 8, "Fire": 10, "Water": 11, "Grass": 12,
             "Electric": 13, "Psychic": 14, "Ice": 15, "Dragon": 16, "Dark": 17}
    kind = types[entry["type"]]
    obs["moves" if side == 0 else "opponent_moves"][party, slot] = [entry["id"], kind, pp, 10]
    obs["v2_move_details"][side, party, slot] = [entry["power"], accuracy, int(accuracy == 0),
        {"physical": 0, "special": 1, "status": 2}[entry["category"]], 0, kind]
    if side == 0 and party == 0:
        obs["action_mask"][slot] = 1


def test_damage_types_accuracy_immunities_and_ko(policy):
    obs = observation(policy)
    move(policy, obs, 0, "Thunder", accuracy=70)
    move(policy, obs, 1, "Thunderbolt")
    obs["active"][1, 5:7] = 11
    assert policy.choose(obs)[0] == 1
    obs["v2_active"][1, 3] = 10  # Volt Absorb.
    move(policy, obs, 2, "Tackle")
    assert policy.choose(obs)[0] == 2
    obs["v2_active"][1, 3] = 0
    obs["active"][1, 5:7] = 4
    assert policy.choose(obs)[0] == 2
    obs["active"][1, 5:7] = 11
    obs["v2_active"][1, 0] = 15
    assert policy.choose(obs)[0] == 1  # Reliable KO beats inaccurate high power.


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_sparse_moves_masked_moves_and_input_unchanged(policy, count):
    obs = observation(policy)
    for i in range(count):
        move(policy, obs, i, "Tackle")
    obs["action_mask"][:count-1] = 0
    before = {k: x.copy() for k, x in obs.items()}
    assert policy.choose(obs)[0] == count-1
    assert all(np.array_equal(before[k], obs[k]) for k in obs)
    obs["action_mask"][:] = 0
    obs["action_mask"][10] = 1
    assert policy.choose(obs)[0] == 10
    obs["action_mask"][:] = 0
    with pytest.raises(ValueError, match="no legal"):
        policy.choose(obs)


def test_status_recovery_setup_and_no_repeated_setup(policy):
    obs = observation(policy)
    move(policy, obs, 0, "Toxic")
    move(policy, obs, 1, "Tackle")
    assert policy.choose(obs)[0] == 0
    obs["status_bits"][7, 3] = 1
    assert policy.choose(obs)[0] == 1
    obs["status_bits"][:] = 0
    obs["active"][1, 5:7] = 8
    assert policy.choose(obs)[0] == 1  # Steel cannot be poisoned.
    obs["active"][1, 5:7] = 0
    obs["v2_volatile_bits"][1, 24] = 1
    assert policy.choose(obs)[0] == 1
    move(policy, obs, 0, "Recover")
    assert policy.choose(obs)[0] == 1  # Full HP.
    obs["v2_active"][0, 0] = 40
    assert policy.choose(obs)[0] == 0
    obs["v2_active"][0, 0] = 150
    move(policy, obs, 0, "Swords Dance")
    assert policy.choose(obs)[0] == 0
    obs["v2_boosts"][0, 0] = 2
    assert policy.choose(obs)[0] == 1


def test_switch_cost_forced_replacement_and_trapping(policy):
    obs = observation(policy)
    move(policy, obs, 0, "Tackle")
    move(policy, obs, 0, "Thunderbolt", side=1)
    obs["active"][0, 5:7] = [11, 2]
    obs["v2_active"][0, 0] = 20
    for party, kind in ((1, 4), (2, 11)):
        obs["v2_party"][0, party] = [1, 50, 150, 150, kind, kind, 0, 0, 0, 0]
        obs["v2_stats"][0, party] = 100
        move(policy, obs, 0, "Earthquake", party=party)
        obs["action_mask"][4+party] = 1
    assert policy.choose(obs)[0] == 5  # Immune Ground switch, not vulnerable Water.
    obs["action_mask"][5:7] = 0
    assert policy.choose(obs)[0] == 0  # Trapped: never invent switch legality.
    obs["action_mask"][:] = 0
    obs["action_mask"][5:7] = 1
    obs["context"][1] = 1
    assert policy.choose(obs)[0] == 5
    # Equivalent alternatives cost a turn, so staying is preferred when safe.
    obs["context"][1] = 0
    obs["action_mask"][0] = 1
    obs["v2_active"][0, 0] = 150
    obs["active"][0, 5:7] = 4
    move(policy, obs, 0, "Earthquake")
    assert policy.choose(obs)[0] == 0


def test_batched_deterministic_decisions_and_bounded_log(policy, tmp_path):
    obs = observation(policy)
    move(policy, obs, 0, "Tackle")
    move(policy, obs, 1, "Surf")
    batch = {k: np.stack([v, v]) for k, v in obs.items()}
    args = SimpleNamespace(rule_log=True, log_max_mb=.002, log_backups=1)
    policy.configure_log(tmp_path, args)
    for _ in range(15):
        result, _ = policy.predict(batch, action_masks=batch["action_mask"])
        assert result.tolist() == [1, 1]
    files = list(tmp_path.glob("rule-decisions.jsonl*"))
    assert len(files) == 2 and all(p.stat().st_size <= 2097 for p in files)
    assert all(json.loads(line)["candidates"] for p in files for line in p.read_text().splitlines())


def test_real_training_continue_and_evaluation(tmp_path, monkeypatch):
    import importlib
    import shutil
    import sys
    if not shutil.which("node") or not (ROOT / "sim/showdown/node_modules/pokemon-showdown").exists():
        pytest.skip("Showdown not installed")
    monkeypatch.syspath_prepend(str(ROOT / "mgba"))
    train = importlib.import_module("train_ppo")
    evaluation = importlib.import_module("evaluate_ppo")
    teams = []
    for side, (species, moves) in enumerate((("Pikachu", ["Thunderbolt", "Quick Attack"]), ("Squirtle", ["Surf", "Tackle"]))):
        path = tmp_path / f"team-{side}.json"
        path.write_text(json.dumps(dict(party=[dict(species=species, level=50, moves=moves)])))
        teams.append(str(path))
    run = tmp_path / "train"
    def invoke(module, flags):
        monkeypatch.setattr(sys, "argv", ["cli", *map(str, flags)])
        return module.main()
    assert invoke(train, ["--output-dir", run, "--backend", "showdown", "--rule-based-opponent", "--rule-log",
                         "--teams", *teams, "--observation-version", "v2", "--environments", 2,
                         "--rollout-steps", 4, "--batch-size", 4, "--epochs", 1, "--max-steps", 5,
                         "--total-timesteps", 8, "--showdown-workers", "process", "--no-step-logs"]) == 0
    config = json.loads((run / "config.json").read_text())
    assert config["learning_environments"] == 1 and config["rule_opponent"]["version"] == "gen3-rules-v1"
    assert not config.get("frozen_opponent")
    assert invoke(train, ["--output-dir", run, "--if-exists", "continue", "--total-timesteps", 4]) == 0
    from sb3_contrib import MaskablePPO
    checkpoint = run / "checkpoints/final.zip"
    assert MaskablePPO.load(checkpoint).num_timesteps == 12
    output = tmp_path / "eval"
    common = ["--checkpoint", checkpoint, "--environments", 2, "--showdown-workers", "process"]
    assert invoke(evaluation, [*common, "--rule-based-opponent", "--rule-log", "--teams", *teams,
                              "--episodes", 4, "--max-steps", 5, "--output-dir", output]) == 0
    replay = tmp_path / "replay"
    assert invoke(evaluation, [*common, "--suite", output / "suite.json", "--output-dir", replay]) == 0
    first = json.loads((output / "summary.json").read_text())
    second = json.loads((replay / "summary.json").read_text())
    for key in ("candidate_wins", "opponent_wins", "draws", "truncated", "suite_sha256"):
        assert first[key] == second[key]
    assert invoke(train, ["--output-dir", run, "--if-exists", "continue", "--total-timesteps", 4,
                          "--eval-suite", output / "suite.json", "--eval-every", 4]) == 0
    assert (run / "checkpoints/best.zip").is_file()
    assert "rule-based player 2" in (run / "config.json").read_text()
    suite = json.loads((output / "suite.json").read_text())
    suite["rule_opponent"]["sha256"] = "0" * 64
    changed = tmp_path / "changed-suite.json"
    changed.write_text(json.dumps(suite))
    with pytest.raises(SystemExit) as error:
        invoke(evaluation, [*common, "--suite", changed, "--output-dir", tmp_path / "bad"])
    assert error.value.code == 2 and not (tmp_path / "bad").exists()
