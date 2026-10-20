import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from firered_rl.curriculum import (DEFAULT_RULES, digest, evaluation_rates, load_curriculum,
    promotion_report, read_json, stage_suites, write_json)
from firered_rl.evaluation import summary

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/curricula/type_progression/curriculum.json"


def cli_module():
    spec = importlib.util.spec_from_file_location("curriculum_cli", ROOT / "mgba/run_curriculum.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_stage_teams_and_curriculum_suites():
    manifest, paths = load_curriculum(MANIFEST)
    assert len(manifest["stages"]) == 10 and len(paths) == 11
    assert paths[0].read_bytes() == (ROOT / "config/training_profiles/six_types/teams/six_types.json").read_bytes()
    catalog = read_json(ROOT / "config/generation/firered_us_v1_0.json")
    for path in paths:
        for mon in read_json(path)["party"]:
            assert len(set(mon["moves"])) == 4
            assert set(mon["moves"]) <= set(catalog["species"][mon["species"]]["moves"])
            assert mon["held_item"] == "None" or mon["held_item"] in catalog["items"]
    for stage in range(1, 11):
        current, expanded = stage_suites(paths, stage=stage, episodes=1000, seed=10, max_steps=200)
        assert (current, expanded) == stage_suites(paths, stage=stage, episodes=1000, seed=10, max_steps=200)
        incoming = digest(dict(team=read_json(paths[stage])))
        assert incoming not in current["assets"]
        assert len(current["assets"]) == stage
        assert expanded["cases"][:500] == current["cases"][:500]
        assert len(expanded["cases"]) == 1000
        for a, b in zip(expanded["cases"][500::2], expanded["cases"][501::2]):
            assert incoming in a["assets"] and a["assets"] == b["assets"]
            assert (a["candidate_player"], b["candidate_player"]) == (1, 2)
            assert a["cohort"] == b["cohort"] == "incoming"
            assert a["seed"] == b["seed"]


def rates(win, trunc=0):
    return dict(win_rate=win, truncation_rate=trunc, attempts=1000, completed=1000, wins=round(win*1000))


def test_promotion_needs_new_team_performance_and_preserves_previous_skill():
    before = dict(all=rates(.65), retained=rates(.65))
    expanded_before = dict(all=rates(.57), retained=rates(.65), incoming=rates(.49))
    current = dict(all=rates(.70), retained=rates(.70))
    expanded = dict(all=rates(.63), retained=rates(.70), incoming=rates(.56))
    assert promotion_report(before, expanded_before, current, expanded, DEFAULT_RULES)["status"] == "promote"
    expanded["incoming"] = rates(.49)
    assert promotion_report(before, expanded_before, current, expanded, DEFAULT_RULES)["status"] == "hold"
    expanded["incoming"] = rates(.56)
    current["all"] = rates(.61)  # Above absolute 60%, but four points below anchor.
    assert promotion_report(before, expanded_before, current, expanded, DEFAULT_RULES)["status"] == "hold"
    current["all"] = rates(.70)
    expanded["incoming"] = rates(.90, .02)  # Truncation must not inflate the win gate.
    assert promotion_report(before, expanded_before, current, expanded, DEFAULT_RULES)["status"] == "hold"


def completed_evaluation(path, suite, candidate, opponent, *, lose_incoming=False):
    path.mkdir(parents=True)
    cli = cli_module()
    write_json(path / "suite.json", suite)
    write_json(path / "config.json", dict(suite_sha256=digest(suite), models=dict(
        candidate=dict(sha256=cli.sha256(candidate)), opponent=dict(sha256=cli.sha256(opponent)))))
    rows = [dict(case, candidate_outcome="lost" if lose_incoming and case["cohort"] == "incoming" else "won",
                 terminated=True, truncated=False, candidate_reward=0, opponent_reward=0, decisions=1)
            for case in suite["cases"]]
    (path / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(path / "summary.json", summary(rows, requested=len(rows), elapsed=1,
               status="completed", suite_hash=digest(suite)))


def test_missing_case_cannot_pass_even_when_summary_claims_completed(tmp_path):
    _, paths = load_curriculum(MANIFEST)
    suite, _ = stage_suites(paths, stage=1, episodes=4, seed=9, max_steps=200)
    checkpoint = tmp_path / "agent.zip"
    checkpoint.write_bytes(b"test fixture")
    directory = tmp_path / "eval"
    completed_evaluation(directory, suite, checkpoint, checkpoint)
    assert evaluation_rates(directory, suite)["all"]["win_rate"] == 1
    rows = (directory / "episodes.jsonl").read_text().splitlines()
    (directory / "episodes.jsonl").write_text("\n".join(rows[:-1]))
    with pytest.raises(ValueError, match="missing cases"):
        evaluation_rates(directory, suite)


@pytest.mark.parametrize("opponent_mode", ["fixed", "previous-stage"])
@pytest.mark.parametrize("switch_chance", [0, .3])
def test_pipeline_pins_inputs_and_retries_then_adds_exactly_one_team(tmp_path, opponent_mode, switch_chance):
    cli = cli_module()
    source = tmp_path / "source.zip"
    opponent = tmp_path / "opponent.zip"
    source.write_bytes(b"initial policy")
    opponent.write_bytes(b"reference policy")
    run = tmp_path / "campaign"
    run.mkdir()
    args = argparse.Namespace(resume=source, opponent_checkpoint=opponent, curriculum=MANIFEST,
        eval_episodes=4, max_steps=200, promotion_rules=None, reward_config=None, seed=7,
        opponent_mode=opponent_mode, opponent_switch_chance=switch_chance)
    config = cli.initialize(args, run)
    assert config["opponent_switch"] == dict(chance=switch_chance, cooldown=2)
    # A live training run replacing an external latest.zip cannot change this run.
    source.write_bytes(b"new unrelated training update")
    opponent.write_bytes(b"new unrelated opponent")
    trained, evaluations = [], []
    lose = [True]

    def execute(command, root):
        values = command[2:]
        get = lambda flag: Path(values[values.index(flag)+1])
        output = get("--output-dir")
        stage = int(output.relative_to(run).parts[0].split("-")[1])
        expected = (Path(read_json(run / f"stage-{stage-1:02d}/result.json")["candidate"]).read_bytes()
                    if stage > 1 and opponent_mode == "previous-stage" else b"reference policy")
        assert get("--opponent-checkpoint") == run / f"stage-{stage:02d}/opponent.zip"
        assert get("--opponent-checkpoint").read_bytes() == expected
        assert "--no-step-logs" in command
        assert str(get("--log-max-mb")) == "10"
        assert str(get("--log-backups")) == "2"
        assert str(get("--keep-episode-logs")) == "2"
        if Path(command[1]).name == "train_ppo.py":
            assert str(get("--latest-every-updates")) == "10"
            assert str(get("--checkpoint-every")) == "0"
            assert str(get("--keep-checkpoints")) == "2"
            assert ("--opponent-switch-chance" in command) == bool(switch_chance)
            if switch_chance:
                assert float(str(get("--opponent-switch-chance"))) == switch_chance
            pool = sorted(get("--team-dir").glob("*.json"))
            resume = get("--resume")
            trained.append((pool, resume, resume.read_bytes()))
            (output / "checkpoints").mkdir(parents=True)
            (output / "checkpoints/final.zip").write_bytes(resume.read_bytes() + b" updated")
            write_json(output / "summary.json", dict(status="completed"))
        else:
            assert "--latest-every-updates" not in command and "--checkpoint-every" not in command
            evaluations.append(output)
            assert "--opponent-switch-chance" not in command
            completed_evaluation(output, read_json(get("--suite")), get("--checkpoint"),
                get("--opponent-checkpoint"), lose_incoming=lose[0] and "attempt-001" in str(output))

    execution = dict(cli.EXECUTION, latest_every_updates=10, checkpoint_every=0)
    campaign = cli.Campaign(run, config, execution, execute=execute)
    assert campaign.next_stage() == 1
    held = campaign.run_stage(1)
    assert held["status"] == "hold" and campaign.next_stage() == 1
    assert len(trained[0][0]) == 1
    # An interruption after evaluation but before reporting must not train again.
    (run / "stage-01/attempt-001/report.json").unlink()
    (run / "stage-01/result.json").unlink()
    assert campaign.run_stage(1)["status"] == "hold"
    assert len(trained) == 1 and len(evaluations) == 4
    lose[0] = False
    accepted = campaign.run_stage(1)
    assert accepted["status"] == "promote" and campaign.next_stage() == 2
    assert trained[1][1] == Path(held["candidate"])
    assert len(evaluations) == 6  # Two baselines reused; only the two after evaluations repeat.
    lose[0] = True
    second = campaign.run_stage(2)
    assert second["status"] == "hold"
    assert len(trained[2][0]) == 2
    assert trained[2][2] == Path(accepted["candidate"]).read_bytes()
    assert not any("03_familiar" in str(p) for p in trained[2][0])
    assert (run / "inputs/opponent.zip").read_bytes() == b"reference policy"
    assert (run / "stage-02/report.md").is_file()
    stage_opponent = (run / "stage-02/opponent.zip").read_bytes()
    assert stage_opponent == (Path(accepted["candidate"]).read_bytes() if opponent_mode == "previous-stage"
                              else b"reference policy")
    assert second["opponent"]["source_stage"] == (1 if opponent_mode == "previous-stage" else None)
    # A fresh runner instance must keep the stage-2 opponent when resuming a held attempt.
    lose[0] = False
    campaign = cli.Campaign(run, read_json(run / "campaign.json"), execution, execute=execute)
    retried = campaign.run_stage(2)
    assert retried["status"] == "promote"
    assert trained[3][1] == Path(second["candidate"])
    assert retried["opponent"] == second["opponent"]
    assert retried["evaluations"]["before_current"] == second["evaluations"]["before_current"]
    assert len(evaluations) == 12
    assert (run / "stage-02/opponent.zip").read_bytes() == stage_opponent
    campaign.prepare_stage(3)
    assert campaign.opponent.read_bytes() == (Path(retried["candidate"]).read_bytes()
                                             if opponent_mode == "previous-stage" else b"reference policy")
    (run / "stage-02/opponent.zip").write_bytes(b"modified snapshot")
    with pytest.raises(ValueError, match="opponent.*changed"):
        campaign.next_stage()


def test_old_fixed_campaign_and_stage_can_resume(tmp_path):
    cli = cli_module()
    source = tmp_path / "start.zip"
    source.write_bytes(b"start")
    opponent = tmp_path / "opponent.zip"
    opponent.write_bytes(b"reference")
    run = tmp_path / "old-campaign"
    run.mkdir()
    config = cli.initialize(argparse.Namespace(resume=source, opponent_checkpoint=opponent,
        curriculum=MANIFEST, eval_episodes=4, max_steps=200, promotion_rules=None, reward_config=None, seed=7), run)
    config.pop("opponent_mode")  # v0.22 campaign configuration.
    config.pop("team_groups")
    config.pop("validation_groups")
    stage = run / "stage-01"
    stage.mkdir()
    (stage / "anchor.zip").write_bytes(b"start")
    write_json(stage / "stage.json", dict(stage=1, source=str(run / "inputs/start.zip"),
                                         anchor_sha256=cli.sha256(source)))
    campaign = cli.Campaign(run, config, cli.EXECUTION)
    campaign.prepare_stage(1)
    assert campaign.opponent_mode == "fixed"
    assert campaign.opponent.read_bytes() == b"reference"
    assert (stage / "anchor.zip").read_bytes() == b"start"


def test_checkpoint_settings_can_change_on_legacy_campaign_resume(tmp_path, monkeypatch):
    cli = cli_module()
    source = tmp_path / "start.zip"
    source.write_bytes(b"policy")
    run = tmp_path / "campaign"
    run.mkdir()
    config = cli.initialize(argparse.Namespace(resume=source, opponent_checkpoint=source,
        curriculum=MANIFEST, eval_episodes=4, max_steps=200, promotion_rules=None,
        reward_config=None, seed=7), run)
    for name in ("latest_every_updates", "checkpoint_every", "retention"):
        config["execution"].pop(name)  # Simulate a campaign created before these settings existed.
    write_json(run / "campaign.json", config)
    observed = []
    def held(self, stage):
        observed.append((self.execution["latest_every_updates"], self.execution["checkpoint_every"]))
        return dict(status="hold")
    monkeypatch.setattr(cli.Campaign, "run_stage", held)
    base = ["run_curriculum", "--run-dir", str(run), "--attempts-per-stage", "1"]
    for options, expected in (([], (1, 1024)),
                             (["--latest-every-updates", "10", "--checkpoint-every", "0"], (10, 0)),
                             ([], (10, 0))):
        monkeypatch.setattr(sys, "argv", base + options)
        assert cli.main() == 2
        assert observed[-1] == expected
        saved = read_json(run / "campaign.json")
        assert saved["input_hashes"] == config["input_hashes"]
        assert saved["execution"]["rollout_steps"] == config["execution"]["rollout_steps"]
        assert saved["execution"]["retention"] == "off"
    monkeypatch.setattr(sys, "argv", base + ["--retention", "compact", "--keep-attempts", "1",
                                          "--keep-best", "1", "--no-step-logs"])
    assert cli.main() == 2
    saved = read_json(run / "campaign.json")
    assert saved["execution"]["retention"] == "compact"
    assert saved["execution"]["keep_attempts"] == saved["execution"]["keep_best"] == 1
    assert saved["execution"]["step_logs"] is False
    for invalid in (["--latest-every-updates", "0"], ["--checkpoint-every", "-1"]):
        monkeypatch.setattr(sys, "argv", base + invalid)
        with pytest.raises(SystemExit) as error:
            cli.parse_args()
        assert error.value.code == 2


def test_manifest_opponent_mode_and_explicit_override(tmp_path):
    cli = cli_module()
    source = tmp_path / "start.zip"
    source.write_bytes(b"policy")
    manifest = ROOT / "config/curricula/type_progression_selfplay/curriculum.json"
    data, paths = load_curriculum(manifest)
    assert data["opponent_mode"] == "previous-stage"
    assert paths == load_curriculum(MANIFEST)[1]
    for option, expected in ((None, "previous-stage"), ("fixed", "fixed")):
        run = tmp_path / expected
        run.mkdir()
        config = cli.initialize(argparse.Namespace(resume=source, opponent_checkpoint=source,
            curriculum=manifest, eval_episodes=4, max_steps=200, promotion_rules=None,
            reward_config=None, seed=7, opponent_mode=option), run)
        assert config["opponent_mode"] == expected


def test_auto_advance_stops_on_hold_and_cannot_skip_stage(tmp_path, monkeypatch):
    cli = cli_module()
    source = tmp_path / "source.zip"
    source.write_bytes(b"policy fixture")
    run = tmp_path / "campaign"
    run.mkdir()
    cli.initialize(argparse.Namespace(resume=source, opponent_checkpoint=source, curriculum=MANIFEST,
        eval_episodes=4, max_steps=200, promotion_rules=None, reward_config=None, seed=7), run)
    stages = []

    def held(self, stage):
        stages.append(stage)
        return dict(status="hold")

    monkeypatch.setattr(cli.Campaign, "run_stage", held)
    monkeypatch.setattr(sys, "argv", ["run_curriculum", "--run-dir", str(run), "--through-stage", "10"])
    assert cli.main() == 2 and stages == [1]
    monkeypatch.setattr(sys, "argv", ["run_curriculum", "--run-dir", str(run), "--stage", "2"])
    assert cli.main() == 1 and stages == [1]
    monkeypatch.setattr(sys, "argv", ["run_curriculum", "--run-dir", str(run),
                                      "--opponent-mode", "previous-stage"])
    assert cli.main() == 1 and stages == [1]


@pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(), reason="Showdown not installed")
def test_real_curriculum_training_evaluation_and_resume(tmp_path):
    pytest.importorskip("sb3_contrib")
    import gymnasium as gym
    import torch
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import BattleFeatures, policy_space
    torch.set_num_threads(1)

    class ShapeEnv(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)

    model = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2, n_epochs=1,
                        policy_kwargs=dict(features_extractor_class=BattleFeatures), seed=9)
    source = tmp_path / "initial.zip"
    model.save(source)
    original = source.read_bytes()
    run = tmp_path / "real-curriculum"
    command = [sys.executable, str(ROOT / "mgba/run_curriculum.py"), "--run-dir", str(run),
        "--resume", str(source), "--opponent-checkpoint", str(source), "--environments", "2",
        "--timesteps", "2", "--rollout-steps", "2", "--batch-size", "2", "--epochs", "1",
        "--eval-episodes", "4", "--max-steps", "2", "--no-battle-log", "--no-readable-log"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert first.returncode == 2, first.stdout + first.stderr  # Short cap holds on truncation.
    report = read_json(run / "stage-01/result.json")
    trained = MaskablePPO.load(report["candidate"])
    assert trained.num_timesteps == 2
    assert source.read_bytes() == original

    for label in ("current", "expanded"):
        cfg = read_json(Path(report["evaluations"][label]) / "config.json")
        assert cfg["models"]["candidate"]["sha256"] == report["candidate_sha256"]
        assert cfg["models"]["opponent"]["sha256"] == cli_module().sha256(source)
    second = subprocess.run([sys.executable, str(ROOT / "mgba/run_curriculum.py"),
        "--run-dir", str(run), "--stage", "1"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert second.returncode == 2, second.stdout + second.stderr
    retry = read_json(run / "stage-01/result.json")
    assert retry["evaluations"]["before_current"] == report["evaluations"]["before_current"]
    assert MaskablePPO.load(retry["candidate"]).num_timesteps == 4
    assert source.read_bytes() == original


@pytest.mark.skipif(not shutil.which("node") or not
    (ROOT / "sim/showdown/node_modules/pokemon-showdown/package.json").is_file(), reason="Showdown not installed")
def test_real_previous_stage_checkpoint_handoff(tmp_path):
    pytest.importorskip("sb3_contrib")
    import gymnasium as gym
    import torch
    from sb3_contrib import MaskablePPO
    from firered_rl.ppo_features import BattleFeatures, policy_space
    torch.set_num_threads(1)

    class ShapeEnv(gym.Env):
        observation_space = policy_space()
        action_space = gym.spaces.Discrete(11)

    model = MaskablePPO("MultiInputPolicy", ShapeEnv(), n_steps=2, batch_size=2, n_epochs=1,
                        policy_kwargs=dict(features_extractor_class=BattleFeatures), seed=9)
    source = tmp_path / "initial.zip"
    model.save(source)
    original = source.read_bytes()
    for i, species in enumerate(("Raticate", "Linoone", "Furret"), 1):
        write_json(tmp_path / f"team{i}.json", dict(party=[dict(species=species, level=1,
                   moves=["Double Edge", "Body Slam", "Strength", "Facade"])]))
    manifest = tmp_path / "curriculum.json"
    write_json(manifest, dict(version=1, name="smoke", opponent_mode="previous-stage", stages=[
        dict(stage=1, team="team1.json", focus="first"), dict(stage=2, team="team2.json", focus="second")],
        next_team="team3.json"))
    rules = tmp_path / "rules.json"
    # Permissive smoke-test thresholds exercise handoff; they are not training recommendations.
    write_json(rules, dict(current_win_rate=0, expanded_win_rate=0, incoming_win_rate=0,
                           max_regression=1, max_truncation_rate=1))
    run = tmp_path / "selfplay"
    result = subprocess.run([sys.executable, str(ROOT / "mgba/run_curriculum.py"),
        "--run-dir", str(run), "--curriculum", str(manifest), "--promotion-rules", str(rules),
        "--resume", str(source), "--opponent-checkpoint", str(source), "--through-stage", "2",
        "--environments", "2", "--timesteps", "2", "--rollout-steps", "2", "--batch-size", "2",
        "--epochs", "1", "--eval-episodes", "4", "--max-steps", "12", "--no-battle-log", "--no-readable-log"],
        cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    first = read_json(run / "stage-01/result.json")
    second = read_json(run / "stage-02/result.json")
    assert first["status"] == second["status"] == "promote"
    assert MaskablePPO.load(first["candidate"]).num_timesteps == 2
    assert MaskablePPO.load(second["candidate"]).num_timesteps == 4
    assert (run / "stage-01/opponent.zip").read_bytes() == original
    assert (run / "stage-02/opponent.zip").read_bytes() == Path(first["candidate"]).read_bytes()
    assert second["opponent"]["sha256"] == first["candidate_sha256"]
    assert second["opponent"]["source_stage"] == 1
    training = read_json(run / "stage-02/attempt-001/train/config.json")
    assert training["frozen_opponent"]["sha256"] == first["candidate_sha256"]
    assert training["resume_checkpoint"]["sha256"] == first["candidate_sha256"]
    for path in second["evaluations"].values():
        evaluation = read_json(Path(path) / "config.json")
        assert evaluation["models"]["opponent"]["sha256"] == first["candidate_sha256"]
    assert source.read_bytes() == (run / "inputs/opponent.zip").read_bytes() == original


def test_battle_budget_single_evaluation_and_named_history(tmp_path):
    cli = cli_module()
    source, opponent = tmp_path / 'smart.zip', tmp_path / 'dumb.zip'
    source.write_bytes(b'smart'); opponent.write_bytes(b'dumb')
    run = tmp_path / 'campaign'; run.mkdir()
    args = argparse.Namespace(resume=source, opponent_checkpoint=opponent, curriculum=MANIFEST,
        eval_episodes=1000, max_steps=200, promotion_rules=None, reward_config=None, seed=7,
        opponent_mode='previous-stage', train_battles=1000, promotion_win_rate=.75)
    config = cli.initialize(args, run)
    evaluations = []
    wins = [749, 750, 900]
    def execute(command, root):
        values = command[2:]
        get = lambda flag: Path(values[values.index(flag)+1])
        output = get('--output-dir')
        if Path(command[1]).name == 'train_ppo.py':
            assert '--total-timesteps' not in values
            assert str(get('--train-battles')) == '1000'
            (output / 'checkpoints').mkdir(parents=True)
            (output / 'checkpoints/final.zip').write_bytes(get('--resume').read_bytes()+b' trained')
            write_json(output / 'summary.json', dict(status='completed'))
        else:
            assert 'before-' not in str(output) and 'expanded' not in str(output)
            evaluations.append(output)
            suite = read_json(get('--suite'))
            completed_evaluation(output, suite, get('--checkpoint'), get('--opponent-checkpoint'))
            rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
            count = wins.pop(0)
            for row in rows[count:]:
                row['candidate_outcome'] = 'lost'
            (output / 'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            write_json(output / 'summary.json', summary(rows, requested=1000, elapsed=1, status='completed', suite_hash=digest(suite)))
    campaign = cli.Campaign(run, config, cli.EXECUTION, execute=execute)
    assert campaign.run_stage(1)['status'] == 'hold'
    assert not (run / 'models/smart_v2.zip').exists()
    assert campaign.run_stage(1)['status'] == 'promote'
    assert campaign.run_stage(2)['status'] == 'promote'
    assert len(evaluations) == 3
    assert (run / 'models/smart_v1.zip').read_bytes() == b'smart'
    assert (run / 'models/dumb.zip').read_bytes() == b'dumb'
    assert (run / 'stage-02/opponent.zip').read_bytes() == (run / 'models/smart_v2.zip').read_bytes()
    assert (run / 'models/smart_v3.zip').exists()
    before = (run / 'evaluations.csv').read_text()
    campaign.track_record()
    assert (run / 'evaluations.csv').read_text() == before
    assert len(before.splitlines()) == 4
    assert '75.0%' in (run / 'evaluations.md').read_text()


def test_resume_changes_gates_and_pool_without_changing_past_results_or_training_opponents(tmp_path, monkeypatch):
    cli = cli_module()
    source, opponent = tmp_path / "smart.zip", tmp_path / "dumb.zip"
    source.write_bytes(b"initial smart"); opponent.write_bytes(b"initial dumb")
    run = tmp_path / "campaign"; run.mkdir()
    config = cli.initialize(argparse.Namespace(resume=source, opponent_checkpoint=opponent,
        curriculum=MANIFEST, eval_episodes=20, max_steps=20, promotion_rules=None,
        reward_config=None, seed=7, train_battles=10, promotion_win_rate=.75,
        opponent_mode="previous-stage"), run)
    original_inputs = dict(config["input_hashes"])
    train_opponents, evaluations = [], []

    def execute(command, root):
        values = command[2:]
        get = lambda flag: Path(values[values.index(flag)+1])
        output = get("--output-dir")
        if Path(command[1]).name == "train_ppo.py":
            train_opponents.append(get("--opponent-checkpoint").read_bytes())
            (output / "checkpoints").mkdir(parents=True)
            (output / "checkpoints/final.zip").write_bytes(get("--resume").read_bytes() + b" trained")
            write_json(output / "summary.json", dict(status="completed"))
            return
        suite = read_json(get("--suite"))
        evaluations.append(suite)
        if suite.get("opponents"):
            assert "--opponent-checkpoint" not in values
        else:
            assert get("--opponent-checkpoint").read_bytes() == b"initial dumb"
        completed_evaluation(output, suite, get("--checkpoint"), opponent)
        rows = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
        for row in rows:
            # Stage-only: 50% wins. Pool: Basic 100%, Smart 0%.
            won = row.get("opponent_group") in ("random", "dumb") if suite.get("opponents") else row["episode"] % 2 == 0
            row["candidate_outcome"] = "won" if won else "lost"
        (output / "episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        write_json(output / "summary.json", summary(rows, requested=len(rows), elapsed=1,
            status="completed", suite_hash=digest(suite)))

    monkeypatch.setattr(cli, "run_child", execute)
    common = ["run_curriculum", "--run-dir", str(run), "--environments", "2", "--rollout-steps", "2", "--batch-size", "2"]
    monkeypatch.setattr(sys, "argv", common)
    assert cli.main() == 2  # 50% is below original 75%.
    held_path = run / "stage-01/attempt-001/report.json"
    held_bytes = held_path.read_bytes()
    monkeypatch.setattr(sys, "argv", [*common, "--promotion-win-rate", ".5"])
    assert cli.main() == 0  # Same score reaches newly requested threshold.
    assert held_path.read_bytes() == held_bytes
    assert train_opponents[:2] == [b"initial dumb"]*2
    promoted_bytes = (run / "stage-01/result.json").read_bytes()
    previous_smart = (run / "models/smart_v2.zip").read_bytes()
    monkeypatch.setattr(sys, "argv", [*common, "--eval-opponents", "mixed",
        "--promotion-basic-win-rate", ".75", "--promotion-smart-win-rate", ".6", "--promotion-require", "both"])
    assert cli.main() == 2  # Basic passes, Smart fails; both are required.
    assert train_opponents[-1] == previous_smart
    pooled_hold = run / "stage-02/attempt-001/report.json"
    pooled_bytes = pooled_hold.read_bytes()
    monkeypatch.setattr(sys, "argv", [*common, "--promotion-require", "either"])
    assert cli.main() == 0
    assert pooled_hold.read_bytes() == pooled_bytes
    assert (run / "stage-01/result.json").read_bytes() == promoted_bytes
    assert train_opponents[-1] == previous_smart
    assert evaluations[-1]["cases"] != evaluations[-2]["cases"]
    assert len([v for v in evaluations[-1]["opponents"].values() if v["group"] == "smart"]) == 2
    saved = read_json(run / "campaign.json")
    assert saved["input_hashes"] == original_inputs
    assert saved["settings_revision"] == 3
    assert len((run / "rule-changes.jsonl").read_text().splitlines()) == 3
    assert "Basic" in (run / "evaluations.md").read_text()
    assert len((run / "evaluations.csv").read_text().splitlines()) == 5


def test_real_battle_budget_finishes_update(tmp_path):
    pytest.importorskip('sb3_contrib')
    team = tmp_path / 'team.json'
    write_json(team, dict(party=[dict(species='Raticate', level=1,
        moves=['Double Edge', 'Body Slam', 'Strength', 'Facade'])]))
    run = tmp_path / 'train'
    result = subprocess.run([sys.executable, str(ROOT / 'mgba/train_ppo.py'),
        '--backend', 'showdown', '--environments', '2', '--teams', str(team), str(team),
        '--train-battles', '3', '--total-timesteps', '10000', '--max-steps', '1',
        '--rollout-steps', '2', '--batch-size', '2', '--epochs', '1',
        '--output-dir', str(run), '--no-readable-log', '--no-battle-log'],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = read_json(run / 'summary.json')
    assert report['status'] == 'completed'
    assert 3 <= report['finished_attempts'] <= 4
    assert report['player_timesteps'] < 10000
    assert (run / 'checkpoints/final.zip').is_file()
