"""Best-model selection, fixed-suite continuation, and real isolated validation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("sb3_contrib")

from firered_rl.evaluation import create_suite, digest
from firered_rl.periodic_evaluation import EvaluationSetup, PeriodicEvaluation, sha256
from firered_rl.training_teams import TrainingTeams


def read(path):
    return json.loads(path.read_text())


def suite_file(directory, teams, opponent=None):
    suite = create_suite(TrainingTeams(teams=teams), backend="showdown", episodes=4,
                         seed=777, swap_sides=True, max_steps=2)
    entry = (dict(kind="checkpoint", group="dumb", label="Frozen", path=str(opponent),
                  sha256=sha256(opponent.read_bytes())) if opponent else
             dict(kind="random", group="random", label="Random"))
    suite["opponents"] = {"reference": entry}
    for case in suite["cases"]:
        case.update(opponent_id="reference", opponent_group=entry["group"], opponent_seed=case["seed"] + 1)
    path = directory / "suite.json"
    path.write_text(json.dumps(suite))
    return path


@pytest.fixture
def controller(tmp_path):
    teams = [tmp_path / f"team-{i}.json" for i in range(2)]
    for team in teams:
        team.write_text(json.dumps(dict(party=[dict(species="Pikachu", level=50, moves=["Thunderbolt"])])))
    suite = suite_file(tmp_path, teams)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    args = SimpleNamespace(eval_suite=suite, backend="showdown", eval_environments=2, environments=2,
                           opponent_checkpoint=None, eval_every=10, keep_evaluations=2, showdown_workers="process",
                           device="cpu", torch_threads=1, node="node", command_timeout=10,
                           log_max_mb=1, log_backups=1, battle_log=True)
    plan = SimpleNamespace(run=run, action="new", saved_config=None)
    setup = EvaluationSetup(args, plan, {}, None)
    description = setup.install(run, args)
    instance = PeriodicEvaluation(tmp_path, run, args, setup)
    instance.test_args, instance.test_plan, instance.test_description = args, plan, description
    return instance


def evaluator_stub(monkeypatch, controller, *, wins, truncated=0, exit_code=0, incomplete=False):
    def execute(command):
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        data = checkpoint.read_bytes()
        (output / "candidate.zip").write_bytes(data)
        steps = int(data.decode())
        summary = dict(status="completed", requested_battles=4, attempted_battles=3 if incomplete else 4,
                       completed_battles=4-truncated, suite_sha256=controller.history["identity"]["suite_sha256"],
                       candidate_wins=wins, opponent_wins=4-wins-truncated, draws=0,
                       truncated=truncated, candidate_reward_mean=10,
                       by_opponent_group={"random": {"win_rate": wins/4}})
        (output / "summary.json").write_text(json.dumps(summary))
        (output / "config.json").write_text(json.dumps(dict(reward={}, models={"candidate": {
            "sha256": sha256(data), "training_timesteps": steps}})))
        return exit_code
    monkeypatch.setattr(controller, "_execute", execute)


def run_validation(controller, steps):
    checkpoint = controller.run / "checkpoints/latest.zip"
    checkpoint.write_bytes(str(steps).encode())
    return controller.evaluate(checkpoint, steps, reason="interval")


def test_best_survives_worse_tied_truncated_results_and_rotation(controller, monkeypatch):
    c = controller
    for steps, wins, truncated in [(10, 3, 0), (20, 2, 0), (30, 3, 1), (40, 2, 0)]:
        evaluator_stub(monkeypatch, c, wins=wins, truncated=truncated)
        run_validation(c, steps)
    best = c.run / "checkpoints/best.zip"
    assert best.read_bytes() == b"10"
    assert read(best.with_suffix(".json"))["win_rate"] == .75
    assert c.records[2]["win_rate"] == .75  # Truncated match is not dropped from denominator.
    assert not c.records[2]["new_best"]
    assert sorted(p.name for p in c.directory.glob("evaluation-*")) == [
        "evaluation-000001", "evaluation-000003", "evaluation-000004"]
    evaluator_stub(monkeypatch, c, wins=4)
    run_validation(c, 50)
    assert best.read_bytes() == b"50"
    assert sorted(p.name for p in c.directory.glob("evaluation-*")) == ["evaluation-000004", "evaluation-000005"]
    assert len(read(c.run / "evaluation-history.json")["records"]) == 5
    assert len((c.run / "evaluations.csv").read_text().splitlines()) == 6
    assert "100.0%" in (c.run / "evaluations.md").read_text()


@pytest.mark.parametrize("mode", ["failed", "incomplete", "interrupt"])
def test_failed_validation_keeps_best_and_records_diagnostics(controller, monkeypatch, mode):
    c = controller
    evaluator_stub(monkeypatch, c, wins=2)
    run_validation(c, 10)
    evaluator_stub(monkeypatch, c, wins=4, exit_code=1 if mode == "failed" else 0,
                   incomplete=mode == "incomplete")
    if mode == "interrupt":
        def interrupted(command):
            raise KeyboardInterrupt
        monkeypatch.setattr(c, "_execute", interrupted)
    with pytest.raises(KeyboardInterrupt if mode == "interrupt" else ValueError):
        run_validation(c, 20)
    assert (c.run / "checkpoints/best.zip").read_bytes() == b"10"
    assert c.records[-1]["status"] == ("interrupted" if mode == "interrupt" else "failed")
    assert c.due(20) and c.needs_final(20)


def test_continuation_reconstructs_best_and_enforces_suite_identity(controller, monkeypatch):
    c = controller
    evaluator_stub(monkeypatch, c, wins=3)
    run_validation(c, 10)
    best = c.run / "checkpoints/best.zip"
    best.write_bytes(b"interrupted publication")
    plan = SimpleNamespace(run=c.run, action="continue", saved_config={"evaluation": c.test_description})
    setup = EvaluationSetup(c.test_args, plan, {}, None)
    resumed = PeriodicEvaluation(c.root, c.run, c.test_args, setup)
    assert best.read_bytes() == b"10"
    assert not resumed.due(19) and resumed.due(20)
    assert not resumed.needs_final(10) and resumed.needs_final(12)
    c.test_args.eval_every = 0
    assert not resumed.due(100) and not resumed.needs_final(100)
    changed = read(c.test_args.eval_suite)
    changed["seed"] += 1
    c.test_args.eval_suite.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="evaluation suite or reference changed"):
        EvaluationSetup(c.test_args, plan, {}, None)


def test_best_is_excluded_from_numbered_retention(controller, monkeypatch):
    from test_observation_v2_evaluation import load_cli
    cli = load_cli("train_ppo", monkeypatch)
    directory = controller.run / "checkpoints"
    for name in ("best.zip", "best.json", "latest.zip", "final.zip", "ppo_000000010.zip"):
        (directory / name).write_bytes(b"checkpoint")
    cli.prune_numbered_checkpoints(directory, 0)
    assert {p.name for p in directory.iterdir()} == {"best.zip", "best.json", "latest.zip", "final.zip"}


def test_pinned_opponents_remain_usable_without_the_original_files(controller, tmp_path):
    c = controller
    original_suite = read(c.test_args.eval_suite)
    source = tmp_path / "frozen.zip"
    source.write_bytes(b"fixed checkpoint")
    checksum = sha256(source.read_bytes())
    suite = deepcopy(original_suite)
    suite["opponents"]["reference"] = dict(kind="checkpoint", group="smart", label="Reference",
                                           path=str(source), sha256=checksum)
    for case in suite["cases"]:
        case["opponent_group"] = "smart"
    c.test_args.eval_suite.write_text(json.dumps(suite))
    setup = EvaluationSetup(c.test_args, c.test_plan, {}, None)
    description = setup.install(c.run, c.test_args)
    source.unlink()
    plan = SimpleNamespace(run=c.run, action="continue", saved_config={"evaluation": description})
    setup = EvaluationSetup(c.test_args, plan, {}, None)
    assert setup.opponents[checksum] == b"fixed checkpoint"
    assert setup.identity["suite_sha256"] == digest(original_suite | {"opponents": suite["opponents"], "cases": suite["cases"]})


def test_suite_without_pool_pins_training_reference_and_reward(controller, tmp_path):
    c = controller
    suite = read(c.test_args.eval_suite)
    del suite["opponents"]
    for case in suite["cases"]:
        for field in ("opponent_id", "opponent_group", "opponent_seed"):
            del case[field]
    c.test_args.eval_suite.write_text(json.dumps(suite))
    reference = tmp_path / "reference.zip"
    reference.write_bytes(b"training opponent")
    c.test_args.opponent_checkpoint = reference
    reward = {"win": 5000}
    setup = EvaluationSetup(c.test_args, c.test_plan, reward, sha256(reference.read_bytes()))
    setup.install(c.run, c.test_args)
    reference.unlink()
    evaluator = PeriodicEvaluation(c.root, c.run, c.test_args, setup)
    command = evaluator._command(c.run / "checkpoints/latest.zip", c.run / "evaluation")
    pinned_reference = Path(command[command.index("--opponent-checkpoint") + 1])
    assert pinned_reference.read_bytes() == b"training opponent"
    assert read(Path(command[command.index("--reward-config") + 1])) == reward


def test_success_stop_requires_distinct_complete_untruncated_checkpoints(controller, monkeypatch):
    c = controller
    c.args.eval_stop_win_rate, c.args.eval_stop_consecutive = .75, 2
    evaluator_stub(monkeypatch, c, wins=3)
    run_validation(c, 0)
    assert c.stopping_status(0)["consecutive"] == 1
    assert not c.stopping_status(0)["achieved"]
    # An unchanged checkpoint re-evaluated twice is not two PPO checkpoints.
    run_validation(c, 0)
    assert c.stopping_status(0)["consecutive"] == 1
    run_validation(c, 10)
    assert c.stopping_status(10)["achieved"]
    assert not c.stopping_status(11)["achieved"]  # Current weights have not been validated.
    evaluator_stub(monkeypatch, c, wins=3, truncated=1)
    run_validation(c, 20)
    assert not c.stopping_status(20)["achieved"]
    assert c.stopping_status(20)["consecutive"] == 0
    evaluator_stub(monkeypatch, c, wins=3)
    run_validation(c, 30)
    assert c.stopping_status(30)["consecutive"] == 1
    evaluator_stub(monkeypatch, c, wins=2)
    run_validation(c, 40)
    # Deliberately misleading displayed rate must not override requested denominator.
    c.records[-1]["win_rate"] = 1.0
    assert not c.stopping_status(40)["achieved"]


def test_success_streak_restores_from_history_and_failure_breaks_it(controller, monkeypatch):
    c = controller
    c.args.eval_stop_win_rate, c.args.eval_stop_consecutive = .75, 2
    evaluator_stub(monkeypatch, c, wins=3)
    run_validation(c, 0)
    plan = SimpleNamespace(run=c.run, action="continue", saved_config={"evaluation": c.test_description})
    resumed = PeriodicEvaluation(c.root, c.run, c.args, EvaluationSetup(c.args, plan, {}, None))
    evaluator_stub(monkeypatch, resumed, wins=3)
    run_validation(resumed, 10)
    assert resumed.stopping_status(10)["achieved"]
    evaluator_stub(monkeypatch, resumed, wins=3, exit_code=1)
    with pytest.raises(ValueError, match="evaluation exited"):
        run_validation(resumed, 20)
    assert not resumed.stopping_status(20)["achieved"]
    evaluator_stub(monkeypatch, resumed, wins=3)
    run_validation(resumed, 30)
    assert resumed.stopping_status(30)["consecutive"] == 1
    resumed.args.eval_stop_win_rate = None
    assert not resumed.stopping_status(30)["enabled"]
