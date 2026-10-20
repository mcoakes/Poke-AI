"""Same-directory run selection, reconstruction, and destructive-operation guards."""
import argparse
from io import StringIO
import json
from pathlib import Path
import zipfile

import pytest

from firered_rl.training_run import (RunCancelled, checkpoint_timesteps,
    explicit_arguments, newest_run_checkpoint, select_training_run)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--if-exists", choices=("ask", "continue", "erase", "cancel"), default="ask")
    p.add_argument("--backend", default="mgba")
    p.add_argument("--observation-version", default=None)
    p.add_argument("--resume", type=Path)
    p.add_argument("--opponent-checkpoint", "--player2-checkpoint", type=Path)
    p.add_argument("--reward-config", type=Path, default=Path("default.json"))
    p.add_argument("--team-dir", type=Path)
    p.add_argument("--state-dir", type=Path)
    p.add_argument("--teams", nargs=2, type=Path)
    p.add_argument("--states", nargs=2, type=Path)
    p.add_argument("--team-profile")
    p.add_argument("--training-profile")
    p.add_argument("--team-size", type=int)
    p.add_argument("--mirror-teams", action="store_true")
    p.add_argument("--shuffle-move-slots", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--shuffle-bench-order", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--opponent-stochastic", action="store_true")
    p.add_argument("--opponent-switch-chance", type=float, default=0)
    p.add_argument("--opponent-switch-cooldown", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--environments", type=int, default=4)
    p.add_argument("--rollout-steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--train-battles", type=int)
    p.add_argument("--total-timesteps", type=int, default=10000)
    p.add_argument("--readable-log", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--showdown-workers", default="thread")
    def speed(value):
        return "unlimited" if value.lower() == "unlimited" else float(value)
    p.add_argument("--speed", type=speed, default=1.0)
    return p


def checkpoint(path, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": steps}))
    return path


@pytest.fixture
def saved_run(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    run = project / "runs" / "ppo-v2"
    run.mkdir(parents=True)
    team = project / "teams" / "one.json"
    team.parent.mkdir()
    team.write_text('{}')
    description = dict(mode="random_directory", seed=17, mirror_teams=True,
                       teams=[dict(path=str(team), sha256="team-fingerprint")])
    config = dict(algorithm="MaskablePPO", backend="showdown", observation_version="v2",
                  teams=description, team_dir=str(team.parent), states=None, state_dir=None,
                  team_profile=None, team_size=None, training_profile=None, mirror_teams=True,
                  seed=17, max_steps=100, environments=48, total_timesteps=100000,
                  train_battles=None, opponent_checkpoint="frozen.zip", opponent_stochastic=False,
                  opponent_switch_chance=0, opponent_switch_cooldown=2,
                  frozen_opponent=dict(path=str(project / "frozen.zip"), sha256="frozen-fingerprint"),
                  reward_config=str(project / "reward.json"), reward=dict(win=5000, loss=-5000),
                  rollout_steps=128, batch_size=128, showdown_workers="process", readable_log=True)
    (run / "config.json").write_text(json.dumps(config))
    checkpoint(run / "checkpoints" / "final.zip", 101376)
    return project, run, config


def select(saved_run, *extra, input_fn=input, stream=None):
    project, run, _ = saved_run
    p = parser()
    argv = ["--output-dir", str(run), *extra]
    args = p.parse_args(argv)
    plan = select_training_run(p, args, argv, project, input_fn=input_fn, stream=stream)
    return plan, args


def validated(plan, config):
    plan.validate(config["teams"], config["reward"], "frozen-fingerprint")
    return plan


def test_minimal_continue_restores_actual_settings_and_additional_budget(saved_run):
    plan, args = select(saved_run, "--if-exists", "continue", "--total-timesteps", "400000")
    project, run, config = saved_run
    assert args.backend == "showdown" and args.observation_version == "v2"
    assert args.team_dir == project / "teams" and args.teams is None
    assert args.environments == 48 and args.showdown_workers == "process"
    assert args.mirror_teams and args.max_steps == 100 and args.seed == 17
    assert args.opponent_checkpoint == project / "frozen.zip"
    assert args.reward_config == project / "reward.json"
    assert args.total_timesteps == 400000 and args.train_battles is None
    # Omitted hyperparameters come from the model, not the run's CLI defaults.
    assert args.rollout_steps is None and args.batch_size is None
    assert args.resume == run / "checkpoints" / "final.zip"
    assert plan.action == "continue"
    assert (run / "config.json").read_text() == json.dumps(config)
    assert not (run.parent / f".{run.name}.training.lock").exists()


def test_checkpoint_selection_prefers_progress_over_stale_final_or_mtime(saved_run):
    _, run, _ = saved_run
    checkpoint(run / "checkpoints" / "latest.zip", 110592)
    newest = checkpoint(run / "checkpoints" / "ppo_000120000.zip", 120000)
    checkpoint(run / "checkpoints" / "ppo_999999999.zip", "corrupt")
    assert newest_run_checkpoint(run) == newest
    plan, args = select(saved_run, "--if-exists", "continue", "--resume", "latest")
    assert args.resume == newest and plan.checkpoint == newest
    checkpoint(run / "checkpoints" / "final.zip", 120000)
    assert newest_run_checkpoint(run).name == "final.zip"


def test_aliases_boolean_equal_flags_and_hyperparameter_overrides(saved_run):
    project, _, _ = saved_run
    plan, args = select(saved_run, "--if-exists=continue", "--no-readable-log",
                       "--player2-checkpoint", str(project / "frozen.zip"),
                       "--environments", "16", "--rollout-steps", "256")
    assert not args.readable_log and args.environments == 16 and args.rollout_steps == 256
    assert {"opponent_checkpoint", "if_exists", "readable_log", "rollout_steps"} <= plan.explicit
    assert "opponent_checkpoint" in explicit_arguments(parser(), ["--player2-checkpoint=x.zip"])


@pytest.mark.parametrize("selection", ["fixed", "team_profile", "training_profile", "state_dir"])
def test_restore_legacy_source_descriptions(saved_run, selection):
    project, run, config = saved_run
    description = config["teams"]
    if selection == "fixed":
        description["mode"] = "fixed"
        description["teams"] *= 2
        config["team_dir"] = None
    elif selection == "team_profile":
        description.update(mode="team_profile", team_profile="mixed", team_size=3)
        config["team_dir"] = None
    elif selection == "training_profile":
        config["team_dir"] = None
        config["training_profile"] = "six_types"
    else:
        config.update(backend="mgba", observation_version="v1", team_dir=None,
                      state_dir=str(project / "states"))
        config["states"] = config.pop("teams")
        config["states"]["checkpoints"] = config["states"].pop("teams")
    (run / "config.json").write_text(json.dumps(config))
    _, args = select(saved_run, "--if-exists", "continue")
    if selection == "fixed":
        assert args.teams == [project / "teams/one.json"] * 2
    elif selection == "team_profile":
        assert args.team_profile == "mixed" and args.team_size == 3
    elif selection == "training_profile":
        assert args.training_profile == "six_types"
    else:
        assert args.state_dir == project / "states" and args.backend == "mgba"


@pytest.mark.parametrize("flag,value", [("--backend", "mgba"), ("--observation-version", "v1"),
                                         ("--max-steps", "200"), ("--seed", "2"),
                                         ("--opponent-switch-chance", ".3")])
def test_immutable_cli_settings_rejected_without_mutation(saved_run, flag, value):
    _, run, _ = saved_run
    before = (run / "config.json").read_bytes()
    with pytest.raises(ValueError, match="differs from this run"):
        select(saved_run, "--if-exists", "continue", flag, value)
    assert (run / "config.json").read_bytes() == before


@pytest.mark.parametrize("changed", ["team", "reward", "opponent"])
def test_changed_file_content_or_frozen_weights_rejected(saved_run, changed):
    _, _, config = saved_run
    plan, _ = select(saved_run, "--if-exists", "continue")
    source = json.loads(json.dumps(config["teams"]))
    reward = dict(config["reward"])
    frozen_hash = "frozen-fingerprint"
    if changed == "team":
        source["teams"][0]["sha256"] = "changed"
    elif changed == "reward":
        reward["win"] = 99999
    else:
        frozen_hash = "changed"
    with pytest.raises(ValueError, match="changed"):
        plan.validate(source, reward, frozen_hash)
    assert not plan.validated


def test_explicit_new_selector_is_validated_as_data(saved_run):
    _, _, config = saved_run
    plan, args = select(saved_run, "--if-exists", "continue", "--team-profile", "mixed")
    assert args.team_dir is None and args.team_profile == "mixed"
    with pytest.raises(ValueError, match="team/state selection"):
        plan.validate(dict(config["teams"], mode="team_profile"), config["reward"], "frozen-fingerprint")


def test_cancel_precedes_config_or_dependencies_and_non_tty_never_prompts(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    fixture = project, project, None
    with pytest.raises(RunCancelled):
        select(fixture, "--if-exists", "cancel")
    with pytest.raises(ValueError, match="non-interactive"):
        select(fixture, stream=StringIO())
    class TTY(StringIO):
        def isatty(self):
            return True
    with pytest.raises(RunCancelled):
        select(fixture, input_fn=lambda _: "", stream=TTY())


def test_interactive_erase_needs_typed_confirmation(saved_run):
    class TTY(StringIO):
        def isatty(self):
            return True
    answers = iter(["erase", "wrong-name"])
    with pytest.raises(RunCancelled):
        select(saved_run, input_fn=lambda _: next(answers), stream=TTY())
    answers = iter(["e", saved_run[1].name])
    plan, _ = select(saved_run, input_fn=lambda _: next(answers), stream=TTY())
    assert plan.action == "erase" and (plan.run / "config.json").exists()


def test_erasing_happens_after_preflight_and_rejects_descendant_inputs(saved_run):
    _, run, config = saved_run
    plan, _ = select(saved_run, "--if-exists", "erase")
    with pytest.raises(ValueError, match="validated"):
        plan.prepare()
    plan.validate({}, {}, None)
    with pytest.raises(ValueError, match="selected training input"):
        plan.prepare(input_paths=[run / "checkpoints/final.zip"])
    assert (run / "config.json").exists()
    with plan.prepare(input_paths=[run.parent / "frozen.zip"]):
        assert run.is_dir() and list(run.iterdir()) == []
    assert not list(run.parent.glob("*.erasing-*"))


def test_active_run_lock_blocks_concurrent_continue_and_erase(saved_run):
    _, run, config = saved_run
    plan, _ = select(saved_run, "--if-exists", "continue")
    with validated(plan, config).prepare():
        for action in ("continue", "erase"):
            with pytest.raises(ValueError, match="active/locked"):
                select(saved_run, "--if-exists", action)
    # A stale lock file is harmless once its OS lock has been released.
    select(saved_run, "--if-exists", "continue")
    assert (run / "config.json").exists()


@pytest.mark.parametrize("change", ["config", "checkpoint", "new_checkpoint"])
def test_concurrent_preflight_changes_detected_before_any_write(saved_run, change):
    _, run, config = saved_run
    plan, _ = select(saved_run, "--if-exists", "continue")
    validated(plan, config)
    if change == "config":
        (run / "config.json").write_text(json.dumps(dict(config, seed=20)))
    elif change == "checkpoint":
        checkpoint(plan.checkpoint, 333333)
    else:
        checkpoint(run / "checkpoints/latest.zip", 333333)
    with pytest.raises(ValueError, match="changed during preflight"):
        plan.prepare()
    assert (run / "config.json").is_file()


@pytest.mark.parametrize("kind", ["project", "arbitrary", "symlink", "campaign"])
def test_erase_protects_nonrun_and_managed_directories(saved_run, kind):
    project, run, config = saved_run
    if kind == "project":
        target = project
    elif kind == "arbitrary":
        target = project / "arbitrary"
        target.mkdir()
        (target / "keep.txt").write_text("keep")
    elif kind == "symlink":
        target = project / "alias"
        target.symlink_to(run, target_is_directory=True)
    else:
        target = project / "campaign/stage-01/attempt-001/train"
        target.mkdir(parents=True)
        (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        select((project, target, config), "--if-exists", "erase")
    assert target.exists()


def test_new_run_preparation_and_saved_args_restore(saved_run):
    project, run, config = saved_run
    config["saved_args"] = dict(seed=17, backend="showdown", team_dir=str(project / "teams"),
                                train_battles=1000, readable_log=False)
    (run / "config.json").write_text(json.dumps(config))
    _, args = select(saved_run, "--if-exists", "continue", "--total-timesteps", "400000")
    assert args.train_battles is None and not args.readable_log
    target = project / "new/run"
    plan, args = select((project, target, None), "--if-exists", "continue", "--backend", "showdown")
    assert plan.action == "new" and args.backend == "showdown"
    assert not target.exists()
    plan.validate({}, {}, None)
    with plan.prepare():
        assert target.is_dir()


def test_legacy_run_has_shuffling_off_and_requires_new_run_to_enable(saved_run):
    _, args = select(saved_run, "--if-exists", "continue")
    assert args.shuffle_move_slots is False
    with pytest.raises(ValueError, match="shuffle-move-slots differs"):
        select(saved_run, "--if-exists", "continue", "--shuffle-move-slots")


def test_bench_shuffle_is_restored_and_cannot_be_toggled_during_continue(saved_run):
    _, run, config = saved_run
    _, args = select(saved_run, "--if-exists", "continue")
    assert args.shuffle_bench_order is False
    with pytest.raises(ValueError, match="shuffle-bench-order differs"):
        select(saved_run, "--if-exists", "continue", "--shuffle-bench-order")
    config['shuffle_bench_order'] = True
    (run / 'config.json').write_text(json.dumps(config))
    _, args = select(saved_run, "--if-exists", "continue")
    assert args.shuffle_bench_order is True
    with pytest.raises(ValueError, match="shuffle-bench-order differs"):
        select(saved_run, "--if-exists", "continue", "--no-shuffle-bench-order")


def test_external_or_older_resume_rejected_for_same_directory(saved_run):
    _, run, _ = saved_run
    checkpoint(run / "checkpoints/latest.zip", 120000)
    with pytest.raises(ValueError, match="most advanced checkpoint"):
        select(saved_run, "--if-exists", "continue", "--resume", str(run / "checkpoints/final.zip"))


def test_dotdot_cannot_bypass_protected_project_guard(saved_run):
    project, run, config = saved_run
    # A syntactically different path is still the project directory.
    (project / "child").mkdir()
    (project / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="project, home, working directory"):
        select((project, project / "child/..", config), "--if-exists", "erase")
    assert (project / "config.json").exists()


def test_campaign_manifest_blocks_other_nested_run_names(saved_run):
    project, run, config = saved_run
    campaign = project / "campaign"
    target = campaign / "unusual/nested/run"
    target.mkdir(parents=True)
    (campaign / "campaign.json").write_text('{}')
    (target / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="campaign-managed"):
        select((project, target, config), "--if-exists", "erase")


def test_saved_args_none_version_does_not_override_resolved_model_version(saved_run):
    _, run, config = saved_run
    config["saved_args"] = {"observation_version": None, "backend": "showdown"}
    (run / "config.json").write_text(json.dumps(config))
    _, args = select(saved_run, "--if-exists", "continue")
    assert args.observation_version == "v2"


@pytest.mark.parametrize("name", ["inputs", "sessions", "checkpoints", "summary.json",
                                  "summary.tmp", "episodes.jsonl", "updates.jsonl",
                                  "battle-blocks.jsonl", "switching.json", "failure.json"])
def test_continuation_rejects_writable_internal_symlinks(saved_run, name):
    project, run, config = saved_run
    outside = project / "outside"
    if name in {"inputs", "sessions", "checkpoints"}:
        outside.mkdir()
        if (run / name).exists():
            (run / name).rename(project / "old-checkpoints")
    else:
        outside.write_text("untouched")
    (run / name).symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        select(saved_run, "--if-exists", "continue")
    if outside.is_file():
        assert outside.read_text() == "untouched"


def test_nested_checkpoint_and_archived_metadata_symlinks_rejected(saved_run):
    project, run, config = saved_run
    outside = project / "outside"
    outside.write_text("untouched")
    linked = run / "checkpoints/latest.tmp.zip"
    linked.symlink_to(outside)
    with pytest.raises(ValueError, match="linked run output"):
        select(saved_run, "--if-exists", "continue")
    linked.unlink()
    archived = run / "sessions/000001"
    archived.mkdir(parents=True)
    (archived / "config.json").symlink_to(outside)
    with pytest.raises(ValueError, match="session metadata"):
        select(saved_run, "--if-exists", "continue")
    assert outside.read_text() == "untouched"


def test_later_alias_is_scoped_to_current_run(saved_run):
    _, run, _ = saved_run
    _, args = select(saved_run, "--if-exists", "continue", "--resume", "later")
    assert args.resume == run / "checkpoints/final.zip"


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
@pytest.mark.parametrize("during_confirmation", [False, True])
def test_interrupted_menu_or_erase_confirmation_cancels_cleanly(saved_run, error, during_confirmation):
    _, run, config = saved_run
    before = (run / "config.json").read_bytes()
    class TTY(StringIO):
        def isatty(self):
            return True
    calls = 0
    def interrupted(message):
        nonlocal calls
        calls += 1
        if during_confirmation and calls == 1:
            return "erase"
        raise error()
    with pytest.raises(RunCancelled, match="left unchanged"):
        select(saved_run, input_fn=interrupted, stream=TTY())
    assert (run / "config.json").read_bytes() == before
    assert (run / "checkpoints/final.zip").is_file()


@pytest.mark.parametrize("alias", ["latest", "later"])
def test_opponent_latest_alias_keeps_saved_frozen_identity(saved_run, alias):
    project, run, config = saved_run
    plan, args = select(saved_run, "--if-exists", "continue", "--opponent-checkpoint", alias)
    assert args.opponent_checkpoint == project / "frozen.zip"
    assert args.resume == run / "checkpoints/final.zip"
    with pytest.raises(ValueError, match="frozen opponent checkpoint changed"):
        plan.validate(config["teams"], config["reward"], "new-global-learner-hash")
    plan.validate(config["teams"], config["reward"], "frozen-fingerprint")


def test_latest_alias_cannot_change_selfplay_into_frozen_training(saved_run):
    _, run, config = saved_run
    config["opponent_checkpoint"] = None
    config.pop("frozen_opponent")
    (run / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="frozen-opponent/self-play mode"):
        select(saved_run, "--if-exists", "continue", "--opponent-checkpoint", "latest")


@pytest.mark.parametrize("speed", [1.0, 4.0, "unlimited"])
def test_saved_numeric_speed_goes_through_cli_converter_as_text(saved_run, speed):
    _, run, config = saved_run
    config["speed"] = speed
    (run / "config.json").write_text(json.dumps(config))
    _, args = select(saved_run, "--if-exists", "continue")
    assert args.speed == speed
