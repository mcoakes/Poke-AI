"""Read-only policy checks across the real Showdown/SB3 vector boundary.

Scripted actions establish battle outcomes; policy inference checks legality only.
Nothing here calls ``learn`` or writes a model checkpoint.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
from threading import Event
from types import SimpleNamespace

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _same_observation(first, second):
    return first.keys() == second.keys() and all(
        np.array_equal(first[key], second[key]) for key in first)


def _policy_digest(policy):
    """Include buffers as well as trainable parameters, without retaining copies."""
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _real_vector(root, directory, parties, *, node, timeout, seed, max_steps,
                 log_max_mb, log_backups):
    from .ppo_vec_env import LinkedVecEnv
    from .showdown_env import ShowdownBattleSession, ShowdownFactory

    directory.mkdir(parents=True, exist_ok=False)
    teams = []
    for index, party in enumerate(parties):
        path = directory / f"team-{index + 1}.json"
        path.write_text(json.dumps(dict(name=f"audit-{index + 1}", party=party)) + "\n")
        teams.append(path)
    cancel = Event()
    args = SimpleNamespace(seed=seed, node=node, command_timeout=timeout,
                           log_max_mb=log_max_mb, log_backups=log_backups,
                           step_logs=False, keep_episode_logs=0)
    factory = ShowdownFactory(root, directory, [55355, 55356], lambda: teams, args, cancel)
    session = ShowdownBattleSession(factory, max_steps=max_steps,
        readable_log=directory / "battle-readable.log", battle_log=directory / "battle.log")
    events = []

    def completed(pair, **fields):
        # LinkedVecEnv calls this before its automatic reset. Keep the actual
        # simulator reports to verify the terminal_observation's provenance.
        events.append(dict(reports=deepcopy(session.reports), episode=factory.episode,
                           terminated=fields["terminated"], truncated=fields["truncated"]))

    vec = None
    try:
        vec = LinkedVecEnv([session], cancel=cancel, aborters=[factory.stop],
                           on_episode=completed)
        vec.seed(seed)
        yield vec, session, events
    finally:
        if vec is not None:
            vec.close()
        else:
            session.close()


def run_pipeline_checks(root, output_dir, *, node="node", timeout=30, checkpoint=None,
                        log_max_mb=2, log_backups=1, seed=26002):
    """Return JSON-safe pass/fail evidence; reserve ``output_dir/pipeline``.

    Requires the installed training dependencies and pinned local simulator.
    A checkpoint is loaded once from an in-memory snapshot, used in evaluation
    mode, and checked for unchanged parameters, buffers and source bytes. Without
    one, a temporary untrained policy tests interfaces, not behavioral quality.
    """
    if not np.isfinite(log_max_mb) or log_max_mb <= 0:
        raise ValueError("log_max_mb must be finite and strictly positive for bounded audit logs")
    if log_backups < 0:
        raise ValueError("log_backups must be nonnegative")

    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    from .ppo_features import BattleFeatures, policy_observation, policy_space
    from .showdown_env import showdown_observation, validate_showdown

    root, output_dir = Path(root), Path(output_dir)
    validate_showdown(root, node)
    directory = output_dir / "pipeline"
    directory.mkdir(parents=True, exist_ok=False)
    checks, samples, inference_samples, truncation_infos = [], [], [], []
    terminal_infos, model_state = [], {}
    model = None

    def check(name, work):
        try:
            details = work()
        except Exception as error:
            checks.append(dict(name=f"pipeline_{name}", status="fail",
                               details=dict(error=f"{type(error).__name__}: {error}")))
            return None
        checks.append(dict(name=f"pipeline_{name}", status="pass", details=details))
        return details

    def sample(label, observations, *, inference=False):
        saved = {key: value.copy() for key, value in observations.items()}
        samples.append((label, saved))
        if inference:
            inference_samples.append((label, saved))

    def load_policy(vec):
        nonlocal model
        if checkpoint is None:
            model = MaskablePPO("MultiInputPolicy", vec, n_steps=2, batch_size=4,
                n_epochs=1, device="cpu", seed=seed,
                policy_kwargs=dict(features_extractor_class=BattleFeatures, net_arch=[32]))
            source = "temporary untrained MaskablePPO"
        else:
            path = Path(checkpoint).expanduser().resolve()
            content = path.read_bytes()
            model_state["checkpoint_path"] = path
            model_state["checkpoint_sha256"] = hashlib.sha256(content).hexdigest()
            model = MaskablePPO.load(BytesIO(content), env=vec, device="cpu")
            source = str(path)
        model_state["digest"] = _policy_digest(model.policy)
        model_state["timesteps"] = model.num_timesteps
        model_state["updates"] = model._n_updates
        model.policy.set_training_mode(False)
        _require(model.observation_space == policy_space(), "checkpoint observation space mismatch")
        _require(model.action_space == vec.action_space, "checkpoint action space mismatch")
        return dict(source=source, trained_checkpoint=checkpoint is not None,
                    checkpoint_sha256=model_state.get("checkpoint_sha256"),
                    behavioral_quality="untested; this audit checks contracts and legality",
                    num_timesteps=int(model.num_timesteps), training_performed=False,
                    checkpoint_writes=0)

    common = dict(node=node, timeout=timeout, seed=seed,
                  log_max_mb=log_max_mb, log_backups=log_backups)

    def mon(species, move, level=50):
        return dict(species=species, moves=[move], level=level)

    def verify_final(vec, session, events, observations, dones, infos, *, terminated):
        _require(dones.tolist() == [True, True], "both vector players must finish together")
        _require(len(events) == 1, "expected one paired episode callback")
        _require(events[0]["terminated"] is terminated, "wrong termination flag at callback")
        _require(events[0]["truncated"] is (not terminated), "wrong truncation flag at callback")
        _require(session.factory.episode == events[0]["episode"] + 1,
                 "the linked pair must automatically reset exactly once")
        _require(session.steps == 0 and not session.done, "returned vector must be a fresh battle")
        for index, info in enumerate(infos):
            _require(info["TimeLimit.truncated"] is (not terminated),
                     "TimeLimit.truncated must distinguish truncation from game termination")
            final = info["terminal_observation"]
            actual = policy_observation(showdown_observation(events[0]["reports"][index], terminated))
            _require(_same_observation(final, actual), "terminal_observation lost the pre-reset state")
            reset = {key: value[index] for key, value in observations.items()}
            _require(not _same_observation(final, reset), "terminal_observation was replaced by reset")
            _require(info["episode"]["l"] == 1, "scripted episode must take one decision")
        sample("terminal_final" if terminated else "truncation_final",
               vec.stack([info["terminal_observation"] for info in infos]),
               inference=not terminated)
        sample("terminal_autoreset" if terminated else "truncation_autoreset", observations)

    def terminal_case():
        parties = [[mon("Magikarp", "Splash", 1)], [mon("Blastoise", "Surf")]]
        with _real_vector(root, directory / "terminal", parties, max_steps=10, **common) as bundle:
            vec, session, events = bundle
            before = vec.reset()
            sample("terminal_initial", before, inference=True)
            _require(np.array_equal(get_action_masks(vec), before["action_mask"].astype(bool)),
                     "SB3 action-mask accessor differs from the real vector observation")
            check("policy_load", lambda: load_policy(vec))
            observations, rewards, dones, infos = vec.step(np.array([0, 0]))
            terminal_infos.extend(infos)
            verify_final(vec, session, events, observations, dones, infos, terminated=True)
            _require([info["outcome"] for info in infos] == ["lost", "won"],
                     "level-1 Magikarp should lose to Blastoise Surf")
            _require(rewards.tolist() == [-1.0, 1.0], "default terminal rewards must be loss/win")
            _require(all(not info["terminal_observation"]["action_mask"].any() for info in infos),
                     "game terminal observations must have no legal actions")
            _require(all(info["terminal_observation"]["context"][2] == 1 for info in infos),
                     "game terminal observations must mark terminal context")
            return dict(scripted_actions=[0, 0], outcomes=["lost", "won"],
                        rewards=rewards.tolist(), terminal_observation_before_reset=True,
                        time_limit_truncated=[False, False], autoresets=1)

    def replacement_case():
        parties = [[mon("Magikarp", "Splash", 1), mon("Magikarp", "Splash", 1)],
                   [mon("Blastoise", "Surf")]]
        with _real_vector(root, directory / "replacement", parties, max_steps=10, **common) as bundle:
            vec, session, _ = bundle
            sample("replacement_initial", vec.reset(), inference=True)
            before, _, dones, infos = vec.step(np.array([0, 0]))
            sample("forced_replacement", before, inference=True)
            _require(not dones.any(), "one KO with a living reserve must not end the battle")
            _require(not any(info["TimeLimit.truncated"] for info in infos),
                     "forced replacement must not truncate")
            legal = [np.flatnonzero(mask).tolist() for mask in before["action_mask"]]
            _require(legal == [[5], [10]], "replacement player must switch and partner must WAIT only")
            _require(before["context"][:, 1].tolist() == [1.0, 0.0],
                     "forced-switch context must identify the replacing side only")
            steps = session.steps
            rejected = False
            try:
                vec.step_async(np.array([5, 0]))
            except ValueError:
                rejected = True
            _require(rejected and session.steps == steps and vec.pending is None,
                     "partner move during WAIT must be rejected before reaching Showdown")
            after, _, done_after, _ = vec.step(np.array([5, 10]))
            sample("replacement_completed", after, inference=True)
            _require(not done_after.any(), "replacement must leave the battle ongoing")
            _require(after["active"][0, 0, 8] == 2, "replacement must select original party slot 2")
            _require(after["party"][0, 1, 3] == before["party"][0, 1, 3],
                     "WAIT must not attack the replacement")
            _require(after["moves"][1, 0, 0, 2] == before["moves"][1, 0, 0, 2],
                     "WAIT must not spend another Surf PP")
            return dict(forced_legal_actions=legal, invalid_partner_move_rejected=True,
                        scripted_replacement=[5, 10], active_party_slot=2,
                        partner_extra_move=False)

    def truncation_case():
        parties = [[mon("Magikarp", "Splash")], [mon("Magikarp", "Splash")]]
        with _real_vector(root, directory / "truncation", parties, max_steps=1, **common) as bundle:
            vec, session, events = bundle
            sample("truncation_initial", vec.reset())
            observations, rewards, dones, infos = vec.step(np.array([0, 0]))
            truncation_infos.extend(infos)
            verify_final(vec, session, events, observations, dones, infos, terminated=False)
            _require([info["outcome"] for info in infos] == ["ongoing", "ongoing"],
                     "time-limit fixture must remain a live battle")
            _require(all(info["reason"] == "max_steps_or_counter" for info in infos),
                     "time-limit reason must survive vectorization")
            _require(rewards.tolist() == [0.0, 0.0], "live Splash truncation must have no terminal reward")
            _require(all(info["terminal_observation"]["context"][2] == 0 for info in infos),
                     "a time limit must not mark the live observation as a game terminal")
            _require(all(info["terminal_observation"]["action_mask"].any() for info in infos),
                     "live final observations must retain legal actions for bootstrapping")
            return dict(max_steps=1, outcomes=["ongoing", "ongoing"], rewards=rewards.tolist(),
                        time_limit_truncated=[True, True], terminal_observation_before_reset=True,
                        autoresets=1)

    # Policy construction seeds global RNGs; preserve the caller's RNG state.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            check("terminal_autoreset", terminal_case)
            check("forced_replacement_wait", replacement_case)
            check("live_truncation", truncation_case)

            def observation_contract():
                _require(samples, "no live Showdown observations were collected")
                space = policy_space()
                for label, batch in samples:
                    _require(set(batch) == set(space.spaces), f"{label}: policy fields differ")
                    for key, field in space.spaces.items():
                        _require(batch[key].shape == (2, *field.shape), f"{label}/{key}: shape differs")
                        _require(batch[key].dtype == np.float32, f"{label}/{key}: expected float32")
                        _require(np.isfinite(batch[key]).all(), f"{label}/{key}: nonfinite observation")
                    for index in range(2):
                        _require(space.contains({key: value[index] for key, value in batch.items()}),
                                 f"{label}: observation falls outside policy_space")
                return dict(batch_size=2, observations_checked=len(samples) * 2,
                            fields={key: list(field.shape) for key, field in space.spaces.items()},
                            status_bits=True, volatile_bits=True)

            check("observation_contract", observation_contract)

            def masked_inference():
                _require(model is not None and "digest" in model_state, "policy initialization failed")
                evidence = []
                for label, batch in inference_samples:
                    masks = batch["action_mask"].astype(bool)
                    _require(masks.any(axis=1).all(), f"{label}: inference has no legal action")
                    tensor, _ = model.policy.obs_to_tensor(batch)
                    with torch.no_grad():
                        distribution = model.policy.get_distribution(tensor, action_masks=masks)
                        probabilities = distribution.distribution.probs.detach().cpu().numpy()
                    _require(probabilities.shape == (2, 11), f"{label}: wrong probability shape")
                    _require(np.isfinite(probabilities).all(), f"{label}: probabilities are nonfinite")
                    _require((probabilities >= 0).all() and np.allclose(probabilities.sum(axis=1), 1),
                             f"{label}: invalid probability distribution")
                    _require((probabilities[~masks] == 0).all(), f"{label}: masked actions have probability")
                    actions, _ = model.predict(batch, action_masks=masks, deterministic=True)
                    _require(all(masks[index, action] for index, action in enumerate(actions)),
                             f"{label}: MaskablePPO.predict selected an unavailable action")
                    evidence.append(dict(observation=label, predicted_actions=actions.tolist(),
                                         observation_snapshot={key: value.tolist()
                                                               for key, value in batch.items()},
                                         predicted_probabilities=probabilities.tolist(),
                                         legal_actions=[np.flatnonzero(row).tolist() for row in masks],
                                         invalid_probability_mass=float(probabilities[~masks].sum())))
                return dict(cases=evidence, behavioral_quality="untested")

            check("masked_inference", masked_inference)

            def bootstrap():
                _require(model is not None, "policy initialization failed")
                _require(len(truncation_infos) == len(terminal_infos) == 2,
                         "both real terminal and truncation results are required")
                terminal_mask = [bool(info["TimeLimit.truncated"]) for info in terminal_infos]
                truncated_mask = [bool(info["TimeLimit.truncated"]) for info in truncation_infos]
                _require(terminal_mask == [False, False] and truncated_mask == [True, True],
                         "bootstrap eligibility must distinguish real terminals and time limits")
                values, turns, masks = [], [], []
                for info in truncation_infos:
                    final = info["terminal_observation"]
                    _require(final["action_mask"].any() and final["context"][2] == 0,
                             "bootstrap requires the live final observation and its legal mask")
                    _require(final["context"][0] == 1, "bootstrap must use turn 1, not autoreset turn 0")
                    tensor, _ = model.policy.obs_to_tensor(final)
                    with torch.no_grad():
                        value = float(model.policy.predict_values(tensor).item())
                    _require(np.isfinite(value), "final observation has a nonfinite bootstrap value")
                    values.append(value)
                    turns.append(int(final["context"][0]))
                    masks.append(np.flatnonzero(final["action_mask"]).tolist())
                return dict(terminal_bootstrap_mask=terminal_mask, truncation_bootstrap_mask=truncated_mask,
                            final_observation_turns=turns, final_observation_legal_actions=masks,
                            final_observation_values=values,
                            scope="SB3 bootstrap inputs and value path; no rollout collection or training")

            check("truncation_bootstrap", bootstrap)

            def immutable():
                _require(model is not None and "digest" in model_state, "policy initialization failed")
                _require(_policy_digest(model.policy) == model_state["digest"],
                         "inference changed policy parameters or buffers")
                _require(model.num_timesteps == model_state["timesteps"] and
                         model._n_updates == model_state["updates"], "policy training counters changed")
                if "checkpoint_path" in model_state:
                    current = hashlib.sha256(model_state["checkpoint_path"].read_bytes()).hexdigest()
                    _require(current == model_state["checkpoint_sha256"], "source checkpoint bytes changed")
                return dict(parameters_and_buffers_unchanged=True, training_counters_unchanged=True,
                            checkpoint_bytes_unchanged=True if checkpoint is not None else None,
                            policy_state_sha256=model_state["digest"], checkpoint_writes=0)

            check("policy_immutable", immutable)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)

    def bounded_logs():
        files = []
        maximum = max(1, int(log_max_mb * 1024 * 1024))
        for case in ("terminal", "replacement", "truncation"):
            for filename in ("battle-readable.log", "battle.log", "showdown-stderr.log"):
                path = directory / case / filename
                _require(path.is_file() and path.stat().st_size > 0, f"missing diagnostic log: {path}")
                related = [path, *path.parent.glob(path.name + ".*")]
                _require(len(related) <= log_backups + 1, f"too many rotated logs: {path}")
                _require(all(item.stat().st_size <= maximum for item in related), f"oversized log: {path}")
                files.extend(str(item.relative_to(output_dir)) for item in related)
        _require(not list(directory.glob("*/episode-*")), "full per-step episode logs must remain disabled")
        return dict(readable_log=True, battle_log=True, step_logs=False,
                    max_bytes_per_file=maximum, backups=log_backups, files=files)

    check("bounded_logs", bounded_logs)
    return checks
