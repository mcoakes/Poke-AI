"""Independent native-state checks for the opt-in Showdown v2 observation.

Legacy audits stay in their original modules. These checks exercise listed
fixtures, not every Showdown mechanic or the quality of a policy's decisions.
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

from .audit_cases import BattleAudit, key, major_status, mon, reference_mask, same
from .audit_pipeline import _policy_digest, _require, _same_observation
from .showdown_env import showdown_observation, spectator_lines


STATS = ("atk", "def", "spa", "spd", "spe")
BOOSTS = (*STATS, "accuracy", "evasion")
WEATHER = {"": 0, "sunnyday": 1, "raindance": 2, "sandstorm": 3, "hail": 4}


def observation_inventory_v2():
    """Describe the added contract without relabeling v1 omissions as v2 gaps."""
    rows = [
        ("both_party_state", "v2_party", "Exact HP/max HP, level, base types, item, ability, fainted state and status for both parties", "Complete supplied teams are deliberately visible."),
        ("stored_stats", "v2_stats / v2_active_stats", "Five unboosted base party stats and current active stored stats", "These are stored stats, not final event-modified damage or speed values."),
        ("active_hp_item_ability", "v2_active", "Exact active HP/max HP and current item/ability for both players", "Temporary active state remains separate from base party stats/types."),
        ("stat_stages", "v2_boosts", "Seven signed stages / 6 for each active", "Stages are explicit; effective multipliers are not precomputed."),
        ("move_metadata", "v2_move_details", "Power, accuracy, always-hit flag, category, priority and computed type for both teams", "Dex metadata is not a prediction of target-dependent power, hit chance or final priority."),
        ("weather", "v2_weather", "Identity, remaining duration and permanent flag", "Weather does not describe terrain, side conditions or every field effect."),
        ("both_major_statuses", "v2_party status bits", "Exact integer bit expansion for twelve party slots", "Only statuses supported by the report contract are represented."),
        ("both_active_volatiles", "v2_volatiles", "Exact integer bit expansion for both active battlers", "The cartridge-style bitfield represents a documented subset of simulator volatiles."),
    ]
    return [dict(name=name, status="present", source=source,
                 policy_representation=representation, limitation=note)
            for name, source, representation, note in rows]


def _native_volatile(pokemon, absolute_side):
    """Interpret native effect counters/flags without consuming adapter values."""
    effects = pokemon["volatiles"]
    result = 0
    for name, shift, limit, counter in (("confusion", 0, 7, "time"),
            ("uproar", 4, 7, "duration"), ("bide", 8, 3, "duration"),
            ("lockedmove", 10, 3, "duration"), ("partiallytrapped", 13, 7, "duration")):
        if name in effects:
            result |= max(1, min(limit, effects[name].get(counter, 1) or 1)) << shift
    flags = {"flinch": 3, "focusenergy": 20, "mustrecharge": 22, "rage": 23,
             "substitute": 24, "destinybond": 25, "trapped": 26, "nightmare": 27,
             "curse": 28, "foresight": 29, "defensecurl": 30, "torment": 31}
    for name, bit in flags.items():
        if name in effects:
            result |= 1 << bit
    if pokemon["transformed"]:
        result |= 1 << 21
    if any(name in effects for name in ("twoturnmove", "lockedmove", "bide", "uproar", "rollout", "iceball")):
        result |= 1 << 12
    if "attract" in effects:
        result |= 1 << (17 - absolute_side)
    return result


class NativeV2Audit(BattleAudit):
    """Every captured v2 value is compared with separate audit_inspect facts."""

    def capture(self):
        self.oracle = self.inspect()
        self.observations = [showdown_observation(r, self.raw["ended"], version="v2")
                             for r in self.raw["reports"]]
        self.checkpoints += 1
        self.log.append(json.dumps(dict(case=self.case, checkpoint=self.checkpoints,
            oracle=self.oracle, observations=[{k: v.tolist() for k, v in o.items()}
                                               for o in self.observations])) + "\n")
        self.battle_log.append(f"\n[{self.case}]\n" + "\n".join(
            spectator_lines(self.raw.get("protocol", []), self.hp)) + "\n")
        for perspective, observation in enumerate(self.observations):
            self.verify_observation(perspective, observation)
        return self.observations

    def verify_observation(self, perspective, obs):
        from .observation_v2 import versioned_raw_space
        from .ppo_features import policy_observation, policy_space

        prefix = f"P{perspective+1} v2"
        _require(versioned_raw_space("v2").contains(obs), f"{prefix} raw schema mismatch")
        policy = policy_observation(obs, version="v2")
        _require(policy_space("v2").contains(policy), f"{prefix} policy schema mismatch")
        for relative, absolute in enumerate((perspective, 1-perspective)):
            side = self.oracle["sides"][absolute]
            for slot in range(6):
                if slot >= len(side["party"]):
                    for field in ("v2_party", "v2_stats", "v2_move_details"):
                        same(bool(obs[field][relative, slot].any()), False,
                             f"{prefix} unused {field}/{relative}/{slot}")
                    continue
                p = side["party"][slot]
                item = self.ids["items"][key(p["item"]["name"])] if p["item"]["id"] else 0
                expected = [1, p["level"], p["hp"], p["maxhp"], *self.types(p["base_types"]),
                            item, p["ability"]["num"], int(p["fainted"]), major_status(p)]
                same(obs["v2_party"][relative, slot], expected,
                     f"{prefix} party side {relative} slot {slot+1}")
                same(policy["v2_status_bits"][relative, slot],
                     [(major_status(p) >> bit) & 1 for bit in range(32)],
                     f"{prefix} native major-status bit expansion side {relative} slot {slot+1}")
                same(obs["v2_stats"][relative, slot],
                     [p["base_stored_stats"][name] for name in STATS],
                     f"{prefix} base stored stats side {relative} slot {slot+1}")
                moves = p["moves"] if p["active"] else p["base_moves"]
                expected_moves = [[0]*6 for _ in range(4)]
                for index, move in enumerate(moves):
                    always = move["accuracy"] is True
                    move_type, power = move["type"], move["power"]
                    category = {"Physical": 0, "Special": 1, "Status": 2}[move["category"]]
                    if move["id"] == "hiddenpower":
                        move_type = p["hp_type"] if p["active"] else p["base_hp_type"]
                        power = p["hp_power"] if p["active"] else p["base_hp_power"]
                        category = int(move_type in ("Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark"))
                    expected_moves[index] = [power, 0 if always else move["accuracy"],
                        int(always), category, move["priority"], self.ids["types"][key(move_type)]]
                same(obs["v2_move_details"][relative, slot], expected_moves,
                     f"{prefix} native move metadata side {relative} slot {slot+1}")
            p = side["party"][side["active_slot"]-1]
            item = self.ids["items"][key(p["item"]["name"])] if p["item"]["id"] else 0
            same(obs["v2_active"][relative], [p["hp"], p["maxhp"], item, p["ability"]["num"]],
                 f"{prefix} exact active HP/item/ability side {relative}")
            same(obs["v2_active_stats"][relative], [p["stored_stats"][name] for name in STATS],
                 f"{prefix} active stored stats side {relative}")
            same(obs["v2_boosts"][relative], [p["boosts"][name] for name in BOOSTS],
                 f"{prefix} native boosts side {relative}")
            native_volatile = _native_volatile(p, absolute)
            same(int(obs["v2_volatiles"][relative]), native_volatile,
                 f"{prefix} native volatile counters/flags side {relative}")
            same(policy["v2_volatile_bits"][relative],
                 [(native_volatile >> bit) & 1 for bit in range(32)],
                 f"{prefix} native volatile bit expansion side {relative}")
        field = self.oracle["field"]
        weather = field["weather"]
        duration = field["weather_state"].get("duration", 0) or 0
        same(obs["v2_weather"], [WEATHER[weather], duration, int(bool(weather) and not duration)],
             f"{prefix} native weather")
        same(obs["action_mask"], reference_mask(self.oracle["sides"][perspective], self.oracle["ended"]),
             f"{prefix} native action mask")


def _feature_sensitivity(observation):
    """Native-validated input ablations isolate the added feature channels."""
    import torch
    from stable_baselines3.common.preprocessing import preprocess_obs
    from .ppo_features import BattleFeatures, policy_observation, policy_space

    schema = policy_space("v2")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1729)
        extractor = BattleFeatures(schema, features_dim=8).eval()
    def flatten(raw):
        converted = policy_observation(raw, version="v2")
        tensors = {k: torch.as_tensor(v).unsqueeze(0) for k, v in converted.items()}
        with torch.no_grad():
            return extractor._flatten(preprocess_obs(tensors, schema))
    before = flatten(observation)
    probes = [("v2_party", (1, 1, column)) for column in range(1, 10)]
    probes += [(field, index) for field, indices in (
        ("v2_stats", [(1, 1, n) for n in range(5)]),
        ("v2_active_stats", [(1, n) for n in range(5)]),
        ("v2_active", [(1, n) for n in range(4)]),
        ("v2_boosts", [(1, n) for n in range(7)]),
        ("v2_move_details", [(1, 1, 0, n) for n in range(6)]),
        ("v2_weather", [(n,) for n in range(3)]),
        ("v2_volatiles", [(0,), (1,)])) for index in indices]
    evidence = []
    for field, index in probes:
        changed = {name: value.copy() for name, value in observation.items()}
        value = int(changed[field][index])
        changed[field][index] = 0 if value else 1
        count = int(torch.count_nonzero(before != flatten(changed)).item())
        _require(count > 0, f"{field}{index} does not reach BattleFeatures")
        evidence.append(dict(field=field, index=list(index), changed_features=count))
    # These explicit synthetic high/low combinations test float safety; they do
    # not claim that these major-status combinations occur in a legal battle.
    for field, index in (("v2_party", (1, 1, 9)), ("v2_volatiles", (1,))):
        high = {name: value.copy() for name, value in observation.items()}
        high[field][index] = 1 << 31
        first = flatten(high)
        high[field][index] |= 1
        count = int(torch.count_nonzero(first != flatten(high)).item())
        _require(count == 1, f"{field} loses the low bit next to bit 31")
    return dict(probes=evidence, high_and_low_bits_preserved=True,
                scope="Single-field ablations of native-validated input, before dense projection; representability only")


def run_v2_battle_checks(root, output_dir, *, node="node", timeout=30, seed=26002,
                         log_max_mb=2, log_backups=1):
    audit = NativeV2Audit(root, output_dir, node=node, timeout=timeout, seed=seed,
                         log_max_mb=log_max_mb, log_backups=log_backups)
    rows = []
    def check(name, action):
        audit.case = name
        before = audit.checkpoints
        try:
            details = action() or {}
            rows.append(dict(name="v2_"+name, status="pass",
                             details=dict(snapshots=audit.checkpoints-before, **details)))
        except Exception as error:
            rows.append(dict(name="v2_"+name, status="fail", details=dict(
                error=f"{type(error).__name__}: {error}", snapshots=audit.checkpoints-before)))

    def metadata():
        audit.reset([mon("Mew", ["Tackle", "Surf", "Quick Attack", "Roar"], held_item="Leftovers"),
                     mon("Blastoise", ["Swift", "Toxic", "Hidden Power"], nature="Modest", evs={"spa": 252, "hp": 252})],
                    [mon("Blissey", ["Splash", "Protect"]),
                     mon("Lapras", ["Ice Beam", "Rest", "Hidden Power"], ivs={"atk": 30, "def": 30}, held_item="Sitrus Berry")])
        same(audit.inspect(), audit.inspect(), "v2 native inspection is read-only")
        details = _feature_sensitivity(audit.observations[0])
        audit.step([5, 5])
        return dict(**details, hidden_power_base_and_active=True)

    def boosts():
        for side in range(2):
            teams = [[mon("Mew", ["Swords Dance", "Agility", "Calm Mind", "Double Team"])],
                     [mon("Blissey", ["Growl", "Tail Whip", "Sand Attack", "Splash"])]]
            if side:
                teams.reverse()
            audit.reset(*teams)
            for action in range(4):
                audit.step([action, action])
            expected = dict(atk=1, def_=-1, spa=1, spd=1, spe=2, accuracy=-1, evasion=1)
            same([audit.active(side)["boosts"][name] for name in BOOSTS], list(expected.values()),
                 "seven native boost fixture values")
        return dict(player_sides=[1, 2], stages=list(BOOSTS), positive_and_negative=True)

    def weather():
        for move, native in (("Sunny Day", "sunnyday"), ("Rain Dance", "raindance"),
                             ("Sandstorm", "sandstorm"), ("Hail", "hail")):
            audit.reset([mon("Mew", [move, "Splash"])], [mon("Blissey", ["Splash"])])
            audit.step([0, 0])
            same(audit.oracle["field"]["weather"], native, "native weather established")
            _require(audit.oracle["field"]["weather_state"].get("duration", 0) > 0,
                     "move weather must have a remaining duration")
            for _ in range(4):
                audit.step([1, 0])
            same(audit.oracle["field"]["weather"], "", "weather expires")
        audit.reset([mon("Tyranitar", ["Splash"])], [mon("Blissey", ["Splash"])])
        same(audit.observations[0]["v2_weather"], [3, 0, 1], "permanent Sand Stream")
        return dict(timed_weather=list(WEATHER)[1:], expiry=True, permanent_ability_weather="sandstorm")

    def bench_status():
        audit.reset([mon("Mew", ["Toxic", "Splash"])],
                    [mon("Lapras", ["Splash"]), mon("Blastoise", ["Splash"], held_item="Leftovers")])
        for _ in range(15):
            audit.step([0, 0])
            if audit.active(1)["status"] == "tox":
                break
        same(audit.active(1)["status"], "tox", "native Toxic established")
        damaged = audit.active(1)["hp"]
        _require(damaged < audit.active(1)["maxhp"], "Toxic must damage the target")
        audit.step([1, 5])
        bench = audit.oracle["sides"][1]["party"][0]
        same(bench["hp"], damaged, "damaged bench retains exact HP")
        same(bench["status"], "tox", "bench retains status")
        return dict(opponent_bench_hp=damaged, status="tox", original_slot=1)

    def fainted():
        audit.reset([mon("Magikarp", ["Splash"], 1), mon("Lapras", ["Splash"])],
                    [mon("Blastoise", ["Surf", "Splash"])])
        audit.step([0, 0])
        same(audit.active(0)["fainted"], True, "native faint fixture")
        audit.step([5, 10])
        same(audit.observations[1]["v2_party"][1, 0, [2, 8]], [0, 1], "opponent fainted bench")

    def changed_items_abilities():
        audit.reset([mon("Mew", ["Trick", "Skill Swap"], held_item="Leftovers")],
                    [mon("Blissey", ["Splash"], held_item="Choice Band")])
        before = [audit.active(i)["ability"]["num"] for i in range(2)]
        audit.step([0, 0])
        same(audit.active(0)["item"]["id"], "choiceband", "Trick changed item")
        # Choice Band locks Trick, so begin a separate Skill Swap fixture.
        audit.reset([mon("Mew", ["Skill Swap"])], [mon("Blissey", ["Splash"])])
        audit.step([0, 0])
        same([audit.active(i)["ability"]["num"] for i in range(2)], before[::-1], "Skill Swap changed abilities")

    def volatiles():
        for side in range(2):
            teams = [[mon("Mew", ["Substitute", "Defense Curl"])], [mon("Blissey", ["Torment"])]]
            if side:
                teams.reverse()
            audit.reset(*teams)
            audit.step([0, 0])
            _require("substitute" in audit.active(side)["volatiles"], "Substitute fixture missing")
            _require("torment" in audit.active(side)["volatiles"], "Torment fixture missing")
            audit.step([0, 1] if side else [1, 0])
            _require("defensecurl" in audit.active(side)["volatiles"], "Defense Curl fixture missing")
            teams = [[mon("Mew", ["Splash"])], [mon("Blissey", ["Confuse Ray"])]]
            if side:
                teams.reverse()
            audit.reset(*teams)
            audit.step([0, 0])
            _require(2 <= audit.active(side)["volatiles"]["confusion"]["time"] <= 5,
                     "confusion counter fixture missing")
        return dict(player_sides=[1, 2], native_flags=[24, 30, 31], native_confusion_counter=True)

    def transform():
        audit.reset([mon("Mew", ["Transform", "Psychic", "Protect", "Rest"])],
                    [mon("Magikarp", ["Splash"])])
        base = audit.observations[0]["v2_stats"][0, 0].copy()
        audit.step([0, 0])
        same(audit.active(0)["transformed"], True, "Transform fixture")
        same(audit.observations[0]["v2_stats"][0, 0], base.tolist(), "Transform preserves party base stats")
        _require(not np.array_equal(base, audit.observations[0]["v2_active_stats"][0]),
                 "Transform should change active stored stats")
        same(bool(audit.observations[0]["v2_move_details"][0, 0, 1:].any()), False,
             "removed Transform move metadata is cleared")

    try:
        for name, work in (("native_metadata_and_feature_sensitivity", metadata),
                           ("signed_boosts", boosts), ("weather_duration_and_expiry", weather),
                           ("damaged_status_bench", bench_status), ("fainted_bench", fainted),
                           ("changed_items_and_abilities", changed_items_abilities),
                           ("both_volatile_flags_and_counters", volatiles),
                           ("transformed_stats_and_move_metadata", transform)):
            check(name, work)
    finally:
        audit.close()
    return rows


@contextmanager
def _v2_vector(root, directory, parties, *, node, timeout, seed, max_steps,
               log_max_mb, log_backups):
    from .ppo_vec_env import LinkedVecEnv
    from .showdown_env import ShowdownBattleSession, ShowdownFactory
    directory.mkdir(parents=True, exist_ok=False)
    teams = []
    for index, party in enumerate(parties):
        path = directory / f"team-{index+1}.json"
        path.write_text(json.dumps(dict(name=f"v2-audit-{index+1}", party=party)) + "\n")
        teams.append(path)
    cancel = Event()
    args = SimpleNamespace(seed=seed, node=node, command_timeout=timeout,
        log_max_mb=log_max_mb, log_backups=log_backups, step_logs=False,
        keep_episode_logs=0, observation_version="v2", session_observation_version="v2")
    factory = ShowdownFactory(root, directory, [55355, 55356], lambda: teams, args, cancel)
    session = ShowdownBattleSession(factory, max_steps=max_steps,
        readable_log=directory / "battle-readable.log", battle_log=directory / "battle.log")
    events = []
    def completed(pair, **fields):
        events.append(dict(reports=deepcopy(session.reports), episode=factory.episode,
                           terminated=fields["terminated"], truncated=fields["truncated"]))
    vec = None
    try:
        vec = LinkedVecEnv([session], cancel=cancel, aborters=[factory.stop], on_episode=completed,
                           observation_version="v2")
        vec.seed(seed)
        yield vec, session, events
    finally:
        if vec is not None:
            vec.close()
        else:
            session.close()


def run_v2_pipeline_checks(root, output_dir, *, node="node", timeout=30, checkpoint=None,
                            log_max_mb=2, log_backups=1, seed=26002):
    """Read-only MaskablePPO inference and final-observation paths on real v2 vectors."""
    import torch
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from .ppo_features import BattleFeatures, observation_version, policy_observation, policy_space
    _require(np.isfinite(log_max_mb) and log_max_mb > 0 and log_backups >= 0, "invalid log bounds")
    root, output_dir = Path(root), Path(output_dir)
    directory = output_dir / "pipeline-v2"
    directory.mkdir(parents=True, exist_ok=False)
    rows, samples, state = [], [], {}
    model = None
    def check(name, work):
        try:
            details = work() or {}
            rows.append(dict(name="v2_pipeline_"+name, status="pass", details=details))
        except Exception as error:
            rows.append(dict(name="v2_pipeline_"+name, status="fail",
                             details=dict(error=f"{type(error).__name__}: {error}")))
    def load_policy(vec):
        nonlocal model
        if checkpoint:
            path = Path(checkpoint).expanduser().resolve()
            content = path.read_bytes()
            state.update(path=path, checkpoint_sha256=hashlib.sha256(content).hexdigest())
            model = MaskablePPO.load(BytesIO(content), env=vec, device="cpu")
        else:
            model = MaskablePPO("MultiInputPolicy", vec, n_steps=2, batch_size=4,
                n_epochs=1, device="cpu", seed=seed,
                policy_kwargs=dict(features_extractor_class=BattleFeatures, net_arch=[32]))
        actual = observation_version(model.observation_space)
        same(actual, "v2", "actual checkpoint observation schema")
        same(model.observation_space, policy_space("v2"), "checkpoint observation space")
        state.update(digest=_policy_digest(model.policy), timesteps=model.num_timesteps, updates=model._n_updates)
        model.policy.set_training_mode(False)
        return dict(source=str(checkpoint) if checkpoint else "temporary untrained MaskablePPO",
                    observation_version=actual, trained_checkpoint=bool(checkpoint),
                    checkpoint_sha256=state.get("checkpoint_sha256"), num_timesteps=int(model.num_timesteps),
                    training_performed=False, checkpoint_writes=0)
    common = dict(node=node, timeout=timeout, seed=seed, log_max_mb=log_max_mb, log_backups=log_backups)
    def save_sample(label, batch):
        for index in range(2):
            _require(policy_space("v2").contains({k: v[index] for k, v in batch.items()}),
                     f"{label}: vector observation outside v2 policy space")
        samples.append((label, {k: v.copy() for k, v in batch.items()}))
    def boundary(terminated):
        label = "terminal" if terminated else "truncation"
        teams = ([[mon("Magikarp", ["Splash"], 1)], [mon("Blastoise", ["Surf"])]] if terminated
                 else [[mon("Mew", ["Rain Dance"])], [mon("Blissey", ["Splash"])]])
        with _v2_vector(root, directory / label, teams, max_steps=10 if terminated else 1, **common) as bundle:
            vec, session, events = bundle
            before = vec.reset()
            save_sample(label+"_initial", before)
            same(get_action_masks(vec), before["action_mask"].astype(bool).tolist(), "vector action masks")
            if terminated:
                check("policy_load", lambda: load_policy(vec))
            reset, _, dones, infos = vec.step(np.array([0, 0]))
            same(dones.tolist(), [True, True], "paired episode completed")
            _require(len(events) == 1 and session.factory.episode == events[0]["episode"]+1,
                     "paired autoreset must happen once")
            values = []
            for player, info in enumerate(infos):
                final = info["terminal_observation"]
                expected = policy_observation(showdown_observation(
                    events[0]["reports"][player], terminated, version="v2"), version="v2")
                _require(_same_observation(final, expected), "terminal_observation lost pre-reset v2 fields")
                _require(not _same_observation(final, {k: v[player] for k, v in reset.items()}),
                         "final observation was overwritten by autoreset")
                same(info["TimeLimit.truncated"], not terminated, "bootstrap eligibility")
                same(bool(final["action_mask"].any()), not terminated, "final legal mask")
                if not terminated:
                    _require(final["v2_weather"][0] == 2, "final observation lost Rain Dance before autoreset")
                    _require(model is not None, "policy initialization failed")
                    tensor, _ = model.policy.obs_to_tensor(final)
                    with torch.no_grad():
                        value = float(model.policy.predict_values(tensor).item())
                    _require(np.isfinite(value), "nonfinite final bootstrap value")
                    values.append(value)
            if not terminated:
                save_sample("truncation_final", vec.stack([i["terminal_observation"] for i in infos]))
            save_sample(label+"_autoreset", reset)
            return dict(terminal_observation_before_reset=True, observation_version="v2", autoresets=1,
                        bootstrap_eligible=[not terminated]*2, final_values=values,
                        scope="Final-state and value path; no training rollouts")
    def replacement():
        parties = [[mon("Magikarp", ["Splash"], 1), mon("Lapras", ["Splash"])], [mon("Blastoise", ["Surf"])]]
        with _v2_vector(root, directory / "replacement", parties, max_steps=10, **common) as bundle:
            vec, _, _ = bundle
            vec.reset()
            forced, _, dones, _ = vec.step(np.array([0, 0]))
            _require(not dones.any(), "replacement fixture ended early")
            same([np.flatnonzero(row).tolist() for row in forced["action_mask"]], [[5], [10]], "replacement legality")
            save_sample("forced_replacement", forced)
            after, _, _, _ = vec.step(np.array([5, 10]))
            save_sample("replacement_completed", after)
            return dict(legal_actions=[[5], [10]], scripted_replacement=[5, 10])
    def inference():
        _require(model is not None and samples, "policy or observations unavailable")
        evidence = []
        for label, batch in samples:
            mask = batch["action_mask"].astype(bool)
            tensor, _ = model.policy.obs_to_tensor(batch)
            with torch.no_grad():
                probs = model.policy.get_distribution(tensor, action_masks=mask).distribution.probs.cpu().numpy()
                values = model.policy.predict_values(tensor).cpu().numpy()
            _require(np.isfinite(probs).all() and np.isfinite(values).all(), "nonfinite inference")
            _require(np.allclose(probs.sum(1), 1) and (probs[~mask] == 0).all(), "invalid masked probabilities")
            actions, _ = model.predict(batch, action_masks=mask, deterministic=True)
            _require(all(mask[i, action] for i, action in enumerate(actions)), "illegal predicted action")
            evidence.append(dict(observation=label, predicted_actions=actions.tolist(),
                predicted_probabilities=probs.tolist(), legal_actions=[np.flatnonzero(row).tolist() for row in mask],
                invalid_probability_mass=float(probs[~mask].sum()), values=values.tolist()))
        return dict(cases=evidence, behavioral_quality="untested", observation_version="v2")
    def immutable():
        _require(model is not None and "digest" in state, "policy initialization failed")
        same(_policy_digest(model.policy), state["digest"], "policy parameters/buffers unchanged")
        same([model.num_timesteps, model._n_updates], [state["timesteps"], state["updates"]], "training counters unchanged")
        if checkpoint:
            same(hashlib.sha256(state["path"].read_bytes()).hexdigest(), state["checkpoint_sha256"], "source checkpoint bytes unchanged")
        return dict(parameters_and_buffers_unchanged=True, training_counters_unchanged=True,
                    checkpoint_bytes_unchanged=True if checkpoint else None, checkpoint_writes=0)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            check("terminal_autoreset", lambda: boundary(True))
            check("forced_replacement_wait", replacement)
            check("truncation_bootstrap", lambda: boundary(False))
            check("masked_inference", inference)
            check("policy_immutable", immutable)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    def bounded_logs():
        maximum = max(1, int(log_max_mb*1024*1024))
        files = []
        for case in ("terminal", "replacement", "truncation"):
            for filename in ("battle-readable.log", "battle.log", "showdown-stderr.log"):
                path = directory / case / filename
                related = [path, *path.parent.glob(path.name+".*")]
                _require(path.is_file() and path.stat().st_size > 0, f"missing {path}")
                _require(len(related) <= log_backups+1 and all(p.stat().st_size <= maximum for p in related),
                         f"unbounded {path}")
                files.extend(str(p.relative_to(output_dir)) for p in related)
        _require(not list(directory.glob("*/episode-*")), "step logs must remain disabled")
        return dict(files=files, max_bytes_per_file=maximum, backups=log_backups, step_logs=False)
    check("bounded_logs", bounded_logs)
    return rows
