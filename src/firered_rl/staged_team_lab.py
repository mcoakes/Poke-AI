"""Fixed-team curriculum assets, immutable checks, and bounded frozen-policy audits.

Directory stages deliberately use the production sampler: two distinct pool
entries sampled without replacement. They are varied pairings, not identical
mirrors and not independent draws with replacement.
"""
from collections import Counter
import copy
import hashlib
from itertools import permutations, product
import json
import math
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np

from .evaluation import create_suite, digest, evaluate, materialize_assets, validate_suite
from .fixed_team_lab import GreedyDamage, make_suite, pokemon, teams
from .linked_reward import LinkedRewardConfig
from .periodic_evaluation import atomic_write
from .ppo_features import observation_version, policy_for_version
from .readable_log import names
from .rule_opponent import RuleBasedOpponent
from .showdown_env import ShowdownBattleSession, ShowdownFactory
from .training_teams import TrainingTeams


COMPOSITION_HOLDOUT = {"001", "110"}
SPECIES_CHOICES = (("Typhlosion", "Arcanine"), ("Blastoise", "Vaporeon"),
                   ("Sceptile", "Venusaur"))
STAB_MOVES = dict(Typhlosion="Flamethrower", Arcanine="Flamethrower",
                  Blastoise="Surf", Vaporeon="Surf", Sceptile="LeafBlade",
                  Venusaur="RazorLeaf")


def _stage_name(stage):
    return "mirror" if stage in ("mirror-discovery", "mirror-stability") else stage


def _write_team(inputs, relative, value):
    path = inputs / relative
    content = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"pinned curriculum asset differs: {path}")
    if not path.exists():
        path.write_text(content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_assets(inputs):
    """Write fixed assets and return JSON-ready entries with input-relative paths.

    Each entry's root is absolute. Join root / train_dir for a trainer directory,
    or use asset_paths(entry) for the fixed pair. Existing differing files fail.
    The original 000 composition is in training; 001 and 110 are held out.
    """
    inputs = Path(inputs).resolve()
    result = {}
    for stage in ("move-choice", "mirror", "order", "composition"):
        base = Path("staged-teams") / stage
        entry = dict(root=str(inputs), fixed=stage in ("move-choice", "mirror"),
                     train_dir=str(base / "train"), train_paths=[], holdout_dir=None,
                     holdout_paths=[], sha256={},
                     pair_sampling="fixed" if stage in ("move-choice", "mirror") else
                         "two_distinct_pool_entries_without_replacement")
        records = []
        if entry["fixed"]:
            records = [("train", f"p{i+1}", team) for i, team in enumerate(teams()[stage])]
        elif stage == "order":
            party = teams()["mirror"][0]["party"]
            for order in permutations(range(3)):
                key = "".join(str(i) for i in order)
                records.append(("train", key, dict(name=f"order-{key}",
                    party=[copy.deepcopy(party[i]) for i in order])))
        else:
            entry["holdout_dir"] = str(base / "holdout")
            entry["heldout_compositions"] = sorted(COMPOSITION_HOLDOUT)
            entry["training_compositions"] = []
            for choice in product(range(2), repeat=3):
                code = "".join(str(i) for i in choice)
                split = "holdout" if code in COMPOSITION_HOLDOUT else "train"
                if split == "train":
                    entry["training_compositions"].append(code)
                mons = [pokemon(SPECIES_CHOICES[i][j],
                                ["Strength", STAB_MOVES[SPECIES_CHOICES[i][j]]])
                        for i, j in enumerate(choice)]
                for order in permutations(range(3)):
                    key = code + "-" + "".join(str(i) for i in order)
                    records.append((split, key, dict(name=f"composition-{key}",
                        party=[copy.deepcopy(mons[i]) for i in order])))
        for split, key, team in records:
            relative = str(base / split / f"{key}.json")
            entry["sha256"][relative] = _write_team(inputs, relative, team)
            entry[f"{split}_paths"].append(relative)
        result[stage] = entry
    return result


def asset_paths(entry, *, heldout=False):
    """Resolve the frozen entry's relative asset paths without changing its data."""
    key = "holdout_paths" if heldout else "train_paths"
    return tuple(Path(entry["root"]) / path for path in entry[key])


def validate_assets(entry):
    """Reject a changed pinned input before starting or continuing a stage."""
    root = Path(entry["root"])
    for split in ("train", "holdout"):
        directory = entry[f"{split}_dir"]
        if directory is None:
            continue
        expected_paths = {root / path for path in entry[f"{split}_paths"]}
        if set((root / directory).glob("*.json")) != expected_paths:
            raise ValueError(f"pinned curriculum asset list changed: {root / directory}")
    for relative, expected in entry["sha256"].items():
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned curriculum asset changed: {path}")


def make_stage_suite(project, entry, episodes, seed, stage, *, permute_moves=False,
                     heldout=False):
    """Balanced candidate seats, fixed schedule, and optional candidate slot rotation.

    Held-out composition tests draw BOTH teams from the held-out composition
    pool. Ordinary tests draw both from the training pool with fresh seeds.
    A permuted suite must be called with the same seed and count as its ordinary
    counterpart: battle randomness and team assignments then remain identical.
    """
    stage = _stage_name(stage)
    validate_assets(entry)
    if heldout and (stage != "composition" or not entry["holdout_paths"]):
        raise ValueError("held-out composition check is only available for composition")
    if entry["fixed"]:
        suite = make_suite(project, asset_paths(entry), episodes, seed, stage,
                           permute_moves=permute_moves)
    else:
        directory = entry["holdout_dir" if heldout else "train_dir"]
        source = TrainingTeams(directory=Path(entry["root"]) / directory, seed=seed)
        suite = create_suite(source, backend="showdown", episodes=episodes, seed=seed,
                             swap_sides=True, max_steps=100)
        if permute_moves:
            original = copy.deepcopy(suite["assets"])
            for case in suite["cases"]:
                seat = case["candidate_player"] - 1
                asset = copy.deepcopy(original[case["assets"][seat]])
                for mon in asset["team"]["party"]:
                    shift = case["matchup"] % len(mon["moves"])
                    mon["moves"] = mon["moves"][shift:] + mon["moves"][:shift]
                key = digest(asset)
                suite["assets"][key] = asset
                case["assets"] = list(case["assets"])
                case["assets"][seat] = key
            # Candidate-only rotations give the two seat cases different asset
            # lists; seed/matchup/semantic pairing and balanced seats are retained.
            suite["swap_sides"] = False
        suite["rule_opponent"] = RuleBasedOpponent(project).identity
    suite["diagnostic"] = dict(stage=stage, balanced_candidate_seats=True,
        heldout_compositions=heldout, permuted_moves=permute_moves,
        pair_sampling=entry["pair_sampling"],
        candidate_keeps_team=stage == "move-choice")
    return validate_suite(suite)


def json_native(value):
    """Make all nested audit values strict JSON, including NumPy diagnostics."""
    if isinstance(value, np.ndarray):
        return json_native(value.tolist())
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class StageDecisionAudit:
    """All-battle counts plus traces bounded by case and decision, never by timing.

    The production evaluator invokes observers on its central inference thread.
    Worker completion order cannot determine which cases are retained.
    """
    # These Gen III moves use special typing/delayed or indirect execution that
    # the base descriptors do not fully describe. Power <= 1 and status moves
    # below also exclude fixed-damage/counter/OHKO/called-move mechanics.
    MATCHUP_EXCLUDED_MOVES = frozenset({165, 248, 251, 267, 311, 353})
    MATCHUP_COUNTERS = (
        "matchup_classified_move_actions", "matchup_unclassified_move_actions",
        "current_target_immune_move_actions", "current_target_type_immune_move_actions",
        "current_target_ability_immune_move_actions",
        "current_target_immune_with_nonimmune_alternative",
    )

    def __init__(self, project, candidate, *, trace_cases=16, trace_decisions=24):
        self.candidate = candidate
        self.rules = RuleBasedOpponent(project)
        self.names = names("firered_us_v1_0")
        self.trace_cases, self.trace_decisions = trace_cases, trace_decisions
        self.counts, self.by_episode, self.first_moves = Counter(), {}, Counter()
        self.counts.update(dict.fromkeys(self.MATCHUP_COUNTERS, 0))
        self.moves, self.switches, self.rows = Counter(), Counter(), []

    def label(self, group, value):
        return self.names.get(group, {}).get(str(int(value)), str(int(value)))

    def action_data(self, obs, action):
        action = int(action)
        if action == 10:
            return dict(kind="wait", action=action)
        if action >= 4:
            species = int(obs["party"][action - 4, 1])
            return dict(kind="switch", action=action, party_slot=action - 3,
                        species_id=species, species=self.label("species", species))
        slot = int(obs["active"][0, 8]) - 1
        move = int(obs["moves"][slot, action, 0]) if 0 <= slot < 6 else 0
        return dict(kind="move", action=action, move_slot=action + 1,
                    move_id=move, move=self.label("moves", move))

    def probabilities(self, obs):
        policy = getattr(self.candidate, "policy", None)
        if policy is None or not hasattr(policy, "get_distribution"):
            return None
        import torch
        version = observation_version(self.candidate.observation_space)
        with torch.no_grad():
            tensor, _ = policy.obs_to_tensor(policy_for_version(obs, version))
            distribution = policy.get_distribution(tensor,
                action_masks=np.asarray(obs["action_mask"], dtype=bool))
            probabilities = distribution.distribution.probs.detach().cpu().numpy()[0]
            entropy = distribution.entropy().detach().cpu().numpy()[0]
        return dict(action_probabilities=probabilities, entropy_nats=entropy)

    def current_target_matchup(self, obs, action):
        """Classify supported attacks against the visible defender, read-only.

        This is a decision-time matchup diagnostic, not a predicted execution:
        either battler can switch/change state, and accuracy/protection are not
        modeled. Excluded attacks are never called nonimmune alternatives.
        """
        if not 0 <= action < 4:
            return dict(assessed=False, reason="not_a_move")
        slot = int(obs["active"][0, 8]) - 1
        if not 0 <= slot < 6 or "v2_move_details" not in obs:
            return dict(assessed=False, reason="missing_move_details")
        move_id, _, pp, _ = obs["moves"][slot, action]
        power, _, _, category, _, kind = obs["v2_move_details"][0, slot, action]
        move_id, kind = int(move_id), int(kind)
        if (move_id not in self.rules.moves or move_id in self.MATCHUP_EXCLUDED_MOVES
                or pp <= 0 or power <= 1 or category not in (0, 1)):
            return dict(assessed=False, reason="not_a_supported_damaging_move")
        if not bool(obs["active"][1, 0]) or obs["v2_active"][1, 0] <= 0:
            return dict(assessed=False, reason="no_living_current_target")
        types = set(map(int, obs["active"][1, 5:7]))
        if kind not in range(18) or any(t not in range(18) for t in types):
            return dict(assessed=False, reason="unknown_type")
        ability = int(obs["v2_active"][1, 3])
        effective_types, effective_ability = set(types), ability
        if kind in (0, 1) and 7 in types:
            # Foresight removes Ghost's Normal/Fighting immunity in Gen III.
            if "v2_volatile_bits" in obs:
                foresight = bool(obs["v2_volatile_bits"][1, 29])
            elif "v2_volatiles" in obs:
                foresight = bool(int(obs["v2_volatiles"][1]) & (1 << 29))
            else:
                return dict(assessed=False, reason="missing_foresight_state")
            if foresight:
                effective_types.remove(7)
        if kind == 10 and ability == 18:
            # Gen III Flash Fire is inactive while its holder is frozen.
            if "status_bits" in obs:
                frozen = bool(obs["status_bits"][7, 5])
            else:
                return dict(assessed=False, reason="missing_flash_fire_status")
            if frozen:
                effective_ability = 0
        type_multiplier = math.prod(float(self.rules.chart[kind, t]) for t in effective_types)
        multiplier = self.rules.effectiveness(kind,
            dict(types=effective_types, ability=effective_ability))
        # Soundproof blocks these ordinary damaging sound moves in Gen III.
        soundproof = ability == 43 and move_id in (173, 253, 304)
        if soundproof:
            multiplier = 0.
        type_immune = type_multiplier == 0
        # Classify ability independently so overlaps are explicit, not a partition.
        ability_immune = (soundproof or
            (effective_ability, kind) in ((26, 4), (10, 13), (11, 11), (18, 10)) or
            (effective_ability == 25 and type_multiplier <= 1))
        return dict(assessed=True, move_type_id=kind, target_type_ids=sorted(types),
            target_ability_id=ability, type_multiplier=type_multiplier,
            known_immunity_multiplier=multiplier, currently_immune=multiplier == 0,
            type_immunity=type_immune, known_ability_immunity=bool(ability_immune))

    def __call__(self, case, decision, obs, action):
        action = int(action)
        legal = np.flatnonzero(obs["action_mask"]).tolist()
        forced = bool(obs["context"][1])
        choice = self.action_data(obs, action)
        counts = Counter(decisions=1)
        counts["automatic" if len(legal) == 1 else "choices"] += 1
        if forced:
            counts["forced_replacement_decisions"] += 1
            if choice["kind"] == "switch":
                counts["forced_switches"] += 1
        elif action != 10:
            counts["voluntary_decisions"] += 1
            if choice["kind"] == "switch":
                counts["voluntary_switches"] += 1
        if choice["kind"] == "move":
            counts["move_actions"] += 1
            self.moves[str(choice["move_id"])] += 1
        elif choice["kind"] == "switch":
            self.switches[str(choice["species_id"])] += 1
        if decision == 1:
            self.first_moves[str(choice.get("move_id", -1))] += 1
        matchup = self.current_target_matchup(obs, action)
        alternatives = []
        if choice["kind"] == "move":
            counts["matchup_classified_move_actions" if matchup["assessed"] else
                   "matchup_unclassified_move_actions"] += 1
            if matchup.get("currently_immune"):
                counts["current_target_immune_move_actions"] += 1
                counts["current_target_type_immune_move_actions"] += int(matchup["type_immunity"])
                counts["current_target_ability_immune_move_actions"] += int(matchup["known_ability_immunity"])
                for other in legal:
                    if other == action or other >= 4:
                        continue
                    other_matchup = self.current_target_matchup(obs, other)
                    if other_matchup["assessed"] and not other_matchup["currently_immune"]:
                        alternatives.append(dict(**self.action_data(obs, other),
                                                 current_target_matchup=other_matchup))
                counts["current_target_immune_with_nonimmune_alternative"] += bool(alternatives)
        self.counts.update(counts)
        self.by_episode.setdefault(str(case["episode"]), Counter()).update(counts)
        if int(case["episode"]) > self.trace_cases or decision > self.trace_decisions:
            return
        suggested, _ = self.rules.choose(obs)
        row = dict(episode=case["episode"], matchup=case["matchup"], seed=case["seed"],
            candidate_player=case["candidate_player"], decision=decision,
            turn=int(obs["context"][0]), forced_replacement=forced,
            legal_actions=legal, chosen=choice,
            rules_suggestion=self.action_data(obs, suggested),
            active_species=[self.label("species", row[1]) for row in obs["active"]],
            active_species_ids=[int(row[1]) for row in obs["active"]],
            active_hp=obs["v2_active"][:, :2],
            party_species=[self.label("species", row[1]) for row in obs["party"] if row[0]])
        row.update(current_target_matchup=matchup,
                   legal_nonimmune_damaging_alternatives=alternatives)
        probabilities = self.probabilities(obs)
        if probabilities is not None:
            row.update(probabilities)
        self.rows.append(json_native(row))

    def finish(self, run):
        denominator = self.counts["voluntary_decisions"]
        result = dict(totals=dict(self.counts), by_episode=self.by_episode,
            first_move_ids=dict(self.first_moves), move_ids=dict(self.moves),
            switch_target_species_ids=dict(self.switches),
            voluntary_switch_rate=self.counts["voluntary_switches"] / denominator if denominator else None,
            voluntary_switch_denominator="candidate non-forced decisions excluding wait",
            trace_limits=dict(cases=self.trace_cases, decisions_per_case=self.trace_decisions),
            traced_decisions=len(self.rows), rule_identity=self.rules.identity,
            current_target_immunity_scope={
                "version": 1,
                "denominator": "matchup_classified_move_actions; all observed candidate battles",
                "meaning": "Currently immune target at decision time, not guaranteed move execution or a proven lost battle.",
                "alternatives": "Legal supported damaging moves with no classified current type/ability immunity; not a damage or hit guarantee. Populated when the chosen move is classified immune.",
                "known_abilities": ["Levitate", "Volt Absorb", "Water Absorb", "Flash Fire", "Wonder Guard", "Soundproof"],
                "excluded_move_ids": sorted(self.MATCHUP_EXCLUDED_MOVES),
                "exclusions": "Status, unknown, zero-PP, power <= 1, and listed special moves; missing required state. Unknown mechanics do not count as nonimmune alternatives.",
                "overlap": "Type and ability immunity counters can overlap.",
            },
            limitation="Rules suggestions are heuristic comparisons, not proven best actions. "
                       "Policy probabilities are re-evaluated without learning for retained trace rows only.")
        atomic_write(Path(run) / "decision-audit.json",
                     json.dumps(json_native(result), indent=2, allow_nan=False) + "\n")
        ordered = sorted(self.rows, key=lambda row: (row["episode"], row["decision"]))
        atomic_write(Path(run) / "decision-traces.jsonl",
                     "".join(json.dumps(row, allow_nan=False) + "\n" for row in ordered))
        return json_native(result)


def baseline_policy(project, name):
    """Candidate baselines; their opponent is still the unchanged Rules policy."""
    if name == "random":
        return None
    if name == "greedy":
        return GreedyDamage(project)
    if name == "rules":
        return RuleBasedOpponent(project)
    raise ValueError(f"unknown diagnostic baseline: {name}")


def score_stage(project, run, suite, candidate, reward_path, *, pairs=4,
                trace_cases=16, trace_decisions=24, opponent=None, audit_factory=None):
    """Frozen evaluation using real sessions and bounded candidate traces.

    The default opponent is Rules. Callers supplying another frozen policy must
    label the suite accordingly; Rules suggestions in traces remain a diagnostic
    reference independently of the actual opponent. An optional audit factory
    receives the same project, candidate, and trace limits as StageDecisionAudit.
    """
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    atomic_write(run / "suite.json", json.dumps(suite, indent=2) + "\n")
    paths = materialize_assets(suite, run)
    selected = [[] for _ in range(min(pairs, len(suite["cases"])))]
    if not selected:
        raise ValueError("evaluation needs at least one pair")
    cancel, sessions = Event(), []
    args = SimpleNamespace(seed=suite["seed"], node="node", command_timeout=120,
        observation_version="v2", step_logs=False, log_max_mb=2, log_backups=1,
        keep_episode_logs=0)
    factory = StageDecisionAudit if audit_factory is None else audit_factory
    audit = factory(project, candidate, trace_cases=trace_cases,
                    trace_decisions=trace_decisions)
    try:
        for i in range(len(selected)):
            factory = ShowdownFactory(project, run / f"pair-{i+1}",
                [55355+2*i, 55356+2*i], lambda i=i: tuple(selected[i]), args, cancel)
            sessions.append(ShowdownBattleSession(factory, max_steps=suite["max_steps"],
                reward_config=LinkedRewardConfig.load(reward_path)))
        result = evaluate(sessions, selected, suite, paths,
            [candidate, RuleBasedOpponent(project) if opponent is None else opponent], run, cancel,
            aborters=[s.factory.stop for s in sessions], announce=lambda message: None,
            decision_observer=audit)
        counts = audit.finish(run)
        result.update(wins_per_requested=result["candidate_wins"] / result["requested_battles"],
            first_move_ids=counts["first_move_ids"],
            first_surf_rate=counts["first_move_ids"].get("57", 0) / result["requested_battles"],
            voluntary_switch_rate=counts["voluntary_switch_rate"],
            audit_totals=counts["totals"])
        atomic_write(run / "summary.json", json.dumps(json_native(result), indent=2,
                                                       allow_nan=False) + "\n")
        print(f"{run.name}: wins={result['candidate_wins']}/{result['requested_battles']} "
              f"({result['wins_per_requested']:.1%}); truncated={result['truncated']}", flush=True)
        return result
    finally:
        try:
            # Preserve partial audit data even when a simulation or model failed.
            audit.finish(run)
        finally:
            for session in sessions:
                session.close()
