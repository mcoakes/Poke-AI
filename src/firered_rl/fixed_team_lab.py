"""Small, calibrated learnability problems using the production battle pipeline."""
import copy
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np

from .evaluation import create_suite, evaluate, materialize_assets, validate_suite
from .linked_reward import LinkedRewardConfig
from .periodic_evaluation import atomic_write
from .rule_opponent import RuleBasedOpponent
from .showdown_env import ShowdownBattleSession, ShowdownFactory
from .training_teams import TrainingTeams


def pokemon(species, moves):
    return dict(species=species, level=50, nature="Hardy", held_item="None",
                moves=moves, evs={k: 0 for k in ("hp", "atk", "def", "spa", "spd", "spe")},
                ivs={k: 31 for k in ("hp", "atk", "def", "spa", "spd", "spe")})


def teams():
    """Explicit stats/items/natures; no generated team pool or hidden curriculum."""
    mirror = dict(name="fixed-fire-water-grass", party=[
        pokemon("Typhlosion", ["Strength", "Flamethrower"]),
        pokemon("Blastoise", ["Strength", "Surf"]),
        pokemon("Sceptile", ["Strength", "LeafBlade"]),
    ])
    return {
        "move-choice": (
            dict(name="starmie-move-choice", party=[pokemon(
                "Starmie", ["Thunderbolt", "Swift", "Surf", "Harden"])]),
            dict(name="golem-rules", party=[pokemon(
                "Golem", ["Earthquake", "RockThrow", "Tackle", "DefenseCurl"])])),
        "mirror": (mirror, copy.deepcopy(mirror)),
    }


def make_suite(project, paths, episodes, seed, scenario, *, permute_moves=False):
    source = TrainingTeams(teams=paths, seed=seed)
    suite = create_suite(source, backend="showdown", episodes=episodes,
                         seed=seed, swap_sides=True, max_steps=100)
    if scenario == "move-choice":
        # Standard side swaps also exchange teams. Here the candidate must keep
        # Starmie on BOTH seats, otherwise the advantage test averages to ~50%.
        suite["swap_sides"] = False
        for case in suite["cases"]:
            if case["candidate_player"] == 2:
                case["assets"] = list(reversed(case["assets"]))
    if permute_moves:
        from .evaluation import digest
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
        # Each side gets the same semantic team, with its own move presentation.
        suite["swap_sides"] = False
    suite["rule_opponent"] = RuleBasedOpponent(project).identity
    suite["diagnostic"] = dict(scenario=scenario, balanced_candidate_seats=True,
                               candidate_keeps_team=True, permuted_moves=permute_moves)
    return validate_suite(suite)


class GreedyDamage(RuleBasedOpponent):
    """Attack with largest estimated damage; Rules chooses forced replacements.

    This provides an achievable baseline, not an optimal-policy certificate.
    """
    def choose(self, obs, mask=None):
        legal = np.flatnonzero(obs["action_mask"])
        attacks = [int(a) for a in legal if a < 4]
        if not attacks:
            return super().choose(obs, mask)
        own, foe = self.slot(obs, 0), self.slot(obs, 1)
        action = max(attacks, key=lambda a: (self.damage(obs, 0, own, a, foe)[0], -a))
        return action, []


class FirstMoveObserver:
    """Record the executed policy's initial semantic move, not its action slot."""
    def __init__(self):
        self.moves = {}

    def __call__(self, case, decision, obs, action):
        if decision == 1:
            slot = int(obs["active"][0, 8]) - 1
            move = int(obs["moves"][slot, action, 0]) if action < 4 else -1
            self.moves[str(move)] = self.moves.get(str(move), 0) + 1


def score(project, run, suite, candidate, reward_path, *, pairs=4):
    """Frozen candidate evaluation; used for calibration and final diagnostics."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=False)
    (run / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    paths = materialize_assets(suite, run)
    selected = [[] for _ in range(min(pairs, len(suite["cases"])))]
    cancel, sessions = Event(), []
    args = SimpleNamespace(seed=suite["seed"], node="node", command_timeout=120,
                           observation_version="v2", step_logs=False,
                           log_max_mb=2, log_backups=1, keep_episode_logs=0)
    observer = FirstMoveObserver()
    try:
        for i in range(len(selected)):
            factory = ShowdownFactory(project, run / f"pair-{i+1}", [55355+2*i, 55356+2*i],
                                      lambda i=i: tuple(selected[i]), args, cancel)
            sessions.append(ShowdownBattleSession(factory, max_steps=suite["max_steps"],
                            reward_config=LinkedRewardConfig.load(reward_path)))
        result = evaluate(sessions, selected, suite, paths, [candidate, RuleBasedOpponent(project)],
                          run, cancel, aborters=[s.factory.stop for s in sessions],
                          announce=lambda message: None, decision_observer=observer)
        # Explicitly count all requested battles for comparisons, including truncations.
        result["wins_per_requested"] = result["candidate_wins"] / result["requested_battles"]
        result["first_move_ids"] = observer.moves
        result["first_surf_rate"] = observer.moves.get("57", 0) / result["requested_battles"]
        atomic_write(run / "summary.json", json.dumps(result, indent=2) + "\n")
        print(f"{run.name}: wins={result['candidate_wins']}/{result['requested_battles']} "
              f"({result['wins_per_requested']:.1%}); truncated={result['truncated']}", flush=True)
        return result
    finally:
        for session in sessions:
            session.close()
