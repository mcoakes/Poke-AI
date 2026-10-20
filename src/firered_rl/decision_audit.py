"""Read-only heuristic comparison of candidate choices; not an optimal-action oracle."""
from collections import Counter
import json

import numpy as np

from .readable_log import names
from .rule_opponent import RuleBasedOpponent
from .runtime_logs import RotatingLog, log_settings


def _json_scalar(value):
    """Preserve numeric diagnostic values when NumPy arithmetic returns scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class DecisionAudit:
    def __init__(self, project, run, args):
        self.rules = RuleBasedOpponent(project)
        self.run = run
        settings, _ = log_settings(args)
        self.log = RotatingLog(run / 'candidate-decisions.jsonl', **settings)
        self.counts = {}
        self.examples = []
        self.names = names('firered_us_v1_0')

    def label(self, obs, action):
        if action == 10:
            return 'Wait / automatic continuation'
        if action >= 4:
            slot = action - 4
            species = int(obs['party'][slot, 1])
            return f"Switch slot {slot+1}: {self.names.get('species', {}).get(str(species), species)}"
        slot = self.rules.slot(obs, 0)
        move = int(obs['moves'][slot, action, 0]) if 0 <= slot < 6 else 0
        return f"Move {action+1}: {self.names.get('moves', {}).get(str(move), move)}"

    def __call__(self, case, decision, obs, action):
        counters = self.counts.setdefault(str(case['episode']), Counter())
        counters['decisions'] += 1
        action = int(action)
        legal = np.flatnonzero(obs['action_mask']).tolist()
        forced = bool(obs['context'][1])
        automatic = len(legal) == 1
        counters['automatic' if automatic else 'choices'] += 1
        if forced:
            counters['replacement_decisions'] += 1
        elif not automatic and 4 <= action < 10:
            counters['voluntary_switches'] += 1
        suggested, scores = self.rules.choose(obs)
        row = dict(episode=case['episode'], pair=case.get('diagnostic_pair'), decision=decision,
                   turn=int(obs['context'][0]), candidate_player=case['candidate_player'],
                   action=action, chosen=self.label(obs, action),
                   suggested=suggested, suggestion=self.label(obs, suggested),
                   forced_replacement=forced, automatic=automatic, legal=legal,
                   hp=obs['v2_active'][:, :2].tolist(),
                   active_species=[self.names.get('species', {}).get(str(int(x)), str(int(x)))
                                   for x in obs['active'][:, 1]],
                   major_status_bits=[np.flatnonzero(obs['status_bits'][6+i]).tolist() for i in (0, 1)],
                   scores=scores, flags=[])
        if not automatic:
            if action != suggested:
                row['flags'].append('heuristic_disagreement')
            if action < 4 and not forced:
                slot, target = self.rules.slot(obs, 0), self.rules.slot(obs, 1)
                category = int(obs['v2_move_details'][0, slot, action, 3])
                counters['status_choices' if category == 2 else 'attack_choices'] += 1
                estimates = {a: self.rules.damage(obs, 0, slot, a, target)[0]
                             for a in legal if a < 4}
                best = max(estimates, key=estimates.get)
                row['estimated_direct_damage'] = estimates
                row['largest_damage_estimate_action'] = best
                # Review flag only: zero estimates include unsupported mechanics.
                if estimates[best] - estimates[action] >= .25 * self.rules.mon(obs, 1, target)['maximum']:
                    row['flags'].append('lower_direct_damage_estimate')
                reason = next(x['reason'] for x in scores if x['action'] == action)
                if 'unsupported or ineffective' in reason:
                    row['flags'].append('unsupported_or_ineffective_by_heuristic')
        counters.update(row['flags'])
        if row['flags'] and len(self.examples) < 40:
            self.examples.append(row)
        self.log.append(json.dumps(row, default=_json_scalar) + '\n')

    def finish(self):
        data = dict(by_episode={key: dict(value) for key, value in self.counts.items()},
                    examples=self.examples, rule_identity=self.rules.identity,
                    limitation='Heuristic disagreement and damage estimates are review signals, not proven mistakes or counterfactual outcomes.')
        (self.run / 'decision-audit.json').write_text(json.dumps(data, indent=2, default=_json_scalar) + '\n')
