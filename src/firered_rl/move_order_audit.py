"""Inference-only move permutation probes and observed protocol execution checks."""
from collections import Counter
import json
import re

import numpy as np
import torch

from .readable_log import names
from .runtime_logs import RotatingLog, log_settings


def permute_moves(obs, order):
    order = np.asarray(order, dtype=int)
    if sorted(order.tolist()) != [0, 1, 2, 3]:
        raise ValueError('order must permute four move slots')
    slot = int(obs['active'][0, 8]) - 1
    if not 0 <= slot < 6:
        raise ValueError('missing active slot')
    result = {k: v.copy() for k, v in obs.items()}
    result['moves'][slot] = obs['moves'][slot, order]
    result['v2_move_details'][0, slot] = obs['v2_move_details'][0, slot, order]
    result['action_mask'][:4] = obs['action_mask'][order]
    return result


def probabilities(model, observations):
    batch = {k: np.stack([o[k] for o in observations]) for k in observations[0]}
    tensor, _ = model.policy.obs_to_tensor(batch)
    with torch.no_grad():
        return model.policy.get_distribution(tensor, action_masks=batch['action_mask'].astype(bool)).distribution.probs.cpu().numpy()


def execution_check(lines, side, expected, *, action=None, choice=None):
    emitted = [line.split('|')[3] for line in lines if line.startswith(f'|move|p{side}a:') and len(line.split('|')) > 3]
    def normal(name):
        move_id = re.sub('[^a-z0-9]', '', name.lower())
        # Gen III's name and Showdown's modern name refer to the same move (136).
        return {"hijumpkick": "highjumpkick"}.get(move_id, move_id)
    if not emitted:
        return 'not_observed', emitted
    # Called moves can follow the originally selected move; compare its first announcement.
    if normal(emitted[0]) == normal(expected):
        return 'matched', emitted
    # Do not hide unexpected Struggle announcements: require native request AND
    # queued-choice evidence from before this exact submitted action executed.
    if (normal(emitted[0]) == 'struggle' and action is not None and choice
            and choice.get('action') == action and choice.get('command') == 'move 1'
            and choice.get('requested_move') == 'struggle'
            and choice.get('resolved_move') == 'struggle'):
        return 'struggle_fallback', emitted
    return 'review_mismatch', emitted


class MoveOrderAudit:
    def __init__(self, model, run, args):
        self.model, self.run = model, run
        settings, _ = log_settings(args)
        self.log = RotatingLog(run/'move-order-decisions.jsonl', **settings)
        self.table = names('firered_us_v1_0')
        self.counts, self.per_case, self.examples, self.pending = Counter(), {}, [], {}
        self.orders = [np.roll(np.arange(4), i) for i in (1, 2, 3)]
        self.probability_sum = np.zeros(11)
        self.tv_sum = 0.

    def __call__(self, case, decision, obs, action):
        episode = case['episode']
        counts = self.per_case.setdefault(str(episode), Counter())
        counts['decisions'] += 1
        counts[f'action_{action}'] += 1
        legal = np.flatnonzero(obs['action_mask'])
        slot = int(obs['active'][0, 8])-1
        original = probabilities(self.model, [obs])[0]
        self.probability_sum += original
        row = dict(episode=episode, decision=decision, candidate_player=case['candidate_player'],
                   action=action, legal=legal.tolist(), probabilities=original.tolist(), probes=[])
        if int(np.argmax(original)) != action:
            counts['inference_recheck_mismatch'] += 1
        row['automatic'] = len(legal) == 1
        if len(legal) == 1:
            counts['automatic'] += 1
        else:
            counts['choices'] += 1
            counts[f'choice_action_{action}'] += 1
        if 0 <= slot < 6:
            ids = obs['moves'][slot, :, 0].astype(int)
            row['moves'] = [self.table.get('moves', {}).get(str(i), str(i)) for i in ids]
            row['active_species'] = self.table.get('species', {}).get(str(int(obs['active'][0,1])), '?')
            # All entries, including empty/disabled slots, move together. Duplicate moves
            # make semantic identity ambiguous and are excluded from permutation metrics.
            present = ids[ids > 0]
            eligible = (not bool(obs['context'][1]) and len(legal[legal < 4]) >= 2
                        and len(set(present.tolist())) == len(present))
            if eligible:
                counts['permuted_positions'] += 1
                variants = [permute_moves(obs, p) for p in self.orders]
                probs = probabilities(self.model, variants)
                for order, prob in zip(self.orders, probs):
                    chosen = int(np.argmax(prob))
                    semantic = int(order[chosen]) if chosen < 4 else chosen
                    aligned = prob.copy()
                    aligned[order] = prob[:4]
                    tv = float(np.abs(aligned-original).sum()/2)
                    self.tv_sum += tv
                    counts['permutations'] += 1
                    if semantic == action:
                        counts['follows_original_action'] += 1
                    if action < 4:
                        counts['original_move_permutations'] += 1
                        if chosen == action:
                            counts['stays_in_original_slot'] += 1
                    if np.sort(original)[-1]-np.sort(original)[-2] < 1e-7 or np.sort(prob)[-1]-np.sort(prob)[-2] < 1e-7:
                        counts['near_tie_permutations'] += 1
                    row['probes'].append(dict(new_to_old_slots=(order+1).tolist(), action=chosen,
                                              semantic_action=semantic, probabilities=prob.tolist(),
                                              aligned_total_variation=tv))
            if action < 4 and len(legal) > 1 and not bool(obs['context'][1]):
                self.pending[episode] = (row['moves'][action], case['candidate_player'], decision, action)
        self.log.append(json.dumps(row)+'\n')
        if row['probes'] and len(self.examples) < 24:
            self.examples.append(row)

    def after_step(self, case, decision, info):
        pending = self.pending.pop(case['episode'], None)
        if pending is None:
            return
        expected, side, _, action = pending
        choice = info.get('audit_choice')
        status, emitted = execution_check(info.get('audit_protocol', []), side, expected,
                                         action=action, choice=choice)
        self.per_case[str(case['episode'])][f'execution_{status}'] += 1
        row = dict(episode=case['episode'], decision=decision, execution=status,
                   expected=expected, announced=emitted, submitted_choice=choice)
        self.log.append(json.dumps(row)+'\n')
        if status in ('review_mismatch', 'struggle_fallback') and len(self.examples) < 48:
            self.examples.append(row)

    def finish(self):
        for counts in self.per_case.values():
            self.counts.update(counts)
        c = self.counts
        data = dict(counts=dict(c), by_episode={k:dict(v) for k,v in self.per_case.items()},
                    mean_action_probabilities=(self.probability_sum/max(1,c['decisions'])).tolist(),
                    mean_aligned_total_variation=self.tv_sum/max(1,c['permutations']), examples=self.examples)
        (self.run/'move-order-audit.json').write_text(json.dumps(data, indent=2)+'\n')
        lines = ['# Checkpoint move-order audit', '',
                 'All sampled decisions include winning and losing battles. Forced/single-action decisions are separated.', '',
                 '| Action | All choices | Nonautomatic choices | Mean probability (all decisions) |', '|---|---:|---:|---:|']
        for a, p in enumerate(data['mean_action_probabilities']):
            label = f'Move {a+1}' if a < 4 else (f'Switch {a-3}' if a < 10 else 'Wait / escape')
            lines.append(f"| {label} | {c[f'action_{a}']} | {c[f'choice_action_{a}']} | {p:.3f} |")
        lines += ['', '| Check | Count |', '|---|---:|']
        lines += [f'| {key} | {c[key]} |' for key in ('decisions','automatic','choices','permuted_positions','permutations','follows_original_action','original_move_permutations','stays_in_original_slot','near_tie_permutations','inference_recheck_mismatch','execution_matched','execution_struggle_fallback','execution_not_observed','execution_review_mismatch')]
        lines += ['', '`execution_struggle_fallback` requires Struggle in both the native request and queued move before the submitted action executed. The stored slot name is retained in the JSON evidence. Missing or conflicting evidence remains a review mismatch.']
        def rate(numerator, denominator):
            return f'{numerator/denominator:.1%} ({numerator}/{denominator})' if denominator else 'not applicable (no eligible decisions)'
        lines += ['', '## Slot sensitivity', '',
                  f"- Move 1 among nonautomatic choices: {rate(c['choice_action_0'], c['choices'])}.",
                  f"- Same numeric slot after rearranging moves: {rate(c['stays_in_original_slot'], c['original_move_permutations'])}.",
                  f"- Follows original semantic action: {rate(c['follows_original_action'], c['permutations'])}.",
                  'Semantic agreement includes unchanged switch/wait actions. These denominators can differ.',
                  'Lower probability sensitivity can also reflect flatter probabilities; it does not by itself show better tactical decisions. Different checkpoints may visit different positions even on the same matchups.']
        mismatches = [row for row in self.examples if row.get('execution') == 'review_mismatch']
        if c['execution_review_mismatch']:
            def cell(value):
                return str(value).replace('|', r'\|').replace('\n', ' ').replace('\r', ' ')
            lines += ['', '## Execution differences to review', '',
                      f"Showing {min(12, len(mismatches))} retained examples of {c['execution_review_mismatch']} flagged decisions. These are not automatically action-mapping errors.", '',
                      '| Replay episode | Decision | Selected move | Announced move(s) |',
                      '|---:|---:|---|---|']
            for row in mismatches[:12]:
                lines.append(f"| {row['episode']} | {row['decision']} | {cell(row['expected'])} | {cell(' → '.join(row['announced']))} |")
            lines += ['', 'Check naming aliases, forced/called moves and the battle protocol for each example. Further retained examples are in `move-order-audit.json`.']
        lines += ['', f"Mean probability change after mapping back to original actions (total variation): {data['mean_aligned_total_variation']:.4f}.", '',
                  'Three cyclic permutations move every move to every other slot. `follows_original_action` includes unchanged switch choices; compare `stays_in_original_slot` with `original_move_permutations` for move-slot preference. Near ties can change argmax without meaningful preference.',
                  '', 'Execution checks compare the first actual move announcement for the candidate side with its selected move. No announcement can mean fainting, flinching, sleep or another interruption. Different announcements require review (for example Encore/Struggle/called-move behavior), not an automatic mapping-bug conclusion.',
                  '', 'Permutations are inference-only: their actions are never submitted. Inputs move IDs, types, PP, power/accuracy/category/priority and legal mask together. This measures sensitivity, not the best tactical action. Diagnostic inference slows execution; do not use this run for throughput benchmarking.',
                  '', 'Details and bounded examples: `move-order-audit.json`; rotated trace: `move-order-decisions.jsonl`.']
        (self.run/'move-order-audit.md').write_text('\n'.join(lines)+'\n')
