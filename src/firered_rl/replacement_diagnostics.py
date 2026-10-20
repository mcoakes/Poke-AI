"""Frozen tests of party-order sensitivity and forced-replacement decisions.

Rules assistance is an evaluation intervention, never a learned-policy score.
It changes only forced replacements with at least two legal replacement choices.
"""
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .evaluation import digest, validate_suite
from .periodic_evaluation import atomic_write
from .ppo_features import observation_version, policy_for_version
from .rule_opponent import RuleBasedOpponent
from .staged_team_lab import StageDecisionAudit, json_native


VERSION = 1


def transform_suite(suite):
    """Reverse each candidate's occupied bench slots, keeping its lead fixed.

    Opponent assets, all Pokémon data, move order, cases, seats and battle seeds
    stay unchanged. Candidate-only transformations cannot retain the validator's
    stricter identical-assets side-swap flag; paired case IDs/seeds remain intact.
    Calling twice restores each case's original team order and asset digest.
    """
    validate_suite(suite)
    if suite['backend'] != 'showdown':
        raise ValueError('bench-order diagnostics require showdown team assets')
    result = copy.deepcopy(suite)
    originals = suite['assets']
    for case in result['cases']:
        side = case['candidate_player'] - 1
        asset = copy.deepcopy(originals[case['assets'][side]])
        party = asset['team']['party']
        asset['team']['party'] = party[:1] + party[:0:-1]
        key = digest(asset)
        result['assets'][key] = asset
        # create_suite may share this same list object between side-swapped
        # cases before a JSON roundtrip. Never mutate its opponent's case too.
        case['assets'] = list(case['assets'])
        case['assets'][side] = key
    result['swap_sides'] = False
    result['replacement_transform'] = dict(version=VERSION,
        kind='candidate_bench_reversal', lead='unchanged', opponent='unchanged',
        source_suite_sha256=digest(suite),
        pairing='original case IDs, candidate seats, matchup IDs and RNG seeds retained')
    return validate_suite(result)


def _eligible(obs):
    legal = np.flatnonzero(np.asarray(obs['action_mask'], dtype=bool))
    replacements = legal[(legal >= 4) & (legal < 10)]
    return bool(obs['context'][1]) and len(replacements) >= 2, replacements


def _validate_mask(observation, action_masks):
    mask = np.asarray(observation['action_mask'], dtype=bool)
    supplied = np.asarray(action_masks, dtype=bool)
    if mask.shape != supplied.shape or not np.array_equal(mask, supplied):
        raise ValueError('replacement diagnostic requires the authoritative action mask')
    if mask.ndim not in (1, 2) or mask.shape[-1] != 11 or not mask.any(axis=-1).all():
        raise ValueError('replacement diagnostic received invalid action masks')
    return mask


def _single_prediction(model, obs):
    # Production observers receive the complete policy observation. Checkpoint
    # models still get their exact serialized schema, as in evaluation.evaluate.
    if callable(getattr(model, 'choose', None)):
        # The Rules/greedy baselines expose the exact stateless single-row chooser
        # used by their predict method and need no PPO schema projection.
        action, _ = model.choose(obs, np.asarray(obs['action_mask'], dtype=bool))
    else:
        selected = obs
        if hasattr(model, 'observation_space'):
            selected = policy_for_version(obs, observation_version(model.observation_space))
        action, _ = model.predict(selected,
            action_masks=np.asarray(selected['action_mask'], dtype=bool), deterministic=True)
    values = np.asarray(action)
    if values.size != 1:
        raise ValueError('base policy did not return one diagnostic action')
    action = int(values.reshape(-1)[0])
    if action not in np.flatnonzero(obs['action_mask']):
        raise ValueError('base policy selected an illegal diagnostic action')
    return action


class _InferenceMode:
    """Mode-only facade; base PPO probabilities are not hybrid probabilities."""
    def __init__(self, base):
        self.base = base

    def set_training_mode(self, mode):
        if mode:
            raise ValueError('replacement diagnostic policy is frozen')
        policy = getattr(self.base, 'policy', None)
        if policy is not None and hasattr(policy, 'set_training_mode'):
            policy.set_training_mode(False)


class FrozenForcedReplacementPolicy:
    """Use Rules only for multi-option forced replacements; otherwise use PPO.

    Stateless deterministic inference supports both batched and single inputs.
    No training/optimizer methods are exposed. The policy facade deliberately
    has no get_distribution: hybrid actions do not follow the PPO distribution.
    """
    assisted = True

    def __init__(self, base_model, project):
        self.base_model = base_model
        self.rules = RuleBasedOpponent(project)
        self.policy = _InferenceMode(base_model)
        for name in ('observation_space', 'action_space'):
            if hasattr(base_model, name):
                setattr(self, name, getattr(base_model, name))
        self.policy.set_training_mode(False)

    def predict(self, observation, *, action_masks, deterministic=True, **kwargs):
        if not deterministic:
            raise ValueError('replacement diagnostic requires deterministic inference')
        mask = _validate_mask(observation, action_masks)
        actions, state = self.base_model.predict(observation, action_masks=action_masks,
                                                deterministic=True, **kwargs)
        batch = mask.ndim == 2
        count = len(mask) if batch else 1
        actions = np.asarray(actions).reshape(-1).copy()
        if len(actions) != count:
            raise ValueError('base policy returned the wrong number of actions')
        for index in range(count):
            obs = {key: value[index] if batch else value
                   for key, value in observation.items()}
            row_mask = mask[index] if batch else mask
            action = int(actions[index])
            if not 0 <= action < 11 or not row_mask[action]:
                raise ValueError('base policy selected an illegal action')
            eligible, replacements = _eligible(obs)
            if eligible:
                proposed, _ = self.rules.choose(obs, row_mask)
                if int(proposed) not in replacements:
                    raise ValueError('Rules did not choose a legal forced replacement')
                actions[index] = int(proposed)
        actions = actions.astype(np.int64)
        return (actions if batch else actions[0]), state


def make_candidate(model, assisted, project):
    if type(assisted) is not bool:
        raise ValueError('assisted must be a boolean')
    return FrozenForcedReplacementPolicy(model, project) if assisted else model


class ReplacementAudit(StageDecisionAudit):
    """All-battle replacement counters plus bounded, explicitly labeled traces."""
    REPLACEMENT_COUNTERS = (
        'replacement_eligible', 'replacement_actual_lowest_legal',
        'replacement_base_lowest_legal', 'replacement_base_rules_disagreements',
        'replacement_override_eligible', 'replacement_overrides_changed',
    )

    def __init__(self, project, candidate, *, trace_cases=16, trace_decisions=24):
        super().__init__(project, candidate, trace_cases=trace_cases,
                         trace_decisions=trace_decisions)
        self.assisted = isinstance(candidate, FrozenForcedReplacementPolicy)
        self.base_model = candidate.base_model if self.assisted else candidate
        self.counts.update(dict.fromkeys(self.REPLACEMENT_COUNTERS, 0))
        self.replacement_species = Counter()
        self.replacement_species_by_episode = {}

    def probabilities(self, obs):
        values = StageDecisionAudit.probabilities(
            SimpleNamespace(candidate=self.base_model), obs)
        if values is None or not self.assisted:
            return values
        return dict(base_policy_action_probabilities=values['action_probabilities'],
                    base_policy_entropy_nats=values['entropy_nats'],
                    probability_scope='unassisted base policy, not the executed hybrid policy')

    def __call__(self, case, decision, obs, action):
        action = int(action)
        eligible, replacements = _eligible(obs)
        details = None
        if eligible:
            base_action = _single_prediction(self.base_model, obs)
            rule_action, _ = self.rules.choose(obs)
            rule_action = int(rule_action)
            if base_action not in replacements or rule_action not in replacements:
                raise ValueError('forced replacement proposals must be legal replacements')
            expected = rule_action if self.assisted else base_action
            if action != expected:
                raise ValueError('observed action differs from frozen diagnostic policy')
            lowest = int(replacements.min())
            extra = Counter(replacement_eligible=1,
                replacement_actual_lowest_legal=int(action == lowest),
                replacement_base_lowest_legal=int(base_action == lowest),
                replacement_base_rules_disagreements=int(base_action != rule_action),
                replacement_override_eligible=int(self.assisted),
                replacement_overrides_changed=int(self.assisted and action != base_action))
            self.counts.update(extra)
            episode = str(case['episode'])
            self.by_episode.setdefault(episode, Counter()).update(extra)
            species = str(int(obs['party'][action - 4, 1]))
            self.replacement_species[species] += 1
            self.replacement_species_by_episode.setdefault(episode, Counter())[species] += 1
            details = dict(eligible=True, intervention='rules_forced_replacement' if self.assisted else 'none',
                base_proposal=self.action_data(obs, base_action),
                rules_proposal=self.action_data(obs, rule_action),
                executed=self.action_data(obs, action),
                actual_override=bool(self.assisted and action != base_action),
                lowest_legal_replacement=self.action_data(obs, lowest))
        before = len(self.rows)
        super().__call__(case, decision, obs, action)
        if len(self.rows) != before:
            self.rows[-1]['replacement_diagnostic'] = details or dict(eligible=False,
                intervention='none', reason='not a multi-option forced replacement')
            self.rows[-1]['policy_kind'] = ('diagnostic_hybrid' if self.assisted else 'unassisted')

    def finish(self, run):
        result = super().finish(run)
        eligible = self.counts['replacement_eligible']
        result['replacement_diagnostic'] = dict(version=VERSION,
            policy_kind='diagnostic_hybrid' if self.assisted else 'unassisted',
            assisted=self.assisted, all_battle_counts=True,
            eligibility='forced replacement with at least two legal replacement actions',
            rates_denominator='replacement_eligible',
            actual_lowest_legal_rate=(self.counts['replacement_actual_lowest_legal'] / eligible
                                      if eligible else None),
            base_lowest_legal_rate=(self.counts['replacement_base_lowest_legal'] / eligible
                                    if eligible else None),
            base_rules_disagreement_rate=(self.counts['replacement_base_rules_disagreements'] / eligible
                                         if eligible else None),
            actual_species_ids=dict(self.replacement_species),
            actual_species_ids_by_episode=self.replacement_species_by_episode,
            interpretation='Rules assistance is a frozen diagnostic intervention, not learned performance. '
                           'Rules proposals are heuristic, not optimal-action certificates. '
                           'Order-test outcomes measure sensitivity; they do not identify a unique cause.')
        result['limitation'] += (' Hybrid trace probabilities, when present, are explicitly labeled '
                                'as the unassisted base model distribution.')
        result = json_native(result)
        atomic_write(Path(run) / 'decision-audit.json',
                     json.dumps(result, indent=2, allow_nan=False) + '\n')
        return result
