"""Learned, shared residual scores for multi-option forced replacements.

Only already-visible v2 fields are used. Power/stat proxies are comparative
features, not simulator damage calculations or predictions of the foe's move.
The original move, switch and value pathways remain intact. Conversion starts
the residual at exactly zero and retains existing optimizer state.
"""
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from .matchup_move_policy import MatchupMoveFeatures, MatchupMovePolicy, MatchupRelations


FEATURE_NAMES = ('hp_fraction', 'maximum_hp', 'level', 'attack', 'defense',
    'special_attack', 'special_defense', 'speed', 'status_present',
    'outgoing_assessed_fraction', 'outgoing_max_multiplier', 'outgoing_power_proxy',
    'outgoing_stat_proxy', 'incoming_assessed_fraction', 'incoming_max_multiplier',
    'incoming_stat_proxy', 'speed_fraction', 'opponent_hp_fraction')
FEATURE_DIM = len(FEATURE_NAMES)


class BenchMatchupFeatures(nn.Module):
    """B x 6 x 18 candidate features without species or party-slot indices.

Outgoing compares each candidate's supported damaging moves with the current
foe; incoming compares the foe's currently observed moves with that candidate.
No future foe action is selected or assumed. Unsupported/status/special moves
have assessed=0. Maxima pool over moves without depending on their order.

Proxies include power, nominal accuracy, STAB, type/known-ability immunities,
and (for stat_proxy) the relevant attack/defense ratio. They exclude weather,
items, offensive abilities, status stat penalties, accuracy/evasion stages,
priority, and special move mechanics. They are not damage or KO estimates.
"""
    feature_names = FEATURE_NAMES

    def __init__(self):
        super().__init__()
        self.relations = MatchupRelations()

    def _attacks(self, moves, details, own_types, target_types, ability,
                 foresight, frozen, valid, attack_stats, defense_stats):
        ids = moves[..., 0].long()
        kinds = details[..., 5].long()
        categories = details[..., 3]
        assessed = ((ids > 0) & (ids < 512)
            & self.relations.supported_moves[ids.clamp(0, 511)]
            & (moves[..., 2] > 0) & (details[..., 0] > 1)
            & ((categories == 0) | (categories == 1))
            & (kinds >= 0) & (kinds < 18)
            & ((target_types >= 0) & (target_types < 18)).all(-1)[..., None]
            & ((own_types >= 0) & (own_types < 18)).all(-1)[..., None]
            & valid[..., None])
        target = target_types.long().clamp(0, 31)
        kind = kinds.clamp(0, 31)
        first = self.relations.type_chart[kind, target[..., None, 0]]
        second = self.relations.type_chart[kind, target[..., None, 1]]
        second = torch.where((target_types[..., 0] == target_types[..., 1])[..., None], 1., second)
        ignore_ghost = foresight[..., None] & ((kinds == 0) | (kinds == 1))
        first = torch.where(ignore_ghost & (target[..., None, 0] == 7), 1., first)
        second = torch.where(ignore_ghost & (target[..., None, 1] == 7), 1., second)
        multiplier = first * second
        ability = ability.long()[..., None]
        immune = (((ability == 26) & (kinds == 4))
            | ((ability == 10) & (kinds == 13))
            | ((ability == 11) & (kinds == 11))
            | ((ability == 18) & (kinds == 10) & ~frozen[..., None])
            | ((ability == 25) & (multiplier <= 1))
            | ((ability == 43) & self.relations.sound_moves[ids.clamp(0, 511)]))
        multiplier = torch.where(immune, 0., multiplier)
        stab = torch.where((kinds[..., None] == own_types[..., None, :]).any(-1), 1.5, 1.)
        accuracy = torch.where(details[..., 2] > 0, 1., details[..., 1] / 100).clamp(0, 1)
        power = details[..., 0] * accuracy * stab * multiplier
        attack = torch.where(categories == 0, attack_stats[..., 0, None], attack_stats[..., 2, None])
        defense = torch.where(categories == 0, defense_stats[..., 1, None], defense_stats[..., 3, None])
        stats_known = (attack > 0) & (defense > 0)
        stat_proxy = power * attack / defense.clamp(min=1)
        return torch.stack([assessed.float().mean(-1),
            (multiplier / 4 * assessed).amax(-1),
            (power / 1500 * assessed).clamp(0, 4).amax(-1),
            (stat_proxy / 1500 * assessed * stats_known).clamp(0, 4).amax(-1)], -1)

    def forward(self, obs):
        party = obs['v2_party'][:, 0]
        own_stats = obs['v2_stats'][:, 0]
        batch = torch.arange(len(party), device=party.device)
        raw_slot = obs['opponent_active_index'][:, 0] - 1
        slot_valid = (raw_slot >= 0) & (raw_slot < 6) & (raw_slot == raw_slot.floor())
        slot = raw_slot.long().clamp(0, 5)
        valid_foe = (slot_valid & (obs['active'][:, 1, 0] > 0)
            & (obs['v2_active'][:, 1, 0] > 0) & (obs['context'][:, 2] == 0))
        valid = ((party[..., 0] > 0) & (party[..., 2] > 0)
            & (party[..., 8] == 0) & valid_foe[:, None])
        foe_types = obs['active'][:, 1, 5:7, None].transpose(-1, -2).expand(-1, 6, -1)
        foe_ability = obs['v2_active'][:, 1, 3, None].expand(-1, 6)
        foe_stats = obs['v2_active_stats'][:, 1]
        boosts = obs['v2_boosts'][:, 1, :5]
        factor = torch.where(boosts >= 0, (2 + boosts) / 2, 2 / (2 - boosts))
        foe_stats = (foe_stats * factor)[:, None].expand(-1, 6, -1)
        foe_foresight = (obs['v2_volatile_bits'][:, 1, 29] > 0)[:, None].expand(-1, 6)
        foe_frozen = (obs['status_bits'][:, 7, 5] > 0)[:, None].expand(-1, 6)
        outgoing = self._attacks(obs['moves'], obs['v2_move_details'][:, 0],
            party[..., 4:6], foe_types, foe_ability, foe_foresight, foe_frozen,
            valid, own_stats, foe_stats)
        incoming_moves = obs['opponent_moves'][batch, slot][:, None].expand(-1, 6, -1, -1)
        incoming_details = obs['v2_move_details'][batch, 1, slot][:, None].expand(-1, 6, -1, -1)
        # A benched replacement has no active Foresight volatile. Status remains.
        incoming = self._attacks(incoming_moves, incoming_details, foe_types,
            party[..., 4:6], party[..., 7], torch.zeros_like(valid),
            obs['v2_status_bits'][:, 0, :, 5] > 0, valid, foe_stats, own_stats)
        hp = (party[..., 2:3] / party[..., 3:4].clamp(min=1)).clamp(0, 1)
        status = (obs['v2_status_bits'][:, 0].sum(-1, keepdim=True) > 0).float()
        speed = own_stats[..., 4:5] / (own_stats[..., 4:5] + foe_stats[..., 4:5]).clamp(min=1)
        foe_hp = (obs['v2_active'][:, 1, :1] / obs['v2_active'][:, 1, 1:2].clamp(min=1)).clamp(0, 1)
        values = torch.cat([hp, party[..., 3:4] / 1024, party[..., 1:2] / 100,
            own_stats / 1024, status, outgoing, incoming[..., :2], incoming[..., 3:4],
            speed, foe_hp[:, None].expand(-1, 6, -1)], -1)
        return values * valid[..., None]


class MatchupSwitchFeatures(MatchupMoveFeatures):
    def __init__(self, observation_space, context_dim=128, move_dim=32,
                 relational_enabled=True, bench_features_enabled=True):
        if type(bench_features_enabled) is not bool:
            raise ValueError('bench_features_enabled must be a boolean')
        super().__init__(observation_space, context_dim=context_dim, move_dim=move_dim,
                         relational_enabled=relational_enabled)
        self.bench_features_enabled = bench_features_enabled
        self.base_features_dim = self._features_dim
        self._features_dim += 6 * (FEATURE_DIM + 1)
        self.bench = BenchMatchupFeatures()

    def forward(self, obs):
        original = super().forward(obs)
        tokens = self.bench(obs)
        if not self.bench_features_enabled:
            tokens = torch.zeros_like(tokens)
        mask = obs['action_mask']
        # Native forced replacement has only switch actions. The extra guard
        # makes the disabled arm a common logit offset over every legal action.
        gate = ((obs['context'][:, 1] > 0) & (obs['context'][:, 2] == 0)
            & (mask[:, 4:10].sum(-1) >= 2) & (mask[:, :4].sum(-1) == 0)
            & (mask[:, 10] == 0))
        packed = torch.cat([tokens, (gate[:, None] * mask[:, 4:10])[..., None]], -1)
        return torch.cat([original, packed.flatten(1)], -1)


class _SwitchResidualScores(nn.Module):
    def __init__(self, original, base_width):
        super().__init__()
        self.context_dim, self.move_dim = original.context_dim, original.move_dim
        self.base_width = base_width
        # Keep every original parameter's name and shape for exact migration.
        self.moves, self.other = original.moves, original.other
        self.switch_residual = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        nn.init.zeros_(self.switch_residual[-1].weight)

    def residual_scores(self, features):
        packed = features[:, self.base_width:].reshape(-1, 6, FEATURE_DIM + 1)
        return self.switch_residual(packed[..., :FEATURE_DIM]).squeeze(-1) * packed[..., -1]

    def forward(self, features):
        context = features[:, :self.context_dim]
        moves = features[:, self.context_dim:self.base_width].reshape(-1, 4, self.move_dim)
        move_scores = self.moves(torch.cat([context[:, None].expand(-1, 4, -1), moves], -1)).squeeze(-1)
        other = self.other(context)
        return torch.cat([move_scores, other[:, :6] + self.residual_scores(features), other[:, 6:]], -1)


class MatchupSwitchPolicy(MatchupMovePolicy):
    """A learned correction, never a Rules action override or oracle."""
    ARCHITECTURE_VERSION = 1

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if not isinstance(action_space, gym.spaces.Discrete) or action_space.n != 11:
            raise ValueError('matchup-switches requires Discrete(11) actions')
        if kwargs.pop('features_extractor_class', MatchupSwitchFeatures) is not MatchupSwitchFeatures:
            raise ValueError('matchup-switches requires MatchupSwitchFeatures')
        if kwargs.get('share_features_extractor', True) is not True:
            raise ValueError('matchup-switches requires shared features')
        if kwargs.pop('net_arch', []) not in (None, []):
            raise ValueError('matchup-switches has a fixed actor/critic architecture')
        kwargs.setdefault('normalize_images', False)
        MaskableActorCriticPolicy.__init__(self, observation_space, action_space, lr_schedule,
            features_extractor_class=MatchupSwitchFeatures, net_arch=[], **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = _SwitchResidualScores(self.action_net, self.features_extractor.base_features_dim)
        self.mlp_extractor.latent_dim_pi = self.features_extractor.features_dim
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def create_switch_checkpoint(template_model, destination, *, enabled, seed, training_settings=None):
    """Convert a trained matchup policy without changing initial decisions.

Original parameter values and optimizer moments are copied by parameter name.
Only new residual parameters start with fresh optimizer state. Each arm must
use the same seed/settings; its feature switch is serialized in policy kwargs.
"""
    if type(enabled) is not bool:
        raise ValueError('enabled must be a boolean')
    if type(template_model.policy) is not MatchupMovePolicy:
        raise ValueError('template must be an original matchup-moves policy')
    if not template_model.policy.features_extractor.relational_enabled:
        raise ValueError('template must have matchup features enabled')
    destination = Path(destination)
    if destination.suffix != '.zip':
        destination = destination.with_suffix('.zip')
    if destination.exists():
        raise ValueError(f'checkpoint already exists: {destination}')
    names = ('learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda',
        'clip_range', 'clip_range_vf', 'normalize_advantage', 'ent_coef', 'vf_coef',
        'max_grad_norm', 'target_kl', 'device')
    settings = {name: getattr(template_model, name) for name in names}
    changes = dict(training_settings or {})
    if set(changes) - set(names):
        raise ValueError(f'unsupported training settings: {sorted(set(changes) - set(names))}')
    settings.update(changes)
    kwargs = deepcopy(template_model.policy_kwargs)
    kwargs.pop('features_extractor_class', None)
    features = template_model.policy.features_extractor
    kwargs['features_extractor_kwargs'] = dict(context_dim=features.context_dim,
        move_dim=features.base_move_dim, relational_enabled=True, bench_features_enabled=enabled)
    model = MaskablePPO(MatchupSwitchPolicy, None, policy_kwargs=kwargs, seed=seed,
        _init_setup_model=False, **settings)
    model.observation_space = deepcopy(template_model.observation_space)
    model.action_space = deepcopy(template_model.action_space)
    model.n_envs = template_model.n_envs
    model._setup_model()
    original, updated = template_model.policy.state_dict(), model.policy.state_dict()
    for name, value in original.items():
        if name not in updated or updated[name].shape != value.shape:
            raise ValueError(f'template policy cannot be migrated: {name}')
        updated[name].copy_(value.to(updated[name].device))
    model.policy.load_state_dict(updated)
    old_parameters = dict(template_model.policy.named_parameters())
    new_parameters = dict(model.policy.named_parameters())
    old_optimizer, new_optimizer = template_model.policy.optimizer, model.policy.optimizer
    if type(old_optimizer) is not type(new_optimizer) or len(old_optimizer.param_groups) != 1:
        raise ValueError('optimizer migration requires the same single-group optimizer')
    for key, value in old_optimizer.param_groups[0].items():
        if key != 'params':
            new_optimizer.param_groups[0][key] = deepcopy(value)
    for name, parameter in old_parameters.items():
        if parameter in old_optimizer.state:
            target = new_parameters[name]
            new_optimizer.state[target] = {key: (value.detach().clone().to(target.device)
                if torch.is_tensor(value) else deepcopy(value))
                for key, value in old_optimizer.state[parameter].items()}
    for name in ('num_timesteps', '_n_updates', '_current_progress_remaining'):
        if hasattr(template_model, name):
            setattr(model, name, deepcopy(getattr(template_model, name)))
    for name in ('battle_backend', 'reward_settings'):
        if hasattr(template_model, name):
            setattr(model, name, deepcopy(getattr(template_model, name)))
    model.observation_version = 'v2'
    model.policy_architecture = 'matchup-switches'
    model.policy_architecture_version = MatchupSwitchPolicy.ARCHITECTURE_VERSION
    model.switch_ablation = dict(version=1, enabled=enabled, template_timesteps=template_model.num_timesteps,
        initialization='original weights copied; shared forced-switch residual output zero',
        optimizer='preserved-existing/new-empty', seed=seed, feature_names=list(FEATURE_NAMES),
        scope='multi-option forced replacements only; original move/value paths retained')
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    return model
