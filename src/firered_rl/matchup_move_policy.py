"""Controlled observation-derived matchup features for the shared move policy.

The external v2 observation, legal actions and rewards are unchanged. Features
describe attacks against the current defender, not a prediction of its action.
Status and unusual damage mechanics are deliberately unassessed. The serialized
``features_extractor_kwargs.relational_enabled`` switch makes the control emit
zeros with exactly the same architecture and trainable parameter count.
"""
from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from .shared_move_policy import SharedMoveFeatures, SharedMovePolicy


RELATION_DIM = 5
# Fixed/counter/OHKO/delayed/special damage is not ordinary typed direct damage.
# Weather Ball's dynamically changed type is not represented by native v2.
UNASSESSED_MOVES = frozenset({165, 248, 251, 267, 311, 353})
SOUND_ATTACKS = frozenset({173, 253, 304})  # Snore, Uproar, Hyper Voice.


class MatchupRelations(nn.Module):
    """Vectorized, RNG-free rules from already-visible v2 fields.

    Output columns: assessed; chart multiplier / 4; current type/ability-adjusted
    multiplier / 4; current immunity; STAB multiplier / 1.5. Unassessed rows are
    all zero. Zero adjusted multiplier means immunity only when assessed is 1.
    Buffers are serialized, travel with the policy device and never train.
    """
    def __init__(self):
        super().__init__()
        root = Path(__file__).resolve().parents[2]
        data = json.loads((root / 'config/generation/gen3_type_chart.json').read_text())
        chart = torch.ones(32, 32)
        values = torch.tensor(data['multipliers'], dtype=torch.float32)
        chart[:values.shape[0], :values.shape[1]] = values
        catalog = json.loads((root / 'config/generation/firered_us_v1_0.json').read_text())
        supported = torch.zeros(512, dtype=torch.bool)
        for move in catalog['moves'].values():
            if move['id'] not in UNASSESSED_MOVES:
                supported[move['id']] = True
        sound = torch.zeros(512, dtype=torch.bool)
        sound[list(SOUND_ATTACKS)] = True
        self.register_buffer('type_chart', chart)
        self.register_buffer('supported_moves', supported)
        self.register_buffer('sound_moves', sound)

    def forward(self, obs):
        batch = torch.arange(len(obs['active']), device=obs['active'].device)
        raw_slot = obs['active'][:, 0, 8] - 1
        slot_valid = (raw_slot >= 0) & (raw_slot < 6) & (raw_slot == raw_slot.floor())
        slot = raw_slot.long().clamp(0, 5)
        moves = obs['moves'][batch, slot]
        details = obs['v2_move_details'][batch, 0, slot]
        move_ids = moves[..., 0].long()
        kinds = details[..., 5].long()
        target = obs['active'][:, 1, 5:7].long()
        own_types = obs['active'][:, 0, 5:7].long()
        move_valid = ((move_ids > 0) & (move_ids < len(self.supported_moves))
                      & self.supported_moves[move_ids.clamp(0, 511)])
        known_types = ((kinds >= 0) & (kinds < 18)
                       & ((target >= 0) & (target < 18)).all(-1)[:, None])
        assessed = (move_valid & known_types & (details[..., 0] > 1)
                    & (moves[..., 2] > 0)
                    & ((details[..., 3] == 0) | (details[..., 3] == 1))
                    & slot_valid[:, None] & (obs['active'][..., 0] > 0).all(-1)[:, None]
                    & (obs['v2_active'][:, 1, 0] > 0)[:, None]
                    & (obs['context'][:, 2] == 0)[:, None])
        kind_index = kinds.clamp(0, 31)
        target_index = target.clamp(0, 31)
        first = self.type_chart[kind_index, target_index[:, None, 0]]
        second = self.type_chart[kind_index, target_index[:, None, 1]]
        second = torch.where((target[:, 0] == target[:, 1])[:, None], 1., second)
        pure = first * second

        # Foresight removes Ghost's Normal/Fighting immunity, but not other
        # resistances or ability immunities. The pure chart column stays pure.
        foresight = (obs['v2_volatile_bits'][:, 1, 29] > 0)[:, None]
        ignores_ghost = foresight & ((kinds == 0) | (kinds == 1))
        first = torch.where(ignores_ghost & (target[:, None, 0] == 7), 1., first)
        second = torch.where(ignores_ghost & (target[:, None, 1] == 7), 1., second)
        adjusted = first * second
        ability = obs['v2_active'][:, 1, 3].long()[:, None]
        frozen = (obs['status_bits'][:, 7, 5] > 0)[:, None]
        ability_immune = (((ability == 26) & (kinds == 4))
                          | ((ability == 10) & (kinds == 13))
                          | ((ability == 11) & (kinds == 11))
                          | ((ability == 18) & (kinds == 10) & ~frozen)
                          | ((ability == 25) & (adjusted <= 1))
                          | ((ability == 43) & self.sound_moves[move_ids.clamp(0, 511)]))
        adjusted = torch.where(ability_immune, 0., adjusted)
        stab = torch.where((kinds[..., None] == own_types[:, None]).any(-1), 1.5, 1.)
        return torch.stack([assessed.float(), pure / 4, adjusted / 4,
                            (adjusted == 0).float(), stab / 1.5], -1) * assessed[..., None]


class MatchupMoveFeatures(SharedMoveFeatures):
    """Original encoder plus five relationships for each active move.

    Defaults match SharedMoveFeatures: context_dim=128, move_dim=32. Here
    ``move_dim`` is the original learned token width, before five extra values.
    A zero-initialized mean-pooling residual also exposes relationships to the
    switch scorer and critic while preserving move permutation symmetry.
    """
    def __init__(self, observation_space, context_dim=128, move_dim=32,
                 relational_enabled=True):
        if type(relational_enabled) is not bool:
            raise ValueError('relational_enabled must be a boolean')
        super().__init__(observation_space, context_dim=context_dim, move_dim=move_dim)
        self.base_move_dim = move_dim
        self.move_dim = move_dim + RELATION_DIM
        self._features_dim = context_dim + 4 * self.move_dim
        self.relational_enabled = relational_enabled
        self.relations = MatchupRelations()
        self.relation_context = nn.Linear(RELATION_DIM, context_dim, bias=False)
        nn.init.zeros_(self.relation_context.weight)

    def forward(self, obs):
        original = super().forward(obs)
        context = original[:, :self.context_dim]
        moves = original[:, self.context_dim:].reshape(-1, 4, self.base_move_dim)
        relationships = self.relations(obs)
        if not self.relational_enabled:
            relationships = torch.zeros_like(relationships)
        context = context + self.relation_context(relationships.mean(1))
        return torch.cat([context, torch.cat([moves, relationships], -1).flatten(1)], -1)


class MatchupMovePolicy(SharedMovePolicy):
    """Shared learned scorer; game rules are features, never forced actions."""
    ARCHITECTURE_VERSION = 1

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if not isinstance(action_space, gym.spaces.Discrete) or action_space.n != 11:
            raise ValueError('matchup-moves requires Discrete(11) actions')
        if kwargs.pop('features_extractor_class', MatchupMoveFeatures) is not MatchupMoveFeatures:
            raise ValueError('matchup-moves requires MatchupMoveFeatures')
        if kwargs.get('share_features_extractor', True) is not True:
            raise ValueError('matchup-moves requires shared features')
        if kwargs.pop('net_arch', []) not in (None, []):
            raise ValueError('matchup-moves has a fixed actor/critic architecture')
        kwargs.setdefault('normalize_images', False)
        # SharedMovePolicy intentionally rejects other extractors, so enter its
        # common parent here and retain SharedMovePolicy's build and heads.
        MaskableActorCriticPolicy.__init__(self, observation_space, action_space,
            lr_schedule, features_extractor_class=MatchupMoveFeatures, net_arch=[], **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        features = self.features_extractor
        with torch.no_grad():
            self.action_net.moves[0].weight[:, features.context_dim + features.base_move_dim:] = 0
            features.relation_context.weight.zero_()


def create_ablation_checkpoint(template_model, destination, *, enabled, seed,
                               training_settings=None):
    """Save a feature-on/off model from the same genuinely untrained template.

    All old actor, critic and encoder weights are copied. New input weights are
    zero, so old and new policies have equivalent initial logits and values.
    Both ablation arms have the same parameter count and empty optimizer state.
    Allowed training_settings are the PPO constructor settings listed below;
    absent settings retain the template's values. The template is not mutated.
    """
    if type(enabled) is not bool:
        raise ValueError('enabled must be a boolean')
    if type(template_model.policy) is not SharedMovePolicy:
        raise ValueError('template must be an original shared-moves policy')
    if template_model.num_timesteps != 0 or template_model.policy.optimizer.state:
        raise ValueError('template must have zero training timesteps and empty optimizer state')
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
        move_dim=features.move_dim, relational_enabled=enabled)
    model = MaskablePPO(MatchupMovePolicy, None, policy_kwargs=kwargs, seed=seed,
                        _init_setup_model=False, **settings)
    model.observation_space = deepcopy(template_model.observation_space)
    model.action_space = deepcopy(template_model.action_space)
    model.n_envs = template_model.n_envs
    model._setup_model()
    old, new = template_model.policy.state_dict(), model.policy.state_dict()
    for name, value in old.items():
        if name == 'action_net.moves.0.weight':
            new[name].zero_()
            new[name][:, :value.shape[1]].copy_(value.to(new[name].device))
        elif name in new and new[name].shape == value.shape:
            new[name].copy_(value.to(new[name].device))
        else:
            raise ValueError(f'template policy cannot be migrated: {name}')
    model.policy.load_state_dict(new)
    model.observation_version = 'v2'
    model.policy_architecture = 'matchup-moves'
    model.policy_architecture_version = MatchupMovePolicy.ARCHITECTURE_VERSION
    for name in ('battle_backend', 'reward_settings'):
        if hasattr(template_model, name):
            setattr(model, name, deepcopy(getattr(template_model, name)))
    model.matchup_ablation = dict(version=1, enabled=enabled, template_timesteps=0,
        initialization='all shared-moves weights copied; new relation weights zero',
        optimizer='fresh', seed=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(destination)
    return model
