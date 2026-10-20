"""Optional v2 policy whose move scores follow move contents, not slot position.

The external observation/action contract is unchanged. Party order is retained;
only the four move slots within each Pokemon are exchangeable. All four move
logits use the same scorer, and actor/critic context pools moves symmetrically.
"""
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from .ppo_features import BattleFeatures, observation_version


class SharedMoveFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, context_dim=128, move_dim=32):
        if observation_version(observation_space) != 'v2':
            raise ValueError('shared-moves requires complete v2 policy observations')
        if context_dim < 1 or move_dim < 1:
            raise ValueError('context_dim and move_dim must be positive')
        super().__init__(observation_space, context_dim + 4 * move_dim)
        self.context_dim, self.move_dim = context_dim, move_dim
        self.state = BattleFeatures(observation_space, features_dim=context_dim, include_moves=False)
        # Move embedding, base type, PP; v2 power/accuracy/category/priority/type;
        # active learner legality. There is no move-slot index or positional bias.
        self.move_encoder = nn.Sequential(nn.Linear(34, 64), nn.ReLU(),
                                          nn.Linear(64, move_dim), nn.ReLU())
        self.context = nn.Sequential(nn.Linear(context_dim + 12 * move_dim, context_dim), nn.ReLU())

    def forward(self, obs):
        moves = torch.stack([obs['moves'], obs['opponent_moves']], dim=1)
        details = obs['v2_move_details']
        categories = torch.nn.functional.one_hot(self.state.ids(details[..., 3], 3), 3).float()
        slots = obs['active'][:, 0, 8] - 1
        valid_active = (slots >= 0) & (slots < 6) & (slots == slots.floor())
        indices = slots.long().clamp(0, 5)
        active_row = torch.nn.functional.one_hot(indices, 6) * valid_active[:, None]
        own_legal = active_row[..., None] * obs['action_mask'][:, None, :4]
        legal = torch.stack([own_legal, torch.zeros_like(own_legal)], dim=1).unsqueeze(-1)
        values = torch.cat([
            self.state.move(self.state.ids(moves[..., 0])),
            self.state.types(self.state.ids(moves[..., 1], 32)),
            (moves[..., 2:4] / 64).clamp(0, 1),
            details[..., :1] / 250, details[..., 1:2] / 100, details[..., 2:3],
            categories, details[..., 4:5] / 7,
            self.state.types(self.state.ids(details[..., 5], 32)), legal,
        ], dim=-1)
        valid = (moves[..., :1] != 0) * obs['v2_party'][..., :1].unsqueeze(-2)
        tokens = self.move_encoder(values) * valid

        # Never flatten ordered moves or ordered move-mask bits into context.
        # Keep every party position, but average its four masked move tokens.
        invariant = dict(obs)
        mask = obs['action_mask']
        invariant['action_mask'] = torch.cat([
            mask[:, :4].mean(dim=1, keepdim=True).expand(-1, 4), mask[:, 4:]
        ], dim=1)
        state = self.state(invariant)
        context = self.context(torch.cat([state, tokens.mean(dim=3).flatten(1)], dim=1))
        active = tokens[torch.arange(len(slots), device=slots.device), 0, indices]
        active = active * valid_active[:, None, None]
        return torch.cat([context, active.flatten(1)], dim=1)


class _SharedMoveLatents(nn.Module):
    def __init__(self, context_dim, move_dim):
        super().__init__()
        self.context_dim = context_dim
        self.latent_dim_pi = context_dim + 4 * move_dim
        self.latent_dim_vf = 64
        self.critic = nn.Sequential(nn.Linear(context_dim, 128), nn.Tanh(),
                                    nn.Linear(128, 64), nn.Tanh())

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return self.critic(features[:, :self.context_dim])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class _SharedMoveScores(nn.Module):
    def __init__(self, context_dim, move_dim):
        super().__init__()
        self.context_dim, self.move_dim = context_dim, move_dim
        self.moves = nn.Sequential(nn.Linear(context_dim + move_dim, 128), nn.Tanh(),
                                   nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        # Switch actions keep their original party-slot meaning. Action 10 is
        # wait/forced continuation; no new item-use actions are introduced.
        self.other = nn.Sequential(nn.Linear(context_dim, 128), nn.Tanh(),
                                   nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 7))

    def forward(self, features):
        context = features[:, :self.context_dim]
        moves = features[:, self.context_dim:].reshape(-1, 4, self.move_dim)
        scores = self.moves(torch.cat([context[:, None].expand(-1, 4, -1), moves], dim=-1)).squeeze(-1)
        return torch.cat([scores, self.other(context)], dim=1)


class SharedMovePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy; serialized class is authoritative when resuming."""
    ARCHITECTURE_VERSION = 1

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        if not isinstance(action_space, gym.spaces.Discrete) or action_space.n != 11:
            raise ValueError('shared-moves requires Discrete(11) actions')
        extractor = kwargs.pop('features_extractor_class', SharedMoveFeatures)
        if extractor is not SharedMoveFeatures:
            raise ValueError('shared-moves requires SharedMoveFeatures')
        if kwargs.get('share_features_extractor', True) is not True:
            raise ValueError('shared-moves uses one actor/critic feature extractor')
        if kwargs.pop('net_arch', []) not in (None, []):
            raise ValueError('shared-moves has a fixed actor/critic architecture')
        kwargs.setdefault('normalize_images', False)
        super().__init__(observation_space, action_space, lr_schedule,
                         features_extractor_class=extractor, net_arch=[], **kwargs)

    def _build(self, lr_schedule):
        features = self.features_extractor
        self.mlp_extractor = _SharedMoveLatents(features.context_dim, features.move_dim)
        self.action_net = _SharedMoveScores(features.context_dim, features.move_dim)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            for module in (self.features_extractor, self.mlp_extractor, self.action_net):
                module.apply(partial(self.init_weights, gain=np.sqrt(2)))
            for output in (self.action_net.moves[-1], self.action_net.other[-1]):
                output.apply(partial(self.init_weights, gain=.01))
            self.value_net.apply(partial(self.init_weights, gain=1))
        # Build the optimizer only after every custom head has been registered.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def policy_architecture(model):
    # Check the most specific serialized subclass first; old checkpoints keep
    # their original classes and do not acquire a new head on ordinary load.
    from .matchup_switch_policy import MatchupSwitchPolicy
    from .matchup_move_policy import MatchupMovePolicy
    if isinstance(model.policy, MatchupSwitchPolicy):
        return 'matchup-switches'
    if isinstance(model.policy, MatchupMovePolicy):
        return 'matchup-moves'
    return 'shared-moves' if isinstance(model.policy, SharedMovePolicy) else 'flat'
