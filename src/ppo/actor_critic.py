"""ActorCritic model for PPO (spec: model section).

Reuses SolutionEncoder unchanged (GATv2 + HeteroConv, joint mean/max
pooling); state = encoder(data) || g_t -> d_state = 135 with defaults.
g_t is attached to the (batched) HeteroData as ``data.g``, matching the
QNet / ReplayBuffer convention. Heads carry no Dropout/BatchNorm so
rollout and update see identical outputs.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..gnn_dqn.encoder import SolutionEncoder
from ..gnn_dqn.global_features import G_DIM


def _ortho(layer, std):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = SolutionEncoder(cfg)
        d_state = 2 * cfg.hidden_dim + G_DIM
        self.actor = nn.Sequential(
            _ortho(nn.Linear(d_state, 64), 2 ** 0.5), nn.Tanh(),
            _ortho(nn.Linear(64, cfg.n_actions), 0.01))
        self.critic = nn.Sequential(
            _ortho(nn.Linear(d_state, 64), 2 ** 0.5), nn.Tanh(),
            _ortho(nn.Linear(64, 1), 1.0))

    def _state(self, data):
        g = data.g.view(-1, G_DIM)
        return torch.cat([self.encoder(data), g], dim=1)

    def get_action_and_value(self, data, action=None):
        """Single entry point for rollout and update (spec).

        action=None samples (rollout); a given action gets its log_prob
        re-evaluated (update). Returns (action, log_prob, entropy,
        value), each of shape (B,).
        """
        s = self._state(data)
        dist = Categorical(logits=self.actor(s))
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action), dist.entropy(),
                self.critic(s).squeeze(-1))

    def get_value(self, data):
        """Critic value (B,) only — bootstrap at rollout boundaries."""
        return self.critic(self._state(data)).squeeze(-1)
