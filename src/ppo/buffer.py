"""On-policy rollout storage for PPO (spec: buffer).

Holds exactly one rollout (t_rollout x n_envs) and is reset after
every update — never a replay buffer (PPO is on-policy). Graphs are
stored on CPU with g_t attached as ``data.g`` (QNet/ReplayBuffer
convention); minibatches are assembled with Batch.from_data_list.
"""

import torch
from torch_geometric.data import Batch

from .gae import compute_gae


class RolloutBuffer:
    def __init__(self, t_rollout, n_envs):
        self.t_rollout = t_rollout
        self.n_envs = n_envs
        self.reset()

    def reset(self):
        self.obs = []           # flat, t-major: index = t * n_envs + i
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.actions)        # steps stored (of t_rollout)

    def add(self, obs, actions, log_probs, values, rewards, dones):
        """Store one vector step: obs is a list of n_envs HeteroData
        (with .g); the rest are tensors/sequences of shape (n_envs,).
        """
        assert len(self.actions) < self.t_rollout, "buffer full"
        self.obs.extend(obs)
        self.actions.append(
            torch.as_tensor(actions, dtype=torch.long))
        self.log_probs.append(
            torch.as_tensor(log_probs, dtype=torch.float32).detach())
        self.values.append(
            torch.as_tensor(values, dtype=torch.float32).detach())
        self.rewards.append(
            torch.as_tensor(rewards, dtype=torch.float32))
        self.dones.append(
            torch.as_tensor(dones, dtype=torch.float32))

    def compute_returns(self, last_value, gamma, gae_lambda):
        """GAE from the stored rollout-time values (spec)."""
        self.advantages, self.returns = compute_gae(
            torch.stack(self.rewards), torch.stack(self.values),
            torch.stack(self.dones),
            torch.as_tensor(last_value, dtype=torch.float32).detach(),
            gamma, gae_lambda)

    def minibatches(self, n_minibatch, generator=None):
        """Shuffled minibatches covering the rollout exactly once.

        Yields (graph Batch, actions, log_probs, advantages, returns,
        values), each flattened t-major and index-aligned. Advantages
        are raw — normalization is per-minibatch in the update (spec).
        """
        flat = lambda xs: torch.stack(xs).view(-1)  # noqa: E731
        actions = flat(self.actions)
        log_probs = flat(self.log_probs)
        values = flat(self.values)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)
        perm = torch.randperm(len(self.obs), generator=generator)
        for chunk in perm.chunk(n_minibatch):
            idx = chunk.tolist()
            yield (Batch.from_data_list([self.obs[i] for i in idx]),
                   actions[chunk], log_probs[chunk], advantages[chunk],
                   returns[chunk], values[chunk])
