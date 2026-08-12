"""Generalized Advantage Estimation (spec: GAE section).

Computed from the values stored at rollout time — update.py must never
recompute them. done_t marks a true episode end (t == max_iter); a
rollout-boundary truncation keeps done=0 so the last state bootstraps
through last_value.
"""

import torch


@torch.no_grad()
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Backward-accumulated GAE over a [T, N] rollout.

    rewards/values/dones: [T, N] float tensors (dones in {0, 1});
    last_value: [N] critic value of the state after the final stored
    step (used only where dones[-1] == 0).
    Returns (advantages, returns), both [T, N]; returns = adv + values.
    """
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(rewards.size(0))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    return advantages, advantages + values
