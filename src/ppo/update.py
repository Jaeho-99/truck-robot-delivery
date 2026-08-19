"""Clipped PPO update step (spec: PPO update section).

One rollout -> up to k_epochs of minibatch updates through a single
Adam and a single backward per minibatch (no optimizer split). Stored
log_probs/values come from rollout time; only new_logp/new_v are
recomputed here.
"""

import torch
import torch.nn as nn


def make_optimizer(model, cfg):
    """Single Adam over ALL parameters (spec: eps=1e-5, no split)."""
    return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                            eps=cfg.adam_eps)


def c2_schedule(cfg, progress):
    """Entropy coefficient: linear c2_start -> c2_end over training.

    progress = update_idx / (n_updates - 1), clipped to [0, 1].
    """
    p = min(max(progress, 0.0), 1.0)
    return cfg.c2_start + (cfg.c2_end - cfg.c2_start) * p


def value_loss(new_v, old_v, ret, eps):
    """Clipped value loss, pessimistic MAX of the two errors (spec)."""
    v_clip = old_v + torch.clamp(new_v - old_v, -eps, eps)
    return torch.max((new_v - ret) ** 2, (v_clip - ret) ** 2).mean()


def ppo_update(model, optimizer, buffer, cfg, progress, generator=None):
    """Run the PPO epochs on one rollout; returns a metrics dict.

    Stops remaining epochs when the epoch-end approx_kl exceeds
    1.5 * target_kl. Advantages are normalized per minibatch.
    """
    device = next(model.parameters()).device
    c2 = c2_schedule(cfg, progress)
    eps = cfg.clip_eps
    pg_losses, v_losses, entropies, clipfracs = [], [], [], []
    approx_kl = 0.0
    epochs_run = 0

    for _ in range(cfg.k_epochs):
        kls = []
        for batch, actions, old_logp, adv, ret, old_v in \
                buffer.minibatches(cfg.n_minibatch, generator):
            batch = batch.to(device)
            actions, old_logp, adv, ret, old_v = (
                x.to(device)
                for x in (actions, old_logp, adv, ret, old_v))
            _, new_logp, entropy, new_v = model.get_action_and_value(
                batch, action=actions)
            log_ratio = new_logp - old_logp
            ratio = log_ratio.exp()

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
            v_loss = value_loss(new_v, old_v, ret, eps)
            ent = entropy.mean()
            loss = pg_loss + cfg.c1 * v_loss - c2 * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                                     cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kls.append(((ratio - 1) - log_ratio).mean().item())
                clipfracs.append(
                    ((ratio - 1).abs() > eps).float().mean().item())
            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(ent.item())
        epochs_run += 1
        approx_kl = sum(kls) / len(kls)
        # SB3/DR-ALNS default target_kl=None disables the early stop
        if (cfg.target_kl is not None
                and approx_kl > 1.5 * cfg.target_kl):
            break

    # explained variance of the rollout-time value estimates
    y = buffer.returns.view(-1)
    v = torch.stack(buffer.values).view(-1)
    var_y = y.var()
    ev = (float("nan") if var_y == 0
          else (1.0 - (y - v).var() / var_y).item())

    def mean(xs):
        return sum(xs) / len(xs)

    return {"pg_loss": mean(pg_losses), "v_loss": mean(v_losses),
            "entropy": mean(entropies), "approx_kl": approx_kl,
            "clipfrac": mean(clipfracs), "explained_variance": ev,
            "c2": c2, "epochs_run": epochs_run}
