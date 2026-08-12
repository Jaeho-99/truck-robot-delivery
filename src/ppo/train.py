"""PPO training loop (spec: rollout section).

n_envs synchronous ALNS environments (VecALNS); every t_rollout vector
steps one PPO update. Environments keep running to max_iter across
rollout boundaries — the boundary truncation bootstraps through
get_value while done=True happens only at t == max_iter. Forward
passes are batched over the env vector (the destroy/repair work itself
is pure Python and stays sequential).
"""

import time

import torch
from torch_geometric.data import Batch

from ..gnn_dqn.graph_builder import GraphBuilder
from .actor_critic import ActorCritic
from .buffer import RolloutBuffer
from .env import ALNSEnv, VecALNS
from .update import make_optimizer, ppo_update


def make_envs(cfg, provider, builder):
    return VecALNS([ALNSEnv(provider, builder, cfg, cfg.seed + i)
                    for i in range(cfg.n_envs)])


def save_checkpoint(model, cfg, norms, path):
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "norms": norms}, path)


def train(cfg, provider, out_path, norms, log_rows=None):
    """Train and save a checkpoint to out_path. Returns the model."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = ActorCritic(cfg).to(device)
    opt = make_optimizer(model, cfg)
    vec = make_envs(cfg, provider, GraphBuilder(norms, cfg))
    gen = torch.Generator().manual_seed(cfg.seed)   # minibatch shuffle
    obs = vec.reset()
    t0 = time.time()

    for upd in range(cfg.n_updates):
        buf = RolloutBuffer(cfg.t_rollout, cfg.n_envs)
        done_bests = []         # f_best of episodes finished this rollout
        for _ in range(cfg.t_rollout):
            with torch.no_grad():
                a, logp, _, v = model.get_action_and_value(
                    Batch.from_data_list(obs).to(device))
            nxt, rewards, dones, infos = vec.step(a.tolist())
            buf.add(obs, a.cpu(), logp.cpu(), v.cpu(), rewards, dones)
            obs = nxt
            done_bests.extend(info["f_best"]
                              for d, info in zip(dones, infos) if d)
        with torch.no_grad():
            last_v = model.get_value(
                Batch.from_data_list(obs).to(device)).cpu()
        buf.compute_returns(last_v, cfg.gamma, cfg.gae_lambda)

        progress = upd / max(1, cfg.n_updates - 1)
        metrics = ppo_update(model, opt, buf, cfg, progress, gen)
        row = {"update": upd, **{k: round(v, 6)
                                 for k, v in metrics.items()},
               "mean_best_objective":
                   round(sum(done_bests) / len(done_bests), 4)
                   if done_bests else None,
               "episodes_done": len(done_bests),
               "elapsed_s": round(time.time() - t0, 1)}
        if log_rows is not None:
            log_rows.append(row)
        print(f"[upd {upd}] ent={row['entropy']:.3f} "
              f"kl={row['approx_kl']:.5f} pg={row['pg_loss']:.5f} "
              f"v={row['v_loss']:.5f} ev={row['explained_variance']:.3f} "
              f"best={row['mean_best_objective']} "
              f"({row['episodes_done']} eps) {row['elapsed_s']}s",
              flush=True)

        if (upd + 1) % 10 == 0 or upd == cfg.n_updates - 1:
            save_checkpoint(model, cfg, norms, out_path)
    save_checkpoint(model, cfg, norms, out_path)
    return model
