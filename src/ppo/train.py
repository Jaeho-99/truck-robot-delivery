"""PPO training loop (DR-ALNS protocol).

cfg.n_envs synchronous ALNS environments (VecALNS); every
cfg.t_rollout vector steps one PPO update, for
cfg.n_updates = total_steps / (t_rollout * n_envs) updates.
Environments run episodes of cfg.search_iterations and keep running
across rollout boundaries — boundary truncation bootstraps through
get_value while done=True happens only at episode end. Episode
rewards are logged with rolling mean/std (window 100), matching the
DQN trainer's CSV format so plot_training.py reads both.
"""

import statistics
import time
from collections import deque

import torch
from torch_geometric.data import Batch

from ..gnn_dqn.graph_builder import GraphBuilder
from .actor_critic import ActorCritic
from .buffer import RolloutBuffer
from .env import ALNSEnv, VecALNS
from .update import make_optimizer, ppo_update

REWARD_WINDOW = 100     # rolling stats window (episodes)


def make_envs(cfg, provider, builder):
    return VecALNS([ALNSEnv(provider, builder, cfg, cfg.seed + i)
                    for i in range(cfg.n_envs)])


def save_checkpoint(model, cfg, norms, path):
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "norms": norms}, path)


def train(cfg, provider, out_path, norms, log_rows=None,
          update_rows=None):
    """Train and save a checkpoint to out_path. Returns the model.

    log_rows: per-episode reward rows (CSV/plot format);
    update_rows: per-update PPO metrics.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = ActorCritic(cfg).to(device)
    opt = make_optimizer(model, cfg)
    vec = make_envs(cfg, provider, GraphBuilder(norms, cfg))
    gen = torch.Generator().manual_seed(cfg.seed)   # minibatch shuffle
    obs = vec.reset()
    ep_reward = [0.0] * cfg.n_envs
    recent_rewards = deque(maxlen=REWARD_WINDOW)
    episodes_done = 0
    step = 0                    # cumulative env steps over all workers
    t0 = time.time()

    for upd in range(cfg.n_updates):
        buf = RolloutBuffer(cfg.t_rollout, cfg.n_envs)
        for _ in range(cfg.t_rollout):
            with torch.no_grad():
                a, logp, _, v = model.get_action_and_value(
                    Batch.from_data_list(obs).to(device))
            nxt, rewards, dones, infos = vec.step(a.tolist())
            buf.add(obs, a.cpu(), logp.cpu(), v.cpu(), rewards, dones)
            obs = nxt
            step += cfg.n_envs
            for i, (r, d, info) in enumerate(zip(rewards, dones,
                                                 infos)):
                ep_reward[i] += r
                if not d:
                    continue
                recent_rewards.append(ep_reward[i])
                roll_mean = statistics.fmean(recent_rewards)
                roll_std = (statistics.pstdev(recent_rewards)
                            if len(recent_rewards) > 1 else 0.0)
                if log_rows is not None:
                    log_rows.append({
                        "episode": episodes_done, "step": step,
                        "instance_id": info["instance_id"],
                        "episode_reward": round(ep_reward[i], 6),
                        "reward_roll_mean": round(roll_mean, 6),
                        "reward_roll_std": round(roll_std, 6),
                        "best_cost": round(info["f_best"], 4),
                        "init_cost": round(info["f_init"], 4),
                        "elapsed_s": round(time.time() - t0, 1)})
                episodes_done += 1
                ep_reward[i] = 0.0
        with torch.no_grad():
            last_v = model.get_value(
                Batch.from_data_list(obs).to(device)).cpu()
        buf.compute_returns(last_v, cfg.gamma, cfg.gae_lambda)

        progress = upd / max(1, cfg.n_updates - 1)
        metrics = ppo_update(model, opt, buf, cfg, progress, gen)
        if update_rows is not None:
            update_rows.append({"update": upd, "step": step, **metrics})
        roll = (statistics.fmean(recent_rewards)
                if recent_rewards else 0.0)
        print(f"[upd {upd}] step={step} ent={metrics['entropy']:.3f} "
              f"kl={metrics['approx_kl']:.5f} "
              f"pg={metrics['pg_loss']:.5f} "
              f"v={metrics['v_loss']:.5f} "
              f"ev={metrics['explained_variance']:.3f} "
              f"eps_done={episodes_done} rollR={roll:.3f} "
              f"{round(time.time() - t0, 1)}s", flush=True)

        if (upd + 1) % 10 == 0 or upd == cfg.n_updates - 1:
            save_checkpoint(model, cfg, norms, out_path)
    save_checkpoint(model, cfg, norms, out_path)
    return model
