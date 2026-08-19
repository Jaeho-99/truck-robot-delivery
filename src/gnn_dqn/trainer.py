"""Offline training loop for the GNN+DQN operator selector.

DR-ALNS protocol (Reijnen et al., ICAPS 2024): training runs for
cfg.total_steps search iterations, chopped into episodes of
cfg.search_iterations (100, fixed regardless of instance size). At
each episode start one instance is drawn with replacement
(random.choice) from the provider's train pool. One model is trained
per instance size (see experiments/train_gnn_dqn.py --size).

Terminology: one destroy -> repair -> accept cycle is a "search
iteration" (`it` within an episode); `step` is the cumulative counter
across all episodes. The per-episode ALNS body mirrors solve_alns
(destroy -> repair -> SA accept) with the Santini-style linear
temperature decay used for training episodes; solve_alns itself is
untouched so the roulette path stays byte-identical.
"""

import math
import random
import statistics
import time
from collections import deque

from ..heuristics.alns import NOISE_FRAC, congestion_aware_initial
from ..heuristics.operators import (DESTROY, repair_greedy,
                                    repair_regret2)
from ..heuristics.solution import eval_solution
from .dqn_agent import DQNAgent
from .global_features import global_features
from .graph_builder import GraphBuilder
from .reward import compute_reward

W_START = 0.25          # SA start temperature fraction (as solve_alns)

REWARD_WINDOW = 100     # rolling stats window (episodes)


def repair_ops(noise_amp):
    return [lambda p, s, pool, rng: repair_greedy(p, s, pool, rng, 0.0),
            lambda p, s, pool, rng: repair_greedy(p, s, pool, rng,
                                                  noise_amp),
            repair_regret2]


def train(cfg, provider, out_path, norms, log_rows=None):
    """Train and save a checkpoint to out_path. Returns the agent."""
    agent = DQNAgent(cfg, cfg.total_steps)
    builder = GraphBuilder(norms, cfg)
    rng = random.Random(cfg.seed)
    step = 0                        # cumulative search iterations
    recent_rewards = deque(maxlen=REWARD_WINDOW)
    t0 = time.time()

    for episode in range(cfg.n_episodes):
        pr = provider.sample()      # random.choice, with replacement
        current_solution = congestion_aware_initial(pr, rng)
        current_cost, _, _, _ = eval_solution(pr, current_solution)
        init_cost = best_cost = current_cost
        stagcount = 0               # iterations since best improved
        nC = len(pr.C)
        # destroy size ("degree of destruction"): q customers removed,
        # drawn uniformly from [qmin, qmax] each iteration
        qmin, qmax = 1, max(2, round(0.35 * nC))
        noise_amp = NOISE_FRAC * init_cost
        repairs = repair_ops(noise_amp)
        T0 = (W_START * init_cost) / math.log(2)

        G = builder.build(pr, current_solution)
        G.g = global_features(pr, current_solution, 0.0, 0,
                              current_cost, best_cost)
        ep_reward, ep_losses = 0.0, []
        ep_actions = [0] * cfg.n_actions

        for it in range(cfg.search_iterations):
            a = agent.act(G, step)
            ep_actions[a] += 1
            di, ri = divmod(a, 3)

            cand = current_solution.clone()
            pool = DESTROY[di][1](pr, cand, rng.randint(qmin, qmax),
                                  rng)
            repairs[ri](pr, cand, pool, rng)
            cand_cost, ok, _, _ = eval_solution(pr, cand)

            T = T0 * (1.0 - it / cfg.search_iterations)  # linear decay
            accepted = ok and (
                cand_cost < current_cost - 1e-9
                or rng.random() < math.exp(-(cand_cost - current_cost)
                                           / max(T, 1e-9)))
            # reward_mode "binary" (+5 on new best only) corresponds to
            # DR-ALNS's reward function; see reward.py
            r = compute_reward(current_cost, cand_cost, init_cost,
                               best_cost, accepted, cfg) if ok else 0.0
            improved = ok and cand_cost < best_cost - 1e-9
            if improved:
                best_cost = cand_cost
            stagcount = 0 if improved else stagcount + 1
            if accepted:
                current_solution, current_cost = cand, cand_cost

            G_new = builder.build(pr, current_solution)
            G_new.g = global_features(
                pr, current_solution,
                (it + 1) / cfg.search_iterations, stagcount,
                current_cost, best_cost)
            agent.buffer.push(G, a, r, G_new)
            G = G_new
            ep_reward += r
            step += 1

            if (len(agent.buffer) >= cfg.warmup
                    and step % cfg.train_freq == 0):
                ep_losses.append(agent.update())
            if step % cfg.target_sync == 0:
                agent.sync_target()

        recent_rewards.append(ep_reward)
        roll_mean = statistics.fmean(recent_rewards)
        roll_std = (statistics.pstdev(recent_rewards)
                    if len(recent_rewards) > 1 else 0.0)
        row = {"episode": episode, "step": step,
               "instance_id": pr.inst.get("instance_id"),
               "n_cust": nC,
               "episode_reward": round(ep_reward, 6),
               "reward_roll_mean": round(roll_mean, 6),
               "reward_roll_std": round(roll_std, 6),
               "best_cost": round(best_cost, 4),
               "init_cost": round(init_cost, 4),
               "eps": round(agent.epsilon(step), 3),
               "mean_loss": round(sum(ep_losses) / len(ep_losses), 6)
               if ep_losses else None,
               "actions": ep_actions,
               "elapsed_s": round(time.time() - t0, 1)}
        if log_rows is not None:
            log_rows.append(row)
        print(f"[ep {episode}] {row['instance_id']} "
              f"best={best_cost:.2f} (init {init_cost:.2f}) "
              f"R={ep_reward:.4f} (roll {roll_mean:.4f}) "
              f"eps={row['eps']} loss={row['mean_loss']} "
              f"{row['elapsed_s']}s", flush=True)

        if (episode + 1) % 10 == 0 or episode == cfg.n_episodes - 1:
            agent.save(out_path, norms)
    agent.save(out_path, norms)
    return agent
