"""Offline training loop for the GNN+DQN operator selector.

Each episode: sample a training instance, build the initial solution
(reusing alns.congestion_aware_initial), then run episode_len ALNS
iterations where the DQN picks the (destroy, repair) pair. The ALNS
step below mirrors the per-iteration body of solve_alns (destroy ->
repair -> SA accept) with the Santini-style linear temperature decay
used for training episodes; solve_alns itself is left untouched so the
roulette path stays byte-identical.
"""

import math
import random
import time

from ..heuristics.alns import NOISE_FRAC, congestion_aware_initial
from ..heuristics.operators import (DESTROY, repair_greedy,
                                    repair_regret2)
from ..heuristics.solution import eval_solution
from .dqn_agent import DQNAgent
from .global_features import global_features
from .graph_builder import GraphBuilder
from .reward import compute_reward

W_START = 0.25          # SA start temperature fraction (as solve_alns)


def repair_ops(noise_amp):
    return [lambda p, s, pool, rng: repair_greedy(p, s, pool, rng, 0.0),
            lambda p, s, pool, rng: repair_greedy(p, s, pool, rng,
                                                  noise_amp),
            repair_regret2]


def train(cfg, provider, out_path, norms, log_rows=None):
    """Train and save a checkpoint to out_path. Returns the agent."""
    total_steps = cfg.n_episodes * cfg.episode_len
    agent = DQNAgent(cfg, total_steps)
    builder = GraphBuilder(norms, cfg)
    rng = random.Random(cfg.seed)
    global_step = 0
    t0 = time.time()

    for ep in range(cfg.n_episodes):
        pr = provider.sample()
        sol = congestion_aware_initial(pr, rng)
        f_cur, _, _, _ = eval_solution(pr, sol)
        f_init = f_best = f_cur
        best_it = 0
        nC = len(pr.C)
        qmin, qmax = 1, max(2, round(0.35 * nC))
        noise_amp = NOISE_FRAC * f_init
        repairs = repair_ops(noise_amp)
        T0 = (W_START * f_init) / math.log(2)

        G = builder.build(pr, sol)
        G.g = global_features(pr, sol, 0.0, 0, f_cur, f_best)
        ep_r, ep_losses, ep_actions = 0.0, [], [0] * cfg.n_actions

        for t in range(cfg.episode_len):
            a = agent.act(G, global_step)
            ep_actions[a] += 1
            di, ri = divmod(a, 3)

            cand = sol.clone()
            pool = DESTROY[di][1](pr, cand, rng.randint(qmin, qmax),
                                  rng)
            repairs[ri](pr, cand, pool, rng)
            f_new, ok, _, _ = eval_solution(pr, cand)

            T = T0 * (1.0 - t / cfg.episode_len)   # linear decay to 0
            accepted = ok and (
                f_new < f_cur - 1e-9
                or rng.random() < math.exp(-(f_new - f_cur)
                                           / max(T, 1e-9)))
            r = compute_reward(f_cur, f_new, f_init, f_best, accepted,
                               cfg) if ok else 0.0
            if ok and f_new < f_best - 1e-9:
                f_best, best_it = f_new, t
            if accepted:
                sol, f_cur = cand, f_new

            G_new = builder.build(pr, sol)
            G_new.g = global_features(pr, sol, (t + 1) / cfg.episode_len,
                                      t - best_it, f_cur, f_best)
            agent.buffer.push(G, a, r, G_new)
            G = G_new
            ep_r += r
            global_step += 1

            if (len(agent.buffer) >= cfg.warmup
                    and global_step % cfg.train_freq == 0):
                ep_losses.append(agent.update())
            if global_step % cfg.target_sync == 0:
                agent.sync_target()

        row = {"episode": ep, "n_cust": nC,
               "best_f": round(f_best, 4), "init_f": round(f_init, 4),
               "mean_r": round(ep_r / cfg.episode_len, 6),
               "eps": round(agent.epsilon(global_step), 3),
               "mean_loss": round(sum(ep_losses) / len(ep_losses), 6)
               if ep_losses else None,
               "actions": ep_actions,
               "elapsed_s": round(time.time() - t0, 1)}
        if log_rows is not None:
            log_rows.append(row)
        print(f"[ep {ep}] n={nC} best={f_best:.2f} "
              f"(init {f_init:.2f}) mean_r={row['mean_r']:.5f} "
              f"eps={row['eps']} loss={row['mean_loss']} "
              f"actions={ep_actions} {row['elapsed_s']}s", flush=True)

        if (ep + 1) % 10 == 0 or ep == cfg.n_episodes - 1:
            agent.save(out_path, norms)
    agent.save(out_path, norms)
    return agent
