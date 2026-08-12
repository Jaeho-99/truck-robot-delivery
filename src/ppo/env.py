"""ALNS as a step-interface environment for PPO rollouts (spec: env).

Mirrors the per-iteration body of solve_alns exactly — destroy ->
repair -> SA accept with geometric cooling over cfg.max_iter — so the
acceptance criterion is identical in training and evaluation by
construction (solve_alns itself stays untouched). The DQN trainer's
linear temperature decay is deliberately NOT reproduced.

g_t redefines two features relative to the DQN (spec):
  progress   = t / max_iter          (deployment horizon)
  stagnation = min(iters_since_improve / 200, 5.0)
The remaining five solution features are reused via global_features.
"""

import math
import random

import torch

from ..gnn_dqn.global_features import global_features
from ..gnn_dqn.graph_builder import EDGE_DIMS, EDGE_TYPES
from ..heuristics.alns import NOISE_FRAC, congestion_aware_initial
from ..heuristics.operators import (DESTROY, repair_greedy,
                                    repair_regret2)
from ..heuristics.solution import eval_solution

W_START = 0.25          # SA start temperature fraction (as solve_alns)

STAGNATION_SCALE = 200.0
STAGNATION_CAP = 5.0


def sa_accept(f_new, f_cur, T, rng):
    """solve_alns acceptance: improving always, worsening via SA.

    Improving candidates short-circuit before the rng draw, matching
    solve_alns's branch order.
    """
    return (f_new < f_cur - 1e-9
            or rng.random() < math.exp(-(f_new - f_cur)
                                       / max(T, 1e-9)))


def pad_edge_types(data):
    """Give every EDGE_TYPES triplet a (possibly empty) store.

    Batch.from_data_list mis-collates HeteroData lists whose edge-store
    key sets differ: edges of a graph missing elsewhere get node
    offsets from the wrong graph, silently rewiring them across graph
    boundaries. That makes batched (update) outputs diverge from
    single-graph (rollout) outputs and corrupts the PPO ratio. Padding
    to one shared key set makes collation exact.
    """
    for et in EDGE_TYPES:
        if et not in data.edge_types:
            data[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[et].edge_attr = torch.zeros((0, EDGE_DIMS[et[1]]))
    return data


def ppo_global_features(pr, sol, t, since_improve, f_cur, f_best,
                        max_iter):
    """g_t with the spec's progress/stagnation definitions.

    Reuses global_features for the five solution features (indices
    0-3, 6) and overwrites only the stagnation entry (index 5); the
    progress entry (index 4) is already just a [0,1] clip of t/max_iter.
    """
    g = global_features(pr, sol, t / max_iter, 0, f_cur, f_best)
    g[0, 5] = min(since_improve / STAGNATION_SCALE, STAGNATION_CAP)
    return g


class ALNSEnv:
    """Single-instance ALNS with a gym-style step interface.

    reset() -> obs; step(a) -> (obs, reward, done, info). obs is a
    HeteroData with g_t attached as .g. done=True only at
    t == cfg.max_iter; rollout-boundary truncation is the caller's
    concern (spec). reward = max(0, f_best_prev - f_best) / f_init.
    """

    def __init__(self, provider, builder, cfg, seed):
        self.provider = provider
        self.builder = builder
        self.cfg = cfg
        self.rng = random.Random(seed)

    def reset(self):
        self.pr = self.provider.sample()
        self.sol = congestion_aware_initial(self.pr, self.rng)
        f, _, _, _ = eval_solution(self.pr, self.sol)
        self.f_init = self.f_cur = self.f_best = f
        self.t = 0
        self.since_improve = 0
        nC = len(self.pr.C)
        self.qmin, self.qmax = 1, max(2, round(0.35 * nC))
        noise_amp = NOISE_FRAC * self.f_init
        self.repairs = [
            lambda p, s, pool, r: repair_greedy(p, s, pool, r, 0.0),
            lambda p, s, pool, r: repair_greedy(p, s, pool, r,
                                                noise_amp),
            repair_regret2]
        self.T = (W_START * self.f_init) / math.log(2)
        self.cooling = (1e-3) ** (1.0 / max(1, self.cfg.max_iter))
        return self._obs()

    def _obs(self):
        data = self.builder.build(self.pr, self.sol)
        data.g = ppo_global_features(self.pr, self.sol, self.t,
                                     self.since_improve, self.f_cur,
                                     self.f_best, self.cfg.max_iter)
        return pad_edge_types(data)

    def step(self, a):
        di, ri = divmod(int(a), 3)
        cand = self.sol.clone()
        pool = DESTROY[di][1](self.pr, cand,
                              self.rng.randint(self.qmin, self.qmax),
                              self.rng)
        self.repairs[ri](self.pr, cand, pool, self.rng)
        f_new, ok, _, _ = eval_solution(self.pr, cand)

        f_best_prev = self.f_best
        improved = False
        if ok:      # infeasible candidates are discarded (solve_alns)
            if f_new < self.f_best - 1e-9:
                self.f_best = f_new
                improved = True
            if sa_accept(f_new, self.f_cur, self.T, self.rng):
                self.sol, self.f_cur = cand, f_new
        reward = max(0.0, f_best_prev - self.f_best) / self.f_init

        self.since_improve = 0 if improved else self.since_improve + 1
        self.T *= self.cooling      # cools on infeasible too
        self.t += 1
        done = self.t >= self.cfg.max_iter
        return (self._obs(), reward, done,
                {"f_best": self.f_best, "f_cur": self.f_cur})


class VecALNS:
    """Synchronous vector of ALNSEnv with auto-reset on done.

    A done env's returned obs is the reset obs of its next episode, so
    GAE must key off done=True (never bootstrap through it) — matching
    the spec's truncation-vs-termination rules.
    """

    def __init__(self, envs):
        self.envs = envs

    def reset(self):
        return [e.reset() for e in self.envs]

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, d, info = env.step(a)
            if d:
                o = env.reset()
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return obs, rewards, dones, infos
