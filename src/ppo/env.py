"""ALNS as a step-interface environment for PPO rollouts.

DR-ALNS-aligned: episodes of cfg.search_iterations (100, fixed),
linear SA temperature decay from T0 = W_START * f_init / ln 2 to 0,
fixed degree of destruction q = round(DOD * n) — identical to
solve_alns and the DQN trainer (shared constants from
src.heuristics.alns). g_t comes from the shared
gnn_dqn.global_features (single source for both RL paths). reward is
the DR-ALNS reward function: +5 on a new best-known solution, else 0.
"""

import math
import random

from ..gnn_dqn.global_features import global_features
from ..gnn_dqn.graph_builder import pad_edge_types      # noqa: F401
# (re-export: padding now happens inside GraphBuilder.build, shared by
# the DQN replay path and this env)
from ..heuristics.alns import (DOD, NOISE_FRAC, W_START,
                               congestion_aware_initial)
from ..heuristics.operators import (DESTROY, repair_greedy,
                                    repair_regret2)
from ..heuristics.solution import eval_solution


def sa_accept(f_new, f_cur, T, rng):
    """solve_alns acceptance: improving always, worsening via SA.

    Improving candidates short-circuit before the rng draw, matching
    solve_alns's branch order.
    """
    return (f_new < f_cur - 1e-9
            or rng.random() < math.exp(-(f_new - f_cur)
                                       / max(T, 1e-9)))


class ALNSEnv:
    """Single-instance ALNS with a gym-style step interface.

    reset() -> obs; step(a) -> (obs, reward, done, info). obs is a
    HeteroData with g_t attached as .g. done=True only at
    t == cfg.search_iterations; rollout-boundary truncation is the
    caller's concern.
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
        self.stagcount = 0
        # previous-iteration outcome flags (DR-ALNS obs); reset -> 0
        self.flags = (False, False, False)
        nC = len(self.pr.C)
        # degree of destruction: fixed 30% of customers (DR-ALNS)
        self.q_destroy = max(1, round(DOD * nC))
        noise_amp = NOISE_FRAC * self.f_init
        self.repairs = [
            lambda p, s, pool, r: repair_greedy(p, s, pool, r, 0.0),
            lambda p, s, pool, r: repair_greedy(p, s, pool, r,
                                                noise_amp),
            repair_regret2]
        self.T0 = (W_START * self.f_init) / math.log(2)
        return self._obs()

    def _temperature(self):
        """Linear decay T0 -> 0 over the episode (as solve_alns)."""
        return self.T0 * (1.0 - self.t / self.cfg.search_iterations)

    def _obs(self):
        data = self.builder.build(self.pr, self.sol)
        best_improved, accepted, cur_improved = self.flags
        data.g = global_features(
            self.pr, self.sol, self.t, self.cfg.search_iterations,
            self.stagcount, self.f_cur, self.f_best,
            best_improved=best_improved, current_accepted=accepted,
            current_improved=cur_improved)
        return data          # GraphBuilder.build already pads

    def step(self, a):
        di, ri = divmod(int(a), 3)
        cand = self.sol.clone()
        pool = DESTROY[di][1](self.pr, cand, self.q_destroy, self.rng)
        self.repairs[ri](self.pr, cand, pool, self.rng)
        f_new, ok, _, _ = eval_solution(self.pr, cand)

        T = self._temperature()
        improved_best = False
        accepted = False
        improved_current = False
        if ok:      # infeasible candidates are discarded (solve_alns)
            if f_new < self.f_best - 1e-9:
                self.f_best = f_new
                improved_best = True
            if sa_accept(f_new, self.f_cur, T, self.rng):
                accepted = True
                improved_current = f_new < self.f_cur - 1e-9
                self.sol, self.f_cur = cand, f_new
        # DR-ALNS reward: +5 on a new best-known solution, else 0
        reward = 5.0 if improved_best else 0.0

        self.flags = (improved_best, accepted, improved_current)
        self.stagcount = 0 if improved_best else self.stagcount + 1
        self.t += 1
        done = self.t >= self.cfg.search_iterations
        return (self._obs(), reward, done,
                {"f_best": self.f_best, "f_cur": self.f_cur,
                 "f_init": self.f_init,
                 "instance_id": self.pr.inst.get("instance_id")})


class VecALNS:
    """Synchronous vector of ALNSEnv with auto-reset on done.

    A done env's returned obs is the reset obs of its next episode, so
    GAE must key off done=True (never bootstrap through it).
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
