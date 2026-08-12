"""Phase 2 tests for the PPO ALNS environment (spec: env/rollout).

Run with:  .venv/bin/python -m pytest tests/test_ppo_env.py -q
"""

import dataclasses
import math
import os
import random
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance                                    # noqa: E402
from src.gnn_dqn.graph_builder import GraphBuilder          # noqa: E402
from src.gnn_dqn.normalization import compute_norms         # noqa: E402
from src.heuristics import Params                           # noqa: E402
from src.ppo.config import PPOConfig                        # noqa: E402
from src.ppo.env import (ALNSEnv, VecALNS, W_START,         # noqa: E402
                         sa_accept)


class FixedProvider:
    def __init__(self, pr):
        self.pr = pr

    def sample(self):
        return self.pr


@pytest.fixture(scope="module")
def cfg():
    return PPOConfig()


@pytest.fixture(scope="module")
def pr():
    master = instance.build_master(2)
    inst, e_c, l_c = instance.build_scaling_instance(master, 5)
    return Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])


@pytest.fixture(scope="module")
def builder(pr, cfg):
    return GraphBuilder(compute_norms([pr]), cfg)


def _env(pr, builder, cfg, seed=0, **overrides):
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return ALNSEnv(FixedProvider(pr), builder, cfg, seed)


@pytest.fixture(scope="module")
def trace(pr, builder, cfg):
    """One full episode (max_iter=500) with random actions."""
    env = _env(pr, builder, cfg, seed=0)
    obs = env.reset()
    rng = random.Random(123)
    rows = []
    for t in range(cfg.max_iter):
        f_best_prev = env.f_best
        obs, r, done, info = env.step(rng.randrange(9))
        rows.append({"t": t, "reward": r, "done": done,
                     "f_best_prev": f_best_prev,
                     "f_best": info["f_best"], "g": obs.g})
    return env, rows


# ---- reward: best-based, no penalty (spec) ----
def test_reward_is_best_based(trace):
    env, rows = trace
    for row in rows:
        expected = max(0.0, row["f_best_prev"] - row["f_best"]) \
            / env.f_init
        assert row["reward"] == pytest.approx(expected)
        assert row["reward"] >= 0.0
        if row["f_best"] == row["f_best_prev"]:    # incl. worsening
            assert row["reward"] == 0.0
    assert any(row["reward"] > 0 for row in rows)  # search did improve


# ---- smoke: full episode sane ----
def test_episode_smoke(trace, cfg):
    env, rows = trace
    bests = [row["f_best"] for row in rows]
    assert all(b1 >= b2 for b1, b2 in zip(bests, bests[1:]))
    for row in rows:
        assert torch.isfinite(row["g"]).all()
        assert math.isfinite(row["reward"])
    # done only at t == max_iter
    assert not any(row["done"] for row in rows[:-1])
    assert rows[-1]["done"]
    assert env.t == cfg.max_iter


# ---- acceptance: geometric cooling identical to solve_alns ----
def test_geometric_cooling_schedule(pr, builder, cfg):
    env = _env(pr, builder, cfg, seed=1, max_iter=10)
    env.reset()
    T0 = (W_START * env.f_init) / math.log(2)
    assert env.T == pytest.approx(T0)
    cooling = (1e-3) ** (1.0 / 10)
    assert env.cooling == pytest.approx(cooling)
    for k in range(1, 11):
        env.step(0)
        assert env.T == pytest.approx(T0 * cooling ** k)


def test_sa_accept_criterion():
    class NoDraw:                       # improving: no rng draw
        def random(self):
            raise AssertionError("rng consumed on improving move")

    assert sa_accept(90.0, 100.0, 5.0, NoDraw())
    # worsening at T -> 0: exp underflows to 0, never accepted
    rng = random.Random(0)
    assert not any(sa_accept(101.0, 100.0, 1e-12, rng)
                   for _ in range(100))
    # worsening at huge T: exp(-eps) rounds to 1.0 > random() in [0,1)
    assert sa_accept(101.0, 100.0, 1e18, random.Random(0))


# ---- g_t: spec progress/stagnation definitions ----
def test_g_progress_and_stagnation(pr, builder, cfg):
    env = _env(pr, builder, cfg, seed=2, max_iter=50)
    obs = env.reset()
    assert obs.g.shape == (1, 7)
    assert obs.g[0, 4].item() == 0.0
    rng = random.Random(7)
    for _ in range(20):
        obs, _, _, _ = env.step(rng.randrange(9))
        assert obs.g[0, 4].item() == pytest.approx(env.t / 50)
        assert obs.g[0, 5].item() == pytest.approx(
            min(env.since_improve / 200.0, 5.0))
    env.since_improve = 1500            # cap at 5.0, not 1.0 (spec)
    assert env._obs().g[0, 5].item() == 5.0


# ---- vectorized wrapper: auto-reset, done flags ----
def test_vec_auto_reset(pr, builder, cfg):
    vec = VecALNS([_env(pr, builder, cfg, seed=s, max_iter=5)
                   for s in (0, 1)])
    vec.reset()
    for t in range(5):
        obs, rewards, dones, infos = vec.step([t % 9, (t + 3) % 9])
        assert dones == [t == 4, t == 4]
    for o in obs:                       # auto-reset obs: fresh episode
        assert o.g[0, 4].item() == 0.0


# ---- determinism: same seed + actions -> same trajectory ----
def test_determinism(pr, builder, cfg):
    traces = []
    for _ in range(2):
        env = _env(pr, builder, cfg, seed=42, max_iter=30)
        env.reset()
        rng = random.Random(9)
        traces.append([env.step(rng.randrange(9))[1:3]
                       for _ in range(30)])
        traces[-1].append(env.f_best)
    assert traces[0] == traces[1]
