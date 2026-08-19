"""Tests for the PPO ALNS environment (DR-ALNS-aligned).

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
from src.gnn_dqn.graph_builder import EDGE_TYPES            # noqa: E402
from src.ppo.env import (ALNSEnv, DOD, VecALNS, W_START,    # noqa: E402
                         sa_accept)


class FixedProvider:
    def __init__(self, pr):
        self.pr = pr

    def sample(self):
        return self.pr


@pytest.fixture(scope="module")
def cfg():
    from src.ppo.config import PPOConfig
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
    """One full episode (search_iterations=100) with random actions."""
    env = _env(pr, builder, cfg, seed=0)
    obs = env.reset()
    rng = random.Random(123)
    rows = []
    for t in range(cfg.search_iterations):
        f_best_prev = env.f_best
        obs, r, done, info = env.step(rng.randrange(9))
        rows.append({"t": t, "reward": r, "done": done,
                     "f_best_prev": f_best_prev,
                     "f_best": info["f_best"], "g": obs.g,
                     "flags": env.flags})
    return env, rows


# ---- reward: DR-ALNS binary, +5 only on a new best-known ----
def test_reward_is_binary_new_best(trace):
    env, rows = trace
    for row in rows:
        improved = row["f_best"] < row["f_best_prev"] - 1e-9
        assert row["reward"] == (5.0 if improved else 0.0)
    assert any(row["reward"] == 5.0 for row in rows)  # search improved


# ---- smoke: full episode sane ----
def test_episode_smoke(trace, cfg):
    env, rows = trace
    bests = [row["f_best"] for row in rows]
    assert all(b1 >= b2 for b1, b2 in zip(bests, bests[1:]))
    for row in rows:
        assert torch.isfinite(row["g"]).all()
        assert math.isfinite(row["reward"])
    # done only at t == search_iterations
    assert not any(row["done"] for row in rows[:-1])
    assert rows[-1]["done"]
    assert env.t == cfg.search_iterations


# ---- SA: 5% start temperature, linear decay to 0 ----
def test_linear_temperature_schedule(pr, builder, cfg):
    env = _env(pr, builder, cfg, seed=1, search_iterations=10)
    env.reset()
    assert W_START == 0.05
    T0 = (0.05 * env.f_init) / math.log(2)
    assert env.T0 == pytest.approx(T0)
    for k in range(10):
        env.t = k
        assert env._temperature() == pytest.approx(T0 * (1 - k / 10))


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


# ---- degree of destruction: fixed 30% ----
def test_degree_of_destruction(pr, builder, cfg):
    env = _env(pr, builder, cfg, seed=4)
    env.reset()
    assert DOD == 0.3
    assert env.q_destroy == max(1, round(0.3 * len(pr.C)))


# ---- g_t: 9 dims, DR-ALNS features tracked by the env ----
def test_g_shape_and_dynamic_features(pr, builder, cfg):
    env = _env(pr, builder, cfg, seed=2, search_iterations=50)
    obs = env.reset()
    assert obs.g.shape == (1, 9)
    assert torch.all(obs.g[0, 2:6] == 0.0)      # first state: zeros
    assert obs.g[0, 8].item() == 0.0            # search_budget
    rng = random.Random(7)
    for _ in range(20):
        obs, _, _, _ = env.step(rng.randrange(9))
        best_improved, accepted, cur_improved = env.flags
        assert obs.g[0, 2].item() == float(best_improved)
        assert obs.g[0, 3].item() == float(accepted)
        assert obs.g[0, 4].item() == float(cur_improved)
        assert obs.g[0, 5].item() == \
            (1.0 if abs(env.f_cur - env.f_best) <= 1e-9 else 0.0)
        assert obs.g[0, 7].item() == pytest.approx(
            min(1.0, env.stagcount / 50))
        assert obs.g[0, 8].item() == pytest.approx(env.t / 50)


# ---- obs padding: uniform edge-type key set across all obs ----
def test_obs_edge_types_padded(pr, builder, cfg):
    """Graphs with differing edge-store key sets are silently
    mis-collated by Batch.from_data_list (edges rewired across graph
    boundaries), which breaks rollout-vs-update ratio consistency —
    every obs must carry the full EDGE_TYPES key set."""
    env = _env(pr, builder, cfg, seed=3, search_iterations=40)
    obs = env.reset()
    rng = random.Random(11)
    for _ in range(40):
        assert set(obs.edge_types) == set(EDGE_TYPES)
        obs, _, _, _ = env.step(rng.randrange(9))


# ---- vectorized wrapper: auto-reset, done flags ----
def test_vec_auto_reset(pr, builder, cfg):
    vec = VecALNS([_env(pr, builder, cfg, seed=s, search_iterations=5)
                   for s in (0, 1)])
    vec.reset()
    for t in range(5):
        obs, rewards, dones, infos = vec.step([t % 9, (t + 3) % 9])
        assert dones == [t == 4, t == 4]
    for o in obs:                       # auto-reset obs: fresh episode
        assert o.g[0, 8].item() == 0.0
        assert torch.all(o.g[0, 2:6] == 0.0)


# ---- determinism: same seed + actions -> same trajectory ----
def test_determinism(pr, builder, cfg):
    traces = []
    for _ in range(2):
        env = _env(pr, builder, cfg, seed=42, search_iterations=30)
        env.reset()
        rng = random.Random(9)
        traces.append([env.step(rng.randrange(9))[1:3]
                       for _ in range(30)])
        traces[-1].append(env.f_best)
    assert traces[0] == traces[1]
