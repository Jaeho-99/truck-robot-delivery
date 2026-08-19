"""DR-ALNS alignment checks: shared 9-dim g_t, 5% SA rule, dod 30%.

Run with:  .venv/bin/python -m pytest tests/test_dralns_alignment.py -q
"""

import inspect
import math
import os
import random
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.gnn_dqn import global_features as gf_module        # noqa: E402
from src.gnn_dqn import trainer as dqn_trainer              # noqa: E402
from src.gnn_dqn.dataset import DirectoryInstanceProvider   # noqa: E402
from src.gnn_dqn.global_features import (G_DIM,             # noqa: E402
                                         global_features)
from src.gnn_dqn.graph_builder import GraphBuilder          # noqa: E402
from src.gnn_dqn.normalization import compute_norms         # noqa: E402
from src.heuristics import alns                             # noqa: E402
from src.heuristics.alns import (congestion_aware_initial,  # noqa: E402
                                 solve_alns)
from src.heuristics.solution import eval_solution           # noqa: E402
from src.ppo import env as ppo_env                          # noqa: E402
from src.ppo.config import PPOConfig                        # noqa: E402
from src.ppo.env import ALNSEnv                             # noqa: E402


@pytest.fixture(scope="module")
def provider():
    return DirectoryInstanceProvider(size=20, root=os.path.join(
        REPO_ROOT, "data"), seed=0)


@pytest.fixture(scope="module")
def pr(provider):
    return provider._params(provider.train[0])


@pytest.fixture(scope="module")
def sol(pr):
    return congestion_aware_initial(pr, random.Random(0))


# ---- (a) binary-feature scenarios (indices 2-5) ----
def test_g_flag_scenarios(pr, sol):
    f, _, _, _ = eval_solution(pr, sol)

    def flags4(**kw):
        g = global_features(pr, sol, kw.pop("it", 10), 100,
                            kw.pop("stagcount", 0),
                            kw.pop("current_cost", f),
                            kw.pop("best_cost", f), **kw)
        return g[0, 2:6].tolist()

    # right after a new best: current == best, all outcome flags set
    assert flags4(best_improved=True, current_accepted=True,
                  current_improved=True) == [1.0, 1.0, 1.0, 1.0]
    # right after accepting a worsening move: current > best
    assert flags4(current_accepted=True,
                  current_cost=f * 1.1) == [0.0, 1.0, 0.0, 0.0]
    # right after a reject: flags 0; is_current_best from costs
    assert flags4(current_cost=f * 1.1) == [0.0, 0.0, 0.0, 0.0]
    assert flags4() == [0.0, 0.0, 0.0, 1.0]     # current still == best
    # first state of an episode (it == 0): all zeros, like the
    # DR-ALNS environment reset()
    assert flags4(it=0) == [0.0, 0.0, 0.0, 0.0]


# ---- (b) exactly 9 dims, all components within [0, 1.5] ----
def test_g_shape_and_ranges(pr, sol):
    assert G_DIM == 9
    f, _, _, _ = eval_solution(pr, sol)
    for kw in ({}, {"it": 0}, {"it": 100, "stagcount": 500},
               {"current_cost": f * 5.0, "best_improved": True,
                "current_accepted": True, "current_improved": True}):
        g = global_features(pr, sol, kw.pop("it", 10), 100,
                            kw.pop("stagcount", 3),
                            kw.pop("current_cost", f), f, **kw)
        assert g.shape == (1, 9)
        assert torch.all(g >= 0.0) and torch.all(g <= 1.5)


# ---- (c) DQN path and PPO path share ONE g_t function ----
def test_g_single_source(pr, provider):
    from src.gnn_dqn import selector_gnn
    assert ppo_env.global_features is global_features
    assert selector_gnn.global_features is global_features
    assert dqn_trainer.global_features is global_features
    # env-produced g equals a direct call with the same state
    cfg = PPOConfig()
    builder = GraphBuilder(compute_norms([pr]), cfg)
    env = ALNSEnv(provider, builder, cfg, seed=0)
    obs = env.reset()
    direct = global_features(env.pr, env.sol, env.t,
                             cfg.search_iterations, env.stagcount,
                             env.f_cur, env.f_best, *env.flags)
    assert torch.equal(obs.g, direct)
    obs, _, _, _ = env.step(4)
    direct = global_features(env.pr, env.sol, env.t,
                             cfg.search_iterations, env.stagcount,
                             env.f_cur, env.f_best, *env.flags)
    assert torch.equal(obs.g, direct)


# ---- (d) 5% temperature rule + dod 30% on every path ----
def test_temperature_and_dod_all_paths():
    assert alns.W_START == 0.05
    assert alns.DOD == 0.3
    # solve_alns (vanilla/QL/DQN inference) defaults to the shared rule
    assert (inspect.signature(solve_alns).parameters["w_start"].default
            == alns.W_START)
    src = inspect.getsource(solve_alns)
    assert "q_destroy = max(1, round(DOD * nC))" in src
    assert "T = T0 * (1.0 - (it - 1) / iters)" in src
    # DQN trainer and PPO env import the same constants
    assert dqn_trainer.W_START == 0.05 and dqn_trainer.DOD == 0.3
    assert ppo_env.W_START == 0.05 and ppo_env.DOD == 0.3
    tsrc = inspect.getsource(dqn_trainer.train)
    assert "q_destroy = max(1, round(DOD * nC))" in tsrc
    assert "T0 * (1.0 - it / cfg.search_iterations)" in tsrc


# ---- (e) PPO on the Ulsan directory data ----
def test_ppo_with_directory_provider(provider):
    assert len(provider.train) == 250 and len(provider.test) == 50
    cfg = PPOConfig()
    pr0 = provider._params(provider.train[0])
    builder = GraphBuilder(compute_norms([pr0]), cfg)
    env = ALNSEnv(provider, builder, cfg, seed=0)
    obs = env.reset()
    assert obs.g.shape == (1, 9)
    for a in (0, 4, 8):
        obs, r, done, info = env.step(a)
        assert math.isfinite(r) and not done
        assert torch.isfinite(obs.g).all()
