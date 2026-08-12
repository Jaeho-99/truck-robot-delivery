"""Phase 3 tests: RolloutBuffer + GAE (spec tests 3, 4).

Run with:  .venv/bin/python -m pytest tests/test_ppo_gae.py -q
"""

import os
import random
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance                                    # noqa: E402
from src.gnn_dqn.global_features import global_features     # noqa: E402
from src.gnn_dqn.graph_builder import GraphBuilder          # noqa: E402
from src.gnn_dqn.normalization import compute_norms         # noqa: E402
from src.heuristics import Params                           # noqa: E402
from src.heuristics.alns import congestion_aware_initial    # noqa: E402
from src.heuristics.solution import eval_solution           # noqa: E402
from src.ppo.buffer import RolloutBuffer                    # noqa: E402
from src.ppo.config import PPOConfig                        # noqa: E402
from src.ppo.gae import compute_gae                         # noqa: E402


# ---- spec test 3: hand-computed T=3 case, done in the middle ----
def test_gae_hand_computed():
    # column 0 has done at t=1; column 1 never terminates
    rewards = torch.tensor([[1.0, 0.5], [2.0, 1.0], [3.0, 2.0]])
    values = torch.tensor([[0.5, 1.0], [1.0, 0.5], [1.5, 1.0]])
    dones = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    last_value = torch.tensor([2.0, 1.5])
    adv, ret = compute_gae(rewards, values, dones, last_value,
                           gamma=0.9, gae_lambda=0.8)
    # col 0: A2 = 3 + .9*2 - 1.5 = 3.3
    #        A1 = 2 - 1 = 1.0                    (done: no bootstrap)
    #        A0 = (1 + .9*1 - .5) + .72*1 = 2.12
    # col 1: A2 = 2 + .9*1.5 - 1 = 2.35
    #        A1 = (1 + .9*1 - .5) + .72*2.35 = 3.092
    #        A0 = (.5 + .9*.5 - 1) + .72*3.092 = 2.17624
    expected = torch.tensor([[2.12, 2.17624], [1.0, 3.092],
                             [3.3, 2.35]])
    assert torch.allclose(adv, expected, atol=1e-6)
    assert torch.allclose(ret, expected + values, atol=1e-6)


# ---- spec test 4: rollout truncation vs episode termination ----
def test_truncation_bootstraps_termination_does_not():
    rewards = torch.zeros(3, 1)
    values = torch.zeros(3, 1)
    last_value = torch.tensor([10.0])
    # truncation (done=0 at the boundary): last_value flows back
    adv_tr, _ = compute_gae(rewards, values, torch.zeros(3, 1),
                            last_value, gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(adv_tr, torch.full((3, 1), 10.0))
    # termination (done=1 at the last step): last_value is ignored
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    for lv in (10.0, 999.0):
        adv_te, _ = compute_gae(rewards, values, dones,
                                torch.tensor([lv]), 1.0, 1.0)
        assert torch.all(adv_te == 0.0)


def test_gae_is_no_grad():
    values = torch.zeros(3, 1, requires_grad=True)
    adv, ret = compute_gae(torch.ones(3, 1), values,
                           torch.zeros(3, 1), torch.zeros(1),
                           0.99, 0.95)
    assert not adv.requires_grad and not ret.requires_grad


# ---- buffer: shapes, alignment, minibatch coverage ----
@pytest.fixture(scope="module")
def graph():
    cfg = PPOConfig()
    master = instance.build_master(2)
    inst, e_c, l_c = instance.build_scaling_instance(master, 5)
    pr = Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])
    sol = congestion_aware_initial(pr, random.Random(0))
    f, _, _, _ = eval_solution(pr, sol)
    g = GraphBuilder(compute_norms([pr]), cfg).build(pr, sol)
    g.g = global_features(pr, sol, 0.1, 3, f, f)
    return g


def _filled_buffer(graph, T=4, N=2):
    """Buffer where action == g[0,0] == flat index (alignment probe)."""
    buf = RolloutBuffer(T, N)
    for t in range(T):
        obs = []
        for i in range(N):
            o = graph.clone()
            o.g = torch.full((1, 7), float(t * N + i))
            obs.append(o)
        base = t * N
        buf.add(obs, actions=[base, base + 1],
                log_probs=[-0.1 * base, -0.1 * (base + 1)],
                values=[0.5 * base, 0.5 * (base + 1)],
                rewards=[float(t), float(t)],
                dones=[0.0, 1.0 if t == 2 else 0.0])
    return buf


def test_buffer_stacking_and_gae_consistency(graph):
    buf = _filled_buffer(graph)
    buf.compute_returns(torch.tensor([1.0, 2.0]), 0.99, 0.95)
    assert buf.advantages.shape == (4, 2)
    assert buf.returns.shape == (4, 2)
    adv, ret = compute_gae(torch.stack(buf.rewards),
                           torch.stack(buf.values),
                           torch.stack(buf.dones),
                           torch.tensor([1.0, 2.0]), 0.99, 0.95)
    assert torch.equal(buf.advantages, adv)
    assert torch.equal(buf.returns, ret)


def test_buffer_minibatch_alignment_and_coverage(graph):
    buf = _filled_buffer(graph)
    buf.compute_returns(torch.zeros(2), 0.99, 0.95)
    gen = torch.Generator().manual_seed(0)
    seen = []
    for batch, actions, logps, advs, rets, vals in \
            buf.minibatches(n_minibatch=4, generator=gen):
        assert batch.num_graphs == 2
        # graph <-> tensor alignment: g[0,0] encodes the flat index
        assert torch.equal(batch.g.view(-1, 7)[:, 0].long(), actions)
        assert torch.allclose(logps, -0.1 * actions.float())
        assert torch.allclose(vals, 0.5 * actions.float())
        assert advs.shape == rets.shape == (2,)
        seen.extend(actions.tolist())
    assert sorted(seen) == list(range(8))   # exhaustive, disjoint


def test_buffer_minibatch_determinism(graph):
    buf = _filled_buffer(graph)
    buf.compute_returns(torch.zeros(2), 0.99, 0.95)
    orders = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(7)
        orders.append([mb[1].tolist()
                       for mb in buf.minibatches(4, generator=gen)])
    assert orders[0] == orders[1]


def test_buffer_reset_and_capacity(graph):
    buf = _filled_buffer(graph, T=2, N=2)
    assert len(buf) == 2
    with pytest.raises(AssertionError):
        buf.add([graph, graph], [0, 0], [0.0, 0.0], [0.0, 0.0],
                [0.0, 0.0], [0.0, 0.0])
    buf.reset()
    assert len(buf) == 0 and buf.obs == [] and buf.advantages is None
