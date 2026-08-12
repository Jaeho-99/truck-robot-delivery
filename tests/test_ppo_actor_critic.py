"""Phase 1 tests for the PPO ActorCritic (spec tests 1, 5, 6).

Run with:  .venv/bin/python -m pytest tests/test_ppo_actor_critic.py -q
"""

import os
import random
import sys

import pytest
import torch
from torch_geometric.data import Batch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance                                    # noqa: E402
from src.gnn_dqn.global_features import global_features     # noqa: E402
from src.gnn_dqn.graph_builder import GraphBuilder          # noqa: E402
from src.gnn_dqn.normalization import compute_norms         # noqa: E402
from src.heuristics import Params                           # noqa: E402
from src.heuristics.alns import congestion_aware_initial    # noqa: E402
from src.heuristics.solution import eval_solution           # noqa: E402
from src.ppo.actor_critic import ActorCritic                # noqa: E402
from src.ppo.config import PPOConfig                        # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return PPOConfig()


@pytest.fixture(scope="module")
def graphs(cfg):
    """Two graphs from instances with different node counts (n=5, 10)."""
    master = instance.build_master(2)
    prs = []
    for n in (5, 10):
        inst, e_c, l_c = instance.build_scaling_instance(master, n)
        prs.append(Params(inst, e_c, l_c, beta_robot=inst["beta_robot"]))
    builder = GraphBuilder(compute_norms(prs), cfg)
    gs = []
    for i, pr in enumerate(prs):
        rng = random.Random(i)
        sol = congestion_aware_initial(pr, rng)
        f, _, _, _ = eval_solution(pr, sol)
        g = builder.build(pr, sol)
        g.g = global_features(pr, sol, 0.1, 3, f, f)
        gs.append(g)
    return gs


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return ActorCritic(cfg)


def _all_zero(params):
    return all(p.grad is None or torch.all(p.grad == 0) for p in params)


def _any_nonzero(params):
    return any(p.grad is not None and torch.any(p.grad != 0)
               for p in params)


# ---- spec test 1: near-uniform policy at initialization ----
def test_initial_entropy(graphs, cfg):
    model = _model(cfg)
    with torch.no_grad():
        _, _, ent, _ = model.get_action_and_value(
            Batch.from_data_list(graphs))
    assert ent.mean().item() >= 2.15       # ln 9 ~ 2.197


# ---- spec test 5: actor/critic gradient isolation ----
def test_gradient_isolation_value_loss(graphs, cfg):
    model = _model(cfg)
    _, _, _, v = model.get_action_and_value(Batch.from_data_list(graphs))
    v.pow(2).mean().backward()
    assert _all_zero(model.actor.parameters())
    assert _any_nonzero(model.critic.parameters())
    assert _any_nonzero(model.encoder.parameters())    # shared encoder


def test_gradient_isolation_pg_loss(graphs, cfg):
    model = _model(cfg)
    _, logp, _, _ = model.get_action_and_value(
        Batch.from_data_list(graphs))
    (-logp.mean()).backward()
    assert _all_zero(model.critic.parameters())
    assert _any_nonzero(model.actor.parameters())
    assert _any_nonzero(model.encoder.parameters())    # shared encoder


# ---- spec test 6: batching instances of different node counts ----
def test_hetero_batch_shapes(graphs, cfg):
    assert (graphs[0]["customer"].x.size(0)
            != graphs[1]["customer"].x.size(0))
    model = _model(cfg)
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        a, logp, ent, v = model.get_action_and_value(batch)
        logits = model.actor(model._state(batch))
    for t in (a, logp, ent, v):
        assert t.shape == (2,)
    assert logits.shape == (2, cfg.n_actions)
    assert a.dtype == torch.long
    assert torch.all((a >= 0) & (a < cfg.n_actions))


# ---- supporting checks (phase 1 plan) ----
def test_sample_and_reeval_paths_agree(graphs, cfg):
    model = _model(cfg)
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        a, logp, ent, v = model.get_action_and_value(batch)
        a2, logp2, ent2, v2 = model.get_action_and_value(batch, action=a)
    assert torch.equal(a, a2)
    assert torch.allclose(logp, logp2)
    assert torch.allclose(ent, ent2)
    assert torch.allclose(v, v2)


def test_orthogonal_init_scales(cfg):
    model = _model(cfg)
    w = model.actor[-1].weight             # std 0.01 -> W W^T = 1e-4 I
    assert torch.allclose(w @ w.t(), 1e-4 * torch.eye(cfg.n_actions),
                          atol=1e-8)
    w = model.critic[-1].weight            # std 1.0 -> W W^T = 1
    assert torch.allclose(w @ w.t(), torch.ones(1, 1), atol=1e-5)
    for head in (model.actor, model.critic):
        w = head[0].weight                 # std sqrt(2) -> W W^T = 2 I
        assert torch.allclose(w @ w.t(), 2.0 * torch.eye(w.size(0)),
                              atol=1e-4)
        assert torch.all(head[0].bias == 0)
        assert torch.all(head[-1].bias == 0)
