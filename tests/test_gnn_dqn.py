"""Smoke/correctness tests for the GNN+DQN selector (spec section 8).

Run with:  .venv/bin/python -m pytest tests/test_gnn_dqn.py -q
"""

import os
import random
import sys
import time

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance                                    # noqa: E402
from src.gnn_dqn.config import Config                       # noqa: E402
from src.gnn_dqn.dqn_agent import DQNAgent                  # noqa: E402
from src.gnn_dqn.global_features import global_features     # noqa: E402
from src.gnn_dqn.graph_builder import GraphBuilder          # noqa: E402
from src.gnn_dqn.normalization import compute_norms         # noqa: E402
from src.gnn_dqn.reward import compute_reward               # noqa: E402
from src.heuristics import Params, solve_alns               # noqa: E402
from src.heuristics.alns import congestion_aware_initial    # noqa: E402
from src.heuristics.solution import eval_solution           # noqa: E402


@pytest.fixture(scope="module")
def pr():
    inst = instance.build_grid_instance(seed=1)
    e_c, l_c = instance.reachability_tw(inst, seed=1)
    return Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])


@pytest.fixture(scope="module")
def norms(pr):
    return compute_norms([pr])


@pytest.fixture(scope="module")
def cfg():
    return Config()


def _graph(pr, norms, cfg, seed=0):
    rng = random.Random(seed)
    sol = congestion_aware_initial(pr, rng)
    f, _, _, _ = eval_solution(pr, sol)
    g = GraphBuilder(norms, cfg).build(pr, sol)
    g.g = global_features(pr, sol, 10, 100, 3, f, f)
    return g, sol


def test_graph_structure(pr, norms, cfg):
    g, sol = _graph(pr, norms, cfg)
    assert g["customer"].x.shape == (len(pr.C), 5)
    assert g["parking"].x.shape == (len(pr.P), 6)
    assert g["depot"].x.shape == (2, 3)
    # coordinates normalized to [0,1]
    for t in ("customer", "parking", "depot"):
        assert g[t].x[:, :2].min() >= 0.0
        assert g[t].x[:, :2].max() <= 1.0
    # each truck arc appears with its reverse: total even per family
    n_truck = sum(g[et].edge_index.size(1) for et in g.edge_types
                  if et[1] == "truck_arc")
    assert n_truck > 0 and n_truck % 2 == 0
    # proximity: k edges per customer, both directions
    n_prox = sum(g[et].edge_index.size(1) for et in g.edge_types
                 if et[1] == "proximity")
    assert n_prox == 2 * cfg.knn_k * len(pr.C)
    # served_by flag consistent with the solution
    from src.heuristics.qlearning import robot_served_customers
    served = robot_served_customers(sol)
    for i, c in enumerate(pr.C):
        assert g["customer"].x[i, 4] == (1.0 if c in served else 0.0)
    assert g.g.shape == (1, 9)


def test_state_dim_and_batching(pr, norms, cfg):
    from torch_geometric.data import Batch
    from src.gnn_dqn.encoder import QNet
    net = QNet(cfg)
    g1, _ = _graph(pr, norms, cfg, seed=0)
    g2, _ = _graph(pr, norms, cfg, seed=1)
    q = net(Batch.from_data_list([g1, g2]))
    assert q.shape == (2, 9)
    assert torch.isfinite(q).all()


def test_dueling_mean_equals_value(pr, norms, cfg):
    """Q = V + (A - mean A) implies mean_a Q(s,a) == V(s)."""
    from torch_geometric.data import Batch
    from src.gnn_dqn.encoder import QNet
    net = QNet(cfg)
    g1, _ = _graph(pr, norms, cfg, seed=0)
    g2, _ = _graph(pr, norms, cfg, seed=1)
    batch = Batch.from_data_list([g1, g2])
    with torch.no_grad():
        q = net(batch)
        v, a = net.value_advantage(batch)
    assert q.shape == (2, 9) and torch.isfinite(q).all()
    assert torch.allclose(q.mean(dim=1), v.squeeze(1), atol=1e-5)
    assert torch.allclose(a.mean(dim=1), torch.zeros(2), atol=1e-5)


def test_dueling_update_reaches_all_streams(pr, norms, cfg):
    """One update must move trunk, V and A stream parameters."""
    agent = DQNAgent(cfg, total_steps=1000)
    g, _ = _graph(pr, norms, cfg)
    for i in range(70):
        agent.buffer.push(g, i % 9, 0.1 * (i % 3), g)
    head = agent.online.head
    before = {name: p.clone() for name, p in head.named_parameters()}
    agent.update()
    for stream in ("trunk", "V", "A"):
        assert any((before[n] != p).any()
                   for n, p in head.named_parameters()
                   if n.startswith(stream)), f"{stream} unchanged"


def test_dqn_update_changes_params(pr, norms, cfg):
    agent = DQNAgent(cfg, total_steps=1000)
    g, _ = _graph(pr, norms, cfg)
    for i in range(70):
        agent.buffer.push(g, i % 9, 0.1, g)
    before = [p.clone() for p in agent.online.head.parameters()]
    loss = agent.update()
    assert torch.isfinite(torch.tensor(loss))
    changed = any((a != b).any()
                  for a, b in zip(before,
                                  agent.online.head.parameters()))
    assert changed


def test_mixed_structure_batch_consistency(pr, norms, cfg):
    """Graphs with different edge-type sets (robot trips vs truck-only)
    batch to the SAME Q-values as individual forwards. Without the
    edge-type padding in GraphBuilder.build, Batch.from_data_list
    rewires edges across graph boundaries (measured Q corruption
    ~0.06) — this pins the replay-batch fix."""
    from torch_geometric.data import Batch
    from src.gnn_dqn.encoder import QNet
    from src.gnn_dqn.graph_builder import EDGE_TYPES
    from src.heuristics.alns import truck_only_initial

    torch.manual_seed(0)
    net = QNet(cfg)
    net.eval()
    graphs = []
    for init in (congestion_aware_initial, truck_only_initial):
        sol = init(pr, random.Random(0))
        f, _, _, _ = eval_solution(pr, sol)
        g = GraphBuilder(norms, cfg).build(pr, sol)
        g.g = global_features(pr, sol, 5, 100, 0, f, f)
        assert set(g.edge_types) == set(EDGE_TYPES)   # padded
        graphs.append(g)
    with torch.no_grad():
        q_ind = torch.cat([net(Batch.from_data_list([g]))
                           for g in graphs])
        q_bat = net(Batch.from_data_list(graphs))
    assert torch.allclose(q_ind, q_bat, atol=1e-5)


def test_reward_modes():
    cfg = Config(reward_mode="R1")
    # accepted improvement
    assert compute_reward(100, 90, 200, 95, True, cfg) == \
        pytest.approx(0.05)
    # accepted-worse keeps its negative R1
    assert compute_reward(100, 110, 200, 95, True, cfg) == \
        pytest.approx(-0.05)
    # rejected -> 0
    assert compute_reward(100, 110, 200, 95, False, cfg) == 0.0
    cfg2 = Config(reward_mode="R2")
    assert compute_reward(100, 90, 200, 95, True, cfg2) == \
        pytest.approx(0.05 + cfg2.kappa)
    cfg3 = Config(reward_mode="binary")
    assert compute_reward(100, 90, 200, 95, True, cfg3) == 5.0
    assert compute_reward(100, 96, 200, 95, True, cfg3) == 0.0


def test_smoke_train(pr, norms, tmp_path):
    from src.gnn_dqn.trainer import train

    class FixedProvider:
        def sample(self):
            return pr

    cfg = Config(total_steps=50, search_iterations=25, warmup=10,
                 buffer_capacity=200, target_sync=20)
    t0 = time.time()
    agent = train(cfg, FixedProvider(), str(tmp_path / "m.pt"), norms,
                  log_rows=[])
    assert time.time() - t0 < 300
    assert os.path.exists(tmp_path / "m.pt")
    assert len(agent.buffer) == 50


def test_gnn_selector_end_to_end(pr, norms, tmp_path):
    """Trained checkpoint drives solve_alns(selector='gnn_dqn')."""
    from src.gnn_dqn.trainer import train

    class FixedProvider:
        def sample(self):
            return pr

    cfg = Config(total_steps=15, search_iterations=15, warmup=5,
                 buffer_capacity=100)
    path = str(tmp_path / "m.pt")
    train(cfg, FixedProvider(), path, norms, log_rows=[])
    best, cost, stats = solve_alns(pr, iters=30, seed=0,
                                   selector="gnn_dqn",
                                   q_params={"model_path": path})
    _, feas, _, _ = eval_solution(pr, best)
    assert feas and cost > 0
