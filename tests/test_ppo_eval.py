"""Phase 6 tests: PPO evaluation (checkpoint roundtrip, determinism,
solution validity).

Run with:  .venv/bin/python -m pytest tests/test_ppo_eval.py -q
"""

import os
import sys

import pytest
import torch
from torch_geometric.data import Batch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance                                    # noqa: E402
from src.gnn_dqn.graph_builder import GraphBuilder          # noqa: E402
from src.gnn_dqn.normalization import compute_norms         # noqa: E402
from src.heuristics import Params                           # noqa: E402
from src.heuristics.solution import eval_solution           # noqa: E402
from src.ppo.actor_critic import ActorCritic                # noqa: E402
from src.ppo.config import PPOConfig                        # noqa: E402
from src.ppo.env import ALNSEnv                             # noqa: E402
from src.ppo.eval import (FixedInstanceProvider,            # noqa: E402
                          evaluate_instance, load_model)
from src.ppo.train import save_checkpoint                   # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    master = instance.build_master(2)
    inst, e_c, l_c = instance.build_scaling_instance(master, 5)
    pr = Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])
    cfg = PPOConfig(max_iter=60, seed=0)
    norms = compute_norms([pr])
    torch.manual_seed(0)
    model = ActorCritic(cfg)
    path = str(tmp_path_factory.mktemp("eval") / "ppo.pt")
    save_checkpoint(model, cfg, norms, path)
    return pr, cfg, norms, model, path


# ---- checkpoint roundtrip: identical logits after load ----
def test_load_model_roundtrip_logits(setup):
    pr, cfg, norms, model, path = setup
    model2, cfg2, norms2 = load_model(path)
    assert cfg2 == cfg and norms2 == norms
    assert not model2.training                    # eval mode
    env = ALNSEnv(FixedInstanceProvider(pr),
                  GraphBuilder(norms, cfg), cfg, seed=0)
    batch = Batch.from_data_list([env.reset()])
    with torch.no_grad():
        l1 = model.actor(model._state(batch))
        l2 = model2.actor(model2._state(batch))
    assert torch.equal(l1, l2)


# ---- greedy evaluation: deterministic, valid solution ----
@pytest.fixture(scope="module")
def run(setup):
    pr, cfg, norms, _, path = setup
    model, cfg, norms = load_model(path)
    builder = GraphBuilder(norms, cfg)
    return pr, cfg, builder, model, \
        evaluate_instance(model, cfg, builder, pr, seed=1)


def test_evaluate_deterministic(run):
    pr, cfg, builder, model, (sol, stats) = run
    sol2, stats2 = evaluate_instance(model, cfg, builder, pr, seed=1)
    assert stats2["best_cost"] == stats["best_cost"]
    assert stats2["action_hist"] == stats["action_hist"]
    assert [(i, c) for i, _, c in stats2["best_trace"]] \
        == [(i, c) for i, _, c in stats["best_trace"]]


def test_evaluate_solution_valid(run):
    pr, cfg, _, _, (sol, stats) = run
    cost, feasible, _, _ = eval_solution(pr, sol)
    assert feasible
    assert cost == pytest.approx(stats["best_cost"])
    assert stats["best_cost"] <= stats["init_cost"]
    assert stats["improve_pct"] == pytest.approx(
        100.0 * (stats["init_cost"] - stats["best_cost"])
        / stats["init_cost"])
    assert stats["iters_done"] == cfg.max_iter
    assert sum(stats["action_hist"]) == cfg.max_iter
