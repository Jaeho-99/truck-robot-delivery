"""Phase 4 tests: PPO update step (spec test 2 + update rules).

Run with:  .venv/bin/python -m pytest tests/test_ppo_update.py -q
"""

import dataclasses
import math
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
from src.ppo.actor_critic import ActorCritic                # noqa: E402
from src.ppo.buffer import RolloutBuffer                    # noqa: E402
from src.ppo.config import PPOConfig                        # noqa: E402
from src.ppo.env import ALNSEnv                             # noqa: E402
from src.ppo.update import (c2_schedule, make_optimizer,    # noqa: E402
                            ppo_update, value_loss)


class FixedProvider:
    def __init__(self, pr):
        self.pr = pr

    def sample(self):
        return self.pr


T_ROLLOUT = 32


@pytest.fixture(scope="module")
def cfg():
    return PPOConfig(t_rollout=T_ROLLOUT, n_envs=1, n_minibatch=8,
                     k_epochs=4)


@pytest.fixture(scope="module")
def env(cfg):
    master = instance.build_master(2)
    inst, e_c, l_c = instance.build_scaling_instance(master, 5)
    pr = Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])
    builder = GraphBuilder(compute_norms([pr]), cfg)
    return ALNSEnv(FixedProvider(pr), builder, cfg, seed=0)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return ActorCritic(cfg)


def _rollout(model, env, cfg):
    """Collect one on-policy rollout (N=1) with GAE computed."""
    buf = RolloutBuffer(cfg.t_rollout, 1)
    obs = env.reset()
    for _ in range(cfg.t_rollout):
        with torch.no_grad():
            a, logp, _, v = model.get_action_and_value(
                Batch.from_data_list([obs]))
        nxt, r, done, _ = env.step(int(a))
        buf.add([obs], [int(a)], [float(logp)], [float(v)], [r],
                [float(done)])
        obs = env.reset() if done else nxt
    with torch.no_grad():
        last_v = model.get_value(Batch.from_data_list([obs]))
    buf.compute_returns(last_v, cfg.gamma, cfg.gae_lambda)
    return buf


# ---- spec test 2: ratio == 1 on the first minibatch, first epoch ----
def test_first_minibatch_ratio_is_one(env, cfg):
    model = _model(cfg)
    buf = _rollout(model, env, cfg)
    gen = torch.Generator().manual_seed(0)
    batch, actions, old_logp, _, _, _ = next(
        buf.minibatches(cfg.n_minibatch, generator=gen))
    with torch.no_grad():
        _, new_logp, _, _ = model.get_action_and_value(batch,
                                                       action=actions)
    ratio = (new_logp - old_logp).exp()
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


# ---- full update: metrics, single-optimizer training signal ----
def test_ppo_update_metrics_and_training(env, cfg):
    model = _model(cfg)
    buf = _rollout(model, env, cfg)
    before = [p.clone() for p in model.parameters()]
    opt = make_optimizer(model, cfg)
    assert opt.defaults["eps"] == cfg.adam_eps

    metrics = ppo_update(model, opt, buf, cfg, progress=0.0,
                         generator=torch.Generator().manual_seed(0))
    for key in ("entropy", "approx_kl", "clipfrac",
                "explained_variance", "pg_loss", "v_loss"):
        assert key in metrics and math.isfinite(metrics[key])
    assert 1 <= metrics["epochs_run"] <= cfg.k_epochs
    assert metrics["c2"] == cfg.c2_start
    # single loss trains actor, critic and shared encoder together
    after = list(model.parameters())
    for group in (model.actor, model.critic, model.encoder):
        params = set(map(id, group.parameters()))
        assert any(not torch.equal(b, a)
                   for b, a in zip(before, after)
                   if id(a) in params)


# ---- KL early stop: remaining epochs are skipped ----
def test_kl_early_stop(env, cfg):
    model = _model(cfg)
    buf = _rollout(model, env, cfg)
    hot = dataclasses.replace(cfg, lr=0.5, target_kl=1e-8)
    metrics = ppo_update(model, make_optimizer(model, hot), buf, hot,
                         progress=0.0,
                         generator=torch.Generator().manual_seed(0))
    assert metrics["epochs_run"] < hot.k_epochs


# ---- c2 (ent_coef): DR-ALNS default 0.0, schedule still linear ----
def test_c2_schedule(cfg):
    for p in (0.0, 0.5, 1.0):          # DR-ALNS: ent_coef fixed at 0
        assert c2_schedule(cfg, p) == 0.0
    legacy = dataclasses.replace(cfg, c2_start=0.02, c2_end=0.003)
    assert c2_schedule(legacy, 0.0) == pytest.approx(0.02)
    assert c2_schedule(legacy, 0.5) == pytest.approx(0.0115)
    assert c2_schedule(legacy, 1.7) == pytest.approx(0.003)  # clipped


# ---- value loss: pessimistic max, not min ----
def test_value_loss_uses_max():
    new_v = torch.tensor([10.0])
    old_v = torch.tensor([0.0])
    ret = torch.tensor([0.0])
    # unclipped err = 100, clipped err = 0.2^2; max -> 100
    assert value_loss(new_v, old_v, ret, eps=0.2).item() \
        == pytest.approx(100.0)


# ---- advantage normalization is per minibatch ----
def test_constant_advantage_gives_zero_pg_loss(env, cfg):
    model = _model(cfg)
    buf = _rollout(model, env, cfg)
    buf.advantages = torch.ones_like(buf.advantages)   # (A-mean) == 0
    buf.returns = buf.advantages + torch.stack(buf.values)
    metrics = ppo_update(model, make_optimizer(model, cfg), buf, cfg,
                         progress=0.0,
                         generator=torch.Generator().manual_seed(0))
    assert metrics["pg_loss"] == pytest.approx(0.0, abs=1e-6)
