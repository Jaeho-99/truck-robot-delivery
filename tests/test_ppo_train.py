"""Phase 5 tests: PPO training loop smoke test (spec test 7).

Scaled-down parameters (n=5 instance, 2 envs, t_rollout 32, 50
updates) to keep the runtime test-friendly; max_iter=100 < 2*t_rollout
so episode terminations and rollout-boundary truncations both occur.

Run with:  .venv/bin/python -m pytest tests/test_ppo_train.py -q
"""

import math
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance                                    # noqa: E402
from src.gnn_dqn.normalization import compute_norms         # noqa: E402
from src.heuristics import Params                           # noqa: E402
from src.ppo.actor_critic import ActorCritic                # noqa: E402
from src.ppo.config import PPOConfig                        # noqa: E402
from src.ppo.train import train                             # noqa: E402


class FixedProvider:
    def __init__(self, pr):
        self.pr = pr

    def sample(self):
        return self.pr


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    master = instance.build_master(2)
    inst, e_c, l_c = instance.build_scaling_instance(master, 5)
    pr = Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])
    cfg = PPOConfig(n_envs=2, t_rollout=32, max_iter=100, n_updates=50,
                    n_minibatch=4, k_epochs=2, seed=0)
    out = str(tmp_path_factory.mktemp("ppo") / "ppo_smoke.pt")
    log_rows = []
    model = train(cfg, FixedProvider(pr), out,
                  compute_norms([pr]), log_rows)
    return cfg, model, log_rows, out


# ---- spec test 7: no NaN/Inf over 50 updates, no entropy collapse ----
def test_smoke_metrics_finite_and_entropy(smoke):
    cfg, model, log_rows, _ = smoke
    assert len(log_rows) == cfg.n_updates
    for row in log_rows:
        for key in ("pg_loss", "v_loss", "entropy", "approx_kl",
                    "clipfrac"):
            assert math.isfinite(row[key]), (row["update"], key)
        # explained_variance may start meaningless but must not blow up
        assert not math.isinf(row["explained_variance"])
        assert row["entropy"] > 0.5          # no premature collapse
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_smoke_episodes_complete_and_log_fields(smoke):
    cfg, _, log_rows, _ = smoke
    # max_iter=100, 2 envs x 32 steps/update -> episodes finish
    assert sum(row["episodes_done"] for row in log_rows) >= 10
    finished = [row for row in log_rows if row["episodes_done"] > 0]
    assert all(row["mean_best_objective"] > 0 for row in finished)
    assert {"update", "c2", "epochs_run",
            "explained_variance"} <= set(log_rows[0])
    # c2 decays linearly across updates
    assert log_rows[0]["c2"] == pytest.approx(cfg.c2_start)
    assert log_rows[-1]["c2"] == pytest.approx(cfg.c2_end)


def test_checkpoint_roundtrip(smoke):
    cfg, model, _, out = smoke
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert {"model", "config", "norms"} <= set(ckpt)
    model2 = ActorCritic(PPOConfig(**ckpt["config"]))
    model2.load_state_dict(ckpt["model"])    # raises on mismatch
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
