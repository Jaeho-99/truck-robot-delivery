"""PPO training loop smoke test (DR-ALNS protocol, scaled down).

Reduced parameters (n=5 procedural instance, 2 envs, t_rollout 32,
total_steps 3200 -> 50 updates, episodes of 100) keep the runtime
test-friendly while episode ends and rollout-boundary truncations both
occur (episode 100 vs rollout window 32).

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
    cfg = PPOConfig(n_envs=2, t_rollout=32, search_iterations=100,
                    total_steps=3200, n_minibatch=4, k_epochs=2,
                    seed=0)
    assert cfg.n_updates == 50
    out = str(tmp_path_factory.mktemp("ppo") / "ppo_smoke.pt")
    log_rows, update_rows = [], []
    model = train(cfg, FixedProvider(pr), out, compute_norms([pr]),
                  log_rows, update_rows)
    return cfg, model, log_rows, update_rows, out


def test_smoke_update_metrics_finite(smoke):
    cfg, model, _, update_rows, _ = smoke
    assert len(update_rows) == cfg.n_updates
    for row in update_rows:
        for key in ("pg_loss", "v_loss", "entropy", "approx_kl",
                    "clipfrac"):
            assert math.isfinite(row[key]), (row["update"], key)
        assert not math.isinf(row["explained_variance"])
        assert row["entropy"] > 0.5          # no premature collapse
        assert row["c2"] == 0.0              # DR-ALNS ent_coef
        assert row["epochs_run"] == cfg.k_epochs   # no KL early stop
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_smoke_episode_log(smoke):
    cfg, _, log_rows, _, _ = smoke
    # 3200 total steps / 100-iteration episodes -> 32 finished episodes
    assert len(log_rows) == cfg.total_steps // cfg.search_iterations
    assert {"episode", "step", "episode_reward", "reward_roll_mean",
            "reward_roll_std", "best_cost", "init_cost"} \
        <= set(log_rows[0])
    steps = [row["step"] for row in log_rows]
    assert steps == sorted(steps)
    for row in log_rows:
        # binary reward: episode reward = 5 * (# best improvements)
        assert row["episode_reward"] % 5.0 == 0.0
        assert row["best_cost"] <= row["init_cost"] + 1e-9


def test_checkpoint_roundtrip(smoke):
    cfg, model, _, _, out = smoke
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert {"model", "config", "norms"} <= set(ckpt)
    assert ckpt["config"]["search_iterations"] == 100
    model2 = ActorCritic(PPOConfig(**ckpt["config"]))
    model2.load_state_dict(ckpt["model"])    # raises on mismatch
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
