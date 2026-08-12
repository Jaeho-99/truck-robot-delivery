"""Inference/evaluation for the PPO operator selector (spec: eval).

Runs the SAME ALNSEnv used during training — the geometric-cooling
acceptance criterion and the max_iter horizon are identical by
construction (both come from the checkpoint config). Actions are
greedy argmax over the policy logits, mirroring the DQN's greedy
inference (selector_gnn.py).
"""

import dataclasses
import time

import torch
from torch_geometric.data import Batch

from .actor_critic import ActorCritic
from .config import PPOConfig
from .env import ALNSEnv


class FixedInstanceProvider:
    """provider.sample() interface over a single Params (for eval)."""

    def __init__(self, pr):
        self.pr = pr

    def sample(self):
        return self.pr


def load_model(path, device="cpu"):
    """Load a train.py checkpoint -> (model in eval mode, cfg, norms)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in ckpt["config"].items()
                       if k in known})
    cfg.device = device
    model = ActorCritic(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt["norms"]


def evaluate_instance(model, cfg, builder, pr, seed=0):
    """One greedy episode on pr; returns (best_solution, stats).

    stats keys follow solve_alns's stats dict where they overlap
    (init_cost, best_cost, improve_pct, iters_done, action_hist,
    best_trace) so results are directly comparable.
    """
    device = next(model.parameters()).device
    env = ALNSEnv(FixedInstanceProvider(pr), builder, cfg, seed)
    obs = env.reset()
    best_sol, best_cost = env.sol.clone(), env.f_best
    action_hist = [0] * cfg.n_actions
    t0 = time.time()
    best_trace = [(0, 0.0, round(best_cost, 6))]

    for _ in range(cfg.max_iter):
        with torch.no_grad():
            logits = model.actor(model._state(
                Batch.from_data_list([obs]).to(device)))
        a = int(logits.argmax(dim=1).item())
        action_hist[a] += 1
        obs, _, done, info = env.step(a)
        if info["f_best"] < best_cost - 1e-9:
            # a new best is always accepted, so env.sol holds it now
            best_sol, best_cost = env.sol.clone(), info["f_best"]
            best_trace.append((env.t, round(time.time() - t0, 3),
                               round(best_cost, 6)))
        if done:
            break

    stats = {"init_cost": env.f_init, "best_cost": best_cost,
             "improve_pct":
                 100.0 * (env.f_init - best_cost) / env.f_init,
             "iters_done": env.t,
             "runtime_s": time.time() - t0,
             "action_hist": action_hist, "best_trace": best_trace}
    return best_sol, stats
