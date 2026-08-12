"""Train the PPO operator selector on mixed-size scaling instances.

  .venv/bin/python experiments/train_ppo.py \
      --updates 100 --out models/ppo.pt

Training draws from master pools with seeds 2-4 (the test set, master
seed 1, is never sampled), matching train_gnn_dqn.py. Normalization
constants are computed once from the training distribution and stored
inside the checkpoint.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.gnn_dqn.normalization import compute_norms       # noqa: E402
from src.gnn_dqn.provider import ScalingInstanceProvider  # noqa: E402
from src.ppo.config import PPOConfig                      # noqa: E402
from src.ppo.train import train                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="PPO training")
    ap.add_argument("--updates", type=int, default=500)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--t-rollout", type=int, default=512)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[5, 10, 15, 20])
    ap.add_argument("--trucks", type=int, default=5)
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--out", default="models/ppo.pt")
    args = ap.parse_args()

    cfg = PPOConfig(n_updates=args.updates, n_envs=args.n_envs,
                    t_rollout=args.t_rollout, max_iter=args.max_iter,
                    seed=args.seed, device=args.device)
    provider = ScalingInstanceProvider(sizes=args.sizes, seed=args.seed,
                                       num_trucks=args.trucks,
                                       num_robots=args.robots)
    print(f"[train] {cfg.n_updates} updates x {cfg.n_envs} envs x "
          f"{cfg.t_rollout} steps, max_iter {cfg.max_iter}, "
          f"sizes {args.sizes}", flush=True)

    norm_sample = [provider.sample() for _ in range(20)]
    norms = compute_norms(norm_sample)
    print(f"[train] norms: {norms}", flush=True)

    out = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log_rows = []
    try:
        train(cfg, provider, out, norms, log_rows)
    finally:
        log_path = out.replace(".pt", "_train_log.json")
        with open(log_path, "w") as f:
            json.dump(log_rows, f, indent=2)
        print(f"[train] checkpoint -> {out}\n[train] log -> {log_path}",
              flush=True)


if __name__ == "__main__":
    main()
