"""Train the GNN+DQN operator selector on mixed-size scaling instances.

  .venv/bin/python experiments/train_gnn_dqn.py \
      --episodes 100 --episode-len 300 --out models/gnn_dqn.pt

Training draws from master pools with seeds 2-4; the test instances
(master seed 1, the ones used by run_qlearning.py) are never sampled.
The normalization constants are computed once from the training
distribution and stored inside the checkpoint.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.gnn_dqn.config import Config                     # noqa: E402
from src.gnn_dqn.normalization import compute_norms       # noqa: E402
from src.gnn_dqn.provider import ScalingInstanceProvider  # noqa: E402
from src.gnn_dqn.trainer import train                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="GNN+DQN offline training")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--episode-len", type=int, default=300)
    ap.add_argument("--train-freq", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--reward-mode", default="R1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[5, 10, 15, 20])
    ap.add_argument("--trucks", type=int, default=5)
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--out", default="models/gnn_dqn.pt")
    args = ap.parse_args()

    cfg = Config(n_episodes=args.episodes,
                 episode_len=args.episode_len,
                 train_freq=args.train_freq, warmup=args.warmup,
                 reward_mode=args.reward_mode, seed=args.seed)
    provider = ScalingInstanceProvider(sizes=args.sizes, seed=args.seed,
                                       num_trucks=args.trucks,
                                       num_robots=args.robots)
    print(f"[train] {cfg.n_episodes} episodes x {cfg.episode_len} "
          f"iters, sizes {args.sizes}", flush=True)

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
