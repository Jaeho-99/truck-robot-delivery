"""Train one GNN+PPO operator selector per instance size (DR-ALNS).

  .venv/bin/python experiments/train_ppo.py --size 20 \
      --out models/gnn_ppo_n20.pt

Same protocol as train_gnn_dqn.py: 300,000 total steps (summed over
10 workers) in episodes of 100 search iterations; instances drawn with
replacement from data/train/n{size}; per-size normalization constants
stored in the checkpoint. Episode-reward CSV (rolling mean/std, window
100) is written next to the checkpoint and is readable by
plot_training.py.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.gnn_dqn.dataset import DirectoryInstanceProvider  # noqa: E402
from src.gnn_dqn.normalization import compute_norms       # noqa: E402
from src.ppo.config import PPOConfig                      # noqa: E402
from src.ppo.train import train                           # noqa: E402
from src.utils import write_csv                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="GNN+PPO training (one model per size)")
    ap.add_argument("--size", type=int, required=True,
                    choices=[20, 50, 100])
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--search-iterations", type=int, default=100)
    ap.add_argument("--n-envs", type=int, default=10)
    ap.add_argument("--t-rollout", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data", default="data")
    ap.add_argument("--trucks", type=int, default=None,
                    help="override the per-size default fleet "
                         "(dataset.DEFAULT_FLEET)")
    ap.add_argument("--robots", type=int, default=None)
    ap.add_argument("--beta-robot", type=int, default=3)
    ap.add_argument("--norm-sample", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="default: models/gnn_ppo_n{size}.pt")
    args = ap.parse_args()

    cfg = PPOConfig(total_steps=args.total_steps,
                    search_iterations=args.search_iterations,
                    n_envs=args.n_envs, t_rollout=args.t_rollout,
                    seed=args.seed, device=args.device)
    provider = DirectoryInstanceProvider(
        size=args.size, root=os.path.join(REPO_ROOT, args.data),
        num_trucks=args.trucks, num_robots=args.robots,
        beta_robot=args.beta_robot, seed=args.seed)
    print(f"[train] n{args.size}: {cfg.total_steps} steps = "
          f"{cfg.n_updates} updates x {cfg.n_envs} envs x "
          f"{cfg.t_rollout} steps, episodes of "
          f"{cfg.search_iterations}", flush=True)

    # per-size normalization constants, deterministic sample of the
    # train split (independent of the episode-sampling rng)
    norm_prs = [provider._params(p)
                for p in provider.train[:args.norm_sample]]
    norms = compute_norms(norm_prs)
    print(f"[train] norms (n{args.size}): {norms}", flush=True)

    out = os.path.join(REPO_ROOT,
                       args.out or f"models/gnn_ppo_n{args.size}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log_rows, update_rows = [], []
    try:
        train(cfg, provider, out, norms, log_rows, update_rows)
    finally:
        if log_rows:
            log_path = out.replace(".pt", "_train_log.csv")
            write_csv(log_path, log_rows, list(log_rows[0]))
            print(f"[train] episode log -> {log_path}", flush=True)
        if update_rows:
            upd_path = out.replace(".pt", "_update_log.json")
            with open(upd_path, "w") as f:
                json.dump(update_rows, f, indent=2)
            print(f"[train] update log -> {upd_path}", flush=True)
        print(f"[train] checkpoint -> {out}", flush=True)


if __name__ == "__main__":
    main()
