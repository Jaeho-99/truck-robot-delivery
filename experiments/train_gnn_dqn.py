"""Train one GNN+DQN operator selector per instance size (DR-ALNS).

  .venv/bin/python experiments/train_gnn_dqn.py --size 20 \
      --out models/gnn_dqn_n20.pt

DR-ALNS protocol (Reijnen et al., ICAPS 2024): 300,000 total steps in
episodes of 100 search iterations; at each episode start one instance
is drawn with replacement from data/train/n{size}. Normalization
constants are computed per size from the training split and stored in
the checkpoint (each model keeps its own constants, also under
cross-size evaluation). The episode-reward log (rolling mean/std over
100 episodes) is written next to the checkpoint as *_train_log.csv.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.gnn_dqn.config import Config                     # noqa: E402
from src.gnn_dqn.dataset import DirectoryInstanceProvider  # noqa: E402
from src.gnn_dqn.normalization import compute_norms       # noqa: E402
from src.gnn_dqn.trainer import train                     # noqa: E402
from src.utils import write_csv                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="GNN+DQN offline training (one model per size)")
    ap.add_argument("--size", type=int, required=True,
                    choices=[20, 50, 100])
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--search-iterations", type=int, default=100)
    ap.add_argument("--reward-mode", default="R1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data")
    ap.add_argument("--trucks", type=int, default=None,
                    help="override the per-size default fleet "
                         "(dataset.DEFAULT_FLEET)")
    ap.add_argument("--robots", type=int, default=None)
    ap.add_argument("--beta-robot", type=int, default=3)
    ap.add_argument("--norm-sample", type=int, default=20,
                    help="train instances used for normalization "
                         "constants")
    ap.add_argument("--out", default=None,
                    help="default: models/gnn_dqn_n{size}.pt")
    args = ap.parse_args()

    cfg = Config(total_steps=args.total_steps,
                 search_iterations=args.search_iterations,
                 reward_mode=args.reward_mode, seed=args.seed)
    provider = DirectoryInstanceProvider(
        size=args.size, root=os.path.join(REPO_ROOT, args.data),
        num_trucks=args.trucks, num_robots=args.robots,
        beta_robot=args.beta_robot, seed=args.seed)
    print(f"[train] n{args.size}: {cfg.total_steps} steps = "
          f"{cfg.n_episodes} episodes x {cfg.search_iterations} "
          f"search iterations", flush=True)

    # per-size normalization constants, deterministic sample of the
    # train split (independent of the episode-sampling rng)
    norm_prs = [provider._params(p)
                for p in provider.train[:args.norm_sample]]
    norms = compute_norms(norm_prs)
    print(f"[train] norms (n{args.size}): {norms}", flush=True)

    out = os.path.join(REPO_ROOT,
                       args.out or f"models/gnn_dqn_n{args.size}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log_rows = []
    try:
        train(cfg, provider, out, norms, log_rows)
    finally:
        if log_rows:
            log_path = out.replace(".pt", "_train_log.csv")
            rows = [dict(r, actions=json.dumps(r["actions"]))
                    for r in log_rows]
            write_csv(log_path, rows, list(rows[0]))
            print(f"[train] checkpoint -> {out}\n"
                  f"[train] log -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
