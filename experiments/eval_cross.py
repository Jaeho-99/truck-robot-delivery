"""Cross-size evaluation of size-specific GNN+DQN models (DR-ALNS
Table 5 format).

  .venv/bin/python experiments/eval_cross.py --seeds 10

Rows = trained models (n20 / n50 / n100 checkpoints), columns =
evaluation sizes; each cell = mean +- std of the best objective over
all test instances of that size x repetition seeds. Iteration budget
follows DR-ALNS: 100 for n=20/50, 200 for n=100 (--iterations
overrides all cells). Each model uses the normalization constants
stored in its own checkpoint (also on foreign sizes). If a cell's
budget exceeds the model's trained search_iterations, a warning is
printed: the progress/stagnation features then leave the distribution
seen in training.
"""

import argparse
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch                                              # noqa: E402

from src.gnn_dqn.dataset import DirectoryInstanceProvider  # noqa: E402
from src.heuristics import solve_alns                     # noqa: E402
from src.utils import write_csv                           # noqa: E402

DEFAULT_BUDGET = {20: 100, 50: 100, 100: 200}


def main():
    ap = argparse.ArgumentParser(
        description="model x size cross evaluation")
    ap.add_argument("--models", nargs="+",
                    default=["20=models/gnn_dqn_n20.pt",
                             "50=models/gnn_dqn_n50.pt",
                             "100=models/gnn_dqn_n100.pt"],
                    help="SIZE=CHECKPOINT entries (row per model)")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[20, 50, 100])
    ap.add_argument("--seeds", type=int, default=10,
                    help="repetition seeds per instance (0..N-1)")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override the per-size budget "
                         f"(default {DEFAULT_BUDGET})")
    ap.add_argument("--max-instances", type=int, default=None,
                    help="cap test instances per size (smoke tests)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--trucks", type=int, default=None,
                    help="override the per-size default fleet "
                         "(dataset.DEFAULT_FLEET; fleet follows the "
                         "EVAL size, not the model)")
    ap.add_argument("--robots", type=int, default=None)
    ap.add_argument("--beta-robot", type=int, default=3)
    ap.add_argument("--out", default="results/cross_eval.csv")
    args = ap.parse_args()

    models = []
    for entry in args.models:
        size, path = entry.split("=", 1)
        path = os.path.join(REPO_ROOT, path)
        ckpt_cfg = torch.load(path, map_location="cpu",
                              weights_only=False)["config"]
        models.append((f"n{size}", path,
                       ckpt_cfg.get("search_iterations")))

    test_sets = {}
    for size in args.sizes:
        provider = DirectoryInstanceProvider(
            size=size, root=os.path.join(REPO_ROOT, args.data),
            num_trucks=args.trucks, num_robots=args.robots,
            beta_robot=args.beta_robot)
        insts = provider.test_set()
        if args.max_instances:
            insts = insts[:args.max_instances]
        test_sets[size] = insts

    runs, cells = [], {}
    for model_name, path, trained_iters in models:
        for size in args.sizes:
            budget = args.iterations or DEFAULT_BUDGET[size]
            if trained_iters and budget > trained_iters:
                print(f"[warn] {model_name} on n{size}: budget "
                      f"{budget} > trained search_iterations "
                      f"{trained_iters} — progress/stagnation "
                      f"features leave the training distribution",
                      flush=True)
            objs = []
            for inst_id, pr in test_sets[size]:
                for seed in range(args.seeds):
                    _, best_cost, _ = solve_alns(
                        pr, iters=budget, seed=seed,
                        selector="gnn_dqn",
                        q_params={"model_path": path})
                    objs.append(best_cost)
                    runs.append({"model": model_name,
                                 "eval_size": size,
                                 "instance": inst_id, "seed": seed,
                                 "iterations": budget,
                                 "obj": round(best_cost, 4)})
            mean = statistics.fmean(objs)
            std = statistics.pstdev(objs) if len(objs) > 1 else 0.0
            cells[(model_name, size)] = (mean, std)
            print(f"[cell] {model_name} x n{size}: "
                  f"{mean:.2f} +- {std:.2f} ({len(objs)} runs)",
                  flush=True)

    # ---- table: stdout ----
    col_names = [f"n{s}" for s in args.sizes]
    print("\nmodel \\ eval " + " | ".join(f"{c:>18}"
                                          for c in col_names))
    summary = []
    for model_name, _, _ in models:
        row = {"model": model_name}
        line = []
        for size in args.sizes:
            mean, std = cells[(model_name, size)]
            row[f"n{size}"] = f"{mean:.2f}+-{std:.2f}"
            line.append(f"{mean:9.2f} +- {std:6.2f}")
        summary.append(row)
        print(f"{model_name:>12} " + " | ".join(f"{c:>18}"
                                                for c in line))

    # ---- table: CSV (summary + per-run detail) ----
    out = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_csv(out, summary, ["model"] + col_names)
    detail = out.replace(".csv", "_runs.csv")
    write_csv(detail, runs, list(runs[0]))
    print(f"\n[eval] summary -> {out}\n[eval] runs -> {detail}",
          flush=True)


if __name__ == "__main__":
    main()
