"""Evaluate a trained PPO checkpoint on the scaling test set.

  .venv/bin/python experiments/eval_ppo.py \
      --model models/ppo.pt --sizes 5 10 15 20 --run-seeds 0 1 2

Mirrors run_qlearning.py's compare output so PPO rows are directly
comparable with the roulette / qlearning / gnn_dqn results: per-run
rows in runs_ppo.csv plus best solutions under solutions/. The
acceptance criterion and horizon come from the checkpoint config
(identical to training by construction). Every solution is re-checked
with the independent validator.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.gnn_dqn.graph_builder import GraphBuilder         # noqa: E402
from src.gnn_dqn.provider import ScalingInstanceProvider   # noqa: E402
from src.heuristics import eval_solution                   # noqa: E402
from src.heuristics.validator import (check_solution,      # noqa: E402
                                      cost_params_from)
from src.ppo.eval import evaluate_instance, load_model     # noqa: E402
from src.utils import (alns_solution_payload,              # noqa: E402
                       fleet_stats, write_csv)


def main():
    ap = argparse.ArgumentParser(description="PPO evaluation")
    ap.add_argument("--model", default="models/ppo.pt")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[5, 10, 15, 20])
    ap.add_argument("--run-seeds", type=int, nargs="+",
                    default=[0, 1, 2])
    ap.add_argument("--trucks", type=int, default=5)
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--experiment", default="main_ppo",
                    help="results go to results/<experiment>/")
    args = ap.parse_args()

    model, cfg, norms = load_model(
        os.path.join(REPO_ROOT, args.model), args.device)
    builder = GraphBuilder(norms, cfg)
    provider = ScalingInstanceProvider(sizes=args.sizes,
                                       num_trucks=args.trucks,
                                       num_robots=args.robots)
    insts = provider.test_set(args.sizes)
    print(f"[eval] {args.model} on {[n for n, _ in insts]}, "
          f"seeds {args.run_seeds}, max_iter {cfg.max_iter}",
          flush=True)

    root = os.path.join(REPO_ROOT, "results", args.experiment)
    sdir = os.path.join(root, "solutions")
    os.makedirs(sdir, exist_ok=True)

    rows = []
    for name, pr in insts:
        for s in args.run_seeds:
            sol, stats = evaluate_instance(model, cfg, builder, pr,
                                           seed=s)
            obj = stats["best_cost"]
            ev_obj, feas, brk, _ = eval_solution(pr, sol)
            ok_ind, viols, ind_obj = check_solution(
                pr.inst, pr.e_c, pr.l_c, sol, cost_params_from(pr))
            ind_diff = ind_obj - ev_obj
            if viols or abs(ind_diff) > 1e-4:
                print(f"  !! VALIDATOR [{name}] seed={s} "
                      f"diff={ind_diff:+.6f} violations={viols}",
                      flush=True)

            with open(os.path.join(
                    sdir, f"sol_{name}_ppo_{s}.json"), "w") as f:
                json.dump({
                    "instance": name, "selector": "ppo", "seed": s,
                    "obj": round(obj, 6),
                    "init_obj": round(stats["init_cost"], 6),
                    "routes": {str(k): v
                               for k, v in sol.routes.items()},
                    "plot_payload": alns_solution_payload(
                        pr, sol, f"ppo_{name}", obj),
                }, f, ensure_ascii=False, indent=2)

            ntr, nrb, rc = fleet_stats(sol)
            rows.append({
                "instance": name, "n_cust": len(pr.C),
                "selector": "ppo", "seed": s,
                "obj": round(obj, 4),
                "runtime_s": round(stats["runtime_s"], 1),
                "iters_done": stats["iters_done"],
                "improve_pct": round(stats["improve_pct"], 1),
                "init_obj": round(stats["init_cost"], 4),
                "feasible": feas,
                "indep_feasible": ok_ind,
                "indep_obj_diff": round(ind_diff, 6),
                "trucks": ntr, "robots": nrb, "robot_cust": rc,
                "lateness": round(brk["lateness"], 4),
                "action_hist": json.dumps(stats["action_hist"])})
            print(f"  [{name}] seed={s} obj={obj:.4f} "
                  f"(init {stats['init_cost']:.2f}, "
                  f"-{stats['improve_pct']:.1f}%) "
                  f"{stats['runtime_s']:.0f}s", flush=True)

    csv_path = os.path.join(root, "runs_ppo.csv")
    write_csv(csv_path, rows, list(rows[0]))
    print(f"[eval] rows -> {csv_path}\n[eval] solutions -> {sdir}",
          flush=True)


if __name__ == "__main__":
    main()
