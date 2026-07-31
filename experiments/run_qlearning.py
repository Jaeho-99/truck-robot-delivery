"""ALNS vs RL-ALNS comparison and Q-learning hyperparameter search.

Config-driven, mirroring run_experiment.py. Two parts, each optional:

  1. compare — roulette-wheel ALNS vs Q-learning ALNS (RL-ALNS) on the
     same instances over multiple run seeds.
  2. search  — grid search over the Q-learning rate (eta) and discount
     (gamma) for RL-ALNS.

No exact solver is involved, so this script deliberately avoids
importing model/gurobipy:

  python experiments/run_qlearning.py \
      --config experiments/configs/qlearning_toy.json

Config schema
-------------
{
  "experiment": str,              # results go to results/<experiment>/
  "instance_set": {...},          # same schema as run_experiment.py
  "alns": {"iters": 3000, "time_limit_s": null},
  "run_seeds": [0, 1, 2, 3, 4],   # ALNS restarts per configuration
  "compare": true,                # part 1 on/off
  "search": {                     # part 2 (omit to skip)
    "eta": [0.05, 0.1, 0.3],
    "gamma": [0.5, 0.9, 0.99]
  }
}

Outputs (under results/<experiment>/)
-------------------------------------
  runs_compare.csv        per-run rows (instance x selector x seed)
  summary_compare.csv     aggregated per (instance, selector)
  qtab/qtab_{name}.json   learned policy of the best RL-ALNS run
  runs_search.csv         per-run rows (instance x eta x gamma x seed)
  summary_search.csv      aggregated per (instance, eta, gamma)
  summary_search_overall.csv  per (eta, gamma) across instances,
                              gap measured vs each instance's best
"""

import argparse
import json
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance                                  # noqa: E402
from src.heuristics import Params, solve_alns             # noqa: E402
from src.utils import write_csv                           # noqa: E402


# ============================================================
# 1. Instances
# ============================================================
def iter_instances(cfg):
    """Yield (name, inst, e_c, l_c) per the config.

    Same schema as run_experiment.iter_instances minus the exact-solver
    flag; kept separate so this script runs without gurobipy.
    """
    iset = cfg["instance_set"]
    if iset["type"] == "grid":
        beta_robot = iset.get("beta_robot", 3)
        for grid in iset["grids"]:
            kwargs = {k: v for k, v in grid.items() if k != "name"}
            for seed in iset["seeds"]:
                inst = instance.build_grid_instance(
                    seed=seed, beta_robot=beta_robot, **kwargs)
                e_c, l_c = instance.reachability_tw(inst, seed)
                yield f"{grid['name']}_s{seed}", inst, e_c, l_c
    elif iset["type"] == "scaling":
        seed = iset["seed"]
        master = instance.build_master(seed)
        for n in iset["sizes"]:
            inst, e_c, l_c = instance.build_scaling_instance(
                master, n,
                num_trucks=iset.get("num_trucks", 5),
                num_robots=iset.get("num_robots", 3),
                num_parking_copies=iset.get("num_parking_copies", 2),
                beta_robot=iset.get("beta_robot", 3))
            yield f"n{n}_s{seed}", inst, e_c, l_c
    else:
        raise ValueError(f"unknown instance_set type: {iset['type']}")


# ============================================================
# 2. Run + aggregation helpers
# ============================================================
def run_once(pr, acfg, seed, selector, q_params=None):
    """One solve_alns run; returns (obj, runtime_s, stats)."""
    t0 = time.time()
    _, cost, stats = solve_alns(
        pr, iters=acfg.get("iters", 3000), seed=seed,
        time_limit_s=acfg.get("time_limit_s"),
        selector=selector, q_params=q_params)
    return cost, time.time() - t0, stats


def aggregate(rows):
    """Mean/std/min statistics over the per-run rows of one config."""
    objs = [r["obj"] for r in rows]
    return {
        "n_runs": len(rows),
        "mean_obj": round(statistics.mean(objs), 4),
        "std_obj": round(statistics.stdev(objs), 4) if len(objs) > 1
        else 0.0,
        "min_obj": round(min(objs), 4),
        "mean_runtime_s": round(statistics.mean(
            r["runtime_s"] for r in rows), 1),
        "mean_improve_pct": round(statistics.mean(
            r["improve_pct"] for r in rows), 1),
    }


# ============================================================
# 3. Part 1 — ALNS vs RL-ALNS
# ============================================================
def run_compare(cfg, root, insts):
    acfg = cfg.get("alns", {})
    seeds = cfg.get("run_seeds", [0, 1, 2])
    selectors = cfg.get("selectors", ["roulette", "qlearning"])
    sel_params = cfg.get("selector_params", {})
    qdir = os.path.join(root, "qtab")
    os.makedirs(qdir, exist_ok=True)

    raw, summary = [], []
    for name, pr in insts:
        by_sel = {}
        for sel in selectors:
            rows, best_run = [], None
            for s in seeds:
                obj, rt, stats = run_once(pr, acfg, s, sel,
                                          sel_params.get(sel))
                print(f"  [{name}] {sel} seed={s} obj={obj:.4f} "
                      f"{rt:.0f}s", flush=True)
                row = {"instance": name, "n_cust": len(pr.C),
                       "selector": sel, "seed": s,
                       "obj": round(obj, 4), "runtime_s": round(rt, 1),
                       "iters_done": stats["iters_done"],
                       "improve_pct": round(stats["improve_pct"], 1)}
                rows.append(row)
                if best_run is None or obj < best_run[0]:
                    best_run = (obj, stats)
            raw.extend(rows)
            by_sel[sel] = aggregate(rows)
            summary.append({"instance": name, "n_cust": len(pr.C),
                            "selector": sel, **by_sel[sel]})
            if sel == "qlearning":
                with open(os.path.join(qdir, f"qtab_{name}.json"),
                          "w") as f:
                    json.dump(best_run[1]["q_summary"], f,
                              ensure_ascii=False, indent=2)
        base = selectors[0]
        parts = [f"{sel} mean={by_sel[sel]['mean_obj']:.4f} "
                 f"min={by_sel[sel]['min_obj']:.4f}"
                 + (f" (diff {by_sel[sel]['mean_obj'] - by_sel[base]['mean_obj']:+.4f})"
                    if sel != base else "")
                 for sel in selectors]
        print(f"[compare {name}] " + " | ".join(parts), flush=True)

    write_csv(os.path.join(root, "runs_compare.csv"), raw,
              ["instance", "n_cust", "selector", "seed", "obj",
               "runtime_s", "iters_done", "improve_pct"])
    write_csv(os.path.join(root, "summary_compare.csv"), summary,
              ["instance", "n_cust", "selector", "n_runs", "mean_obj",
               "std_obj", "min_obj", "mean_runtime_s",
               "mean_improve_pct"])


# ============================================================
# 4. Part 2 — eta x gamma grid search for RL-ALNS
# ============================================================
def run_search(cfg, root, insts):
    acfg = cfg.get("alns", {})
    seeds = cfg.get("run_seeds", [0, 1, 2])
    etas = cfg["search"]["eta"]
    gammas = cfg["search"]["gamma"]

    raw, summary = [], []
    for name, pr in insts:
        for eta in etas:
            for gamma in gammas:
                rows = []
                for s in seeds:
                    obj, rt, stats = run_once(
                        pr, acfg, s, "qlearning",
                        q_params={"eta": eta, "gamma": gamma})
                    rows.append({"instance": name, "n_cust": len(pr.C),
                                 "eta": eta, "gamma": gamma, "seed": s,
                                 "obj": round(obj, 4),
                                 "runtime_s": round(rt, 1),
                                 "iters_done": stats["iters_done"],
                                 "improve_pct": round(
                                     stats["improve_pct"], 1)})
                raw.extend(rows)
                a = aggregate(rows)
                summary.append({"instance": name, "n_cust": len(pr.C),
                                "eta": eta, "gamma": gamma, **a})
                print(f"[search {name}] eta={eta} gamma={gamma} "
                      f"mean={a['mean_obj']:.4f} "
                      f"min={a['min_obj']:.4f}", flush=True)

    # Overall ranking: per-run gap vs the instance's best objective
    # (normalizes away instance scale), averaged per (eta, gamma).
    best = {}
    for r in raw:
        best[r["instance"]] = min(best.get(r["instance"], r["obj"]),
                                  r["obj"])
    overall = []
    for eta in etas:
        for gamma in gammas:
            gaps = [100.0 * (r["obj"] - best[r["instance"]])
                    / best[r["instance"]]
                    for r in raw
                    if r["eta"] == eta and r["gamma"] == gamma]
            overall.append({"eta": eta, "gamma": gamma,
                            "n_runs": len(gaps),
                            "mean_gap_vs_best_pct": round(
                                statistics.mean(gaps), 3),
                            "max_gap_vs_best_pct": round(max(gaps), 3)})
    overall.sort(key=lambda r: r["mean_gap_vs_best_pct"])
    print("[search] best (eta, gamma) by mean gap: "
          f"({overall[0]['eta']}, {overall[0]['gamma']}) "
          f"{overall[0]['mean_gap_vs_best_pct']}%")

    write_csv(os.path.join(root, "runs_search.csv"), raw,
              ["instance", "n_cust", "eta", "gamma", "seed", "obj",
               "runtime_s", "iters_done", "improve_pct"])
    write_csv(os.path.join(root, "summary_search.csv"), summary,
              ["instance", "n_cust", "eta", "gamma", "n_runs",
               "mean_obj", "std_obj", "min_obj", "mean_runtime_s",
               "mean_improve_pct"])
    write_csv(os.path.join(root, "summary_search_overall.csv"), overall,
              ["eta", "gamma", "n_runs", "mean_gap_vs_best_pct",
               "max_gap_vs_best_pct"])


# ============================================================
# 5. Driver
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="ALNS vs RL-ALNS comparison + Q-learning "
                    "hyperparameter search")
    ap.add_argument("--config", required=True,
                    help="path to a JSON experiment config")
    ap.add_argument("--results-dir", default=None,
                    help="output root (default: results/<experiment>)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    root = args.results_dir or os.path.join(REPO_ROOT, "results",
                                            cfg["experiment"])
    os.makedirs(root, exist_ok=True)

    # Params is built once per instance and reused across all runs
    # (arc caches shared; solve_alns itself never mutates it).
    insts = []
    for name, inst, e_c, l_c in iter_instances(cfg):
        insts.append((name, Params(inst, e_c, l_c,
                                   beta_robot=inst["beta_robot"])))
        print(f"[setup] built {name} ({len(inst['C'])} customers)",
              flush=True)
    print(f"[setup] {len(insts)} instances, "
          f"run seeds {cfg.get('run_seeds', [0, 1, 2])}", flush=True)

    if cfg.get("compare", True):
        run_compare(cfg, root, insts)
    if cfg.get("search"):
        run_search(cfg, root, insts)
    print(f"[done] results -> {root}")


if __name__ == "__main__":
    main()
