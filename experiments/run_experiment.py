"""Unified comparison framework: exact (MILP) vs ALNS.

Every comparison experiment is described by a JSON config (see
experiments/configs/); no per-experiment script is needed. The config
chooses the instance set, the methods to run, and the solver
parameters:

  python experiments/run_experiment.py \
      --config experiments/configs/toy_small.json

Config schema
-------------
{
  "experiment": str,              # results go to results/<experiment>/
  "methods": ["exact", "alns"],   # any subset, in run order
  "instance_set": {
    "type": "grid",               # grid instances (one cfg x seeds)
    "seeds": [1, 2, ...],
    "beta_robot": 3,
    "grids": [ {"name": str, ...build_grid_instance kwargs...}, ... ]
  }
  # -- or --
  "instance_set": {
    "type": "scaling",            # nested master-pool instances
    "seed": 1,
    "sizes": [5, 10, ...],        # ALNS sizes
    "exact_sizes": [5, 10],       # exact sizes (default: sizes)
    "num_trucks": 5, "num_robots": 3,
    "num_parking_copies": 2, "beta_robot": 3
  },
  "exact": {"time_limit_s": 300, "mip_gap": 0.0,
            "time_limit_overrides": {"<instance>": sec, ...}},
  "alns": {"iters": 3000, "seed": 0, "time_limit_s": null},
  "figures": true,                      # instance/route SVGs
  "check_evaluator_consistency": true   # rebuild the exact solution in
}                                       # the ALNS representation and
                                        # re-evaluate (diff must be ~0)

Outputs (under results/<experiment>/)
-------------------------------------
  instances/instance_{name}.json      instance payload (plot input)
  logs/exact_{name}.log               Gurobi log
  exact/exact_{name}.json             full exact result
  exact/solution_{name}.json          plot payload
  exact/report_{name}.txt             cost breakdown, routes, custody
                                      diagnostics
  alns/alns_{name}.json               ALNS summary
  alns/solution_{name}.json           plot payload
  alns/report_{name}.txt              cost breakdown, routes
  img/fig_instance_{name}.svg, img/fig_route_{method}_{name}.svg
  summary_exact.csv / summary_alns.csv / summary_compare.csv
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance, model, plotting                # noqa: E402
from src.heuristics import Params, eval_solution, solve_alns  # noqa: E402
from src.utils import (COST_ROWS, GRB_STATUS, alns_report,
                           alns_solution_payload, diagnose,
                           exact_to_solution, fleet_stats,
                           format_routes, milp_solution_payload,
                           write_csv)                    # noqa: E402


# ============================================================
# 1. Instance sets
# ============================================================
def iter_instances(cfg):
    """Yield (name, inst, e_c, l_c, run_exact) per the config."""
    iset = cfg["instance_set"]
    if iset["type"] == "grid":
        beta_robot = iset.get("beta_robot", 3)
        for grid in iset["grids"]:
            kwargs = {k: v for k, v in grid.items() if k != "name"}
            for seed in iset["seeds"]:
                inst = instance.build_grid_instance(
                    seed=seed, beta_robot=beta_robot, **kwargs)
                e_c, l_c = instance.reachability_tw(inst, seed)
                yield f"{grid['name']}_s{seed}", inst, e_c, l_c, True
    elif iset["type"] == "scaling":
        seed = iset["seed"]
        master = instance.build_master(seed)
        exact_sizes = set(iset.get("exact_sizes", iset["sizes"]))
        for n in iset["sizes"]:
            inst, e_c, l_c = instance.build_scaling_instance(
                master, n,
                num_trucks=iset.get("num_trucks", 5),
                num_robots=iset.get("num_robots", 3),
                num_parking_copies=iset.get("num_parking_copies", 2),
                beta_robot=iset.get("beta_robot", 3))
            yield f"n{n}_s{seed}", inst, e_c, l_c, n in exact_sizes
    else:
        raise ValueError(f"unknown instance_set type: {iset['type']}")


# ============================================================
# 2. Methods
# ============================================================
def run_exact(cfg, dirs, name, inst, e_c, l_c, figures):
    ecfg = cfg.get("exact", {})
    tl = ecfg.get("time_limit_overrides", {}).get(
        name, ecfg.get("time_limit_s", 300))
    print(f"[exact {name}] time limit {tl}s ...", flush=True)
    res = model.run_model(
        inst, inst["alpha_traffic"], inst["alpha_ped"], e_c, l_c,
        model_name=f"exact_{name}", time_limit_sec=tl,
        mip_gap=ecfg.get("mip_gap", 0.0),
        beta_robot=inst["beta_robot"],
        log_path=os.path.join(dirs["logs"], f"exact_{name}.log"))
    with open(os.path.join(dirs["exact"], f"exact_{name}.json"),
              "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    if res.get("obj") is None:
        status = GRB_STATUS.get(res.get("status"), res.get("status"))
        print(f"  no solution (status {status}, "
              f"{res['runtime_s']:.0f}s)")
        return {"instance": name, "n_cust": len(inst["C"]),
                "status": res.get("status"),
                "runtime_s": round(res["runtime_s"], 1)}, None

    sol_json = os.path.join(dirs["exact"], f"solution_{name}.json")
    with open(sol_json, "w") as f:
        json.dump(milp_solution_payload(inst, f"exact_{name}", res), f,
                  ensure_ascii=False, indent=2)
    write_exact_report(
        os.path.join(dirs["exact"], f"report_{name}.txt"), name, tl, res)
    if figures:
        plotting.make_route_svg(
            dirs["inst_json"][name], sol_json,
            os.path.join(dirs["img"], f"fig_route_exact_{name}.svg"))

    optimal = res["status"] == 2            # GRB.OPTIMAL
    row = {"instance": name, "n_cust": len(inst["C"]),
           "status": res["status"], "optimal": optimal,
           "time_limit_s": tl, "obj": round(res["obj"], 4),
           "mip_gap": round(res["gap"], 4),
           "runtime_s": round(res["runtime_s"], 1),
           "trucks": res["n_trucks_used"],
           "robots": res["n_robots_used"],
           "robot_cust": res["robot_customers"],
           "lateness_min": round(res["total_lateness_min"], 2),
           "n_vars": res["n_vars"], "n_constrs": res["n_constrs"]}
    print(f"  obj={res['obj']:.4f}  gap={res['gap']:.2%}  "
          f"{res['runtime_s']:.0f}s  trucks={res['n_trucks_used']} "
          f"robots={res['n_robots_used']}"
          f"{'  (OPTIMAL)' if optimal else ''}")
    return row, res


def write_exact_report(path, name, tl, res):
    """Cost breakdown, routes and custody diagnostics (text report)."""
    (n_sorties, n_overlaps, overlap_detail,
     custody_ok, custody_detail) = diagnose(res)
    lines = [
        f"run     : {name}",
        f"status  : {GRB_STATUS.get(res['status'], res['status'])}  "
        f"runtime={res['runtime_s']:.1f}s  time_limit={tl}s",
        f"objective = {res['obj']:.4f}  (MIP gap {res['gap']:.2%})",
        "",
        "[objective breakdown]",
    ]
    for lab, key in COST_ROWS:
        lines.append(f"  {lab:<30}{res[key]:>12.4f}")
    lines += [
        "",
        f"[fleet] trucks used={res['n_trucks_used']}  "
        f"robots used={res['n_robots_used']}  "
        f"robot-served customers={res['robot_customers']}  "
        f"truck-served customers={res['truck_customers']}  "
        f"total lateness={res['total_lateness_min']:.1f} min",
        "",
        format_routes(res),
        "",
        f"[diagnostics] sorties={n_sorties}  "
        f"time-overlaps={n_overlaps}  custody_ok={custody_ok}",
    ]
    for (k, r, i1, i2) in overlap_detail:
        lines.append(f"  ! robot {k}-{r} overlap: {i1[2]} "
                     f"[{i1[0]},{i1[1]}]  vs  {i2[2]} [{i2[0]},{i2[1]}]")
    for msg in custody_detail:
        lines.append(f"  ! {msg}")
    if n_overlaps == 0 and custody_ok:
        lines.append("  OK — delivery trips strictly sequential, "
                     "no overlap (custody (19)-(25) satisfied).")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_alns(cfg, dirs, name, inst, e_c, l_c, figures):
    acfg = cfg.get("alns", {})
    iters = acfg.get("iters", 3000)
    alns_seed = acfg.get("seed", 0)
    alns_tl = acfg.get("time_limit_s")
    pr = Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])
    print(f"[alns  {name}] iters {iters}"
          f"{f' (TL {alns_tl}s)' if alns_tl else ''} ...", flush=True)
    t0 = time.time()
    best, cost, stats = solve_alns(pr, iters=iters, seed=alns_seed,
                                   time_limit_s=alns_tl)
    rt = time.time() - t0
    ntr, nrb, rc = fleet_stats(best)
    _, feas, brk, _ = eval_solution(pr, best)

    sol_json = os.path.join(dirs["alns"], f"solution_{name}.json")
    with open(sol_json, "w") as f:
        json.dump(alns_solution_payload(pr, best, f"alns_{name}", cost),
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(dirs["alns"], f"alns_{name}.json"),
              "w") as f:
        json.dump({"instance": name, "n_cust": len(inst["C"]),
                   "obj": cost, "runtime_s": rt, "feasible": feas,
                   "iters": iters, "iters_done": stats["iters_done"],
                   "time_limit_s": alns_tl, "alns_seed": alns_seed,
                   "init_obj": stats["init_cost"],
                   "improve_pct": stats["improve_pct"],
                   "trucks": ntr, "robots": nrb, "robot_cust": rc,
                   "breakdown": brk},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(dirs["alns"], f"report_{name}.txt"),
              "w") as f:
        f.write(alns_report(pr, best, cost, stats) + "\n")
    if figures:
        plotting.make_route_svg(
            dirs["inst_json"][name], sol_json,
            os.path.join(dirs["img"], f"fig_route_alns_{name}.svg"))

    row = {"instance": name, "n_cust": len(inst["C"]),
           "obj": round(cost, 4), "runtime_s": round(rt, 1),
           "iters_done": stats["iters_done"], "feasible": feas,
           "init_obj": round(stats["init_cost"], 4),
           "improve_pct": round(stats["improve_pct"], 1),
           "trucks": ntr, "robots": nrb, "robot_cust": rc,
           "lateness_cost": round(brk["lateness"], 4)}
    print(f"  obj={cost:.4f}  {rt:.1f}s ({stats['iters_done']} iters)  "
          f"init={stats['init_cost']:.2f} "
          f"(-{stats['improve_pct']:.1f}%)  "
          f"trucks={ntr} robots={nrb} robot_cust={rc}")
    return row, pr


# ============================================================
# 3. Driver
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Config-driven exact vs ALNS comparison")
    ap.add_argument("--config", required=True,
                    help="path to a JSON experiment config")
    ap.add_argument("--results-dir", default=None,
                    help="output root (default: results/<experiment>)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    methods = cfg.get("methods", ["exact", "alns"])
    figures = cfg.get("figures", False)
    check_recon = cfg.get("check_evaluator_consistency", False)

    root = args.results_dir or os.path.join(REPO_ROOT, "results",
                                            cfg["experiment"])
    dirs = {"root": root,
            "instances": os.path.join(root, "instances"),
            "logs": os.path.join(root, "logs"),
            "exact": os.path.join(root, "exact"),
            "alns": os.path.join(root, "alns"),
            "img": os.path.join(root, "img"),
            "inst_json": {}}
    for key in ("instances", "logs", "exact", "alns", "img"):
        os.makedirs(dirs[key], exist_ok=True)

    exact_rows, alns_rows, recon_rows = [], [], {}
    for name, inst, e_c, l_c, exact_ok in iter_instances(cfg):
        inst_json = os.path.join(dirs["instances"],
                                 f"instance_{name}.json")
        with open(inst_json, "w") as f:
            json.dump(instance.instance_payload(inst), f,
                      ensure_ascii=False, indent=2)
        dirs["inst_json"][name] = inst_json
        if figures:
            plotting.make_instance_svg(
                inst_json,
                os.path.join(dirs["img"], f"fig_instance_{name}.svg"))

        exact_res = alns_pr = None
        for method in methods:
            if method == "exact":
                if not exact_ok:
                    continue
                row, exact_res = run_exact(cfg, dirs, name, inst,
                                           e_c, l_c, figures)
                exact_rows.append(row)
            elif method == "alns":
                row, alns_pr = run_alns(cfg, dirs, name, inst,
                                        e_c, l_c, figures)
                alns_rows.append(row)
            else:
                raise ValueError(f"unknown method: {method}")

        # Evaluator consistency: the exact solution rebuilt in the ALNS
        # representation must reproduce the exact objective.
        if check_recon and exact_res is not None and exact_res.get(
                "obj") is not None:
            pr = alns_pr or Params(inst, e_c, l_c,
                                   beta_robot=inst["beta_robot"])
            recon = exact_to_solution(pr, inst, exact_res)
            rcost, rok, _, _ = eval_solution(pr, recon)
            recon_rows[name] = {
                "recon_diff": round(rcost - exact_res["obj"], 6),
                "recon_feasible": rok}
            print(f"  recon diff {rcost - exact_res['obj']:+.6f} "
                  f"(feasible={rok})")

    # ---- summaries ----
    if exact_rows:
        write_csv(os.path.join(root, "summary_exact.csv"), exact_rows,
                  ["instance", "n_cust", "status", "optimal",
                   "time_limit_s", "obj", "mip_gap", "runtime_s",
                   "trucks", "robots", "robot_cust", "lateness_min",
                   "n_vars", "n_constrs"])
    if alns_rows:
        write_csv(os.path.join(root, "summary_alns.csv"), alns_rows,
                  ["instance", "n_cust", "obj", "runtime_s",
                   "iters_done", "feasible", "init_obj", "improve_pct",
                   "trucks", "robots", "robot_cust", "lateness_cost"])
    if exact_rows and alns_rows:
        by_e = {r["instance"]: r for r in exact_rows
                if r.get("obj") is not None}
        by_a = {r["instance"]: r for r in alns_rows}
        rows = []
        for name in [r["instance"] for r in alns_rows]:
            e, a = by_e.get(name), by_a.get(name)
            row = {"instance": name, "n_cust": a["n_cust"]}
            if e:
                row.update({"exact_obj": e["obj"],
                            "exact_mip_gap": e["mip_gap"],
                            "exact_optimal": e["optimal"],
                            "exact_runtime_s": e["runtime_s"],
                            "exact_robots": e["robots"]})
            row.update({"alns_obj": a["obj"],
                        "alns_runtime_s": a["runtime_s"],
                        "alns_robots": a["robots"]})
            if e:
                row["alns_gap_vs_exact_pct"] = round(
                    (a["obj"] - e["obj"]) / e["obj"] * 100.0, 3)
            row.update(recon_rows.get(name, {}))
            rows.append(row)
        write_csv(os.path.join(root, "summary_compare.csv"), rows,
                  ["instance", "n_cust", "exact_obj", "exact_mip_gap",
                   "exact_optimal", "exact_runtime_s", "exact_robots",
                   "alns_obj", "alns_runtime_s", "alns_robots",
                   "alns_gap_vs_exact_pct", "recon_diff",
                   "recon_feasible"])
    print(f"[done] results -> {root}")


if __name__ == "__main__":
    main()
