"""Verify a saved ALNS solution against the MILP (gold standard).

Loads a solution JSON written by run_qlearning.py, rebuilds its
instance from the embedded instance_set, fixes the MILP's routing
binaries (x, y, u, uhat) to the solution's arcs — loads, timing and
lateness stay free, as they are implied by the routes — and re-solves
with symmetry breaking off. OPTIMAL status plus
|milp_obj - stored obj| <= 1e-3 confirms the ALNS evaluator and the
exact model agree on this solution; on infeasibility the Gurobi IIS
constraint names are printed. Requires gurobipy:

  /opt/anaconda3/bin/python experiments/verify_solution_milp.py \
      --solution results/<exp>/solutions/sol_n5_s1_roulette_0.json
"""

import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance, model                           # noqa: E402
from src.heuristics import Params, eval_solution          # noqa: E402
from src.heuristics.solution import Solution              # noqa: E402

BRK = [("truck fixed", "obj_truck_fixed", "truck_fixed"),
       ("robot fixed", "obj_robot_fixed", "robot_fixed"),
       ("truck travel", "obj_truck_travel", "truck_travel"),
       ("robot travel", "obj_robot_travel", "robot_travel"),
       ("lateness", "obj_lateness", "lateness")]


def rebuild_instance(meta):
    iset = meta["instance_set"]
    name = meta["instance"]
    if iset["type"] == "scaling":
        n = int(re.match(r"n(\d+)_s\d+", name).group(1))
        master = instance.build_master(iset["seed"])
        return instance.build_scaling_instance(
            master, n,
            num_trucks=iset.get("num_trucks", 5),
            num_robots=iset.get("num_robots", 3),
            num_parking_copies=iset.get("num_parking_copies", 2),
            beta_robot=iset.get("beta_robot", 3))
    if iset["type"] == "grid":
        seed = int(name.rsplit("_s", 1)[1])
        grid = next(g for g in iset["grids"]
                    if name.startswith(g["name"] + "_s"))
        kwargs = {k: v for k, v in grid.items() if k != "name"}
        inst = instance.build_grid_instance(
            seed=seed, beta_robot=iset.get("beta_robot", 3), **kwargs)
        e_c, l_c = instance.reachability_tw(inst, seed)
        return inst, e_c, l_c
    raise ValueError(f"unknown instance_set type: {iset['type']}")


def binaries_from_routes(routes, D):
    """ALNS routes -> support sets of the MILP binaries x/y/u/uhat."""
    fix = {"x": set(), "y": set(), "u": set(), "uhat": set()}
    for k_str, route in routes.items():
        k = int(k_str)
        if not route:
            continue
        fix["u"].add(k)
        prev = 0
        for st in route:
            node = st["c"] if st["kind"] == "cust" else st["p"]
            fix["x"].add((k, prev, node))
            prev = node
            if st["kind"] == "park":
                for tr in st["deploys"]:
                    r = tr["r"]
                    fix["uhat"].add((k, r))
                    rprev = st["p"]
                    for c in tr["custs"]:
                        fix["y"].add((k, r, rprev, c))
                        rprev = c
                    fix["y"].add((k, r, rprev, tr["ret_p"]))
        fix["x"].add((k, prev, D))
    return fix


def main():
    ap = argparse.ArgumentParser(
        description="MILP fixing check of a saved ALNS solution")
    ap.add_argument("--solution", required=True)
    ap.add_argument("--size-cap", type=int, default=25)
    ap.add_argument("--time-limit", type=int, default=60)
    args = ap.parse_args()

    with open(args.solution) as f:
        meta = json.load(f)
    inst, e_c, l_c = rebuild_instance(meta)
    if len(inst["C"]) > args.size_cap:
        print(f"SKIP: n={len(inst['C'])} > size cap {args.size_cap}")
        return 0

    # ALNS-side re-evaluation of the stored routes
    pr = Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])
    sol = Solution(inst["K"])
    sol.routes = {int(k): v for k, v in meta["routes"].items()}
    alns_obj, feas, brk, _ = eval_solution(pr, sol)
    print(f"[{meta['instance']} {meta['selector']} seed "
          f"{meta['seed']}] stored obj={meta['obj']:.4f}  "
          f"re-eval obj={alns_obj:.4f}  feasible={feas}")

    fix = binaries_from_routes(meta["routes"], inst["D"])
    res = model.run_model(
        inst, inst["alpha_traffic"], inst["alpha_ped"], e_c, l_c,
        model_name=f"verify_{meta['instance']}_{meta['selector']}",
        time_limit_sec=args.time_limit, mip_gap=0.0,
        beta_robot=inst["beta_robot"], symmetry_breaking=False,
        fix_binaries=fix)

    if res.get("obj") is None:
        print(f"MILP: no solution (status {res['status']})")
        for cname in res.get("iis", []):
            print(f"  IIS: {cname}")
        print("VERDICT: FAIL (infeasible under fixed binaries)")
        return 1

    diff = res["obj"] - meta["obj"]
    print(f"MILP: obj={res['obj']:.4f}  status={res['status']} "
          f"(2=OPTIMAL)  runtime={res['runtime_s']:.1f}s")
    print(f"{'component':<14}{'MILP':>12}{'ALNS eval':>12}")
    for label, mkey, akey in BRK:
        print(f"{label:<14}{res[mkey]:>12.4f}{brk[akey]:>12.4f}")
    print(f"{'total':<14}{res['obj']:>12.4f}{alns_obj:>12.4f}  "
          f"(stored {meta['obj']:.4f})")
    ok = res["status"] == 2 and abs(diff) <= 1e-3
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(milp - stored = {diff:+.6f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
