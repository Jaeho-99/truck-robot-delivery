"""Shared reporting utilities: route strings, custody diagnostics,
plot payloads, CSV helpers, and the exact-to-ALNS solution
reconstruction used to verify evaluator consistency."""

import csv
from collections import defaultdict

from .heuristics.solution import Solution, eval_solution

GRB_STATUS = {2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD",
              5: "UNBOUNDED", 9: "TIME_LIMIT", 11: "INTERRUPTED",
              13: "SUBOPTIMAL"}

COST_ROWS = [
    ("Truck fixed cost", "obj_truck_fixed"),
    ("Robot fixed cost", "obj_robot_fixed"),
    ("Truck travel cost (fuel+env)", "obj_truck_travel"),
    ("Robot travel cost (fuel+env)", "obj_robot_travel"),
    ("Lateness penalty cost", "obj_lateness"),
    ("Truck environment cost", "obj_truck_env"),
    ("Robot environment cost", "obj_robot_env"),
    ("Total environment cost", "obj_total_env"),
]


def _is_parking(label):
    return label.startswith("P")


def _node_name(label):
    return "Depot" if label == "D" else label


def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


# ==========================================
# 1. Exact (MILP) result helpers
# ==========================================
def format_routes(res):
    """Ordered truck/robot route strings for a run_model result.

    Trucks as Depot -> ... -> Depot; robots per sortie (sorted by
    departure time) as deploy parking -> customers -> retrieve parking.
    """
    lines = ["[routes]"]
    truck_succ = defaultdict(dict)
    for a in res.get("truck_arcs", []):
        truck_succ[a["k"]][a["i"]] = a["j"]
    for k in sorted(truck_succ):
        succ = truck_succ[k]
        path, cur, guard = ["Depot"], succ.get("D"), 0
        while cur is not None and cur != "D" and guard < 1000:
            path.append(_node_name(cur))
            cur = succ.get(cur)
            guard += 1
        path.append("Depot")
        lines.append(f"  truck {k}: " + " -> ".join(path))
    times = {(t["k"], t["r"], t["node"]): t
             for t in res.get("robot_node_times", [])}
    dep_arcs = defaultdict(list)
    cust_succ = defaultdict(dict)
    for a in res.get("robot_arcs", []):
        key = (a["k"], a["r"])
        if _is_parking(a["i"]):
            dep_arcs[key].append((a["i"], a["j"]))
        else:
            cust_succ[key][a["i"]] = a["j"]
    for (k, r) in sorted(dep_arcs):
        sorties = []
        for (p, c0) in dep_arcs[(k, r)]:
            chain, cur, guard = [p, c0], c0, 0
            while (not _is_parking(cur) and cur in cust_succ[(k, r)]
                   and guard < 100):
                cur = cust_succ[(k, r)][cur]
                chain.append(cur)
                guard += 1
            t_dep = times.get((k, r, p), {}).get("bhat", 0.0)
            sorties.append((t_dep, chain))
        sorties.sort()
        for si, (td, chain) in enumerate(sorties, 1):
            lines.append(f"  robot {k}-{r} sortie {si} "
                         f"(depart {td:.1f} min): " + " -> ".join(chain))
    if not truck_succ and not dep_arcs:
        lines.append("  (no routes)")
    return "\n".join(lines)


def diagnose(res):
    """Custody integrity diagnostics for a run_model result.

    Checks (1) that no two delivery trips of one robot overlap in time
    and (2) that w = 1 (aboard) at deploy nodes and w = 0 (away) at
    retrieve nodes. Returns (n_sorties, n_overlaps, overlap_detail,
    custody_ok, custody_detail).
    """
    times = {(t["k"], t["r"], t["node"]): t
             for t in res.get("robot_node_times", [])}

    dep_arcs = defaultdict(list)
    cust_succ = defaultdict(dict)
    for a in res.get("robot_arcs", []):
        key = (a["k"], a["r"])
        if _is_parking(a["i"]):
            dep_arcs[key].append((a["i"], a["j"]))
        else:
            cust_succ[key][a["i"]] = a["j"]

    n_sorties = n_overlaps = 0
    detail = []
    for (k, r) in sorted(dep_arcs):
        trips = []
        for (p, c0) in dep_arcs[(k, r)]:
            cur, guard = c0, 0
            while (not _is_parking(cur) and cur in cust_succ[(k, r)]
                   and guard < 100):
                cur = cust_succ[(k, r)][cur]
                guard += 1
            if (k, r, p) not in times or (k, r, cur) not in times:
                continue
            trips.append((times[(k, r, p)]["bhat"],
                          times[(k, r, cur)]["ahat"], (p, cur)))
        n_sorties += len(trips)
        trips.sort()
        for i in range(len(trips)):
            for j in range(i + 1, len(trips)):
                a1, b1, ch1 = trips[i]
                a2, b2, ch2 = trips[j]
                if a1 < b2 - 1e-6 and a2 < b1 - 1e-6:
                    n_overlaps += 1
                    detail.append((k, r,
                                   (round(a1, 1), round(b1, 1), ch1),
                                   (round(a2, 1), round(b2, 1), ch2)))

    wmap = {(c["k"], c["r"], c["node"]): c["w"]
            for c in res.get("custody", [])}
    deploy_nodes = defaultdict(set)
    retr_nodes = defaultdict(set)
    for a in res.get("robot_arcs", []):
        if _is_parking(a["i"]):
            deploy_nodes[(a["k"], a["r"])].add(a["i"])
        if _is_parking(a["j"]):
            retr_nodes[(a["k"], a["r"])].add(a["j"])
    custody_detail = []
    for (k, r), nds in deploy_nodes.items():
        for p in nds:
            if wmap.get((k, r, p), 1) != 1:
                custody_detail.append(
                    f"robot {k}-{r}: deploy@{p} but "
                    f"w={wmap.get((k, r, p))} (should be aboard)")
    for (k, r), nds in retr_nodes.items():
        for p in nds:
            if wmap.get((k, r, p), 0) != 0:
                custody_detail.append(
                    f"robot {k}-{r}: retrieve@{p} but "
                    f"w={wmap.get((k, r, p))} (should be away)")
    custody_ok = len(custody_detail) == 0
    return n_sorties, n_overlaps, detail, custody_ok, custody_detail


def milp_solution_payload(inst, scenario, res):
    """Plot payload for a run_model result — nodes (labels) + arcs.

    Parking copies share coordinates and are merged in the figure.
    """
    cust = {inst["meta"][c]["label"] for c in inst["C"]}
    rserved = {a["j"] for a in res.get("robot_arcs", []) if a["j"] in cust}
    tserved = {a["j"] for a in res.get("truck_arcs", []) if a["j"] in cust}
    served = {}
    for lab in cust:
        if lab in rserved:
            served[lab] = "robot"
        elif lab in tserved:
            served[lab] = "truck"
        else:
            served[lab] = "none"
    nodes = []
    for i in inst["C"] + inst["P"] + [0]:
        mt = inst["meta"][i]
        if mt["type"] == "customer":
            served_by = served.get(mt["label"])
        else:
            served_by = None
        nodes.append({"label": mt["label"], "type": mt["type"],
                      "x": inst["nodes"][i][0], "y": inst["nodes"][i][1],
                      "zone": inst["node_zone"][i],
                      "served_by": served_by})
    return {"scenario": scenario, "objective": res.get("obj"),
            "gap": res.get("gap"), "nodes": nodes,
            "truck_arcs": res.get("truck_arcs", []),
            "robot_arcs": res.get("robot_arcs", []),
            "depot_label": "D"}


# ==========================================
# 2. ALNS solution helpers
# ==========================================
def fleet_stats(sol):
    """(trucks used, robots used, robot-served customers)."""
    ntr = sum(1 for r in sol.routes.values() if r)
    robots = {(k, tr["r"]) for k, route in sol.routes.items()
              for st in route if st["kind"] == "park"
              for tr in st["deploys"]}
    rob_cust = sum(len(tr["custs"]) for route in sol.routes.values()
                   for st in route if st["kind"] == "park"
                   for tr in st["deploys"])
    return ntr, len(robots), rob_cust


def solution_arcs(pr, sol):
    """Arcs compatible with milp_solution_payload (for plotting)."""
    def lbl(i):
        return pr.inst["meta"][i]["label"]

    truck_arcs, robot_arcs = [], []
    for k, route in sol.routes.items():
        if not route:
            continue
        prev = 0
        for st in route:
            node = st["c"] if st["kind"] == "cust" else st["p"]
            truck_arcs.append({"k": k, "i": lbl(prev), "j": lbl(node)})
            if st["kind"] == "park":
                for tr in st["deploys"]:
                    rprev = st["p"]
                    for c in tr["custs"]:
                        robot_arcs.append({"k": k, "r": tr["r"],
                                           "i": lbl(rprev),
                                           "j": lbl(c)})
                        rprev = c
                    robot_arcs.append({"k": k, "r": tr["r"],
                                       "i": lbl(rprev),
                                       "j": lbl(tr["ret_p"])})
            prev = node
        truck_arcs.append({"k": k, "i": lbl(prev), "j": lbl(pr.D)})
    return truck_arcs, robot_arcs


def alns_solution_payload(pr, sol, name, cost):
    """Same format as milp_solution_payload (make_route_svg input)."""
    truck_arcs, robot_arcs = solution_arcs(pr, sol)
    cust_lbls = {pr.inst["meta"][c]["label"] for c in pr.C}
    rserved = {a["j"] for a in robot_arcs if a["j"] in cust_lbls}
    tserved = {a["j"] for a in truck_arcs if a["j"] in cust_lbls}
    served = {ll: ("robot" if ll in rserved
                   else "truck" if ll in tserved else "none")
              for ll in cust_lbls}
    nodes = []
    for i in pr.C + pr.P + [0]:
        mt = pr.inst["meta"][i]
        nodes.append({"label": mt["label"], "type": mt["type"],
                      "x": pr.nodes[i][0], "y": pr.nodes[i][1],
                      "zone": pr.inst["node_zone"][i],
                      "served_by": (served.get(mt["label"])
                                    if mt["type"] == "customer"
                                    else None)})
    return {"scenario": name, "objective": cost, "gap": None,
            "nodes": nodes, "truck_arcs": truck_arcs,
            "robot_arcs": robot_arcs, "depot_label": "D"}


def alns_report(pr, sol, cost, stats):
    """Human-readable report of an ALNS solution."""
    def lbl(i):
        return pr.inst["meta"][i]["label"]

    _, feas, brk, _ = eval_solution(pr, sol)
    lines = []
    lines.append(f"objective = {cost:.4f}   (feasible={feas})")
    lines.append(f"  init={stats['init_cost']:.4f} -> "
                 f"best={stats['best_cost']:.4f}"
                 f"  ({stats['improve_pct']:.1f}% improvement)")
    lines.append("")
    lines.append("[objective breakdown]")
    lines.append(f"  {'Truck fixed cost':<30}{brk['truck_fixed']:>12.4f}")
    lines.append(f"  {'Robot fixed cost':<30}{brk['robot_fixed']:>12.4f}")
    lines.append(f"  {'Truck travel (fuel+env)':<30}"
                 f"{brk['truck_travel']:>12.4f}")
    lines.append(f"  {'Robot travel (fuel+env)':<30}"
                 f"{brk['robot_travel']:>12.4f}")
    lines.append(f"  {'Lateness penalty':<30}{brk['lateness']:>12.4f}")
    ntr, nrb, rob_cust = fleet_stats(sol)
    lines.append("")
    lines.append(f"[fleet] trucks used={ntr}  robots used={nrb}  "
                 f"robot-served customers={rob_cust}  "
                 f"truck-served={len(pr.C) - rob_cust}")
    lines.append("")
    lines.append("[routes]")
    for k, route in sol.routes.items():
        if not route:
            continue
        seq = ["Depot"]
        for st in route:
            if st["kind"] == "cust":
                seq.append(lbl(st["c"]))
            else:
                seq.append(f"[{lbl(st['p'])}]")
        seq.append("Depot")
        lines.append(f"  truck {k}: " + " -> ".join(seq))
        for st in route:
            if st["kind"] == "park":
                for tr in st["deploys"]:
                    chain = ([lbl(st["p"])] + [lbl(c) for c in tr["custs"]]
                             + [lbl(tr["ret_p"])])
                    lines.append(f"    robot {k}-{tr['r']}: "
                                 + " -> ".join(chain))
    lines.append("")
    lines.append(f"[adaptive weights] destroy={stats['destroy_w']}  "
                 f"repair={stats['repair_w']}")
    return "\n".join(lines)


# ==========================================
# 3. Exact solution -> ALNS representation (evaluator consistency)
# ==========================================
def exact_to_solution(pr, inst, res):
    """Rebuild a run_model result (label arcs) as an ALNS Solution.

    Parking labels contain '#' (e.g. P0#1); the depot is 'D'.
    Evaluating the reconstruction with eval_solution must reproduce the
    exact objective (difference ~ 0), which certifies that the ALNS
    evaluator agrees with the MILP.
    """
    lbl2idx = {inst["meta"][i]["label"]: i for i in inst["C"] + inst["P"]}

    def is_park(s):
        return "#" in s

    rsucc = defaultdict(dict)   # (k, r): cust label -> next label
    rdep = defaultdict(list)    # (k, r): [(deploy parking, 1st cust)]
    for a in res["robot_arcs"]:
        key = (a["k"], a["r"])
        if is_park(a["i"]):
            rdep[key].append((a["i"], a["j"]))
        else:
            rsucc[key][a["i"]] = a["j"]
    trips_at = defaultdict(list)    # (k, deploy parking) -> [trip, ...]
    for (k, r), deps in rdep.items():
        for (p0, c0) in deps:
            chain, cur = [c0], c0
            while True:
                nxt = rsucc[(k, r)][cur]
                if is_park(nxt):
                    ret = nxt
                    break
                chain.append(nxt)
                cur = nxt
            trips_at[(k, p0)].append(
                {"r": r, "custs": [lbl2idx[c] for c in chain],
                 "ret_p": lbl2idx[ret]})

    tsucc = defaultdict(dict)
    for a in res["truck_arcs"]:
        tsucc[a["k"]][a["i"]] = a["j"]
    sol = Solution(pr.K)
    for k, succ in tsucc.items():
        route, cur, guard = [], succ.get("D"), 0
        while cur is not None and cur != "D" and guard < 1000:
            if is_park(cur):
                route.append({"kind": "park", "p": lbl2idx[cur],
                              "deploys": trips_at.get((k, cur), [])})
            else:
                route.append({"kind": "cust", "c": lbl2idx[cur]})
            cur = succ.get(cur)
            guard += 1
        sol.routes[k] = route
    return sol
