"""Independent feasibility/objective validator for ALNS solutions.

Deliberately does NOT import solution.py (eval_truck/eval_solution):
every check and the objective are recomputed from instance primitives
(nodes, arc_zones, alpha dicts, lam, park_groups) and explicit cost
constants, so it can catch evaluator bugs instead of repeating them.
Constraint list follows the MILP formulation families (18)-(51) as
documented in model.py / solution.py headers:

  coverage (18)          every customer served exactly once
  custody (19)-(25)      deploy while aboard, retrieve while away,
                         deploy copy != retrieval copy, retrieval on a
                         strictly later stop of the same route, all
                         robots aboard / no pending retrievals at end
  copy budget            each parking copy used at most once
                         solution-wide, <= copies per physical group
  capacity/range         trip size <= beta_robot, robot distance <=
                         phi_hat, truck load <= beta_truck, truck
                         distance <= phi_truck (39)(44)(50)(51)
  schedule (26)-(38)     forward schedule recomputed from scratch,
                         lateness from l_c
"""

from collections import defaultdict


def cost_params_from(pr):
    """Extract the cost-constant dict from a Params object."""
    keys = ("s_kc", "s_hat", "zeta_load", "zeta_unload", "beta_truck",
            "beta_robot", "phi_truck", "phi_hat", "gamma_late",
            "gamma_fixed", "gammahat_fixed", "truck_arc_coef",
            "robot_arc_coef", "V_T", "V_R")
    return {k: getattr(pr, k) for k in keys}


def _routes_of(sol_like):
    """Normalize Solution-object / plain-dict / JSON routes."""
    routes = getattr(sol_like, "routes", sol_like)
    return {int(k): v for k, v in routes.items()}


def check_solution(inst, e_c, l_c, sol_like, cost_params):
    """Return (ok, violations, recomputed_obj)."""
    cp = cost_params
    nodes = inst["nodes"]
    az = inst["arc_zones"]
    at, ap = inst["alpha_traffic"], inst["alpha_ped"]
    lam = inst["lam"]
    C, D = set(inst["C"]), inst["D"]
    routes = _routes_of(sol_like)
    violations = []

    def dist(i, j):
        return (abs(nodes[i][0] - nodes[j][0])
                + abs(nodes[i][1] - nodes[j][1]))

    def tau_truck(i, j):
        eff = sum(km * at[z] for z, km in az[(i, j)].items())
        return eff / cp["V_T"] * 60.0

    def tau_robot(i, j):
        eff = sum(km * ap[z] for z, km in az[(i, j)].items())
        return eff / cp["V_R"] * 60.0

    # ---- coverage (18): exactly once over the whole solution ----
    served_count = defaultdict(int)
    for k, route in routes.items():
        for st in route:
            if st["kind"] == "cust":
                served_count[st["c"]] += 1
            else:
                for tr in st["deploys"]:
                    for c in tr["custs"]:
                        served_count[c] += 1
    for c in C:
        if served_count[c] != 1:
            violations.append(f"coverage: customer {c} served "
                              f"{served_count[c]} times")
    for c in served_count:
        if c not in C:
            violations.append(f"coverage: unknown customer {c}")

    # ---- copy budget: each copy at most once solution-wide ----
    copy_uses = defaultdict(int)
    for route in routes.values():
        for st in route:
            if st["kind"] == "park":
                copy_uses[st["p"]] += 1
    for p, n in copy_uses.items():
        if n > 1:
            violations.append(f"copy: parking copy {p} visited {n}x")
    ncopy = inst.get("num_parking_copies", 2)
    for gi, grp in enumerate(inst["park_groups"]):
        used = sum(1 for p in grp if copy_uses.get(p))
        if used > ncopy:
            violations.append(f"copy: group {gi} uses {used} > {ncopy}")

    # ---- per-truck replay: custody, capacity, range, schedule ----
    truck_travel = robot_travel = 0.0
    lateness = {}
    trucks_used = 0
    robots_used = set()
    robot_dist = defaultdict(float)

    for k, route in routes.items():
        if not route:
            continue
        trucks_used += 1
        aboard = defaultdict(lambda: True)
        pending = {}                    # ret_p -> [(r, arr_ret), ...]
        load = 0
        tdist = 0.0
        prev, b_prev = 0, 0.0
        for si, st in enumerate(route):
            node = st["c"] if st["kind"] == "cust" else st["p"]
            a_node = b_prev + tau_truck(prev, node)
            truck_travel += tau_truck(prev, node) * cp["truck_arc_coef"]
            tdist += dist(prev, node)
            if st["kind"] == "cust":
                c = st["c"]
                load += lam[c]
                lateness[c] = max(0.0, a_node - l_c[c])
                b_node = a_node + cp["s_kc"]
            else:
                p = st["p"]
                b_node = a_node
                later_parks = {s2["p"] for s2 in route[si + 1:]
                               if s2["kind"] == "park"}
                for tr in st["deploys"]:
                    r, custs, ret_p = tr["r"], tr["custs"], tr["ret_p"]
                    if not aboard[(k, r)]:
                        violations.append(
                            f"custody: truck {k} deploys robot {r} at "
                            f"{p} while away")
                    aboard[(k, r)] = False
                    robots_used.add((k, r))
                    if not custs:
                        violations.append(
                            f"trip: empty trip at {p} (truck {k})")
                    if len(custs) > cp["beta_robot"]:
                        violations.append(
                            f"trip: {len(custs)} custs > beta_robot "
                            f"{cp['beta_robot']} at {p}")
                    if ret_p == p:
                        violations.append(
                            f"custody: same-copy retrieval at {p}")
                    elif ret_p not in later_parks:
                        violations.append(
                            f"custody: ret_p {ret_p} not a later stop "
                            f"of truck {k}'s route (deploy at {p})")
                    load += sum(lam[c] for c in custs)
                    t = a_node + cp["zeta_unload"]
                    rprev = p
                    for c in custs:
                        ahat = t + tau_robot(rprev, c)
                        robot_travel += (tau_robot(rprev, c)
                                         * cp["robot_arc_coef"])
                        robot_dist[(k, r)] += dist(rprev, c)
                        lateness[c] = max(0.0, ahat - l_c[c])
                        t = ahat + cp["s_hat"]
                        rprev = c
                    arr = t + tau_robot(rprev, ret_p)
                    robot_travel += (tau_robot(rprev, ret_p)
                                     * cp["robot_arc_coef"])
                    robot_dist[(k, r)] += dist(rprev, ret_p)
                    pending.setdefault(ret_p, []).append((r, arr))
                if st["deploys"]:
                    b_node = max(b_node, a_node + cp["zeta_unload"])
                for (r, arr) in pending.pop(p, []):
                    if aboard[(k, r)]:
                        violations.append(
                            f"custody: truck {k} retrieves robot {r} "
                            f"at {p} while aboard")
                    aboard[(k, r)] = True
                    b_node = max(b_node, arr + cp["zeta_load"])
            prev, b_prev = node, b_node
        truck_travel += tau_truck(prev, D) * cp["truck_arc_coef"]
        tdist += dist(prev, D)

        if pending:
            violations.append(
                f"custody: truck {k} ends with pending retrievals at "
                f"{sorted(pending)}")
        away = [r for (kk, r), ab in aboard.items()
                if kk == k and not ab]
        if away:
            violations.append(
                f"custody: robots {away} of truck {k} not retrieved")
        if load > cp["beta_truck"] + 1e-9:
            violations.append(
                f"capacity: truck {k} load {load} > {cp['beta_truck']}")
        if tdist > cp["phi_truck"] + 1e-6:
            violations.append(
                f"range: truck {k} distance {tdist:.2f} > "
                f"{cp['phi_truck']}")

    for (k, r), dd in robot_dist.items():
        if dd > cp["phi_hat"] + 1e-6:
            violations.append(
                f"range: robot {k}-{r} distance {dd:.2f} > "
                f"{cp['phi_hat']}")

    obj = (trucks_used * cp["gamma_fixed"]
           + len(robots_used) * cp["gammahat_fixed"]
           + truck_travel + robot_travel
           + cp["gamma_late"] * sum(lateness.values()))
    return (not violations), violations, obj
