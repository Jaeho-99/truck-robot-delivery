"""Solution representation and evaluator.

Solution representation (launch and retrieval may differ):
  truck_routes: dict k -> list of stops
    stop = {"kind": "cust", "c": c}
    stop = {"kind": "park", "p": copy_node, "deploys": [trip, ...]}
    trip = {"r": r, "custs": [c, ...], "ret_p": copy_node'}
           (ret_p != p; a later parking stop on the same truck route)

  * A delivery trip is attached to its deploy parking stop; retrieval
    happens at a *later* parking stop (ret_p) of the same truck route.
    The truck does not wait after deploying, so it works in parallel
    with the robot — the same flexibility as the exact model.
  * Following the MILP custody constraints (24)(25), one copy cannot
    host a deploy and a retrieve simultaneously, so a "waiting-style"
    trip at the same physical location also consumes two copies
    (deploy at copy 1, retrieve at copy 2, zero distance in between) —
    exactly the copy budget of the exact model.
  * Custody sequencing is validated by tracking the aboard state in the
    evaluator: deploy only while aboard, retrieve only while away, all
    robots aboard at route end (constraints (19)-(25)).
  * Robot range (51): total accumulated distance <= phi-hat (no
    swapping). Trip capacity: number of customers <= beta-hat (lambda
    = 1).
  * Soft time windows (e_c = 0): lateness is an objective penalty.
    Forward scheduling matches MILP constraints (26)-(34).
"""

import copy
from collections import defaultdict


class Solution:
    def __init__(self, K):
        self.routes = {k: [] for k in K}

    def clone(self):
        s = Solution(self.routes.keys())
        s.routes = copy.deepcopy(self.routes)
        return s

    def customers(self):
        """Set of customers placed in the solution."""
        out = set()
        for route in self.routes.values():
            for st in route:
                if st["kind"] == "cust":
                    out.add(st["c"])
                else:
                    for tr in st["deploys"]:
                        out.update(tr["custs"])
        return out

    def used_copies(self):
        out = set()
        for route in self.routes.values():
            for st in route:
                if st["kind"] == "park":
                    out.add(st["p"])
        return out


def eval_truck(pr, k, route):
    """Evaluate one route of truck k.

    Returns (cost, feasible, lateness {c: min}, robot_dist {r: km}).

    Forward schedule = MILP (26)-(34):
      deploy stop:   robot departs at a + zeta_unload; the truck leaves
                     without waiting (32)(33)
      retrieve stop: truck departure >= robot arrival + zeta_load (34)
    Custody (19)-(25) via aboard-state tracking: deploy only while
    aboard, retrieve only while away; at one stop deploys are processed
    before retrieves (state on arrival = w_kri); all robots aboard and
    no pending retrievals at route end.
    """
    if not route:
        return 0.0, True, {}, {}

    D = pr.D
    truck_travel = 0.0
    robot_travel = 0.0
    truck_dist = 0.0
    lateness = {}
    robot_dist = defaultdict(float)   # accumulated robot distance (51)
    parcels = 0
    feasible = True
    aboard = {r: True for r in pr.R_k[k]}
    pending = {}            # ret_p copy -> [(r, robot arrival), ...]
    robots_used = set()

    prev, b_prev = 0, 0.0             # leave depot-out at minute 0
    for st in route:
        node = st["c"] if st["kind"] == "cust" else st["p"]
        a_node = b_prev + pr.tau_truck(prev, node)
        truck_travel += pr.tau_truck(prev, node) * pr.truck_arc_coef
        truck_dist += pr.dist(prev, node)

        if st["kind"] == "cust":
            c = st["c"]
            parcels += pr.lam[c]
            lateness[c] = max(0.0, a_node - pr.l_c[c])
            b_node = a_node + pr.s_kc
        else:
            p = st["p"]
            b_node = a_node
            # --- deploys first (custody state on arrival: (24) needs
            #     w = 1) ---
            for tr in st["deploys"]:
                r, custs, ret_p = tr["r"], tr["custs"], tr["ret_p"]
                if not aboard.get(r, False):   # no re-deploy while away
                    feasible = False
                aboard[r] = False
                robots_used.add(r)
                # empty trip / capacity (44)
                if not custs or len(custs) > pr.beta_robot:
                    feasible = False
                if ret_p == p:      # same-copy retrieval banned (24)(25)
                    feasible = False
                parcels += sum(pr.lam[c] for c in custs)
                t = a_node + pr.zeta_unload    # (32) deploy departure
                rprev = p
                for c in custs:
                    ahat_c = t + pr.tau_robot(rprev, c)
                    robot_travel += (pr.tau_robot(rprev, c)
                                     * pr.robot_arc_coef)
                    robot_dist[r] += pr.dist(rprev, c)
                    lateness[c] = max(0.0, ahat_c - pr.l_c[c])
                    t = ahat_c + pr.s_hat
                    rprev = c
                arr_ret = t + pr.tau_robot(rprev, ret_p)   # at retrieval
                robot_travel += (pr.tau_robot(rprev, ret_p)
                                 * pr.robot_arc_coef)
                robot_dist[r] += pr.dist(rprev, ret_p)
                pending.setdefault(ret_p, []).append((r, arr_ret))
            if st["deploys"]:
                # (33) the truck cannot leave before the robot departure
                # (a + zeta_unload).
                b_node = max(b_node, a_node + pr.zeta_unload)
            # --- retrieves at this stop ((25) needs w = 0) ---
            for (r, arr) in pending.pop(p, []):
                if aboard.get(r, False):
                    feasible = False
                aboard[r] = True
                # (34) truck waits for the robot
                b_node = max(b_node, arr + pr.zeta_load)
        prev, b_prev = node, b_node

    # return to depot
    truck_travel += pr.tau_truck(prev, D) * pr.truck_arc_coef
    truck_dist += pr.dist(prev, D)

    # custody closure (19): no pending retrievals / robots left away
    if pending or not all(aboard.values()):
        feasible = False
    # driving ranges (50)(51), truck capacity (39)
    if truck_dist > pr.phi_truck + 1e-6:
        feasible = False
    for r, dd in robot_dist.items():
        if dd > pr.phi_hat + 1e-6:
            feasible = False
    if parcels > pr.beta_truck + 1e-6:
        feasible = False

    cost = (pr.gamma_fixed                                # truck fixed
            + len(robots_used) * pr.gammahat_fixed        # robot fixed
            + truck_travel + robot_travel                 # travel
            + pr.gamma_late * sum(lateness.values()))     # lateness
    return cost, feasible, lateness, dict(robot_dist)


def eval_solution(pr, sol):
    """Total cost, feasibility and per-component breakdown.

    Coverage (18): a solution missing any customer is infeasible — this
    prevents a (cheaper) solution with dropped customers from becoming
    the incumbent after a failed repair.
    """
    total = 0.0
    feasible = sol.customers() == set(pr.C)
    brk = dict(truck_fixed=0.0, robot_fixed=0.0, truck_travel=0.0,
               robot_travel=0.0, lateness=0.0)
    per_truck = {}
    for k, route in sol.routes.items():
        c, ok, late, _ = eval_truck(pr, k, route)
        per_truck[k] = c
        total += c
        feasible = feasible and ok
        if route:
            brk["truck_fixed"] += pr.gamma_fixed
            robots_used = {tr["r"] for st in route
                           if st["kind"] == "park"
                           for tr in st["deploys"]}
            brk["robot_fixed"] += len(robots_used) * pr.gammahat_fixed
            brk["lateness"] += pr.gamma_late * sum(late.values())
    # Travel-cost breakdown (recomputed so it matches the total).
    for k, route in sol.routes.items():
        tt = rt = 0.0
        prev = 0
        for st in route:
            node = st["c"] if st["kind"] == "cust" else st["p"]
            tt += pr.tau_truck(prev, node) * pr.truck_arc_coef
            if st["kind"] == "park":
                for tr in st["deploys"]:
                    rprev = st["p"]
                    for c in tr["custs"]:
                        rt += pr.tau_robot(rprev, c) * pr.robot_arc_coef
                        rprev = c
                    rt += (pr.tau_robot(rprev, tr["ret_p"])
                           * pr.robot_arc_coef)
            prev = node
        if route:
            tt += pr.tau_truck(prev, pr.D) * pr.truck_arc_coef
        brk["truck_travel"] += tt
        brk["robot_travel"] += rt
    return total, feasible, brk, per_truck
