"""Internal route scoring for the v2 operators.

The public evaluators in the original implementations still produce final
objectives and breakdowns.  This context computes their cost and feasibility
for insertion candidates using Python float lookups, without constructing the
lateness/robot-distance breakdown returned to callers by ``eval_truck``.
Contexts belong to one immutable problem and have no module-level cache.
"""


class CostContext:
    """Precompute fast scalar matrix lookups for one immutable ALNS problem.

    ``evaluate(k, route)`` returns ``(cost, feasible)``.  Cost remains valid
    even when the route is infeasible, because destroy/repair also scores
    incomplete base routes.  Accumulation order and duplicate-customer
    behavior intentionally match the original ``eval_truck`` evaluator.
    """

    def __init__(self, pr):
        # NumPy scalar indexing plus float conversion is expensive in this
        # inner loop.  Conversion happens once per context, outside scoring.
        self.truck_times = pr.tau_truck_matrix.tolist()
        self.robot_times = pr.tau_robot_matrix.tolist()
        self.distances = pr.d.tolist()
        self.D = pr.D
        self.R_k = pr.R_k
        self.lam = pr.lam
        self.l_c = pr.l_c
        self.s_kc = pr.s_kc
        self.s_hat = pr.s_hat
        self.zeta_load = pr.zeta_load
        self.zeta_unload = pr.zeta_unload
        self.beta_robot = pr.beta_robot
        self.beta_truck = pr.beta_truck
        self.phi_truck = pr.phi_truck
        self.phi_hat = pr.phi_hat
        self.gamma_fixed = pr.gamma_fixed
        self.gammahat_fixed = pr.gammahat_fixed
        self.gamma_late = pr.gamma_late
        self.truck_arc_coef = pr.truck_arc_coef
        self.robot_arc_coef = pr.robot_arc_coef

    def evaluate(self, k, route, *, reject_infeasible=False):
        """Score a route, retaining all original scheduling constraints.

        Candidate scans may discard a provably infeasible route immediately;
        its returned cost is then unspecified. Base/removal scoring leaves
        this off because the exact cost of an infeasible route can matter.
        """
        if not route:
            return 0.0, True

        truck_times = self.truck_times
        robot_times = self.robot_times
        distances = self.distances
        lam = self.lam
        l_c = self.l_c
        s_kc = self.s_kc
        s_hat = self.s_hat
        zeta_load = self.zeta_load
        zeta_unload = self.zeta_unload
        beta_robot = self.beta_robot
        truck_arc_coef = self.truck_arc_coef
        robot_arc_coef = self.robot_arc_coef

        truck_travel = 0.0
        robot_travel = 0.0
        truck_dist = 0.0
        # Retain dictionaries so an invalid duplicate visit has exactly the
        # original overwrite and insertion-order behavior in the objective.
        lateness = {}
        robot_dist = {}
        parcels = 0
        feasible = True
        aboard = {r: True for r in self.R_k[k]}
        pending = {}
        robots_used = set()
        prev, b_prev = 0, 0.0

        for st in route:
            is_customer = st["kind"] == "cust"
            node = st["c"] if is_customer else st["p"]
            travel = truck_times[prev][node]
            a_node = b_prev + travel
            truck_travel += travel * truck_arc_coef
            truck_dist += distances[prev][node]

            if is_customer:
                parcels += lam[node]
                lateness[node] = max(0.0, a_node - l_c[node])
                b_node = a_node + s_kc
            else:
                p = node
                b_node = a_node
                deploys = st["deploys"]
                # Deploy precedes retrieval at the same stop, so a robot
                # arriving at this stop cannot be redeployed here.
                for tr in deploys:
                    r = tr["r"]
                    custs = tr["custs"]
                    ret_p = tr["ret_p"]
                    if not aboard.get(r, False):
                        if reject_infeasible:
                            return 0.0, False
                        feasible = False
                    aboard[r] = False
                    robots_used.add(r)
                    if not custs or len(custs) > beta_robot:
                        if reject_infeasible:
                            return 0.0, False
                        feasible = False
                    if ret_p == p:
                        if reject_infeasible:
                            return 0.0, False
                        feasible = False
                    parcels += sum(lam[c] for c in custs)
                    t = a_node + zeta_unload
                    rprev = p
                    for c in custs:
                        travel = robot_times[rprev][c]
                        ahat_c = t + travel
                        robot_travel += travel * robot_arc_coef
                        robot_dist[r] = (robot_dist.get(r, 0.0)
                                         + distances[rprev][c])
                        lateness[c] = max(0.0, ahat_c - l_c[c])
                        t = ahat_c + s_hat
                        rprev = c
                    travel = robot_times[rprev][ret_p]
                    arr_ret = t + travel
                    robot_travel += travel * robot_arc_coef
                    robot_dist[r] = (robot_dist.get(r, 0.0)
                                     + distances[rprev][ret_p])
                    pending.setdefault(ret_p, []).append((r, arr_ret))
                if deploys:
                    b_node = max(b_node, a_node + zeta_unload)
                for r, arr in pending.pop(p, []):
                    if aboard.get(r, False):
                        if reject_infeasible:
                            return 0.0, False
                        feasible = False
                    aboard[r] = True
                    b_node = max(b_node, arr + zeta_load)
            prev, b_prev = node, b_node

        truck_travel += truck_times[prev][self.D] * truck_arc_coef
        truck_dist += distances[prev][self.D]
        if pending or not all(aboard.values()):
            feasible = False
        if truck_dist > self.phi_truck + 1e-6:
            feasible = False
        for dd in robot_dist.values():
            if dd > self.phi_hat + 1e-6:
                feasible = False
        if parcels > self.beta_truck + 1e-6:
            feasible = False

        cost = (self.gamma_fixed
                + len(robots_used) * self.gammahat_fixed
                + truck_travel + robot_travel
                + self.gamma_late * sum(lateness.values()))
        return cost, feasible
