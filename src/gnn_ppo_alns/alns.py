"""Combinatorial search mechanics for GNN-PPO-ALNS.

This module deliberately contains no standalone operator selector or search
driver. The graph-conditioned PPO actor in :mod:`gnn_ppo_alns.ppo` selects
one of the nine joint destroy/repair actions and calls
:func:`apply_actor_action` for exactly one search transition. Geometry and
travel coefficients are loaded from the preprocessed instance and are never
reconstructed here; node coordinates are retained only for graph features.
"""

import copy
from collections import defaultdict
import math
from pathlib import Path
import random
import re

import numpy as np

from common.params import Instance as ProcessedInstance
from common.params import Params as SharedParams
from common.params import REPO_ROOT, load_params, load_problem


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
__all__ = [
    "ACTION_COUNT", "ACTION_LABELS", "DOD", "NOISE_FRAC", "W_START",
    "DirectoryInstanceProvider", "Params", "Solution",
    "apply_actor_action", "congestion_aware_initial", "eval_solution",
]

# ---- caps on trip-insertion candidates (combinatorial control) ----
L_RET_EXIST = 3   # existing later parking stops tried as retrieval
W_RET_NEW = 4     # positions after the deploy tried for a new stop
N_PHYS_NEAR = 2   # nearest physical locations tried for a new stop


# ============================================================
# Insertion-candidate enumeration (modes A/B/C/D — shared by
# greedy and regret repair)
# ============================================================
def _with_new_trip(route, si, st, trip):
    """Copy-on-write: route with ``trip`` appended at stop index si."""
    new_st = dict(st)
    new_st["deploys"] = st["deploys"] + [trip]
    nr = list(route)
    nr[si] = new_st
    return nr


def enum_insertions(pr, sol, c):
    """Yield every insertion candidate (k, new_route) for customer c.

      A. direct truck visit
      B. insertion into the customer chain of an existing trip
      C. new trip deployed at an existing parking stop (retrieval at a
         later existing stop / a new copy at the same location / a new
         copy at a nearby location)
      D. new deploy stop plus new trip (retrieval at the second copy of
         the same location, waiting style / at a later existing stop)

    Feasibility (custody, range, capacity) is judged by eval_truck, so
    only the structures are generated here.

    Candidates are built copy-on-write: only containers on the
    modified path are fresh objects, untouched stops/trips are SHARED
    with sol's route and must never be mutated in place (eval_truck
    only reads; apply_insertion swaps the whole route list; in-place
    edits happen only on Solution.clone() deep copies). This replaces
    the per-candidate route deepcopy that dominated ALNS runtime
    (~65% in profiling).
    """
    used = sol.used_copies()
    for k in pr.K:
        route = sol.routes[k]

        # --- A. direct truck visit ---
        for pos in range(len(route) + 1):
            yield k, (route[:pos] + [{"kind": "cust", "c": c}]
                      + route[pos:])

        park_pos = [(si, st) for si, st in enumerate(route)
                    if st["kind"] == "park"]

        # --- B. insertion into an existing trip ---
        for si, st in park_pos:
            for ti, tr in enumerate(st["deploys"]):
                if len(tr["custs"]) >= pr.beta_robot:
                    continue
                for pos in range(len(tr["custs"]) + 1):
                    new_tr = dict(tr)
                    new_tr["custs"] = (tr["custs"][:pos] + [c]
                                       + tr["custs"][pos:])
                    new_st = dict(st)
                    new_st["deploys"] = list(st["deploys"])
                    new_st["deploys"][ti] = new_tr
                    nr = list(route)
                    nr[si] = new_st
                    yield k, nr

        # --- C. deploy at an existing parking stop (all robots tried —
        #        robots are asymmetric) ---
        for si, st in park_pos:
            for r in pr.R_k[k]:
                # ret 1) later existing parking stops (L_RET_EXIST max)
                for sj, st2 in [pp for pp in park_pos
                                if pp[0] > si][:L_RET_EXIST]:
                    yield k, _with_new_trip(
                        route, si, st,
                        {"r": r, "custs": [c], "ret_p": st2["p"]})
                # ret 2) fresh copy at the same location right behind
                #        (waiting style, consumes two copies)
                grp = pr.park_groups[pr.copy_to_phys[st["p"]]]
                free = [cp for cp in grp if cp not in used]
                if free:
                    nr = _with_new_trip(
                        route, si, st,
                        {"r": r, "custs": [c], "ret_p": free[0]})
                    nr.insert(si + 1, {"kind": "park", "p": free[0],
                                       "deploys": []})
                    yield k, nr
                # ret 3) fresh copy at a location near c, inserted
                #        within W positions after the deploy
                for gi in pr.phys_near[c][:N_PHYS_NEAR]:
                    grp2 = pr.park_groups[gi]
                    free2 = [cp for cp in grp2
                             if cp not in used and cp != st["p"]]
                    if not free2:
                        continue
                    for pos in range(si + 1,
                                     min(len(route), si + W_RET_NEW) + 1):
                        nr = _with_new_trip(
                            route, si, st,
                            {"r": r, "custs": [c], "ret_p": free2[0]})
                        nr.insert(pos, {"kind": "park", "p": free2[0],
                                        "deploys": []})
                        yield k, nr

        # --- D. new deploy stop plus new trip ---
        for gi in pr.phys_near[c][:N_PHYS_NEAR]:
            grp = pr.park_groups[gi]
            free = [cp for cp in grp if cp not in used]
            if not free:
                continue
            dep_cp = free[0]
            for r in pr.R_k[k]:
                for pos in range(len(route) + 1):
                    # ret a) second copy of the same location (waiting)
                    if len(free) >= 2:
                        nr = list(route)
                        nr.insert(pos, {"kind": "park", "p": dep_cp,
                                        "deploys": [{"r": r,
                                                     "custs": [c],
                                                     "ret_p": free[1]}]})
                        nr.insert(pos + 1, {"kind": "park", "p": free[1],
                                            "deploys": []})
                        yield k, nr
                    # ret b) later existing parking stops
                    later = [st2 for si2, st2 in park_pos
                             if si2 >= pos][:L_RET_EXIST]
                    for st2 in later:
                        nr = list(route)
                        nr.insert(pos, {"kind": "park", "p": dep_cp,
                                        "deploys": [{"r": r,
                                                     "custs": [c],
                                                     "ret_p": st2["p"]}]})
                        yield k, nr


def best_insertion(pr, sol, c, rng=None, noise=0.0):
    """Cheapest insertion of customer c.

    Returns (delta, (k, new_route)) or (inf, None). With noise > 0 a
    U(-noise, noise) perturbation is added to each candidate's delta
    (Ropke & Pisinger noise insertion) — locally poor insertions such
    as the first customer of a new robot trip are then chosen
    occasionally, allowing escapes from truck-only local optima.
    """
    base = {k: eval_truck(pr, k, sol.routes[k])[0] for k in pr.K}
    best_delta, best_apply = math.inf, None
    for k, nr in enum_insertions(pr, sol, c):
        cost, ok, _, _ = eval_truck(pr, k, nr)
        if not ok:
            continue
        delta = cost - base[k]
        if noise > 0.0 and rng is not None:
            delta += rng.uniform(-noise, noise)
        if delta < best_delta - 1e-9:
            best_delta, best_apply = delta, (k, nr)
    return best_delta, best_apply


def apply_insertion(sol, apply_tuple):
    k, new_route = apply_tuple
    sol.routes[k] = new_route


# ============================================================
# Repair operators
# ============================================================
def repair_greedy(pr, sol, pool, rng, noise=0.0):
    remaining = list(pool)
    while remaining:
        best = None
        for c in remaining:
            delta, ap = best_insertion(pr, sol, c, rng=rng, noise=noise)
            if ap is not None and (best is None or delta < best[0] - 1e-9):
                best = (delta, ap, c)
        if best is None:
            return False
        apply_insertion(sol, best[1])
        remaining.remove(best[2])
    return True


def repair_regret2(pr, sol, pool, rng):
    """Regret-2: insert first the customer whose gap between its best
    and second-best per-truck insertion delta is largest."""
    remaining = list(pool)
    while remaining:
        pick = None       # (regret, delta, apply, c)
        for c in remaining:
            base = {k: eval_truck(pr, k, sol.routes[k])[0] for k in pr.K}
            local = {}    # k -> (delta, apply)
            for k, nr in enum_insertions(pr, sol, c):
                cost, ok, _, _ = eval_truck(pr, k, nr)
                if not ok:
                    continue
                delta = cost - base[k]
                if k not in local or delta < local[k][0] - 1e-9:
                    local[k] = (delta, (k, nr))
            if not local:
                continue
            deltas = sorted(local.values(), key=lambda t: t[0])
            best_d = deltas[0][0]
            second = deltas[1][0] if len(deltas) > 1 else best_d + 1e6
            regret = second - best_d
            if pick is None or regret > pick[0] + 1e-9:
                pick = (regret, best_d, deltas[0][1], c)
        if pick is None:
            return False
        apply_insertion(sol, pick[2])
        remaining.remove(pick[3])
    return True


# ============================================================
# Destroy operators
# ============================================================
def remove_customers(sol, custs):
    """Remove the given customers, drop emptied trips, and drop parking
    stops that no longer host a deploy nor are referenced as a
    retrieval."""
    cset = set(custs)
    for k, route in sol.routes.items():
        # 1) remove customers, drop empty trips
        mid = []
        for st in route:
            if st["kind"] == "cust":
                if st["c"] in cset:
                    continue
                mid.append(st)
            else:
                st["deploys"] = [
                    dict(tr, custs=[c for c in tr["custs"]
                                    if c not in cset])
                    for tr in st["deploys"]]
                st["deploys"] = [tr for tr in st["deploys"]
                                 if tr["custs"]]
                mid.append(st)
        # 2) collect retrieval references of surviving trips, then drop
        #    parking stops without a role
        refs = {tr["ret_p"] for st in mid if st["kind"] == "park"
                for tr in st["deploys"]}
        sol.routes[k] = [st for st in mid
                         if st["kind"] == "cust"
                         or st["deploys"] or st["p"] in refs]


def destroy_random(pr, sol, q, rng):
    custs = list(sol.customers())
    q = min(q, len(custs))
    chosen = rng.sample(custs, q)
    remove_customers(sol, chosen)
    return chosen


def destroy_worst(pr, sol, q, rng, p=3.0):
    """Prefer customers with the largest removal gain (current cost
    minus cost after removal), randomized by exponent p."""
    base, _, _, _ = eval_solution(pr, sol)
    contrib = []
    for c in sol.customers():
        tmp = sol.clone()
        remove_customers(tmp, [c])
        after, _, _, _ = eval_solution(pr, tmp)
        contrib.append((base - after, c))   # larger = worse placed
    contrib.sort(reverse=True)
    chosen = []
    pool = contrib[:]
    while pool and len(chosen) < q:
        y = rng.random()
        idx = int((y ** p) * len(pool))
        chosen.append(pool.pop(idx)[1])
    remove_customers(sol, chosen)
    return chosen


def destroy_related(pr, sol, q, rng, p=6.0):
    """Shaw-style: remove customers close in distance and time window."""
    custs = list(sol.customers())
    if not custs:
        return []
    q = min(q, len(custs))
    seed = rng.choice(custs)
    removed = [seed]

    def relatedness(a, b):
        d = pr.dist(a, b)
        t = abs(pr.l_c[a] - pr.l_c[b])
        return d + 0.1 * t

    while len(removed) < q:
        ref = rng.choice(removed)
        cand = [c for c in custs if c not in removed]
        cand.sort(key=lambda c: relatedness(ref, c))
        y = rng.random()
        idx = int((y ** p) * len(cand))
        removed.append(cand[idx])
    remove_customers(sol, removed)
    return removed


DESTROY_OPERATORS = (
    ("random", destroy_random),
    ("worst", destroy_worst),
    ("related", destroy_related),
)
REPAIR_NAMES = ("greedy", "greedy_noise", "regret2")
ACTION_LABELS = tuple(
    f"{destroy_name}+{repair_name}"
    for destroy_name, _ in DESTROY_OPERATORS
    for repair_name in REPAIR_NAMES
)
ACTION_COUNT = len(ACTION_LABELS)


def apply_actor_action(pr, current_solution, action, q_destroy, rng,
                       noise_amplitude):
    """Apply one graph-conditioned actor-selected destroy/repair pair.

    ``action = destroy_index * 3 + repair_index`` is the PPO action-space
    contract. Returns ``(candidate, objective, feasible)``; acceptance and
    reward remain the responsibility of the GNN-PPO environment.
    """
    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError(f"actor action must be an integer, got {action!r}")
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(
            f"actor action must be in [0, {ACTION_COUNT - 1}], got {action}")
    if q_destroy <= 0:
        raise ValueError("q_destroy must be positive")
    destroy_index, repair_index = divmod(action, len(REPAIR_NAMES))
    candidate = current_solution.clone()
    pool = DESTROY_OPERATORS[destroy_index][1](
        pr, candidate, q_destroy, rng)
    if repair_index == 0:
        repair_greedy(pr, candidate, pool, rng, noise=0.0)
    elif repair_index == 1:
        repair_greedy(pr, candidate, pool, rng, noise=noise_amplitude)
    else:
        repair_regret2(pr, candidate, pool, rng)
    objective, feasible, _, _ = eval_solution(pr, candidate)
    return candidate, objective, feasible


# Robot-candidate threshold theta: only customers with a normalized
# score > 0 are considered by the congestion-aware initial solution.
INIT_SCORE_MIN = 1e-9

# Actor action ``greedy_noise`` uses initial objective x NOISE_FRAC.
NOISE_FRAC = 0.25

# SA start-temperature fraction (DR-ALNS / Roozbeh rule of thumb): a
# solution 5% worse than the initial one is accepted with probability
# 0.5 at T_start = W_START * f_init / ln 2 in the PPO environment.
W_START = 0.05

# A fixed 30% of customers is removed per actor-selected transition.
DOD = 0.3


# ============================================================
# 1. Parameter container
# ============================================================
class Params:
    """ALNS view of one processed instance.

    The processed file has one depot, while the route formulation uses a
    departure and an arrival depot. ``source_index`` duplicates that depot
    and reorders all three precomputed matrices once. No geometry or travel
    time is calculated by the search algorithm.
    """

    def __init__(self, data: ProcessedInstance, config: SharedParams,
                 size: int):
        n_customers = int(data.customer_idx.size)
        if n_customers != size:
            raise ValueError(
                f"requested n{size}, but {data.source_path.name} contains "
                f"{n_customers} customers")

        source_index = np.asarray([
            data.depot_index, *data.customer_idx.tolist(),
            *data.parking_idx.tolist(), data.depot_index,
        ], dtype=np.int64)
        self.source_index = source_index
        self.nodes = np.ascontiguousarray(data.node_xy[source_index])
        self.node_zone = np.ascontiguousarray(data.node_zone[source_index])
        self.labels = tuple(data.node_label[source_index].tolist())
        self.d = np.ascontiguousarray(data.d[np.ix_(source_index, source_index)])
        self.tau_truck_matrix = np.ascontiguousarray(
            data.tau_truck[np.ix_(source_index, source_index)])
        self.tau_robot_matrix = np.ascontiguousarray(
            data.tau_robot[np.ix_(source_index, source_index)])
        for array in (self.nodes, self.node_zone, self.d,
                      self.tau_truck_matrix, self.tau_robot_matrix):
            array.setflags(write=False)

        self.C = list(range(1, n_customers + 1))
        first_parking = n_customers + 1
        self.P = list(range(first_parking,
                            first_parking + int(data.parking_idx.size)))
        self.D = len(source_index) - 1
        self.K = list(range(1, config.fleet.n_trucks_for(size) + 1))
        robots = config.fleet.n_robots_per_truck
        self.R_k = {k: list(range(1, robots + 1)) for k in self.K}
        self.lam = {c: int(value) for c, value in
                    zip(self.C, data.demand, strict=True)}
        self.e_c = {c: float(value) for c, value in
                    zip(self.C, data.e, strict=True)}
        self.l_c = {c: float(value) for c, value in
                    zip(self.C, data.l, strict=True)}
        self.alpha_traffic = data.alpha_traffic
        self.alpha_ped = data.alpha_ped
        self.instance_id = str(data.meta["instance_id"])
        self.preproc_hash = str(data.meta["preproc_hash"])

        # Physical parking groups (copy-budget bookkeeping).
        groups = defaultdict(list)
        for p in self.P:
            groups[self.labels[p].rsplit("#", 1)[0]].append(p)
        self.park_groups = [groups[key] for key in sorted(
            groups, key=lambda label: int(label[1:]))]
        self.copy_to_phys = {}          # copy node -> group index
        for gi, grp in enumerate(self.park_groups):
            for cp in grp:
                self.copy_to_phys[cp] = gi
        # Per-customer ranking of physical locations by proximity
        # (candidates for new parking stops).
        self.phys_near = {}
        for c in self.C:
            order = sorted(
                range(len(self.park_groups)),
                key=lambda gi: self.dist(c, self.park_groups[gi][0]))
            self.phys_near[c] = order

        self.s_kc = config.truck.service_time_min
        self.s_hat = config.robot.service_time_min
        self.zeta_load = config.robot.load_time_min
        self.zeta_unload = config.robot.unload_time_min
        self.beta_truck = config.truck.capacity
        self.beta_robot = config.robot.capacity
        self.phi_hat = config.robot.range_km
        self.phi_truck = config.truck.range_km
        self.gamma_late = config.lateness_cost_per_min
        self.gamma_fixed = config.truck.fixed_cost
        self.gammahat_fixed = config.robot.fixed_cost
        self.truck_arc_coef = (config.truck.fuel_cost_per_min
                               + config.truck.env_cost_per_min)
        self.robot_arc_coef = (config.robot.fuel_cost_per_min
                               + config.robot.env_cost_per_min)

    def dist(self, i, j):
        return float(self.d[i, j])

    def tau_truck(self, i, j):
        return float(self.tau_truck_matrix[i, j])

    def tau_robot(self, i, j):
        return float(self.tau_robot_matrix[i, j])


# ============================================================
# 2. Initial solutions
# ============================================================
def truck_only_initial(pr, rng):
    """Greedy initial solution using direct truck dispatch only."""
    sol = Solution(pr.K)
    ok = repair_greedy(pr, sol, list(pr.C), rng)
    if not ok:
        raise RuntimeError("initial solution failed — check truck "
                           "capacity/range")
    return sol


def _congestion_score(pr):
    """Per-customer normalized congestion score = (normalized traffic
    congestion) - (normalized pedestrian congestion).

    The differing scales of alpha_z in [1, 1+eps] and alpha-hat_z in
    [1, 1+eps-hat] are removed by min-max normalization over the
    observed zone range. score > 0 iff the zone is relatively worse for
    trucks than for robots.
    """
    zc = pr.node_zone
    ta, pa = pr.alpha_traffic, pr.alpha_ped
    tmin = float(ta.min())
    tspan = max(float(ta.max() - ta.min()), 1e-12)
    pmin = float(pa.min())
    pspan = max(float(pa.max() - pa.min()), 1e-12)
    return {c: (ta[zc[c]] - tmin) / tspan - (pa[zc[c]] - pmin) / pspan
            for c in pr.C}


def _build_trips(pr, gi, custs, leftovers):
    """Split the customers assigned to physical parking gi into
    nearest-neighbor trips under the beta-hat capacity and phi-hat
    battery limits.

    Returns [(custs, trip_distance), ...]; customers that cannot form a
    trip go to ``leftovers``.
    """
    grp = pr.park_groups[gi]
    p0, p1 = grp[0], grp[1]
    rem = list(custs)
    trips = []
    while rem:
        trip, last, ddist = [], p0, 0.0
        while rem and len(trip) < pr.beta_robot:
            nxt = min(rem, key=lambda c: pr.tau_robot(last, c))
            if (ddist + pr.dist(last, nxt) + pr.dist(nxt, p1)
                    > pr.phi_hat + 1e-9):
                break
            ddist += pr.dist(last, nxt)
            trip.append(nxt)
            rem.remove(nxt)
            last = nxt
        if not trip:            # defensive guard: even a solo round
            leftovers.append(rem.pop(0))    # trip is impossible
            continue
        ddist += pr.dist(last, p1)
        trips.append((trip, ddist))
    return trips


def congestion_aware_initial(pr, rng):
    """Congestion-aware initial solution (two-stage constructive).

    Stage 1 — robot-customer selection: sort customers by the
      normalized (traffic - pedestrian) congestion score of their zone,
      descending (ties broken by distance to the nearest parking).
      Customers with score > 0 whose round trip from the nearest
      physical parking fits the battery limit (phi-hat) become robot
      candidates, assigned to that parking.
    Stage 2 — route construction: cluster each parking's candidates
      into nearest-neighbor delivery trips of size <= beta-hat, assign
      parkings to trucks round-robin in order of depot proximity, and
      build waiting-style stop pairs (deploy at copy 1 -> retrieve at
      copy 2) as a skeleton while tracking each robot's phi-hat budget.
      Remaining customers are inserted by greedy repair (which may also
      join existing trips — insertion mode B).

    Falls back to the truck-only greedy on failure or infeasibility
    (never crashes).
    """
    scores = _congestion_score(pr)

    # ---- stage 1: candidate selection + nearest feasible parking ----
    cand = []
    for c in pr.C:
        if scores[c] <= INIT_SCORE_MIN:
            continue
        for gi in pr.phys_near[c]:
            grp = pr.park_groups[gi]
            # a waiting-style stop pair needs two copies, plus
            # round-trip battery feasibility
            if (len(grp) >= 2
                    and 2.0 * pr.dist(grp[0], c) <= pr.phi_hat + 1e-9):
                cand.append((c, gi))
                break
    cand.sort(key=lambda t: (-scores[t[0]],
                             pr.dist(pr.park_groups[t[1]][0], t[0])))

    by_group = defaultdict(list)
    for c, gi in cand:
        by_group[gi].append(c)

    # ---- stage 2: trip clustering -> truck/robot assignment ->
    #      skeleton ----
    leftovers = []
    robot_budget = {(k, r): pr.phi_hat for k in pr.K for r in pr.R_k[k]}
    pairs = {k: [] for k in pr.K}
    order = sorted(by_group,
                   key=lambda gi: pr.tau_truck(0, pr.park_groups[gi][0]))
    for ti, gi in enumerate(order):
        grp = pr.park_groups[gi]
        k = pr.K[ti % len(pr.K)]
        deploys = []
        for custs, ddist in _build_trips(pr, gi, by_group[gi], leftovers):
            # no duplicate robot within one stop pair (custody: no
            # re-deploy while away)
            r = next((r for r in pr.R_k[k]
                      if robot_budget[(k, r)] >= ddist - 1e-9
                      and all(d["r"] != r for d in deploys)), None)
            if r is None:
                leftovers.extend(custs)
                continue
            robot_budget[(k, r)] -= ddist
            deploys.append({"r": r, "custs": custs, "ret_p": grp[1]})
        if deploys:
            pairs[k].append((
                {"kind": "park", "p": grp[0], "deploys": deploys},
                {"kind": "park", "p": grp[1], "deploys": []}))

    sol = Solution(pr.K)
    for k in pr.K:
        sol.routes[k] = [st for pair in pairs[k] for st in pair]
        _, ok, _, _ = eval_truck(pr, k, sol.routes[k])
        if not ok:      # defensive: dissolve skeleton into truck pool
            for pair in pairs[k]:
                for tr in pair[0]["deploys"]:
                    leftovers.extend(tr["custs"])
            sol.routes[k] = []

    # ---- greedy insertion of remaining customers (non-candidates +
    #      leftovers) ----
    pool = [c for c in pr.C if c not in sol.customers()]
    if not repair_greedy(pr, sol, pool, rng):
        return truck_only_initial(pr, rng)
    _, feas, _, _ = eval_solution(pr, sol)
    if not feas:
        return truck_only_initial(pr, rng)
    return sol


# ============================================================
# 3. Processed-instance provider for GNN-PPO training and testing
# ============================================================


_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class DirectoryInstanceProvider:
    """Load validated NPZ files and expose immutable ALNS problems."""

    def __init__(self, size, params_path=None, tag=None, seed=0,
                 train_count=None, split="all"):
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be 'train', 'test', or 'all'")
        if tag is not None and not _TAG_PATTERN.fullmatch(tag):
            raise ValueError("invalid tag; use letters, digits, '.', '_', or '-'")
        self.size = size
        self.tag = tag
        self.params_path = params_path
        self.config = load_params(params_path)
        self.fleet = {
            "n_trucks": self.config.fleet.n_trucks_for(size),
            "n_robots_per_truck": self.config.fleet.n_robots_per_truck,
        }
        self.rng = random.Random(seed)
        data_dir = "processed" if tag is None else f"processed_{tag}"
        self.data_root = REPO_ROOT / "data" / data_dir
        self._cache = {}
        self._train_all = (self._load("train")
                           if split in {"train", "all"} else [])
        self.test = (self._load("test")
                     if split in {"test", "all"} else [])
        if train_count is not None and (not self._train_all
                                        or not 0 < train_count <= len(
                                            self._train_all)):
            raise ValueError(
                f"train_count must be in [1, {len(self._train_all)}], "
                f"got {train_count}")
        self.train = (self._train_all if train_count is None
                      else self._train_all[:train_count])
        held_out = len(self._train_all) - len(self.train)
        split_note = (f" / {held_out} validation" if held_out else "")
        print(f"[data] n{size}: {len(self.train)} train{split_note} / "
              f"{len(self.test)} test instances loaded "
              f"(fleet: {self.fleet})", flush=True)

    def _load(self, split):
        path = self.data_root / split / f"n{self.size}"
        if not path.is_dir():
            raise FileNotFoundError(f"processed instance directory not found: {path}")
        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no instances under {path}")
        return files

    def _params(self, path):
        path = Path(path).resolve()
        if path not in self._cache:
            config, data = load_problem(path, self.params_path)
            self._cache[path] = Params(data, config, self.size)
        return self._cache[path]

    @property
    def checkpoint_metadata(self):
        return {
            "preproc_hash": self.config.preproc_hash,
            "n_trucks": self.fleet["n_trucks"],
            "n_robots_per_truck": self.fleet["n_robots_per_truck"],
            "size": self.size,
            "tag": self.tag,
        }

    def sample(self):
        """Random draw with replacement from the train pool."""
        if not self.train:
            raise RuntimeError("provider has no training split")
        return self._params(self.rng.choice(self.train))

    def test_set(self):
        """[(instance_id, Params), ...] over the full test split."""
        return [(path.stem, self._params(path)) for path in self.test]

    def validation_set(self, start=200, end=None):
        """Materialize a slice of the original train files for tuning.

        This is independent of ``train_count`` so evaluation can use
        indices 200--249 while a model is trained only on 0--199.
        """
        paths = self._train_all[start:end]
        if not paths:
            raise ValueError(
                f"empty validation split [{start}:{end}] from "
                f"{len(self._train_all)} train instances")
        return [(path.stem, self._params(path)) for path in paths]
