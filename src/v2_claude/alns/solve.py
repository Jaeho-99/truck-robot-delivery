"""Vanilla ALNS over immutable, precomputed truck-robot instances.

Same experiment as ``src/alns/solve.py`` -- same CLI, same instances,
same objective, same artifacts -- with faster destroy/repair
operators that search the identical candidate space. Artifacts are
written under ``output/v2_claude/alns/``.

Geometry is never reconstructed here. Distances and truck/robot travel
times are loaded from ``data/processed*/`` and used as lookup matrices;
this module only performs combinatorial search over routes and assignments.
"""

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

from collections import defaultdict
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _copy_route(route):
    """Structural deep copy of one route.

    ``copy.deepcopy`` spends most of its time on memo bookkeeping and
    dispatch that this fixed, three-level shape does not need.
    """
    out = []
    for st in route:
        if st["kind"] == "cust":
            out.append({"kind": "cust", "c": st["c"]})
        else:
            out.append({"kind": "park", "p": st["p"],
                        "deploys": [{"r": tr["r"],
                                     "custs": list(tr["custs"]),
                                     "ret_p": tr["ret_p"]}
                                    for tr in st["deploys"]]})
    return out


class Solution:
    def __init__(self, K):
        self.routes = {k: [] for k in K}

    def clone(self):
        s = Solution(self.routes.keys())
        s.routes = {k: _copy_route(route) for k, route in self.routes.items()}
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


# Evaluator state, snapshotted before each stop so an insertion
# candidate can resume where it starts to differ from the base route:
#   (prev, b_prev, truck_travel, robot_travel, truck_dist, lateness_sum,
#    parcels, robots_used_mask, feasible, aboard, robot_dist, pending)
def _initial_state(pr):
    width = pr.n_robots + 1
    return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, True,
            (True,) * width, (0.0,) * width, ())


def _run(pr, k, route, start, state, snapshots=None):
    """Evaluate ``route[start:]`` from ``state``; return (cost, feasible).

    Same forward schedule, custody tracking and objective as the
    reference evaluator (:func:`eval_truck` below), and the same
    accumulation order for every float, so the results agree bit for
    bit. It is written to be cheap rather than readable:

      * arcs are read from the flat ``pr.dl``/``pr.ttl``/``pr.trl``
        lists instead of numpy matrices behind accessor methods;
      * lateness is a running sum, not a per-customer dict;
      * ``aboard``/``robot_dist`` are lists indexed by robot id and the
        set of used robots is a bitmask;
      * ``start``/``state`` let a candidate skip the prefix it shares
        with the base route (see :func:`_prefix_states`).
    """
    (prev, b_prev, truck_travel, robot_travel, truck_dist, late_sum,
     parcels, rused, feasible, aboard_t, rdist_t, pending_t) = state
    aboard = list(aboard_t)
    rdist = list(rdist_t)
    pending = {node: list(pairs) for node, pairs in pending_t}

    dl, ttl, trl = pr.dl, pr.ttl, pr.trl
    lam, lcv = pr.lam_arr, pr.lc_arr
    tcoef, rcoef = pr.truck_arc_coef, pr.robot_arc_coef
    s_kc, s_hat = pr.s_kc, pr.s_hat
    zeta_unload, zeta_load = pr.zeta_unload, pr.zeta_load
    beta_robot = pr.beta_robot

    for si in range(start, len(route)):
        if snapshots is not None:
            snapshots.append(
                (prev, b_prev, truck_travel, robot_travel, truck_dist,
                 late_sum, parcels, rused, feasible, tuple(aboard),
                 tuple(rdist),
                 tuple((nd, tuple(v)) for nd, v in pending.items())))
        st = route[si]
        if st["kind"] == "cust":
            node = st["c"]
            leg = ttl[prev][node]
            a_node = b_prev + leg
            truck_travel += leg * tcoef
            truck_dist += dl[prev][node]
            parcels += lam[node]
            late = a_node - lcv[node]
            if late > 0.0:
                late_sum += late
            b_node = a_node + s_kc
        else:
            node = st["p"]
            leg = ttl[prev][node]
            a_node = b_prev + leg
            truck_travel += leg * tcoef
            truck_dist += dl[prev][node]
            b_node = a_node
            deploys = st["deploys"]
            if deploys:
                # --- deploys first (custody state on arrival) ---
                for tr in deploys:
                    r = tr["r"]
                    custs = tr["custs"]
                    ret_p = tr["ret_p"]
                    if not aboard[r]:          # no re-deploy while away
                        feasible = False
                    aboard[r] = False
                    rused |= 1 << r
                    n_custs = len(custs)
                    if n_custs == 0 or n_custs > beta_robot or ret_p == node:
                        feasible = False
                    t = a_node + zeta_unload   # (32) deploy departure
                    rprev = node
                    acc = rdist[r]
                    for c in custs:
                        parcels += lam[c]
                        leg_r = trl[rprev][c]
                        ahat_c = t + leg_r
                        robot_travel += leg_r * rcoef
                        acc += dl[rprev][c]
                        late = ahat_c - lcv[c]
                        if late > 0.0:
                            late_sum += late
                        t = ahat_c + s_hat
                        rprev = c
                    leg_r = trl[rprev][ret_p]
                    arr_ret = t + leg_r
                    robot_travel += leg_r * rcoef
                    acc += dl[rprev][ret_p]
                    rdist[r] = acc
                    slot = pending.get(ret_p)
                    if slot is None:
                        pending[ret_p] = [(r, arr_ret)]
                    else:
                        slot.append((r, arr_ret))
                # (33) the truck cannot leave before the robot departure
                departure = a_node + zeta_unload
                if departure > b_node:
                    b_node = departure
            # --- retrieves at this stop ---
            arrivals = pending.pop(node, None)
            if arrivals is not None:
                for (r, arr) in arrivals:
                    if aboard[r]:
                        feasible = False
                    aboard[r] = True
                    ready = arr + zeta_load    # (34) truck waits
                    if ready > b_node:
                        b_node = ready
        prev, b_prev = node, b_node

    if snapshots is not None:
        snapshots.append(
            (prev, b_prev, truck_travel, robot_travel, truck_dist,
             late_sum, parcels, rused, feasible, tuple(aboard),
             tuple(rdist),
             tuple((nd, tuple(v)) for nd, v in pending.items())))

    # return to depot
    D = pr.D
    truck_travel += ttl[prev][D] * tcoef
    truck_dist += dl[prev][D]

    # custody closure (19) / driving ranges (50)(51) / capacity (39)
    if pending:
        feasible = False
    else:
        for r in pr.R_k[k]:
            if not aboard[r]:
                feasible = False
                break
    if truck_dist > pr.phi_truck + 1e-6:
        feasible = False
    phi_hat = pr.phi_hat + 1e-6
    for r in pr.R_k[k]:
        if rdist[r] > phi_hat:
            feasible = False
            break
    if parcels > pr.beta_truck + 1e-6:
        feasible = False

    cost = (pr.gamma_fixed                                # truck fixed
            + rused.bit_count() * pr.gammahat_fixed       # robot fixed
            + truck_travel + robot_travel                 # travel
            + pr.gamma_late * late_sum)                   # lateness
    return cost, feasible


def eval_route(pr, k, route):
    """(cost, feasible) for one truck route -- the hot-path evaluator."""
    if not route:
        return 0.0, True
    return _run(pr, k, route, 0, _initial_state(pr))


def _prefix_states(pr, k, route):
    """Evaluator states before each stop (length ``len(route) + 1``).

    Every insertion candidate agrees with ``route`` on ``route[:si]``,
    so it can be evaluated from ``si`` against ``_prefix_states(...)[si]``
    instead of from the depot. The shared prefix is accumulated exactly
    once per repair round rather than once per candidate.
    """
    if not route:
        return [_initial_state(pr)]
    snapshots = []
    _run(pr, k, route, 0, _initial_state(pr), snapshots)
    return snapshots


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

    This is the readable reference form, kept for the initial-solution
    constructor and the reporting breakdown, which need the per-customer
    lateness and per-robot distance. The search itself calls
    :func:`eval_route`, which computes the same cost and feasibility
    without building those dicts.
    """
    if not route:
        return 0.0, True, {}, {}
    dl, ttl, trl = pr.dl, pr.ttl, pr.trl
    lam, lcv = pr.lam_arr, pr.lc_arr
    truck_travel = robot_travel = truck_dist = 0.0
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
        leg = ttl[prev][node]
        a_node = b_prev + leg
        truck_travel += leg * pr.truck_arc_coef
        truck_dist += dl[prev][node]
        if st["kind"] == "cust":
            c = st["c"]
            parcels += lam[c]
            lateness[c] = max(0.0, a_node - lcv[c])
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
                parcels += sum(lam[c] for c in custs)
                t = a_node + pr.zeta_unload    # (32) deploy departure
                rprev = p
                for c in custs:
                    leg_r = trl[rprev][c]
                    ahat_c = t + leg_r
                    robot_travel += leg_r * pr.robot_arc_coef
                    robot_dist[r] += dl[rprev][c]
                    lateness[c] = max(0.0, ahat_c - lcv[c])
                    t = ahat_c + pr.s_hat
                    rprev = c
                leg_r = trl[rprev][ret_p]
                arr_ret = t + leg_r            # arrival at retrieval
                robot_travel += leg_r * pr.robot_arc_coef
                robot_dist[r] += dl[rprev][ret_p]
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
    truck_travel += ttl[prev][pr.D] * pr.truck_arc_coef
    truck_dist += dl[prev][pr.D]

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


def eval_solution_cost(pr, sol):
    """(total cost, feasible) -- everything the acceptance test needs.

    Coverage (18): a solution missing any customer is infeasible -- this
    prevents a (cheaper) solution with dropped customers from becoming
    the incumbent after a failed repair.
    """
    total = 0.0
    feasible = sol.customers() == pr.C_set
    for k, route in sol.routes.items():
        cost, ok = eval_route(pr, k, route)
        total += cost
        feasible = feasible and ok
    return total, feasible


def eval_solution(pr, sol):
    """Total cost, feasibility and per-component breakdown.

    Coverage (18) as in :func:`eval_solution_cost`. Used for reporting;
    the search loop calls ``eval_solution_cost``, which returns the same
    total and feasibility without the breakdown.
    """
    total = 0.0
    feasible = sol.customers() == pr.C_set
    brk = dict(truck_fixed=0.0, robot_fixed=0.0, truck_travel=0.0,
               robot_travel=0.0, lateness=0.0)
    per_truck = {}
    for k, route in sol.routes.items():
        cost, ok, late, _ = eval_truck(pr, k, route)
        per_truck[k] = cost
        total += cost
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
            tt += pr.ttl[prev][node] * pr.truck_arc_coef
            if st["kind"] == "park":
                for tr in st["deploys"]:
                    rprev = st["p"]
                    for c in tr["custs"]:
                        rt += pr.trl[rprev][c] * pr.robot_arc_coef
                        rprev = c
                    rt += pr.trl[rprev][tr["ret_p"]] * pr.robot_arc_coef
            prev = node
        if route:
            tt += pr.ttl[prev][pr.D] * pr.truck_arc_coef
        brk["truck_travel"] += tt
        brk["robot_travel"] += rt
    return total, feasible, brk, per_truck
"""Destroy and repair operators, and insertion-candidate enumeration."""

import math


__all__ = ["DESTROY", "enum_insertions", "best_insertion",
           "apply_insertion", "repair_greedy", "repair_regret2",
           "remove_customers", "destroy_random", "destroy_worst",
           "destroy_related"]


# ---- caps on trip-insertion candidates (combinatorial control) ----
L_RET_EXIST = 3   # existing later parking stops tried as retrieval
W_RET_NEW = 4     # positions after the deploy tried for a new stop
N_PHYS_NEAR = 2   # nearest physical locations tried for a new stop

# Reuse the per-(customer, truck) best insertion across repair rounds.
# Never applied to the noise operator, whose per-candidate draws would
# otherwise stop being redrawn every round.
INSERTION_CACHE = True

# Enumerate only the robots already deployed on a route plus the
# lowest-index idle one (robots are homogeneous, so the rest are
# duplicates of exactly the same cost).
ROBOT_SYMMETRY = True

# The noise operator draws one U(-noise, noise) per *candidate*, so
# dropping duplicate candidates shifts its random stream even though
# every dropped candidate was a cost-identical duplicate. With this flag
# set, the symmetry reduction is skipped there and the whole search
# reproduces the reference implementation bit for bit; clear it to trade
# that guarantee for roughly a further 1.5x on noise iterations.
EXACT_NOISE_STREAM = True


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


def _robots_for(pr, k, route, reduce_robots=True):
    """Robots worth enumerating on this route, in ascending index order.

    Robots are homogeneous (same range, same fixed cost), so the idle
    ones are interchangeable and every idle robot beyond the first
    produces a duplicate candidate of exactly the same cost. The strict
    ``<`` tie-break in the callers keeps the first of those, so keeping
    the lowest-index idle robot reproduces the same winner.
    """
    fleet = pr.R_k[k]
    if not (ROBOT_SYMMETRY and reduce_robots):
        return fleet
    deployed = {tr["r"] for st in route if st["kind"] == "park"
                for tr in st["deploys"]}
    if not deployed:
        return fleet[:1]
    out = []
    idle_taken = False
    for r in fleet:
        if r in deployed:
            out.append(r)
        elif not idle_taken:
            out.append(r)
            idle_taken = True
    return out


def _custody_profile(pr, k, route):
    """Custody timeline of the base route, for O(1) candidate screening.

    Returns ``(aboard_arr, aboard_after, next_deploy)`` where

    * ``aboard_arr[i]``    -- bitmask of robots aboard on arrival at stop i
    * ``aboard_after[i]``  -- the same after stop i's own deploys
    * ``next_deploy[r][i]``-- first stop index >= i that deploys robot r,
      or ``len(route)`` when there is none.

    A new trip that takes robot r away between two points of the route is
    custody-feasible only if r is aboard where the trip starts and the
    base route deploys r nowhere in between -- both O(1) against these
    tables. In profiling, custody alone rejected ~70% of all candidates,
    each after a full route build and evaluation.

    The state machine mirrors :func:`_run` exactly, violations included,
    so the screen never rejects a candidate the evaluator would accept.
    """
    n = len(route)
    state = 0
    for r in pr.R_k[k]:
        state |= 1 << r
    aboard_arr = [state] * (n + 1)
    aboard_after = [state] * (n + 1)
    pending = {}
    for i, st in enumerate(route):
        aboard_arr[i] = state
        if st["kind"] == "park":
            for tr in st["deploys"]:
                state &= ~(1 << tr["r"])
                pending.setdefault(tr["ret_p"], []).append(tr["r"])
            aboard_after[i] = state
            for r in pending.pop(st["p"], []):
                state |= 1 << r
        else:
            aboard_after[i] = state
    aboard_arr[n] = aboard_after[n] = state

    next_deploy = {}
    for r in pr.R_k[k]:
        column = [n] * (n + 2)
        nxt = n
        for i in range(n - 1, -1, -1):
            st = route[i]
            if st["kind"] == "park" and any(tr["r"] == r
                                            for tr in st["deploys"]):
                nxt = i
            column[i] = nxt
        next_deploy[r] = column
    return aboard_arr, aboard_after, next_deploy


def enum_insertions_truck(pr, route, c, k, used, robots, custody):
    """Yield ``(new_route, resume_index, claimed_copies)`` for customer c
    on truck k.

      A. direct truck visit
      B. insertion into the customer chain of an existing trip
      C. new trip deployed at an existing parking stop (retrieval at a
         later existing stop / a new copy at the same location / a new
         copy at a nearby location)
      D. new deploy stop plus new trip (retrieval at the second copy of
         the same location, waiting style / at a later existing stop)

    ``resume_index`` is the first index at which the candidate departs
    from ``route``; the evaluator resumes from the prefix state there.
    ``claimed_copies`` are the parking copies the candidate newly
    occupies -- the insertion cache invalidates on those.

    ``custody`` is :func:`_custody_profile` of ``route``. Trips whose
    robot is not aboard where they would launch, or whose robot the base
    route re-deploys before the retrieval, are skipped without building
    or evaluating the route. The evaluator would have rejected exactly
    these, and the noise operator draws only for candidates that pass
    the feasibility test, so the screen changes neither the winner nor
    the random stream.

    Remaining feasibility (range, capacity, coverage) is judged by the
    evaluator, so only the structures are generated here.

    Candidates are built copy-on-write: only containers on the modified
    path are fresh objects, untouched stops/trips are SHARED with the
    route and must never be mutated in place (the evaluator only reads;
    applying an insertion swaps the whole route list; in-place edits
    happen only on Solution.clone() copies).
    """
    # --- A. direct truck visit ---
    for pos in range(len(route) + 1):
        yield (route[:pos] + [{"kind": "cust", "c": c}] + route[pos:],
               pos, ())

    park_pos = [(si, st) for si, st in enumerate(route)
                if st["kind"] == "park"]

    # --- B. insertion into an existing trip ---
    for si, st in park_pos:
        for ti, tr in enumerate(st["deploys"]):
            if len(tr["custs"]) >= pr.beta_robot:
                continue
            for pos in range(len(tr["custs"]) + 1):
                new_tr = dict(tr)
                new_tr["custs"] = tr["custs"][:pos] + [c] + tr["custs"][pos:]
                new_st = dict(st)
                new_st["deploys"] = list(st["deploys"])
                new_st["deploys"][ti] = new_tr
                nr = list(route)
                nr[si] = new_st
                yield nr, si, ()

    aboard_arr, aboard_after, next_deploy = custody
    n_stops = len(route)

    # --- C. deploy at an existing parking stop ---
    for si, st in park_pos:
        for r in robots:
            if not aboard_after[si] & (1 << r):
                continue                  # robot is away at this stop
            after_si = next_deploy[r][si + 1]   # base re-deploy of r
            # ret 1) later existing parking stops (L_RET_EXIST max)
            for sj, st2 in [pp for pp in park_pos
                            if pp[0] > si][:L_RET_EXIST]:
                if after_si <= sj:        # re-deployed before retrieval
                    continue
                yield (_with_new_trip(route, si, st,
                                      {"r": r, "custs": [c],
                                       "ret_p": st2["p"]}),
                       si, ())
            # ret 2) fresh copy at the same location right behind
            #        (waiting style, consumes two copies)
            grp = pr.park_groups[pr.copy_to_phys[st["p"]]]
            free = [cp for cp in grp if cp not in used]
            if free:
                nr = _with_new_trip(route, si, st,
                                    {"r": r, "custs": [c],
                                     "ret_p": free[0]})
                nr.insert(si + 1, {"kind": "park", "p": free[0],
                                   "deploys": []})
                yield nr, si, (free[0],)
            # ret 3) fresh copy at a location near c, inserted within
            #        W positions after the deploy
            for gi in pr.phys_near[c][:N_PHYS_NEAR]:
                grp2 = pr.park_groups[gi]
                free2 = [cp for cp in grp2
                         if cp not in used and cp != st["p"]]
                if not free2:
                    continue
                for pos in range(si + 1,
                                 min(n_stops, si + W_RET_NEW) + 1):
                    if after_si < pos:    # re-deployed before retrieval
                        continue
                    nr = _with_new_trip(route, si, st,
                                        {"r": r, "custs": [c],
                                         "ret_p": free2[0]})
                    nr.insert(pos, {"kind": "park", "p": free2[0],
                                    "deploys": []})
                    yield nr, si, (free2[0],)

    # --- D. new deploy stop plus new trip ---
    for gi in pr.phys_near[c][:N_PHYS_NEAR]:
        grp = pr.park_groups[gi]
        free = [cp for cp in grp if cp not in used]
        if not free:
            continue
        dep_cp = free[0]
        for r in robots:
            bit = 1 << r
            column = next_deploy[r]
            for pos in range(n_stops + 1):
                if not aboard_arr[pos] & bit:
                    continue              # robot is away at this point
                # ret a) second copy of the same location (waiting)
                if len(free) >= 2:
                    nr = list(route)
                    nr.insert(pos, {"kind": "park", "p": dep_cp,
                                    "deploys": [{"r": r, "custs": [c],
                                                 "ret_p": free[1]}]})
                    nr.insert(pos + 1, {"kind": "park", "p": free[1],
                                        "deploys": []})
                    yield nr, pos, (dep_cp, free[1])
                # ret b) later existing parking stops
                after_pos = column[pos]
                later = [(si2, st2) for si2, st2 in park_pos
                         if si2 >= pos][:L_RET_EXIST]
                for si2, st2 in later:
                    if after_pos <= si2:  # re-deployed before retrieval
                        continue
                    nr = list(route)
                    nr.insert(pos, {"kind": "park", "p": dep_cp,
                                    "deploys": [{"r": r, "custs": [c],
                                                 "ret_p": st2["p"]}]})
                    yield nr, pos, (dep_cp,)


def enum_insertions(pr, sol, c):
    """Yield every insertion candidate ``(k, new_route)`` for customer c.

    Thin wrapper over :func:`enum_insertions_truck` for callers that do
    not keep the per-truck bookkeeping (the repair operators do, through
    :class:`_RepairContext`).
    """
    used = sol.used_copies()
    for k in pr.K:
        route = sol.routes[k]
        for nr, _si, _copies in enum_insertions_truck(
                pr, route, c, k, used, _robots_for(pr, k, route),
                _custody_profile(pr, k, route)):
            yield k, nr


_MISS = object()


class _RepairContext:
    """Per-repair bookkeeping shared by the greedy and regret operators.

    Holds, per truck: its base cost, the prefix states of its route, the
    robots worth enumerating, the custody profile, and the best
    insertion found so far for each pending customer.

    Applying an insertion invalidates only the truck that changed, plus
    any cached entry whose claimed parking copy has just been taken:
    inserting into truck k leaves every other truck's route untouched,
    and the used-copy set only grows during a repair, so those trucks'
    candidate sets can only shrink -- and a replacement copy at the same
    physical location costs exactly the same, so no better candidate can
    appear in the meantime.
    """

    def __init__(self, pr, sol, use_cache, reduce_robots=True):
        self.pr = pr
        self.sol = sol
        self.use_cache = use_cache
        self.reduce_robots = reduce_robots
        self.used = sol.used_copies()
        self.base = {}
        self.prefix = {}
        self.robots = {}
        self.custody = {}
        self.cache = {}        # (c, k) -> (delta, new_route, claimed)
        for k in pr.K:
            self._refresh(k)

    def _refresh(self, k):
        pr, route = self.pr, self.sol.routes[k]
        self.base[k] = eval_route(pr, k, route)[0]
        self.prefix[k] = _prefix_states(pr, k, route)
        self.robots[k] = _robots_for(pr, k, route, self.reduce_robots)
        self.custody[k] = _custody_profile(pr, k, route)

    def best_on(self, k, c, rng=None, noise=0.0):
        """Cheapest feasible insertion of c on truck k.

        Returns ``(delta, new_route, claimed_copies)`` or ``None``.
        """
        if self.use_cache:
            hit = self.cache.get((c, k), _MISS)
            if hit is not _MISS:
                return hit
        pr = self.pr
        route = self.sol.routes[k]
        prefix = self.prefix[k]
        base_k = self.base[k]
        best_delta, best_route, best_claim = math.inf, None, ()
        for nr, si, claimed in enum_insertions_truck(
                pr, route, c, k, self.used, self.robots[k],
                self.custody[k]):
            cost, ok = _run(pr, k, nr, si, prefix[si])
            if not ok:
                continue
            delta = cost - base_k
            if noise > 0.0 and rng is not None:
                delta += rng.uniform(-noise, noise)
            if delta < best_delta - 1e-9:
                best_delta, best_route, best_claim = delta, nr, claimed
        found = (None if best_route is None
                 else (best_delta, best_route, best_claim))
        if self.use_cache:
            self.cache[(c, k)] = found
        return found

    def best(self, c, rng=None, noise=0.0):
        """``(delta, k, new_route, claimed)`` over all trucks, or None."""
        best = None
        for k in self.pr.K:
            found = self.best_on(k, c, rng, noise)
            if found is None:
                continue
            if best is None or found[0] < best[0] - 1e-9:
                best = (found[0], k, found[1], found[2])
        return best

    def per_truck(self, c, rng=None, noise=0.0):
        """``{k: (delta, k, new_route, claimed)}`` for accepting trucks."""
        local = {}
        for k in self.pr.K:
            found = self.best_on(k, c, rng, noise)
            if found is not None:
                local[k] = (found[0], k, found[1], found[2])
        return local

    def apply(self, k, new_route, claimed):
        self.sol.routes[k] = new_route
        if self.use_cache:
            taken = set(claimed)
            stale = [key for key, entry in self.cache.items()
                     if key[1] == k
                     or (entry is not None and taken.intersection(entry[2]))]
            for key in stale:
                del self.cache[key]
        self.used = self.sol.used_copies()
        self._refresh(k)


def best_insertion(pr, sol, c, rng=None, noise=0.0):
    """Cheapest insertion of customer c.

    Returns (delta, (k, new_route)) or (inf, None). With noise > 0 a
    U(-noise, noise) perturbation is added to each candidate's delta
    (Ropke & Pisinger noise insertion) — locally poor insertions such
    as the first customer of a new robot trip are then chosen
    occasionally, allowing escapes from truck-only local optima.
    """
    noisy = noise > 0.0 and rng is not None
    ctx = _RepairContext(pr, sol, use_cache=False,
                         reduce_robots=not (noisy and EXACT_NOISE_STREAM))
    found = ctx.best(c, rng, noise)
    if found is None:
        return math.inf, None
    return found[0], (found[1], found[2])


def apply_insertion(sol, apply_tuple):
    k, new_route = apply_tuple
    sol.routes[k] = new_route


# ============================================================
# Repair operators
# ============================================================
def repair_greedy(pr, sol, pool, rng, noise=0.0):
    """Greedy insertion: the customer with the cheapest insertion first.

    The per-(customer, truck) results are reused across rounds when the
    truck did not change -- except under noise, where every round must
    redraw (see ``EXACT_NOISE_STREAM``).
    """
    noisy = noise > 0.0 and rng is not None
    remaining = list(pool)
    ctx = _RepairContext(pr, sol,
                         use_cache=INSERTION_CACHE and not noisy,
                         reduce_robots=not (noisy and EXACT_NOISE_STREAM))
    while remaining:
        best = None                      # (delta, k, new_route, claimed, c)
        for c in remaining:
            found = ctx.best(c, rng, noise)
            if found is None:
                continue
            if best is None or found[0] < best[0] - 1e-9:
                best = (*found, c)
        if best is None:
            return False
        ctx.apply(best[1], best[2], best[3])
        remaining.remove(best[4])
    return True


def repair_regret2(pr, sol, pool, rng):
    """Regret-2: insert first the customer whose gap between its best
    and second-best per-truck insertion delta is largest."""
    remaining = list(pool)
    ctx = _RepairContext(pr, sol, use_cache=INSERTION_CACHE)
    while remaining:
        pick = None       # (regret, k, new_route, claimed, c)
        for c in remaining:
            local = ctx.per_truck(c)
            if not local:
                continue
            # Stable sort on the delta alone, so ties keep truck order.
            deltas = sorted(local.values(), key=lambda t: t[0])
            best_d = deltas[0][0]
            second = deltas[1][0] if len(deltas) > 1 else best_d + 1e6
            regret = second - best_d
            if pick is None or regret > pick[0] + 1e-9:
                _, k, new_route, claimed = deltas[0]
                pick = (regret, k, new_route, claimed, c)
        if pick is None:
            return False
        ctx.apply(pick[1], pick[2], pick[3])
        remaining.remove(pick[4])
    return True


# ============================================================
# Destroy operators
# ============================================================
def _route_without(route, cset):
    """One route with ``cset`` removed.

    Drops emptied trips and parking stops that no longer host a deploy
    nor are referenced as a retrieval. Non-mutating, so ``destroy_worst``
    can price a removal without cloning the solution.
    """
    mid = []
    for st in route:
        if st["kind"] == "cust":
            if st["c"] in cset:
                continue
            mid.append(st)
        else:
            deploys = [dict(tr, custs=[c for c in tr["custs"]
                                       if c not in cset])
                       for tr in st["deploys"]]
            deploys = [tr for tr in deploys if tr["custs"]]
            mid.append({"kind": "park", "p": st["p"], "deploys": deploys})
    # Retrieval references of the surviving trips decide which parking
    # stops still have a role.
    refs = {tr["ret_p"] for st in mid if st["kind"] == "park"
            for tr in st["deploys"]}
    return [st for st in mid
            if st["kind"] == "cust" or st["deploys"] or st["p"] in refs]


def remove_customers(sol, custs):
    """Remove the given customers, drop emptied trips, and drop parking
    stops that no longer host a deploy nor are referenced as a
    retrieval."""
    cset = set(custs)
    for k, route in sol.routes.items():
        sol.routes[k] = _route_without(route, cset)


def destroy_random(pr, sol, q, rng):
    custs = list(sol.customers())
    q = min(q, len(custs))
    chosen = rng.sample(custs, q)
    remove_customers(sol, chosen)
    return chosen


def destroy_worst(pr, sol, q, rng, p=3.0):
    """Prefer customers with the largest removal gain (current cost
    minus cost after removal), randomized by exponent p.

    Removing one customer touches exactly one truck, so only that
    truck's route is rebuilt and re-evaluated. The solution total is
    re-summed over the trucks in their original order, which keeps every
    gain bit-identical to a full re-evaluation of a cloned solution.
    """
    per_truck = {}
    owner = {}
    base = 0.0
    for k, route in sol.routes.items():
        per_truck[k] = eval_route(pr, k, route)[0]
        base += per_truck[k]
        for st in route:
            if st["kind"] == "cust":
                owner[st["c"]] = k
            else:
                for tr in st["deploys"]:
                    for c in tr["custs"]:
                        owner[c] = k
    contrib = []
    for c in sol.customers():
        kc = owner[c]
        stripped = eval_route(pr, kc, _route_without(sol.routes[kc], {c}))[0]
        after = 0.0
        for k in sol.routes:
            after += stripped if k == kc else per_truck[k]
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
    """Shaw-style: remove customers close in distance and time window.

    The relatedness ranking of every customer is precomputed once per
    instance (``pr.related_order``), so each round filters a ready-made
    order instead of sorting with a Python-level key.
    """
    custs = sol.customers()
    if not custs:
        return []
    q = min(q, len(custs))
    seed = rng.choice(list(custs))
    removed = [seed]
    removed_set = {seed}
    order = pr.related_order
    while len(removed) < q:
        ref = rng.choice(removed)
        cand = [c for c in order[ref]
                if c in custs and c not in removed_set]
        y = rng.random()
        idx = int((y ** p) * len(cand))
        pick = cand[idx]
        removed.append(pick)
        removed_set.add(pick)
    remove_customers(sol, removed)
    return removed


DESTROY = [("random", destroy_random), ("worst", destroy_worst),
           ("related", destroy_related)]
"""ALNS for truck-robot collaborative last-mile delivery.

Standard Ropke & Pisinger (2006) skeleton: destroy (random / worst /
related) + repair (greedy / greedy-noise / regret-2) with roulette-wheel
adaptive weights and simulated-annealing acceptance.
The evaluator above reproduces the MILP's objective and
feasibility definition exactly, so ALNS and exact objective values are
directly comparable.

This module provides the parameter container (``Params``), the
initial-solution constructors, and the ALNS driver (``solve_alns``).
``congestion_aware_initial`` is the default initial solution: a
two-stage constructive heuristic that assigns customers in zones where
trucks are relatively more congested than robots to robot delivery
trips. ``truck_only_initial`` (greedy truck-only dispatch) is kept as
its fallback.
"""

import math
import numpy as np
import random
import time
from collections import defaultdict

from common.params import Instance as ProcessedInstance
from common.params import Params as SharedParams
from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT, load_params, load_problem


# Robot-candidate threshold theta: only customers with a normalized
# score > 0 are considered by the congestion-aware initial solution.
INIT_SCORE_MIN = 1e-9

# Noise amplitude = initial-solution cost x NOISE_FRAC. The default was
# tuned on an instance with a high entry barrier for multi-customer
# robot trips: noise 0.25 with w_start 0.25 reached exact +0.5% on 3/3
# restarts.
NOISE_FRAC = 0.25

# SA start-temperature fraction (DR-ALNS / Roozbeh rule of thumb): a
# solution 5% worse than the initial one is accepted with probability
# 0.5 at T_start = W_START * f_init / ln 2. Shared by solve_alns, the
# PPO environments.
W_START = 0.05

# Degree of destruction (DR-ALNS vanilla): a FIXED 30% of customers is
# removed each iteration, q = max(1, round(DOD * n)). Shared by all
# training/test paths.
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
        self.node_zone = np.ascontiguousarray(data.node_zone[source_index])
        self.labels = tuple(data.node_label[source_index].tolist())
        self.d = np.ascontiguousarray(data.d[np.ix_(source_index, source_index)])
        self.tau_truck_matrix = np.ascontiguousarray(
            data.tau_truck[np.ix_(source_index, source_index)])
        self.tau_robot_matrix = np.ascontiguousarray(
            data.tau_robot[np.ix_(source_index, source_index)])
        for array in (self.node_zone, self.d,
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

        # ---- flat lookup tables read by the evaluator ----
        # The search reads arcs millions of times per run, and nested
        # Python lists index far faster than numpy scalars behind an
        # accessor method. Built once per instance, never mutated.
        self.dl = [[float(v) for v in row] for row in self.d.tolist()]
        self.ttl = [[float(v) for v in row]
                    for row in self.tau_truck_matrix.tolist()]
        self.trl = [[float(v) for v in row]
                    for row in self.tau_robot_matrix.tolist()]
        n_nodes = self.D + 1
        self.lam_arr = [0] * n_nodes      # demand, indexed by node id
        self.lc_arr = [0.0] * n_nodes     # due time, indexed by node id
        for c in self.C:
            self.lam_arr[c] = self.lam[c]
            self.lc_arr[c] = self.l_c[c]
        self.n_robots = max(max(ids) for ids in self.R_k.values())
        self.C_set = set(self.C)
        # Shaw relatedness ranking for destroy_related, materialized once
        # (ties break on the customer index, which is the iteration order
        # of the small integer set the operator used to sort).
        self.related_order = {
            a: sorted(self.C,
                      key=lambda b: self.dl[a][b]
                      + 0.1 * abs(self.l_c[a] - self.l_c[b]))
            for a in self.C}

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
# 3. ALNS driver (SA acceptance + adaptive weights)
# ============================================================
def roulette(weights, rng):
    tot = sum(weights)
    y = rng.random() * tot
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if y <= acc:
            return i
    return len(weights) - 1


def solve_alns(pr, iters=3000, seed=0, segment=None,
               sigma=(5.0, 3.0, 1.0), reaction=0.2, w_start=W_START,
               time_limit_s=None, initial=None, iter_trace=None, dod=DOD):
    """Run ALNS and return (best_solution, best_cost, stats).

    ``initial``: initial-solution constructor ``f(pr, rng) -> Solution``
    (default: congestion_aware_initial).
    ``iter_trace``: optional list; if given, one
    (it, accepted, current_cost) tuple is appended per iteration
    (instrumentation only — never touches rng).
    ``time_limit_s``: wall-clock cap in seconds; exceeding it stops the
    run early (for large instances). The cooling schedule stays based
    on ``iters``, so an early stop may end in the hot phase.
    Operators are chosen by classic adaptive roulette weights.
    """
    t_start = time.time()
    rng = random.Random(seed)
    segment = segment or max(20, iters // 30)
    nC = len(pr.C)
    # degree of destruction: fixed fraction of customers (DR-ALNS
    # vanilla uses 30%; ``dod`` allows regime studies)
    q_destroy = max(1, round(dod * nC))

    if initial is None:
        initial = congestion_aware_initial
    current_solution = initial(pr, rng)
    current_cost, feas = eval_solution_cost(pr, current_solution)
    best_solution, best_cost = current_solution.clone(), current_cost
    init_cost = current_cost

    # Repair operators (noise amplitude scales with instance cost).
    noise_amp = NOISE_FRAC * init_cost
    repair_ops = [
        ("greedy",
         lambda p_, s_, pool_, rng_: repair_greedy(p_, s_, pool_,
                                                   rng_, 0.0)),
        ("greedy_noise",
         lambda p_, s_, pool_, rng_: repair_greedy(p_, s_, pool_,
                                                   rng_, noise_amp)),
        ("regret2", repair_regret2),
    ]

    # SA start temperature: a solution worse than the initial one by
    # w_start (fraction) is accepted with probability 0.5. Linear
    # decay to 0 over the run (Santini et al.; same rule in PPO).
    T0 = (w_start * init_cost) / math.log(2)

    dW = [1.0] * len(DESTROY)
    rW = [1.0] * len(repair_ops)
    dScore = [0.0] * len(DESTROY)
    rScore = [0.0] * len(repair_ops)
    dCnt = [0] * len(DESTROY)
    rCnt = [0] * len(repair_ops)
    seen = set()

    # Instrumentation (timers/counters only — never touches rng, so
    # results stay byte-identical to the uninstrumented loop).
    n_actions = len(DESTROY) * len(repair_ops)
    pair_cnt = [0] * n_actions          # a = di * len(repair_ops) + ri
    pair_time = [0.0] * n_actions       # destroy+repair+eval seconds
    sel_time = 0.0                      # selector overhead seconds
    accept_cnt = infeas_cnt = best_updates = best_hit_it = 0
    best_trace = []                     # (iter, elapsed_s, best_cost)

    it_done = 0
    for it in range(1, iters + 1):
        if time_limit_s is not None and time.time() - t_start > time_limit_s:
            break
        it_done = it
        # linear temperature decay T0 -> 0 over the run
        T = T0 * (1.0 - (it - 1) / iters)
        t_sel = time.perf_counter()
        di = roulette(dW, rng)
        ri = roulette(rW, rng)
        sel_time += time.perf_counter() - t_sel
        a_idx = di * len(repair_ops) + ri
        t_op = time.perf_counter()      # clone+destroy+repair+eval
        cand = current_solution.clone()
        pool = DESTROY[di][1](pr, cand, q_destroy, rng)
        repair_ops[ri][1](pr, cand, pool, rng)
        cand_cost, ok = eval_solution_cost(pr, cand)
        pair_cnt[a_idx] += 1
        pair_time[a_idx] += time.perf_counter() - t_op
        dCnt[di] += 1
        rCnt[ri] += 1

        if not ok:      # discard coverage/custody/range violations
            infeas_cnt += 1
            if iter_trace is not None:
                iter_trace.append((it, 0, round(current_cost, 9)))
            continue

        # Operator scores (DR-ALNS weights w1..w4 = 5, 3, 1, 0):
        # 5 new best / 3 improving the current solution / 1 accepted /
        # 0 otherwise (improving/accepted only for unseen solutions).
        key = round(cand_cost, 4)
        reward = 0.0
        accept = False
        found_best = False
        was_improving = cand_cost < current_cost - 1e-9
        if cand_cost < best_cost - 1e-9:
            found_best = True
            best_solution, best_cost = cand.clone(), cand_cost
            best_updates += 1
            best_hit_it = it
            best_trace.append((it, round(time.time() - t_start, 3),
                               round(cand_cost, 6)))
            reward = sigma[0]
            accept = True
        elif cand_cost < current_cost - 1e-9 and key not in seen:
            reward = sigma[1]
            accept = True
        else:
            if cand_cost < current_cost - 1e-9 or rng.random() < math.exp(
                    -(cand_cost - current_cost) / max(T, 1e-9)):
                accept = True
                if key not in seen:
                    reward = sigma[2]
        seen.add(key)
        if accept:
            accept_cnt += 1
            current_solution, current_cost = cand, cand_cost
        if iter_trace is not None:
            iter_trace.append((it, int(accept),
                               round(current_cost, 9)))
        dScore[di] += reward
        rScore[ri] += reward

        if it % segment == 0:                    # adaptive weight update
            for i in range(len(DESTROY)):
                if dCnt[i] > 0:
                    dW[i] = (dW[i] * (1 - reaction)
                             + reaction * (dScore[i] / dCnt[i]))
                dScore[i] = 0.0
                dCnt[i] = 0
            for i in range(len(repair_ops)):
                if rCnt[i] > 0:
                    rW[i] = (rW[i] * (1 - reaction)
                             + reaction * (rScore[i] / rCnt[i]))
                rScore[i] = 0.0
                rCnt[i] = 0

    stats = {"init_cost": init_cost, "best_cost": best_cost,
             "iters_done": it_done, "selector": "roulette",
             "improve_pct": 100.0 * (init_cost - best_cost) / init_cost,
             "destroy_w": dict(zip([d[0] for d in DESTROY],
                                   [round(x, 3) for x in dW])),
             "repair_w": dict(zip([r[0] for r in repair_ops],
                                  [round(x, 3) for x in rW])),
             # instrumentation (a = destroy_index * 3 + repair_index)
             "pair_labels": [f"{d[0]}+{r[0]}" for d in DESTROY
                             for r in repair_ops],
             "action_hist": pair_cnt,
             "pair_time_s": [round(x, 3) for x in pair_time],
             "selector_overhead_s": round(sel_time, 3),
             "accept_count": accept_cnt,
             "infeasible_count": infeas_cnt,
             "best_update_count": best_updates,
             "best_first_hit_iter": best_hit_it,
             "best_trace": best_trace}
    return best_solution, best_cost, stats
"""Processed-instance provider used by ALNS experiments."""

import re


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


import argparse
from datetime import datetime
import multiprocessing
from multiprocessing.connection import wait
import os
import traceback

import yaml

from v2_claude.common.artifacts import (reserve_artifacts, validate_run_label,
                              write_csv, write_json)


def _case_result(task, size, params_path, cache, capture_trace):
    """Solve one independent case; Params and Solution never cross a pipe."""
    task_id, instance_path, seed, iterations = task
    load_started = time.perf_counter()
    if instance_path not in cache:
        config, data = load_problem(instance_path, params_path)
        cache[instance_path] = Params(data, config, size)
    params = cache[instance_path]
    load_seconds = time.perf_counter() - load_started
    iter_trace = [] if capture_trace else None
    started = time.perf_counter()
    solution, objective, stats = solve_alns(
        params, iters=iterations, seed=seed, iter_trace=iter_trace)
    result = {
        "task_id": task_id,
        # Preserve the old CLI's filename-derived instance identifier.
        "instance_id": Path(instance_path).stem,
        "seed": seed,
        "obj": objective,
        "stats": stats,
        "routes": solution.routes,
        "runtime_s": time.perf_counter() - started,
        "load_seconds": load_seconds,
        "cache_size": len(cache),
    }
    if capture_trace:
        result["iter_trace"] = iter_trace
    return result


def _alns_case_worker(worker_id, connection, size, params_path, capture_trace):
    """Spawn-safe CPU worker. Imports neither Torch nor a CUDA runtime."""
    cache = {}
    task_id = None
    phase = "startup"
    try:
        connection.send(("ready", worker_id, os.getpid()))
        while True:
            phase = "receive"
            message = connection.recv()
            if message == ("close",):
                return
            if not isinstance(message, tuple) or len(message) != 2:
                raise ValueError("invalid ALNS worker request")
            command, task = message
            if command != "solve":
                raise ValueError(f"unknown ALNS command: {command!r}")
            task_id = task[0]
            phase = "solve"
            result = _case_result(task, size, params_path, cache, capture_trace)
            phase = "send"
            connection.send(("ok", worker_id, task_id, result))
            task_id = None
    except KeyboardInterrupt:
        # Parent owns the user-facing interruption and whole-pool cleanup.
        return
    except EOFError:
        return
    except BaseException:
        try:
            connection.send(("error", worker_id, task_id, phase,
                             traceback.format_exc()))
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


def _close_case_workers(processes, connections, idle, grace_seconds=5.0):
    """Bounded shutdown, including partial startup and blocked/failed jobs."""
    for worker_id in idle:
        process = processes.get(worker_id)
        if process is not None and process.is_alive():
            try:
                connections[worker_id].send(("close",))
            except (EOFError, OSError):
                pass
    deadline = time.monotonic() + grace_seconds
    for process in processes.values():
        if process.pid is not None:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
    remaining = [process for process in processes.values()
                 if process.is_alive()]
    for process in remaining:
        process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in remaining:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    # Never wait without a deadline, even if a platform termination fails.
    survivors = [process for process in remaining if process.is_alive()]
    for process in survivors:
        process.kill()
    deadline = time.monotonic() + grace_seconds
    for process in survivors:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for connection in connections.values():
        connection.close()
    for process in processes.values():
        if process.is_alive():
            print(f"[v2_claude/alns cleanup] worker PID {process.pid} did not exit; "
                  "inspect this PID before manual recovery", flush=True)
        else:
            process.close()


def run_cases(instance_paths, *, size, params_path=DEFAULT_PARAMS_PATH,
              iterations=100, seeds=5, workers=1, capture_trace=False,
              progress=True, heartbeat_interval=30.0, startup_timeout=180.0):
    """Return case results in the original instance-major, seed-minor order.

    ``workers=1`` is the serial baseline. More workers parallelize whole
    independent solves, not the dependent iterations within a solve. The
    same seed is used for a case regardless of which worker receives it.
    Each worker loads and caches its own immutable Params from file paths.
    No output files are written here, so verification cannot overwrite runs.
    """
    for name, value in (("iterations", iterations), ("seeds", seeds),
                        ("workers", workers)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if workers > 30:
        raise ValueError("workers must be <= 30 (pipe + process wait handles)")
    if heartbeat_interval <= 0 or startup_timeout <= 0:
        raise ValueError("heartbeat_interval and startup_timeout must be positive")
    paths = [str(Path(path).expanduser().resolve()) for path in instance_paths]
    if not paths:
        raise ValueError("at least one instance path is required")
    params_path = str(Path(params_path).expanduser().resolve())
    tasks = [(index * seeds + seed, path, seed, iterations)
             for index, path in enumerate(paths) for seed in range(seeds)]
    results = [None] * len(tasks)
    if workers == 1:
        cache = {}
        for task in tasks:
            if progress:
                print(f"[v2_claude/alns {task[0] + 1}/{len(tasks)}] "
                      f"instance={Path(task[1]).stem} seed={task[2]} starting",
                      flush=True)
            result = _case_result(task, size, params_path, cache, capture_trace)
            results[task[0]] = result
            if progress:
                print(f"[v2_claude/alns {task[0] + 1}/{len(tasks)}] "
                      f"obj={result['obj']:.6f} "
                      f"solve={result['runtime_s']:.2f}s", flush=True)
        return results

    # There is no benefit in spawning idle workers when a batch is smaller.
    worker_count = min(workers, len(tasks))
    context = multiprocessing.get_context("spawn")
    processes, connections = {}, {}
    idle = set()
    assignments = {}
    try:
        for worker_id in range(worker_count):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_alns_case_worker,
                args=(worker_id, child, size, params_path, capture_trace),
                name=f"ALNS-case-{worker_id}", daemon=False)
            # Register before start so an interrupt/failure during startup
            # still reaches the shared cleanup path for every owned handle.
            processes[worker_id] = process
            connections[worker_id] = parent
            try:
                process.start()
            finally:
                child.close()

        ready_workers = set()
        startup_started = time.monotonic()
        last_heartbeat = startup_started
        while len(ready_workers) != worker_count:
            pending = set(processes) - ready_workers
            objects = [connections[i] for i in pending]
            objects.extend(process.sentinel for process in processes.values())
            readable = wait(objects, timeout=1.0)
            for worker_id in sorted(pending):
                connection = connections[worker_id]
                if connection not in readable:
                    continue
                message = connection.recv()
                if (not isinstance(message, tuple) or len(message) != 3
                        or message[:2] != ("ready", worker_id)
                        or message[2] != processes[worker_id].pid):
                    raise RuntimeError(f"worker {worker_id}: invalid READY {message!r}")
                ready_workers.add(worker_id)
                idle.add(worker_id)
            for worker_id, process in processes.items():
                if not process.is_alive():
                    raise RuntimeError(
                        f"worker {worker_id} died during startup; "
                        f"exitcode={process.exitcode}")
            now = time.monotonic()
            if (len(ready_workers) < worker_count
                    and now - startup_started >= startup_timeout):
                raise TimeoutError(f"ALNS worker startup timed out: {sorted(pending)}")
            if progress and now - last_heartbeat >= heartbeat_interval:
                print(f"[v2_claude/alns startup] waiting={sorted(set(processes) - ready_workers)} "
                      f"elapsed={now - startup_started:.1f}s", flush=True)
                last_heartbeat = now

        next_task = 0
        completed = 0
        while completed < len(tasks):
            for worker_id in sorted(idle):
                if next_task >= len(tasks):
                    break
                task = tasks[next_task]
                assignments[worker_id] = (task[0], time.monotonic())
                idle.remove(worker_id)
                connections[worker_id].send(("solve", task))
                next_task += 1
            objects = [connections[i] for i in assignments]
            objects.extend(process.sentinel for process in processes.values())
            readable = wait(objects, timeout=1.0)
            for worker_id in list(assignments):
                connection = connections[worker_id]
                if connection not in readable:
                    continue
                try:
                    message = connection.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(
                        f"ALNS worker {worker_id} disconnected while solving "
                        f"task {assignments[worker_id][0]}") from error
                expected_id = assignments[worker_id][0]
                if (not isinstance(message, tuple) or len(message) < 3
                        or message[1:3] != (worker_id, expected_id)):
                    raise RuntimeError(f"worker {worker_id}: invalid response {message!r}")
                if message[0] == "error" and len(message) == 5:
                    raise RuntimeError(
                        f"ALNS worker {worker_id}, task {expected_id}, "
                        f"phase={message[3]} failed:\n{message[4]}")
                if message[0] != "ok" or len(message) != 4:
                    raise RuntimeError(f"worker {worker_id}: invalid response {message!r}")
                result = message[3]
                if (result.get("task_id") != expected_id
                        or result.get("instance_id") != Path(tasks[expected_id][1]).stem
                        or result.get("seed") != tasks[expected_id][2]
                        or results[expected_id] is not None):
                    raise RuntimeError(f"worker {worker_id}: mismatched result for task {expected_id}")
                results[expected_id] = result
                del assignments[worker_id]
                idle.add(worker_id)
                completed += 1
                if progress:
                    print(f"[v2_claude/alns completed={completed}/{len(tasks)}] "
                          f"task={expected_id} worker={worker_id} "
                          f"instance={result['instance_id']} seed={result['seed']} "
                          f"obj={result['obj']:.6f} "
                          f"solve={result['runtime_s']:.2f}s", flush=True)
            for worker_id, process in processes.items():
                if not process.is_alive():
                    # The process sentinel can become ready just after wait's
                    # pipe snapshot. Preserve an already queued traceback.
                    if (worker_id in assignments
                            and connections[worker_id].poll()):
                        try:
                            message = connections[worker_id].recv()
                        except (EOFError, OSError):
                            message = None
                        if (isinstance(message, tuple) and len(message) == 5
                                and message[:3] == (
                                    "error", worker_id, assignments[worker_id][0])):
                            raise RuntimeError(
                                f"ALNS worker {worker_id}, task={message[2]}, "
                                f"phase={message[3]} failed:\n{message[4]}")
                    raise RuntimeError(
                        f"ALNS worker {worker_id} died; exitcode={process.exitcode}")
            now = time.monotonic()
            if progress and now - last_heartbeat >= heartbeat_interval:
                waiting = ", ".join(
                    f"w{i}:task={task_id},elapsed={now - started:.1f}s"
                    for i, (task_id, started) in sorted(assignments.items()))
                print(f"[v2_claude/alns waiting] {waiting}", flush=True)
                last_heartbeat = now
        return results
    finally:
        _close_case_workers(processes, connections, idle)


def _parser():
    parser = argparse.ArgumentParser(
        description="Run vanilla ALNS on precomputed test instances "
                    "(v2_claude fast operators)")
    parser.add_argument("--size", type=int, choices=(5, 10, 20, 50, 100),
                        required=True)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of repetition seeds, starting at zero")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="independent case processes (1-30); 1 keeps the serial baseline")
    parser.add_argument("--run-label",
                        help="unique output label; existing artifacts are never overwritten")
    return parser


def main():
    main_started = time.perf_counter()
    args = _parser().parse_args()
    if args.iterations <= 0 or args.seeds <= 0 or args.workers <= 0:
        raise ValueError("--iterations, --seeds and --workers must be positive")
    if args.workers > 30:
        raise ValueError("--workers must be <= 30")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    validate_run_label(args.run_label)
    provider = DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag, split="test")
    params_path = args.params.expanduser().resolve()
    with params_path.open("r", encoding="utf-8") as f:
        parameter_settings = yaml.safe_load(f)
    # Send paths, not Params (which contains non-picklable immutable config).
    cases = provider.test
    if args.limit is not None:
        cases = cases[:args.limit]
    method_dir = "alns" if args.tag is None else f"alns_{args.tag}"
    out_dir = REPO_ROOT / "output" / "v2_claude" / method_dir / f"n{args.size}"
    if args.run_label is not None:
        out_dir = out_dir / "runs" / args.run_label
    route_paths = [out_dir / f"{path.stem}_s{seed}.json"
                   for path in cases for seed in range(args.seeds)]
    summary_path = out_dir / "summary.csv"
    metadata_path = out_dir / "test_metadata.json"
    with reserve_artifacts([*route_paths, summary_path, metadata_path]):
        actual_workers = min(args.workers, len(route_paths))
        print(f"[v2_claude/alns] cases={len(cases)} seeds={args.seeds} "
              f"tasks={len(route_paths)} iterations={args.iterations} "
              f"workers={actual_workers} "
              f"backend={'serial' if args.workers == 1 else 'process'} "
              f"run_label={args.run_label!r}", flush=True)
        solve_started = time.perf_counter()
        results = run_cases(
            cases, size=args.size, params_path=params_path,
            iterations=args.iterations, seeds=args.seeds, workers=args.workers)
        solve_wall_seconds = time.perf_counter() - solve_started
        rows = []
        for result, route_path in zip(results, route_paths, strict=True):
            write_json(route_path, {
                key: result[key] for key in
                ("instance_id", "seed", "obj", "stats", "routes")})
            rows.append({"instance_id": result["instance_id"],
                         "seed": result["seed"], "obj": result["obj"],
                         "runtime_s": round(result["runtime_s"], 3),
                         "improve_pct": result["stats"]["improve_pct"]})
        write_csv(summary_path, rows)
        metadata = {
            "schema_version": 2,
            "status": "completed",
            "model": "alns",
            "variant": "v2_claude",
            "selection_mode": "roulette",
            "size": args.size,
            "tag": args.tag,
            "run_label": args.run_label,
            "env_backend": "serial" if args.workers == 1 else "process",
            "worker_count": actual_workers,
            "task_count": len(results),
            "solve_wall_seconds": solve_wall_seconds,
            "sum_case_solve_seconds": sum(r["runtime_s"] for r in results),
            "sum_case_load_seconds": sum(r["load_seconds"] for r in results),
            "main_wall_seconds": time.perf_counter() - main_started,
            "main_wall_scope": "main entry through result publication; excludes interpreter/import and final metadata write",
            "completed_at": datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "parameters": parameter_settings,
            "checkpoint_metadata": provider.checkpoint_metadata,
        }
        # Completion metadata is published last, after every case and CSV.
        write_json(metadata_path, metadata)
        print(f"[v2_claude/alns complete] tasks={len(results)} "
              f"solve_wall={solve_wall_seconds:.2f}s output={out_dir}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
