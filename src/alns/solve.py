"""Run vanilla ALNS on immutable, precomputed truck/robot instances.

Workflow: load NPZ matrices -> build an initial feasible route -> repeat
destroy/repair with adaptive roulette selection and annealing -> save results.
Geometry is never reconstructed: distance and travel-time matrices are read
from ``data/processed*/``. The evaluator follows the exact model's objective
and custody, capacity, range, and scheduling constraints.

Run from the repository root, for example::

    python src/alns/solve.py --size 5 --limit 1 --seeds 1 --iterations 20
    python src/alns/solve.py --size 20 --workers 2 --run-label example_20

Use a fresh ``--run-label`` when repeating a run: artifact reservations
prevent accidental overwrites. Output remains under ``output/alns/n<size>/``
(or the existing tagged/run-label subdirectories).

Route representation
--------------------
``Solution.routes`` maps a truck ID to its ordered stops. A customer stop
is ``{"kind": "cust", "c": 1}``; a parking stop is
``{"kind": "park", "p": 6, "deploys": [...]}``. Each robot trip records
``{"r": 1, "custs": [2, 3], "ret_p": 7}``, attached to its launch stop.
The retrieval copy must occur later on the same truck route. Even when
launch and retrieval share a physical location, two distinct copies are
consumed. The truck can serve other stops while the robot is away.

The evaluator tracks whether each robot is aboard, waits at retrieval when
needed, and requires all robots aboard at route end. Robot distance is
accumulated across trips without battery swapping; lateness is a soft cost.
"""

import argparse
import copy
import csv
import json
import math
import multiprocessing
import os
import random
import re
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import yaml

if __package__ in (None, ""):
    # Support both ``python src/alns/solve.py`` and package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.params import (
    DEFAULT_PARAMS_PATH,
    REPO_ROOT,
    load_params,
    load_problem,
)
from common.params import Instance as ProcessedInstance
from common.params import Params as SharedParams


class Solution:
    """Mutable route plan; clone before modifying an ALNS candidate."""

    def __init__(self, K):
        self.routes = {k: [] for k in K}

    def clone(self):
        """Copy nested stops/trips so operators cannot change the incumbent."""
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
        """Return occupied parking-copy IDs, including retrieval stops."""
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
    robot_dist = defaultdict(float)  # accumulated robot distance (51)
    parcels = 0
    feasible = True
    aboard = {r: True for r in pr.R_k[k]}
    pending = {}  # ret_p copy -> [(r, robot arrival), ...]
    robots_used = set()

    prev, b_prev = 0, 0.0  # leave depot-out at minute 0
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
                if not aboard.get(r, False):  # no re-deploy while away
                    feasible = False
                aboard[r] = False
                robots_used.add(r)
                # empty trip / capacity (44)
                if not custs or len(custs) > pr.beta_robot:
                    feasible = False
                if ret_p == p:  # same-copy retrieval banned (24)(25)
                    feasible = False
                parcels += sum(pr.lam[c] for c in custs)
                t = a_node + pr.zeta_unload  # (32) deploy departure
                rprev = p
                for c in custs:
                    ahat_c = t + pr.tau_robot(rprev, c)
                    robot_travel += pr.tau_robot(rprev, c) * pr.robot_arc_coef
                    robot_dist[r] += pr.dist(rprev, c)
                    lateness[c] = max(0.0, ahat_c - pr.l_c[c])
                    t = ahat_c + pr.s_hat
                    rprev = c
                arr_ret = t + pr.tau_robot(rprev, ret_p)  # at retrieval
                robot_travel += pr.tau_robot(rprev, ret_p) * pr.robot_arc_coef
                robot_dist[r] += pr.dist(rprev, ret_p)
                pending.setdefault(ret_p, []).append((r, arr_ret))
            if st["deploys"]:
                # (33) the truck cannot leave before the robot departure
                # (a + zeta_unload).
                b_node = max(b_node, a_node + pr.zeta_unload)
            # --- retrieves at this stop ((25) needs w = 0) ---
            for r, arr in pending.pop(p, []):
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

    cost = (
        pr.gamma_fixed  # truck fixed
        + len(robots_used) * pr.gammahat_fixed  # robot fixed
        + truck_travel
        + robot_travel  # travel
        + pr.gamma_late * sum(lateness.values())
    )  # lateness
    return cost, feasible, lateness, dict(robot_dist)


def eval_solution(pr, sol):
    """Total cost, feasibility and per-component breakdown.

    Coverage (18): a solution missing any customer is infeasible — this
    prevents a (cheaper) solution with dropped customers from becoming
    the incumbent after a failed repair.
    """
    total = 0.0
    feasible = sol.customers() == set(pr.C)
    brk = dict(
        truck_fixed=0.0,
        robot_fixed=0.0,
        truck_travel=0.0,
        robot_travel=0.0,
        lateness=0.0,
    )
    per_truck = {}
    for k, route in sol.routes.items():
        c, ok, late, _ = eval_truck(pr, k, route)
        per_truck[k] = c
        total += c
        feasible = feasible and ok
        if route:
            brk["truck_fixed"] += pr.gamma_fixed
            robots_used = {
                tr["r"]
                for st in route
                if st["kind"] == "park"
                for tr in st["deploys"]
            }
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
                    rt += pr.tau_robot(rprev, tr["ret_p"]) * pr.robot_arc_coef
            prev = node
        if route:
            tt += pr.tau_truck(prev, pr.D) * pr.truck_arc_coef
        brk["truck_travel"] += tt
        brk["robot_travel"] += rt
    return total, feasible, brk, per_truck


# Destroy/repair operators and candidate enumeration.


__all__ = [
    "DESTROY",
    "enum_insertions",
    "best_insertion",
    "apply_insertion",
    "repair_greedy",
    "repair_regret2",
    "remove_customers",
    "destroy_random",
    "destroy_worst",
    "destroy_related",
]

# ---- caps on trip-insertion candidates (combinatorial control) ----
L_RET_EXIST = 3  # existing later parking stops tried as retrieval
W_RET_NEW = 4  # positions after the deploy tried for a new stop
N_PHYS_NEAR = 2  # nearest physical locations tried for a new stop


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
            yield k, (route[:pos] + [{"kind": "cust", "c": c}] + route[pos:])

        park_pos = [
            (si, st) for si, st in enumerate(route) if st["kind"] == "park"
        ]

        # --- B. insertion into an existing trip ---
        for si, st in park_pos:
            for ti, tr in enumerate(st["deploys"]):
                if len(tr["custs"]) >= pr.beta_robot:
                    continue
                for pos in range(len(tr["custs"]) + 1):
                    new_tr = dict(tr)
                    new_tr["custs"] = (
                        tr["custs"][:pos] + [c] + tr["custs"][pos:]
                    )
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
                for sj, st2 in [pp for pp in park_pos if pp[0] > si][
                    :L_RET_EXIST
                ]:
                    yield (
                        k,
                        _with_new_trip(
                            route,
                            si,
                            st,
                            {"r": r, "custs": [c], "ret_p": st2["p"]},
                        ),
                    )
                # ret 2) fresh copy at the same location right behind
                #        (waiting style, consumes two copies)
                grp = pr.park_groups[pr.copy_to_phys[st["p"]]]
                free = [cp for cp in grp if cp not in used]
                if free:
                    nr = _with_new_trip(
                        route, si, st, {"r": r, "custs": [c], "ret_p": free[0]}
                    )
                    nr.insert(
                        si + 1, {"kind": "park", "p": free[0], "deploys": []}
                    )
                    yield k, nr
                # ret 3) fresh copy at a location near c, inserted
                #        within W positions after the deploy
                for gi in pr.phys_near[c][:N_PHYS_NEAR]:
                    grp2 = pr.park_groups[gi]
                    free2 = [
                        cp for cp in grp2 if cp not in used and cp != st["p"]
                    ]
                    if not free2:
                        continue
                    for pos in range(
                        si + 1, min(len(route), si + W_RET_NEW) + 1
                    ):
                        nr = _with_new_trip(
                            route,
                            si,
                            st,
                            {"r": r, "custs": [c], "ret_p": free2[0]},
                        )
                        nr.insert(
                            pos, {"kind": "park", "p": free2[0], "deploys": []}
                        )
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
                        nr.insert(
                            pos,
                            {
                                "kind": "park",
                                "p": dep_cp,
                                "deploys": [
                                    {"r": r, "custs": [c], "ret_p": free[1]}
                                ],
                            },
                        )
                        nr.insert(
                            pos + 1,
                            {"kind": "park", "p": free[1], "deploys": []},
                        )
                        yield k, nr
                    # ret b) later existing parking stops
                    later = [st2 for si2, st2 in park_pos if si2 >= pos][
                        :L_RET_EXIST
                    ]
                    for st2 in later:
                        nr = list(route)
                        nr.insert(
                            pos,
                            {
                                "kind": "park",
                                "p": dep_cp,
                                "deploys": [
                                    {"r": r, "custs": [c], "ret_p": st2["p"]}
                                ],
                            },
                        )
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
        pick = None  # (regret, delta, apply, c)
        for c in remaining:
            base = {k: eval_truck(pr, k, sol.routes[k])[0] for k in pr.K}
            local = {}  # k -> (delta, apply)
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
                    dict(tr, custs=[c for c in tr["custs"] if c not in cset])
                    for tr in st["deploys"]
                ]
                st["deploys"] = [tr for tr in st["deploys"] if tr["custs"]]
                mid.append(st)
        # 2) collect retrieval references of surviving trips, then drop
        #    parking stops without a role
        refs = {
            tr["ret_p"]
            for st in mid
            if st["kind"] == "park"
            for tr in st["deploys"]
        }
        sol.routes[k] = [
            st
            for st in mid
            if st["kind"] == "cust" or st["deploys"] or st["p"] in refs
        ]


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
        contrib.append((base - after, c))  # larger = worse placed
    contrib.sort(reverse=True)
    chosen = []
    pool = contrib[:]
    while pool and len(chosen) < q:
        y = rng.random()
        idx = int((y**p) * len(pool))
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
        idx = int((y**p) * len(cand))
        removed.append(cand[idx])
    remove_customers(sol, removed)
    return removed


DESTROY = [
    ("random", destroy_random),
    ("worst", destroy_worst),
    ("related", destroy_related),
]


# Problem adapter, initial solutions, and adaptive search.


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

    def __init__(
        self, data: ProcessedInstance, config: SharedParams, size: int
    ):
        n_customers = int(data.customer_idx.size)
        if n_customers != size:
            raise ValueError(
                f"requested n{size}, but {data.source_path.name} contains "
                f"{n_customers} customers"
            )

        source_index = np.asarray(
            [
                data.depot_index,
                *data.customer_idx.tolist(),
                *data.parking_idx.tolist(),
                data.depot_index,
            ],
            dtype=np.int64,
        )
        self.source_index = source_index
        self.node_zone = np.ascontiguousarray(data.node_zone[source_index])
        self.labels = tuple(data.node_label[source_index].tolist())
        self.d = np.ascontiguousarray(
            data.d[np.ix_(source_index, source_index)]
        )
        self.tau_truck_matrix = np.ascontiguousarray(
            data.tau_truck[np.ix_(source_index, source_index)]
        )
        self.tau_robot_matrix = np.ascontiguousarray(
            data.tau_robot[np.ix_(source_index, source_index)]
        )
        for array in (
            self.node_zone,
            self.d,
            self.tau_truck_matrix,
            self.tau_robot_matrix,
        ):
            array.setflags(write=False)

        self.C = list(range(1, n_customers + 1))
        first_parking = n_customers + 1
        self.P = list(
            range(first_parking, first_parking + int(data.parking_idx.size))
        )
        self.D = len(source_index) - 1
        self.K = list(range(1, config.fleet.n_trucks_for(size) + 1))
        robots = config.fleet.n_robots_per_truck
        self.R_k = {k: list(range(1, robots + 1)) for k in self.K}
        self.lam = {
            c: int(value) for c, value in zip(self.C, data.demand, strict=True)
        }
        self.e_c = {
            c: float(value) for c, value in zip(self.C, data.e, strict=True)
        }
        self.l_c = {
            c: float(value) for c, value in zip(self.C, data.l, strict=True)
        }
        self.alpha_traffic = data.alpha_traffic
        self.alpha_ped = data.alpha_ped
        self.instance_id = str(data.meta["instance_id"])
        self.preproc_hash = str(data.meta["preproc_hash"])

        # Physical parking groups (copy-budget bookkeeping).
        groups = defaultdict(list)
        for p in self.P:
            groups[self.labels[p].rsplit("#", 1)[0]].append(p)
        self.park_groups = [
            groups[key]
            for key in sorted(groups, key=lambda label: int(label[1:]))
        ]
        self.copy_to_phys = {}  # copy node -> group index
        for gi, grp in enumerate(self.park_groups):
            for cp in grp:
                self.copy_to_phys[cp] = gi
        # Per-customer ranking of physical locations by proximity
        # (candidates for new parking stops).
        self.phys_near = {}
        for c in self.C:
            order = sorted(
                range(len(self.park_groups)),
                key=lambda gi: self.dist(c, self.park_groups[gi][0]),
            )
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
        self.truck_arc_coef = (
            config.truck.fuel_cost_per_min + config.truck.env_cost_per_min
        )
        self.robot_arc_coef = (
            config.robot.fuel_cost_per_min + config.robot.env_cost_per_min
        )

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
        raise RuntimeError(
            "initial solution failed — check truck capacity/range"
        )
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
    return {
        c: (ta[zc[c]] - tmin) / tspan - (pa[zc[c]] - pmin) / pspan
        for c in pr.C
    }


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
            if (
                ddist + pr.dist(last, nxt) + pr.dist(nxt, p1)
                > pr.phi_hat + 1e-9
            ):
                break
            ddist += pr.dist(last, nxt)
            trip.append(nxt)
            rem.remove(nxt)
            last = nxt
        if not trip:  # defensive guard: even a solo round
            leftovers.append(rem.pop(0))  # trip is impossible
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
            if len(grp) >= 2 and 2.0 * pr.dist(grp[0], c) <= pr.phi_hat + 1e-9:
                cand.append((c, gi))
                break
    cand.sort(
        key=lambda t: (-scores[t[0]], pr.dist(pr.park_groups[t[1]][0], t[0]))
    )

    by_group = defaultdict(list)
    for c, gi in cand:
        by_group[gi].append(c)

    # ---- stage 2: trip clustering -> truck/robot assignment ->
    #      skeleton ----
    leftovers = []
    robot_budget = {(k, r): pr.phi_hat for k in pr.K for r in pr.R_k[k]}
    pairs = {k: [] for k in pr.K}
    order = sorted(
        by_group, key=lambda gi: pr.tau_truck(0, pr.park_groups[gi][0])
    )
    for ti, gi in enumerate(order):
        grp = pr.park_groups[gi]
        k = pr.K[ti % len(pr.K)]
        deploys = []
        for custs, ddist in _build_trips(pr, gi, by_group[gi], leftovers):
            # no duplicate robot within one stop pair (custody: no
            # re-deploy while away)
            r = next(
                (
                    r
                    for r in pr.R_k[k]
                    if robot_budget[(k, r)] >= ddist - 1e-9
                    and all(d["r"] != r for d in deploys)
                ),
                None,
            )
            if r is None:
                leftovers.extend(custs)
                continue
            robot_budget[(k, r)] -= ddist
            deploys.append({"r": r, "custs": custs, "ret_p": grp[1]})
        if deploys:
            pairs[k].append(
                (
                    {"kind": "park", "p": grp[0], "deploys": deploys},
                    {"kind": "park", "p": grp[1], "deploys": []},
                )
            )

    sol = Solution(pr.K)
    for k in pr.K:
        sol.routes[k] = [st for pair in pairs[k] for st in pair]
        _, ok, _, _ = eval_truck(pr, k, sol.routes[k])
        if not ok:  # defensive: dissolve skeleton into truck pool
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
    """Draw an index proportionally to its adaptive operator weight."""
    tot = sum(weights)
    y = rng.random() * tot
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if y <= acc:
            return i
    return len(weights) - 1


def solve_alns(
    pr,
    iters=3000,
    seed=0,
    segment=None,
    sigma=(5.0, 3.0, 1.0),
    reaction=0.2,
    w_start=W_START,
    time_limit_s=None,
    initial=None,
    iter_trace=None,
    dod=DOD,
):
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
    current_cost, feas, _, _ = eval_solution(pr, current_solution)
    best_solution, best_cost = current_solution.clone(), current_cost
    init_cost = current_cost

    # Repair operators (noise amplitude scales with instance cost).
    noise_amp = NOISE_FRAC * init_cost
    repair_ops = [
        (
            "greedy",
            lambda p_, s_, pool_, rng_: repair_greedy(
                p_, s_, pool_, rng_, 0.0
            ),
        ),
        (
            "greedy_noise",
            lambda p_, s_, pool_, rng_: repair_greedy(
                p_, s_, pool_, rng_, noise_amp
            ),
        ),
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
    pair_cnt = [0] * n_actions  # a = di * len(repair_ops) + ri
    pair_time = [0.0] * n_actions  # destroy+repair+eval seconds
    sel_time = 0.0  # selector overhead seconds
    accept_cnt = infeas_cnt = best_updates = best_hit_it = 0
    best_trace = []  # (iter, elapsed_s, best_cost)

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
        t_op = time.perf_counter()  # clone+destroy+repair+eval
        cand = current_solution.clone()
        pool = DESTROY[di][1](pr, cand, q_destroy, rng)
        repair_ops[ri][1](pr, cand, pool, rng)
        cand_cost, ok, _, _ = eval_solution(pr, cand)
        pair_cnt[a_idx] += 1
        pair_time[a_idx] += time.perf_counter() - t_op
        dCnt[di] += 1
        rCnt[ri] += 1

        if not ok:  # discard coverage/custody/range violations
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
        if cand_cost < best_cost - 1e-9:
            best_solution, best_cost = cand.clone(), cand_cost
            best_updates += 1
            best_hit_it = it
            best_trace.append(
                (it, round(time.time() - t_start, 3), round(cand_cost, 6))
            )
            reward = sigma[0]
            accept = True
        elif cand_cost < current_cost - 1e-9 and key not in seen:
            reward = sigma[1]
            accept = True
        else:
            if cand_cost < current_cost - 1e-9 or rng.random() < math.exp(
                -(cand_cost - current_cost) / max(T, 1e-9)
            ):
                accept = True
                if key not in seen:
                    reward = sigma[2]
        seen.add(key)
        if accept:
            accept_cnt += 1
            current_solution, current_cost = cand, cand_cost
        if iter_trace is not None:
            iter_trace.append((it, int(accept), round(current_cost, 9)))
        dScore[di] += reward
        rScore[ri] += reward

        if it % segment == 0:  # adaptive weight update
            for i in range(len(DESTROY)):
                if dCnt[i] > 0:
                    dW[i] = dW[i] * (1 - reaction) + reaction * (
                        dScore[i] / dCnt[i]
                    )
                dScore[i] = 0.0
                dCnt[i] = 0
            for i in range(len(repair_ops)):
                if rCnt[i] > 0:
                    rW[i] = rW[i] * (1 - reaction) + reaction * (
                        rScore[i] / rCnt[i]
                    )
                rScore[i] = 0.0
                rCnt[i] = 0

    stats = {
        "init_cost": init_cost,
        "best_cost": best_cost,
        "iters_done": it_done,
        "selector": "roulette",
        "improve_pct": 100.0 * (init_cost - best_cost) / init_cost,
        "destroy_w": dict(
            zip([d[0] for d in DESTROY], [round(x, 3) for x in dW])
        ),
        "repair_w": dict(
            zip([r[0] for r in repair_ops], [round(x, 3) for x in rW])
        ),
        # instrumentation (a = destroy_index * 3 + repair_index)
        "pair_labels": [f"{d[0]}+{r[0]}" for d in DESTROY for r in repair_ops],
        "action_hist": pair_cnt,
        "pair_time_s": [round(x, 3) for x in pair_time],
        "selector_overhead_s": round(sel_time, 3),
        "accept_count": accept_cnt,
        "infeasible_count": infeas_cnt,
        "best_update_count": best_updates,
        "best_first_hit_iter": best_hit_it,
        "best_trace": best_trace,
    }
    return best_solution, best_cost, stats


# Processed-instance loading for experiment entry points.


_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class DirectoryInstanceProvider:
    """Load validated NPZ files and expose immutable ALNS problems.

    File discovery is sorted for reproducible instance ordering. Loaded
    problems are cached by path, while ``sample()`` uses this provider's
    seeded RNG to draw training instances with replacement.

    Example: ``DirectoryInstanceProvider(5, split="test").test_set()``
    returns ``(instance_id, Params)`` pairs without starting any search.
    """

    def __init__(
        self,
        size,
        params_path=None,
        tag=None,
        seed=0,
        train_count=None,
        split="all",
    ):
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be 'train', 'test', or 'all'")
        if tag is not None and not _TAG_PATTERN.fullmatch(tag):
            raise ValueError(
                "invalid tag; use letters, digits, '.', '_', or '-'"
            )
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
        self._train_all = (
            self._load("train") if split in {"train", "all"} else []
        )
        self.test = self._load("test") if split in {"test", "all"} else []
        if train_count is not None and (
            not self._train_all or not 0 < train_count <= len(self._train_all)
        ):
            raise ValueError(
                f"train_count must be in [1, {len(self._train_all)}], "
                f"got {train_count}"
            )
        self.train = (
            self._train_all
            if train_count is None
            else self._train_all[:train_count]
        )
        held_out = len(self._train_all) - len(self.train)
        split_note = f" / {held_out} validation" if held_out else ""
        print(
            f"[data] n{size}: {len(self.train)} train{split_note} / "
            f"{len(self.test)} test instances loaded "
            f"(fleet: {self.fleet})",
            flush=True,
        )

    def _load(self, split):
        path = self.data_root / split / f"n{self.size}"
        if not path.is_dir():
            raise FileNotFoundError(
                f"processed instance directory not found: {path}"
            )
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
                f"{len(self._train_all)} train instances"
            )
        return [(path.stem, self._params(path)) for path in paths]


# Artifact publication is local to this CLI. Reserve the complete output
# set before solving, then publish each file atomically after success.
def validate_run_label(label):
    if label is None:
        return None
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", label
    ) or label.upper().split(".")[0] in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError(
            "--run-label must be 1-80 ASCII letters/digits/_/-, "
            "start with a letter/digit, and not be a Windows device name"
        )
    return label


@contextmanager
def reserve_artifacts(paths):
    """Fail before expensive work if any target exists or another run owns it."""
    targets = sorted(
        {Path(path).expanduser().resolve() for path in paths}, key=str
    )
    if not targets:
        raise ValueError("at least one artifact target is required")
    locks = []
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite {target}. Use a new --run-label."
                )
            lock = target.with_name(f".{target.name}.lock")
            try:
                handle = lock.open("x", encoding="utf-8")
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Output reserved: {lock}. Choose a new --run-label; only remove "
                    "a stale lock after verifying its recorded PID has no active run."
                ) from exc
            locks.append(lock)
            with handle:
                json.dump({"pid": os.getpid(), "target": str(target)}, handle)
        for target in targets:
            if target.exists():
                raise FileExistsError(
                    f"Artifact appeared during reservation: {target}"
                )
        yield
    finally:
        for lock in reversed(locks):
            lock.unlink(missing_ok=True)


@contextmanager
def atomic_open(path, mode="w", *, encoding="utf-8", newline=None):
    """Write beside the destination, then replace it only on successful close.

    Intended replacement is allowed (periodic checkpoints within a reserved run).
    An exception leaves an existing destination untouched and removes only this
    call's temporary file. This is per-file atomicity, not a multi-file transaction.
    """
    if mode not in {"w", "wb"}:
        raise ValueError("atomic_open only supports w or wb")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary)
    try:
        options = (
            {} if "b" in mode else {"encoding": encoding, "newline": newline}
        )
        with os.fdopen(fd, mode, **options) as handle:
            fd = None
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    with atomic_open(path) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to publish an empty result CSV")
    with atomic_open(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
        params, iters=iterations, seed=seed, iter_trace=iter_trace
    )
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
            result = _case_result(
                task, size, params_path, cache, capture_trace
            )
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
            connection.send(
                ("error", worker_id, task_id, phase, traceback.format_exc())
            )
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
    remaining = [
        process for process in processes.values() if process.is_alive()
    ]
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
            print(
                f"[alns cleanup] worker PID {process.pid} did not exit; "
                "inspect this PID before manual recovery",
                flush=True,
            )
        else:
            process.close()


def run_cases(
    instance_paths,
    *,
    size,
    params_path=DEFAULT_PARAMS_PATH,
    iterations=100,
    seeds=5,
    workers=1,
    capture_trace=False,
    progress=True,
    heartbeat_interval=30.0,
    startup_timeout=180.0,
):
    """Return case results in the original instance-major, seed-minor order.

    ``workers=1`` is the serial baseline. More workers parallelize whole
    independent solves, not the dependent iterations within a solve. The
    same seed is used for a case regardless of which worker receives it.
    Each worker loads and caches its own immutable Params from file paths.
    No output files are written here, so verification cannot overwrite runs.
    """
    for name, value in (
        ("iterations", iterations),
        ("seeds", seeds),
        ("workers", workers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if workers > 30:
        raise ValueError("workers must be <= 30 (pipe + process wait handles)")
    if heartbeat_interval <= 0 or startup_timeout <= 0:
        raise ValueError(
            "heartbeat_interval and startup_timeout must be positive"
        )
    paths = [str(Path(path).expanduser().resolve()) for path in instance_paths]
    if not paths:
        raise ValueError("at least one instance path is required")
    params_path = str(Path(params_path).expanduser().resolve())
    tasks = [
        (index * seeds + seed, path, seed, iterations)
        for index, path in enumerate(paths)
        for seed in range(seeds)
    ]
    results = [None] * len(tasks)
    if workers == 1:
        cache = {}
        for task in tasks:
            if progress:
                print(
                    f"[alns {task[0] + 1}/{len(tasks)}] "
                    f"instance={Path(task[1]).stem} seed={task[2]} starting",
                    flush=True,
                )
            result = _case_result(
                task, size, params_path, cache, capture_trace
            )
            results[task[0]] = result
            if progress:
                print(
                    f"[alns {task[0] + 1}/{len(tasks)}] "
                    f"obj={result['obj']:.6f} "
                    f"solve={result['runtime_s']:.2f}s",
                    flush=True,
                )
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
                name=f"ALNS-case-{worker_id}",
                daemon=False,
            )
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
                if (
                    not isinstance(message, tuple)
                    or len(message) != 3
                    or message[:2] != ("ready", worker_id)
                    or message[2] != processes[worker_id].pid
                ):
                    raise RuntimeError(
                        f"worker {worker_id}: invalid READY {message!r}"
                    )
                ready_workers.add(worker_id)
                idle.add(worker_id)
            for worker_id, process in processes.items():
                if not process.is_alive():
                    raise RuntimeError(
                        f"worker {worker_id} died during startup; "
                        f"exitcode={process.exitcode}"
                    )
            now = time.monotonic()
            if (
                len(ready_workers) < worker_count
                and now - startup_started >= startup_timeout
            ):
                raise TimeoutError(
                    f"ALNS worker startup timed out: {sorted(pending)}"
                )
            if progress and now - last_heartbeat >= heartbeat_interval:
                print(
                    f"[alns startup] waiting={sorted(set(processes) - ready_workers)} "
                    f"elapsed={now - startup_started:.1f}s",
                    flush=True,
                )
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
                        f"task {assignments[worker_id][0]}"
                    ) from error
                expected_id = assignments[worker_id][0]
                if (
                    not isinstance(message, tuple)
                    or len(message) < 3
                    or message[1:3] != (worker_id, expected_id)
                ):
                    raise RuntimeError(
                        f"worker {worker_id}: invalid response {message!r}"
                    )
                if message[0] == "error" and len(message) == 5:
                    raise RuntimeError(
                        f"ALNS worker {worker_id}, task {expected_id}, "
                        f"phase={message[3]} failed:\n{message[4]}"
                    )
                if message[0] != "ok" or len(message) != 4:
                    raise RuntimeError(
                        f"worker {worker_id}: invalid response {message!r}"
                    )
                result = message[3]
                if (
                    result.get("task_id") != expected_id
                    or result.get("instance_id")
                    != Path(tasks[expected_id][1]).stem
                    or result.get("seed") != tasks[expected_id][2]
                    or results[expected_id] is not None
                ):
                    raise RuntimeError(
                        f"worker {worker_id}: mismatched result for task {expected_id}"
                    )
                results[expected_id] = result
                del assignments[worker_id]
                idle.add(worker_id)
                completed += 1
                if progress:
                    print(
                        f"[alns completed={completed}/{len(tasks)}] "
                        f"task={expected_id} worker={worker_id} "
                        f"instance={result['instance_id']} seed={result['seed']} "
                        f"obj={result['obj']:.6f} "
                        f"solve={result['runtime_s']:.2f}s",
                        flush=True,
                    )
            for worker_id, process in processes.items():
                if not process.is_alive():
                    # The process sentinel can become ready just after wait's
                    # pipe snapshot. Preserve an already queued traceback.
                    if (
                        worker_id in assignments
                        and connections[worker_id].poll()
                    ):
                        try:
                            message = connections[worker_id].recv()
                        except (EOFError, OSError):
                            message = None
                        if (
                            isinstance(message, tuple)
                            and len(message) == 5
                            and message[:3]
                            == ("error", worker_id, assignments[worker_id][0])
                        ):
                            raise RuntimeError(
                                f"ALNS worker {worker_id}, task={message[2]}, "
                                f"phase={message[3]} failed:\n{message[4]}"
                            )
                    raise RuntimeError(
                        f"ALNS worker {worker_id} died; exitcode={process.exitcode}"
                    )
            now = time.monotonic()
            if progress and now - last_heartbeat >= heartbeat_interval:
                waiting = ", ".join(
                    f"w{i}:task={task_id},elapsed={now - started:.1f}s"
                    for i, (task_id, started) in sorted(assignments.items())
                )
                print(f"[alns waiting] {waiting}", flush=True)
                last_heartbeat = now
        return results
    finally:
        _close_case_workers(processes, connections, idle)


def _parser():
    parser = argparse.ArgumentParser(
        description="Run vanilla ALNS on precomputed test instances"
    )
    parser.add_argument(
        "--size", type=int, choices=(5, 10, 20, 50, 100), required=True
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="number of repetition seeds, starting at zero",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent case processes (1-30); 1 keeps the serial baseline",
    )
    parser.add_argument(
        "--run-label",
        help="unique output label; existing artifacts are never overwritten",
    )
    return parser


def main():
    """Parse the CLI, solve independent cases, and publish their artifacts."""
    main_started = time.perf_counter()
    args = _parser().parse_args()
    if args.iterations <= 0 or args.seeds <= 0 or args.workers <= 0:
        raise ValueError(
            "--iterations, --seeds and --workers must be positive"
        )
    if args.workers > 30:
        raise ValueError("--workers must be <= 30")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    validate_run_label(args.run_label)
    provider = DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag, split="test"
    )
    params_path = args.params.expanduser().resolve()
    with params_path.open("r", encoding="utf-8") as f:
        parameter_settings = yaml.safe_load(f)
    # Send paths, not Params (which contains non-picklable immutable config).
    cases = provider.test
    if args.limit is not None:
        cases = cases[: args.limit]
    method_dir = "alns" if args.tag is None else f"alns_{args.tag}"
    out_dir = REPO_ROOT / "output" / method_dir / f"n{args.size}"
    if args.run_label is not None:
        out_dir = out_dir / "runs" / args.run_label
    route_paths = [
        out_dir / f"{path.stem}_s{seed}.json"
        for path in cases
        for seed in range(args.seeds)
    ]
    summary_path = out_dir / "summary.csv"
    metadata_path = out_dir / "test_metadata.json"
    with reserve_artifacts([*route_paths, summary_path, metadata_path]):
        actual_workers = min(args.workers, len(route_paths))
        print(
            f"[alns] cases={len(cases)} seeds={args.seeds} "
            f"tasks={len(route_paths)} iterations={args.iterations} "
            f"workers={actual_workers} "
            f"backend={'serial' if args.workers == 1 else 'process'} "
            f"run_label={args.run_label!r}",
            flush=True,
        )
        solve_started = time.perf_counter()
        results = run_cases(
            cases,
            size=args.size,
            params_path=params_path,
            iterations=args.iterations,
            seeds=args.seeds,
            workers=args.workers,
        )
        solve_wall_seconds = time.perf_counter() - solve_started
        rows = []
        for result, route_path in zip(results, route_paths, strict=True):
            write_json(
                route_path,
                {
                    key: result[key]
                    for key in (
                        "instance_id",
                        "seed",
                        "obj",
                        "stats",
                        "routes",
                    )
                },
            )
            rows.append(
                {
                    "instance_id": result["instance_id"],
                    "seed": result["seed"],
                    "obj": result["obj"],
                    "runtime_s": round(result["runtime_s"], 3),
                    "improve_pct": result["stats"]["improve_pct"],
                }
            )
        write_csv(summary_path, rows)
        metadata = {
            "schema_version": 2,
            "status": "completed",
            "model": "alns",
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
            "completed_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "parameters": parameter_settings,
            "checkpoint_metadata": provider.checkpoint_metadata,
        }
        # Completion metadata is published last, after every case and CSV.
        write_json(metadata_path, metadata)
        print(
            f"[alns complete] tasks={len(results)} "
            f"solve_wall={solve_wall_seconds:.2f}s output={out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
