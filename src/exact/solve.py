"""Exact MILP over immutable, precomputed truck-robot instances.

Implements the clean equation numbers (1)-(70) of the final
formulation (battery swapping removed). Key ingredients:
  * Parking-node copies P-bar: ``num_parking_copies`` co-located copies
    per physical location, so a truck may revisit a physical location
    (one deploy-retrieve round trip consumes two copies).
  * Robot custody w_{kri} with constraints (19)-(25): deploy/retrieve
    strictly alternate along the truck's visiting order, preventing a
    robot from being re-deployed before it is retrieved.
  * Parcel conservation (49): a robot leaves for a retrieval arc empty.
  * No battery swapping: robots start fully charged and neither swap
    nor recharge en route, giving the single constraint (51)
    "total accumulated route distance <= phi-hat".
  * Truck driving range (50), standard MTZ (52)-(55).
  * Objective (3): truck fixed + robot fixed + truck travel (fuel +
    environmental) + robot travel (fuel + environmental) + lateness.

All distance and congestion-adjusted travel-time coefficients come directly
from ``data/processed*/``. The MILP performs combinatorial optimization only;
it never reconstructs geometry from node coordinates.

Workflow: read a processed NPZ -> reindex depot/customer/parking nodes ->
build the MILP -> optimize -> export JSON, Gurobi log, and summary CSV.
Equation-number comments connect each constraint block to the formulation.
Variable names keep that mathematical notation so the correspondence stays
visible when checking the model.

Run from the repository root, for example::

    python src/exact/solve.py --size 5 --limit 1 --time-limit 60
    python src/exact/solve.py --size 10 --limit 1 --mip-gap 0.01

Output is written to ``output/exact/n<size>/``; ``--tag`` selects the
matching preprocessed dataset and existing tagged output directory.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT, load_problem
from common.params import Instance as ProcessedInstance
from common.params import Params as SharedParams
from common.sizes import SUPPORTED_SIZES

STATUS_LOOKUP = {
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
}


def run_model(
    inst,
    config,
    model_name,
    time_limit_sec=300,
    mip_gap=0.0,
    symmetry_breaking=True,
    fix_binaries=None,
    threads=0,
    mip_focus=2,
    nodefile_gb=10,
    log_path=None,
):
    """Build and solve one MILP instance; return its result dictionary.

    ``inst`` uses the 0/C/P/D node convention produced by
    ``_to_exact_instance``. ``config`` contains the shared YAML parameters.
    ``log_path`` selects a Gurobi log file (None disables file logging).

    The returned dictionary always includes status and model dimensions.
    Objective components, selected arcs, and schedule diagnostics are added
    only if Gurobi finds an incumbent. ``fix_binaries`` optionally pins a
    heuristic route for independent feasibility/cost verification.
    """
    C, P, D = inst["C"], inst["P"], inst["D"]  # P = P-bar (all copies)
    K, R_k, lam = inst["K"], inst["R_k"], inst["lam"]
    e_c, l_c = inst["e_c"], inst["l_c"]

    N = C + P + [0, D]
    N0 = C + P + [0]
    Nplus = C + P + [D]
    N0_hat = C + P
    Nplus_hat = C + P
    P_set = set(P)

    d_ij = {(i, j): float(inst["d"][i, j]) for i in N for j in N if i != j}
    tau_truck = {
        (i, j): float(inst["tau_truck"][i, j])
        for i in N0
        for j in Nplus
        if i != j
    }
    tau_robot = {
        (i, j): float(inst["tau_robot"][i, j])
        for i in N0_hat
        for j in Nplus_hat
        if i != j
    }

    # ---- parameters ----
    s_kc = {(k, c): config.truck.service_time_min for k in K for c in C}
    s_hat = {
        (k, r, c): config.robot.service_time_min
        for k in K
        for r in R_k[k]
        for c in C
    }
    zeta_load = {
        (k, r, p): config.robot.load_time_min
        for k in K
        for r in R_k[k]
        for p in P
    }
    zeta_unload = {
        (k, r, p): config.robot.unload_time_min
        for k in K
        for r in R_k[k]
        for p in P
    }
    beta_k = {k: config.truck.capacity for k in K}
    beta_hat = {(k, r): config.robot.capacity for k in K for r in R_k[k]}
    phi_hat_param = {(k, r): config.robot.range_km for k in K for r in R_k[k]}
    phi_truck_param = {k: config.truck.range_km for k in K}

    gamma_late = {c: config.lateness_cost_per_min for c in C}
    gamma_fixed = {k: config.truck.fixed_cost for k in K}
    gammahat_fixed = {
        (k, r): config.robot.fixed_cost for k in K for r in R_k[k]
    }
    gamma_fuel = {k: config.truck.fuel_cost_per_min for k in K}
    gamma_env = {k: config.truck.env_cost_per_min for k in K}
    gammahat_fuel = {
        (k, r): config.robot.fuel_cost_per_min for k in K for r in R_k[k]
    }
    gammahat_env = {
        (k, r): config.robot.env_cost_per_min for k in K for r in R_k[k]
    }

    # ---- tight big-M values (formulation, notes for implementation) ----
    max_tau = max(
        max(tau_truck.values(), default=0.0),
        max(tau_robot.values(), default=0.0),
    )
    svc_total = sum(max((s_kc[(k, c)] for k in K), default=0.0) for c in C)
    sync_total = sum(
        max(
            (
                zeta_unload[(k, r, p)] + zeta_load[(k, r, p)]
                for k in K
                for r in R_k[k]
            ),
            default=0.0,
        )
        for p in P
    )
    T_max = (
        max(l_c.values(), default=0.0)
        + len(N) * max_tau
        + svc_total
        + sync_total
    )
    M_time = T_max + max_tau  # time propagation (26)-(38)
    max_lam = max(lam.values())
    # (40)-(43)
    M_q = config.truck.capacity + max(
        max_lam, max(len(R_k[k]) for k in K) * config.robot.capacity
    )
    M_qhat = config.robot.capacity + max_lam  # (46)-(48)
    M_wP = 2  # custody propagation at parking (20)-(21) — tight
    M_wpass = 1  # custody propagation elsewhere (22)-(23) — tight

    m = gp.Model(model_name)
    m.Params.OutputFlag = 1
    m.Params.LogToConsole = 0
    if log_path is not None:
        m.Params.LogFile = log_path
    m.Params.TimeLimit = time_limit_sec
    m.Params.MIPGap = mip_gap
    m.Params.Threads = threads
    if mip_focus:
        m.Params.MIPFocus = mip_focus
    if nodefile_gb:
        m.Params.NodefileStart = nodefile_gb
    m.Params.Cuts = 2

    # ---- variables (56)-(70) ----
    # Robot parking-to-parking arcs are excluded at variable creation,
    # which enforces (13) y_krpq = 0 structurally. The direct depot arc
    # (0 -> D) is also excluded so a dispatched truck cannot do nothing
    # (code-level restriction only).
    # (56)
    x = m.addVars(
        [
            (k, i, j)
            for k in K
            for i in N0
            for j in Nplus
            if i != j and not (i == 0 and j == D)
        ],
        vtype=GRB.BINARY,
        name="x",
    )
    # (57),(13)
    y = m.addVars(
        [
            (k, r, i, j)
            for k in K
            for r in R_k[k]
            for i in N0_hat
            for j in Nplus_hat
            if i != j and not (i in P_set and j in P_set)
        ],
        vtype=GRB.BINARY,
        name="y",
    )
    # (58) custody
    w = m.addVars(
        [(k, r, i) for k in K for r in R_k[k] for i in N],
        vtype=GRB.BINARY,
        name="w",
    )
    u = m.addVars(K, vtype=GRB.BINARY, name="u")  # (59)
    uhat = m.addVars(
        [(k, r) for k in K for r in R_k[k]], vtype=GRB.BINARY, name="uhat"
    )  # (60)
    eta = m.addVars(
        [(k, r, p) for k in K for r in R_k[k] for p in P],
        vtype=GRB.INTEGER,
        lb=0,
        name="eta",
    )  # (61)
    q = m.addVars(
        [(k, i) for k in K for i in N0], vtype=GRB.INTEGER, lb=0, name="q"
    )  # (62)
    qhat = m.addVars(
        [(k, r, i) for k in K for r in R_k[k] for i in N0_hat],
        vtype=GRB.INTEGER,
        lb=0,
        name="qhat",
    )  # (63)
    delta = m.addVars(C, lb=0.0, vtype=GRB.CONTINUOUS, name="delta")  # (64)
    a = m.addVars(
        [(k, i) for k in K for i in Nplus],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="a",
    )  # (65)
    b = m.addVars(
        [(k, i) for k in K for i in N0], lb=0.0, vtype=GRB.CONTINUOUS, name="b"
    )  # (66)
    ahat = m.addVars(
        [(k, r, i) for k in K for r in R_k[k] for i in Nplus_hat],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="ahat",
    )  # (67)
    bhat = m.addVars(
        [(k, r, i) for k in K for r in R_k[k] for i in N0_hat],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="bhat",
    )  # (68)
    pi = m.addVars(
        [(k, i) for k in K for i in C + P],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="pi",
    )  # (69)
    pihat = m.addVars(
        [(k, r, c) for k in K for r in R_k[k] for c in C],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="pihat",
    )  # (70)

    for k in K:
        for i in N0:
            q[k, i].UB = beta_k[k]  # (62)
        for r in R_k[k]:
            for i in N0_hat:
                qhat[k, r, i].UB = beta_hat[(k, r)]  # (63)
            for p in P:
                eta[k, r, p].UB = beta_hat[(k, r)]  # (61)
    for k in K:  # time variables bounded by T_max
        for i in Nplus:
            a[k, i].UB = T_max
        for i in N0:
            b[k, i].UB = T_max
        for r in R_k[k]:
            for i in Nplus_hat:
                ahat[k, r, i].UB = T_max
            for i in N0_hat:
                bhat[k, r, i].UB = T_max

    # ---- objective (3): five components ----
    obj_tfix = gp.quicksum(gamma_fixed[k] * u[k] for k in K)
    obj_rfix = gp.quicksum(
        gammahat_fixed[(k, r)] * uhat[k, r] for k in K for r in R_k[k]
    )
    obj_late = gp.quicksum(gamma_late[c] * delta[c] for c in C)
    obj_truck = gp.quicksum(
        x[k, i, j] * tau_truck[(i, j)] * (gamma_fuel[k] + gamma_env[k])
        for (k, i, j) in x.keys()
    )
    obj_robot = gp.quicksum(
        y[k, r, i, j]
        * tau_robot[(i, j)]
        * (gammahat_fuel[(k, r)] + gammahat_env[(k, r)])
        for (k, r, i, j) in y.keys()
    )
    # The next two expressions are reporting-only (already included in
    # the travel terms above).
    obj_truck_env = gp.quicksum(
        x[k, i, j] * tau_truck[(i, j)] * gamma_env[k] for (k, i, j) in x.keys()
    )
    obj_robot_env = gp.quicksum(
        y[k, r, i, j] * tau_robot[(i, j)] * gammahat_env[(k, r)]
        for (k, r, i, j) in y.keys()
    )
    m.setObjective(
        obj_tfix + obj_rfix + obj_truck + obj_robot + obj_late, GRB.MINIMIZE
    )

    # -------- 3.5.2.1 Truck routing (4)-(9) --------
    for k in K:
        m.addConstr(
            gp.quicksum(x[k, 0, j] for j in Nplus if (k, 0, j) in x) == u[k]
        )  # (4)
        m.addConstr(
            gp.quicksum(x[k, i, D] for i in N0 if (k, i, D) in x) == u[k]
        )  # (5)
    for k, i, j in x.keys():
        m.addConstr(x[k, i, j] <= u[k])  # (6)
    for k in K:
        for j in C + P:
            inflow = gp.quicksum(x[k, i, j] for i in N0 if (k, i, j) in x)
            outflow = gp.quicksum(x[k, j, h] for h in Nplus if (k, j, h) in x)
            m.addConstr(inflow - outflow == 0)  # (7)
            m.addConstr(inflow <= 1)  # (8)
            m.addConstr(outflow <= 1)  # (9)

    # -------- 3.5.2.2 Robot routing (10)-(13) --------
    for k, r, i, j in y.keys():
        m.addConstr(y[k, r, i, j] <= uhat[k, r])  # (10)
    for k in K:
        for r in R_k[k]:
            for c in C:
                inflow = gp.quicksum(
                    y[k, r, i, c] for i in N0_hat if (k, r, i, c) in y
                )
                outflow = gp.quicksum(
                    y[k, r, c, j] for j in Nplus_hat if (k, r, c, j) in y
                )
                m.addConstr(inflow - outflow == 0)  # (11)
            dep = gp.quicksum(
                y[k, r, p, c] for p in P for c in C if (k, r, p, c) in y
            )
            ret = gp.quicksum(
                y[k, r, c, p] for p in P for c in C if (k, r, c, p) in y
            )
            m.addConstr(dep == ret)  # (12)
            # No robot use without a deployment (code-level).
            m.addConstr(uhat[k, r] <= dep)
            # (13) no parking-to-parking robot arcs — excluded
            # structurally at variable creation.

    # -------- 3.5.2.3 Truck-robot coordination (14)-(18) --------
    for k in K:
        for r in R_k[k]:
            m.addConstr(uhat[k, r] <= u[k])  # (14)
            for p in P:
                truck_visit = gp.quicksum(
                    x[k, i, p] for i in N0 if (k, i, p) in x
                )
                m.addConstr(
                    gp.quicksum(y[k, r, p, c] for c in C if (k, r, p, c) in y)
                    <= truck_visit
                )  # (15)
                m.addConstr(
                    gp.quicksum(y[k, r, c, p] for c in C if (k, r, c, p) in y)
                    <= truck_visit
                )  # (16)
    for k in K:
        for p in P:
            truck_visit = gp.quicksum(x[k, i, p] for i in N0 if (k, i, p) in x)
            robot_interact = gp.quicksum(
                y[k, r, p, c] + y[k, r, c, p]
                for r in R_k[k]
                for c in C
                if (k, r, p, c) in y and (k, r, c, p) in y
            )
            m.addConstr(truck_visit <= robot_interact)  # (17)
    for c in C:
        truck_served = gp.quicksum(
            x[k, i, c] for k in K for i in N0 if (k, i, c) in x
        )
        robot_served = gp.quicksum(
            y[k, r, i, c]
            for k in K
            for r in R_k[k]
            for i in N0_hat
            if (k, r, i, c) in y
        )
        m.addConstr(truck_served + robot_served == 1)  # (18)

    # -------- 3.5.2.4 Robot custody & delivery-trip sequencing
    #          (19)-(25) --------
    for k in K:
        for r in R_k[k]:
            m.addConstr(w[k, r, 0] == uhat[k, r])  # (19) depot-out
            m.addConstr(w[k, r, D] == uhat[k, r])  # (19) depot-in
    for k, i, j in x.keys():
        for r in R_k[k]:
            if i in P_set:  # arcs leaving a parking node
                dep_p = gp.quicksum(
                    y[k, r, i, c] for c in C if (k, r, i, c) in y
                )  # deploy
                ret_p = gp.quicksum(
                    y[k, r, c, i] for c in C if (k, r, c, i) in y
                )  # retrieve
                m.addConstr(
                    w[k, r, j]
                    >= w[k, r, i] - dep_p + ret_p - M_wP * (1 - x[k, i, j])
                )  # (20)
                m.addConstr(
                    w[k, r, j]
                    <= w[k, r, i] - dep_p + ret_p + M_wP * (1 - x[k, i, j])
                )  # (21)
            else:  # arcs leaving depot/customer: state carries over
                m.addConstr(
                    w[k, r, j] >= w[k, r, i] - M_wpass * (1 - x[k, i, j])
                )  # (22)
                m.addConstr(
                    w[k, r, j] <= w[k, r, i] + M_wpass * (1 - x[k, i, j])
                )  # (23)
    for k in K:
        for r in R_k[k]:
            for p in P:
                # (24) deploying requires the robot aboard
                m.addConstr(
                    w[k, r, p]
                    >= gp.quicksum(
                        y[k, r, p, c] for c in C if (k, r, p, c) in y
                    )
                )
                # (25) retrieving requires the robot away
                m.addConstr(
                    w[k, r, p]
                    <= 1
                    - gp.quicksum(
                        y[k, r, c, p] for c in C if (k, r, c, p) in y
                    )
                )

    # -------- 3.5.2.5 Temporal (26)-(38) --------
    for k, i, j in x.keys():
        m.addConstr(
            a[k, j] >= b[k, i] + tau_truck[(i, j)] - M_time * (1 - x[k, i, j])
        )  # (26)
    for k, r, i, j in y.keys():
        m.addConstr(
            ahat[k, r, j]
            >= bhat[k, r, i] + tau_robot[(i, j)] - M_time * (1 - y[k, r, i, j])
        )  # (27)
    for k in K:
        for p in P:
            m.addConstr(b[k, p] >= a[k, p])  # (28)
            for r in R_k[k]:
                m.addConstr(bhat[k, r, p] >= ahat[k, r, p])  # (29)
        for c in C:
            visit_t = gp.quicksum(x[k, i, c] for i in N0 if (k, i, c) in x)
            m.addConstr(
                b[k, c] >= a[k, c] + s_kc[(k, c)] - M_time * (1 - visit_t)
            )  # (30)
            for r in R_k[k]:
                visit_r = gp.quicksum(
                    y[k, r, i, c] for i in N0_hat if (k, r, i, c) in y
                )
                m.addConstr(
                    bhat[k, r, c]
                    >= ahat[k, r, c]
                    + s_hat[(k, r, c)]
                    - M_time * (1 - visit_r)
                )  # (31)
    for k in K:
        for r in R_k[k]:
            for p in P:
                dep_p = gp.quicksum(
                    y[k, r, p, c] for c in C if (k, r, p, c) in y
                )
                ret_p = gp.quicksum(
                    y[k, r, c, p] for c in C if (k, r, c, p) in y
                )
                m.addConstr(
                    bhat[k, r, p]
                    >= a[k, p] + zeta_unload[(k, r, p)] - M_time * (1 - dep_p)
                )  # (32)
                m.addConstr(
                    bhat[k, r, p] <= b[k, p] + M_time * (1 - dep_p)
                )  # (33)
                # (34) — no swap term
                m.addConstr(
                    b[k, p]
                    >= ahat[k, r, p]
                    + zeta_load[(k, r, p)]
                    - M_time * (1 - ret_p)
                )
    for k in K:
        for c in C:
            visit_t = gp.quicksum(x[k, i, c] for i in N0 if (k, i, c) in x)
            m.addConstr(a[k, c] >= e_c[c] - M_time * (1 - visit_t))  # (35)
            m.addConstr(
                delta[c] >= a[k, c] - l_c[c] - M_time * (1 - visit_t)
            )  # (37)
            for r in R_k[k]:
                visit_r = gp.quicksum(
                    y[k, r, i, c] for i in N0_hat if (k, r, i, c) in y
                )
                m.addConstr(
                    ahat[k, r, c] >= e_c[c] - M_time * (1 - visit_r)
                )  # (36)
                m.addConstr(
                    delta[c] >= ahat[k, r, c] - l_c[c] - M_time * (1 - visit_r)
                )  # (38)

    # -------- 3.5.2.6 Capacity & driving range (39)-(51) --------
    for k in K:
        m.addConstr(q[k, 0] <= beta_k[k] * u[k])  # (39)
        for c in C:
            for i in N0:
                if i != c and (k, i, c) in x:
                    m.addConstr(
                        q[k, i] - lam[c] - M_q * (1 - x[k, i, c]) <= q[k, c]
                    )  # (40)
                    m.addConstr(
                        q[k, c] <= q[k, i] - lam[c] + M_q * (1 - x[k, i, c])
                    )  # (41)
        for p in P:
            for i in N0:
                if i != p and (k, i, p) in x:
                    eta_sum = gp.quicksum(eta[k, r, p] for r in R_k[k])
                    m.addConstr(
                        q[k, i] - eta_sum - M_q * (1 - x[k, i, p]) <= q[k, p]
                    )  # (42)
                    m.addConstr(
                        q[k, p] <= q[k, i] - eta_sum + M_q * (1 - x[k, i, p])
                    )  # (43)
    for k in K:
        for r in R_k[k]:
            for p in P:
                dep_p = gp.quicksum(
                    y[k, r, p, c] for c in C if (k, r, p, c) in y
                )
                m.addConstr(eta[k, r, p] <= beta_hat[(k, r)] * dep_p)  # (44)
                m.addConstr(qhat[k, r, p] == eta[k, r, p])  # (45)
            for c in C:
                for i in N0_hat:
                    if i != c and (k, r, i, c) in y:
                        m.addConstr(
                            qhat[k, r, i]
                            - lam[c]
                            - M_qhat * (1 - y[k, r, i, c])
                            <= qhat[k, r, c]
                        )  # (46)
                        m.addConstr(
                            qhat[k, r, c]
                            <= qhat[k, r, i]
                            - lam[c]
                            + M_qhat * (1 - y[k, r, i, c])
                        )  # (47)
                        m.addConstr(
                            qhat[k, r, i]
                            >= lam[c] - M_qhat * (1 - y[k, r, i, c])
                        )  # (48)
            for c in C:  # (49) parcel conservation
                for p in P:
                    if (k, r, c, p) in y:
                        m.addConstr(
                            qhat[k, r, c]
                            <= beta_hat[(k, r)] * (1 - y[k, r, c, p])
                        )
    for k in K:
        total_truck_dist = gp.quicksum(
            x[k, i, j] * d_ij[(i, j)] for (kk, i, j) in x.keys() if kk == k
        )
        m.addConstr(total_truck_dist <= phi_truck_param[k] * u[k])  # (50)
        # (51) robot range: total accumulated route distance bounded by
        # the full-charge range (no swapping or recharging).
        for r in R_k[k]:
            m.addConstr(
                gp.quicksum(
                    y[k, r, i, j] * d_ij[(i, j)]
                    for (kk, rr, i, j) in y.keys()
                    if kk == k and rr == r
                )
                <= phi_hat_param[(k, r)] * uhat[k, r]
            )

    # -------- 3.5.2.7 Subtour elimination — standard MTZ (52)-(55) ----
    nT = len(C) + len(P)
    for k in K:
        for i in C + P:
            out_i = gp.quicksum(x[k, i, j] for j in Nplus if (k, i, j) in x)
            m.addConstr(pi[k, i] <= nT * out_i)  # (52)
            for j in C + P:
                if i != j and (k, i, j) in x:
                    m.addConstr(
                        pi[k, j] >= pi[k, i] + 1 - nT * (1 - x[k, i, j])
                    )  # (53)
    nC = len(C)
    for k in K:
        for r in R_k[k]:
            for c in C:
                out_c = gp.quicksum(
                    y[k, r, c, j] for j in Nplus_hat if (k, r, c, j) in y
                )
                m.addConstr(pihat[k, r, c] <= nC * out_c)  # (54)
            for i in C:
                for j in C:
                    if i != j and (k, r, i, j) in y:
                        m.addConstr(
                            pihat[k, r, j]
                            >= pihat[k, r, i] + 1 - nC * (1 - y[k, r, i, j])
                        )  # (55)

    # -------- (optional) symmetry breaking — valid inequalities
    #          outside the formulation (non-normative) --------
    # Removes exchange symmetry of parking copies and of homogeneous
    # trucks/robots; never cuts off an optimal solution.
    if symmetry_breaking:
        for ki in range(len(K) - 1):
            m.addConstr(u[K[ki]] >= u[K[ki + 1]])
        for k in K:
            rs = R_k[k]
            for ri in range(len(rs) - 1):
                m.addConstr(uhat[k, rs[ri]] >= uhat[k, rs[ri + 1]])
        for grp in inst["park_groups"]:
            for ci in range(len(grp) - 1):
                p_cur, p_nxt = grp[ci], grp[ci + 1]
                use_cur = gp.quicksum(
                    x[k, i, p_cur] for k in K for i in N0 if (k, i, p_cur) in x
                )
                use_nxt = gp.quicksum(
                    x[k, i, p_nxt] for k in K for i in N0 if (k, i, p_nxt) in x
                )
                m.addConstr(use_nxt <= use_cur)

    # -------- optional binary fixing (verification mode) --------
    # fix_binaries = {"x": {(k, i, j)}, "y": {(k, r, i, j)},
    #                 "u": {k}, "uhat": {(k, r)}}: pins every routing/
    # assignment binary to the given support (all others to 0) so the
    # MILP re-derives loads/timing/lateness for a heuristic solution.
    # Use symmetry_breaking=False with this — the valid inequalities
    # assume canonical truck/robot/copy ordering.
    if fix_binaries is not None:
        for fam, vs in (("x", x), ("y", y), ("u", u), ("uhat", uhat)):
            keep = set(fix_binaries.get(fam, ()))
            for key, v in vs.items():
                val = 1.0 if key in keep else 0.0
                v.lb = v.ub = val

    m.update()  # apply lazy updates before reading model size
    n_vars, n_constrs = m.NumVars, m.NumConstrs
    m.optimize()

    if m.SolCount == 0:
        res = {
            "name": model_name,
            "status": STATUS_LOOKUP.get(m.Status, f"UNKNOWN_{m.Status}"),
            "obj": None,
            "runtime_s": m.Runtime,
            "n_vars": n_vars,
            "n_constrs": n_constrs,
        }
        if fix_binaries is not None and m.Status == GRB.INFEASIBLE:
            # identify the violated constraint family for diagnostics
            m.computeIIS()
            res["iis"] = [c.ConstrName for c in m.getConstrs() if c.IISConstr][
                :100
            ]
        m.dispose()
        return res

    used_t = [k for k in K if u[k].X > 0.5]
    used_r = [
        (k, r)
        for k in used_t
        for r in R_k[k]
        if any(
            v.X > 0.5 for (kk, rr, i, j), v in y.items() if kk == k and rr == r
        )
    ]
    robot_cust = sum(
        1
        for c in C
        if any(v.X > 0.5 for (k, r, i, j), v in y.items() if j == c)
    )

    def lbl(i):
        return inst["labels"][i]

    res = {
        "name": model_name,
        "status": STATUS_LOOKUP.get(m.Status, f"UNKNOWN_{m.Status}"),
        "obj": m.ObjVal,
        "gap": m.MIPGap,
        "runtime_s": m.Runtime,
        "n_vars": n_vars,
        "n_constrs": n_constrs,
        "obj_truck_fixed": obj_tfix.getValue(),
        "obj_robot_fixed": obj_rfix.getValue(),
        "obj_lateness": obj_late.getValue(),
        "obj_truck_travel": obj_truck.getValue(),
        "obj_robot_travel": obj_robot.getValue(),
        "obj_truck_env": obj_truck_env.getValue(),
        "obj_robot_env": obj_robot_env.getValue(),
        "obj_total_env": (obj_truck_env.getValue() + obj_robot_env.getValue()),
        "n_trucks_used": len(used_t),
        "n_robots_used": len(used_r),
        "robot_customers": robot_cust,
        "truck_customers": len(C) - robot_cust,
        "total_lateness_min": sum(delta[c].X for c in C),
        "truck_arcs": [
            {"k": k, "i": lbl(i), "j": lbl(j)}
            for (k, i, j), v in x.items()
            if v.X > 0.5
        ],
        "robot_arcs": [
            {"k": k, "r": r, "i": lbl(i), "j": lbl(j)}
            for (k, r, i, j), v in y.items()
            if v.X > 0.5
        ],
        # Custody states (diagnostics): w at nodes on the truck route.
        "custody": [
            {"k": k, "r": r, "node": lbl(i), "w": round(w[k, r, i].X)}
            for (k, r) in used_r
            for i in N
            if any(
                v.X > 0.5
                for (kk, ii, jj), v in x.items()
                if kk == k and (ii == i or jj == i)
            )
        ],
        # Robot per-node times (overlap diagnostics).
        "robot_node_times": [
            {
                "k": k,
                "r": r,
                "node": lbl(i),
                "ahat": round(ahat[k, r, i].X, 3),
                "bhat": round(bhat[k, r, i].X, 3),
            }
            for (k, r) in used_r
            for i in N0_hat
            if any(
                v.X > 0.5
                for (kk, rr, ii, jj), v in y.items()
                if kk == k and rr == r and (ii == i or jj == i)
            )
        ],
        # Truck per-node times (arrival a / departure b) — parking-wait
        # diagnostics.
        "truck_node_times": [
            {
                "k": k,
                "node": lbl(i),
                "a": (round(a[k, i].X, 3) if i in Nplus else None),
                "b": (round(b[k, i].X, 3) if i in N0 else None),
            }
            for k in used_t
            for i in N
            if any(
                v.X > 0.5
                for (kk, ii, jj), v in x.items()
                if kk == k and (ii == i or jj == i)
            )
        ],
    }
    m.dispose()
    return res


def _to_exact_instance(
    data: ProcessedInstance, config: SharedParams, size: int
) -> dict:
    """Reindex NPZ arrays to the MILP's 0/C/P/D convention.

    NPZ files contain one physical depot. Duplicate its index at both ends
    of ``source_index`` to model separate departure (0) and arrival (D)
    nodes without recomputing travel data. For example, every copied
    matrix's ``[0, D]`` entry refers to the original depot-to-depot entry.
    Parking copies retain their labels and physical-location grouping.
    """
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
    C = list(range(1, n_customers + 1))
    first_parking = n_customers + 1
    P = list(range(first_parking, first_parking + int(data.parking_idx.size)))
    D = len(source_index) - 1
    labels = tuple(data.node_label[source_index].tolist())
    groups = {}
    for p in P:
        groups.setdefault(labels[p].rsplit("#", 1)[0], []).append(p)
    K = list(range(1, config.fleet.n_trucks_for(size) + 1))

    def reindex(matrix):
        result = np.ascontiguousarray(
            matrix[np.ix_(source_index, source_index)]
        )
        result.setflags(write=False)
        return result

    return {
        "C": C,
        "P": P,
        "D": D,
        "K": K,
        "R_k": {
            k: list(range(1, config.fleet.n_robots_per_truck + 1)) for k in K
        },
        "lam": {
            c: int(value) for c, value in zip(C, data.demand, strict=True)
        },
        "e_c": {c: float(value) for c, value in zip(C, data.e, strict=True)},
        "l_c": {c: float(value) for c, value in zip(C, data.l, strict=True)},
        "labels": labels,
        "park_groups": [
            groups[key]
            for key in sorted(groups, key=lambda label: int(label[1:]))
        ],
        "d": reindex(data.d),
        "tau_truck": reindex(data.tau_truck),
        "tau_robot": reindex(data.tau_robot),
    }


_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _tag(value):
    if not _TAG_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "tag must use only letters, digits, '.', '_', or '-'"
        )
    return value


def _parser():
    parser = argparse.ArgumentParser(
        description="Solve precomputed test instances with the exact MILP"
    )
    parser.add_argument(
        "--size", type=int, choices=SUPPORTED_SIZES, required=True
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag", type=_tag)
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="number of instances (default: 1; 0 means all)",
    )
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-symmetry-breaking", action="store_true")
    return parser


def main():
    """Solve the selected test files and export the existing result schema."""
    args = _parser().parse_args()
    if args.limit < 0 or args.time_limit <= 0 or args.mip_gap < 0:
        raise ValueError(
            "--limit/--mip-gap must be nonnegative and "
            "--time-limit must be positive"
        )
    data_dir = "processed" if args.tag is None else f"processed_{args.tag}"
    source = REPO_ROOT / "data" / data_dir / "test" / f"n{args.size}"
    if not source.is_dir():
        raise FileNotFoundError(
            f"processed instance directory not found: {source}"
        )
    paths = sorted(source.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no processed instances under {source}")
    if args.limit:
        paths = paths[: args.limit]
    method_dir = "exact" if args.tag is None else f"exact_{args.tag}"
    out_dir = REPO_ROOT / "output" / method_dir / f"n{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        config, data = load_problem(path, args.params)
        instance_id = str(data.meta["instance_id"])
        inst = _to_exact_instance(data, config, args.size)
        log_file_path = out_dir / f"{instance_id}.log"
        result = run_model(
            inst,
            config,
            model_name=instance_id,
            time_limit_sec=args.time_limit,
            mip_gap=args.mip_gap,
            threads=args.threads,
            symmetry_breaking=not args.no_symmetry_breaking,
            log_path=str(log_file_path),
        )
        with (out_dir / f"{instance_id}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(result, f, indent=2)
        rows.append(
            {
                key: result.get(key)
                for key in (
                    "name",
                    "status",
                    "obj",
                    "gap",
                    "runtime_s",
                    "n_vars",
                    "n_constrs",
                )
            }
        )
        print(instance_id, result.get("status"), result.get("obj"))
    if rows:
        with (out_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
