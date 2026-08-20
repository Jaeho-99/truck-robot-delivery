"""ALNS for truck-robot collaborative last-mile delivery.

Standard Ropke & Pisinger (2006) skeleton: destroy (random / worst /
related) + repair (greedy / greedy-noise / regret-2) with roulette-wheel
adaptive weights and simulated-annealing acceptance (see operators.py).
The evaluator (solution.py) reproduces the MILP's objective and
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
import random
import time
from collections import defaultdict

from ..instance import (EMIS_ROBOT_KM, EMIS_TRUCK_KM, GFUEL_ROBOT_KM,
                        GFUEL_TRUCK_KM, V_ROBOT, V_TRUCK)
from .operators import DESTROY, repair_greedy, repair_regret2
from .qlearning import N_STATES, QTable, get_state
from .solution import Solution, eval_solution, eval_truck

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
# DQN trainer and the PPO environment.
W_START = 0.05

# Degree of destruction (DR-ALNS vanilla): a FIXED 30% of customers is
# removed each iteration, q = max(1, round(DOD * n)). Shared by all
# training/inference paths.
DOD = 0.3


# ============================================================
# 1. Parameter container
# ============================================================
class Params:
    """Instance-plus-cost parameters shared by all ALNS components.

    The cost parameters replicate the defaults of ``model.run_model``
    so that ALNS and exact objective values are directly comparable.
    """

    def __init__(self, inst, e_c, l_c,
                 carbon_price=0.19, phi_hat=40.0, phi_truck=480.0,
                 fixed_cost_hours=8.0, beta_truck=100, beta_robot=3):
        self.inst = inst
        self.nodes = inst["nodes"]
        self.C = list(inst["C"])
        self.P = list(inst["P"])
        self.D = inst["D"]
        self.K = list(inst["K"])
        self.R_k = {k: list(inst["R_k"][k]) for k in inst["K"]}
        self.lam = inst["lam"]
        self.arc_zones = inst["arc_zones"]
        self.alpha_traffic = inst["alpha_traffic"]
        self.alpha_ped = inst["alpha_ped"]
        self.e_c, self.l_c = e_c, l_c

        # Physical parking groups (copy-budget bookkeeping).
        self.park_groups = inst["park_groups"]
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
                key=lambda gi: self._manh(c, self.park_groups[gi][0]))
            self.phys_near[c] = order

        self.V_T, self.V_R = V_TRUCK, V_ROBOT
        self.s_kc = 5.0
        self.s_hat = 1.0
        self.zeta_load = 0.5
        self.zeta_unload = 1.0
        self.beta_truck = beta_truck
        self.beta_robot = beta_robot
        self.phi_hat = phi_hat
        self.phi_truck = phi_truck

        self.gamma_late = 0.0958
        self.gamma_fixed = 7.87 * fixed_cost_hours      # truck fixed
        self.gammahat_fixed = 3.7                       # robot fixed
        self.gamma_fuel = GFUEL_TRUCK_KM * (self.V_T / 60.0)
        self.gamma_env = EMIS_TRUCK_KM * carbon_price * (self.V_T / 60.0)
        self.gammahat_fuel = GFUEL_ROBOT_KM * (self.V_R / 60.0)
        self.gammahat_env = (EMIS_ROBOT_KM * carbon_price
                             * (self.V_R / 60.0))
        self.truck_arc_coef = self.gamma_fuel + self.gamma_env
        self.robot_arc_coef = self.gammahat_fuel + self.gammahat_env

        # Arc caches.
        self._d = {}
        self._tt = {}
        self._tr = {}

    def _manh(self, i, j):
        ni, nj = self.nodes[i], self.nodes[j]
        return abs(ni[0] - nj[0]) + abs(ni[1] - nj[1])

    # ---- distance / travel time (same definition as the MILP) ----
    def dist(self, i, j):
        v = self._d.get((i, j))
        if v is None:
            v = self._manh(i, j)
            self._d[(i, j)] = v
        return v

    def tau_truck(self, i, j):
        v = self._tt.get((i, j))
        if v is None:
            eff = sum(km * self.alpha_traffic[z]
                      for z, km in self.arc_zones[(i, j)].items())
            v = eff / self.V_T * 60.0
            self._tt[(i, j)] = v
        return v

    def tau_robot(self, i, j):
        v = self._tr.get((i, j))
        if v is None:
            eff = sum(km * self.alpha_ped[z]
                      for z, km in self.arc_zones[(i, j)].items())
            v = eff / self.V_R * 60.0
            self._tr[(i, j)] = v
        return v


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
    zc = pr.inst["node_zone"]
    ta, pa = pr.alpha_traffic, pr.alpha_ped
    tmin = min(ta.values())
    tspan = max(max(ta.values()) - min(ta.values()), 1e-12)
    pmin = min(pa.values())
    pspan = max(max(pa.values()) - min(pa.values()), 1e-12)
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
               time_limit_s=None, initial=None, selector="roulette",
               q_params=None, iter_trace=None):
    """Run ALNS and return (best_solution, best_cost, stats).

    ``initial``: initial-solution constructor ``f(pr, rng) -> Solution``
    (default: congestion_aware_initial).
    ``iter_trace``: optional list; if given, one
    (it, accepted, current_cost) tuple is appended per iteration
    (instrumentation only — never touches rng).
    ``time_limit_s``: wall-clock cap in seconds; exceeding it stops the
    run early (for large instances). The cooling schedule stays based
    on ``iters``, so an early stop may end in the hot phase.
    ``selector``: operator-selection scheme — "roulette" (adaptive
    weights, default), "qlearning" (tabular Q-learning over
    (destroy, repair) pairs with the sigma scores as rewards; see
    qlearning.py) or "gnn_dqn" (trained GNN+DQN policy, greedy; see
    src/gnn_dqn, requires torch).
    ``q_params``: optional selector-specific dict — QTable keyword
    overrides for "qlearning" (eta, gamma, eps_start, eps_end,
    eps_decay_iters), {"model_path", "device"} for "gnn_dqn";
    ignored under "roulette".
    """
    t_start = time.time()
    rng = random.Random(seed)
    segment = segment or max(20, iters // 30)
    nC = len(pr.C)
    # degree of destruction: fixed 30% of customers (DR-ALNS vanilla)
    q_destroy = max(1, round(DOD * nC))

    if initial is None:
        initial = congestion_aware_initial
    current_solution = initial(pr, rng)
    current_cost, feas, _, _ = eval_solution(pr, current_solution)
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
    # decay to 0 over the run (Santini et al.; same rule in the DQN
    # trainer and the PPO environment).
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

    # Q-learning selection over (destroy, repair) pairs (see
    # qlearning.py). Epsilon decays over the first half of the run.
    use_q = selector == "qlearning"
    if use_q:
        qkw = {"eps_decay_iters": max(1, iters // 2), "seed": seed + 1}
        qkw.update(q_params or {})
        qtab = QTable(N_STATES, len(DESTROY) * len(repair_ops), **qkw)
        state = get_state(pr, current_solution)
    # Trained GNN+DQN policy (inference only; lazy import keeps this
    # module usable without torch).
    use_gnn = selector == "gnn_dqn"
    if use_gnn:
        from ..gnn_dqn.selector_gnn import GNNSelector
        gsel = GNNSelector(**(q_params or {}))
        best_it = 0
        # previous-iteration outcome flags for g_t (DR-ALNS obs):
        # (best_improved, current_accepted, current_improved)
        g_flags = (False, False, False)

    it_done = 0
    for it in range(1, iters + 1):
        if time_limit_s is not None and time.time() - t_start > time_limit_s:
            break
        it_done = it
        # linear temperature decay T0 -> 0 over the run
        T = T0 * (1.0 - (it - 1) / iters)
        t_sel = time.perf_counter()
        if use_q:
            act = qtab.select(state, it)
            di, ri = divmod(act, len(repair_ops))
        elif use_gnn:
            di, ri = gsel.select(pr, current_solution, it - 1, iters,
                                 it - 1 - best_it, current_cost,
                                 best_cost, *g_flags)
        else:
            di = roulette(dW, rng)
            ri = roulette(rW, rng)
        sel_time += time.perf_counter() - t_sel
        a_idx = di * len(repair_ops) + ri
        t_op = time.perf_counter()      # clone+destroy+repair+eval
        cand = current_solution.clone()
        pool = DESTROY[di][1](pr, cand, q_destroy, rng)
        repair_ops[ri][1](pr, cand, pool, rng)
        cand_cost, ok, _, _ = eval_solution(pr, cand)
        pair_cnt[a_idx] += 1
        pair_time[a_idx] += time.perf_counter() - t_op
        dCnt[di] += 1
        rCnt[ri] += 1

        if not ok:      # discard coverage/custody/range violations
            infeas_cnt += 1
            if use_q:   # current solution unchanged: zero reward, same state
                t_sel = time.perf_counter()
                qtab.update(state, act, 0.0, state)
                sel_time += time.perf_counter() - t_sel
            if use_gnn:
                g_flags = (False, False, False)
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
            if use_gnn:
                best_it = it
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
        if use_gnn:
            g_flags = (found_best, accept, accept and was_improving)
        if use_q:
            t_sel = time.perf_counter()
            s_next = get_state(pr, current_solution) if accept else state
            qtab.update(state, act, reward, s_next)
            state = s_next
            sel_time += time.perf_counter() - t_sel
        dScore[di] += reward
        rScore[ri] += reward

        if (not use_q and not use_gnn
                and it % segment == 0):         # adaptive weight update
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
             "iters_done": it_done, "selector": selector,
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
    if use_q:
        labels = [f"{d[0]}+{r[0]}" for d in DESTROY for r in repair_ops]
        stats["q_summary"] = qtab.summary(labels)
    return best_solution, best_cost, stats
