"""Destroy and repair operators, and insertion-candidate enumeration."""

import math

from .solution import eval_solution, eval_truck

__all__ = ["DESTROY", "enum_insertions", "best_insertion",
           "apply_insertion", "repair_greedy", "repair_regret2",
           "remove_customers", "destroy_random", "destroy_worst",
           "destroy_related"]

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


DESTROY = [("random", destroy_random), ("worst", destroy_worst),
           ("related", destroy_related)]
