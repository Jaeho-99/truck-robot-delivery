"""Exact-neighborhood operators with repair-local candidate reuse.

Only routes touched by an insertion need rescoring, unless the global parking
copy budget changes. Noisy repair reuses costs, never perturbed deltas: every
feasible candidate consumes the same fresh RNG draw in the same order.
"""

from collections import OrderedDict
import math

from .candidates import enum_route
from .cost import CostContext


# Bounded per-process lookup conversion; no cache is stored on shared Params.
_CONTEXTS = OrderedDict()


def _context(pr):
    key = id(pr)
    if key in _CONTEXTS:
        _CONTEXTS.move_to_end(key)
        return _CONTEXTS[key][1]
    context = CostContext(pr)
    _CONTEXTS[key] = (pr, context)
    if len(_CONTEXTS) > 8:
        _CONTEXTS.popitem(last=False)
    return context


class _RepairSession:
    def __init__(self, pr, sol):
        self.pr = pr
        self.sol = sol
        self.cost = _context(pr)
        self.used = sol.used_copies()
        self.base = {k: self.cost.evaluate(k, sol.routes[k])[0] for k in pr.K}
        self.cache = {}

    def candidates(self, c, k):
        key = (c, k)
        if key not in self.cache:
            base = self.base[k]
            candidates = []
            for _, nr in enum_route(self.pr, k, self.sol.routes[k], c, self.used):
                cost, ok = self.cost.evaluate(k, nr, reject_infeasible=True)
                if ok:
                    candidates.append((cost - base, (k, nr)))
            self.cache[key] = candidates
        return self.cache[key]

    def best(self, c, rng=None, noise=0.0):
        best_delta, best_apply = math.inf, None
        noisy = noise > 0.0 and rng is not None
        for k in self.pr.K:
            for delta, ap in self.candidates(c, k):
                if noisy:
                    delta += rng.uniform(-noise, noise)
                if delta < best_delta - 1e-9:
                    best_delta, best_apply = delta, ap
        return best_delta, best_apply

    def insert(self, ap, c):
        k, route = ap
        self.sol.routes[k] = route
        self.base[k] = self.cost.evaluate(k, route)[0]
        used = self.sol.used_copies()
        if used != self.used:
            # A new parking copy can invalidate candidates on every truck.
            self.cache.clear()
            self.used = used
        else:
            self.cache = {key: value for key, value in self.cache.items()
                          if key[1] != k and key[0] != c}


def best_insertion(pr, sol, c, rng=None, noise=0.0):
    return _RepairSession(pr, sol).best(c, rng, noise)


def repair_greedy(pr, sol, pool, rng, noise=0.0):
    remaining = list(pool)
    if not remaining:
        return True
    session = _RepairSession(pr, sol)
    while remaining:
        best = None
        for c in remaining:
            delta, ap = session.best(c, rng, noise)
            if ap is not None and (best is None or delta < best[0] - 1e-9):
                best = (delta, ap, c)
        if best is None:
            return False
        session.insert(best[1], best[2])
        remaining.remove(best[2])
    return True


def repair_regret2(pr, sol, pool, rng):
    remaining = list(pool)
    if not remaining:
        return True
    session = _RepairSession(pr, sol)
    while remaining:
        pick = None
        for c in remaining:
            local = {}
            for k in pr.K:
                for delta, ap in session.candidates(c, k):
                    if k not in local or delta < local[k][0] - 1e-9:
                        local[k] = (delta, ap)
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
        session.insert(pick[2], pick[3])
        remaining.remove(pick[3])
    return True


def _without_customers(route, cset):
    """Build only modified containers; never mutate shared stops or trips."""
    mid = []
    for st in route:
        if st['kind'] == 'cust':
            if st['c'] not in cset:
                mid.append(st)
        else:
            trips = []
            for tr in st['deploys']:
                custs = [c for c in tr['custs'] if c not in cset]
                if custs:
                    trips.append(tr if len(custs) == len(tr['custs'])
                                 else dict(tr, custs=custs))
            if len(trips) == len(st['deploys']) and all(
                    a is b for a, b in zip(trips, st['deploys'])):
                mid.append(st)
            else:
                mid.append(dict(st, deploys=trips))
    refs = {tr['ret_p'] for st in mid if st['kind'] == 'park'
            for tr in st['deploys']}
    return [st for st in mid if st['kind'] == 'cust'
            or st['deploys'] or st['p'] in refs]


def remove_customers(sol, custs):
    cset = set(custs)
    for k, route in sol.routes.items():
        sol.routes[k] = _without_customers(route, cset)


def destroy_random(pr, sol, q, rng):
    custs = list(sol.customers())
    chosen = rng.sample(custs, min(q, len(custs)))
    remove_customers(sol, chosen)
    return chosen


def destroy_worst(pr, sol, q, rng, p=3.0):
    context = _context(pr)
    costs = {k: context.evaluate(k, route)[0] for k, route in sol.routes.items()}
    base = sum(costs.values())
    # Legacy removal also cleans orphan parking stops on unrelated routes.
    # Normal search states have none; retain the behavior for direct callers.
    cleaned_costs = {}
    for k, route in sol.routes.items():
        cleaned = _without_customers(route, set())
        cleaned_costs[k] = (costs[k] if cleaned == route else
                            context.evaluate(k, cleaned)[0])
    # Keep legacy set iteration and total summation order (ties can be tiny).
    owners = {}
    for k, route in sol.routes.items():
        for st in route:
            custs = ([st['c']] if st['kind'] == 'cust' else
                     [c for tr in st['deploys'] for c in tr['custs']])
            for c in custs:
                owners.setdefault(c, set()).add(k)
    contrib = []
    for c in sol.customers():
        changed = {k: context.evaluate(k, _without_customers(sol.routes[k], {c}))[0]
                   for k in owners[c]}
        after = sum(changed.get(k, cost) for k, cost in cleaned_costs.items())
        contrib.append((base - after, c))
    contrib.sort(reverse=True)
    chosen = []
    while contrib and len(chosen) < q:
        idx = int((rng.random() ** p) * len(contrib))
        chosen.append(contrib.pop(idx)[1])
    remove_customers(sol, chosen)
    return chosen


def destroy_related(pr, sol, q, rng, p=6.0):
    custs = list(sol.customers())
    if not custs:
        return []
    q = min(q, len(custs))
    removed = [rng.choice(custs)]
    while len(removed) < q:
        ref = rng.choice(removed)
        cand = [c for c in custs if c not in removed]
        cand.sort(key=lambda c: pr.dist(ref, c) + 0.1 * abs(pr.l_c[ref] - pr.l_c[c]))
        idx = int((rng.random() ** p) * len(cand))
        removed.append(cand[idx])
    remove_customers(sol, removed)
    return removed
