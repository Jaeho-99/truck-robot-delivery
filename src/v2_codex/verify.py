"""Differential checks against the untouched operators; no output artifacts.

Example: python src/v2_codex/verify.py --sizes 5 10 20 --iterations 30
Uses the first processed test instance of each size by default.  Assertions
check exact Python float values, route structures, ordering and RNG state.
"""

import argparse
import copy
import importlib
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path[:] = [p for p in sys.path
                   if Path(p).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alns import solve as old
from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT, load_problem
from v2_codex.alns import solve as new
from v2_codex.candidates import enum_route
from v2_codex.cost import CostContext
from v2_codex import operators

assert Path(old.__file__).resolve() == REPO_ROOT / "src" / "alns" / "solve.py"
assert old.repair_greedy is not new.repair_greedy, "must compare independent implementations"


def assert_cost(pr, k, route, context=None):
    expected = old.eval_truck(pr, k, route)[:2]
    context = context or CostContext(pr)
    actual = context.evaluate(k, route)
    assert actual == expected, ("candidate cost/feasibility", actual, expected)
    pruned = context.evaluate(k, route, reject_infeasible=True)
    assert pruned[1] == expected[1], ("early rejection feasibility", pruned, expected)
    if expected[1]:
        assert pruned[0] == expected[0], ("early rejection cost", pruned, expected)
    return expected


def check_candidates(pr, initial):
    sol = initial.clone()
    pool = old.destroy_random(pr, sol, max(1, round(0.3 * len(pr.C))),
                              random.Random(1729))
    context = CostContext(pr)
    checked = 0
    snapshot = copy.deepcopy(sol.routes)
    for c in pool:
        expected = list(old.enum_insertions(pr, sol, c))
        actual = [candidate for k in pr.K for candidate in
                  enum_route(pr, k, sol.routes[k], c, sol.used_copies())]
        assert actual == expected, ("candidate contents/order", c)
        for k, route in actual:
            assert_cost(pr, k, route, context)
            checked += 1
        for noise in (0.0, 5.0):
            left_rng, right_rng = random.Random(83), random.Random(83)
            left = old.best_insertion(pr, sol, c, left_rng, noise)
            right = operators.best_insertion(pr, sol, c, right_rng, noise)
            assert left == right, ("best insertion", c, noise)
            assert left_rng.getstate() == right_rng.getstate(), "insertion RNG"
    assert sol.routes == snapshot, "candidate scoring mutated the solution"
    return checked


def check_action_pairs(pr, initial, seed):
    for destroy_name in ("destroy_random", "destroy_worst", "destroy_related"):
        for repair_index in range(3):
            left, right = initial.clone(), initial.clone()
            left_rng, right_rng = random.Random(seed), random.Random(seed)
            q = max(1, round(0.3 * len(pr.C)))
            left_pool = getattr(old, destroy_name)(pr, left, q, left_rng)
            right_pool = getattr(operators, destroy_name)(pr, right, q, right_rng)
            assert left_pool == right_pool, ("destroy pool", destroy_name)
            assert left.routes == right.routes, ("destroy routes", destroy_name)
            if repair_index < 2:
                noise = 0.0 if repair_index == 0 else 5.0
                left_ok = old.repair_greedy(pr, left, left_pool, left_rng, noise)
                right_ok = operators.repair_greedy(
                    pr, right, right_pool, right_rng, noise)
            else:
                left_ok = old.repair_regret2(pr, left, left_pool, left_rng)
                right_ok = operators.repair_regret2(pr, right, right_pool, right_rng)
            assert left_ok == right_ok, ("repair success", repair_index)
            assert left.routes == right.routes, (destroy_name, repair_index)
            assert old.eval_solution(pr, left) == old.eval_solution(pr, right)
            assert left_rng.getstate() == right_rng.getstate(), "operator RNG"

    # Exercise the actual PPO and GNN-PPO integration entry points as well.
    for package in ("ppo_alns", "gnn_ppo_alns"):
        original = importlib.import_module(package + ".alns")
        optimized = importlib.import_module("v2_codex." + package + ".alns")
        left, right = initial.clone(), initial.clone()
        left_rng, right_rng = random.Random(seed), random.Random(seed)
        for action in range(9):
            q = max(1, round(0.3 * len(pr.C)))
            lnext, lcost, lok = original.apply_actor_action(
                pr, left, action, q, left_rng, 5.0)
            rnext, rcost, rok = optimized.apply_actor_action(
                pr, right, action, q, right_rng, 5.0)
            assert (lcost, lok) == (rcost, rok), (package, action, "objective")
            assert lnext.routes == rnext.routes, (package, action, "routes")
            assert left_rng.getstate() == right_rng.getstate(), (package, "RNG")
            if lok:
                left, right = lnext, rnext


def check_trajectory(pr, seed, iterations):
    left_trace, right_trace = [], []
    left, left_cost, left_stats = old.solve_alns(
        pr, iters=iterations, seed=seed, iter_trace=left_trace)
    right, right_cost, right_stats = new.solve_alns(
        pr, iters=iterations, seed=seed, iter_trace=right_trace)
    assert left_trace == right_trace, ("ALNS acceptance/cost trajectory", seed)
    assert left.routes == right.routes and left_cost == right_cost, "ALNS result"
    for stats in (left_stats, right_stats):
        stats.pop("pair_time_s")
        stats.pop("selector_overhead_s")
        stats["best_trace"] = [(it, cost) for it, _, cost in stats["best_trace"]]
    assert left_stats == right_stats, "ALNS non-timing statistics"


def check_mutation_isolation(pr, initial):
    snapshot = copy.deepcopy(initial.routes)
    child = initial.clone()
    pool = operators.destroy_random(pr, child, 2, random.Random(44))
    operators.repair_greedy(pr, child, pool, random.Random(44), 5.0)
    assert initial.routes == snapshot, "clone mutation leaked to parent"

    # A copy-on-write candidate shares unchanged stops/trips with its source.
    # Removal must also copy changed containers when passed such a candidate.
    shared = old.Solution(pr.K)
    shared.routes = dict(initial.routes)
    operators.remove_customers(shared, pr.C[:2])
    assert initial.routes == snapshot, "removal mutated a shared stop/trip"


def check_copy_budget(pr):
    assert len(pr.K) >= 2, "copy-budget check requires two trucks"
    sol = old.Solution(pr.K)
    k1, k2 = pr.K[:2]
    c1, c2 = pr.C[:2]
    session = operators._RepairSession(pr, sol)
    session.candidates(c2, k2)
    chosen = None
    for _, ap in session.candidates(c1, k1):
        if any(st["kind"] == "park" for st in ap[1]):
            chosen = ap
            break
    assert chosen is not None, "fixture needs a feasible robot candidate"
    session.insert(chosen, c1)
    assert session.used == sol.used_copies() and session.used, "used-copy tracking"
    actual = session.candidates(c2, k2)
    expected = []
    base = old.eval_truck(pr, k2, sol.routes[k2])[0]
    for k, route in old.enum_insertions(pr, sol, c2):
        cost, ok, _, _ = old.eval_truck(pr, k, route)
        if k == k2 and ok:
            expected.append((cost - base, (k, route)))
    assert actual == expected, "stale candidates after another truck consumes copies"
    for _, (_, route) in actual:
        used_here = {st["p"] for st in route if st["kind"] == "park"}
        assert not used_here.intersection(session.used), "copy reused across trucks"

    # Direct insertion leaves the global parking budget unchanged but changes
    # one route's positions and costs.  Compare all remaining candidates anew.
    sol = old.Solution(pr.K)
    session = operators._RepairSession(pr, sol)
    session.candidates(c2, k1)
    session.candidates(c2, k2)
    session.insert((k1, [{"kind": "cust", "c": c1}]), c1)
    for k in pr.K:
        fresh = operators._RepairSession(pr, sol).candidates(c2, k)
        assert session.candidates(c2, k) == fresh, "changed-route invalidation"


def check_edge_cases(pr):
    k = pr.K[0]
    r = pr.R_k[k][0]
    c1, c2 = pr.C[:2]
    p1, p2, p3 = pr.P[:3]

    def park(p, *trips):
        return {"kind": "park", "p": p, "deploys": list(trips)}

    def trip(customers, ret, robot=r):
        return {"r": robot, "custs": list(customers), "ret_p": ret}

    waiting = [park(p1, trip([c1], p2)), park(p2)]
    cases = [
        ("empty", [], None),
        ("robot waiting", waiting, True),
        ("same-copy retrieval", [park(p1, trip([c1], p1))], False),
        ("missing retrieval", [park(p1, trip([c1], p2))], False),
        ("empty trip", [park(p1, trip([], p2)), park(p2)], False),
        ("trip capacity", [park(p1, trip([c1] * (pr.beta_robot + 1), p2)),
                           park(p2)], False),
        ("unknown robot", [park(p1, trip([c1], p2, max(pr.R_k[k]) + 1)),
                           park(p2)], False),
        ("deploy before retrieve", [park(p1, trip([c1], p2)),
                                    park(p2, trip([c2], p3)), park(p3)], False),
        ("duplicate lateness overwrite", [{"kind": "cust", "c": c1},
                                          {"kind": "cust", "c": c1}], None),
    ]
    for label, route, expected_ok in cases:
        _, ok = assert_cost(pr, k, route)
        if expected_ok is not None:
            assert ok is expected_ok, label
    for field in ("phi_truck", "phi_hat", "beta_truck"):
        restricted = copy.copy(pr)
        setattr(restricted, field, 0.0)
        _, ok = assert_cost(restricted, k, waiting)
        assert not ok, ("constraint not rejected", field)
    late = copy.copy(pr)
    late.l_c = {c: -10.0 for c in pr.C}
    assert_cost(late, k, [park(p1, trip([c1, c2], p2)), park(p2),
                        {"kind": "cust", "c": c1}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.iterations < 1 or args.instances < 1:
        parser.error("iterations and instances must be positive")
    data_dir = "processed" if args.tag is None else "processed_" + args.tag
    total_candidates = 0
    for size in args.sizes:
        paths = sorted((REPO_ROOT / "data" / data_dir / "test" / f"n{size}").glob("*.npz"))
        if len(paths) < args.instances:
            raise FileNotFoundError(f"need {args.instances} processed test instances for n{size}")
        for path in paths[:args.instances]:
            config, data = load_problem(path, args.params)
            pr = old.Params(data, config, size)
            initial = old.congestion_aware_initial(pr, random.Random(0))
            total_candidates += check_candidates(pr, initial)
            check_edge_cases(pr)
            check_copy_budget(pr)
            check_mutation_isolation(pr, initial)
            for seed in args.seeds:
                check_action_pairs(pr, initial, seed)
                check_trajectory(pr, seed, args.iterations)
            print(f"[verify] {path.stem}: candidate/cost/order, 9 actions, RNG, "
                  "PPO/GNN integration, trajectory, custody, ranges, capacity, "
                  "copy invalidation and mutation isolation passed", flush=True)
    print(f"[verify] PASS: {total_candidates} candidate scores matched exactly; "
          "all requested checks passed", flush=True)


if __name__ == "__main__":
    main()
