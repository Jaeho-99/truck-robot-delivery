"""Copy-on-write enum_insertions: equivalence and immutability.

The permanent guarantee behind the byte-identical claim: candidates
produced by the copy-on-write generator are STRUCTURALLY IDENTICAL to
the ones the old per-candidate-deepcopy generator produced (reference
implementation kept below), and candidate generation never mutates the
input solution. Together with rng-draw-order preservation this makes
the whole ALNS trajectory invariant, which the determinism tests pin
for both the roulette and the DQN-selector paths.

Run with:  .venv/bin/python -m pytest tests/test_operators_cow.py -q
"""

import copy
import os
import random
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.gnn_dqn.dataset import DirectoryInstanceProvider   # noqa: E402
from src.heuristics.alns import (congestion_aware_initial,  # noqa: E402
                                 solve_alns)
from src.heuristics.operators import (DESTROY,              # noqa: E402
                                      L_RET_EXIST, N_PHYS_NEAR,
                                      W_RET_NEW, best_insertion,
                                      enum_insertions)
from src.heuristics.solution import eval_solution           # noqa: E402


def _reference_enum_insertions(pr, sol, c):
    """The pre-CoW generator (per-candidate deepcopy), verbatim."""
    used = sol.used_copies()
    for k in pr.K:
        route = sol.routes[k]
        for pos in range(len(route) + 1):
            yield k, (route[:pos] + [{"kind": "cust", "c": c}]
                      + route[pos:])
        park_pos = [(si, st) for si, st in enumerate(route)
                    if st["kind"] == "park"]
        for si, st in park_pos:
            for ti, tr in enumerate(st["deploys"]):
                if len(tr["custs"]) >= pr.beta_robot:
                    continue
                for pos in range(len(tr["custs"]) + 1):
                    nr = copy.deepcopy(route)
                    nr[si]["deploys"][ti]["custs"].insert(pos, c)
                    yield k, nr
        for si, st in park_pos:
            for r in pr.R_k[k]:
                for sj, st2 in [pp for pp in park_pos
                                if pp[0] > si][:L_RET_EXIST]:
                    nr = copy.deepcopy(route)
                    nr[si]["deploys"].append(
                        {"r": r, "custs": [c], "ret_p": st2["p"]})
                    yield k, nr
                grp = pr.park_groups[pr.copy_to_phys[st["p"]]]
                free = [cp for cp in grp if cp not in used]
                if free:
                    nr = copy.deepcopy(route)
                    nr[si]["deploys"].append(
                        {"r": r, "custs": [c], "ret_p": free[0]})
                    nr.insert(si + 1, {"kind": "park", "p": free[0],
                                       "deploys": []})
                    yield k, nr
                for gi in pr.phys_near[c][:N_PHYS_NEAR]:
                    grp2 = pr.park_groups[gi]
                    free2 = [cp for cp in grp2
                             if cp not in used and cp != st["p"]]
                    if not free2:
                        continue
                    for pos in range(si + 1,
                                     min(len(route), si + W_RET_NEW) + 1):
                        nr = copy.deepcopy(route)
                        nr[si]["deploys"].append(
                            {"r": r, "custs": [c], "ret_p": free2[0]})
                        nr.insert(pos, {"kind": "park", "p": free2[0],
                                        "deploys": []})
                        yield k, nr
        for gi in pr.phys_near[c][:N_PHYS_NEAR]:
            grp = pr.park_groups[gi]
            free = [cp for cp in grp if cp not in used]
            if not free:
                continue
            dep_cp = free[0]
            for r in pr.R_k[k]:
                for pos in range(len(route) + 1):
                    if len(free) >= 2:
                        nr = copy.deepcopy(route)
                        nr.insert(pos, {"kind": "park", "p": dep_cp,
                                        "deploys": [{"r": r,
                                                     "custs": [c],
                                                     "ret_p": free[1]}]})
                        nr.insert(pos + 1,
                                  {"kind": "park", "p": free[1],
                                   "deploys": []})
                        yield k, nr
                    later = [st2 for si2, st2 in park_pos
                             if si2 >= pos][:L_RET_EXIST]
                    for st2 in later:
                        nr = copy.deepcopy(route)
                        nr.insert(pos, {"kind": "park", "p": dep_cp,
                                        "deploys": [{"r": r,
                                                     "custs": [c],
                                                     "ret_p": st2["p"]}]})
                        yield k, nr


@pytest.fixture(scope="module")
def pr():
    prov = DirectoryInstanceProvider(size=20, root=os.path.join(
        REPO_ROOT, "data"), seed=0)
    return prov._params(prov.train[0])


@pytest.fixture(scope="module")
def destroyed(pr):
    """A solution with robot trips + a pool of removed customers."""
    rng = random.Random(0)
    sol = congestion_aware_initial(pr, rng)
    pool = DESTROY[0][1](pr, sol, 6, rng)      # random destroy, q=6
    assert pool
    return sol, pool


def test_candidates_equal_reference(pr, destroyed):
    sol, pool = destroyed
    for c in pool:
        got = list(enum_insertions(pr, sol, c))
        ref = list(_reference_enum_insertions(pr, sol, c))
        assert got == ref, f"candidate mismatch for customer {c}"
        assert len(got) > 0


def test_enum_does_not_mutate_solution(pr, destroyed):
    sol, pool = destroyed
    snapshot = copy.deepcopy(sol.routes)
    for c in pool:
        for _k, _nr in enum_insertions(pr, sol, c):
            pass                                   # consume everything
        best_insertion(pr, sol, c, random.Random(1), noise=0.0)
    assert sol.routes == snapshot


def test_repair_reject_leaves_current_solution_intact(pr):
    """Full iteration shape: candidate built and discarded -> the
    current solution must be untouched (the rejection path never
    'restores' anything, it just keeps the original)."""
    rng = random.Random(3)
    sol = congestion_aware_initial(pr, rng)
    snapshot = copy.deepcopy(sol.routes)
    cand = sol.clone()
    pool = DESTROY[2][1](pr, cand, 6, rng)         # related destroy
    from src.heuristics.operators import repair_greedy
    repair_greedy(pr, cand, pool, rng)
    eval_solution(pr, cand)                        # evaluate candidate
    assert sol.routes == snapshot                  # reject = no-op


def test_solve_alns_deterministic_roulette(pr):
    fingerprints = []
    for _ in range(2):
        trace = []
        _, best_cost, stats = solve_alns(pr, iters=40, seed=7,
                                         iter_trace=trace)
        fingerprints.append((round(best_cost, 9), trace,
                             stats["accept_count"],
                             stats["action_hist"]))
    assert fingerprints[0] == fingerprints[1]
