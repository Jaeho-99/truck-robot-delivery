"""Unit tests for the independent solution validator.

Run with:  .venv/bin/python -m pytest tests/test_validator.py -q
"""

import copy
import os
import random
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance                                    # noqa: E402
from src.heuristics import Params                           # noqa: E402
from src.heuristics.alns import congestion_aware_initial    # noqa: E402
from src.heuristics.solution import eval_solution           # noqa: E402
from src.heuristics.validator import (check_solution,       # noqa: E402
                                      cost_params_from)


@pytest.fixture(scope="module")
def setup():
    inst = instance.build_grid_instance(seed=1)
    e_c, l_c = instance.reachability_tw(inst, seed=1)
    pr = Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])
    sol = congestion_aware_initial(pr, random.Random(0))
    return inst, e_c, l_c, pr, sol


def _robot_stop(sol):
    """(truck, stop_index, stop) of the first stop with a deploy."""
    for k, route in sol.routes.items():
        for si, st in enumerate(route):
            if st["kind"] == "park" and st["deploys"]:
                return k, si, st
    raise AssertionError("initial solution has no robot trip")


def test_valid_solution_matches_evaluator(setup):
    inst, e_c, l_c, pr, sol = setup
    obj, feas, _, _ = eval_solution(pr, sol)
    assert feas
    ok, violations, obj2 = check_solution(inst, e_c, l_c, sol,
                                          cost_params_from(pr))
    assert ok, violations
    assert abs(obj2 - obj) <= 1e-4


def test_rejects_duplicated_customer(setup):
    inst, e_c, l_c, pr, sol = setup
    bad = sol.clone()
    c = next(iter(sol.customers()))
    k = next(iter(bad.routes))
    bad.routes[k].append({"kind": "cust", "c": c})
    ok, violations, _ = check_solution(inst, e_c, l_c, bad,
                                       cost_params_from(pr))
    assert not ok
    assert any("coverage" in v for v in violations)


def test_rejects_retrieval_before_deploy(setup):
    inst, e_c, l_c, pr, sol = setup
    bad = sol.clone()
    k, si, st = _robot_stop(bad)
    # make ret_p point at a real copy that never appears later on the
    # route (retrieval stop missing / would-be-before-deploy)
    st = copy.deepcopy(st)
    later = {s2["p"] for s2 in bad.routes[k][si + 1:]
             if s2["kind"] == "park"}
    p2 = next(p for p in inst["P"]
              if p not in later and p != st["p"])
    st["deploys"][0]["ret_p"] = p2
    bad.routes[k][si] = st
    ok, violations, _ = check_solution(inst, e_c, l_c, bad,
                                       cost_params_from(pr))
    assert not ok
    assert any("ret_p" in v or "pending" in v for v in violations)


def test_rejects_overcapacity_trip(setup):
    inst, e_c, l_c, pr, sol = setup
    bad = sol.clone()
    k, si, st = _robot_stop(bad)
    st = copy.deepcopy(st)
    extra = [c for c in inst["C"]][: pr.beta_robot + 1]
    st["deploys"][0]["custs"] = extra
    bad.routes[k][si] = st
    ok, violations, _ = check_solution(inst, e_c, l_c, bad,
                                       cost_params_from(pr))
    assert not ok
    assert any("beta_robot" in v for v in violations)
