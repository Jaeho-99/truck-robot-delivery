"""Original A/B/C/D candidates, enumerated for one truck at a time.

Copy-on-write candidates retain the legacy search neighborhood and ties.
The global used-copy set is supplied by the repair session.
"""

L_RET_EXIST = 3
W_RET_NEW = 4
N_PHYS_NEAR = 2

def _with_new_trip(route, si, st, trip):
    """Copy-on-write: route with ``trip`` appended at stop index si."""
    new_st = dict(st)
    new_st["deploys"] = st["deploys"] + [trip]
    nr = list(route)
    nr[si] = new_st
    return nr


def enum_route(pr, k, route, c, used):
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


