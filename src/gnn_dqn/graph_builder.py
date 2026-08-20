"""Solution + Params -> torch_geometric HeteroData.

The graph is always built from a repair-complete solution (every
customer served); never from a mid-destroy partial solution.

Node types: customer (5 feats), parking (6 feats, one node per copy in
pr.P including unused copies), depot (3 feats, out/in). Edge families:
truck_arc (2 feats), robot_arc (3 feats), proximity k-NN (3 feats,
customer-customer / customer-parking only). Every edge is added in both
directions with identical features.
"""

import torch
from torch_geometric.data import HeteroData

# Fixed edge-family feature dims (encoder builds one conv per family
# and node-type combination; see EDGE_TYPES).
EDGE_DIMS = {"truck_arc": 2, "robot_arc": 3, "proximity": 3}

# All (src, rel, dst) triplets the builder may emit. Truck arcs connect
# any consecutive stop types (incl. depot legs); robot arcs and
# proximity edges are customer/parking only.
EDGE_TYPES = (
    [(s, "truck_arc", d)
     for s in ("depot", "customer", "parking")
     for d in ("depot", "customer", "parking")
     if not (s == "depot" and d == "depot")]
    + [("parking", "robot_arc", "customer"),
       ("customer", "robot_arc", "customer"),
       ("customer", "robot_arc", "parking")]
    + [("customer", "proximity", "customer"),
       ("customer", "proximity", "parking"),
       ("parking", "proximity", "customer")]
)


def pad_edge_types(data):
    """Give every EDGE_TYPES triplet a (possibly empty) store.

    Batch.from_data_list mis-collates HeteroData lists whose edge-store
    key sets differ: edges of a graph missing elsewhere get node
    offsets from the wrong graph, silently rewiring them across graph
    boundaries (measured Q-value corruption up to ~0.06 in the DQN
    replay batch). Padding every graph to one shared key set makes
    collation exact and keeps single-graph and batched forwards
    identical; build() therefore always returns padded graphs.
    """
    for et in EDGE_TYPES:
        if et not in data.edge_types:
            data[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[et].edge_attr = torch.zeros((0, EDGE_DIMS[et[1]]))
    return data


def horizon(pr):
    """Planning horizon T (pr has no explicit horizon field)."""
    return max(pr.l_c.values()) * 1.2


def _epsilons(pr):
    eps_t = max(pr.alpha_traffic.values()) - 1.0
    eps_p = max(pr.alpha_ped.values()) - 1.0
    return max(eps_t, 1e-9), max(eps_p, 1e-9)


def _exposure(pr, i, j, alpha, eps):
    """(congestion exposure - 1) / eps, clipped to [0, 1.5]."""
    d = pr.dist(i, j)
    if d <= 1e-12:
        return 0.0
    eff = sum(km * alpha[z] for z, km in pr.arc_zones[(i, j)].items())
    return min(max((eff / d - 1.0) / eps, 0.0), 1.5)


def forward_schedule(pr, sol):
    """Arrival times and parking-stop stats of the current solution.

    Re-runs the forward schedule of solution.eval_truck (same recursion
    (26)-(34); solution.py is read-only so the timing extraction is
    replicated here rather than returned by the evaluator).

    Returns (arrivals {c: minute}, park_stats {copy: (a, b, parcels)}).
    """
    arrivals, park_stats = {}, {}
    for k, route in sol.routes.items():
        pending = {}
        prev, b_prev = 0, 0.0
        for st in route:
            node = st["c"] if st["kind"] == "cust" else st["p"]
            a_node = b_prev + pr.tau_truck(prev, node)
            if st["kind"] == "cust":
                arrivals[st["c"]] = a_node
                b_node = a_node + pr.s_kc
            else:
                p = st["p"]
                b_node = a_node
                parcels = 0
                for tr in st["deploys"]:
                    parcels += sum(pr.lam[c] for c in tr["custs"])
                    t = a_node + pr.zeta_unload
                    rprev = p
                    for c in tr["custs"]:
                        ahat = t + pr.tau_robot(rprev, c)
                        arrivals[c] = ahat
                        t = ahat + pr.s_hat
                        rprev = c
                    arr_ret = t + pr.tau_robot(rprev, tr["ret_p"])
                    pending.setdefault(tr["ret_p"], []).append(arr_ret)
                if st["deploys"]:
                    b_node = max(b_node, a_node + pr.zeta_unload)
                for arr in pending.pop(p, []):
                    b_node = max(b_node, arr + pr.zeta_load)
                park_stats[p] = (a_node, b_node, parcels)
            prev, b_prev = node, b_node
    return arrivals, park_stats


class GraphBuilder:
    def __init__(self, norms, cfg):
        self.norms = norms
        self.cfg = cfg

    def build(self, pr, sol):
        from ..heuristics.qlearning import robot_served_customers

        T = horizon(pr)
        eps_t, eps_p = _epsilons(pr)
        c_truck = self.norms["c_truck"]
        c_robot = self.norms["c_robot"]
        d_max = self.norms["d_max"]

        ids = [0] + list(pr.C) + list(pr.P) + [pr.D]
        xs = [pr.nodes[i][0] for i in ids]
        ys = [pr.nodes[i][1] for i in ids]
        x0, y0 = min(xs), min(ys)
        xspan = max(max(xs) - x0, 1e-9)
        yspan = max(max(ys) - y0, 1e-9)

        def xy(i):
            return ((pr.nodes[i][0] - x0) / xspan,
                    (pr.nodes[i][1] - y0) / yspan)

        served = robot_served_customers(sol)
        arrivals, park_stats = forward_schedule(pr, sol)

        # ---- node maps ----
        cust_ids = list(pr.C)
        park_ids = list(pr.P)
        cidx = {c: i for i, c in enumerate(cust_ids)}
        pidx = {p: i for i, p in enumerate(park_ids)}
        depot_ids = [0, pr.D]
        didx = {0: 0, pr.D: 1}

        def ref(node):
            if node in cidx:
                return "customer", cidx[node]
            if node in pidx:
                return "parking", pidx[node]
            return "depot", didx[node]

        # ---- node features ----
        cust_x = []
        for c in cust_ids:
            px, py = xy(c)
            a_c = arrivals.get(c, 0.0)
            ttd = min(max((pr.l_c[c] - a_c) / T, -1.0), 1.0)
            cust_x.append([px, py, pr.l_c[c] / T, ttd,
                           1.0 if c in served else 0.0])

        visiting = {}           # copy -> truck k
        for k, route in sol.routes.items():
            for st in route:
                if st["kind"] == "park":
                    visiting[st["p"]] = k
        deploys_at = {}
        for route in sol.routes.values():
            for st in route:
                if st["kind"] == "park":
                    deploys_at[st["p"]] = st["deploys"]
        park_x = []
        for p in park_ids:
            px, py = xy(p)
            if p in visiting:
                k = visiting[p]
                nrob = max(len(pr.R_k[k]), 1)
                a, b, parcels = park_stats[p]
                park_x.append([px, py, 1.0,
                               len(deploys_at.get(p, [])) / nrob,
                               (b - a) / T,
                               parcels / (nrob * pr.beta_robot)])
            else:
                park_x.append([px, py, 0.0, 0.0, 0.0, 0.0])

        depot_x = [[*xy(0), 0.0], [*xy(pr.D), 1.0]]

        # ---- edges ----
        edges = {}      # triplet -> ([src], [dst])
        feats = {}      # triplet -> [featvec]

        def add(u, v, rel, feat):
            (st_, si), (dt_, di) = ref(u), ref(v)
            for key, a_, b_ in (((st_, rel, dt_), si, di),
                                ((dt_, rel, st_), di, si)):
                e = edges.setdefault(key, ([], []))
                e[0].append(a_)
                e[1].append(b_)
                feats.setdefault(key, []).append(feat)

        for route in sol.routes.values():
            if not route:
                continue
            seq = [0] + [st["c"] if st["kind"] == "cust" else st["p"]
                         for st in route] + [pr.D]
            for i, j in zip(seq, seq[1:]):
                add(i, j, "truck_arc",
                    [pr.tau_truck(i, j) / c_truck,
                     _exposure(pr, i, j, pr.alpha_traffic, eps_t)])
            for st in route:
                if st["kind"] != "park":
                    continue
                for tr in st["deploys"]:
                    legs = ([st["p"]] + list(tr["custs"]) + [tr["ret_p"]])
                    for i, j in zip(legs, legs[1:]):
                        add(i, j, "robot_arc",
                            [pr.tau_robot(i, j) / c_robot,
                             _exposure(pr, i, j, pr.alpha_ped, eps_p),
                             pr.dist(i, j) / d_max])

        if self.cfg.use_proximity:
            cand_ids = cust_ids + park_ids
            for c in cust_ids:
                near = sorted((n for n in cand_ids if n != c),
                              key=lambda n: pr.dist(c, n))
                for n in near[:self.cfg.knn_k]:
                    add(c, n, "proximity",
                        [pr.dist(c, n) / d_max,
                         pr.tau_robot(c, n) / c_robot,
                         _exposure(pr, c, n, pr.alpha_ped, eps_p)])

        # ---- assemble ----
        data = HeteroData()
        data["customer"].x = torch.tensor(cust_x, dtype=torch.float32)
        data["parking"].x = torch.tensor(park_x, dtype=torch.float32)
        data["depot"].x = torch.tensor(depot_x, dtype=torch.float32)
        for key, (src, dst) in edges.items():
            data[key].edge_index = torch.tensor([src, dst],
                                                dtype=torch.long)
            data[key].edge_attr = torch.tensor(feats[key],
                                               dtype=torch.float32)
        return pad_edge_types(data)

    def update(self, graph, pr, sol, changed_customers):
        """Incremental update stub — v1 delegates to a full rebuild."""
        return self.build(pr, sol)
