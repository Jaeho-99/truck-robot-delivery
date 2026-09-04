"""Graph features, graph builder, encoder, and normalization."""

import numpy as np


def compute_norms(prs, pct=0.95):
    """Return graph-feature normalizers from a training-instance sample."""
    truck, robot, distance = [], [], []
    for pr in prs:
        mask = ~np.eye(pr.d.shape[0], dtype=bool)
        truck.extend(pr.tau_truck_matrix[mask].tolist())
        robot.extend(pr.tau_robot_matrix[mask].tolist())
        distance.extend(pr.d[mask].tolist())

    def percentile(values):
        values = sorted(values)
        return max(values[min(len(values) - 1, int(pct * len(values)))],
                   1e-9)

    return {"c_truck": percentile(truck), "c_robot": percentile(robot),
            "d_max": percentile(distance)}


def robot_served_customers(sol):
    served = set()
    for route in sol.routes.values():
        for stop in route:
            if stop["kind"] == "park":
                for trip in stop["deploys"]:
                    served.update(trip["custs"])
    return served


def congestion_ratio(pr, robot_served):
    """Assigned-mode congestion divided by its per-customer lower bound."""
    zones = pr.node_zone
    numerator = denominator = 0.0
    for customer in pr.C:
        truck = pr.alpha_traffic[zones[customer]]
        robot = pr.alpha_ped[zones[customer]]
        numerator += robot if customer in robot_served else truck
        denominator += min(truck, robot)
    return numerator / denominator if denominator else 1.0
"""Global search-state features g_t (9-dim float32 vector).

Used by the GNN-PPO-ALNS policy as the non-graph part of its state.

Features 2-8 follow the DR-ALNS observation space (Reijnen et al.,
ICAPS 2024); 0-1 are problem-specific extensions of this work (robot
share / congestion exploitation). Our objective is minimized, so
"improved" means a cost DECREASE (the original paper maximizes).
"""

import torch


G_DIM = 9


def global_features(pr, sol, it, search_iterations, stagcount,
                    current_cost, best_cost, best_improved=False,
                    current_accepted=False, current_improved=False):
    """g_t for the state after `it` completed search iterations.

    The three flags describe the OUTCOME OF THE PREVIOUS iteration
    (new best found / candidate SA-accepted / accepted and cheaper
    than the previous current solution). Defaults False = first state
    of an episode, matching the DR-ALNS environment reset().
    """
    served = robot_served_customers(sol)
    eps = max(float(pr.alpha_traffic.max()) - 1.0,
              float(pr.alpha_ped.max()) - 1.0, 1e-9)
    rho_cong = min(max((congestion_ratio(pr, served) - 1.0) / eps, 0.0),
                   1.5)
    # cost_difference_best: the paper's "objective <= 0 -> -1" special
    # case cannot occur here (costs are strictly positive).
    cost_difference_best = min(
        max(current_cost / max(best_cost, 1e-9) - 1.0, 0.0), 1.0)
    # it == 0 is the episode's first state: features 2-5 are all 0.0
    # like the DR-ALNS environment's zero-initialized reset() (even
    # though current == best holds trivially at reset)
    is_current_best = (1.0 if it > 0
                       and abs(current_cost - best_cost) <= 1e-9
                       else 0.0)
    return torch.tensor([[
        len(served) / max(1, len(pr.C)),            # 0 rho_robot
        rho_cong,                                   # 1 rho_cong
        1.0 if best_improved else 0.0,              # 2
        1.0 if current_accepted else 0.0,           # 3
        1.0 if current_improved else 0.0,           # 4
        is_current_best,                            # 5
        cost_difference_best,                       # 6
        # paper uses the raw stagnation count; normalized here so all
        # inputs share a comparable scale
        min(1.0, stagcount / max(1, search_iterations)),        # 7
        min(max(it / max(1, search_iterations), 0.0), 1.0),     # 8
    ]], dtype=torch.float32)
"""Solution + Params -> torch_geometric HeteroData.

The graph is always built from a repair-complete solution (every
customer served); never from a mid-destroy partial solution.

Node types: customer (5 feats), parking (6 feats, one node per copy in
pr.P including unused copies), depot (3 feats, out/in). Every edge feature
is a normalized lookup from the precomputed distance and truck/robot travel
time matrices. Edge families are truck_arc (2 feats), robot_arc (3 feats),
and proximity k-NN (3 feats, customer-customer / customer-parking only).
Every edge is added in both directions with identical features.
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
    boundaries. Padding every graph to one shared key set makes
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


def forward_schedule(pr, sol):
    """Arrival times and parking-stop stats of the current solution.

    Re-runs the forward schedule used by ``alns.eval_truck`` with the
    same recursion (26)-(34).

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
        T = horizon(pr)
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
                     pr.dist(i, j) / d_max])
            for st in route:
                if st["kind"] != "park":
                    continue
                for tr in st["deploys"]:
                    legs = ([st["p"]] + list(tr["custs"]) + [tr["ret_p"]])
                    for i, j in zip(legs, legs[1:]):
                        add(i, j, "robot_arc",
                            [pr.tau_robot(i, j) / c_robot,
                             pr.dist(i, j) / d_max,
                             pr.tau_truck(i, j) / c_truck])

        if self.cfg.use_proximity:
            cand_ids = cust_ids + park_ids
            for c in cust_ids:
                near = sorted((n for n in cand_ids if n != c),
                              key=lambda n: pr.dist(c, n))
                for n in near[:self.cfg.knn_k]:
                    add(c, n, "proximity",
                        [pr.dist(c, n) / d_max,
                         pr.tau_robot(c, n) / c_robot,
                         pr.tau_truck(c, n) / c_truck])

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
        """Incremental update stub; the current implementation rebuilds."""
        return self.build(pr, sol)
"""Heterogeneous edge-aware GAT encoder + joint mean/max pooling.

State s_t = concat(mean_pool, max_pool over ALL node types jointly,
g_t) -> 2 * hidden_dim + G_DIM dims (137 with defaults). With
cfg.use_graph = False the encoder is skipped and s_t = g_t.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, global_max_pool, \
    global_mean_pool


NODE_DIMS = {"customer": 5, "parking": 6, "depot": 3}


class SolutionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.embed = nn.ModuleDict(
            {t: nn.Linear(dim, d) for t, dim in NODE_DIMS.items()})
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            convs = {et: GATv2Conv(d, d, heads=cfg.heads, concat=False,
                                   edge_dim=EDGE_DIMS[et[1]],
                                   add_self_loops=False)
                     for et in EDGE_TYPES}
            self.layers.append(HeteroConv(convs, aggr="sum"))
            self.norms.append(nn.ModuleDict(
                {t: nn.LayerNorm(d) for t in NODE_DIMS}))
        self.act = nn.GELU()

    def forward(self, data):
        """data: HeteroData or Batch -> [B, 2 * hidden_dim]."""
        x = {t: self.act(self.embed[t](data[t].x)) for t in NODE_DIMS}
        eidx = {et: data[et].edge_index for et in data.edge_types}
        eattr = {et: data[et].edge_attr for et in data.edge_types}
        for conv, ln in zip(self.layers, self.norms):
            out = conv(x, eidx, eattr)
            # residual + LayerNorm; node types unseen by any edge type
            # keep their previous embedding
            x = {t: ln[t](x[t] + self.act(out[t])) if t in out else x[t]
                 for t in x}
        hs, batches = [], []
        for t in NODE_DIMS:
            h = x[t]
            b = data[t].batch if hasattr(data[t], "batch") else \
                torch.zeros(h.size(0), dtype=torch.long,
                            device=h.device)
            hs.append(h)
            batches.append(b)
        h_all = torch.cat(hs, dim=0)
        b_all = torch.cat(batches, dim=0)
        return torch.cat([global_mean_pool(h_all, b_all),
                          global_max_pool(h_all, b_all)], dim=1)
