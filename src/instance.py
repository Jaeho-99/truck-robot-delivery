"""Synthetic grid instances for truck-robot collaborative delivery.

An instance is a plain dict with the following keys:
  nodes        {node_id: (x_km, y_km)}; node 0 is depot-out, node D is
               depot-in (same coordinates)
  node_zone    {node_id: zone_id}
  meta         {node_id: {"label", "type", ...}}
  C            customer node ids
  P            parking node ids (all co-located copies, set P-bar)
  D            depot-in node id
  K            truck ids
  R_k          {truck_id: [robot ids]}
  park_groups  [[copy ids of one physical parking location], ...]
  lam          {customer: demand}
  arc_zones    {(i, j): {zone: km traversed}} (Manhattan decomposition)
  alpha_traffic, alpha_ped   {zone: congestion factor}
  grid_x, grid_y, zone_km    grid geometry (for plotting payloads)

Node structure (formulation Section 3.2):
  0                       depot-out
  C                       customers
  P-bar (= copies)        co-located copies of each parking location
  c_max + p_max + 1 (= D) depot-in
  N = C u P-bar u {0, D}

Two instance families are provided:
  * build_grid_instance   one random customer per zone on a grid
                          (used for the small exact-vs-ALNS comparison)
  * build_master / build_scaling_instance
                          a fixed master pool of customers sliced to the
                          first n, so instances are nested across sizes
                          (used for the scaling comparison)
"""

import random

import numpy as np

# --- speeds (km/h) ---
V_TRUCK = 20.5   # free-flow truck speed
V_ROBOT = 5.0

# --- per-km cost primitives ---
GFUEL_TRUCK_KM = 0.13     # $/km
GFUEL_ROBOT_KM = 0.003    # $/km
EMIS_TRUCK_KM = 0.414     # kgCO2e/km
EMIS_ROBOT_KM = 0.0103    # kgCO2e/km

# --- upper bounds of the random congestion factors ---
ALPHA_TRAFFIC_MAX = 2.56
ALPHA_PED_MAX = 1.76


# ==========================================
# 0. Zone decomposition (grid, Manhattan paths)
# ==========================================
def zone_of(x, y, grid_x, grid_y, zone_km):
    zx = min(grid_x - 1, max(0, int(x // zone_km)))
    zy = min(grid_y - 1, max(0, int(y // zone_km)))
    return zx * grid_y + zy


def manhattan_zone_km(pi, pj, grid_x, grid_y, zone_km, step=0.1):
    """Decompose an arc into per-zone traversed distance (d_ijz, S3.4).

    The two Manhattan L-paths (horizontal-first / vertical-first) are
    sampled at ``step`` km and averaged; each sample point is assigned
    to its grid zone. Returns {zone: km}; values sum to the Manhattan
    distance.
    """
    xi, yi = pi
    xj, yj = pj
    dx, dy = xj - xi, yj - yi
    total = abs(dx) + abs(dy)
    if total == 0:
        return {}
    n = max(1, int(round(total / step)))
    t = (np.arange(n) + 0.5) / n * total
    sx, sy = np.sign(dx), np.sign(dy)
    nzone = grid_x * grid_y
    acc = np.zeros(nzone)
    for order in ("h", "v"):
        x = np.empty(n)
        y = np.empty(n)
        if order == "h":
            m = t <= abs(dx)
            x[m] = xi + sx * t[m]
            y[m] = yi
            x[~m] = xj
            y[~m] = yi + sy * (t[~m] - abs(dx))
        else:
            m = t <= abs(dy)
            y[m] = yi + sy * t[m]
            x[m] = xi
            y[~m] = yj
            x[~m] = xi + sx * (t[~m] - abs(dy))
        zx = np.clip((x // zone_km).astype(int), 0, grid_x - 1)
        zy = np.clip((y // zone_km).astype(int), 0, grid_y - 1)
        np.add.at(acc, zx * grid_y + zy, (total / n) * 0.5)
    return {z: float(acc[z]) for z in range(nzone) if acc[z] > 1e-12}


# ==========================================
# 1. Grid instance (one customer per zone, parking copies P-bar)
# ==========================================
def even_parking_positions(num, grid_x, grid_y, zone_km,
                           avoid=None, avoid_radius=None):
    """Spread ``num`` parking locations evenly; candidates = zone centers.

    Candidates within ``avoid_radius`` of ``avoid`` (e.g. the depot) are
    excluded, then farthest-point sampling picks the locations.
    """
    centers = [((zx + 0.5) * zone_km, (zy + 0.5) * zone_km)
               for zx in range(grid_x) for zy in range(grid_y)]
    map_x, map_y = grid_x * zone_km, grid_y * zone_km
    if avoid is not None:
        ax, ay = avoid
        r2 = (avoid_radius if avoid_radius is not None
              else zone_km * 0.75) ** 2
        centers = [c for c in centers
                   if (c[0] - ax) ** 2 + (c[1] - ay) ** 2 > r2]
        start = max(range(len(centers)),
                    key=lambda i: ((centers[i][0] - ax) ** 2
                                   + (centers[i][1] - ay) ** 2))
    else:
        cx, cy = map_x / 2, map_y / 2
        start = min(range(len(centers)),
                    key=lambda i: ((centers[i][0] - cx) ** 2
                                   + (centers[i][1] - cy) ** 2))
    if num >= len(centers):
        return centers
    chosen = [start]
    while len(chosen) < num:
        best, best_d = None, -1.0
        for i in range(len(centers)):
            if i in chosen:
                continue
            d = min((centers[i][0] - centers[j][0]) ** 2
                    + (centers[i][1] - centers[j][1]) ** 2
                    for j in chosen)
            if d > best_d:
                best_d, best = d, i
        chosen.append(best)
    return [centers[i] for i in chosen]


def build_grid_instance(seed=1, beta_robot=3, num_trucks=2, num_robots=2,
                        num_parking_copies=2, grid_x=3, grid_y=3,
                        zone_km=10.0, cust_per_zone=1, num_parking=5,
                        parking_avoid_radius=None):
    """Build a grid_x x grid_y zone instance.

    ``cust_per_zone`` customers per zone; ``num_parking`` physical
    parking locations, each materialised as ``num_parking_copies``
    co-located copies. The depot sits at the map center.
    ``parking_avoid_radius`` (km) keeps parking candidates away from the
    depot (default: 0.75 x zone_km inside even_parking_positions).
    """
    rng = random.Random(seed)
    map_x, map_y = grid_x * zone_km, grid_y * zone_km
    nzone = grid_x * grid_y
    nodes, node_zone, meta = {}, {}, {}
    nodes[0] = (map_x / 2, map_y / 2)           # depot at map center
    node_zone[0] = zone_of(*nodes[0], grid_x, grid_y, zone_km)
    meta[0] = {"label": "D", "type": "depot"}

    def rand_in_zone(zx, zy):
        return (rng.uniform(zx * zone_km, (zx + 1) * zone_km),
                rng.uniform(zy * zone_km, (zy + 1) * zone_km))

    C, P, idx = [], [], 1
    for zx in range(grid_x):
        for zy in range(grid_y):
            z = zx * grid_y + zy
            for n in range(1, cust_per_zone + 1):
                nodes[idx] = rand_in_zone(zx, zy)
                node_zone[idx] = z
                clabel = f"C{z}" if cust_per_zone == 1 else f"C{z}-{n}"
                meta[idx] = {"label": clabel, "type": "customer"}
                C.append(idx)
                idx += 1
    park_groups = []
    positions = even_parking_positions(num_parking, grid_x, grid_y,
                                       zone_km, avoid=nodes[0],
                                       avoid_radius=parking_avoid_radius)
    for pidx, (px, py) in enumerate(positions):
        z = zone_of(px, py, grid_x, grid_y, zone_km)
        plabel = f"P{pidx}"
        grp = []
        for cpy in range(1, num_parking_copies + 1):
            nodes[idx] = (px, py)
            node_zone[idx] = z
            meta[idx] = {"label": f"{plabel}#{cpy}", "type": "parking",
                         "phys": plabel, "copy": cpy}
            P.append(idx)
            grp.append(idx)
            idx += 1
        park_groups.append(grp)
    D = idx                             # depot-in = c_max + p_max + 1
    nodes[D] = nodes[0]
    node_zone[D] = node_zone[0]
    meta[D] = dict(meta[0])

    K = list(range(1, num_trucks + 1))
    R_k = {k: list(range(1, num_robots + 1)) for k in K}
    arc_zones = {(i, j): manhattan_zone_km(nodes[i], nodes[j],
                                           grid_x, grid_y, zone_km)
                 for i in nodes for j in nodes if i != j}
    alpha_traffic = {z: round(rng.uniform(1.0, ALPHA_TRAFFIC_MAX), 3)
                     for z in range(nzone)}
    alpha_ped = {z: round(rng.uniform(1.0, ALPHA_PED_MAX), 3)
                 for z in range(nzone)}

    return {
        "nodes": nodes, "node_zone": node_zone, "meta": meta,
        "C": C, "P": P, "D": D, "K": K, "R_k": R_k,
        "park_groups": park_groups,
        "num_parking_copies": num_parking_copies,
        "lam": {c: 1 for c in C}, "arc_zones": arc_zones,
        "alpha_traffic": alpha_traffic, "alpha_ped": alpha_ped,
        "beta_robot": beta_robot, "seed": seed,
        "grid_x": grid_x, "grid_y": grid_y, "zone_km": zone_km,
    }


def reachability_tw(inst, seed):
    """Reachability-based soft time windows under free flow (alpha = 1).

    e_c = 0 and l_c = free-flow truck time depot->c x U(3, 5) + 120 min.
    """
    rng = random.Random(seed * 31 + 7)
    nodes = inst["nodes"]

    def tau_ff(i, j):
        d = abs(nodes[i][0] - nodes[j][0]) + abs(nodes[i][1] - nodes[j][1])
        return d / V_TRUCK * 60.0

    e_c = {c: 0.0 for c in inst["C"]}
    l_c = {c: tau_ff(0, c) * rng.uniform(3.0, 5.0) + 120.0
           for c in inst["C"]}
    return e_c, l_c


# ==========================================
# 2. Scaling instances — fixed master pool, sliced to n customers
# ==========================================
def parking_positions(grid_x, grid_y, zone_km):
    """One parking location per zone, fixed at the zone center.

    The center zone's parking would coincide with the depot, so it is
    offset by (+2.5, +2.5) km.
    """
    map_x, map_y = grid_x * zone_km, grid_y * zone_km
    depot = (map_x / 2, map_y / 2)
    pos = []
    for zx in range(grid_x):
        for zy in range(grid_y):
            cx, cy = (zx + 0.5) * zone_km, (zy + 0.5) * zone_km
            if abs(cx - depot[0]) < 1e-9 and abs(cy - depot[1]) < 1e-9:
                cx, cy = cx + 2.5, cy + 2.5
            pos.append((cx, cy))
    return pos


def build_master(seed, grid_x=3, grid_y=3, zone_km=10.0, max_cust=100):
    """Data shared by all sizes of one scaling experiment.

    A single seed fixes: ``max_cust`` customer coordinates and their
    soft time windows, the zone congestion factors (independent random
    stream, hence identical for every n), and the parking locations.
    """
    map_x, map_y = grid_x * zone_km, grid_y * zone_km
    nzone = grid_x * grid_y
    depot = (map_x / 2, map_y / 2)
    rng_c = random.Random(seed)                   # customer coordinates
    cust_xy = [(rng_c.uniform(0.0, map_x), rng_c.uniform(0.0, map_y))
               for _ in range(max_cust)]
    rng_a = random.Random(seed * 97 + 13)         # congestion factors
    alpha_traffic = {z: round(rng_a.uniform(1.0, ALPHA_TRAFFIC_MAX), 3)
                     for z in range(nzone)}
    alpha_ped = {z: round(rng_a.uniform(1.0, ALPHA_PED_MAX), 3)
                 for z in range(nzone)}
    rng_tw = random.Random(seed * 31 + 7)         # same law as
    l_master = []                                 # reachability_tw
    for (x, y) in cust_xy:
        d = abs(x - depot[0]) + abs(y - depot[1])
        tau_ff = d / V_TRUCK * 60.0
        l_master.append(tau_ff * rng_tw.uniform(3.0, 5.0) + 120.0)
    return {"depot": depot, "cust_xy": cust_xy, "l_master": l_master,
            "alpha_traffic": alpha_traffic, "alpha_ped": alpha_ped,
            "park_xy": parking_positions(grid_x, grid_y, zone_km),
            "seed": seed, "grid_x": grid_x, "grid_y": grid_y,
            "zone_km": zone_km}


def build_scaling_instance(master, n, num_trucks=5, num_robots=3,
                           num_parking_copies=2, beta_robot=3):
    """Instance with the first n customers of the master pool.

    Returns (inst, e_c, l_c); inst follows the build_grid_instance
    schema, so the customers of n=5 are a subset of those of n=10, etc.
    """
    gx, gy, zk = master["grid_x"], master["grid_y"], master["zone_km"]
    nodes, node_zone, meta = {}, {}, {}
    nodes[0] = master["depot"]
    node_zone[0] = zone_of(*nodes[0], gx, gy, zk)
    meta[0] = {"label": "D", "type": "depot"}

    C, idx = [], 1
    for ci in range(n):
        nodes[idx] = master["cust_xy"][ci]
        node_zone[idx] = zone_of(*nodes[idx], gx, gy, zk)
        meta[idx] = {"label": f"C{ci + 1}", "type": "customer"}
        C.append(idx)
        idx += 1

    P, park_groups = [], []
    for pidx, (px, py) in enumerate(master["park_xy"]):
        grp = []
        for cpy in range(1, num_parking_copies + 1):
            nodes[idx] = (px, py)
            node_zone[idx] = zone_of(px, py, gx, gy, zk)
            meta[idx] = {"label": f"P{pidx}#{cpy}", "type": "parking",
                         "phys": f"P{pidx}", "copy": cpy}
            P.append(idx)
            grp.append(idx)
            idx += 1
        park_groups.append(grp)

    D = idx
    nodes[D] = nodes[0]
    node_zone[D] = node_zone[0]
    meta[D] = dict(meta[0])

    K = list(range(1, num_trucks + 1))
    R_k = {k: list(range(1, num_robots + 1)) for k in K}
    arc_zones = {(i, j): manhattan_zone_km(nodes[i], nodes[j], gx, gy, zk)
                 for i in nodes for j in nodes if i != j}
    inst = {
        "nodes": nodes, "node_zone": node_zone, "meta": meta,
        "C": C, "P": P, "D": D, "K": K, "R_k": R_k,
        "park_groups": park_groups,
        "num_parking_copies": num_parking_copies,
        "lam": {c: 1 for c in C}, "arc_zones": arc_zones,
        "alpha_traffic": master["alpha_traffic"],
        "alpha_ped": master["alpha_ped"],
        "beta_robot": beta_robot, "seed": master["seed"],
        "grid_x": gx, "grid_y": gy, "zone_km": zk,
    }
    e_c = {c: 0.0 for c in C}
    l_c = {c: master["l_master"][ci] for ci, c in enumerate(C)}
    return inst, e_c, l_c


# ==========================================
# 3. JSON payload (input of plotting.make_instance_svg)
# ==========================================
def instance_payload(inst):
    nodes = [{"label": inst["meta"][i]["label"],
              "type": inst["meta"][i]["type"],
              "x": inst["nodes"][i][0], "y": inst["nodes"][i][1],
              "zone": inst["node_zone"][i]}
             for i in inst["C"] + inst["P"] + [0]]
    return {"grid": inst["grid_x"], "grid_x": inst["grid_x"],
            "grid_y": inst["grid_y"], "zone_km": inst["zone_km"],
            "nodes": nodes,
            "alpha_traffic": inst["alpha_traffic"],
            "alpha_ped": inst["alpha_ped"]}
