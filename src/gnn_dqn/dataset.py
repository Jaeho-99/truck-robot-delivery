"""Directory-based instance loading (Ulsan Nam-gu train/test data).

Replaces on-the-fly instance generation for the learning pipeline:
instances are pre-generated under data/train/n{size}/ and
data/test/n{size}/ and loaded once at start-up (no per-episode disk
I/O; raw payloads are kept in memory and materialized into Params on
demand so per-instance arc caches do not accumulate).

File schema (one JSON per instance):
  instance_id, region, seed [base, n, index], n_customers,
  num_parking_copies, n_zones, zones {id: {admi_cd, admi_nm}},
  origin_katec, nodes [{label, type, x, y, zone}, ...] in the order
  C1..Cn, P0#1..P{G}#2, D, alpha_traffic {zone: a}, alpha_ped
  {zone: a}.

Fields the old pipeline generated but the files do not carry are
reconstructed here:
  - arc_zones: the files store no zone geometry (zones are
    administrative areas), so an arc's km-per-zone split cannot be
    traced like the old grid stepping. Convention: an arc (i, j) of
    length d is split half/half between its endpoint zones
    ({z_i: d/2, z_j: d/2}; {z_i: d} if equal). Computed lazily.
  - soft time windows: regenerated with instance.reachability_tw's
    law, seeded deterministically from the file's seed triple.
  - fleet (num_trucks / num_robots / beta_robot): provider arguments
    (the files carry customers and parkings only).
"""

import hashlib
import json
import os
import random

from ..heuristics import Params
from .. import instance

# Provisional per-size fleet defaults (num_trucks, num_robots_per_
# truck): customers-per-truck kept roughly constant (~10-15) across
# sizes, robots-per-truck fixed so the robot-delivery structure stays
# comparable. NOT final — override via provider args / CLI flags.
DEFAULT_FLEET = {20: (2, 2), 50: (4, 2), 100: (7, 2)}


class _LazyArcZones(dict):
    """arc_zones with the endpoint half-split convention, on demand."""

    def __init__(self, nodes, node_zone):
        super().__init__()
        self._nodes = nodes
        self._zone = node_zone

    def __missing__(self, key):
        i, j = key
        xi, yi = self._nodes[i]
        xj, yj = self._nodes[j]
        d = abs(xi - xj) + abs(yi - yj)
        zi, zj = self._zone[i], self._zone[j]
        val = {zi: d} if zi == zj else {zi: d / 2.0, zj: d / 2.0}
        self[key] = val
        return val


def _tw_seed(seed_triple):
    """Deterministic int for reachability_tw from the file's seed."""
    s0, s1, s2 = seed_triple
    return s0 * 1_000_003 + s1 * 1_009 + s2


def payload_to_inst(payload, num_trucks, num_robots, beta_robot):
    """File payload -> inst dict in the schema Params expects.

    Node indexing follows the old generator convention: 0 = depot
    (route start), 1..n = customers, then parking copies, last = the
    depot end copy.
    """
    by_type = {"customer": [], "parking": [], "depot": []}
    for node in payload["nodes"]:
        by_type[node["type"]].append(node)
    depot = by_type["depot"][0]

    nodes, node_zone, meta = {}, {}, {}

    def put(idx, node):
        nodes[idx] = (node["x"], node["y"])
        node_zone[idx] = int(node["zone"])
        meta[idx] = {"label": node["label"], "type": node["type"]}

    put(0, dict(depot, label="D", type="depot"))
    C = []
    idx = 1
    for node in by_type["customer"]:
        put(idx, node)
        C.append(idx)
        idx += 1
    P, groups = [], {}
    for node in by_type["parking"]:
        put(idx, node)
        P.append(idx)
        groups.setdefault(node["label"].split("#")[0], []).append(idx)
        idx += 1
    D = idx
    put(D, dict(depot, label="D", type="depot"))

    K = list(range(1, num_trucks + 1))
    return {
        "nodes": nodes, "node_zone": node_zone, "meta": meta,
        "C": C, "P": P, "D": D, "K": K,
        "R_k": {k: list(range(1, num_robots + 1)) for k in K},
        "park_groups": [groups[g] for g in sorted(
            groups, key=lambda s: int(s[1:]))],
        "num_parking_copies": payload["num_parking_copies"],
        "lam": {c: 1 for c in C},
        "arc_zones": _LazyArcZones(nodes, node_zone),
        "alpha_traffic": {int(z): a
                          for z, a in payload["alpha_traffic"].items()},
        "alpha_ped": {int(z): a
                      for z, a in payload["alpha_ped"].items()},
        "beta_robot": beta_robot,
        "seed": payload["seed"],
        "instance_id": payload["instance_id"],
    }


def _content_hash(payload):
    """Content fingerprint ignoring the instance_id naming."""
    body = {k: v for k, v in payload.items() if k != "instance_id"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


class DirectoryInstanceProvider:
    """Loads data/{train,test}/n{size} once; samples with replacement.

    sample() materializes a fresh Params per call (random.choice over
    the in-memory train pool — DR-ALNS protocol), so tau/distance
    caches stay per-episode instead of accumulating over 250
    instances.
    """

    def __init__(self, size, root="data", num_trucks=None,
                 num_robots=None, beta_robot=3, seed=0):
        self.size = size
        default_trucks, default_robots = DEFAULT_FLEET.get(size, (4, 2))
        self.fleet = dict(
            num_trucks=num_trucks or default_trucks,
            num_robots=num_robots or default_robots,
            beta_robot=beta_robot)
        self.rng = random.Random(seed)
        self.train = self._load(os.path.join(root, "train", f"n{size}"))
        self.test = self._load(os.path.join(root, "test", f"n{size}"))
        overlap = ({_content_hash(p) for p in self.train}
                   & {_content_hash(p) for p in self.test})
        if overlap:
            raise ValueError(
                f"train/test overlap for n{size}: {len(overlap)} "
                f"identical instance(s) found in both sets")
        print(f"[data] n{size}: {len(self.train)} train / "
              f"{len(self.test)} test instances loaded "
              f"(fleet: {self.fleet})", flush=True)

    @staticmethod
    def _load(path):
        files = sorted(f for f in os.listdir(path)
                       if f.endswith(".json"))
        if not files:
            raise FileNotFoundError(f"no instances under {path}")
        return [json.load(open(os.path.join(path, f))) for f in files]

    def _params(self, payload):
        inst = payload_to_inst(payload, **{
            "num_trucks": self.fleet["num_trucks"],
            "num_robots": self.fleet["num_robots"],
            "beta_robot": self.fleet["beta_robot"]})
        e_c, l_c = instance.reachability_tw(
            inst, _tw_seed(payload["seed"]))
        return Params(inst, e_c, l_c, beta_robot=inst["beta_robot"])

    def sample(self):
        """Random draw with replacement from the train pool."""
        return self._params(self.rng.choice(self.train))

    def test_set(self):
        """[(instance_id, Params), ...] over the full test split."""
        return [(p["instance_id"], self._params(p)) for p in self.test]
