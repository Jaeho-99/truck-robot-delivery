"""Global normalization constants for graph features.

Computed once from a sample of training instances, saved to JSON, and
loaded by both training and inference (never recomputed per instance).
Out-of-distribution values may exceed 1.0 after normalization; only
congestion-exposure features are clipped (in graph_builder), not
tau/distance features.
"""

import json


def compute_norms(prs, pct=0.95):
    """95th-percentile normalizers over all node pairs of ``prs``.

    Returns {"c_truck": ..., "c_robot": ..., "d_max": ...}.
    """
    taus_t, taus_r, dists = [], [], []
    for pr in prs:
        ids = [0] + list(pr.C) + list(pr.P) + [pr.D]
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                taus_t.append(pr.tau_truck(i, j))
                taus_r.append(pr.tau_robot(i, j))
                dists.append(pr.dist(i, j))

    def q(xs):
        xs = sorted(xs)
        k = min(len(xs) - 1, int(pct * len(xs)))
        return max(xs[k], 1e-9)

    return {"c_truck": q(taus_t), "c_robot": q(taus_r),
            "d_max": q(dists)}


def save_norms(norms, path):
    with open(path, "w") as f:
        json.dump(norms, f, indent=2)


def load_norms(path):
    with open(path) as f:
        return json.load(f)
