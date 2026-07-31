"""Global solution/search features g_t (7-dim float32 vector)."""

import torch

from ..heuristics.qlearning import (congestion_ratio,
                                    robot_served_customers)

STAGNATION_WINDOW = 500     # fixed, not max_iter-relative

G_DIM = 7


def global_features(pr, sol, progress, stagnation_iters, f_cur, f_best):
    served = robot_served_customers(sol)
    eps = max(max(pr.alpha_traffic.values()) - 1.0,
              max(pr.alpha_ped.values()) - 1.0, 1e-9)
    rho_cong = min(max((congestion_ratio(pr, served) - 1.0) / eps, 0.0),
                   1.5)
    n_trucks = sum(1 for r in sol.routes.values() if r)
    robots = {(k, tr["r"]) for k, route in sol.routes.items()
              for st in route if st["kind"] == "park"
              for tr in st["deploys"]}
    tot_robots = sum(len(pr.R_k[k]) for k in pr.K)
    rel_gap = min(max(f_cur / max(f_best, 1e-9) - 1.0, 0.0), 1.0)
    return torch.tensor([[
        len(served) / max(1, len(pr.C)),
        rho_cong,
        n_trucks / max(1, len(pr.K)),
        len(robots) / max(1, tot_robots),
        min(max(progress, 0.0), 1.0),
        min(1.0, stagnation_iters / STAGNATION_WINDOW),
        rel_gap,
    ]], dtype=torch.float32)
