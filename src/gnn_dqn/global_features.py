"""Global search-state features g_t (9-dim float32 vector).

Shared by BOTH the DQN path (trainer, selector_gnn/solve_alns) and the
PPO path (src/ppo/env) — this module is the single source of the g_t
definition.

Features 2-8 follow the DR-ALNS observation space (Reijnen et al.,
ICAPS 2024); 0-1 are problem-specific extensions of this work (robot
share / congestion exploitation). Our objective is minimized, so
"improved" means a cost DECREASE (the original paper maximizes).
"""

import torch

from ..heuristics.qlearning import (congestion_ratio,
                                    robot_served_customers)

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
    eps = max(max(pr.alpha_traffic.values()) - 1.0,
              max(pr.alpha_ped.values()) - 1.0, 1e-9)
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
