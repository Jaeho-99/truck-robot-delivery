"""Tabular Q-learning operator selection for ALNS (online, single run).

State  = (robot utilization bin) x (congestion efficiency bin)
Action = (destroy index, repair index) pair
Reward = existing sigma scores (new best / improve / SA-accepted new)

Drop-in replacement for the roulette-wheel selection in ``solve_alns``
(pass ``selector="qlearning"``). No offline training: the Q-table
starts at zero and is updated every iteration within a single run.
"""

import math
import random

# --------------------------------------------------------------------
# State features
# --------------------------------------------------------------------

# Robot-served customer ratio: 6 bins — exactly zero, then five
# width-0.2 bins (0, .2], (.2, .4], ..., (.8, 1].
N_ROBOT_BINS = 6
# Congestion ratio (>= 1.0 by construction): bins split at these edges,
# i.e. <1.1, 1.1-1.2, ..., 1.4-1.5, >=1.5.
CONG_BIN_EDGES = [1.1, 1.2, 1.3, 1.4, 1.5]

N_CONG_BINS = len(CONG_BIN_EDGES) + 1
N_STATES = N_ROBOT_BINS * N_CONG_BINS


def _bin(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


STATE_LABELS = [
    f"(rob {rb}/{N_ROBOT_BINS}, cong {cb}/{N_CONG_BINS})"
    for rb in range(N_ROBOT_BINS) for cb in range(N_CONG_BINS)
]


def robot_served_customers(sol):
    """Set of customers currently served by robot delivery trips."""
    served = set()
    for route in sol.routes.values():
        for stop in route:
            if stop["kind"] == "park":
                for trip in stop["deploys"]:
                    served.update(trip["custs"])
    return served


def congestion_ratio(pr, robot_served):
    """Assigned-mode congestion divided by the per-customer ideal.

    numerator   = sum over customers of the congestion coefficient of
                  the *currently assigned* mode (ped if robot-served,
                  traffic if truck-served)
    denominator = sum over customers of min(traffic, ped) coefficient
                  (mode-assignment lower bound, so ratio >= 1.0)
    """
    zc = pr.inst["node_zone"]
    num = 0.0
    den = 0.0
    for c in pr.C:
        at = pr.alpha_traffic[zc[c]]
        ap = pr.alpha_ped[zc[c]]
        num += ap if c in robot_served else at
        den += min(at, ap)
    return num / den if den > 0 else 1.0


def get_state(pr, sol):
    """Map a solution to a discrete state index in [0, N_STATES)."""
    served = robot_served_customers(sol)
    ratio = len(served) / max(1, len(pr.C))
    rbin = min(N_ROBOT_BINS - 1, math.ceil(ratio / 0.2))
    cbin = _bin(congestion_ratio(pr, served), CONG_BIN_EDGES)
    return rbin * N_CONG_BINS + cbin


# --------------------------------------------------------------------
# Q-table with epsilon-greedy selection
# --------------------------------------------------------------------

class QTable:
    """Minimal tabular Q-learning for operator-pair selection."""

    def __init__(self, n_states, n_actions, eta=0.1, gamma=0.9,
                 eps_start=0.30, eps_end=0.05, eps_decay_iters=1500,
                 seed=0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.eta = eta
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_iters = max(1, eps_decay_iters)
        self.q = [[0.0] * n_actions for _ in range(n_states)]
        self.visits = [[0] * n_actions for _ in range(n_states)]
        # Separate stream so the main ALNS rng consumption pattern is
        # not entangled with selection (determinism per seed preserved).
        self.rng = random.Random(seed)

    def epsilon(self, it):
        frac = min(1.0, it / self.eps_decay_iters)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select(self, s, it):
        """Epsilon-greedy action for state s at iteration it."""
        if self.rng.random() < self.epsilon(it):
            a = self.rng.randrange(self.n_actions)
        else:
            row = self.q[s]
            best = max(row)
            ties = [i for i, v in enumerate(row) if v == best]
            a = self.rng.choice(ties)
        self.visits[s][a] += 1
        return a

    def update(self, s, a, r, s_next):
        target = r + self.gamma * max(self.q[s_next])
        self.q[s][a] += self.eta * (target - self.q[s][a])

    # ----------------------------------------------------------------
    # Diagnostics for the analysis section
    # ----------------------------------------------------------------

    def summary(self, action_labels=None):
        """Per-state argmax and visit counts (for reports/heatmaps)."""
        out = []
        for s in range(self.n_states):
            row = self.q[s]
            a_best = row.index(max(row))
            label = (action_labels[a_best]
                     if action_labels else str(a_best))
            out.append({
                "state": STATE_LABELS[s],
                "argmax": label,
                "q_row": [round(v, 3) for v in row],
                "visits": sum(self.visits[s]),
            })
        return out
