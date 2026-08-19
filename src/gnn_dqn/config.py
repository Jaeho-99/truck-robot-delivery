"""All GNN+DQN hyperparameters and ablation flags in one dataclass.

Training protocol and the DQN section follow DR-ALNS (Reijnen et al.,
ICAPS 2024): 300k total steps, episodes of 100 search iterations,
instances drawn with replacement per episode, one model per instance
size. Exception kept on purpose: learning rate stays at the previous
1e-4 instead of DR-ALNS's 1e-3 — our Q-net carries a GNN encoder and
the larger rate risks instability.
"""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # encoder
    hidden_dim: int = 64
    n_layers: int = 3
    heads: int = 2
    knn_k: int = 5
    use_graph: bool = True          # False -> g_t-only ablation
    use_proximity: bool = True      # ablation
    # dqn (DR-ALNS values unless noted)
    n_actions: int = 9              # 3 destroy x 3 repair (see dqn_agent)
    gamma: float = 0.99
    lr: float = 1e-4                # kept from previous code, NOT the
    #                                 DR-ALNS 1e-3 (GNN encoder stability)
    batch_size: int = 32
    buffer_capacity: int = 20_000
    warmup: int = 1_000             # learning starts
    train_freq: int = 1
    target_sync: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_frac: float = 0.1     # decay over first 10% of steps
    grad_clip: float = 10.0
    # training protocol (DR-ALNS)
    total_steps: int = 300_000      # total search iterations trained on
    search_iterations: int = 100    # episode length, fixed per episode
    # reward
    reward_mode: str = "R1"         # "R1" | "R2" | "binary"
    kappa: float = 0.02
    # misc
    device: str = "cpu"
    seed: int = 0

    @property
    def n_episodes(self):
        """Derived: episode count = total_steps / search_iterations."""
        return self.total_steps // self.search_iterations

    def to_dict(self):
        return asdict(self)
