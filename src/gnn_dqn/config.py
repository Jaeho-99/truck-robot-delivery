"""All GNN+DQN hyperparameters and ablation flags in one dataclass."""

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
    # dqn
    n_actions: int = 9              # 3 destroy x 3 repair (see dqn_agent)
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_capacity: int = 50_000
    warmup: int = 1_000
    train_freq: int = 4
    target_sync: int = 2_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.5
    grad_clip: float = 10.0
    # training
    n_episodes: int = 3_000
    episode_len: int = 500
    # reward
    reward_mode: str = "R1"         # "R1" | "R2" | "binary"
    kappa: float = 0.02
    # misc
    device: str = "cpu"
    seed: int = 0

    def to_dict(self):
        return asdict(self)
