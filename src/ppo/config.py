"""All PPO hyperparameters in one dataclass (spec: config section)."""

from dataclasses import asdict, dataclass


@dataclass
class PPOConfig:
    # encoder (shared architecture with the DQN — controlled
    # comparison; fields consumed by SolutionEncoder / GraphBuilder)
    hidden_dim: int = 64
    n_layers: int = 3
    heads: int = 2
    knn_k: int = 5
    use_proximity: bool = True
    # model
    n_actions: int = 9              # 3 destroy x 3 repair, a = di*3+ri
    # ppo update
    lr: float = 3e-4
    adam_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    c1: float = 0.5
    c2_start: float = 0.02
    c2_end: float = 0.003
    k_epochs: int = 4
    n_minibatch: int = 8
    target_kl: float = 0.02
    max_grad_norm: float = 0.5
    # environment / rollout
    n_envs: int = 16
    t_rollout: int = 512
    max_iter: int = 500             # ALNS horizon (progress + cooling)
    # training
    n_updates: int = 500
    # misc
    device: str = "cpu"
    seed: int = 0

    def to_dict(self):
        return asdict(self)
