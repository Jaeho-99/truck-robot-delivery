"""All PPO hyperparameters in one dataclass.

Values follow the DR-ALNS public repository's PPO config (SB3
defaults where unlisted): n_steps 256 per worker, 10 workers, batch 64,
10 epochs, ent_coef 0, no KL early stop. Training protocol matches the
DQN side: 300k total steps in episodes of 100 search iterations, one
model per instance size.
"""

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
    # ppo update (DR-ALNS repo values)
    lr: float = 3e-4                # learning_rate
    adam_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2           # clip_range
    c1: float = 0.5                 # vf_coef
    c2_start: float = 0.0           # ent_coef = 0.0 (DR-ALNS; the
    c2_end: float = 0.0             #   old 0.02->0.003 decay is off)
    k_epochs: int = 10              # n_epochs
    n_minibatch: int = 40           # rollout 2560 / batch_size 64
    target_kl: float | None = None  # SB3 default: no KL early stop
    max_grad_norm: float = 0.5
    # environment / rollout (DR-ALNS protocol)
    n_envs: int = 10                # n_workers
    t_rollout: int = 256            # n_steps per worker
    search_iterations: int = 100    # episode length, size-independent
    # training
    total_steps: int = 300_000      # summed over workers
    # misc
    device: str = "cpu"
    seed: int = 0

    @property
    def n_updates(self):
        """Derived: updates = total_steps / (t_rollout * n_envs),
        floor (299,520 of 300,000 steps with the defaults)."""
        return self.total_steps // (self.t_rollout * self.n_envs)

    def to_dict(self):
        return asdict(self)
