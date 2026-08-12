"""PPO operator selection for ALNS (docs/ppo_spec.md).

On-policy counterpart to src.gnn_dqn for a controlled DQN-vs-PPO
comparison: the GATv2 encoder (SolutionEncoder) is reused unchanged,
only the head (actor-critic) and the training algorithm differ. All
new code lives in this package; DQN files are never modified.
"""
