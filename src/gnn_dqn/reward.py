"""Reward variants for the DQN agent (spec section 5)."""


def compute_reward(f_prev, f_new, f_init, f_best, accepted, cfg):
    if cfg.reward_mode == "R1":
        return (f_prev - f_new) / f_init if accepted else 0.0
    if cfg.reward_mode == "R2":
        r = (f_prev - f_new) / f_init if accepted else 0.0
        return r + (cfg.kappa if f_new < f_best else 0.0)
    if cfg.reward_mode == "binary":
        return 5.0 if f_new < f_best else 0.0
    raise ValueError(f"unknown reward_mode: {cfg.reward_mode}")
