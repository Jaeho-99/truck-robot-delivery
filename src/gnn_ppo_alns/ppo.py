"""GNN-PPO policy, ALNS environments, training, and evaluation.

The policy selects a destroy/repair pair, the environment applies one ALNS
transition, and the rollout buffer feeds PPO updates. Graph transport and
persistent CPU workers live here so both environment backends use the same
state and reward definitions. Device validation stays in the entry points.

Example from the repository root::

    python src/gnn_ppo_alns/train.py --size 20 --device cpu
"""

import dataclasses
import importlib
import math
import multiprocessing
import operator
import os
import random
import statistics
import sys
import tempfile
import time
import traceback
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data, HeteroData

from .alns import (
    ACTION_COUNT,
    ACTION_LABELS,
    DOD,
    NOISE_FRAC,
    W_START,
    apply_actor_action,
    congestion_aware_initial,
    eval_solution,
)
from .gnn import (
    EDGE_DIMS,
    EDGE_TYPES,
    G_DIM,
    NODE_DIMS,
    GraphBuilder,
    SolutionEncoder,
    global_features,
)

# GNN-PPO-ALNS policy, environment, update, training, and testing.
#
# The environment protocol follows DR-ALNS. Stability defaults include an
# entropy floor, KL early stopping, and magnitude-reward scaling.


REWARD_MODES = ("alns_5310", "new_best_5", "magnitude")


def reward_artifact_token(reward_mode):
    """Return the validated semantic token used in artifact filenames."""
    if reward_mode not in REWARD_MODES:
        raise ValueError(
            f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}"
        )
    return f"reward_{reward_mode}"


def calculate_transition_reward(
    reward_mode,
    reward_scale,
    initial_objective,
    previous_best,
    current_best,
    *,
    improved_best,
    improved_current,
    accepted,
    unseen,
):
    """Calculate one PPO reward from the completed ALNS transition."""
    if reward_mode == "alns_5310":
        if improved_best:
            return 5.0
        if improved_current and unseen:
            return 3.0
        if accepted and unseen:
            return 1.0
        return 0.0
    if reward_mode == "new_best_5":
        return 5.0 if improved_best else 0.0
    if reward_mode == "magnitude":
        return (
            reward_scale
            * max(0.0, previous_best - current_best)
            / initial_objective
        )
    raise ValueError(f"unsupported reward mode: {reward_mode!r}")


@dataclass
class PPOConfig:
    """Keep graph, PPO, and ALNS defaults together for saved checkpoints.

    CLI overrides are explicit: for example, ``--size`` selects the dataset,
    while ``--total-steps`` overrides the transition budget in ``train.py``.
    """

    # graph encoder
    hidden_dim: int = 64
    n_layers: int = 3
    heads: int = 2
    knn_k: int = 5
    use_proximity: bool = True
    use_graph: bool = True  # False -> g_t-only "PPO-ALNS"
    #                                 (no GNN encoder, state = g_t)
    # model
    n_actions: int = ACTION_COUNT  # actor-selected joint operator pairs
    # PPO update stability settings (shared by both model variants)
    lr: float = 3e-4  # learning_rate
    adam_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2  # clip_range
    c1: float = 0.5  # vf_coef
    c2_start: float = 0.01  # constant entropy coefficient;
    c2_end: float = 0.01  # prevents PPO-ALNS collapse
    k_epochs: int = 10  # n_epochs
    n_minibatch: int = 40  # rollout 2560 / batch_size 64
    target_kl: float | None = 0.02  # stop epochs above 1.5 * target
    max_grad_norm: float = 0.5
    # environment / rollout (DR-ALNS protocol)
    n_envs: int = 10  # n_workers
    t_rollout: int = 256  # n_steps per worker
    search_iterations: int = 100  # episode length, size-independent
    # Behavior policy used consistently by rollout, PPO update, and test:
    # pi_eps = (1-eps) * pi_actor + eps / n_actions.
    eps_uniform: float = 0.1
    # ALNS search regime (identical to vanilla ALNS by default)
    w_start: float = W_START  # SA start-temperature fraction
    dod: float = DOD  # degree of destruction
    # training
    total_steps: int = 300_000  # summed over workers
    train_count: int = 200  # train files 0..199; rest held out
    # alns_5310: new best/current improvement/accepted/else = 5/3/1/0
    # new_best_5: new best = 5, otherwise 0
    # magnitude: reward_scale * max(0, dBest) / f_init
    reward_mode: str = "magnitude"
    reward_scale: float = 10.0
    # misc
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self):
        reward_artifact_token(self.reward_mode)
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive")

    @property
    def n_updates(self):
        """Derived: updates = total_steps / (t_rollout * n_envs),
        floor (299,520 of 300,000 steps with the defaults)."""
        return self.total_steps // (self.t_rollout * self.n_envs)

    def to_dict(self):
        return asdict(self)


# ActorCritic model for PPO (spec: model section).
#
# Reuses SolutionEncoder unchanged (GATv2 + HeteroConv, joint mean/max
# pooling); state = encoder(data) || g_t -> d_state = 137 with defaults.
# g_t is attached to the batched HeteroData as ``data.g``. Heads carry no
# Dropout/BatchNorm so
# rollout and update see identical outputs.


def _ortho(layer, std):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """Encode the solution graph, then predict action logits and a value."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.n_actions != ACTION_COUNT:
            raise ValueError(
                f"GNN-PPO actor requires {ACTION_COUNT} ALNS actions, "
                f"got {cfg.n_actions}"
            )
        if not 0.0 <= cfg.eps_uniform <= 1.0:
            raise ValueError("eps_uniform must be between 0 and 1")
        self.eps_uniform = float(cfg.eps_uniform)
        self.n_actions = int(cfg.n_actions)
        # cfg.use_graph=False -> g_t-only "PPO-ALNS" ablation (no
        # encoder; state is just the 9-dim search-state vector)
        self.encoder = SolutionEncoder(cfg) if cfg.use_graph else None
        d_state = 2 * cfg.hidden_dim + G_DIM if cfg.use_graph else G_DIM
        self.actor = nn.Sequential(
            _ortho(nn.Linear(d_state, 64), 2**0.5),
            nn.Tanh(),
            _ortho(nn.Linear(64, cfg.n_actions), 0.01),
        )
        self.critic = nn.Sequential(
            _ortho(nn.Linear(d_state, 64), 2**0.5),
            nn.Tanh(),
            _ortho(nn.Linear(64, 1), 1.0),
        )

    def _state(self, data):
        g = data.g.view(-1, G_DIM)
        if self.encoder is None:
            return g
        return torch.cat([self.encoder(data), g], dim=1)

    def _action_probs_from_state(self, state):
        """Effective policy shared by rollout, update, and testing."""
        logits = self.actor(state)
        probs = torch.softmax(logits, dim=-1)
        if self.eps_uniform > 0.0:
            probs = (
                1.0 - self.eps_uniform
            ) * probs + self.eps_uniform / self.n_actions
        return probs

    def action_probs(self, data):
        """Effective actor probabilities, including uniform exploration."""
        return self._action_probs_from_state(self._state(data))

    def get_action_and_value(self, data, action=None):
        """Single entry point for rollout and update (spec).

        action=None samples (rollout); a given action gets its log_prob
        re-evaluated (update). Returns (action, log_prob, entropy,
        value), each of shape (B,).
        """
        s = self._state(data)
        dist = Categorical(probs=self._action_probs_from_state(s))
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(s).squeeze(-1),
        )

    def get_value(self, data):
        """Critic value (B,) only — bootstrap at rollout boundaries."""
        return self.critic(self._state(data)).squeeze(-1)


# On-policy rollout storage for PPO (spec: buffer).
#
# Holds exactly one rollout (t_rollout x n_envs) and is reset after
# every update — never a replay buffer (PPO is on-policy). Graphs are
# stored on CPU with g_t attached as ``data.g``; minibatches are assembled
# with Batch.from_data_list.


class RolloutBuffer:
    def __init__(self, t_rollout, n_envs):
        self.t_rollout = t_rollout
        self.n_envs = n_envs
        self.reset()

    def reset(self):
        self.obs = []  # flat, t-major: index = t * n_envs + i
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = None
        self.returns = None
        self.collate_seconds = 0.0

    def __len__(self):
        return len(self.actions)  # steps stored (of t_rollout)

    def add(self, obs, actions, log_probs, values, rewards, dones):
        """Store one vector step: obs is a list of n_envs HeteroData
        (with .g); the rest are tensors/sequences of shape (n_envs,).
        """
        assert len(self.actions) < self.t_rollout, "buffer full"
        self.obs.extend(obs)
        self.actions.append(torch.as_tensor(actions, dtype=torch.long))
        self.log_probs.append(
            torch.as_tensor(log_probs, dtype=torch.float32).detach()
        )
        self.values.append(
            torch.as_tensor(values, dtype=torch.float32).detach()
        )
        self.rewards.append(torch.as_tensor(rewards, dtype=torch.float32))
        self.dones.append(torch.as_tensor(dones, dtype=torch.float32))

    def compute_returns(self, last_value, gamma, gae_lambda):
        """GAE from the stored rollout-time values (spec)."""
        self.advantages, self.returns = compute_gae(
            torch.stack(self.rewards),
            torch.stack(self.values),
            torch.stack(self.dones),
            torch.as_tensor(last_value, dtype=torch.float32).detach(),
            gamma,
            gae_lambda,
        )

    def minibatches(self, n_minibatch, generator=None):
        """Shuffled minibatches covering the rollout exactly once.

        Yields (graph Batch, actions, log_probs, advantages, returns,
        values), each flattened t-major and index-aligned. Advantages
        are raw — normalization is per-minibatch in the update (spec).
        """
        flat = lambda xs: torch.stack(xs).view(-1)  # noqa: E731
        actions = flat(self.actions)
        log_probs = flat(self.log_probs)
        values = flat(self.values)
        advantages = self.advantages.view(-1)
        returns = self.returns.view(-1)
        perm = torch.randperm(len(self.obs), generator=generator)
        for chunk in perm.chunk(n_minibatch):
            idx = chunk.tolist()
            collate_started = time.perf_counter()
            batch = Batch.from_data_list([self.obs[i] for i in idx])
            self.collate_seconds += time.perf_counter() - collate_started
            yield (
                batch,
                actions[chunk],
                log_probs[chunk],
                advantages[chunk],
                returns[chunk],
                values[chunk],
            )


# Generalized Advantage Estimation (spec: GAE section).
#
# Computed from the values stored at rollout time — update.py must never
# recompute them. done_t marks a true episode end (t == max_iter); a
# rollout-boundary truncation keeps done=0 so the last state bootstraps
# through last_value.


@torch.no_grad()
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Backward-accumulated GAE over a [T, N] rollout.

    rewards/values/dones: [T, N] float tensors (dones in {0, 1});
    last_value: [N] critic value of the state after the final stored
    step (used only where dones[-1] == 0).
    Returns (advantages, returns), both [T, N]; returns = adv + values.
    """
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(rewards.size(0))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    return advantages, advantages + values


# Clipped PPO update step (spec: PPO update section).
#
# One rollout -> up to k_epochs of minibatch updates through a single
# Adam and a single backward per minibatch (no optimizer split). Stored
# log_probs/values come from rollout time; only new_logp/new_v are
# recomputed here.


def make_optimizer(model, cfg):
    """Single Adam over ALL parameters (spec: eps=1e-5, no split)."""
    return torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps)


def c2_schedule(cfg, progress):
    """Entropy coefficient: linear c2_start -> c2_end over training.

    progress = update_idx / (n_updates - 1), clipped to [0, 1].
    """
    p = min(max(progress, 0.0), 1.0)
    return cfg.c2_start + (cfg.c2_end - cfg.c2_start) * p


def value_loss(new_v, old_v, ret, eps):
    """Clipped value loss, pessimistic MAX of the two errors (spec)."""
    v_clip = old_v + torch.clamp(new_v - old_v, -eps, eps)
    return torch.max((new_v - ret) ** 2, (v_clip - ret) ** 2).mean()


def ppo_update(model, optimizer, buffer, cfg, progress, generator=None):
    """Run the PPO epochs on one rollout; returns a metrics dict.

    Stops remaining epochs when the epoch-end approx_kl exceeds
    1.5 * target_kl. Advantages are normalized per minibatch.
    """
    update_started = time.perf_counter()
    collate_before = buffer.collate_seconds
    optimizer_steps_observed = 0
    device = next(model.parameters()).device
    c2 = c2_schedule(cfg, progress)
    eps = cfg.clip_eps
    pg_losses, v_losses, entropies, clipfracs = [], [], [], []
    approx_kl = 0.0
    epochs_run = 0

    for _ in range(cfg.k_epochs):
        kls = []
        for batch, actions, old_logp, adv, ret, old_v in buffer.minibatches(
            cfg.n_minibatch, generator
        ):
            batch = batch.to(device)
            actions, old_logp, adv, ret, old_v = (
                x.to(device) for x in (actions, old_logp, adv, ret, old_v)
            )
            _, new_logp, entropy, new_v = model.get_action_and_value(
                batch, action=actions
            )
            log_ratio = new_logp - old_logp
            ratio = log_ratio.exp()

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = -torch.min(
                ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv
            ).mean()
            v_loss = value_loss(new_v, old_v, ret, eps)
            ent = entropy.mean()
            loss = pg_loss + cfg.c1 * v_loss - c2 * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer_steps_observed += 1

            with torch.no_grad():
                kls.append(((ratio - 1) - log_ratio).mean().item())
                clipfracs.append(
                    ((ratio - 1).abs() > eps).float().mean().item()
                )
            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(ent.item())
        epochs_run += 1
        approx_kl = sum(kls) / len(kls)
        # SB3/DR-ALNS default target_kl=None disables the early stop
        if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
            break

    # explained variance of the rollout-time value estimates
    y = buffer.returns.view(-1)
    v = torch.stack(buffer.values).view(-1)
    var_y = y.var()
    ev = float("nan") if var_y == 0 else (1.0 - (y - v).var() / var_y).item()

    def mean(xs):
        return sum(xs) / len(xs)

    update_seconds = time.perf_counter() - update_started
    return {
        "pg_loss": mean(pg_losses),
        "v_loss": mean(v_losses),
        "entropy": mean(entropies),
        "approx_kl": approx_kl,
        "clipfrac": mean(clipfracs),
        "explained_variance": ev,
        "c2": c2,
        "epochs_run": epochs_run,
        "optimizer_steps_observed": optimizer_steps_observed,
        "ppo_update_seconds": update_seconds,
        "minibatch_collation_seconds": buffer.collate_seconds - collate_before,
        "update_time_per_mb": (
            update_seconds / optimizer_steps_observed
            if optimizer_steps_observed
            else None
        ),
    }


# ALNS as a step-interface environment for PPO rollouts.
#
# Episodes have cfg.search_iterations graph-conditioned actor transitions,
# linear SA temperature decay from T0 = W_START * f_init / ln 2 to 0, and
# fixed degree of destruction q = round(DOD * n). g_t and graph construction
# are local to this method. The configured reward mode is stored in the
# checkpoint and reused at test.


def sa_accept(f_new, f_cur, T, rng):
    """Accept improving candidates; otherwise use simulated annealing."""
    return f_new < f_cur - 1e-9 or rng.random() < math.exp(
        -(f_new - f_cur) / max(T, 1e-9)
    )


class ALNSEnv:
    """Single-instance ALNS with a gym-style step interface.

    reset() -> obs; step(a) -> (obs, reward, done, info). obs is a
    HeteroData with g_t attached as .g. done=True only at
    t == cfg.search_iterations; rollout-boundary truncation is the
    caller's concern.
    """

    def __init__(self, provider, builder, cfg, seed):
        """builder=None (g_t-only PPO): obs carry only .g — the graph
        is never built, which also skips the GraphBuilder cost."""
        self.provider = provider
        self.builder = builder
        self.cfg = cfg
        self.rng = random.Random(seed)

    def reset(self, pr=None):
        """Reset from an explicit instance or the original sampling provider."""
        started = time.perf_counter()
        if pr is None:
            if self.provider is None:
                raise ValueError(
                    "reset() requires a provider or explicit Params"
                )
            pr = self.provider.sample()
        self.pr = pr
        load_seconds = time.perf_counter() - started
        alns_started = time.perf_counter()
        self.sol = congestion_aware_initial(self.pr, self.rng)
        f, _, _, _ = eval_solution(self.pr, self.sol)
        self.f_init = self.f_cur = self.f_best = f
        self.t = 0
        self.stagcount = 0
        self.seen_objectives = set()
        # previous-iteration outcome flags (DR-ALNS obs); reset -> 0
        self.flags = (False, False, False)
        nC = len(self.pr.C)
        # degree of destruction: fixed fraction of customers (DR-ALNS
        # default 30%; cfg.dod allows regime studies)
        self.q_destroy = max(1, round(getattr(self.cfg, "dod", DOD) * nC))
        self.noise_amplitude = NOISE_FRAC * self.f_init
        self.T0 = (
            getattr(self.cfg, "w_start", W_START) * self.f_init
        ) / math.log(2)
        alns_seconds = time.perf_counter() - alns_started
        obs = self._obs()
        self.last_timing = {
            "phase": "reset",
            "load_seconds": load_seconds,
            "alns_seconds": alns_seconds,
            "graph_seconds": self._last_graph_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        return obs

    def _temperature(self):
        """Linear SA temperature decay over one actor episode."""
        return self.T0 * (1.0 - self.t / self.cfg.search_iterations)

    def _obs(self):
        started = time.perf_counter()
        data = (
            self.builder.build(self.pr, self.sol)
            if self.builder is not None
            else Data(num_nodes=0)
        )
        best_improved, accepted, cur_improved = self.flags
        data.g = global_features(
            self.pr,
            self.sol,
            self.t,
            self.cfg.search_iterations,
            self.stagcount,
            self.f_cur,
            self.f_best,
            best_improved=best_improved,
            current_accepted=accepted,
            current_improved=cur_improved,
        )
        self._last_graph_seconds = time.perf_counter() - started
        return data  # GraphBuilder.build already pads

    def step(self, a):
        started = time.perf_counter()
        cand, f_new, ok = apply_actor_action(
            self.pr,
            self.sol,
            int(a),
            self.q_destroy,
            self.rng,
            self.noise_amplitude,
        )

        T = self._temperature()
        f_best_prev = self.f_best
        improved_best = False
        accepted = False
        improved_current = False
        objective_key = round(f_new, 4) if ok else None
        unseen = ok and objective_key not in self.seen_objectives
        if ok:  # infeasible actor transitions are discarded
            if f_new < self.f_best - 1e-9:
                self.f_best = f_new
                improved_best = True
            if sa_accept(f_new, self.f_cur, T, self.rng):
                accepted = True
                improved_current = f_new < self.f_cur - 1e-9
                self.sol, self.f_cur = cand, f_new
        reward = calculate_transition_reward(
            self.cfg.reward_mode,
            self.cfg.reward_scale,
            self.f_init,
            f_best_prev,
            self.f_best,
            improved_best=improved_best,
            improved_current=improved_current,
            accepted=accepted,
            unseen=unseen,
        )
        if ok:
            self.seen_objectives.add(objective_key)

        self.flags = (improved_best, accepted, improved_current)
        self.stagcount = 0 if improved_best else self.stagcount + 1
        self.t += 1
        done = self.t >= self.cfg.search_iterations
        alns_seconds = time.perf_counter() - started
        obs = self._obs()
        self.last_timing = {
            "phase": "step",
            "load_seconds": 0.0,
            "alns_seconds": alns_seconds,
            "graph_seconds": self._last_graph_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        return (
            obs,
            reward,
            done,
            {
                "f_best": self.f_best,
                "f_cur": self.f_cur,
                "f_init": self.f_init,
                "feasible": ok,
                "accepted": accepted,
                "improved": improved_best,
                "instance_id": self.pr.instance_id,
            },
        )


class VecALNS:
    """Synchronous vector of ALNSEnv with auto-reset on done.

    A done env's returned obs is the reset obs of its next episode, so
    GAE must key off done=True (never bootstrap through it).
    """

    def __init__(self, envs):
        self.envs = list(envs)
        if not self.envs:
            raise ValueError("VecALNS requires at least one environment")
        self.worker_count = len(self.envs)
        self.worker_pids = []
        self.startup_seconds = 0.0
        self.last_timings = {}
        self._closed = False
        self._has_reset = False

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        self._closed = True

    def set_context(self, upd=None, t=None):
        self._context = (upd, t)

    def _require_open(self):
        if self._closed:
            raise RuntimeError("VecALNS is closed")

    @staticmethod
    def _env_timing(env):
        timings = dict(env.last_timing)
        timings["env_seconds"] = timings["total_seconds"]
        timings["encode_seconds"] = 0.0
        cache = getattr(env.provider, "_cache", {})
        timings["cache_size"] = len(cache)
        return timings

    def _record_timings(self, started, workers):
        primary = (
            "step"
            if any("step" in item for item in workers.values())
            else "reset"
        )
        durations = {
            i: item[primary]["env_seconds"]
            for i, item in workers.items()
            if primary in item
        }
        self.last_timings = {
            "env_seconds": time.perf_counter() - started,
            "step_seconds": sum(
                item.get("step", {}).get("env_seconds", 0.0)
                for item in workers.values()
            ),
            "reset_seconds": sum(
                item.get("reset", {}).get("env_seconds", 0.0)
                for item in workers.values()
            ),
            "send_seconds": 0.0,
            "receive_decode_seconds": 0.0,
            "wait_seconds": 0.0,
            "workers": workers,
            "slowest_worker": max(durations, key=durations.get),
            "worker_max_seconds": max(durations.values()),
            "worker_mean_seconds": statistics.fmean(durations.values()),
        }

    def reset(self):
        self._require_open()
        started = time.perf_counter()
        observations, workers = [], {}
        for i, env in enumerate(self.envs):
            observations.append(env.reset())
            workers[i] = {"reset": self._env_timing(env)}
        self._has_reset = True
        self._record_timings(started, workers)
        return observations

    def step(self, actions):
        self._require_open()
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        actions = list(actions)
        if len(actions) != self.worker_count:
            raise ValueError(
                f"expected {self.worker_count} actions, got {len(actions)}"
            )
        # Validate the whole vector before advancing even one environment.
        validated = []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if isinstance(action, bool):
                raise ValueError(f"boolean action for env {i}")
            action = operator.index(action)
            if not 0 <= action < env.cfg.n_actions:
                raise ValueError(f"invalid action for env {i}: {action}")
            validated.append(action)
        started = time.perf_counter()
        workers = {}
        obs, rewards, dones, infos = [], [], [], []
        for i, (env, a) in enumerate(zip(self.envs, validated)):
            o, r, d, info = env.step(a)
            workers[i] = {"step": self._env_timing(env)}
            if d:
                o = env.reset()
                workers[i]["reset"] = self._env_timing(env)
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        self._record_timings(started, workers)
        return obs, rewards, dones, infos


# CPU observation transport with explicit graph-schema preservation.
#
# The default codec leaves HeteroData intact.  The optional NumPy codec never puts
# Torch tensors in its payload, allowing resource comparisons without changing the
# environment or policy.  Both reject unsupported attributes instead of silently
# discarding part of an observation.


OBSERVATION_CODECS = ("direct", "numpy")
_PAYLOAD_VERSION = 1


def _check_codec(codec):
    if codec not in OBSERVATION_CODECS:
        raise ValueError(
            f"unknown observation codec {codec!r}; "
            f"expected one of {OBSERVATION_CODECS}"
        )


def _require_keys(value, keys, label):
    if set(value) != set(keys):
        raise ValueError(
            f"{label} must have exactly these keys: {tuple(keys)!r}"
        )


def _cpu_tensor(value, dtype, shape, label):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a Torch tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{label} must remain on CPU, got {value.device}")
    if value.layout != torch.strided or value.requires_grad:
        raise ValueError(
            f"{label} must be a dense observation without gradients"
        )
    if value.dtype != dtype:
        raise ValueError(f"{label} must have dtype {dtype}, got {value.dtype}")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(
            f"{label} must have shape {shape}, got {tuple(value.shape)}"
        )


def validate_observation(data):
    """Validate one unbatched graph without mutating its store insertion order.

    GraphBuilder creates present edge stores in route order before padding the
    remaining edge types.  Enforcing EDGE_TYPES *order* here would reject valid
    serial observations; the exact key set is what must match the schema.
    """
    if not isinstance(data, HeteroData):
        raise TypeError("GNN observation must be HeteroData")
    _require_keys(data.node_types, NODE_DIMS, "node types")
    _require_keys(data.edge_types, EDGE_TYPES, "edge types")
    stores = data.to_dict()
    _require_keys(
        stores, ["_global_store", *NODE_DIMS, *EDGE_TYPES], "graph stores"
    )
    _require_keys(stores["_global_store"], ("g",), "global store")
    _cpu_tensor(data.g, torch.float32, (1, G_DIM), "g")
    for node_type, dimension in NODE_DIMS.items():
        _require_keys(stores[node_type], ("x",), f"node {node_type!r}")
        _cpu_tensor(
            data[node_type].x,
            torch.float32,
            (None, dimension),
            f"{node_type}.x",
        )
    for edge_type in EDGE_TYPES:
        store = data[edge_type]
        _require_keys(
            stores[edge_type],
            ("edge_index", "edge_attr"),
            f"edge {edge_type!r}",
        )
        _cpu_tensor(
            store.edge_index, torch.int64, (2, None), f"{edge_type}.edge_index"
        )
        _cpu_tensor(
            store.edge_attr,
            torch.float32,
            (store.edge_index.shape[1], EDGE_DIMS[edge_type[1]]),
            f"{edge_type}.edge_attr",
        )
    return data


def encode_observation(data, codec="direct"):
    """Encode a CPU graph; NumPy payloads own snapshots of every tensor's data."""
    _check_codec(codec)
    validate_observation(data)
    if codec == "direct":
        # The current builder allocates fresh tensors at every step/reset and
        # workers must not mutate a returned graph after sending it.
        return data

    def snapshot(tensor):
        return tensor.detach().numpy().copy(order="C")

    return {
        "version": _PAYLOAD_VERSION,
        "node_order": tuple(data.node_types),
        "edge_order": tuple(data.edge_types),
        "nodes": {node: snapshot(data[node].x) for node in NODE_DIMS},
        "edges": {
            edge: (
                snapshot(data[edge].edge_index),
                snapshot(data[edge].edge_attr),
            )
            for edge in EDGE_TYPES
        },
        "g": snapshot(data.g),
    }


def _require_order(order, expected, label):
    if not isinstance(order, (tuple, list)):
        raise TypeError(f"{label} must be a sequence")
    if len(order) != len(expected) or set(order) != set(expected):
        raise ValueError(
            f"{label} must contain each expected store exactly once"
        )


def _numpy_tensor(array, dtype, shape, label):
    if not isinstance(array, np.ndarray) or array.dtype != np.dtype(dtype):
        raise TypeError(f"{label} must be a NumPy array with dtype {dtype}")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    # Own writable storage: neither the caller reusing its payload nor a
    # read-only NumPy array can invalidate or alter an earlier observation.
    return torch.from_numpy(array.copy(order="C"))


def decode_observation(payload, codec="direct"):
    """Restore values and original store order; reject malformed payloads early."""
    _check_codec(codec)
    if codec == "direct":
        return validate_observation(payload)
    if not isinstance(payload, dict):
        raise TypeError("NumPy observation payload must be a dict")
    _require_keys(
        payload,
        ("version", "node_order", "edge_order", "nodes", "edges", "g"),
        "NumPy observation payload",
    )
    if (
        type(payload["version"]) is not int
        or payload["version"] != _PAYLOAD_VERSION
    ):
        raise ValueError("unsupported NumPy observation payload version")
    if not isinstance(payload["nodes"], dict) or not isinstance(
        payload["edges"], dict
    ):
        raise TypeError("NumPy nodes and edges must be dictionaries")
    _require_keys(payload["nodes"], NODE_DIMS, "NumPy nodes")
    _require_keys(payload["edges"], EDGE_TYPES, "NumPy edges")
    _require_order(payload["node_order"], NODE_DIMS, "node_order")
    _require_order(payload["edge_order"], EDGE_TYPES, "edge_order")
    data = HeteroData()
    for node in payload["node_order"]:
        data[node].x = _numpy_tensor(
            payload["nodes"][node],
            np.float32,
            (None, NODE_DIMS[node]),
            f"{node}.x",
        )
    for edge in payload["edge_order"]:
        arrays = payload["edges"][edge]
        if not isinstance(arrays, (tuple, list)) or len(arrays) != 2:
            raise ValueError(
                f"edge {edge!r} must contain index and attribute arrays"
            )
        index, attr = arrays
        data[edge].edge_index = _numpy_tensor(
            index, np.int64, (2, None), f"{edge}.edge_index"
        )
        data[edge].edge_attr = _numpy_tensor(
            attr,
            np.float32,
            (data[edge].edge_index.shape[1], EDGE_DIMS[edge[1]]),
            f"{edge}.edge_attr",
        )
    data.g = _numpy_tensor(payload["g"], np.float32, (1, G_DIM), "g")
    return validate_observation(data)


# Persistent, synchronous ALNS workers with explicit Windows spawn semantics.
#
# Only the parent selects instances and samples policy actions. Each worker owns
# one environment and its RNG for its entire lifetime. Observation transport is
# handled by the codecs below; no model, Params, or Solution crosses the pipe.
#
# Connection.send/recv are blocking APIs. A wait-ready connection is not a promise
# that a whole large payload or its tensor reconstruction has completed. This
# implementation detects process death and reports long requests, but does not
# claim to automatically diagnose a live worker hung inside Python/native code.


def _normal_path(path):
    return os.path.normcase(str(Path(path).resolve()))


def _alns_worker_main(
    worker_id, connection, cache_init, cfg, norms, codec, package
):
    """Spawn-safe entry; heavy imports and all mutable ALNS state stay local."""
    request_id = None
    phase = "startup"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        alns = importlib.import_module(f"{package}.alns")
        ppo = importlib.import_module(f"{package}.ppo")
        transport_package = (
            "gnn_ppo_alns"
            if package == "gnn_ppo_alns" and cfg.use_graph
            else "ppo_alns"
        )
        encode_observation = importlib.import_module(
            f"{transport_package}.ppo"
        ).encode_observation
        builder = None
        if package == "gnn_ppo_alns" and cfg.use_graph:
            builder = importlib.import_module(f"{package}.gnn").GraphBuilder(
                norms, cfg
            )
        cache = alns.InstanceCache(**cache_init)
        env = ppo.ALNSEnv(None, builder, cfg, cfg.seed + worker_id)
        initialized = False
        if torch.cuda.is_initialized():
            raise RuntimeError("ALNS worker unexpectedly initialized CUDA")
        connection.send(
            (
                "ready",
                worker_id,
                os.getpid(),
                {
                    "executable": sys.executable,
                    "python_version": tuple(sys.version_info[:3]),
                    "ppo_module": ppo.__file__,
                    "alns_module": alns.__file__,
                    "torch_version": torch.__version__,
                    "cuda_initialized": torch.cuda.is_initialized(),
                },
            )
        )
        previous_request_id = -1
        while True:
            phase = "receive"
            request_id = None
            try:
                command = connection.recv()
            except EOFError:
                break
            if not isinstance(command, tuple) or len(command) != 3:
                raise RuntimeError("invalid parent command framing")
            phase, request_id, argument = command
            if (
                not isinstance(request_id, int)
                or request_id <= previous_request_id
            ):
                raise RuntimeError("request IDs must strictly increase")
            previous_request_id = request_id
            if phase == "close":
                break
            started = time.perf_counter()
            load_seconds = 0.0
            if phase == "reset":
                load_started = time.perf_counter()
                pr = cache.load_ref(argument)
                load_seconds = time.perf_counter() - load_started
                observation = env.reset(pr)
                initialized = True
                payload = {}
            elif phase == "step":
                if not initialized:
                    raise RuntimeError("STEP received before the first RESET")
                observation, reward, done, info = env.step(argument)
                payload = {
                    "reward": float(reward),
                    "done": bool(done),
                    "info": info,
                }
            else:
                raise RuntimeError(f"unknown command {phase!r}")
            env_seconds = time.perf_counter() - started
            if torch.cuda.is_initialized():
                raise RuntimeError("ALNS worker unexpectedly initialized CUDA")
            encoded_at = time.perf_counter()
            encoded = encode_observation(observation, codec=codec)
            timings = dict(getattr(env, "last_timing", {}))
            timings.update(
                {
                    "load_seconds": load_seconds,
                    "env_seconds": env_seconds,
                    "encode_seconds": time.perf_counter() - encoded_at,
                    "cache_size": len(cache),
                }
            )
            payload.update(observation=encoded, timings=timings)
            connection.send(("ok", request_id, worker_id, payload))
    except KeyboardInterrupt:
        # Never continue from a partially executed ALNS transition.
        pass
    except BaseException:
        try:
            connection.send(
                (
                    "error",
                    request_id,
                    worker_id,
                    phase,
                    traceback.format_exc()[-32768:],
                )
            )
        except (EOFError, OSError, KeyboardInterrupt):
            pass
    finally:
        connection.close()


class ParallelVecALNS:
    """One persistent process per env; auto-reset preserves terminal r/d/info.

    ``last_timings`` measures parent wall time, not the sum of worker times.
    ``workers[id][phase]`` contains worker measurements and current cache size.
    Parent wait and worker execution overlap and must not be added together.
    The class is intentionally single-caller: at most one request per worker is
    outstanding. On any failed operation, all workers are closed, without retry.
    """

    def __init__(
        self,
        cfg,
        provider,
        norms,
        *,
        observation_codec="direct",
        startup_timeout=180.0,
        heartbeat_interval=30.0,
        package="gnn_ppo_alns",
    ):
        if package not in {"gnn_ppo_alns", "ppo_alns"}:
            raise ValueError("unsupported ALNS policy package")
        transport_package = (
            "gnn_ppo_alns"
            if package == "gnn_ppo_alns" and cfg.use_graph
            else "ppo_alns"
        )
        decode_observation = importlib.import_module(
            f"{transport_package}.ppo"
        ).decode_observation

        self.cfg = cfg
        self.package = package
        self.provider = provider
        self.worker_count = operator.index(cfg.n_envs)
        if self.worker_count < 1:
            raise ValueError("cfg.n_envs must be positive")
        if observation_codec not in {"direct", "numpy"}:
            raise ValueError("observation_codec must be 'direct' or 'numpy'")
        for name, value in (
            ("startup_timeout", startup_timeout),
            ("heartbeat_interval", heartbeat_interval),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.observation_codec = observation_codec
        self._decode = decode_observation
        self._heartbeat_interval = float(heartbeat_interval)
        self._closed = False
        self._has_reset = False
        self._next_request_id = 0
        self._context = (None, None)
        self._connections = {}
        self._pending = {}
        self.processes = []
        self.worker_pids = []
        self.worker_runtime = {}
        self.last_timings = {}
        self.startup_seconds = 0.0
        context = multiprocessing.get_context("spawn")
        startup_at = time.perf_counter()
        try:
            cache_init = provider.worker_spec()
            for worker_id in range(self.worker_count):
                parent_connection, child_connection = context.Pipe(duplex=True)
                self._connections[worker_id] = parent_connection
                process = context.Process(
                    target=_alns_worker_main,
                    args=(
                        worker_id,
                        child_connection,
                        cache_init,
                        cfg,
                        norms,
                        observation_codec,
                        package,
                    ),
                    name=f"ALNS-env-{worker_id}",
                    daemon=False,
                )
                self.processes.append(process)
                self._pending[worker_id] = ("startup", None)
                try:
                    process.start()
                    self.worker_pids.append(process.pid)
                finally:
                    # Retaining this duplicate prevents reliable EOF detection.
                    child_connection.close()
            self._collect_ready(startup_at + float(startup_timeout))
            self.startup_seconds = time.perf_counter() - startup_at
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def set_context(self, upd=None, t=None):
        """Attach progress context to long-request heartbeat messages."""
        self._context = (upd, t)

    def _require_open(self):
        if self._closed:
            raise RuntimeError("ParallelVecALNS is closed")

    def _request_id(self):
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _check_workers(self):
        for worker_id, process in enumerate(self.processes):
            if process.exitcode is not None:
                pending = self._pending.get(worker_id)
                # A traceback may arrive just after wait()'s pipe snapshot.
                # Preserve it before falling back to the process-death report.
                if pending is not None and self._connections[worker_id].poll():
                    self._receive(worker_id)
                raise RuntimeError(
                    f"ALNS worker {worker_id} pid={process.pid} died "
                    f"exitcode={process.exitcode}, pending={pending}"
                )

    def _heartbeat(self, pending, phase, started, last_report):
        now = time.perf_counter()
        if now - last_report >= self._heartbeat_interval:
            upd, t = self._context
            print(
                f"[waiting upd={upd} t={t}] phase={phase} "
                f"workers={sorted(pending)} elapsed={now - started:.1f}s",
                flush=True,
            )
            return now
        return last_report

    def _wait_objects(self, pending):
        # Include all sentinels: an already-answered worker can still die while
        # another worker is calculating. Observe that death in this operation.
        return [self._connections[i] for i in pending] + [
            process.sentinel for process in self.processes
        ]

    def _receive(self, worker_id):
        try:
            message = self._connections[worker_id].recv()
        except (EOFError, OSError) as exc:
            process = self.processes[worker_id]
            raise RuntimeError(
                f"ALNS worker {worker_id} pid={process.pid} pipe failed; "
                f"exitcode={process.exitcode}, "
                f"pending={self._pending.get(worker_id)}"
            ) from exc
        if not isinstance(message, tuple) or not message:
            raise RuntimeError(f"invalid reply from ALNS worker {worker_id}")
        if message[0] == "error":
            if len(message) != 5:
                raise RuntimeError(f"malformed ERROR from worker {worker_id}")
            _, request_id, reported_id, phase, detail = message
            if reported_id != worker_id:
                raise RuntimeError("ERROR worker ID does not match its pipe")
            expected = self._pending.get(worker_id)
            # Failures before recv/parsing may legitimately lack a request ID.
            if (
                request_id is not None
                and expected is not None
                and request_id != expected[1]
            ):
                raise RuntimeError(
                    f"worker {worker_id} ERROR request mismatch: "
                    f"expected={expected[1]}, got={request_id}; {detail}"
                )
            raise RuntimeError(
                f"ALNS worker {worker_id} failed in {phase}, "
                f"request_id={request_id}:\n{detail}"
            )
        return message

    def _collect_ready(self, deadline):
        pending = set(range(self.worker_count))
        started = last_report = time.perf_counter()
        package_dir = Path(__file__).resolve().parent.parent / self.package
        expected_ppo = _normal_path(package_dir / "ppo.py")
        expected_alns = _normal_path(package_dir / "alns.py")
        while pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"ALNS startup timed out; pending workers={sorted(pending)}"
                )
            ready = wait(
                self._wait_objects(pending), timeout=min(5.0, remaining)
            )
            # Drain an ERROR reply before reporting its process sentinel.
            for worker_id in sorted(pending):
                if self._connections[worker_id] not in ready:
                    continue
                message = self._receive(worker_id)
                if (
                    len(message) != 4
                    or message[0] != "ready"
                    or message[1] != worker_id
                    or message[2] != self.processes[worker_id].pid
                ):
                    raise RuntimeError(
                        f"invalid READY from worker {worker_id}"
                    )
                runtime = message[3]
                if (
                    _normal_path(runtime["executable"])
                    != _normal_path(sys.executable)
                    or tuple(runtime["python_version"])
                    != tuple(sys.version_info[:3])
                    or _normal_path(runtime["ppo_module"]) != expected_ppo
                    or _normal_path(runtime["alns_module"]) != expected_alns
                    or runtime["cuda_initialized"]
                ):
                    raise RuntimeError(
                        f"worker {worker_id} runtime/source mismatch: {runtime}"
                    )
                self.worker_runtime[worker_id] = runtime
                self._pending.pop(worker_id)
                pending.remove(worker_id)
            self._check_workers()
            last_report = self._heartbeat(
                pending, "startup", started, last_report
            )

    def _dispatch(self, phase, arguments):
        """Send every command first, then collect every reply in env-ID order."""
        self._require_open()
        if self._pending:
            raise RuntimeError(
                "cannot dispatch with outstanding ALNS requests"
            )
        self._check_workers()
        started = time.perf_counter()
        metrics = {
            "send_seconds": 0.0,
            "receive_decode_seconds": 0.0,
            "wait_seconds": 0.0,
            "wall_seconds": 0.0,
        }
        pending = set(arguments)
        for worker_id in sorted(arguments):
            if worker_id not in self._connections:
                raise ValueError(f"unknown ALNS worker {worker_id}")
            request_id = self._request_id()
            self._pending[worker_id] = (phase, request_id)
            send_started = time.perf_counter()
            self._connections[worker_id].send(
                (phase, request_id, arguments[worker_id])
            )
            metrics["send_seconds"] += time.perf_counter() - send_started
        results = {}
        last_report = time.perf_counter()
        while pending:
            wait_started = time.perf_counter()
            ready = wait(self._wait_objects(pending), timeout=5.0)
            metrics["wait_seconds"] += time.perf_counter() - wait_started
            for worker_id in sorted(pending):
                if self._connections[worker_id] not in ready:
                    continue
                receive_started = time.perf_counter()
                message = self._receive(worker_id)
                expected = self._pending[worker_id]
                if (
                    len(message) != 4
                    or message[0] != "ok"
                    or message[1] != expected[1]
                    or message[2] != worker_id
                ):
                    raise RuntimeError(
                        f"invalid {phase} reply from worker {worker_id}: "
                        f"expected request={expected[1]}"
                    )
                payload = message[3]
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"invalid payload from worker {worker_id}"
                    )
                payload["observation"] = self._decode(
                    payload["observation"], codec=self.observation_codec
                )
                results[worker_id] = payload
                self._pending.pop(worker_id)
                pending.remove(worker_id)
                metrics["receive_decode_seconds"] += (
                    time.perf_counter() - receive_started
                )
            self._check_workers()
            last_report = self._heartbeat(pending, phase, started, last_report)
        metrics["wall_seconds"] = time.perf_counter() - started
        return results, metrics

    def _record_timings(
        self, started, steps, resets, step_metrics, reset_metrics
    ):
        workers = {}
        for phase, results in (("step", steps), ("reset", resets)):
            for worker_id, payload in results.items():
                workers.setdefault(worker_id, {})[phase] = payload["timings"]
        primary = steps if steps else resets
        durations = {
            i: payload["timings"]["env_seconds"]
            for i, payload in primary.items()
        }
        slowest = max(durations, key=durations.get) if durations else None
        self.last_timings = {
            "env_seconds": time.perf_counter() - started,
            "step_seconds": step_metrics.get("wall_seconds", 0.0),
            "reset_seconds": reset_metrics.get("wall_seconds", 0.0),
            "workers": workers,
            "slowest_worker": slowest,
            "worker_max_seconds": max(durations.values(), default=0.0),
            "worker_mean_seconds": (
                statistics.mean(durations.values()) if durations else 0.0
            ),
        }
        for name in ("send_seconds", "receive_decode_seconds", "wait_seconds"):
            self.last_timings[name] = step_metrics.get(
                name, 0.0
            ) + reset_metrics.get(name, 0.0)

    def reset(self):
        self._require_open()
        started = time.perf_counter()
        try:
            # Keep the only selection RNG and its consumption order in parent.
            refs = {
                i: self.provider.sample_ref() for i in range(self.worker_count)
            }
            results, metrics = self._dispatch("reset", refs)
            observations = [
                results[i]["observation"] for i in range(self.worker_count)
            ]
            self._has_reset = True
            self._record_timings(started, {}, results, {}, metrics)
            return observations
        except BaseException:
            self.close()
            raise

    def step(self, actions):
        self._require_open()
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        actions = list(actions)
        if len(actions) != self.worker_count:
            raise ValueError(
                f"expected {self.worker_count} actions, got {len(actions)}"
            )
        action_map = {}
        for worker_id, action in enumerate(actions):
            if isinstance(action, bool):
                raise ValueError(f"boolean action for env {worker_id}")
            action = operator.index(action)
            if not 0 <= action < self.cfg.n_actions:
                raise ValueError(
                    f"invalid action for env {worker_id}: {action}"
                )
            action_map[worker_id] = action
        started = time.perf_counter()
        try:
            results, step_metrics = self._dispatch("step", action_map)
            done_ids = [
                i for i in range(self.worker_count) if results[i]["done"]
            ]
            resets, reset_metrics = {}, {}
            if done_ids:
                refs = {i: self.provider.sample_ref() for i in done_ids}
                resets, reset_metrics = self._dispatch("reset", refs)
            observations, rewards, dones, infos = [], [], [], []
            for worker_id in range(self.worker_count):
                payload = results[worker_id]
                observation = (
                    resets[worker_id]["observation"]
                    if worker_id in resets
                    else payload["observation"]
                )
                observations.append(observation)
                rewards.append(payload["reward"])
                dones.append(payload["done"])
                infos.append(payload["info"])
            self._record_timings(
                started, results, resets, step_metrics, reset_metrics
            )
            return observations, rewards, dones, infos
        except BaseException:
            self.close()
            raise

    def close(self, timeout=5.0):
        """Best-effort bounded cleanup, including partially started workers.

        Only idle workers receive CLOSE. For an outstanding request, do not
        enqueue another command behind a potentially blocked large response.
        Such workers are terminated after the shared grace period. A tiny CLOSE
        on an idle dedicated pipe avoids ordinary backpressure, but send() is
        still a blocking OS API and is not a general live-hang guarantee.
        """
        if self._closed:
            return
        self._closed = True
        timeout = max(0.0, float(timeout))
        survivors = []
        for worker_id, process in enumerate(self.processes):
            try:
                if (
                    process.pid is not None
                    and process.is_alive()
                    and worker_id not in self._pending
                ):
                    self._connections[worker_id].send(
                        ("close", self._request_id(), None)
                    )
            except (EOFError, OSError, ValueError, KeyboardInterrupt):
                pass
        deadline = time.perf_counter() + timeout
        for process in self.processes:
            try:
                if process.pid is not None:
                    process.join(max(0.0, deadline - time.perf_counter()))
            except (OSError, ValueError, KeyboardInterrupt):
                pass
        for process in self.processes:
            try:
                if process.pid is not None and process.is_alive():
                    process.terminate()
            except (OSError, ValueError, KeyboardInterrupt):
                pass
        deadline = time.perf_counter() + timeout
        for process in self.processes:
            try:
                if process.pid is not None:
                    process.join(max(0.0, deadline - time.perf_counter()))
                    if process.is_alive():
                        survivors.append(process.pid)
                    else:
                        process.close()
            except (OSError, ValueError, KeyboardInterrupt):
                pass
        for connection in self._connections.values():
            try:
                connection.close()
            except OSError:
                pass
        self._pending.clear()
        if survivors:
            warnings.warn(
                f"ALNS cleanup deadline exceeded; inspect worker PIDs "
                f"{survivors}",
                RuntimeWarning,
                stacklevel=2,
            )


# PPO training loop: collect transitions, estimate advantages, update the
# policy, and save checkpoints. Episode endings and rollout boundaries stay
# independent, so a rollout can bootstrap from an unfinished episode.
REWARD_WINDOW = 100  # rolling stats window (episodes)


def make_envs(
    cfg,
    provider,
    builder,
    *,
    env_backend="serial",
    observation_codec="direct",
    norms=None,
):
    # Backend selection changes execution placement only; each worker keeps
    # the same seed, transition logic, and episode length as a serial env.
    if env_backend == "serial":
        return VecALNS(
            [
                ALNSEnv(provider, builder, cfg, cfg.seed + i)
                for i in range(cfg.n_envs)
            ]
        )
    if env_backend == "process":
        if norms is None and builder is not None:
            norms = builder.norms
        return ParallelVecALNS(
            cfg, provider, norms, observation_codec=observation_codec
        )
    raise ValueError("env_backend must be 'serial' or 'process'")


@contextmanager
def atomic_open(path, mode="w", *, encoding="utf-8", newline=None):
    """Write beside the destination, then replace it only on successful close.

    Intended replacement is allowed (periodic checkpoints within a reserved run).
    An exception leaves an existing destination untouched and removes only this
    call's temporary file. This is per-file atomicity, not a multi-file transaction.
    """
    if mode not in {"w", "wb"}:
        raise ValueError("atomic_open only supports w or wb")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary)
    try:
        options = (
            {} if "b" in mode else {"encoding": encoding, "newline": newline}
        )
        with os.fdopen(fd, mode, **options) as handle:
            fd = None
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def save_checkpoint(model, cfg, norms, metadata, path):
    with atomic_open(path, mode="wb") as handle:
        torch.save(
            {
                "model": model.state_dict(),
                "config": cfg.to_dict(),
                "norms": norms,
                "metadata": dict(metadata),
            },
            handle,
        )


def _resource_snapshot(worker_pids):
    """Best-effort runtime diagnostics; these values never affect training."""
    try:
        import psutil
    except ImportError:
        return {"available": False}
    records = []
    for pid in [os.getpid(), *worker_pids]:
        try:
            process = psutil.Process(pid)
            memory = process.memory_info()
            record = {
                "pid": pid,
                "rss_bytes": memory.rss,
                "vms_bytes": memory.vms,
                "threads": process.num_threads(),
            }
            if hasattr(process, "num_handles"):
                record["handles"] = process.num_handles()
            records.append(record)
        except psutil.Error:
            records.append({"pid": pid, "unavailable": True})
    return {"available": True, "processes": records}


def train(
    cfg,
    provider,
    out_path,
    norms,
    log_rows=None,
    update_rows=None,
    *,
    env_backend="serial",
    observation_codec="direct",
    training_stats=None,
):
    """Train and save a checkpoint to out_path. Returns the model.

    log_rows: per-episode reward rows (CSV/plot format);
    update_rows: per-update PPO metrics, phase timings and resource snapshots.
    Backend/codec/diagnostics are runtime choices, not PPOConfig fields.
    """
    if cfg.n_envs != 10:
        raise ValueError("this training protocol requires cfg.n_envs == 10")
    if cfg.n_updates < 1:
        raise ValueError(
            "total_steps must cover at least one complete rollout"
        )
    train_started = time.perf_counter()
    stats = training_stats if training_stats is not None else {}
    stats.update(
        env_backend=env_backend,
        observation_codec=observation_codec,
        worker_count=cfg.n_envs if env_backend == "process" else 0,
        requested_steps=cfg.total_steps,
        effective_steps=cfg.n_updates * cfg.t_rollout * cfg.n_envs,
        n_updates=cfg.n_updates,
        completed_updates=0,
        completed_steps=0,
    )
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = ActorCritic(cfg).to(device)
    opt = make_optimizer(model, cfg)
    builder = GraphBuilder(norms, cfg) if cfg.use_graph else None
    gen = torch.Generator().manual_seed(cfg.seed)  # minibatch shuffle
    stats["model_optimizer_init_seconds"] = time.perf_counter() - train_started
    print(
        f"[train] device={cfg.device} backend={env_backend} "
        f"n_envs={cfg.n_envs} n_updates={cfg.n_updates} "
        f"requested_steps={cfg.total_steps} "
        f"effective_steps={stats['effective_steps']}",
        flush=True,
    )
    ep_reward = [0.0] * cfg.n_envs
    recent_rewards = deque(maxlen=REWARD_WINDOW)
    episodes_done = 0
    step = 0  # cumulative env steps over all workers
    metadata = provider.checkpoint_metadata
    vec = make_envs(
        cfg,
        provider,
        builder,
        env_backend=env_backend,
        observation_codec=observation_codec,
        norms=norms,
    )
    try:
        stats["worker_startup_seconds"] = vec.startup_seconds
        stats["worker_pids"] = list(vec.worker_pids)
        if hasattr(vec, "worker_runtime"):
            stats["worker_runtime"] = dict(vec.worker_runtime)
        print(
            "[reset] building initial observations for all environments",
            flush=True,
        )
        reset_started = time.perf_counter()
        obs = vec.reset()
        stats["initial_reset_seconds"] = time.perf_counter() - reset_started
        stats["initial_reset_timings"] = vec.last_timings
        stats["initial_resources"] = _resource_snapshot(vec.worker_pids)
        print(
            f"[reset] completed in {stats['initial_reset_seconds']:.1f}s",
            flush=True,
        )
        # Preserve the original episode-log time origin; startup/reset now have
        # separate measurements and train_wall_seconds covers the whole call.
        t0 = time.time()
        for upd in range(cfg.n_updates):
            update_started = time.perf_counter()
            buf = RolloutBuffer(cfg.t_rollout, cfg.n_envs)
            rollout_started = time.perf_counter()
            phase_totals = {
                key: 0.0
                for key in (
                    "policy_seconds",
                    "rollout_collation_seconds",
                    "env_seconds",
                    "step_seconds",
                    "reset_seconds",
                    "send_seconds",
                    "receive_decode_seconds",
                    "wait_seconds",
                    "buffer_seconds",
                    "worker_alns_sum_seconds",
                    "worker_graph_sum_seconds",
                    "worker_encode_sum_seconds",
                    "worker_load_sum_seconds",
                )
            }
            action_hist = [0] * cfg.n_actions
            worker_cache_sizes = {}
            max_worker_ratio = 0.0
            for t in range(cfg.t_rollout):
                policy_started = time.perf_counter()
                collate_started = time.perf_counter()
                batch = Batch.from_data_list(obs)
                phase_totals["rollout_collation_seconds"] += (
                    time.perf_counter() - collate_started
                )
                with torch.no_grad():
                    a, logp, _, v = model.get_action_and_value(
                        batch.to(device)
                    )
                del batch
                actions = a.tolist()  # synchronizes sampled CUDA actions
                phase_totals["policy_seconds"] += (
                    time.perf_counter() - policy_started
                )
                for action in actions:
                    action_hist[action] += 1
                vec.set_context(upd=upd, t=t + 1)
                nxt, rewards, dones, infos = vec.step(actions)
                timings = vec.last_timings
                for key in (
                    "env_seconds",
                    "step_seconds",
                    "reset_seconds",
                    "send_seconds",
                    "receive_decode_seconds",
                    "wait_seconds",
                ):
                    phase_totals[key] += timings.get(key, 0.0)
                for worker_id, phases in timings.get("workers", {}).items():
                    for worker_timing in phases.values():
                        for name in ("alns", "graph", "encode", "load"):
                            phase_totals[f"worker_{name}_sum_seconds"] += (
                                worker_timing.get(f"{name}_seconds", 0.0)
                            )
                        worker_cache_sizes[worker_id] = worker_timing.get(
                            "cache_size", 0
                        )
                mean_worker = timings.get("worker_mean_seconds", 0.0)
                if mean_worker:
                    max_worker_ratio = max(
                        max_worker_ratio,
                        timings.get("worker_max_seconds", 0.0) / mean_worker,
                    )
                buffer_started = time.perf_counter()
                buf.add(obs, a.cpu(), logp.cpu(), v.cpu(), rewards, dones)
                phase_totals["buffer_seconds"] += (
                    time.perf_counter() - buffer_started
                )
                obs = nxt
                step += cfg.n_envs
                for i, (r, d, info) in enumerate(zip(rewards, dones, infos)):
                    ep_reward[i] += r
                    if not d:
                        continue
                    recent_rewards.append(ep_reward[i])
                    roll_mean = statistics.fmean(recent_rewards)
                    roll_std = (
                        statistics.pstdev(recent_rewards)
                        if len(recent_rewards) > 1
                        else 0.0
                    )
                    if log_rows is not None:
                        log_rows.append(
                            {
                                "episode": episodes_done,
                                "step": step,
                                "instance_id": info["instance_id"],
                                "reward_mode": cfg.reward_mode,
                                "episode_reward": round(ep_reward[i], 6),
                                "reward_roll_mean": round(roll_mean, 6),
                                "reward_roll_std": round(roll_std, 6),
                                "best_cost": round(info["f_best"], 4),
                                "init_cost": round(info["f_init"], 4),
                                "elapsed_s": round(time.time() - t0, 1),
                            }
                        )
                    episodes_done += 1
                    ep_reward[i] = 0.0
                if (t + 1) % 32 == 0 or t + 1 == cfg.t_rollout:
                    elapsed = time.perf_counter() - rollout_started
                    rate = (t + 1) * cfg.n_envs / max(elapsed, 1e-9)
                    print(
                        f"[rollout upd={upd} t={t + 1}/{cfg.t_rollout}] "
                        f"{rate:.2f} env-step/s "
                        f"slowest=w{timings.get('slowest_worker')} "
                        f"wait={timings.get('wait_seconds', 0.0):.3f}s",
                        flush=True,
                    )
            phase_totals["rollout_seconds"] = (
                time.perf_counter() - rollout_started
            )
            bootstrap_started = time.perf_counter()
            with torch.no_grad():
                last_v = model.get_value(
                    Batch.from_data_list(obs).to(device)
                ).cpu()
            buf.compute_returns(last_v, cfg.gamma, cfg.gae_lambda)
            phase_totals["bootstrap_gae_seconds"] = (
                time.perf_counter() - bootstrap_started
            )

            progress = upd / max(1, cfg.n_updates - 1)
            metrics = ppo_update(model, opt, buf, cfg, progress, gen)
            metrics.update(phase_totals)
            metrics.update(
                action_hist=action_hist,
                worker_cache_sizes=worker_cache_sizes,
                worker_max_mean_ratio=max_worker_ratio,
                update_wall_seconds=time.perf_counter() - update_started,
                resources=_resource_snapshot(vec.worker_pids),
            )
            if update_rows is not None:
                update_rows.append(
                    {
                        "update": upd,
                        "step": step,
                        "reward_mode": cfg.reward_mode,
                        **metrics,
                    }
                )
            stats.update(
                completed_updates=upd + 1,
                completed_steps=step,
                episodes_done=episodes_done,
                last_update=metrics,
            )
            roll = statistics.fmean(recent_rewards) if recent_rewards else 0.0
            print(
                f"[upd {upd}] step={step} ent={metrics['entropy']:.3f} "
                f"kl={metrics['approx_kl']:.5f} "
                f"pg={metrics['pg_loss']:.5f} "
                f"v={metrics['v_loss']:.5f} "
                f"ev={metrics['explained_variance']:.3f} "
                f"eps_done={episodes_done} rollR={roll:.3f} "
                f"optimizer_steps={metrics['optimizer_steps_observed']} "
                f"{round(time.time() - t0, 1)}s",
                flush=True,
            )

            if (upd + 1) % 10 == 0 or upd == cfg.n_updates - 1:
                save_checkpoint(model, cfg, norms, metadata, out_path)
        save_checkpoint(model, cfg, norms, metadata, out_path)
    finally:
        vec.close()
        stats["train_wall_seconds"] = time.perf_counter() - train_started
    return model


# Testing/evaluation for the PPO operator selector (spec: eval).
#
# Runs the SAME ALNSEnv used during training — the geometric-cooling
# acceptance criterion and the max_iter horizon are identical by
# construction (both come from the checkpoint config).


class FixedInstanceProvider:
    """provider.sample() interface over a single Params (for eval)."""

    def __init__(self, pr):
        self.pr = pr

    def sample(self):
        return self.pr


def load_model(path, expected_metadata, device="cpu"):
    """Load a train.py checkpoint -> (model in eval mode, cfg, norms)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    stored_metadata = ckpt.get("metadata")
    if not isinstance(stored_metadata, dict):
        raise ValueError(
            "checkpoint has no experiment metadata; retrain it with the "
            "current processed-data pipeline"
        )
    expected_metadata = dict(expected_metadata)
    if stored_metadata != expected_metadata:
        differing = {
            key: {
                "checkpoint": stored_metadata.get(key),
                "current": expected_metadata.get(key),
            }
            for key in sorted(set(stored_metadata) | set(expected_metadata))
            if stored_metadata.get(key) != expected_metadata.get(key)
        }
        raise ValueError(
            f"checkpoint metadata does not match this experiment: {differing}"
        )
    stored_config = ckpt.get("config")
    if not isinstance(stored_config, dict):
        raise ValueError("checkpoint has no valid PPO config")
    if "eps_uniform" not in stored_config:
        raise ValueError(
            "checkpoint has no eps_uniform training setting; retrain it so "
            "training and testing use the same sampling policy"
        )
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in stored_config.items() if k in known})
    cfg.device = device
    model = ActorCritic(cfg).to(device)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        raise RuntimeError(
            "checkpoint incompatible with the current model "
            "(g_t changed to 9 dims — retraining required)"
        ) from e
    model.eval()
    return model, cfg, ckpt["norms"]


def evaluate_instance(model, cfg, builder, pr, seed=0, sample=True):
    """One test episode on pr; returns (best_solution, stats).

    sample=False: greedy argmax (deterministic policy).
    sample=True: draw actions from the same epsilon-mixed stochastic
    behavior policy used for training (seeded torch generator, hence
    reproducible). ``cfg.eps_uniform`` comes from the checkpoint, so
    training and testing cannot silently use different epsilon values.
    builder may be None for the g_t-only PPO variant.

    Statistics retain the same objective/improvement fields as Vanilla ALNS
    so results remain directly comparable.
    """
    device = next(model.parameters()).device
    # Timed like Vanilla ALNS: the initial-solution construction inside
    # env.reset() counts toward runtime_s, keeping the two comparable.
    t0 = time.time()
    env = ALNSEnv(FixedInstanceProvider(pr), builder, cfg, seed)
    obs = env.reset()
    best_sol, best_cost = env.sol.clone(), env.f_best
    action_hist = [0] * cfg.n_actions
    gen = torch.Generator().manual_seed(seed) if sample else None
    best_trace = [(0, round(time.time() - t0, 3), round(best_cost, 6))]
    episode_reward = 0.0

    for _ in range(cfg.search_iterations):
        with torch.no_grad():
            batch = Batch.from_data_list([obs]).to(device)
            probs = model.action_probs(batch).cpu()
        if sample:
            a = int(torch.multinomial(probs[0], 1, generator=gen).item())
        else:
            a = int(probs.argmax(dim=1).item())
        action_hist[a] += 1
        obs, reward, done, info = env.step(a)
        episode_reward += reward
        if info["f_best"] < best_cost - 1e-9:
            # a new best is always accepted, so env.sol holds it now
            best_sol, best_cost = env.sol.clone(), info["f_best"]
            best_trace.append(
                (env.t, round(time.time() - t0, 3), round(best_cost, 6))
            )
        if done:
            break

    stats = {
        "init_cost": env.f_init,
        "best_cost": best_cost,
        "improve_pct": 100.0 * (env.f_init - best_cost) / env.f_init,
        "iters_done": env.t,
        "runtime_s": time.time() - t0,
        "selection_mode": "sampling_epsilon" if sample else "argmax",
        "eps_uniform": cfg.eps_uniform if sample else 0.0,
        "reward_mode": cfg.reward_mode,
        "episode_reward": episode_reward,
        "action_labels": list(ACTION_LABELS),
        "action_hist": action_hist,
        "best_trace": best_trace,
    }
    return best_sol, stats
