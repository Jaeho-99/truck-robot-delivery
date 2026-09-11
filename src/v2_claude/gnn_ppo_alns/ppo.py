"""GNN-PPO-ALNS policy, environment, update, training, and testing.

The environment protocol follows DR-ALNS. Stability defaults include an
entropy floor, KL early stopping, and magnitude-reward scaling.
"""

from dataclasses import asdict, dataclass
import operator
import time

from .alns import ACTION_COUNT, DOD, W_START


REWARD_MODES = ("alns_5310", "new_best_5", "magnitude")


def reward_artifact_token(reward_mode):
    """Return the validated semantic token used in artifact filenames."""
    if reward_mode not in REWARD_MODES:
        raise ValueError(
            f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
    return f"reward_{reward_mode}"


def calculate_transition_reward(reward_mode, reward_scale, initial_objective,
                                previous_best, current_best, *,
                                improved_best, improved_current, accepted,
                                unseen):
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
        return (reward_scale * max(0.0, previous_best - current_best)
                / initial_objective)
    raise ValueError(f"unsupported reward mode: {reward_mode!r}")


@dataclass
class PPOConfig:
    # graph encoder
    hidden_dim: int = 64
    n_layers: int = 3
    heads: int = 2
    knn_k: int = 5
    use_proximity: bool = True
    use_graph: bool = True          # False -> g_t-only "PPO-ALNS"
    #                                 (no GNN encoder, state = g_t)
    # model
    n_actions: int = ACTION_COUNT   # actor-selected joint operator pairs
    # PPO update stability settings (shared by both model variants)
    lr: float = 3e-4                # learning_rate
    adam_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2           # clip_range
    c1: float = 0.5                 # vf_coef
    c2_start: float = 0.01          # constant entropy coefficient;
    c2_end: float = 0.01            # prevents PPO-ALNS collapse
    k_epochs: int = 10              # n_epochs
    n_minibatch: int = 40           # rollout 2560 / batch_size 64
    target_kl: float | None = 0.02  # stop epochs above 1.5 * target
    max_grad_norm: float = 0.5
    # environment / rollout (DR-ALNS protocol)
    n_envs: int = 10                # n_workers
    t_rollout: int = 256            # n_steps per worker
    search_iterations: int = 100    # episode length, size-independent
    # Behavior policy used consistently by rollout, PPO update, and test:
    # pi_eps = (1-eps) * pi_actor + eps / n_actions.
    eps_uniform: float = 0.1
    # ALNS search regime (identical to vanilla ALNS by default)
    w_start: float = W_START        # SA start-temperature fraction
    dod: float = DOD                # degree of destruction
    # training
    total_steps: int = 300_000      # summed over workers
    train_count: int = 200          # train files 0..199; rest held out
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
"""ActorCritic model for PPO (spec: model section).

Reuses SolutionEncoder unchanged (GATv2 + HeteroConv, joint mean/max
pooling); state = encoder(data) || g_t -> d_state = 137 with defaults.
g_t is attached to the batched HeteroData as ``data.g``. Heads carry no
Dropout/BatchNorm so
rollout and update see identical outputs.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .gnn import SolutionEncoder
from .gnn import G_DIM


def _ortho(layer, std):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.n_actions != ACTION_COUNT:
            raise ValueError(
                f"GNN-PPO actor requires {ACTION_COUNT} ALNS actions, "
                f"got {cfg.n_actions}")
        if not 0.0 <= cfg.eps_uniform <= 1.0:
            raise ValueError("eps_uniform must be between 0 and 1")
        self.eps_uniform = float(cfg.eps_uniform)
        self.n_actions = int(cfg.n_actions)
        # cfg.use_graph=False -> g_t-only "PPO-ALNS" ablation (no
        # encoder; state is just the 9-dim search-state vector)
        self.encoder = (SolutionEncoder(cfg) if cfg.use_graph
                        else None)
        d_state = (2 * cfg.hidden_dim + G_DIM if cfg.use_graph
                   else G_DIM)
        self.actor = nn.Sequential(
            _ortho(nn.Linear(d_state, 64), 2 ** 0.5), nn.Tanh(),
            _ortho(nn.Linear(64, cfg.n_actions), 0.01))
        self.critic = nn.Sequential(
            _ortho(nn.Linear(d_state, 64), 2 ** 0.5), nn.Tanh(),
            _ortho(nn.Linear(64, 1), 1.0))

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
            probs = ((1.0 - self.eps_uniform) * probs
                     + self.eps_uniform / self.n_actions)
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
        return (action, dist.log_prob(action), dist.entropy(),
                self.critic(s).squeeze(-1))

    def get_value(self, data):
        """Critic value (B,) only — bootstrap at rollout boundaries."""
        return self.critic(self._state(data)).squeeze(-1)
"""On-policy rollout storage for PPO (spec: buffer).

Holds exactly one rollout (t_rollout x n_envs) and is reset after
every update — never a replay buffer (PPO is on-policy). Graphs are
stored on CPU with g_t attached as ``data.g``; minibatches are assembled
with Batch.from_data_list.
"""

import torch
from torch_geometric.data import Batch



class RolloutBuffer:
    def __init__(self, t_rollout, n_envs):
        self.t_rollout = t_rollout
        self.n_envs = n_envs
        self.reset()

    def reset(self):
        self.obs = []           # flat, t-major: index = t * n_envs + i
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = None
        self.returns = None
        self.collate_seconds = 0.0

    def __len__(self):
        return len(self.actions)        # steps stored (of t_rollout)

    def add(self, obs, actions, log_probs, values, rewards, dones):
        """Store one vector step: obs is a list of n_envs HeteroData
        (with .g); the rest are tensors/sequences of shape (n_envs,).
        """
        assert len(self.actions) < self.t_rollout, "buffer full"
        self.obs.extend(obs)
        self.actions.append(
            torch.as_tensor(actions, dtype=torch.long))
        self.log_probs.append(
            torch.as_tensor(log_probs, dtype=torch.float32).detach())
        self.values.append(
            torch.as_tensor(values, dtype=torch.float32).detach())
        self.rewards.append(
            torch.as_tensor(rewards, dtype=torch.float32))
        self.dones.append(
            torch.as_tensor(dones, dtype=torch.float32))

    def compute_returns(self, last_value, gamma, gae_lambda):
        """GAE from the stored rollout-time values (spec)."""
        self.advantages, self.returns = compute_gae(
            torch.stack(self.rewards), torch.stack(self.values),
            torch.stack(self.dones),
            torch.as_tensor(last_value, dtype=torch.float32).detach(),
            gamma, gae_lambda)

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
            yield (batch,
                   actions[chunk], log_probs[chunk], advantages[chunk],
                   returns[chunk], values[chunk])
"""Generalized Advantage Estimation (spec: GAE section).

Computed from the values stored at rollout time — update.py must never
recompute them. done_t marks a true episode end (t == max_iter); a
rollout-boundary truncation keeps done=0 so the last state bootstraps
through last_value.
"""

import torch


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
"""Clipped PPO update step (spec: PPO update section).

One rollout -> up to k_epochs of minibatch updates through a single
Adam and a single backward per minibatch (no optimizer split). Stored
log_probs/values come from rollout time; only new_logp/new_v are
recomputed here.
"""

import torch
import torch.nn as nn


def make_optimizer(model, cfg):
    """Single Adam over ALL parameters (spec: eps=1e-5, no split)."""
    return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                            eps=cfg.adam_eps)


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
        for batch, actions, old_logp, adv, ret, old_v in \
                buffer.minibatches(cfg.n_minibatch, generator):
            batch = batch.to(device)
            actions, old_logp, adv, ret, old_v = (
                x.to(device)
                for x in (actions, old_logp, adv, ret, old_v))
            _, new_logp, entropy, new_v = model.get_action_and_value(
                batch, action=actions)
            log_ratio = new_logp - old_logp
            ratio = log_ratio.exp()

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
            v_loss = value_loss(new_v, old_v, ret, eps)
            ent = entropy.mean()
            loss = pg_loss + cfg.c1 * v_loss - c2 * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                                     cfg.max_grad_norm)
            optimizer.step()
            optimizer_steps_observed += 1

            with torch.no_grad():
                kls.append(((ratio - 1) - log_ratio).mean().item())
                clipfracs.append(
                    ((ratio - 1).abs() > eps).float().mean().item())
            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(ent.item())
        epochs_run += 1
        approx_kl = sum(kls) / len(kls)
        # SB3/DR-ALNS default target_kl=None disables the early stop
        if (cfg.target_kl is not None
                and approx_kl > 1.5 * cfg.target_kl):
            break

    # explained variance of the rollout-time value estimates
    y = buffer.returns.view(-1)
    v = torch.stack(buffer.values).view(-1)
    var_y = y.var()
    ev = (float("nan") if var_y == 0
          else (1.0 - (y - v).var() / var_y).item())

    def mean(xs):
        return sum(xs) / len(xs)

    update_seconds = time.perf_counter() - update_started
    return {"pg_loss": mean(pg_losses), "v_loss": mean(v_losses),
            "entropy": mean(entropies), "approx_kl": approx_kl,
            "clipfrac": mean(clipfracs), "explained_variance": ev,
            "c2": c2, "epochs_run": epochs_run,
            "optimizer_steps_observed": optimizer_steps_observed,
            "ppo_update_seconds": update_seconds,
            "minibatch_collation_seconds": buffer.collate_seconds - collate_before,
            "update_time_per_mb": (update_seconds / optimizer_steps_observed
                                   if optimizer_steps_observed else None)}
"""ALNS as a step-interface environment for PPO rollouts.

Episodes have cfg.search_iterations graph-conditioned actor transitions,
linear SA temperature decay from T0 = W_START * f_init / ln 2 to 0, and
fixed degree of destruction q = round(DOD * n). g_t and graph construction
are local to this method. The configured reward mode is stored in the
checkpoint and reused at test.
"""

import math
import random

from torch_geometric.data import Data

from .gnn import global_features
from .alns import (ACTION_LABELS, NOISE_FRAC, apply_actor_action,
                   congestion_aware_initial, eval_solution)


def sa_accept(f_new, f_cur, T, rng):
    """Accept improving candidates; otherwise use simulated annealing."""
    return (f_new < f_cur - 1e-9
            or rng.random() < math.exp(-(f_new - f_cur)
                                       / max(T, 1e-9)))


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
                raise ValueError("reset() requires a provider or explicit Params")
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
        self.q_destroy = max(1, round(getattr(self.cfg, "dod", DOD)
                                      * nC))
        self.noise_amplitude = NOISE_FRAC * self.f_init
        self.T0 = (getattr(self.cfg, "w_start", W_START)
                   * self.f_init) / math.log(2)
        alns_seconds = time.perf_counter() - alns_started
        obs = self._obs()
        self.last_timing = {
            "phase": "reset", "load_seconds": load_seconds,
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
        data = (self.builder.build(self.pr, self.sol)
                if self.builder is not None else Data(num_nodes=0))
        best_improved, accepted, cur_improved = self.flags
        data.g = global_features(
            self.pr, self.sol, self.t, self.cfg.search_iterations,
            self.stagcount, self.f_cur, self.f_best,
            best_improved=best_improved, current_accepted=accepted,
            current_improved=cur_improved)
        self._last_graph_seconds = time.perf_counter() - started
        return data          # GraphBuilder.build already pads

    def step(self, a):
        started = time.perf_counter()
        cand, f_new, ok = apply_actor_action(
            self.pr, self.sol, int(a), self.q_destroy, self.rng,
            self.noise_amplitude)

        T = self._temperature()
        f_best_prev = self.f_best
        improved_best = False
        accepted = False
        improved_current = False
        objective_key = round(f_new, 4) if ok else None
        unseen = ok and objective_key not in self.seen_objectives
        if ok:      # infeasible actor transitions are discarded
            if f_new < self.f_best - 1e-9:
                self.f_best = f_new
                improved_best = True
            if sa_accept(f_new, self.f_cur, T, self.rng):
                accepted = True
                improved_current = f_new < self.f_cur - 1e-9
                self.sol, self.f_cur = cand, f_new
        reward = calculate_transition_reward(
            self.cfg.reward_mode, self.cfg.reward_scale, self.f_init,
            f_best_prev, self.f_best, improved_best=improved_best,
            improved_current=improved_current, accepted=accepted,
            unseen=unseen)
        if ok:
            self.seen_objectives.add(objective_key)

        self.flags = (improved_best, accepted, improved_current)
        self.stagcount = 0 if improved_best else self.stagcount + 1
        self.t += 1
        done = self.t >= self.cfg.search_iterations
        alns_seconds = time.perf_counter() - started
        obs = self._obs()
        self.last_timing = {
            "phase": "step", "load_seconds": 0.0,
            "alns_seconds": alns_seconds,
            "graph_seconds": self._last_graph_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        return (obs, reward, done,
                {"f_best": self.f_best, "f_cur": self.f_cur,
                 "f_init": self.f_init, "feasible": ok,
                 "accepted": accepted, "improved": improved_best,
                 "instance_id": self.pr.instance_id})


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
        primary = "step" if any("step" in item for item in workers.values()) else "reset"
        durations = {i: item[primary]["env_seconds"]
                     for i, item in workers.items() if primary in item}
        self.last_timings = {
            "env_seconds": time.perf_counter() - started,
            "step_seconds": sum(item.get("step", {}).get("env_seconds", 0.0)
                                for item in workers.values()),
            "reset_seconds": sum(item.get("reset", {}).get("env_seconds", 0.0)
                                 for item in workers.values()),
            "send_seconds": 0.0, "receive_decode_seconds": 0.0,
            "wait_seconds": 0.0, "workers": workers,
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
            raise ValueError(f"expected {self.worker_count} actions, got {len(actions)}")
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
"""PPO training loop (DR-ALNS protocol).

cfg.n_envs synchronous ALNS environments (VecALNS); every
cfg.t_rollout vector steps one PPO update, for
cfg.n_updates = total_steps / (t_rollout * n_envs) updates.
Environments run episodes of cfg.search_iterations and keep running
across rollout boundaries — boundary truncation bootstraps through
get_value while done=True happens only at episode end. Episode
rewards are logged with rolling mean/std (window 100).
"""

import statistics
import time
from collections import deque

import torch
from torch_geometric.data import Batch

from .gnn import GraphBuilder

REWARD_WINDOW = 100     # rolling stats window (episodes)


def make_envs(cfg, provider, builder, *, env_backend="serial",
              observation_codec="direct", norms=None):
    if env_backend == "serial":
        return VecALNS([ALNSEnv(provider, builder, cfg, cfg.seed + i)
                        for i in range(cfg.n_envs)])
    if env_backend == "process":
        from .parallel_env import ParallelVecALNS
        if norms is None and builder is not None:
            norms = builder.norms
        return ParallelVecALNS(cfg, provider, norms,
                               observation_codec=observation_codec)
    raise ValueError("env_backend must be 'serial' or 'process'")


def save_checkpoint(model, cfg, norms, metadata, path):
    from v2_claude.common.artifacts import atomic_open
    with atomic_open(path, mode="wb") as handle:
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                    "norms": norms, "metadata": dict(metadata)}, handle)


def _resource_snapshot(worker_pids):
    """Best-effort runtime diagnostics; these values never affect training."""
    import os
    try:
        import psutil
    except ImportError:
        return {"available": False}
    records = []
    for pid in [os.getpid(), *worker_pids]:
        try:
            process = psutil.Process(pid)
            memory = process.memory_info()
            record = {"pid": pid, "rss_bytes": memory.rss,
                      "vms_bytes": memory.vms,
                      "threads": process.num_threads()}
            if hasattr(process, "num_handles"):
                record["handles"] = process.num_handles()
            records.append(record)
        except psutil.Error:
            records.append({"pid": pid, "unavailable": True})
    return {"available": True, "processes": records}


def train(cfg, provider, out_path, norms, log_rows=None,
          update_rows=None, *, env_backend="serial",
          observation_codec="direct", training_stats=None):
    """Train and save a checkpoint to out_path. Returns the model.

    log_rows: per-episode reward rows (CSV/plot format);
    update_rows: per-update PPO metrics, phase timings and resource snapshots.
    Backend/codec/diagnostics are runtime choices, not PPOConfig fields.
    """
    if cfg.n_envs != 10:
        raise ValueError("this training protocol requires cfg.n_envs == 10")
    if cfg.n_updates < 1:
        raise ValueError("total_steps must cover at least one complete rollout")
    train_started = time.perf_counter()
    stats = training_stats if training_stats is not None else {}
    stats.update(env_backend=env_backend, observation_codec=observation_codec,
                 worker_count=cfg.n_envs if env_backend == "process" else 0,
                 requested_steps=cfg.total_steps,
                 effective_steps=cfg.n_updates * cfg.t_rollout * cfg.n_envs,
                 n_updates=cfg.n_updates, completed_updates=0,
                 completed_steps=0)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = ActorCritic(cfg).to(device)
    opt = make_optimizer(model, cfg)
    builder = GraphBuilder(norms, cfg) if cfg.use_graph else None
    gen = torch.Generator().manual_seed(cfg.seed)   # minibatch shuffle
    stats["model_optimizer_init_seconds"] = time.perf_counter() - train_started
    print(f"[train] device={cfg.device} backend={env_backend} "
          f"n_envs={cfg.n_envs} n_updates={cfg.n_updates} "
          f"requested_steps={cfg.total_steps} "
          f"effective_steps={stats['effective_steps']}", flush=True)
    ep_reward = [0.0] * cfg.n_envs
    recent_rewards = deque(maxlen=REWARD_WINDOW)
    episodes_done = 0
    step = 0                    # cumulative env steps over all workers
    metadata = provider.checkpoint_metadata
    vec = make_envs(cfg, provider, builder, env_backend=env_backend,
                    observation_codec=observation_codec, norms=norms)
    try:
        stats["worker_startup_seconds"] = vec.startup_seconds
        stats["worker_pids"] = list(vec.worker_pids)
        if hasattr(vec, "worker_runtime"):
            stats["worker_runtime"] = dict(vec.worker_runtime)
        print("[reset] building initial observations for all environments", flush=True)
        reset_started = time.perf_counter()
        obs = vec.reset()
        stats["initial_reset_seconds"] = time.perf_counter() - reset_started
        stats["initial_reset_timings"] = vec.last_timings
        stats["initial_resources"] = _resource_snapshot(vec.worker_pids)
        print(f"[reset] completed in {stats['initial_reset_seconds']:.1f}s",
              flush=True)
        # Preserve the original episode-log time origin; startup/reset now have
        # separate measurements and train_wall_seconds covers the whole call.
        t0 = time.time()
        for upd in range(cfg.n_updates):
            update_started = time.perf_counter()
            buf = RolloutBuffer(cfg.t_rollout, cfg.n_envs)
            rollout_started = time.perf_counter()
            phase_totals = {key: 0.0 for key in (
                "policy_seconds", "rollout_collation_seconds", "env_seconds",
                "step_seconds", "reset_seconds", "send_seconds",
                "receive_decode_seconds", "wait_seconds", "buffer_seconds",
                "worker_alns_sum_seconds", "worker_graph_sum_seconds",
                "worker_encode_sum_seconds", "worker_load_sum_seconds")}
            action_hist = [0] * cfg.n_actions
            worker_cache_sizes = {}
            max_worker_ratio = 0.0
            for t in range(cfg.t_rollout):
                policy_started = time.perf_counter()
                collate_started = time.perf_counter()
                batch = Batch.from_data_list(obs)
                phase_totals["rollout_collation_seconds"] += (
                    time.perf_counter() - collate_started)
                with torch.no_grad():
                    a, logp, _, v = model.get_action_and_value(batch.to(device))
                del batch
                actions = a.tolist()  # synchronizes sampled CUDA actions
                phase_totals["policy_seconds"] += time.perf_counter() - policy_started
                for action in actions:
                    action_hist[action] += 1
                vec.set_context(upd=upd, t=t + 1)
                nxt, rewards, dones, infos = vec.step(actions)
                timings = vec.last_timings
                for key in ("env_seconds", "step_seconds", "reset_seconds",
                            "send_seconds", "receive_decode_seconds", "wait_seconds"):
                    phase_totals[key] += timings.get(key, 0.0)
                for worker_id, phases in timings.get("workers", {}).items():
                    for worker_timing in phases.values():
                        for name in ("alns", "graph", "encode", "load"):
                            phase_totals[f"worker_{name}_sum_seconds"] += (
                                worker_timing.get(f"{name}_seconds", 0.0))
                        worker_cache_sizes[worker_id] = worker_timing.get("cache_size", 0)
                mean_worker = timings.get("worker_mean_seconds", 0.0)
                if mean_worker:
                    max_worker_ratio = max(max_worker_ratio,
                        timings.get("worker_max_seconds", 0.0) / mean_worker)
                buffer_started = time.perf_counter()
                buf.add(obs, a.cpu(), logp.cpu(), v.cpu(), rewards, dones)
                phase_totals["buffer_seconds"] += time.perf_counter() - buffer_started
                obs = nxt
                step += cfg.n_envs
                for i, (r, d, info) in enumerate(zip(rewards, dones,
                                                     infos)):
                    ep_reward[i] += r
                    if not d:
                        continue
                    recent_rewards.append(ep_reward[i])
                    roll_mean = statistics.fmean(recent_rewards)
                    roll_std = (statistics.pstdev(recent_rewards)
                                if len(recent_rewards) > 1 else 0.0)
                    if log_rows is not None:
                        log_rows.append({
                            "episode": episodes_done, "step": step,
                            "instance_id": info["instance_id"],
                            "reward_mode": cfg.reward_mode,
                            "episode_reward": round(ep_reward[i], 6),
                            "reward_roll_mean": round(roll_mean, 6),
                            "reward_roll_std": round(roll_std, 6),
                            "best_cost": round(info["f_best"], 4),
                            "init_cost": round(info["f_init"], 4),
                            "elapsed_s": round(time.time() - t0, 1)})
                    episodes_done += 1
                    ep_reward[i] = 0.0
                if (t + 1) % 32 == 0 or t + 1 == cfg.t_rollout:
                    elapsed = time.perf_counter() - rollout_started
                    rate = (t + 1) * cfg.n_envs / max(elapsed, 1e-9)
                    print(f"[rollout upd={upd} t={t+1}/{cfg.t_rollout}] "
                          f"{rate:.2f} env-step/s "
                          f"slowest=w{timings.get('slowest_worker')} "
                          f"wait={timings.get('wait_seconds', 0.0):.3f}s",
                          flush=True)
            phase_totals["rollout_seconds"] = time.perf_counter() - rollout_started
            bootstrap_started = time.perf_counter()
            with torch.no_grad():
                last_v = model.get_value(
                    Batch.from_data_list(obs).to(device)).cpu()
            buf.compute_returns(last_v, cfg.gamma, cfg.gae_lambda)
            phase_totals["bootstrap_gae_seconds"] = time.perf_counter() - bootstrap_started

            progress = upd / max(1, cfg.n_updates - 1)
            metrics = ppo_update(model, opt, buf, cfg, progress, gen)
            metrics.update(phase_totals)
            metrics.update(action_hist=action_hist,
                           worker_cache_sizes=worker_cache_sizes,
                           worker_max_mean_ratio=max_worker_ratio,
                           update_wall_seconds=time.perf_counter() - update_started,
                           resources=_resource_snapshot(vec.worker_pids))
            if update_rows is not None:
                update_rows.append({"update": upd, "step": step,
                                    "reward_mode": cfg.reward_mode, **metrics})
            stats.update(completed_updates=upd + 1, completed_steps=step,
                         episodes_done=episodes_done, last_update=metrics)
            roll = (statistics.fmean(recent_rewards)
                    if recent_rewards else 0.0)
            print(f"[upd {upd}] step={step} ent={metrics['entropy']:.3f} "
                  f"kl={metrics['approx_kl']:.5f} "
                  f"pg={metrics['pg_loss']:.5f} "
                  f"v={metrics['v_loss']:.5f} "
                  f"ev={metrics['explained_variance']:.3f} "
                  f"eps_done={episodes_done} rollR={roll:.3f} "
                  f"optimizer_steps={metrics['optimizer_steps_observed']} "
                  f"{round(time.time() - t0, 1)}s", flush=True)

            if (upd + 1) % 10 == 0 or upd == cfg.n_updates - 1:
                save_checkpoint(model, cfg, norms, metadata, out_path)
        save_checkpoint(model, cfg, norms, metadata, out_path)
    finally:
        vec.close()
        stats["train_wall_seconds"] = time.perf_counter() - train_started
    return model
"""Testing/evaluation for the PPO operator selector (spec: eval).

Runs the SAME ALNSEnv used during training — the geometric-cooling
acceptance criterion and the max_iter horizon are identical by
construction (both come from the checkpoint config).
"""

import dataclasses
import time

import torch
from torch_geometric.data import Batch



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
            "current processed-data pipeline")
    expected_metadata = dict(expected_metadata)
    if stored_metadata != expected_metadata:
        differing = {
            key: {"checkpoint": stored_metadata.get(key),
                  "current": expected_metadata.get(key)}
            for key in sorted(set(stored_metadata) | set(expected_metadata))
            if stored_metadata.get(key) != expected_metadata.get(key)
        }
        raise ValueError(
            f"checkpoint metadata does not match this experiment: {differing}")
    stored_config = ckpt.get("config")
    if not isinstance(stored_config, dict):
        raise ValueError("checkpoint has no valid PPO config")
    if "eps_uniform" not in stored_config:
        raise ValueError(
            "checkpoint has no eps_uniform training setting; retrain it so "
            "training and testing use the same sampling policy")
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in stored_config.items()
                       if k in known})
    cfg.device = device
    model = ActorCritic(cfg).to(device)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        raise RuntimeError(
            "checkpoint incompatible with the current model "
            "(g_t changed to 9 dims — retraining required)") from e
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
            a = int(torch.multinomial(probs[0], 1,
                                      generator=gen).item())
        else:
            a = int(probs.argmax(dim=1).item())
        action_hist[a] += 1
        obs, reward, done, info = env.step(a)
        episode_reward += reward
        if info["f_best"] < best_cost - 1e-9:
            # a new best is always accepted, so env.sol holds it now
            best_sol, best_cost = env.sol.clone(), info["f_best"]
            best_trace.append((env.t, round(time.time() - t0, 3),
                               round(best_cost, 6)))
        if done:
            break

    stats = {"init_cost": env.f_init, "best_cost": best_cost,
             "improve_pct":
                 100.0 * (env.f_init - best_cost) / env.f_init,
             "iters_done": env.t,
             "runtime_s": time.time() - t0,
             "selection_mode": "sampling_epsilon" if sample else "argmax",
             "eps_uniform": cfg.eps_uniform if sample else 0.0,
             "reward_mode": cfg.reward_mode,
             "episode_reward": episode_reward,
             "action_labels": list(ACTION_LABELS),
             "action_hist": action_hist, "best_trace": best_trace}
    return best_sol, stats
