"""PPO-ALNS policy, environment, update, training, and testing."""

from .alns import (ACTION_COUNT, ACTION_LABELS, DOD, NOISE_FRAC, W_START,
                   apply_actor_action, congestion_aware_initial,
                   eval_solution)


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


def robot_served_customers(sol):
    served = set()
    for route in sol.routes.values():
        for stop in route:
            if stop["kind"] == "park":
                for trip in stop["deploys"]:
                    served.update(trip["custs"])
    return served


def congestion_ratio(pr, robot_served):
    """Assigned-mode congestion divided by its per-customer lower bound."""
    zones = pr.node_zone
    numerator = denominator = 0.0
    for customer in pr.C:
        truck = pr.alpha_traffic[zones[customer]]
        robot = pr.alpha_ped[zones[customer]]
        numerator += robot if customer in robot_served else truck
        denominator += min(truck, robot)
    return numerator / denominator if denominator else 1.0
"""Global search-state features g_t (9-dim float32 vector).

This is the complete observation used by the graph-free PPO-ALNS policy.

Features 2-8 follow the DR-ALNS observation space (Reijnen et al.,
ICAPS 2024); 0-1 are problem-specific extensions of this work (robot
share / congestion exploitation). Our objective is minimized, so
"improved" means a cost DECREASE (the original paper maximizes).
"""

import torch


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
    eps = max(float(pr.alpha_traffic.max()) - 1.0,
              float(pr.alpha_ped.max()) - 1.0, 1e-9)
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
"""All PPO hyperparameters in one dataclass.

The environment protocol follows DR-ALNS. Stability defaults include an
entropy floor, KL early stopping, and magnitude-reward scaling.
"""

from dataclasses import asdict, dataclass



@dataclass
class PPOConfig:
    use_graph: bool = False         # fixed: state is the 9-dimensional g_t
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
    device: str = "cpu"
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
"""Graph-free PPO actor-critic over the 9-dimensional g_t state."""

import torch
import torch.nn as nn
from torch.distributions import Categorical



def _ortho(layer, std):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.use_graph:
            raise ValueError("PPO-ALNS uses g_t only; use gnn_ppo_alns")
        if cfg.n_actions != ACTION_COUNT:
            raise ValueError(
                f"PPO actor requires {ACTION_COUNT} ALNS actions, "
                f"got {cfg.n_actions}")
        if not 0.0 <= cfg.eps_uniform <= 1.0:
            raise ValueError("eps_uniform must be between 0 and 1")
        self.eps_uniform = float(cfg.eps_uniform)
        self.n_actions = int(cfg.n_actions)
        d_state = G_DIM
        self.actor = nn.Sequential(
            _ortho(nn.Linear(d_state, 64), 2 ** 0.5), nn.Tanh(),
            _ortho(nn.Linear(64, cfg.n_actions), 0.01))
        self.critic = nn.Sequential(
            _ortho(nn.Linear(d_state, 64), 2 ** 0.5), nn.Tanh(),
            _ortho(nn.Linear(64, 1), 1.0))

    def _state(self, data):
        return data.g.view(-1, G_DIM)

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
"""On-policy rollout storage for PPO (spec: buffer).

Holds exactly one rollout (t_rollout x n_envs) and is reset after
every update — never a replay buffer (PPO is on-policy). States are
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
            yield (Batch.from_data_list([self.obs[i] for i in idx]),
                   actions[chunk], log_probs[chunk], advantages[chunk],
                   returns[chunk], values[chunk])
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

    return {"pg_loss": mean(pg_losses), "v_loss": mean(v_losses),
            "entropy": mean(entropies), "approx_kl": approx_kl,
            "clipfrac": mean(clipfracs), "explained_variance": ev,
            "c2": c2, "epochs_run": epochs_run}
"""ALNS as a step-interface environment for PPO rollouts.

Episodes have cfg.search_iterations actor-selected transitions, linear SA
temperature decay from T0 = W_START * f_init / ln 2 to 0, and fixed degree
of destruction q = round(DOD * n). g_t is computed locally.
The configured reward mode is stored in the checkpoint and reused at test.
"""

import math
import random

from torch_geometric.data import Data



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

    def reset(self):
        self.pr = self.provider.sample()
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
        return self._obs()

    def _temperature(self):
        """Linear SA temperature decay over one actor episode."""
        return self.T0 * (1.0 - self.t / self.cfg.search_iterations)

    def _obs(self):
        data = (self.builder.build(self.pr, self.sol)
                if self.builder is not None else Data(num_nodes=0))
        best_improved, accepted, cur_improved = self.flags
        data.g = global_features(
            self.pr, self.sol, self.t, self.cfg.search_iterations,
            self.stagcount, self.f_cur, self.f_best,
            best_improved=best_improved, current_accepted=accepted,
            current_improved=cur_improved)
        return data          # GraphBuilder.build already pads

    def step(self, a):
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
        return (self._obs(), reward, done,
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
        self.envs = envs

    def reset(self):
        return [e.reset() for e in self.envs]

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, d, info = env.step(a)
            if d:
                o = env.reset()
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
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


REWARD_WINDOW = 100     # rolling stats window (episodes)


def make_envs(cfg, provider, builder):
    return VecALNS([ALNSEnv(provider, builder, cfg, cfg.seed + i)
                    for i in range(cfg.n_envs)])


def save_checkpoint(model, cfg, norms, metadata, path):
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                "norms": norms, "metadata": dict(metadata)}, path)


def train(cfg, provider, out_path, norms, log_rows=None,
          update_rows=None):
    """Train and save a checkpoint to out_path. Returns the model.

    log_rows: per-episode reward rows (CSV/plot format);
    update_rows: per-update PPO metrics.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = ActorCritic(cfg).to(device)
    opt = make_optimizer(model, cfg)
    vec = make_envs(cfg, provider, None)
    gen = torch.Generator().manual_seed(cfg.seed)   # minibatch shuffle
    obs = vec.reset()
    ep_reward = [0.0] * cfg.n_envs
    recent_rewards = deque(maxlen=REWARD_WINDOW)
    episodes_done = 0
    step = 0                    # cumulative env steps over all workers
    t0 = time.time()
    metadata = provider.checkpoint_metadata

    for upd in range(cfg.n_updates):
        buf = RolloutBuffer(cfg.t_rollout, cfg.n_envs)
        for _ in range(cfg.t_rollout):
            with torch.no_grad():
                a, logp, _, v = model.get_action_and_value(
                    Batch.from_data_list(obs).to(device))
            nxt, rewards, dones, infos = vec.step(a.tolist())
            buf.add(obs, a.cpu(), logp.cpu(), v.cpu(), rewards, dones)
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
        with torch.no_grad():
            last_v = model.get_value(
                Batch.from_data_list(obs).to(device)).cpu()
        buf.compute_returns(last_v, cfg.gamma, cfg.gae_lambda)

        progress = upd / max(1, cfg.n_updates - 1)
        metrics = ppo_update(model, opt, buf, cfg, progress, gen)
        if update_rows is not None:
            update_rows.append({"update": upd, "step": step,
                                "reward_mode": cfg.reward_mode, **metrics})
        roll = (statistics.fmean(recent_rewards)
                if recent_rewards else 0.0)
        print(f"[upd {upd}] step={step} ent={metrics['entropy']:.3f} "
              f"kl={metrics['approx_kl']:.5f} "
              f"pg={metrics['pg_loss']:.5f} "
              f"v={metrics['v_loss']:.5f} "
              f"ev={metrics['explained_variance']:.3f} "
              f"eps_done={episodes_done} rollR={roll:.3f} "
              f"{round(time.time() - t0, 1)}s", flush=True)

        if (upd + 1) % 10 == 0 or upd == cfg.n_updates - 1:
            save_checkpoint(model, cfg, norms, metadata, out_path)
    save_checkpoint(model, cfg, norms, metadata, out_path)
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
