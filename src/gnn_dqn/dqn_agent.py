"""DQN agent: replay buffer, eps-greedy acting, Double DQN updates.

Action space (fixed mapping, MUST match tabular qlearning.py for
comparability): a = destroy_index * 3 + repair_index with
destroy = [random, worst, related], repair = [greedy, greedy_noise,
regret2]; decompose with divmod(a, 3).
"""

import copy
import random
from collections import deque

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .encoder import QNet


class ReplayBuffer:
    """FIFO buffer of (HeteroData_t, a, r, HeteroData_{t+1}).

    Graphs are stored on CPU with their g_t vector attached as
    ``data.g``; batches are assembled with Batch.from_data_list and
    moved to the device at sample time.
    """

    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, g0, a, r, g1):
        self.buf.append((g0, a, r, g1))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size, rng):
        items = rng.sample(list(self.buf), batch_size)
        g0, a, r, g1 = zip(*items)
        return (Batch.from_data_list(list(g0)),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                Batch.from_data_list(list(g1)))


class DQNAgent:
    def __init__(self, cfg, total_steps):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.online = QNet(cfg).to(self.device)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.total_steps = max(1, total_steps)
        self.rng = random.Random(cfg.seed)

    def epsilon(self, step):
        decay = self.cfg.eps_decay_frac * self.total_steps
        frac = min(1.0, step / max(1.0, decay))
        return (self.cfg.eps_start
                + frac * (self.cfg.eps_end - self.cfg.eps_start))

    def act(self, data, step, greedy=False):
        if not greedy and self.rng.random() < self.epsilon(step):
            return self.rng.randrange(self.cfg.n_actions)
        with torch.no_grad():
            q = self.online(Batch.from_data_list([data])
                            .to(self.device))
        return int(q.argmax(dim=1).item())

    def update(self):
        """One Double-DQN step; returns the scalar loss."""
        bs = min(self.cfg.batch_size, len(self.buffer))
        s, a, r, s2 = self.buffer.sample(bs, self.rng)
        s, s2 = s.to(self.device), s2.to(self.device)
        a, r = a.to(self.device), r.to(self.device)
        q = self.online(s).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            a_star = self.online(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, a_star).squeeze(1)
            # episodes end by truncation only: always bootstrap
            y = r + self.cfg.gamma * q_next
        loss = self.loss_fn(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),
                                 self.cfg.grad_clip)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    # ---- persistence ----
    def save(self, path, norms):
        torch.save({"model": self.online.state_dict(),
                    "config": self.cfg.to_dict(), "norms": norms}, path)
