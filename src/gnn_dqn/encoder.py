"""Heterogeneous edge-aware GAT encoder + joint mean/max pooling.

State s_t = concat(mean_pool, max_pool over ALL node types jointly,
g_t) -> 2 * hidden_dim + G_DIM dims (135 with defaults). With
cfg.use_graph = False the encoder is skipped and s_t = g_t.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, global_max_pool, \
    global_mean_pool

from .global_features import G_DIM
from .graph_builder import EDGE_DIMS, EDGE_TYPES

NODE_DIMS = {"customer": 5, "parking": 6, "depot": 3}


class SolutionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.embed = nn.ModuleDict(
            {t: nn.Linear(dim, d) for t, dim in NODE_DIMS.items()})
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            convs = {et: GATv2Conv(d, d, heads=cfg.heads, concat=False,
                                   edge_dim=EDGE_DIMS[et[1]],
                                   add_self_loops=False)
                     for et in EDGE_TYPES}
            self.layers.append(HeteroConv(convs, aggr="sum"))
            self.norms.append(nn.ModuleDict(
                {t: nn.LayerNorm(d) for t in NODE_DIMS}))
        self.act = nn.GELU()

    def forward(self, data):
        """data: HeteroData or Batch -> [B, 2 * hidden_dim]."""
        x = {t: self.act(self.embed[t](data[t].x)) for t in NODE_DIMS}
        eidx = {et: data[et].edge_index for et in data.edge_types}
        eattr = {et: data[et].edge_attr for et in data.edge_types}
        for conv, ln in zip(self.layers, self.norms):
            out = conv(x, eidx, eattr)
            # residual + LayerNorm; node types unseen by any edge type
            # keep their previous embedding
            x = {t: ln[t](x[t] + self.act(out[t])) if t in out else x[t]
                 for t in x}
        hs, batches = [], []
        for t in NODE_DIMS:
            h = x[t]
            b = data[t].batch if hasattr(data[t], "batch") else \
                torch.zeros(h.size(0), dtype=torch.long,
                            device=h.device)
            hs.append(h)
            batches.append(b)
        h_all = torch.cat(hs, dim=0)
        b_all = torch.cat(batches, dim=0)
        return torch.cat([global_mean_pool(h_all, b_all),
                          global_max_pool(h_all, b_all)], dim=1)


class QNet(nn.Module):
    """Encoder + DQN head -> Q-values for the 9 operator pairs."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = SolutionEncoder(cfg) if cfg.use_graph else None
        in_dim = (2 * cfg.hidden_dim + G_DIM) if cfg.use_graph else G_DIM
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, cfg.n_actions))

    def forward(self, data):
        g = data.g.view(-1, G_DIM)
        if self.encoder is None:
            return self.head(g)
        s = torch.cat([self.encoder(data), g], dim=1)
        return self.head(s)
