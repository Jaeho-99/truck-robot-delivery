"""Inference-time selector: frozen model, eps = 0 (pure greedy).

Adapter for the selector plug-in of solve_alns:
    solve_alns(pr, selector="gnn_dqn",
               q_params={"model_path": "models/gnn_dqn.pt"})
The same single checkpoint is used for all instance sizes.
"""

import torch
from torch_geometric.data import Batch

from .config import Config
from .encoder import QNet
from .global_features import global_features
from .graph_builder import GraphBuilder


class GNNSelector:
    def __init__(self, model_path, device="cpu"):
        ckpt = torch.load(model_path, map_location=device,
                          weights_only=False)
        self.cfg = Config(**ckpt["config"])
        self.cfg.device = device
        self.net = QNet(self.cfg).to(device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()
        self.builder = GraphBuilder(ckpt["norms"], self.cfg)
        self.device = device

    def select(self, pr, sol, progress, stagnation_iters, f_cur,
               f_best):
        """Greedy (destroy_index, repair_index) for the current state."""
        data = self.builder.build(pr, sol)
        data.g = global_features(pr, sol, progress, stagnation_iters,
                                 f_cur, f_best)
        with torch.no_grad():
            q = self.net(Batch.from_data_list([data]).to(self.device))
        return divmod(int(q.argmax(dim=1).item()), 3)
