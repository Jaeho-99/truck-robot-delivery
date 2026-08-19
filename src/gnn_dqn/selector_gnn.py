"""Inference-time selector: frozen model, eps = 0 (pure greedy).

Adapter for the selector plug-in of solve_alns:
    solve_alns(pr, selector="gnn_dqn",
               q_params={"model_path": "models/gnn_dqn.pt"})
The same single checkpoint is used for all instance sizes.
"""

import dataclasses

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
        known = {f.name for f in dataclasses.fields(Config)}
        self.cfg = Config(**{k: v for k, v in ckpt["config"].items()
                             if k in known})
        self.cfg.device = device
        self.net = QNet(self.cfg).to(device)
        try:
            self.net.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            raise RuntimeError(
                "checkpoint incompatible with the current model "
                "(g_t changed to 9 dims — retraining required)") from e
        self.net.eval()
        self.builder = GraphBuilder(ckpt["norms"], self.cfg)
        self.device = device

    def select(self, pr, sol, it, iters, stagcount, current_cost,
               best_cost, best_improved=False, current_accepted=False,
               current_improved=False):
        """Greedy (destroy_index, repair_index) for the current state.

        it = completed search iterations, iters = the run's budget
        (normalizes stagcount/search_budget in g_t); the three flags
        are the previous iteration's outcome (DR-ALNS obs).
        """
        data = self.builder.build(pr, sol)
        data.g = global_features(pr, sol, it, iters, stagcount,
                                 current_cost, best_cost,
                                 best_improved, current_accepted,
                                 current_improved)
        with torch.no_grad():
            q = self.net(Batch.from_data_list([data]).to(self.device))
        return divmod(int(q.argmax(dim=1).item()), 3)
