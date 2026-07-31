"""GNN+DQN operator selection for ALNS (graph-encoded deep Q-learning).

A GATv2 encoder over the current solution graph plus a DQN head that
outputs Q-values for the 9 (destroy, repair) operator pairs. Training
is offline (trainer.py); inference plugs into ``solve_alns`` via
``selector="gnn_dqn"`` (selector_gnn.py).

Requires torch + torch_geometric (see requirements.txt); the rest of
the package (src.heuristics) stays importable without them.
"""
