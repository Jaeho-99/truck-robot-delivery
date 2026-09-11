"""Mirror of ``src/`` with faster ALNS destroy/repair operators.

``alns``, ``ppo_alns`` and ``gnn_ppo_alns`` are the v1 packages with the
same layout, entry points and artifacts; only the destroy/repair
operators and the evaluator inside each ``alns`` module differ. They
search the same candidate space and reproduce the v1 results bit for
bit. See ``README.md`` in this package.
"""
