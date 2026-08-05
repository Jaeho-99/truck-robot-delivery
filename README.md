# Truck-Robot Collaborative Last-Mile Delivery

Source code for the paper on route optimization for truck-robot
collaborative last-mile delivery under zone-based dual (traffic /
pedestrian) congestion. Trucks carry autonomous delivery robots,
deploy them at parking nodes, and retrieve them at possibly different
parking nodes later on the route.

All methods share one objective and feasibility definition:

* **Exact** — a MILP of the full formulation (constraints (1)-(70)),
  solved with Gurobi ([`src/model.py`](src/model.py)).
* **ALNS** — adaptive large neighborhood search (Ropke & Pisinger
  2006 skeleton: random/worst/related destroy, greedy/greedy-noise/
  regret-2 repair, simulated-annealing acceptance) with a
  congestion-aware initial solution
  ([`src/heuristics/`](src/heuristics/)). Operator selection is
  pluggable (`solve_alns(selector=...)`):
  * `roulette` — classic roulette-wheel adaptive weights (vanilla),
  * `qlearning` — **QL-ALNS**, online tabular Q-learning over
    (destroy, repair) pairs
    ([`src/heuristics/qlearning.py`](src/heuristics/qlearning.py)),
  * `gnn_dqn` — **GNN-DQN-ALNS**, a GATv2 encoder over the current
    solution graph plus a Dueling Double-DQN head, trained offline
    ([`src/gnn_dqn/`](src/gnn_dqn/)).

## Repository structure

The layout follows the [INFORMS Journal on Computing software
template](https://github.com/INFORMSJoC) convention of separating
source code, data, scripts, and results:

```
├── src/                      # algorithm code only (no experiment I/O)
│   ├── model.py              # MILP formulation (+ fix_binaries
│   │                         #   verification mode)
│   ├── heuristics/           # metaheuristics
│   │   ├── alns.py           # Params, initial solutions, ALNS driver
│   │   │                     #   (selector plug-in, instrumentation)
│   │   ├── operators.py      # destroy/repair operators
│   │   ├── solution.py       # solution representation and evaluator
│   │   ├── qlearning.py      # tabular Q-learning selector (QL-ALNS)
│   │   └── validator.py      # independent feasibility/objective
│   │                         #   validator (no evaluator reuse)
│   ├── gnn_dqn/              # GNN-DQN operator selection
│   │   ├── config.py         # all hyperparameters (dataclass)
│   │   ├── graph_builder.py  # Solution -> HeteroData graph
│   │   ├── global_features.py# 7-dim global state g_t
│   │   ├── encoder.py        # GATv2 encoder + Dueling DQN head
│   │   ├── dqn_agent.py      # replay buffer, Double-DQN updates
│   │   ├── reward.py         # R1 / R2 / binary reward modes
│   │   ├── provider.py       # train/test instance providers
│   │   ├── trainer.py        # offline training loop
│   │   ├── normalization.py  # feature normalization constants
│   │   └── selector_gnn.py   # frozen-model inference adapter
│   ├── instance.py           # instance schema and generators
│   ├── plotting.py           # instance/route SVG figures
│   └── utils.py              # reports, diagnostics, CSV, payloads
├── data/
│   └── generator.py          # instance generation script
├── experiments/
│   ├── run_experiment.py     # exact vs ALNS runner (config-driven)
│   ├── run_qlearning.py      # selector comparison runner (saves
│   │                         #   solutions, traces, validator checks)
│   ├── train_gnn_dqn.py      # GNN-DQN offline training CLI
│   ├── verify_solution_milp.py # MILP fixing check of saved solutions
│   ├── make_route_svgs.py    # route SVGs from saved solutions
│   └── configs/              # experiment configs (JSON)
│       ├── main_alns.json / main_ql.json / main_gnn.json
│       └── toy_small.json / toy_scaling.json
├── tests/                    # pytest: gnn_dqn + validator suites
├── models/                   # trained checkpoints (gitignored)
└── results/                  # experiment outputs (gitignored)
```

Algorithm code (`src/`) contains no experiment settings or file paths;
experiments are described entirely by JSON configs under
`experiments/configs/`.

## Installation

Python >= 3.9. Two dependency groups:

* **Exact / plotting**: [Gurobi](https://www.gurobi.com) (tested with
  Gurobi 12.0; free academic license), numpy, matplotlib.
* **GNN-DQN**: torch + torch_geometric (pinned in
  `requirements.txt`); CPU is sufficient. The heuristics themselves
  are pure standard library — ALNS and QL-ALNS run without any of the
  above.

```bash
pip install -r requirements.txt
```

## Usage

Selector comparison (vanilla ALNS / QL-ALNS / GNN-DQN-ALNS) on the
nested scaling instances (n = 5..100, master seed 1):

```bash
python experiments/run_qlearning.py \
    --config experiments/configs/main_alns.json   # or main_ql / main_gnn
```

Each run writes to `results/<experiment>/`: `runs_compare.csv` /
`summary_compare.csv` (objective, runtime, cost breakdown, feasibility
re-checks, selector instrumentation), `solutions/sol_*.json` (best
solution of every run), and `traces/trace_*.csv` (convergence).

Train the GNN-DQN selector (masters seeds 2-4; the test instances of
master seed 1 are never sampled):

```bash
python experiments/train_gnn_dqn.py \
    --episodes 3000 --episode-len 500 --train-freq 4 \
    --trucks 4 --robots 2 --out models/gnn_dqn_final.pt
```

Exact vs ALNS comparison (Gurobi required):

```bash
python experiments/run_experiment.py \
    --config experiments/configs/toy_small.json
```

Post-processing of saved solutions:

```bash
# route figures (SVG) for every saved solution of an experiment
python experiments/make_route_svgs.py --results-dir results/main_ql

# gold-standard check: fix the MILP's routing binaries to a saved
# solution and confirm the objective matches (requires Gurobi)
python experiments/verify_solution_milp.py \
    --solution results/main_ql/solutions/sol_n15_s1_qlearning_0.json
```

Tests:

```bash
python -m pytest tests/ -q
```

## Reproducibility

All randomness is seeded:

* Instances are rebuilt deterministically from the seeds in the config
  (`src/instance.py`); the scaling instances use a fixed master pool of
  100 customers sliced to the first n, so instances are nested across
  sizes.
* ALNS is fully deterministic given the run seed, for every selector
  (the learned selectors use dedicated rng streams / frozen models).
* The exact method reports the proven optimality gap; optimal
  objective values are reproducible, while runtimes and time-limited
  incumbents may vary across machines.

Solution correctness is certified on three independent levels: the
ALNS evaluator, a from-primitives validator
(`src/heuristics/validator.py`, run automatically on every saved
solution), and the MILP fixing check
(`experiments/verify_solution_milp.py`).

## Notes

* `results/` and `models/*.pt` are gitignored; the directory structure
  is kept via `.gitkeep`. Trained checkpoints must be copied between
  machines (or retrained) separately.
* A real-data case study will be added later as an additional config
  plus an instance parser in `src/instance.py`.
