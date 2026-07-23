# Truck-Robot Collaborative Last-Mile Delivery

Source code for the paper on route optimization for truck-robot
collaborative last-mile delivery under zone-based dual (traffic /
pedestrian) congestion. Trucks carry autonomous delivery robots,
deploy them at parking nodes, and retrieve them at possibly different
parking nodes later on the route.

Two solution methods are provided over the same objective and
feasibility definition:

* **Exact** — a MILP of the full formulation (constraints (1)-(70)),
  solved with Gurobi ([`src/model.py`](src/model.py)).
* **ALNS** — adaptive large neighborhood search (Ropke & Pisinger
  2006 skeleton: random/worst/related destroy, greedy/greedy-noise/
  regret-2 repair, roulette-wheel adaptive weights, simulated-annealing
  acceptance) with a congestion-aware initial solution
  ([`src/heuristics/`](src/heuristics/)).

## Repository structure

The layout follows the [INFORMS Journal on Computing software
template](https://github.com/INFORMSJoC) convention of separating
source code, data, scripts, and results:

```
├── src/                      # algorithm code only (no experiment I/O)
│   ├── model.py              # MILP formulation (exact method)
│   ├── heuristics/           # metaheuristics
│   │   ├── alns.py           # Params, initial solutions, ALNS driver
│   │   ├── operators.py      # destroy/repair operators
│   │   └── solution.py       # solution representation and evaluator
│   ├── instance.py           # instance schema and generators
│   ├── plotting.py           # instance/route SVG figures
│   └── utils.py              # reports, diagnostics, CSV, payloads
├── data/
│   ├── generator.py          # instance generation script
│   └── instances/            # exported toy instance payloads
├── experiments/
│   ├── run_experiment.py     # unified config-driven experiment runner
│   └── configs/
│       ├── toy_small.json    # small grids: exact vs ALNS, 5 seeds
│       └── toy_scaling.json  # nested scaling (n = 5..100)
└── results/                  # experiment outputs (gitignored)
```

Algorithm code (`src/`) contains no experiment settings or file paths;
experiments are described entirely by JSON configs under
`experiments/configs/` and executed by the single runner
`experiments/run_experiment.py`. Exact decomposition methods (e.g.
branch-and-price, Benders) would live in a future `src/exact/`
subpackage.

## Installation

Python >= 3.9 with [Gurobi](https://www.gurobi.com) (tested with
Gurobi 12.0; a free academic license is available):

```bash
pip install -r requirements.txt
```

## Usage

Run an experiment from the repository root by pointing the runner at a
config:

```bash
# small grids (4 and 6 customers), exact vs ALNS, seeds 1-5
python experiments/run_experiment.py \
    --config experiments/configs/toy_small.json

# customer-count scaling on nested instances (exact up to n = 20,
# ALNS up to n = 100). Note: exact time limits are up to 3 h / 3 days.
python experiments/run_experiment.py \
    --config experiments/configs/toy_scaling.json
```

Outputs go to `results/<experiment>/`: per-instance JSON payloads,
Gurobi logs, text reports with cost breakdown, routes and custody
diagnostics, route figures (SVG), and `summary_exact.csv` /
`summary_alns.csv` / `summary_compare.csv`.

A new comparison (e.g. a case study) only needs a new config file —
choose the instance set, the methods to run, and the solver
parameters; see the schema documented at the top of
[`run_experiment.py`](experiments/run_experiment.py).

To (re-)export the toy instance payloads:

```bash
python data/generator.py
```

## Reproducibility

All randomness is seeded:

* Instances are rebuilt deterministically from the seeds in the config
  (`src/instance.py`); the scaling instances use a fixed master pool of
  100 customers sliced to the first n, so instances are nested across
  sizes.
* ALNS is fully deterministic given `alns.seed` in the config.
* The exact method reports the proven optimality gap; optimal
  objective values are reproducible, while runtimes and time-limited
  incumbents may vary across machines.

Each comparison run can additionally re-evaluate the exact solution
with the ALNS evaluator (`check_evaluator_consistency`); the reported
`recon_diff` must be ~0, certifying that both methods share one
objective and feasibility definition.

## Notes

* `results/` is gitignored; the directory structure is kept via
  `.gitkeep`.
* A real-data case study will be added later as an additional config
  plus an instance parser in `src/instance.py`.
