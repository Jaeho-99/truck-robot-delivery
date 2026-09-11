# v2_claude — faster ALNS operators, same everything else

`src/v2_claude/` mirrors `src/` file for file. Every module is the v1
file, unchanged, **except the ALNS destroy/repair operators and the
evaluator** inside each package's own `alns` module — exactly where they
live in v1, no shared helper module.

```
src/                              src/v2_claude/
  alns/solve.py                     alns/solve.py
  ppo_alns/{alns,observation,        ppo_alns/{...}          same files
            ppo,train,test}.py
  gnn_ppo_alns/{...}.py              gnn_ppo_alns/{...}.py   same files
  common/policy_evaluation.py        common/policy_evaluation.py
```

Artifacts move one level down so the two can live side by side:

| | v1 | v2_claude |
|---|---|---|
| ALNS results | `output/alns/n20/` | `output/v2_claude/alns/n20/` |
| PPO results | `output/ppo_alns/n20/` | `output/v2_claude/ppo_alns/n20/` |
| GNN results | `output/gnn_ppo_alns/n20/` | `output/v2_claude/gnn_ppo_alns/n20/` |
| checkpoints | `models/ppo_alns_n20_*.pt` | `models/v2_claude/ppo_alns_n20_*.pt` |

Commands are otherwise unchanged — just swap the path:

```
python src/v2_claude/alns/solve.py --size 20 --workers 10
python src/v2_claude/ppo_alns/train.py --size 20 --reward-mode alns_5310 --device cuda --env-backend process --observation-codec numpy
python src/v2_claude/ppo_alns/test.py --size 20 --reward-mode alns_5310 --workers 10
python src/v2_claude/gnn_ppo_alns/train.py --size 20 --reward-mode alns_5310 --device cuda --env-backend process --observation-codec numpy
python src/v2_claude/gnn_ppo_alns/test.py --size 20 --reward-mode alns_5310 --workers 10

python scripts/summarize_results.py --size 20 --variant v2_claude
python scripts/plot_result.py output/v2_claude/alns/n20/test_n20_000_s0.json
```

`run_n20_cuda_v2_claude.bat` and `run_n5_n10_cuda_v2_claude.bat` are the
v2 copies of the existing batch runners.

`summarize_results.py --variant v2_claude` reads `output/v2_claude/` and
`models/v2_claude/` and writes its tables there; without the flag it
behaves exactly as before. `plot_result.py` takes an explicit result path
and needed no change.

## Measured result

Results are not "close" — they are **identical**. Nothing about the
search changed, only how much work each iteration costs.

**Vanilla ALNS**, full test set (50 instances x 5 seeds x 100
iterations), diffed against the existing `output/alns/` runs:

| size | tasks | objective mismatches | route/stats mismatches | CPU (sum of case solves) |
|------|-------|----------------------|------------------------|--------------------------|
| n5   | 250   | 0                    | 0                      | 238.4 s -> 45.5 s (**5.2x**) |
| n10  | 250   | 0                    | 0                      | 1013.4 s -> 129.2 s (**7.9x**) |
| n20  | 250   | 0                    | 0                      | 7110.9 s -> 581.3 s (**12.2x**) |

All 750 published route files match v1 field for field — objective,
routes, operator weights, action histogram, accept/infeasible counts.
Only the wall-clock fields differ. A single n50 case runs **17x** faster,
also bit-identical.

`summarize_results.py` agrees: for n20, `obj_min/obj_mean/obj_max` are
`19.6208 / 33.0493 / 42.5329` in both tables, `runtime_s_mean` drops
`28.444 -> 2.325`.

**PPO training**, one update (2560 env steps, 10 process workers, CPU):

| | rollout throughput | update wall | training stats |
|---|---|---|---|
| `ppo_alns` n20 | 16.9 -> 121.7 env-step/s (**7.2x**) | 152.7 s -> 23.3 s | identical |
| `gnn_ppo_alns` n20 | 15.3 -> 83.6 env-step/s (**5.5x**) | 292.7 s -> 153.6 s | identical |
| `ppo_alns` n5 | 400 -> 971 env-step/s (**2.4x**) | 8.2 s -> 4.4 s | identical |

"identical" means the update log line matches digit for digit:
`ent=2.189 kl=0.01166 pg=-0.01128 v=460.16376 ev=0.002 rollR=150.050`.
The policy saw exactly the same trajectories.

Note the GNN row: its rollout gets 5.5x, but the update wall only 1.9x,
because at that point the graph encoder and the optimizer dominate, not
the operators. The ALNS work is no longer the GNN bottleneck.

## Verifying it yourself

```
python scripts/verify_v2_claude.py --mode artifacts --size 5 10 20   # diff published runs
python scripts/verify_v2_claude.py --mode solve     --size 5 10 20   # head-to-head solve_alns
python scripts/verify_v2_claude.py --mode actor     --size 5 10 20   # all 9 PPO actions, step by step
```

`--mode actor` is the one that covers PPO and GNN: it drives
`apply_actor_action` — the single search transition the actor selects —
from an identical state and random stream for each of the nine
destroy/repair pairs, and compares objective, feasibility and routes.

## Why it is faster

The v1 profile on n20 (100 iterations) was **555,762 `eval_truck` calls**,
94% of runtime, and **70% of those candidates were rejected on custody
alone**. The five changes below attack exactly that.

### 1. O(1) custody screening — the big one

Modes C and D deploy a robot at one point of the route and retrieve it at
a later one. That is custody-feasible only if

* the robot is aboard where the trip would launch, and
* the base route does not re-deploy the same robot in between.

Both are decidable from three small tables built once per truck per
repair round (`_custody_profile`): the aboard bitmask on arrival at each
stop, the aboard bitmask after each stop's own deploys, and
`next_deploy[r][i]`, the first stop at or after `i` that deploys robot
`r`. Candidates that fail are never built and never evaluated.

After this filter **99.7%** of evaluated candidates are feasible, up from
28.6%. Evaluator calls on n20 drop from 555,762 to ~117,000.

This changes nothing observable: the evaluator would have rejected
exactly these candidates, and the noise operator draws its
`U(-noise, noise)` only *after* the feasibility test, so the random
stream is untouched.

### 2. A tight evaluator over flat tables

`Params.__init__` materializes the three travel matrices as nested Python
lists (`pr.dl`, `pr.ttl`, `pr.trl`), so an arc lookup is `ttl[i][j]`
instead of `float(np_matrix[i, j])` behind a bound method — those three
accessors alone were 25% of v1's runtime. `_run` accumulates lateness as
a running sum instead of filling a `dict` and summing it, tracks robots
in a bitmask, and allocates no `defaultdict` per call.

The statement order is preserved exactly, so every float is accumulated
in the same order and the results are bit-identical rather than merely
close. The readable reference form survives as `eval_truck`, which the
initial-solution constructor and the reporting breakdown still use.

### 3. Prefix resume

Every insertion candidate agrees with the base route on `route[:si]`. The
base route is scanned once per repair round with the evaluator state
snapshotted before each stop (`_prefix_states`), so a candidate resumes
from `si` instead of restarting at the depot. On n20 that is ~5.1 stops
evaluated instead of ~7.9.

### 4. Robot symmetry

Robots are homogeneous — same range, same fixed cost — so idle robots are
interchangeable and every idle robot past the first yields a duplicate
candidate of exactly the same cost, which the strict `<` tie-break
discards anyway. `_robots_for` enumerates only the robots already
deployed on the route plus the lowest-index idle one, in ascending order,
which keeps the original winner.

*Caveat, and why `EXACT_NOISE_STREAM` exists:* the noise operator draws
one random number per **feasible** candidate, so dropping cost-identical
duplicates still shifts its random stream. With `EXACT_NOISE_STREAM =
True` (the default) the reduction is skipped for that one operator and
the whole search reproduces v1 bit for bit. Clearing the flag buys
roughly a further 1.5x on noise iterations at the cost of that guarantee.

### 5. Cheaper destroy and repair bookkeeping

* **Insertion cache.** Inserting a customer into truck `k` leaves every
  other truck untouched, and the used-copy set only grows during a
  repair, so `_RepairContext` reuses the per-(customer, truck) best
  insertion across greedy/regret rounds. An entry is recomputed only when
  its truck changed or when a parking copy it claimed was taken
  meanwhile — and a replacement copy at the same physical location costs
  exactly the same, so no better candidate can appear in the meantime.
  Off for the noise operator (see above).
* **`repair_regret2`** recomputed the per-truck base costs once per
  *customer* per round; now once per round.
* **`destroy_worst`** cloned and re-evaluated the whole solution for
  every customer. Removing one customer touches one truck, so only that
  route is rebuilt, and the total is re-summed over the trucks in their
  original order — which keeps the removal gains bit-identical.
* **`destroy_related`** sorted every candidate by a Python-level Shaw key
  each round; the ranking is now precomputed once per instance
  (`pr.related_order`).
* **`Solution.clone`** uses a structural copy (`_copy_route`) instead of
  `copy.deepcopy`, whose memo bookkeeping this fixed three-level shape
  does not need.

## What is *not* changed

The search skeleton is untouched: roulette / actor operator selection,
adaptive weights, the DR-ALNS 5/3/1/0 scores, SA acceptance and its
cooling schedule, `DOD = 0.3`, `W_START = 0.05`, `NOISE_FRAC = 0.25`, the
congestion-aware initial solution, the candidate caps `L_RET_EXIST = 3` /
`W_RET_NEW = 4` / `N_PHYS_NEAR = 2`, the four insertion modes A/B/C/D and
their enumeration order, the objective, the feasibility rules, the
PPO/GNN networks, rewards, observations, worker protocols and artifact
schemas.

## How each file differs from v1

Measured with `diff --strip-trailing-cr` (the v1 originals have mixed
CRLF/LF endings; the copies are LF):

| file | changed lines | what changed |
|------|---------------|--------------|
| `alns/solve.py` | 959 | operator + evaluator section; `output/v2_claude/alns/`; log prefix; `sys.path` depth |
| `ppo_alns/alns.py` | 936 | operator + evaluator section only (see below) |
| `gnn_ppo_alns/alns.py` | 936 | same |
| `common/policy_evaluation.py` | 26 | dotted package name, `models/v2_claude/`, `output/v2_claude/` |
| `gnn_ppo_alns/parallel_env.py` | 21 | accepts the `v2_claude.*` package names |
| `ppo_alns/train.py` | 15 | imports, `sys.path` depth, `models/v2_claude/`, `output/v2_claude/` |
| `gnn_ppo_alns/train.py` | 14 | same |
| `ppo_alns/test.py`, `gnn_ppo_alns/test.py` | 9 each | imports, `sys.path` depth, evaluator package name |
| `ppo_alns/ppo.py` | 4 | vec-env package name |
| `gnn_ppo_alns/observation.py` | 2 | import path |
| `gnn_ppo_alns/gnn.py`, `ppo_alns/observation.py`, `gnn_ppo_alns/ppo.py` | 0 | verbatim |

In the two PPO `alns.py` files, **everything from `DESTROY_OPERATORS`
onward differs by exactly two things**: `apply_actor_action` calls
`eval_solution_cost` instead of `eval_solution`, and `Params.__init__`
builds the flat lookup tables. `DESTROY_OPERATORS`, `ACTION_LABELS`,
`apply_actor_action`, the constants, the initial-solution constructors,
`InstanceCache` and `DirectoryInstanceProvider` are byte-identical to v1.

## Switches

Near the top of each `alns` module (all three carry the same block):

| flag | default | effect |
|------|---------|--------|
| `INSERTION_CACHE` | `True` | reuse per-(customer, truck) insertions across repair rounds |
| `ROBOT_SYMMETRY` | `True` | enumerate only deployed robots plus one idle |
| `EXACT_NOISE_STREAM` | `True` | keep the noise operator's random stream identical to v1 |

With all three at their defaults the run is bit-identical to `src/`.
`EXACT_NOISE_STREAM = False` is the only one that changes results.
