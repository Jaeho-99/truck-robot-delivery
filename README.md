# Truck-Robot Delivery

Four independent methods solve the same preprocessed truck-robot delivery
instances:

- `src/exact/solve.py`: Gurobi exact MILP
- `src/alns/solve.py`: vanilla roulette-wheel ALNS
- `src/ppo_alns/`: PPO-ALNS (`alns.py`, `ppo.py`, `train.py`, `test.py`)
- `src/gnn_ppo_alns/`: GNN-PPO-ALNS (`alns.py`, `gnn.py`, `ppo.py`,
  `train.py`, `test.py`)

## Data flow

```text
data/raw/{train,test}/n{20,50,100}/*.json
                    │
                    ▼  scripts/preprocess.py
data/processed*/{train,test}/n{20,50,100}/*.npz
                    │
                    ├── exact
                    ├── vanilla ALNS
                    ├── PPO-ALNS
                    └── GNN-PPO-ALNS
```

The model code never reconstructs distances or congestion-adjusted travel
times. It loads `d`, `tau_truck`, and `tau_robot` from each NPZ and performs
only route and assignment optimization. Runtime-only fleet, capacity, cost,
range, service-time, and deadline values come from `configs/params.yaml`.

Install the local packages once from the repository root:

```bash
python -m pip install -e .
```

## Preprocessing

For n20 train and test data:

```bash
python scripts/preprocess.py --split all --size 20
```

Use `--force` after changing preprocessing-relevant geometry or speed values.
For a sensitivity dataset, `--tag NAME` writes under
`data/processed_NAME/`.

## Running each method

`--size` is always required. The untagged commands for n20 are:

```bash
python src/exact/solve.py --size 20
python src/alns/solve.py --size 20
python src/ppo_alns/train.py --size 20
python src/ppo_alns/test.py --size 20
python src/gnn_ppo_alns/train.py --size 20
python src/gnn_ppo_alns/test.py --size 20
```

The exact solver processes one instance by default; pass `--limit 0` to run
all test instances. ALNS and learned selectors process all test instances by
default and accept `--limit N`.

PPO-ALNS and GNN-PPO-ALNS use the same epsilon-mixed sampling policy during
training and testing: `0.9 * pi_actor + 0.1 / 9` by default. Training accepts
`--eps-uniform VALUE`; the value is stored in the checkpoint and testing
always reloads it, so the two phases cannot silently use different epsilon
settings. Checkpoints created before this setting was added must be retrained.

Both PPO methods support three semantic reward modes through
`--reward-mode`:

| Mode | Reward | Checkpoint token |
|---|---|---|
| `alns_5310` | new best/current improvement/accepted/else = 5/3/1/0 | `reward_alns_5310` |
| `new_best_5` | new best = 5, else = 0 | `reward_new_best_5` |
| `magnitude` | `10 * max(0, delta_best) / initial_obj` | `reward_magnitude` |

Use the same mode for training and testing. The checkpoint stores the mode,
and testing rejects a mismatch. For example:

```bash
python src/ppo_alns/train.py --size 20 --reward-mode alns_5310
python src/ppo_alns/test.py --size 20 --reward-mode alns_5310

python src/gnn_ppo_alns/train.py --size 20 --reward-mode magnitude
python src/gnn_ppo_alns/test.py --size 20 --reward-mode magnitude
```

The first PPO command writes
`models/ppo_alns_n20_reward_alns_5310.pt`; the GNN command writes
`models/gnn_ppo_alns_n20_reward_magnitude.pt`. Training logs and test outputs
also include the same reward token so reward experiments do not overwrite one
another.

All entry points also accept `--params PATH` and `--tag NAME`. A tag selects
`data/processed_NAME`, writes to `output/METHOD_NAME`, and adds `_NAME` to the
checkpoint filename. PPO checkpoints store and verify the preprocessing hash,
fleet, size, and tag. Older checkpoints without this metadata must be
retrained before testing.

Results are written under:

```text
output/{exact,alns,ppo_alns,gnn_ppo_alns}/n{size}/
```

Trained PPO checkpoints are written under `models/`.

## Plotting one result

`scripts/plot_result.py` accepts one result JSON from any of the four methods.
It infers the matching processed/raw test instance and writes two route maps
next to the JSON:

```bash
python scripts/plot_result.py output/exact/n5/test_n5_000.json
```

The example creates `output/exact/n5/test_n5_000_map.svg` and
`output/exact/n5/test_n5_000_map.html`, an interactive OpenStreetMap (Leaflet)
view of the same routes. Use `--labels` to show node and administrative-dong
labels on the static map. PNG output remains available with `--format png`.

## Summarizing the experiments

`scripts/summarize_results.py` reads what is already under `output/` and
writes two tables per experiment size, straight into `output/`:

```bash
python scripts/summarize_results.py            # all sizes found
python scripts/summarize_results.py --size 10  # one size
```

- `training_time_n{size}.csv` — one row per PPO-ALNS / GNN-PPO-ALNS training
  setting: total wall-clock training time plus the CLI arguments, PPO
  configuration, processed-data metadata, and a snapshot of `params.yaml`.
  New training runs write these values to `training_metadata_reward_*.json`;
  older logs remain supported using their last recorded episode time.
- `comparison_n{size}.csv` — one row per tested method (ALNS, PPO-ALNS,
  GNN-PPO-ALNS) with the min / mean / max of the objective and of the
  inference runtime. Test arguments, selection mode, checkpoint settings, and
  problem parameters are taken from `test_metadata*.json` when available.

The tables contain no opaque run ID. The script compares the readable setting
columns: re-running it refreshes the same setting, while a different setting is
appended below the previous rows. Run the summarizer after each new experiment;
the simple model-specific training and test artifact filenames are overwritten
when the same model, reward mode, and tag are executed again.
