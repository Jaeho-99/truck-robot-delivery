"""Export the toy instances used in the experiments as JSON payloads.

The experiments rebuild instances deterministically from seeds via
src/instance.py, so these files are reference data (also the input of
src/plotting.py), not an input of the experiments.

  python data/generator.py            # export to data/instances/
  python data/generator.py --out-dir <dir>
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance    # noqa: E402

GRIDS = {
    "small4": dict(grid_x=2, grid_y=2, num_parking=3, num_trucks=2,
                   num_robots=2, num_parking_copies=2,
                   parking_avoid_radius=4.5),
    "small6": dict(grid_x=3, grid_y=2, num_parking=4, num_trucks=2,
                   num_robots=2, num_parking_copies=2,
                   parking_avoid_radius=4.5),
}
SEEDS = [1, 2, 3, 4, 5]
SCALING_SEED = 1
SCALING_SIZES = [5, 10, 15, 20, 25, 50, 100]


def dump(payload, out_dir, name):
    path = os.path.join(out_dir, f"instance_{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Export toy instance JSONs")
    ap.add_argument("--out-dir",
                    default=os.path.join(REPO_ROOT, "data", "instances"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for cfg, kwargs in GRIDS.items():
        for seed in SEEDS:
            inst = instance.build_grid_instance(seed=seed, beta_robot=3,
                                                **kwargs)
            dump(instance.instance_payload(inst), args.out_dir,
                 f"{cfg}_s{seed}")

    master = instance.build_master(SCALING_SEED)
    for n in SCALING_SIZES:
        inst, _, _ = instance.build_scaling_instance(master, n)
        dump(instance.instance_payload(inst), args.out_dir,
             f"n{n}_s{SCALING_SEED}")


if __name__ == "__main__":
    main()
