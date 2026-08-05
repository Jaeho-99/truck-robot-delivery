"""Render route SVGs from saved solution JSONs.

Reads the sol_*.json files written by run_qlearning.py (compare mode),
rebuilds each instance from the embedded instance_set, and renders the
stored plot_payload with plotting.make_route_svg. Requires matplotlib
(use the anaconda interpreter if the venv lacks it):

  /opt/anaconda3/bin/python experiments/make_route_svgs.py \
      --results-dir results/main_alns [--out results/main_alns/img]
"""

import argparse
import glob
import json
import os
import re
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import instance, plotting                        # noqa: E402


def rebuild_instance(meta):
    """Instance dict from a solution JSON's embedded instance_set."""
    iset = meta["instance_set"]
    name = meta["instance"]
    if iset["type"] == "scaling":
        n = int(re.match(r"n(\d+)_s\d+", name).group(1))
        master = instance.build_master(iset["seed"])
        inst, _, _ = instance.build_scaling_instance(
            master, n,
            num_trucks=iset.get("num_trucks", 5),
            num_robots=iset.get("num_robots", 3),
            num_parking_copies=iset.get("num_parking_copies", 2),
            beta_robot=iset.get("beta_robot", 3))
        return inst
    if iset["type"] == "grid":
        seed = int(name.rsplit("_s", 1)[1])
        grid = next(g for g in iset["grids"]
                    if name.startswith(g["name"] + "_s"))
        kwargs = {k: v for k, v in grid.items() if k != "name"}
        return instance.build_grid_instance(
            seed=seed, beta_robot=iset.get("beta_robot", 3), **kwargs)
    raise ValueError(f"unknown instance_set type: {iset['type']}")


def main():
    ap = argparse.ArgumentParser(
        description="Route SVGs from saved ALNS solutions")
    ap.add_argument("--results-dir", required=True,
                    help="experiment dir containing solutions/")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <results-dir>/img)")
    ap.add_argument("--instance-svg", action="store_true",
                    help="also render one instance map per instance")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.results_dir, "img")
    os.makedirs(out_dir, exist_ok=True)
    sols = sorted(glob.glob(
        os.path.join(args.results_dir, "solutions", "sol_*.json")))
    if not sols:
        sys.exit(f"no sol_*.json under {args.results_dir}/solutions")

    inst_payloads = {}      # instance name -> temp payload path
    with tempfile.TemporaryDirectory() as tmp:
        for path in sols:
            with open(path) as f:
                meta = json.load(f)
            name = meta["instance"]
            if name not in inst_payloads:
                inst = rebuild_instance(meta)
                p = os.path.join(tmp, f"inst_{name}.json")
                with open(p, "w") as f:
                    json.dump(instance.instance_payload(inst), f)
                inst_payloads[name] = p
                if args.instance_svg:
                    plotting.make_instance_svg(p, os.path.join(
                        out_dir, f"fig_instance_{name}.svg"))
            sol_json = os.path.join(tmp, "payload.json")
            with open(sol_json, "w") as f:
                json.dump(meta["plot_payload"], f)
            out = os.path.join(out_dir, "fig_route_{}_{}_{}.svg".format(
                meta["selector"], name, meta["seed"]))
            plotting.make_route_svg(inst_payloads[name], sol_json, out)
            print(f"written: {out} (obj {meta['obj']:.4f})")


if __name__ == "__main__":
    main()
