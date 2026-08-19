"""Learning-curve plot (DR-ALNS Figure 2 format).

  .venv/bin/python experiments/plot_training.py \
      --log models/gnn_dqn_n20_train_log.csv

x-axis: cumulative training steps; line: rolling mean episode reward
(window 100 episodes, as logged by the trainer); band: +- rolling std.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main():
    ap = argparse.ArgumentParser(description="training curve plot")
    ap.add_argument("--log", required=True,
                    help="*_train_log.csv written by train_gnn_dqn.py")
    ap.add_argument("--out", default=None,
                    help="default: <log>.png next to the CSV")
    args = ap.parse_args()

    with open(args.log, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty log: {args.log}")
    steps = [int(r["step"]) for r in rows]
    mean = [float(r["reward_roll_mean"]) for r in rows]
    std = [float(r["reward_roll_std"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, mean, lw=1.5, label="mean episode reward "
            "(rolling 100)")
    ax.fill_between(steps, [m - s for m, s in zip(mean, std)],
                    [m + s for m, s in zip(mean, std)],
                    alpha=0.25, linewidth=0, label="+- std")
    ax.set_xlabel("training steps (search iterations)")
    ax.set_ylabel("episode reward")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = args.out or args.log.rsplit(".", 1)[0] + ".png"
    fig.savefig(out, dpi=150)
    print(f"[plot] -> {out}")


if __name__ == "__main__":
    main()
