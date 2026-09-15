"""Evaluate a gnn_ppo_alns checkpoint on precomputed test instances."""

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.params import DEFAULT_PARAMS_PATH
from common.policy_evaluation import evaluate_cases as evaluate_cases
from common.policy_evaluation import run_evaluation_cli
from common.sizes import SUPPORTED_SIZES
from gnn_ppo_alns.ppo import REWARD_MODES


def _parser():
    parser = argparse.ArgumentParser(
        description="Test GNN-PPO-ALNS using data/processed*/test"
    )
    parser.add_argument(
        "--size", type=int, choices=SUPPORTED_SIZES, required=True
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="magnitude",
        help="reward category used to train the checkpoint",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--argmax",
        action="store_true",
        help="use deterministic argmax instead of checkpoint sampling",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="1 keeps serial CPU evaluation; 2-30 uses spawn workers",
    )
    parser.add_argument(
        "--run-label",
        help="output label only; select models with --checkpoint",
    )
    return parser


def main():
    started = time.perf_counter()
    run_evaluation_cli(
        "gnn_ppo_alns", _parser().parse_args(), main_started=started
    )


if __name__ == "__main__":
    main()
