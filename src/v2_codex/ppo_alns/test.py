"""Test PPO-ALNS on precomputed instances with optional CPU workers."""

import argparse
from pathlib import Path
import sys
import time

# Support the documented direct invocation from the repository root:
# ``python src/v2_codex/ppo_alns/test.py ...``.  In that mode Python otherwise adds
# only ``src/v2_codex/ppo_alns`` (not ``src``) to sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.params import DEFAULT_PARAMS_PATH
from v2_codex.common.policy_evaluation import run_evaluation_cli
from v2_codex.ppo_alns.ppo import REWARD_MODES


def _parser():
    parser = argparse.ArgumentParser(
        description="Test PPO-ALNS using data/processed*/test")
    parser.add_argument("--size", type=int, choices=(5, 10, 20, 50, 100), required=True)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="magnitude",
                        help="reward category used to train the checkpoint")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--argmax", action="store_true",
                        help="use deterministic argmax instead of checkpoint sampling")
    parser.add_argument("--workers", type=int, default=1,
                        help="1 keeps serial CPU evaluation; 2-30 uses spawn workers")
    parser.add_argument("--run-label", help="output label only; select models with --checkpoint")
    return parser


def main():
    started = time.perf_counter()
    run_evaluation_cli("v2_codex.ppo_alns", _parser().parse_args(), main_started=started)


if __name__ == "__main__":
    main()
