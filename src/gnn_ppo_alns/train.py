"""Train GNN-PPO-ALNS from precomputed train instances."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import yaml

from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT
from gnn_ppo_alns.alns import DirectoryInstanceProvider
from gnn_ppo_alns.gnn import compute_norms
from gnn_ppo_alns.ppo import (PPOConfig, REWARD_MODES,
                              reward_artifact_token, train)


def _parser():
    parser = argparse.ArgumentParser(
        description="Train GNN-PPO-ALNS using data/processed*/train")
    parser.add_argument("--size", type=int, choices=(5, 10, 20, 50, 100),
                        required=True)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--train-count", type=int, default=250)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--search-iterations", type=int, default=100)
    parser.add_argument(
        "--reward-mode", choices=REWARD_MODES, default="magnitude",
        help="training reward and checkpoint filename category")
    parser.add_argument(
        "--eps-uniform", type=float, default=0.1,
        help="uniform exploration mixed into the policy during training "
             "and stored for testing (default: 0.1)")
    parser.add_argument("--norm-instances", type=int, default=20)
    return parser


def main():
    args = _parser().parse_args()
    if (args.train_count <= 0 or args.total_steps <= 0
            or args.search_iterations <= 0 or args.norm_instances <= 0):
        raise ValueError("training counts and iteration budgets must be positive")
    if not 0.0 <= args.eps_uniform <= 1.0:
        raise ValueError("--eps-uniform must be between 0 and 1")
    cfg = PPOConfig(total_steps=args.total_steps,
                    train_count=args.train_count,
                    search_iterations=args.search_iterations,
                    eps_uniform=args.eps_uniform,
                    reward_mode=args.reward_mode,
                    use_graph=True)
    provider = DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag,
        train_count=args.train_count, split="train")
    params_path = args.params.expanduser().resolve()
    with params_path.open("r", encoding="utf-8") as f:
        parameter_settings = yaml.safe_load(f)
    norms = compute_norms([
        provider._params(path)
        for path in provider.train[:args.norm_instances]
    ])
    suffix = "" if args.tag is None else f"_{args.tag}"
    reward_token = reward_artifact_token(args.reward_mode)
    model_path = (REPO_ROOT / "models"
                  / f"gnn_ppo_alns_n{args.size}_{reward_token}{suffix}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logs, updates = [], []
    started = time.perf_counter()
    train(cfg, provider, model_path, norms, logs, updates)
    train_time_s = time.perf_counter() - started

    method_dir = ("gnn_ppo_alns" if args.tag is None
                  else f"gnn_ppo_alns_{args.tag}")
    out_dir = REPO_ROOT / "output" / method_dir / f"n{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"train_{reward_token}.csv").open(
            "w", newline="", encoding="utf-8") as f:
        if not logs:
            raise RuntimeError("training completed without an episode log")
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    with (out_dir / f"updates_{reward_token}.json").open(
            "w", encoding="utf-8") as f:
        json.dump(updates, f, indent=2)
    metadata = {
        "schema_version": 1,
        "model": "gnn_ppo_alns",
        "size": args.size,
        "tag": args.tag,
        "reward_mode": args.reward_mode,
        "completed_at": datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "train_time_s": train_time_s,
        "checkpoint": str(model_path.relative_to(REPO_ROOT)),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "ppo_config": cfg.to_dict(),
        "parameters": parameter_settings,
        "checkpoint_metadata": provider.checkpoint_metadata,
    }
    with (out_dir / f"training_metadata_{reward_token}.json").open(
            "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
