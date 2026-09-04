"""Test a saved PPO-ALNS checkpoint on precomputed instances."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import yaml

from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT
from ppo_alns.alns import DirectoryInstanceProvider
from ppo_alns.ppo import (REWARD_MODES, evaluate_instance, load_model,
                          reward_artifact_token)


def _parser():
    parser = argparse.ArgumentParser(
        description="Test PPO-ALNS using data/processed*/test")
    parser.add_argument("--size", type=int, choices=(5, 10, 20, 50, 100),
                        required=True)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--reward-mode", choices=REWARD_MODES, default="magnitude",
        help="reward category used to train the checkpoint")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--argmax", action="store_true",
                        help="use deterministic argmax instead of the "
                             "checkpoint's epsilon-mixed sampling policy")
    return parser


def main():
    args = _parser().parse_args()
    if args.seeds <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("--seeds and --limit must be positive")
    provider = DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag, split="test")
    params_path = args.params.expanduser().resolve()
    with params_path.open("r", encoding="utf-8") as f:
        parameter_settings = yaml.safe_load(f)
    suffix = "" if args.tag is None else f"_{args.tag}"
    reward_token = reward_artifact_token(args.reward_mode)
    model_path = (args.checkpoint or REPO_ROOT / "models"
                  / f"ppo_alns_n{args.size}_{reward_token}{suffix}.pt")
    model, cfg, _ = load_model(
        model_path, provider.checkpoint_metadata)
    if cfg.reward_mode != args.reward_mode:
        raise ValueError(
            f"selected reward mode {args.reward_mode!r} does not match "
            f"checkpoint mode {cfg.reward_mode!r}")
    method_dir = ("ppo_alns" if args.tag is None
                  else f"ppo_alns_{args.tag}")
    out_dir = REPO_ROOT / "output" / method_dir / f"n{args.size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = provider.test_set()
    if args.limit is not None:
        cases = cases[:args.limit]
    rows = []
    for instance_id, params in cases:
        for seed in range(args.seeds):
            solution, stats = evaluate_instance(model, cfg, None, params,
                                                seed=seed,
                                                sample=not args.argmax)
            with (out_dir / f"{instance_id}_{reward_token}_s{seed}.json").open(
                    "w", encoding="utf-8") as f:
                json.dump({"instance_id": instance_id, "seed": seed,
                           "reward_mode": cfg.reward_mode,
                           "stats": stats, "routes": solution.routes}, f,
                          indent=2)
            rows.append({"instance_id": instance_id, "seed": seed,
                         "reward_mode": cfg.reward_mode,
                         "obj": stats["best_cost"],
                         "runtime_s": round(stats["runtime_s"], 3),
                         "improve_pct": stats["improve_pct"]})
    with (out_dir / f"test_{reward_token}.csv").open(
            "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": 1,
        "model": "ppo_alns",
        "size": args.size,
        "tag": args.tag,
        "reward_mode": cfg.reward_mode,
        "completed_at": datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "checkpoint": str(model_path.resolve()),
        "selection_mode": "argmax" if args.argmax else "sampling_epsilon",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "ppo_config": cfg.to_dict(),
        "parameters": parameter_settings,
        "checkpoint_metadata": provider.checkpoint_metadata,
    }
    with (out_dir / f"test_metadata_{reward_token}.json").open(
            "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
