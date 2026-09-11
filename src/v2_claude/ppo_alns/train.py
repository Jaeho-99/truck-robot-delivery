"""Train PPO-ALNS from precomputed train instances."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2_claude.common.artifacts import (reserve_artifacts, validate_data_tag, validate_run_label, write_csv,
                              write_json)
from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT
from v2_claude.common.runtime import report_training_runtime
from v2_claude.ppo_alns.alns import DirectoryInstanceProvider
from v2_claude.ppo_alns.ppo import (PPOConfig, REWARD_MODES,
                                    reward_artifact_token, train)


def _parser():
    parser = argparse.ArgumentParser(
        description="Train PPO-ALNS using data/processed*/train")
    parser.add_argument("--size", type=int, choices=(5, 10, 20, 50, 100),
                        required=True)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default=PPOConfig().device,
                        help="training device; CUDA errors do not fall back to CPU")
    parser.add_argument("--env-backend", choices=("serial", "process"),
                        default="serial", help="process uses one CPU worker per env")
    parser.add_argument("--observation-codec", choices=("direct", "numpy"),
                        default="direct", help="CPU observation transport for process backend")
    parser.add_argument("--run-label", help="separate artifacts without changing PPOConfig")
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
    return parser


def main():
    main_started = time.perf_counter()
    args = _parser().parse_args()
    args.tag = validate_data_tag(args.tag)
    if (args.train_count <= 0 or args.total_steps <= 0
            or args.search_iterations <= 0):
        raise ValueError("training counts and iteration budgets must be positive")
    if not 0.0 <= args.eps_uniform <= 1.0:
        raise ValueError("--eps-uniform must be between 0 and 1")
    cfg = PPOConfig(total_steps=args.total_steps,
                    train_count=args.train_count,
                    search_iterations=args.search_iterations,
                    eps_uniform=args.eps_uniform,
                    reward_mode=args.reward_mode,
                    use_graph=False,
                    device=args.device)
    if cfg.n_envs != 10:
        raise ValueError("the training protocol requires n_envs=10")
    if cfg.n_updates < 1:
        raise ValueError("--total-steps must cover at least one complete rollout "
                         f"({cfg.t_rollout * cfg.n_envs} environment steps)")
    args.run_label = validate_run_label(args.run_label)
    suffix = "" if args.tag is None else f"_{args.tag}"
    run_suffix = "" if args.run_label is None else f"_run-{args.run_label}"
    reward_token = reward_artifact_token(args.reward_mode)
    model_path = (REPO_ROOT / "models" / "v2_claude"
                  / f"ppo_alns_n{args.size}_{reward_token}{suffix}{run_suffix}.pt")
    method_dir = ("ppo_alns" if args.tag is None
                  else f"ppo_alns_{args.tag}")
    out_dir = REPO_ROOT / "output" / "v2_claude" / method_dir / f"n{args.size}"
    if args.run_label is not None:
        out_dir = out_dir / "runs" / args.run_label
    log_path = out_dir / f"train_{reward_token}.csv"
    updates_path = out_dir / f"updates_{reward_token}.json"
    metadata_path = out_dir / f"training_metadata_{reward_token}.json"
    effective_steps = cfg.n_updates * cfg.t_rollout * cfg.n_envs
    worker_count = cfg.n_envs if args.env_backend == "process" else 0
    print(f"[train] requested_steps={cfg.total_steps} effective_steps={effective_steps} "
          f"n_updates={cfg.n_updates} device={cfg.device} "
          f"env_backend={args.env_backend} worker_count={worker_count} "
          f"run_label={args.run_label!r}", flush=True)
    # Reserve before CUDA setup, data loading or the first checkpoint write.
    # Completion metadata is written last, and never inherited from an old run.
    with reserve_artifacts([model_path, log_path, updates_path, metadata_path]):
        runtime_started = time.perf_counter()
        runtime = report_training_runtime(cfg.device)
        runtime_init_seconds = time.perf_counter() - runtime_started
        data_started = time.perf_counter()
        provider = DirectoryInstanceProvider(
            args.size, params_path=args.params, tag=args.tag,
            train_count=args.train_count, split="train")
        params_path = args.params.expanduser().resolve()
        with params_path.open("r", encoding="utf-8") as stream:
            parameter_settings = yaml.safe_load(stream)
        data_setup_seconds = time.perf_counter() - data_started
        logs, updates, training_stats = [], [], {}
        started = time.perf_counter()
        train(cfg, provider, model_path, {}, logs, updates,
              env_backend=args.env_backend,
              observation_codec=args.observation_codec,
              training_stats=training_stats)
        train_time_s = time.perf_counter() - started
        if not logs:
            raise RuntimeError("training completed without an episode log")
        write_csv(log_path, logs)
        write_json(updates_path, updates)
        metadata = {
            "schema_version": 2, "status": "completed",
            "model": "ppo_alns", "variant": "v2_claude",
            "size": args.size, "tag": args.tag, "reward_mode": args.reward_mode,
            "device": cfg.device, "env_backend": args.env_backend,
            "worker_count": worker_count, "run_label": args.run_label,
            "observation_codec": args.observation_codec,
            "requested_steps": cfg.total_steps, "effective_steps": effective_steps,
            "n_updates": cfg.n_updates, "replay_benchmark": False,
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "train_time_s": train_time_s,
            "main_wall_seconds": time.perf_counter() - main_started,
            "main_wall_scope": "main entry through result publication; excludes interpreter/import and final metadata write",
            "runtime_init_seconds": runtime_init_seconds,
            "data_setup_seconds": data_setup_seconds,
            "checkpoint": str(model_path.relative_to(REPO_ROOT)),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "ppo_config": cfg.to_dict(), "runtime": runtime,
            "training_stats": training_stats, "parameters": parameter_settings,
            "checkpoint_metadata": provider.checkpoint_metadata,
        }
        write_json(metadata_path, metadata)
        print(f"[complete] checkpoint={model_path} metadata={metadata_path}", flush=True)


if __name__ == "__main__":
    main()
