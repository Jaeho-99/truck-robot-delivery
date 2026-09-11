"""Train graph-free PPO-ALNS on precomputed instances.

Workflow: parse unchanged experiment defaults, reserve output files, validate
the requested device, load train instances, and run the PPO loop in ``ppo.py``.
CSV logs and completion metadata are published only after training succeeds.

Example from the repository root::

    python src/ppo_alns/train.py --size 20 --run-label example

Use ``--device cpu`` to request CPU training explicitly. Reusing a run label
does not overwrite existing models or result files.
"""

import argparse
import csv
import json
import os
import platform
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

# Direct execution adds src/ so package imports work from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT
from ppo_alns.alns import DirectoryInstanceProvider
from ppo_alns.ppo import (
    REWARD_MODES,
    PPOConfig,
    reward_artifact_token,
    train,
)

# Output publication is local to this entry point.


def validate_data_tag(tag):
    """Apply the providers' existing tag grammar before constructing paths."""
    if tag is not None and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", tag
    ):
        raise ValueError("invalid --tag; use letters, digits, '.', '_' or '-'")
    return tag


def validate_run_label(label):
    if label is None:
        return None
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", label
    ) or label.upper().split(".")[0] in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise ValueError(
            "--run-label must be 1-80 ASCII letters/digits/_/-, "
            "start with a letter/digit, and not be a Windows device name"
        )
    return label


@contextmanager
def reserve_artifacts(paths):
    """Fail before expensive work if any target exists or another run owns it."""
    targets = sorted(
        {Path(path).expanduser().resolve() for path in paths}, key=str
    )
    if not targets:
        raise ValueError("at least one artifact target is required")
    locks = []
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite {target}. Use a new --run-label."
                )
            lock = target.with_name(f".{target.name}.lock")
            try:
                handle = lock.open("x", encoding="utf-8")
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Output reserved: {lock}. Choose a new --run-label; only remove "
                    "a stale lock after verifying its recorded PID has no active run."
                ) from exc
            locks.append(lock)
            with handle:
                json.dump({"pid": os.getpid(), "target": str(target)}, handle)
        for target in targets:
            if target.exists():
                raise FileExistsError(
                    f"Artifact appeared during reservation: {target}"
                )
        yield
    finally:
        for lock in reversed(locks):
            lock.unlink(missing_ok=True)


@contextmanager
def atomic_open(path, mode="w", *, encoding="utf-8", newline=None):
    """Write beside the destination, then replace it only on successful close.

    Intended replacement is allowed (periodic checkpoints within a reserved run).
    An exception leaves an existing destination untouched and removes only this
    call's temporary file. This is per-file atomicity, not a multi-file transaction.
    """
    if mode not in {"w", "wb"}:
        raise ValueError("atomic_open only supports w or wb")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary)
    try:
        options = (
            {} if "b" in mode else {"encoding": encoding, "newline": newline}
        )
        with os.fdopen(fd, mode, **options) as handle:
            fd = None
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    with atomic_open(path) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to publish an empty result CSV")
    with atomic_open(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Device validation runs inside main(), never during worker imports.


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def report_training_runtime(device: str) -> dict[str, object]:
    """Validate the requested device and report it before expensive setup.

    Call from a training entry point, never at module import or worker startup.
    CPU runs do not query CUDA devices. CUDA validation queries the driver and
    device properties but does not launch a benchmark or change the device.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError(f"unsupported training device: {device!r}")

    import torch

    runtime: dict[str, object] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_geometric_version": _package_version("torch-geometric"),
        "numpy_version": _package_version("numpy"),
        "torch_cuda_version": torch.version.cuda,
        "device": device,
        "cuda_checked": device == "cuda",
        "cuda_available": None,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    print(f"[runtime] python={runtime['python_executable']}", flush=True)
    print(
        f"[runtime] Python={runtime['python_version']} "
        f"platform={runtime['platform']}",
        flush=True,
    )
    print(
        f"[runtime] torch={runtime['torch_version']} "
        f"PyG={runtime['torch_geometric_version']} "
        f"NumPy={runtime['numpy_version']} "
        f"torch CUDA build={runtime['torch_cuda_version']}",
        flush=True,
    )

    if device == "cuda":
        guidance = (
            f"Requested --device cuda with Python {sys.executable!r}, "
            f"PyTorch {torch.__version__}, and torch.version.cuda="
            f"{torch.version.cuda!r}. Verify that this interpreter uses the "
            "Windows CUDA-enabled PyTorch environment and a compatible NVIDIA "
            "driver. Installing the CUDA Toolkit alone does not enable CUDA "
            "in a CPU-only PyTorch build. Use --device cpu for an explicit CPU "
            "run; no automatic CPU fallback was applied."
        )
        if torch.version.cuda is None:
            raise RuntimeError(
                "This PyTorch build has no CUDA support. " + guidance
            )
        try:
            runtime["cuda_available"] = torch.cuda.is_available()
            if not runtime["cuda_available"]:
                raise RuntimeError("torch.cuda.is_available() returned False")
            device_index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device_index)
            runtime.update(
                {
                    "cuda_device_index": device_index,
                    "cuda_device_name": properties.name,
                    "cuda_device_capability": [
                        properties.major,
                        properties.minor,
                    ],
                    "cuda_device_memory_bytes": properties.total_memory,
                    "torch_cuda_arch_list": torch.cuda.get_arch_list(),
                }
            )
        except (RuntimeError, OSError, AssertionError) as exc:
            raise RuntimeError(
                f"CUDA device initialization failed: {exc}. {guidance}"
            ) from exc
        print(
            f"[runtime] device=cuda:{runtime['cuda_device_index']} "
            f"GPU={runtime['cuda_device_name']} "
            f"compute capability={properties.major}.{properties.minor}",
            flush=True,
        )
    else:
        print("[runtime] device=cpu (CUDA devices not queried)", flush=True)

    return runtime


# Command-line configuration and the training workflow.


def _parser():
    """Define the training CLI without loading data or initializing CUDA."""
    parser = argparse.ArgumentParser(
        description="Train PPO-ALNS using data/processed*/train"
    )
    parser.add_argument(
        "--size", type=int, choices=(5, 10, 20, 50, 100), required=True
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=PPOConfig().device,
        help="training device; CUDA errors do not fall back to CPU",
    )
    parser.add_argument(
        "--env-backend",
        choices=("serial", "process"),
        default="serial",
        help="process uses one CPU worker per env",
    )
    parser.add_argument(
        "--observation-codec",
        choices=("direct", "numpy"),
        default="direct",
        help="CPU observation transport for process backend",
    )
    parser.add_argument(
        "--run-label", help="separate artifacts without changing PPOConfig"
    )
    parser.add_argument("--train-count", type=int, default=250)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--search-iterations", type=int, default=100)
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="magnitude",
        help="training reward and checkpoint filename category",
    )
    parser.add_argument(
        "--eps-uniform",
        type=float,
        default=0.1,
        help="uniform exploration mixed into the policy during training "
        "and stored for testing (default: 0.1)",
    )
    return parser


def main():
    """Train with explicit device selection and publish the reserved files."""
    main_started = time.perf_counter()
    args = _parser().parse_args()
    args.tag = validate_data_tag(args.tag)
    if (
        args.train_count <= 0
        or args.total_steps <= 0
        or args.search_iterations <= 0
    ):
        raise ValueError(
            "training counts and iteration budgets must be positive"
        )
    if not 0.0 <= args.eps_uniform <= 1.0:
        raise ValueError("--eps-uniform must be between 0 and 1")
    cfg = PPOConfig(
        total_steps=args.total_steps,
        train_count=args.train_count,
        search_iterations=args.search_iterations,
        eps_uniform=args.eps_uniform,
        reward_mode=args.reward_mode,
        use_graph=False,
        device=args.device,
    )
    if cfg.n_envs != 10:
        raise ValueError("the training protocol requires n_envs=10")
    if cfg.n_updates < 1:
        raise ValueError(
            "--total-steps must cover at least one complete rollout "
            f"({cfg.t_rollout * cfg.n_envs} environment steps)"
        )
    args.run_label = validate_run_label(args.run_label)
    suffix = "" if args.tag is None else f"_{args.tag}"
    run_suffix = "" if args.run_label is None else f"_run-{args.run_label}"
    reward_token = reward_artifact_token(args.reward_mode)
    model_path = (
        REPO_ROOT
        / "models"
        / f"ppo_alns_n{args.size}_{reward_token}{suffix}{run_suffix}.pt"
    )
    method_dir = "ppo_alns" if args.tag is None else f"ppo_alns_{args.tag}"
    out_dir = REPO_ROOT / "output" / method_dir / f"n{args.size}"
    if args.run_label is not None:
        out_dir = out_dir / "runs" / args.run_label
    log_path = out_dir / f"train_{reward_token}.csv"
    updates_path = out_dir / f"updates_{reward_token}.json"
    metadata_path = out_dir / f"training_metadata_{reward_token}.json"
    effective_steps = cfg.n_updates * cfg.t_rollout * cfg.n_envs
    worker_count = cfg.n_envs if args.env_backend == "process" else 0
    print(
        f"[train] requested_steps={cfg.total_steps} effective_steps={effective_steps} "
        f"n_updates={cfg.n_updates} device={cfg.device} "
        f"env_backend={args.env_backend} worker_count={worker_count} "
        f"run_label={args.run_label!r}",
        flush=True,
    )
    # Reserve before CUDA setup, data loading or the first checkpoint write.
    # Completion metadata is written last, and never inherited from an old run.
    with reserve_artifacts(
        [model_path, log_path, updates_path, metadata_path]
    ):
        runtime_started = time.perf_counter()
        runtime = report_training_runtime(cfg.device)
        runtime_init_seconds = time.perf_counter() - runtime_started
        data_started = time.perf_counter()
        provider = DirectoryInstanceProvider(
            args.size,
            params_path=args.params,
            tag=args.tag,
            train_count=args.train_count,
            split="train",
        )
        params_path = args.params.expanduser().resolve()
        with params_path.open("r", encoding="utf-8") as stream:
            parameter_settings = yaml.safe_load(stream)
        data_setup_seconds = time.perf_counter() - data_started
        logs, updates, training_stats = [], [], {}
        started = time.perf_counter()
        train(
            cfg,
            provider,
            model_path,
            {},
            logs,
            updates,
            env_backend=args.env_backend,
            observation_codec=args.observation_codec,
            training_stats=training_stats,
        )
        train_time_s = time.perf_counter() - started
        if not logs:
            raise RuntimeError("training completed without an episode log")
        write_csv(log_path, logs)
        write_json(updates_path, updates)
        metadata = {
            "schema_version": 2,
            "status": "completed",
            "model": "ppo_alns",
            "size": args.size,
            "tag": args.tag,
            "reward_mode": args.reward_mode,
            "device": cfg.device,
            "env_backend": args.env_backend,
            "worker_count": worker_count,
            "run_label": args.run_label,
            "observation_codec": args.observation_codec,
            "requested_steps": cfg.total_steps,
            "effective_steps": effective_steps,
            "n_updates": cfg.n_updates,
            "replay_benchmark": False,
            "completed_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
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
            "ppo_config": cfg.to_dict(),
            "runtime": runtime,
            "training_stats": training_stats,
            "parameters": parameter_settings,
            "checkpoint_metadata": provider.checkpoint_metadata,
        }
        write_json(metadata_path, metadata)
        print(
            f"[complete] checkpoint={model_path} metadata={metadata_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
