"""Test GNN-PPO-ALNS on precomputed instances with optional CPU workers.

Each instance/seed pair runs independently. Results retain input order for
both serial and process evaluation, and a completed run publishes the same
route JSON, aggregate CSV, and metadata fields in either backend.

Example from the repository root::

    python src/gnn_ppo_alns/test.py --size 20 --workers 2 --run-label example_01

Use ``--checkpoint`` to select a model explicitly; ``--run-label`` affects
output filenames only and does not change the checkpoint or policy settings.
"""

import argparse
import csv
import importlib
import json
import multiprocessing
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from multiprocessing.connection import wait
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT
from gnn_ppo_alns.ppo import REWARD_MODES


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


# Training runtime diagnostics without device work at module import time.


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


# Independent CPU policy-evaluation jobs on persistent spawn workers.
#
# The environment's episode seed and the evaluator's private action generator
# remain per-job. No model, Params, Solution, or CUDA tensor crosses a pipe.
# Results are returned in instance-major, seed-minor order, not completion order.


def _worker(worker_id, connection, package, model_path, metadata, cache_spec):
    task_id = None
    phase = "startup"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        ppo = importlib.import_module(f"{package}.ppo")
        alns = importlib.import_module(f"{package}.alns")
        model, cfg, norms = ppo.load_model(model_path, metadata, device="cpu")
        builder = (
            importlib.import_module(f"{package}.gnn").GraphBuilder(norms, cfg)
            if package == "gnn_ppo_alns"
            else None
        )
        cache = alns.InstanceCache(**cache_spec)
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "evaluation worker unexpectedly initialized CUDA"
            )
        connection.send(
            (
                "ready",
                worker_id,
                {
                    "pid": os.getpid(),
                    "executable": sys.executable,
                    "ppo_module": ppo.__file__,
                    "device": "cpu",
                    "torch_threads": torch.get_num_threads(),
                },
            )
        )
        previous_task = -1
        while True:
            phase = "receive"
            try:
                command, task_id, argument = connection.recv()
            except EOFError:
                break
            if command == "close":
                break
            if (
                command != "evaluate"
                or type(task_id) is not int
                or task_id <= previous_task
            ):
                raise RuntimeError("invalid evaluation task protocol")
            previous_task = task_id
            reference, seed, sample = argument
            phase = "load"
            load_started = time.perf_counter()
            params = cache.load_ref(reference)
            load_seconds = time.perf_counter() - load_started
            phase = "evaluate"
            solution, stats = ppo.evaluate_instance(
                model, cfg, builder, params, seed=seed, sample=sample
            )
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "evaluation worker unexpectedly initialized CUDA"
                )
            connection.send(
                (
                    "ok",
                    worker_id,
                    task_id,
                    {
                        "instance_id": Path(reference).stem,
                        "seed": seed,
                        "stats": stats,
                        "routes": solution.routes,
                        "load_seconds": load_seconds,
                        "cache_size": len(cache),
                    },
                )
            )
    except KeyboardInterrupt:
        pass
    except BaseException:
        try:
            connection.send(
                (
                    "error",
                    worker_id,
                    task_id,
                    phase,
                    traceback.format_exc()[-32768:],
                )
            )
        except (EOFError, OSError, KeyboardInterrupt):
            pass
    finally:
        connection.close()


def _cleanup(processes, connections, pending, timeout=5.0):
    """Shared deadlines; never join indefinitely or send behind a busy reply."""
    for worker_id, process in enumerate(processes):
        try:
            if (
                process.pid is not None
                and process.is_alive()
                and worker_id not in pending
            ):
                connections[worker_id].send(("close", None, None))
        except (OSError, ValueError, KeyboardInterrupt):
            pass
    deadline = time.perf_counter() + timeout
    for process in processes:
        try:
            if process.pid is not None:
                process.join(max(0.0, deadline - time.perf_counter()))
                if process.is_alive():
                    process.terminate()
        except (OSError, ValueError, KeyboardInterrupt):
            pass
    deadline = time.perf_counter() + timeout
    survivors = []
    for process in processes:
        try:
            if process.pid is not None:
                process.join(max(0.0, deadline - time.perf_counter()))
                if process.is_alive():
                    survivors.append(process.pid)
                else:
                    process.close()
        except (OSError, ValueError, KeyboardInterrupt):
            pass
    for connection in connections:
        try:
            connection.close()
        except OSError:
            pass
    if survivors:
        warnings.warn(
            f"Evaluation cleanup deadline exceeded; inspect PIDs {survivors}",
            RuntimeWarning,
            stacklevel=2,
        )


def _normal_path(path):
    return os.path.normcase(str(Path(path).resolve()))


def evaluate_cases(
    package,
    model,
    cfg,
    builder,
    provider,
    model_path,
    jobs,
    *,
    workers=1,
    sample=True,
    runtime_stats=None,
):
    """Evaluate (absolute instance reference, seed) jobs; workers=1 stays serial.

    All completed records are returned in input order. A worker error aborts
    the whole run, without retrying a potentially partially executed episode.
    A live worker stuck in native code is not auto-killed by a short task timeout.
    """
    if package not in {"ppo_alns", "gnn_ppo_alns"}:
        raise ValueError("unsupported policy package")
    if type(workers) is not int or not 1 <= workers <= 30:
        raise ValueError("--workers must be an integer in [1, 30]")
    jobs = list(jobs)
    if not jobs:
        raise ValueError("evaluation requires at least one job")
    for reference, seed in jobs:
        if (
            not Path(reference).is_absolute()
            or type(seed) is not int
            or seed < 0
        ):
            raise ValueError(
                "jobs require an absolute instance path and nonnegative seed"
            )
    timings = runtime_stats if runtime_stats is not None else {}
    started = time.perf_counter()
    ppo = importlib.import_module(f"{package}.ppo")
    if workers == 1:
        results = []
        for index, (reference, seed) in enumerate(jobs):
            load_started = time.perf_counter()
            params = provider._params(reference)
            load_seconds = time.perf_counter() - load_started
            solution, stats = ppo.evaluate_instance(
                model, cfg, builder, params, seed=seed, sample=sample
            )
            results.append(
                {
                    "instance_id": Path(reference).stem,
                    "seed": seed,
                    "stats": stats,
                    "routes": solution.routes,
                    "load_seconds": load_seconds,
                    "cache_size": len(provider._cache),
                }
            )
            print(
                f"[test {index + 1}/{len(jobs)}] {params.instance_id} seed={seed} "
                f"runtime={stats['runtime_s']:.2f}s",
                flush=True,
            )
        timings.update(
            {
                "workers_requested": workers,
                "worker_count": 0,
                "backend": "serial",
                "startup_seconds": 0.0,
                "evaluation_wall_seconds": time.perf_counter() - started,
            }
        )
        return results

    worker_count = min(workers, len(jobs))
    context = multiprocessing.get_context("spawn")
    processes, connections = [], []
    pending = {}
    results = [None] * len(jobs)
    runtime = {}

    def receive(worker_id):
        try:
            message = connections[worker_id].recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"evaluation worker {worker_id} pipe failed, "
                f"exitcode={processes[worker_id].exitcode}"
            ) from exc
        if (
            not isinstance(message, tuple)
            or len(message) < 2
            or message[1] != worker_id
        ):
            raise RuntimeError(
                f"invalid reply from evaluation worker {worker_id}"
            )
        if message[0] == "error":
            if len(message) != 5:
                raise RuntimeError(f"malformed worker {worker_id} error")
            raise RuntimeError(
                f"evaluation worker {worker_id} task={message[2]} "
                f"phase={message[3]}:\n{message[4]}"
            )
        return message

    def wait_ready():
        objects = [connections[i] for i in pending] + [
            process.sentinel for process in processes
        ]
        ready = wait(objects, timeout=5.0)
        return [i for i in pending if connections[i] in ready]

    def check_processes():
        for i, process in enumerate(processes):
            if process.exitcode is not None:
                # An ERROR can become readable after wait() took its snapshot.
                # Drain it before replacing its traceback with a generic death.
                if i in pending and connections[i].poll():
                    receive(i)
                raise RuntimeError(
                    f"evaluation worker {i} pid={process.pid} died "
                    f"exitcode={process.exitcode}, task={pending.get(i)}"
                )

    try:
        for worker_id in range(worker_count):
            parent, child = context.Pipe(duplex=True)
            connections.append(parent)
            process = context.Process(
                target=_worker,
                args=(
                    worker_id,
                    child,
                    package,
                    str(Path(model_path).resolve()),
                    provider.checkpoint_metadata,
                    provider.worker_spec(),
                ),
                name=f"policy-evaluation-{worker_id}",
                daemon=False,
            )
            processes.append(process)
            pending[worker_id] = None
            try:
                process.start()
            finally:
                child.close()
        deadline = time.perf_counter() + 180.0
        last_report = started
        while pending:
            for worker_id in wait_ready():
                message = receive(worker_id)
                if len(message) != 3 or message[0] != "ready":
                    raise RuntimeError("expected evaluation READY reply")
                info = message[2]
                if (
                    info["pid"] != processes[worker_id].pid
                    or _normal_path(info["executable"])
                    != _normal_path(sys.executable)
                    or _normal_path(info["ppo_module"])
                    != _normal_path(ppo.__file__)
                    or info["device"] != "cpu"
                ):
                    raise RuntimeError(
                        f"evaluation worker {worker_id} environment mismatch"
                    )
                runtime[worker_id] = info
                del pending[worker_id]
            check_processes()
            now = time.perf_counter()
            if pending and now >= deadline:
                raise TimeoutError(
                    f"evaluation startup timed out: workers {sorted(pending)}"
                )
            if pending and now - last_report >= 30.0:
                print(
                    f"[test startup] waiting={sorted(pending)} elapsed={now - started:.1f}s",
                    flush=True,
                )
                last_report = now
        timings.update(
            {
                "workers_requested": workers,
                "worker_count": worker_count,
                "backend": "process",
                "workers": runtime,
                "startup_seconds": time.perf_counter() - started,
            }
        )
        next_task = 0

        def dispatch(worker_id):
            nonlocal next_task
            task_id = next_task
            reference, seed = jobs[task_id]
            pending[worker_id] = task_id
            connections[worker_id].send(
                ("evaluate", task_id, (reference, seed, sample))
            )
            next_task += 1

        for worker_id in range(worker_count):
            dispatch(worker_id)
        completed = 0
        last_report = time.perf_counter()
        while pending:
            for worker_id in wait_ready():
                message = receive(worker_id)
                task_id = pending[worker_id]
                if (
                    len(message) != 4
                    or message[0] != "ok"
                    or message[2] != task_id
                ):
                    raise RuntimeError(
                        f"evaluation worker {worker_id} task ID mismatch"
                    )
                record = message[3]
                reference, seed = jobs[task_id]
                if (
                    record["instance_id"] != Path(reference).stem
                    or record["seed"] != seed
                ):
                    raise RuntimeError(
                        f"evaluation worker {worker_id} returned the wrong case"
                    )
                results[task_id] = record
                del pending[worker_id]
                completed += 1
                print(
                    f"[test {completed}/{len(jobs)}] {record['instance_id']} "
                    f"seed={seed} worker={worker_id} "
                    f"runtime={record['stats']['runtime_s']:.2f}s",
                    flush=True,
                )
                if next_task < len(jobs):
                    dispatch(worker_id)
            check_processes()
            now = time.perf_counter()
            if pending and now - last_report >= 30.0:
                print(
                    f"[test waiting] workers={sorted(pending)} completed={completed}/"
                    f"{len(jobs)} elapsed={now - started:.1f}s",
                    flush=True,
                )
                last_report = now
        return results
    finally:
        close_started = time.perf_counter()
        _cleanup(processes, connections, pending)
        timings["close_seconds"] = time.perf_counter() - close_started
        timings["evaluation_wall_seconds"] = time.perf_counter() - started


def run_evaluation_cli(package, args, *, main_started=None):
    """Load the checkpoint, evaluate ordered cases, and publish results."""

    main_started = (
        time.perf_counter() if main_started is None else main_started
    )
    if (
        args.seeds <= 0
        or (args.limit is not None and args.limit <= 0)
        or not 1 <= args.workers <= 30
    ):
        raise ValueError(
            "--seeds/--limit must be positive; --workers must be in [1, 30]"
        )
    args.run_label = validate_run_label(args.run_label)
    ppo = importlib.import_module(f"{package}.ppo")
    provider = importlib.import_module(
        f"{package}.alns"
    ).DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag, split="test"
    )
    with (
        args.params.expanduser()
        .resolve()
        .open("r", encoding="utf-8") as stream
    ):
        parameter_settings = yaml.safe_load(stream)
    suffix = "" if args.tag is None else f"_{args.tag}"
    reward_token = ppo.reward_artifact_token(args.reward_mode)
    model_path = (
        args.checkpoint
        or REPO_ROOT
        / "models"
        / f"{package}_n{args.size}_{reward_token}{suffix}.pt"
    )
    # A run label labels outputs only: choosing a trained model stays explicit.
    method_dir = package if args.tag is None else f"{package}_{args.tag}"
    out_dir = REPO_ROOT / "output" / method_dir / f"n{args.size}"
    if args.run_label is not None:
        out_dir = out_dir / "runs" / args.run_label
    paths = (
        provider.test if args.limit is None else provider.test[: args.limit]
    )
    jobs = [
        (str(path.resolve()), seed)
        for path in paths
        for seed in range(args.seeds)
    ]
    record_paths = [
        out_dir / f"{Path(reference).stem}_{reward_token}_s{seed}.json"
        for reference, seed in jobs
    ]
    csv_path = out_dir / f"test_{reward_token}.csv"
    metadata_path = out_dir / f"test_metadata_{reward_token}.json"
    with reserve_artifacts([*record_paths, csv_path, metadata_path]):
        runtime = report_training_runtime("cpu")
        model_started = time.perf_counter()
        model, cfg, norms = ppo.load_model(
            model_path, provider.checkpoint_metadata, device="cpu"
        )
        if cfg.reward_mode != args.reward_mode:
            raise ValueError(
                f"selected reward mode {args.reward_mode!r} does not match "
                f"checkpoint mode {cfg.reward_mode!r}"
            )
        builder = (
            importlib.import_module(f"{package}.gnn").GraphBuilder(norms, cfg)
            if package == "gnn_ppo_alns"
            else None
        )
        model_load_seconds = time.perf_counter() - model_started
        evaluation_stats = {}
        results = evaluate_cases(
            package,
            model,
            cfg,
            builder,
            provider,
            model_path,
            jobs,
            workers=args.workers,
            sample=not args.argmax,
            runtime_stats=evaluation_stats,
        )
        rows = []
        for record, destination in zip(results, record_paths):
            stats = record["stats"]
            write_json(
                destination,
                {
                    "instance_id": record["instance_id"],
                    "seed": record["seed"],
                    "reward_mode": cfg.reward_mode,
                    "stats": stats,
                    "routes": record["routes"],
                },
            )
            rows.append(
                {
                    "instance_id": record["instance_id"],
                    "seed": record["seed"],
                    "reward_mode": cfg.reward_mode,
                    "obj": stats["best_cost"],
                    "runtime_s": round(stats["runtime_s"], 3),
                    "improve_pct": stats["improve_pct"],
                }
            )
        write_csv(csv_path, rows)
        metadata = {
            "schema_version": 2,
            "status": "completed",
            "model": package,
            "size": args.size,
            "tag": args.tag,
            "reward_mode": cfg.reward_mode,
            "device": "cpu",
            "run_label": args.run_label,
            "workers_requested": args.workers,
            "worker_count": evaluation_stats["worker_count"],
            "backend": evaluation_stats["backend"],
            "cases_completed": len(results),
            "completed_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "checkpoint": str(model_path.resolve()),
            "selection_mode": "argmax" if args.argmax else "sampling_epsilon",
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "ppo_config": cfg.to_dict(),
            "parameters": parameter_settings,
            "checkpoint_metadata": provider.checkpoint_metadata,
            "runtime": runtime,
            "evaluation_stats": evaluation_stats,
            "evaluation_wall_seconds": evaluation_stats[
                "evaluation_wall_seconds"
            ],
            "model_load_seconds": model_load_seconds,
            "main_wall_seconds": time.perf_counter() - main_started,
            "main_wall_scope": "main entry through result publication; excludes interpreter/import and final metadata write",
        }
        write_json(metadata_path, metadata)
        print(
            f"[complete] cases={len(results)} csv={csv_path} metadata={metadata_path}",
            flush=True,
        )


def _parser():
    parser = argparse.ArgumentParser(
        description="Test GNN-PPO-ALNS using data/processed*/test"
    )
    parser.add_argument(
        "--size", type=int, choices=(5, 10, 20, 50, 100), required=True
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
