"""Independent CPU policy-evaluation jobs on persistent spawn workers.

The environment's episode seed and the evaluator's private action generator
remain per-job. No model, Params, Solution, or CUDA tensor crosses a pipe.
Results are returned in instance-major, seed-minor order, not completion order.
"""

import importlib
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback
import warnings


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
        builder = (importlib.import_module(f"{package}.gnn").GraphBuilder(norms, cfg)
                   if package == "v2_codex.gnn_ppo_alns" else None)
        cache = alns.InstanceCache(**cache_spec)
        if torch.cuda.is_initialized():
            raise RuntimeError("evaluation worker unexpectedly initialized CUDA")
        connection.send(("ready", worker_id, {
            "pid": os.getpid(), "executable": sys.executable,
            "ppo_module": ppo.__file__, "device": "cpu",
            "torch_threads": torch.get_num_threads()}))
        previous_task = -1
        while True:
            phase = "receive"
            try:
                command, task_id, argument = connection.recv()
            except EOFError:
                break
            if command == "close":
                break
            if (command != "evaluate" or type(task_id) is not int
                    or task_id <= previous_task):
                raise RuntimeError("invalid evaluation task protocol")
            previous_task = task_id
            reference, seed, sample = argument
            phase = "load"
            load_started = time.perf_counter()
            params = cache.load_ref(reference)
            load_seconds = time.perf_counter() - load_started
            phase = "evaluate"
            solution, stats = ppo.evaluate_instance(
                model, cfg, builder, params, seed=seed, sample=sample)
            if torch.cuda.is_initialized():
                raise RuntimeError("evaluation worker unexpectedly initialized CUDA")
            connection.send(("ok", worker_id, task_id, {
                "instance_id": Path(reference).stem, "seed": seed,
                "stats": stats, "routes": solution.routes,
                "load_seconds": load_seconds, "cache_size": len(cache)}))
    except KeyboardInterrupt:
        pass
    except BaseException:
        try:
            connection.send(("error", worker_id, task_id, phase,
                             traceback.format_exc()[-32768:]))
        except (EOFError, OSError, KeyboardInterrupt):
            pass
    finally:
        connection.close()


def _cleanup(processes, connections, pending, timeout=5.0):
    """Shared deadlines; never join indefinitely or send behind a busy reply."""
    for worker_id, process in enumerate(processes):
        try:
            if (process.pid is not None and process.is_alive()
                    and worker_id not in pending):
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
        warnings.warn(f"Evaluation cleanup deadline exceeded; inspect PIDs {survivors}",
                      RuntimeWarning, stacklevel=2)


def _normal_path(path):
    return os.path.normcase(str(Path(path).resolve()))


def evaluate_cases(package, model, cfg, builder, provider, model_path, jobs, *,
                   workers=1, sample=True, runtime_stats=None):
    """Evaluate (absolute instance reference, seed) jobs; workers=1 stays serial.

    All completed records are returned in input order. A worker error aborts
    the whole run, without retrying a potentially partially executed episode.
    A live worker stuck in native code is not auto-killed by a short task timeout.
    """
    if package not in {"v2_codex.ppo_alns", "v2_codex.gnn_ppo_alns"}:
        raise ValueError("unsupported policy package")
    if type(workers) is not int or not 1 <= workers <= 30:
        raise ValueError("--workers must be an integer in [1, 30]")
    jobs = list(jobs)
    if not jobs:
        raise ValueError("evaluation requires at least one job")
    for reference, seed in jobs:
        if not Path(reference).is_absolute() or type(seed) is not int or seed < 0:
            raise ValueError("jobs require an absolute instance path and nonnegative seed")
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
                model, cfg, builder, params, seed=seed, sample=sample)
            results.append({"instance_id": Path(reference).stem, "seed": seed,
                            "stats": stats, "routes": solution.routes,
                            "load_seconds": load_seconds,
                            "cache_size": len(provider._cache)})
            print(f"[test {index + 1}/{len(jobs)}] {params.instance_id} seed={seed} "
                  f"runtime={stats['runtime_s']:.2f}s", flush=True)
        timings.update({"workers_requested": workers, "worker_count": 0,
                        "backend": "serial", "startup_seconds": 0.0,
                        "evaluation_wall_seconds": time.perf_counter() - started})
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
                f"exitcode={processes[worker_id].exitcode}") from exc
        if not isinstance(message, tuple) or len(message) < 2 or message[1] != worker_id:
            raise RuntimeError(f"invalid reply from evaluation worker {worker_id}")
        if message[0] == "error":
            if len(message) != 5:
                raise RuntimeError(f"malformed worker {worker_id} error")
            raise RuntimeError(f"evaluation worker {worker_id} task={message[2]} "
                               f"phase={message[3]}:\n{message[4]}")
        return message

    def wait_ready():
        objects = ([connections[i] for i in pending]
                   + [process.sentinel for process in processes])
        ready = wait(objects, timeout=5.0)
        return [i for i in pending if connections[i] in ready]

    def check_processes():
        for i, process in enumerate(processes):
            if process.exitcode is not None:
                # An ERROR can become readable after wait() took its snapshot.
                # Drain it before replacing its traceback with a generic death.
                if i in pending and connections[i].poll():
                    receive(i)
                raise RuntimeError(f"evaluation worker {i} pid={process.pid} died "
                                   f"exitcode={process.exitcode}, task={pending.get(i)}")

    try:
        for worker_id in range(worker_count):
            parent, child = context.Pipe(duplex=True)
            connections.append(parent)
            process = context.Process(
                target=_worker,
                args=(worker_id, child, package, str(Path(model_path).resolve()),
                      provider.checkpoint_metadata, provider.worker_spec()),
                name=f"policy-evaluation-{worker_id}", daemon=False)
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
                if (info["pid"] != processes[worker_id].pid
                        or _normal_path(info["executable"]) != _normal_path(sys.executable)
                        or _normal_path(info["ppo_module"]) != _normal_path(ppo.__file__)
                        or info["device"] != "cpu"):
                    raise RuntimeError(f"evaluation worker {worker_id} environment mismatch")
                runtime[worker_id] = info
                del pending[worker_id]
            check_processes()
            now = time.perf_counter()
            if pending and now >= deadline:
                raise TimeoutError(f"evaluation startup timed out: workers {sorted(pending)}")
            if pending and now - last_report >= 30.0:
                print(f"[test startup] waiting={sorted(pending)} elapsed={now-started:.1f}s",
                      flush=True)
                last_report = now
        timings.update({"workers_requested": workers, "worker_count": worker_count,
                        "backend": "process", "workers": runtime,
                        "startup_seconds": time.perf_counter() - started})
        next_task = 0

        def dispatch(worker_id):
            nonlocal next_task
            task_id = next_task
            reference, seed = jobs[task_id]
            pending[worker_id] = task_id
            connections[worker_id].send(("evaluate", task_id, (reference, seed, sample)))
            next_task += 1

        for worker_id in range(worker_count):
            dispatch(worker_id)
        completed = 0
        last_report = time.perf_counter()
        while pending:
            for worker_id in wait_ready():
                message = receive(worker_id)
                task_id = pending[worker_id]
                if (len(message) != 4 or message[0] != "ok"
                        or message[2] != task_id):
                    raise RuntimeError(f"evaluation worker {worker_id} task ID mismatch")
                record = message[3]
                reference, seed = jobs[task_id]
                if record["instance_id"] != Path(reference).stem or record["seed"] != seed:
                    raise RuntimeError(f"evaluation worker {worker_id} returned the wrong case")
                results[task_id] = record
                del pending[worker_id]
                completed += 1
                print(f"[test {completed}/{len(jobs)}] {record['instance_id']} "
                      f"seed={seed} worker={worker_id} "
                      f"runtime={record['stats']['runtime_s']:.2f}s", flush=True)
                if next_task < len(jobs):
                    dispatch(worker_id)
            check_processes()
            now = time.perf_counter()
            if pending and now - last_report >= 30.0:
                print(f"[test waiting] workers={sorted(pending)} completed={completed}/"
                      f"{len(jobs)} elapsed={now-started:.1f}s", flush=True)
                last_report = now
        return results
    finally:
        close_started = time.perf_counter()
        _cleanup(processes, connections, pending)
        timings["close_seconds"] = time.perf_counter() - close_started
        timings["evaluation_wall_seconds"] = time.perf_counter() - started


def run_evaluation_cli(package, args, *, main_started=None):
    """Shared artifact-safe CLI plumbing for the two unchanged evaluators."""
    from datetime import datetime
    import yaml

    from v2_codex.common.artifacts import (reserve_artifacts, validate_run_label,
                                  write_csv, write_json)
    from common.params import REPO_ROOT
    from v2_codex.common.runtime import report_training_runtime

    main_started = time.perf_counter() if main_started is None else main_started
    if (args.seeds <= 0 or (args.limit is not None and args.limit <= 0)
            or not 1 <= args.workers <= 30):
        raise ValueError("--seeds/--limit must be positive; --workers must be in [1, 30]")
    args.run_label = validate_run_label(args.run_label)
    ppo = importlib.import_module(f"{package}.ppo")
    provider = importlib.import_module(f"{package}.alns").DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag, split="test")
    with args.params.expanduser().resolve().open("r", encoding="utf-8") as stream:
        parameter_settings = yaml.safe_load(stream)
    method = package.rsplit(".", 1)[-1]
    suffix = "" if args.tag is None else f"_{args.tag}"
    reward_token = ppo.reward_artifact_token(args.reward_mode)
    model_path = (args.checkpoint or REPO_ROOT / "models" / "v2_codex"
                  / f"{method}_n{args.size}_{reward_token}{suffix}.pt")
    # A run label labels outputs only: choosing a trained model stays explicit.
    method_dir = method if args.tag is None else f"{method}_{args.tag}"
    out_dir = REPO_ROOT / "output" / "v2_codex" / method_dir / f"n{args.size}"
    if args.run_label is not None:
        out_dir = out_dir / "runs" / args.run_label
    paths = provider.test if args.limit is None else provider.test[:args.limit]
    jobs = [(str(path.resolve()), seed) for path in paths for seed in range(args.seeds)]
    record_paths = [out_dir / f"{Path(reference).stem}_{reward_token}_s{seed}.json"
                    for reference, seed in jobs]
    csv_path = out_dir / f"test_{reward_token}.csv"
    metadata_path = out_dir / f"test_metadata_{reward_token}.json"
    with reserve_artifacts([*record_paths, csv_path, metadata_path]):
        runtime = report_training_runtime("cpu")
        model_started = time.perf_counter()
        model, cfg, norms = ppo.load_model(
            model_path, provider.checkpoint_metadata, device="cpu")
        if cfg.reward_mode != args.reward_mode:
            raise ValueError(f"selected reward mode {args.reward_mode!r} does not match "
                             f"checkpoint mode {cfg.reward_mode!r}")
        builder = (importlib.import_module(f"{package}.gnn").GraphBuilder(norms, cfg)
                   if package == "v2_codex.gnn_ppo_alns" else None)
        model_load_seconds = time.perf_counter() - model_started
        evaluation_stats = {}
        results = evaluate_cases(
            package, model, cfg, builder, provider, model_path, jobs,
            workers=args.workers, sample=not args.argmax, runtime_stats=evaluation_stats)
        rows = []
        for record, destination in zip(results, record_paths):
            stats = record["stats"]
            write_json(destination, {"instance_id": record["instance_id"],
                                     "seed": record["seed"], "reward_mode": cfg.reward_mode,
                                     "stats": stats, "routes": record["routes"]})
            rows.append({"instance_id": record["instance_id"], "seed": record["seed"],
                         "reward_mode": cfg.reward_mode, "obj": stats["best_cost"],
                         "runtime_s": round(stats["runtime_s"], 3),
                         "improve_pct": stats["improve_pct"]})
        write_csv(csv_path, rows)
        metadata = {
            "schema_version": 2, "status": "completed", "model": method,
            "size": args.size, "tag": args.tag, "reward_mode": cfg.reward_mode,
            "device": "cpu", "run_label": args.run_label,
            "workers_requested": args.workers,
            "worker_count": evaluation_stats["worker_count"],
            "backend": evaluation_stats["backend"], "cases_completed": len(results),
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checkpoint": str(model_path.resolve()),
            "selection_mode": "argmax" if args.argmax else "sampling_epsilon",
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "ppo_config": cfg.to_dict(), "parameters": parameter_settings,
            "checkpoint_metadata": provider.checkpoint_metadata,
            "runtime": runtime, "evaluation_stats": evaluation_stats,
            "evaluation_wall_seconds": evaluation_stats["evaluation_wall_seconds"],
            "model_load_seconds": model_load_seconds,
            "main_wall_seconds": time.perf_counter() - main_started,
            "main_wall_scope": "main entry through result publication; excludes interpreter/import and final metadata write",
        }
        write_json(metadata_path, metadata)
        print(f"[complete] cases={len(results)} csv={csv_path} metadata={metadata_path}",
              flush=True)
