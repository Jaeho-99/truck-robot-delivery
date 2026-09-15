"""Run vanilla ALNS on immutable, precomputed truck/robot instances.

Workflow: load NPZ matrices -> build an initial feasible route -> repeat
destroy/repair with adaptive roulette selection and annealing -> save results.
Geometry is never reconstructed: distance and travel-time matrices are read
from ``data/processed*/``. The evaluator follows the exact model's objective
and custody, capacity, range, and scheduling constraints.

Run from the repository root, for example::

    python src/alns/solve.py --size 5 --limit 1 --seeds 1 --iterations 20
    python src/alns/solve.py --size 20 --workers 2 --run-label example_20

Use a fresh ``--run-label`` when repeating a run: artifact reservations
prevent accidental overwrites. Output remains under ``output/alns/n<size>/``
(or the existing tagged/run-label subdirectories).

Route representation
--------------------
``Solution.routes`` maps a truck ID to its ordered stops. A customer stop
is ``{"kind": "cust", "c": 1}``; a parking stop is
``{"kind": "park", "p": 6, "deploys": [...]}``. Each robot trip records
``{"r": 1, "custs": [2, 3], "ret_p": 7}``, attached to its launch stop.
The retrieval copy must occur later on the same truck route. Even when
launch and retrieval share a physical location, two distinct copies are
consumed. The truck can serve other stops while the robot is away.

The evaluator tracks whether each robot is aboard, waits at retrieval when
needed, and requires all robots aboard at route end. Robot distance is
accumulated across trips without battery swapping; lateness is a soft cost.
"""

import argparse
import math
import multiprocessing
import os
import random
import sys
import time
import traceback
from datetime import datetime
from multiprocessing.connection import wait
from pathlib import Path

import yaml

if __package__ in (None, ""):
    # Support both ``python src/alns/solve.py`` and package imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.artifacts import (
    reserve_artifacts,
    validate_run_label,
    write_csv,
    write_json,
)
from common.params import (
    DEFAULT_PARAMS_PATH,
    REPO_ROOT,
    load_problem,
)
from common.search import (
    DESTROY,
    DOD,
    NOISE_FRAC,
    W_START,
    DirectoryInstanceProvider,
    Params,
    congestion_aware_initial,
    eval_solution_cost,
    repair_greedy,
    repair_regret2,
)
from common.sizes import SUPPORTED_SIZES


def roulette(weights, rng):
    """Draw an index proportionally to its adaptive operator weight."""
    tot = sum(weights)
    y = rng.random() * tot
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if y <= acc:
            return i
    return len(weights) - 1


def solve_alns(
    pr,
    iters=3000,
    seed=0,
    segment=None,
    sigma=(5.0, 3.0, 1.0),
    reaction=0.2,
    w_start=W_START,
    time_limit_s=None,
    initial=None,
    iter_trace=None,
    dod=DOD,
):
    """Run ALNS and return (best_solution, best_cost, stats).

    ``initial``: initial-solution constructor ``f(pr, rng) -> Solution``
    (default: congestion_aware_initial).
    ``iter_trace``: optional list; if given, one
    (it, accepted, current_cost) tuple is appended per iteration
    (instrumentation only — never touches rng).
    ``time_limit_s``: wall-clock cap in seconds; exceeding it stops the
    run early (for large instances). The cooling schedule stays based
    on ``iters``, so an early stop may end in the hot phase.
    Operators are chosen by classic adaptive roulette weights.
    """
    t_start = time.time()
    rng = random.Random(seed)
    segment = segment or max(20, iters // 30)
    nC = len(pr.C)
    # degree of destruction: fixed fraction of customers (DR-ALNS
    # vanilla uses 30%; ``dod`` allows regime studies)
    q_destroy = max(1, round(dod * nC))

    if initial is None:
        initial = congestion_aware_initial
    current_solution = initial(pr, rng)
    current_cost, feas = eval_solution_cost(pr, current_solution)
    best_solution, best_cost = current_solution.clone(), current_cost
    init_cost = current_cost

    # Repair operators (noise amplitude scales with instance cost).
    noise_amp = NOISE_FRAC * init_cost
    repair_ops = [
        (
            "greedy",
            lambda p_, s_, pool_, rng_: repair_greedy(
                p_, s_, pool_, rng_, 0.0
            ),
        ),
        (
            "greedy_noise",
            lambda p_, s_, pool_, rng_: repair_greedy(
                p_, s_, pool_, rng_, noise_amp
            ),
        ),
        ("regret2", repair_regret2),
    ]

    # SA start temperature: a solution worse than the initial one by
    # w_start (fraction) is accepted with probability 0.5. Linear
    # decay to 0 over the run (Santini et al.; same rule in PPO).
    T0 = (w_start * init_cost) / math.log(2)

    dW = [1.0] * len(DESTROY)
    rW = [1.0] * len(repair_ops)
    dScore = [0.0] * len(DESTROY)
    rScore = [0.0] * len(repair_ops)
    dCnt = [0] * len(DESTROY)
    rCnt = [0] * len(repair_ops)
    seen = set()

    # Instrumentation (timers/counters only — never touches rng, so
    # results stay byte-identical to the uninstrumented loop).
    n_actions = len(DESTROY) * len(repair_ops)
    pair_cnt = [0] * n_actions  # a = di * len(repair_ops) + ri
    pair_time = [0.0] * n_actions  # destroy+repair+eval seconds
    sel_time = 0.0  # selector overhead seconds
    accept_cnt = infeas_cnt = best_updates = best_hit_it = 0
    best_trace = []  # (iter, elapsed_s, best_cost)

    it_done = 0
    for it in range(1, iters + 1):
        if time_limit_s is not None and time.time() - t_start > time_limit_s:
            break
        it_done = it
        # linear temperature decay T0 -> 0 over the run
        T = T0 * (1.0 - (it - 1) / iters)
        t_sel = time.perf_counter()
        di = roulette(dW, rng)
        ri = roulette(rW, rng)
        sel_time += time.perf_counter() - t_sel
        a_idx = di * len(repair_ops) + ri
        t_op = time.perf_counter()  # clone+destroy+repair+eval
        cand = current_solution.clone()
        pool = DESTROY[di][1](pr, cand, q_destroy, rng)
        repair_ops[ri][1](pr, cand, pool, rng)
        cand_cost, ok = eval_solution_cost(pr, cand)
        pair_cnt[a_idx] += 1
        pair_time[a_idx] += time.perf_counter() - t_op
        dCnt[di] += 1
        rCnt[ri] += 1

        if not ok:  # discard coverage/custody/range violations
            infeas_cnt += 1
            if iter_trace is not None:
                iter_trace.append((it, 0, round(current_cost, 9)))
            continue

        # Operator scores (DR-ALNS weights w1..w4 = 5, 3, 1, 0):
        # 5 new best / 3 improving the current solution / 1 accepted /
        # 0 otherwise (improving/accepted only for unseen solutions).
        key = round(cand_cost, 4)
        reward = 0.0
        accept = False
        if cand_cost < best_cost - 1e-9:
            best_solution, best_cost = cand.clone(), cand_cost
            best_updates += 1
            best_hit_it = it
            best_trace.append(
                (it, round(time.time() - t_start, 3), round(cand_cost, 6))
            )
            reward = sigma[0]
            accept = True
        elif cand_cost < current_cost - 1e-9 and key not in seen:
            reward = sigma[1]
            accept = True
        else:
            if cand_cost < current_cost - 1e-9 or rng.random() < math.exp(
                -(cand_cost - current_cost) / max(T, 1e-9)
            ):
                accept = True
                if key not in seen:
                    reward = sigma[2]
        seen.add(key)
        if accept:
            accept_cnt += 1
            current_solution, current_cost = cand, cand_cost
        if iter_trace is not None:
            iter_trace.append((it, int(accept), round(current_cost, 9)))
        dScore[di] += reward
        rScore[ri] += reward

        if it % segment == 0:  # adaptive weight update
            for i in range(len(DESTROY)):
                if dCnt[i] > 0:
                    dW[i] = dW[i] * (1 - reaction) + reaction * (
                        dScore[i] / dCnt[i]
                    )
                dScore[i] = 0.0
                dCnt[i] = 0
            for i in range(len(repair_ops)):
                if rCnt[i] > 0:
                    rW[i] = rW[i] * (1 - reaction) + reaction * (
                        rScore[i] / rCnt[i]
                    )
                rScore[i] = 0.0
                rCnt[i] = 0

    stats = {
        "init_cost": init_cost,
        "best_cost": best_cost,
        "iters_done": it_done,
        "selector": "roulette",
        "improve_pct": 100.0 * (init_cost - best_cost) / init_cost,
        "destroy_w": dict(
            zip([d[0] for d in DESTROY], [round(x, 3) for x in dW])
        ),
        "repair_w": dict(
            zip([r[0] for r in repair_ops], [round(x, 3) for x in rW])
        ),
        # instrumentation (a = destroy_index * 3 + repair_index)
        "pair_labels": [f"{d[0]}+{r[0]}" for d in DESTROY for r in repair_ops],
        "action_hist": pair_cnt,
        "pair_time_s": [round(x, 3) for x in pair_time],
        "selector_overhead_s": round(sel_time, 3),
        "accept_count": accept_cnt,
        "infeasible_count": infeas_cnt,
        "best_update_count": best_updates,
        "best_first_hit_iter": best_hit_it,
        "best_trace": best_trace,
    }
    return best_solution, best_cost, stats


def _case_result(task, size, params_path, cache, capture_trace):
    """Solve one independent case; Params and Solution never cross a pipe."""
    task_id, instance_path, seed, iterations = task
    load_started = time.perf_counter()
    if instance_path not in cache:
        config, data = load_problem(instance_path, params_path)
        cache[instance_path] = Params(data, config, size)
    params = cache[instance_path]
    load_seconds = time.perf_counter() - load_started
    iter_trace = [] if capture_trace else None
    started = time.perf_counter()
    solution, objective, stats = solve_alns(
        params, iters=iterations, seed=seed, iter_trace=iter_trace
    )
    result = {
        "task_id": task_id,
        # Preserve the old CLI's filename-derived instance identifier.
        "instance_id": Path(instance_path).stem,
        "seed": seed,
        "obj": objective,
        "stats": stats,
        "routes": solution.routes,
        "runtime_s": time.perf_counter() - started,
        "load_seconds": load_seconds,
        "cache_size": len(cache),
    }
    if capture_trace:
        result["iter_trace"] = iter_trace
    return result


def _alns_case_worker(worker_id, connection, size, params_path, capture_trace):
    """Spawn-safe CPU worker. Imports neither Torch nor a CUDA runtime."""
    cache = {}
    task_id = None
    phase = "startup"
    try:
        connection.send(("ready", worker_id, os.getpid()))
        while True:
            phase = "receive"
            message = connection.recv()
            if message == ("close",):
                return
            if not isinstance(message, tuple) or len(message) != 2:
                raise ValueError("invalid ALNS worker request")
            command, task = message
            if command != "solve":
                raise ValueError(f"unknown ALNS command: {command!r}")
            task_id = task[0]
            phase = "solve"
            result = _case_result(
                task, size, params_path, cache, capture_trace
            )
            phase = "send"
            connection.send(("ok", worker_id, task_id, result))
            task_id = None
    except KeyboardInterrupt:
        # Parent owns the user-facing interruption and whole-pool cleanup.
        return
    except EOFError:
        return
    except BaseException:
        try:
            connection.send(
                ("error", worker_id, task_id, phase, traceback.format_exc())
            )
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


def _close_case_workers(processes, connections, idle, grace_seconds=5.0):
    """Bounded shutdown, including partial startup and blocked/failed jobs."""
    for worker_id in idle:
        process = processes.get(worker_id)
        if process is not None and process.is_alive():
            try:
                connections[worker_id].send(("close",))
            except (EOFError, OSError):
                pass
    deadline = time.monotonic() + grace_seconds
    for process in processes.values():
        if process.pid is not None:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
    remaining = [
        process for process in processes.values() if process.is_alive()
    ]
    for process in remaining:
        process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in remaining:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    # Never wait without a deadline, even if a platform termination fails.
    survivors = [process for process in remaining if process.is_alive()]
    for process in survivors:
        process.kill()
    deadline = time.monotonic() + grace_seconds
    for process in survivors:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for connection in connections.values():
        connection.close()
    for process in processes.values():
        if process.is_alive():
            print(
                f"[alns cleanup] worker PID {process.pid} did not exit; "
                "inspect this PID before manual recovery",
                flush=True,
            )
        else:
            process.close()


def run_cases(
    instance_paths,
    *,
    size,
    params_path=DEFAULT_PARAMS_PATH,
    iterations=100,
    seeds=5,
    workers=1,
    capture_trace=False,
    progress=True,
    heartbeat_interval=30.0,
    startup_timeout=180.0,
):
    """Return case results in the original instance-major, seed-minor order.

    ``workers=1`` is the serial baseline. More workers parallelize whole
    independent solves, not the dependent iterations within a solve. The
    same seed is used for a case regardless of which worker receives it.
    Each worker loads and caches its own immutable Params from file paths.
    No output files are written here, so verification cannot overwrite runs.
    """
    for name, value in (
        ("iterations", iterations),
        ("seeds", seeds),
        ("workers", workers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if workers > 30:
        raise ValueError("workers must be <= 30 (pipe + process wait handles)")
    if heartbeat_interval <= 0 or startup_timeout <= 0:
        raise ValueError(
            "heartbeat_interval and startup_timeout must be positive"
        )
    paths = [str(Path(path).expanduser().resolve()) for path in instance_paths]
    if not paths:
        raise ValueError("at least one instance path is required")
    params_path = str(Path(params_path).expanduser().resolve())
    tasks = [
        (index * seeds + seed, path, seed, iterations)
        for index, path in enumerate(paths)
        for seed in range(seeds)
    ]
    results = [None] * len(tasks)
    if workers == 1:
        cache = {}
        for task in tasks:
            if progress:
                print(
                    f"[alns {task[0] + 1}/{len(tasks)}] "
                    f"instance={Path(task[1]).stem} seed={task[2]} starting",
                    flush=True,
                )
            result = _case_result(
                task, size, params_path, cache, capture_trace
            )
            results[task[0]] = result
            if progress:
                print(
                    f"[alns {task[0] + 1}/{len(tasks)}] "
                    f"obj={result['obj']:.6f} "
                    f"solve={result['runtime_s']:.2f}s",
                    flush=True,
                )
        return results

    # There is no benefit in spawning idle workers when a batch is smaller.
    worker_count = min(workers, len(tasks))
    context = multiprocessing.get_context("spawn")
    processes, connections = {}, {}
    idle = set()
    assignments = {}
    try:
        for worker_id in range(worker_count):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_alns_case_worker,
                args=(worker_id, child, size, params_path, capture_trace),
                name=f"ALNS-case-{worker_id}",
                daemon=False,
            )
            # Register before start so an interrupt/failure during startup
            # still reaches the shared cleanup path for every owned handle.
            processes[worker_id] = process
            connections[worker_id] = parent
            try:
                process.start()
            finally:
                child.close()

        ready_workers = set()
        startup_started = time.monotonic()
        last_heartbeat = startup_started
        while len(ready_workers) != worker_count:
            pending = set(processes) - ready_workers
            objects = [connections[i] for i in pending]
            objects.extend(process.sentinel for process in processes.values())
            readable = wait(objects, timeout=1.0)
            for worker_id in sorted(pending):
                connection = connections[worker_id]
                if connection not in readable:
                    continue
                message = connection.recv()
                if (
                    not isinstance(message, tuple)
                    or len(message) != 3
                    or message[:2] != ("ready", worker_id)
                    or message[2] != processes[worker_id].pid
                ):
                    raise RuntimeError(
                        f"worker {worker_id}: invalid READY {message!r}"
                    )
                ready_workers.add(worker_id)
                idle.add(worker_id)
            for worker_id, process in processes.items():
                if not process.is_alive():
                    raise RuntimeError(
                        f"worker {worker_id} died during startup; "
                        f"exitcode={process.exitcode}"
                    )
            now = time.monotonic()
            if (
                len(ready_workers) < worker_count
                and now - startup_started >= startup_timeout
            ):
                raise TimeoutError(
                    f"ALNS worker startup timed out: {sorted(pending)}"
                )
            if progress and now - last_heartbeat >= heartbeat_interval:
                print(
                    f"[alns startup] waiting={sorted(set(processes) - ready_workers)} "
                    f"elapsed={now - startup_started:.1f}s",
                    flush=True,
                )
                last_heartbeat = now

        next_task = 0
        completed = 0
        while completed < len(tasks):
            for worker_id in sorted(idle):
                if next_task >= len(tasks):
                    break
                task = tasks[next_task]
                assignments[worker_id] = (task[0], time.monotonic())
                idle.remove(worker_id)
                connections[worker_id].send(("solve", task))
                next_task += 1
            objects = [connections[i] for i in assignments]
            objects.extend(process.sentinel for process in processes.values())
            readable = wait(objects, timeout=1.0)
            for worker_id in list(assignments):
                connection = connections[worker_id]
                if connection not in readable:
                    continue
                try:
                    message = connection.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(
                        f"ALNS worker {worker_id} disconnected while solving "
                        f"task {assignments[worker_id][0]}"
                    ) from error
                expected_id = assignments[worker_id][0]
                if (
                    not isinstance(message, tuple)
                    or len(message) < 3
                    or message[1:3] != (worker_id, expected_id)
                ):
                    raise RuntimeError(
                        f"worker {worker_id}: invalid response {message!r}"
                    )
                if message[0] == "error" and len(message) == 5:
                    raise RuntimeError(
                        f"ALNS worker {worker_id}, task {expected_id}, "
                        f"phase={message[3]} failed:\n{message[4]}"
                    )
                if message[0] != "ok" or len(message) != 4:
                    raise RuntimeError(
                        f"worker {worker_id}: invalid response {message!r}"
                    )
                result = message[3]
                if (
                    result.get("task_id") != expected_id
                    or result.get("instance_id")
                    != Path(tasks[expected_id][1]).stem
                    or result.get("seed") != tasks[expected_id][2]
                    or results[expected_id] is not None
                ):
                    raise RuntimeError(
                        f"worker {worker_id}: mismatched result for task {expected_id}"
                    )
                results[expected_id] = result
                del assignments[worker_id]
                idle.add(worker_id)
                completed += 1
                if progress:
                    print(
                        f"[alns completed={completed}/{len(tasks)}] "
                        f"task={expected_id} worker={worker_id} "
                        f"instance={result['instance_id']} seed={result['seed']} "
                        f"obj={result['obj']:.6f} "
                        f"solve={result['runtime_s']:.2f}s",
                        flush=True,
                    )
            for worker_id, process in processes.items():
                if not process.is_alive():
                    # The process sentinel can become ready just after wait's
                    # pipe snapshot. Preserve an already queued traceback.
                    if (
                        worker_id in assignments
                        and connections[worker_id].poll()
                    ):
                        try:
                            message = connections[worker_id].recv()
                        except (EOFError, OSError):
                            message = None
                        if (
                            isinstance(message, tuple)
                            and len(message) == 5
                            and message[:3]
                            == ("error", worker_id, assignments[worker_id][0])
                        ):
                            raise RuntimeError(
                                f"ALNS worker {worker_id}, task={message[2]}, "
                                f"phase={message[3]} failed:\n{message[4]}"
                            )
                    raise RuntimeError(
                        f"ALNS worker {worker_id} died; exitcode={process.exitcode}"
                    )
            now = time.monotonic()
            if progress and now - last_heartbeat >= heartbeat_interval:
                waiting = ", ".join(
                    f"w{i}:task={task_id},elapsed={now - started:.1f}s"
                    for i, (task_id, started) in sorted(assignments.items())
                )
                print(f"[alns waiting] {waiting}", flush=True)
                last_heartbeat = now
        return results
    finally:
        _close_case_workers(processes, connections, idle)


def _parser():
    parser = argparse.ArgumentParser(
        description="Run vanilla ALNS on precomputed test instances"
    )
    parser.add_argument(
        "--size", type=int, choices=SUPPORTED_SIZES, required=True
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="number of repetition seeds, starting at zero",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent case processes (1-30); 1 keeps the serial baseline",
    )
    parser.add_argument(
        "--run-label",
        help="unique output label; existing artifacts are never overwritten",
    )
    return parser


def main():
    """Parse the CLI, solve independent cases, and publish their artifacts."""
    main_started = time.perf_counter()
    args = _parser().parse_args()
    if args.iterations <= 0 or args.seeds <= 0 or args.workers <= 0:
        raise ValueError(
            "--iterations, --seeds and --workers must be positive"
        )
    if args.workers > 30:
        raise ValueError("--workers must be <= 30")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    validate_run_label(args.run_label)
    provider = DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag, split="test"
    )
    params_path = args.params.expanduser().resolve()
    with params_path.open("r", encoding="utf-8") as f:
        parameter_settings = yaml.safe_load(f)
    # Send paths, not Params (which contains non-picklable immutable config).
    cases = provider.test
    if args.limit is not None:
        cases = cases[: args.limit]
    method_dir = "alns" if args.tag is None else f"alns_{args.tag}"
    out_dir = REPO_ROOT / "output" / method_dir / f"n{args.size}"
    if args.run_label is not None:
        out_dir = out_dir / "runs" / args.run_label
    route_paths = [
        out_dir / f"{path.stem}_s{seed}.json"
        for path in cases
        for seed in range(args.seeds)
    ]
    summary_path = out_dir / "summary.csv"
    metadata_path = out_dir / "test_metadata.json"
    with reserve_artifacts([*route_paths, summary_path, metadata_path]):
        actual_workers = min(args.workers, len(route_paths))
        print(
            f"[alns] cases={len(cases)} seeds={args.seeds} "
            f"tasks={len(route_paths)} iterations={args.iterations} "
            f"workers={actual_workers} "
            f"backend={'serial' if args.workers == 1 else 'process'} "
            f"run_label={args.run_label!r}",
            flush=True,
        )
        solve_started = time.perf_counter()
        results = run_cases(
            cases,
            size=args.size,
            params_path=params_path,
            iterations=args.iterations,
            seeds=args.seeds,
            workers=args.workers,
        )
        solve_wall_seconds = time.perf_counter() - solve_started
        rows = []
        for result, route_path in zip(results, route_paths, strict=True):
            write_json(
                route_path,
                {
                    key: result[key]
                    for key in (
                        "instance_id",
                        "seed",
                        "obj",
                        "stats",
                        "routes",
                    )
                },
            )
            rows.append(
                {
                    "instance_id": result["instance_id"],
                    "seed": result["seed"],
                    "obj": result["obj"],
                    "runtime_s": round(result["runtime_s"], 3),
                    "improve_pct": result["stats"]["improve_pct"],
                }
            )
        write_csv(summary_path, rows)
        metadata = {
            "schema_version": 2,
            "status": "completed",
            "model": "alns",
            "selection_mode": "roulette",
            "size": args.size,
            "tag": args.tag,
            "run_label": args.run_label,
            "env_backend": "serial" if args.workers == 1 else "process",
            "worker_count": actual_workers,
            "task_count": len(results),
            "solve_wall_seconds": solve_wall_seconds,
            "sum_case_solve_seconds": sum(r["runtime_s"] for r in results),
            "sum_case_load_seconds": sum(r["load_seconds"] for r in results),
            "main_wall_seconds": time.perf_counter() - main_started,
            "main_wall_scope": "main entry through result publication; excludes interpreter/import and final metadata write",
            "completed_at": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "parameters": parameter_settings,
            "checkpoint_metadata": provider.checkpoint_metadata,
        }
        # Completion metadata is published last, after every case and CSV.
        write_json(metadata_path, metadata)
        print(
            f"[alns complete] tasks={len(results)} "
            f"solve_wall={solve_wall_seconds:.2f}s output={out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
