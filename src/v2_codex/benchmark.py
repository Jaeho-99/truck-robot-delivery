"""Paired original/v2 ALNS experiments; writes only output/v2_codex.

Time includes initial-solution construction and every ALNS iteration, but not
imports, NPZ loading, result comparison or file I/O. Original and v2 runs are
sequential with alternating order, identical instances, seeds and budgets.
"""

import argparse
from datetime import datetime
from pathlib import Path
import platform
import statistics
import sys
import time

if __package__ in (None, ""):
    # The script directory contains regular packages named alns/ppo_alns.
    # Leaving it on sys.path would shadow the originals (namespace packages)
    # even with src inserted first, and accidentally compare v2 with itself.
    sys.path[:] = [p for p in sys.path
                   if Path(p).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alns import solve as original
from v2_codex.common.artifacts import reserve_artifacts, validate_run_label, write_csv, write_json
from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT
from v2_codex.alns import solve as accelerated


def _stable_stats(stats):
    result = {k: v for k, v in stats.items()
              if k not in {"pair_time_s", "selector_overhead_s", "best_trace"}}
    result["best_trace"] = [(row[0], row[2]) for row in stats["best_trace"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, choices=(5, 10, 20, 50, 100),
                        default=[5, 10, 20])
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--run-label", default=datetime.now().strftime("compare_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if (Path(original.__file__).resolve() != REPO_ROOT / "src" / "alns" / "solve.py"
            or original.repair_greedy is accelerated.repair_greedy):
        raise RuntimeError("Original/v2 import isolation failed; refusing a self-comparison")
    validate_run_label(args.run_label)
    if min(args.instances, args.seeds, args.iterations, args.repeats) <= 0:
        parser.error("counts and iterations must be positive")
    destination = REPO_ROOT / "output" / "v2_codex" / "benchmarks" / args.run_label
    csv_path = destination / "cases.csv"
    summary_path = destination / "summary.json"
    rows = []
    with reserve_artifacts([csv_path, summary_path]):
        for size in args.sizes:
            provider = original.DirectoryInstanceProvider(
                size, params_path=args.params, tag=args.tag, split="test")
            if len(provider.test) < args.instances:
                parser.error(f"n{size} has only {len(provider.test)} test instances")
            for path in provider.test[:args.instances]:
                pr = provider._params(path)
                for seed in range(args.seeds):
                    for repeat in range(args.repeats):
                        results = {}
                        order = [("original", original), ("v2_codex", accelerated)]
                        if len(rows) % 2:
                            order.reverse()
                        for name, module in order:
                            trace = []
                            started = time.perf_counter()
                            solution, objective, stats = module.solve_alns(
                                pr, iters=args.iterations, seed=seed, iter_trace=trace)
                            elapsed = time.perf_counter() - started
                            # Always validate with the unchanged public evaluator.
                            evaluated, feasible, _, _ = original.eval_solution(pr, solution)
                            results[name] = (solution, objective, stats, trace, elapsed,
                                             feasible and objective == evaluated)
                        old = results["original"]
                        new = results["v2_codex"]
                        row = {
                            "size": size, "instance_id": path.stem, "seed": seed,
                            "repeat": repeat, "iterations": args.iterations,
                            "first": order[0][0],
                            "original_seconds": old[4], "v2_codex_seconds": new[4],
                            "speedup": old[4] / new[4],
                            "original_objective": old[1], "v2_codex_objective": new[1],
                            "objective_gap_pct": 100 * (new[1] - old[1]) / max(abs(old[1]), 1e-12),
                            "both_feasible": old[5] and new[5],
                            "routes_equal": old[0].routes == new[0].routes,
                            "trace_equal": old[3] == new[3],
                            "stats_equal": _stable_stats(old[2]) == _stable_stats(new[2]),
                        }
                        row["identical"] = (old[1] == new[1] and row["both_feasible"]
                                            and row["routes_equal"] and row["trace_equal"]
                                            and row["stats_equal"])
                        rows.append(row)
                        print(f"[compare] n{size} {path.stem} seed={seed} repeat={repeat} "
                              f"original={old[4]:.3f}s v2={new[4]:.3f}s "
                              f"speedup={row['speedup']:.2f}x identical={row['identical']}",
                              flush=True)
        summary = []
        for size in args.sizes:
            selected = [row for row in rows if row["size"] == size]
            old_total = sum(row["original_seconds"] for row in selected)
            new_total = sum(row["v2_codex_seconds"] for row in selected)
            summary.append({
                "size": size, "cases": len(selected),
                "original_total_seconds": old_total, "v2_codex_total_seconds": new_total,
                "aggregate_speedup": old_total / new_total,
                "time_reduction_pct": 100 * (1 - new_total / old_total),
                "median_case_speedup": statistics.median(row["speedup"] for row in selected),
                "max_abs_objective_gap_pct": max(abs(row["objective_gap_pct"]) for row in selected),
                "identical_cases": sum(row["identical"] for row in selected),
            })
        write_csv(csv_path, rows)
        write_json(summary_path, {
            "status": "completed" if all(row["identical"] for row in rows) else "mismatch",
            "completed_at": datetime.now().astimezone().isoformat(),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "python": sys.version, "platform": platform.platform(),
            "original_source": str(Path(original.__file__).resolve()),
            "v2_codex_source": str(Path(accelerated.__file__).resolve()),
            "timing_scope": "initial solution + ALNS; excludes imports, NPZ loading, comparison and I/O",
            "summary": summary,
        })
        for result in summary:
            print(f"[summary] n{result['size']} speedup={result['aggregate_speedup']:.2f}x "
                  f"time_reduction={result['time_reduction_pct']:.1f}% "
                  f"identical={result['identical_cases']}/{result['cases']}", flush=True)
        print(f"[complete] {summary_path}", flush=True)
    if not all(row["identical"] for row in rows):
        raise SystemExit("Comparison mismatch: inspect cases.csv before using v2 results.")


if __name__ == "__main__":
    main()
