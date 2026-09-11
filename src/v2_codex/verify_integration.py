"""Verify isolated entrypoints, CPU spawn routing, and checkpoint compatibility.

Run from the repository root::

    python src/v2_codex/verify_integration.py

This is a correctness smoke check, not a speed benchmark or training run.
Short verification-only environment episodes leave the production defaults
unchanged. Existing checkpoints are read only; no result artifacts are written.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path[:] = [p for p in sys.path
                   if Path(p).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT


SRC = REPO_ROOT / "src"
METHODS = ("ppo_alns", "gnn_ppo_alns")


def _equal(left, right, where):
    import torch

    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or left.shape != right.shape or not torch.equal(left, right)):
            raise AssertionError(f"{where}: tensor mismatch")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or list(left) != list(right):
            raise AssertionError(f"{where}: dictionary keys/order mismatch")
        for key in left:
            _equal(left[key], right[key], f"{where}/{key}")
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"{where}: sequence mismatch")
        for index, (a, b) in enumerate(zip(left, right)):
            _equal(a, b, f"{where}/{index}")
    elif type(left) is not type(right) or left != right:
        raise AssertionError(f"{where}: {left!r} != {right!r}")


def _source_hashes():
    return {str(path.relative_to(SRC)): hashlib.sha256(path.read_bytes()).hexdigest()
            for package in ("alns", *METHODS, "common")
            for path in (SRC / package).glob("*.py")}


def check_preserved_source():
    # Only namespace/worker plumbing and artifact paths may alter these bodies.
    allowed = {
        "alns/solve.py": {"main"},
        "ppo_alns/ppo.py": {"make_envs"},
        "ppo_alns/train.py": {"main"},
        "ppo_alns/test.py": {"main"},
        "gnn_ppo_alns/train.py": {"main"},
        "gnn_ppo_alns/test.py": {"main"},
        "gnn_ppo_alns/parallel_env.py": {"_alns_worker_main", "ParallelVecALNS"},
        "common/policy_evaluation.py": {"_worker", "evaluate_cases", "run_evaluation_cli"},
    }
    count = 0
    for package in ("alns", *METHODS, "common"):
        for copied in (SRC / "v2_codex" / package).glob("*.py"):
            original = SRC / package / copied.name
            if not original.is_file():
                continue
            definitions = []
            for path in (original, copied):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                definitions.append({node.name: ast.dump(node, include_attributes=False)
                                    for node in tree.body
                                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))})
            key = original.relative_to(SRC).as_posix()
            for name, body in definitions[0].items():
                if name in allowed.get(key, set()):
                    continue
                if definitions[1].get(name) != body:
                    raise AssertionError(f"copied source drift: {key}:{name}")
                count += 1
    print(f"[PASS source] {count} preserved function/class definitions", flush=True)


def check_cli():
    entrypoints = ("alns/solve.py", "ppo_alns/train.py", "ppo_alns/test.py",
                   "gnn_ppo_alns/train.py", "gnn_ppo_alns/test.py")
    for entry in entrypoints:
        outputs = []
        commands = ([sys.executable, str(SRC / entry), "--help"],
                    [sys.executable, str(SRC / "v2_codex" / entry), "--help"],
                    [sys.executable, "-m", "v2_codex." + entry[:-3].replace("/", "."),
                     "--help"])
        for index, command in enumerate(commands):
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            if index == 2:
                env["PYTHONPATH"] = str(SRC)
            else:
                # Direct scripts must bootstrap themselves; src on PYTHONPATH
                # could otherwise hide a broken parents[] import path.
                env.pop("PYTHONPATH", None)
            result = subprocess.run(command, cwd=REPO_ROOT, env=env,
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise AssertionError(f"CLI failed: {command}\n{result.stderr}")
            outputs.append(result.stdout)
        _equal(outputs[0], outputs[1], f"{entry}/direct help")
        _equal(outputs[0], outputs[2], f"{entry}/module help")
        print(f"[PASS CLI] {entry}: original, direct v2, and module v2 match", flush=True)


def check_bindings():
    from v2_codex import operators

    for name in ("alns.solve", "ppo_alns.alns", "gnn_ppo_alns.alns"):
        original = importlib.import_module(name)
        copied = importlib.import_module(f"v2_codex.{name}")
        if original is copied or original.Solution is copied.Solution:
            raise AssertionError(f"module alias detected: {name}")
        for operation in ("best_insertion", "repair_greedy", "repair_regret2",
                          "remove_customers", "destroy_random", "destroy_worst",
                          "destroy_related"):
            if getattr(copied, operation) is not getattr(operators, operation):
                raise AssertionError(f"v2 operator not bound: {name}.{operation}")
            if getattr(original, operation) is getattr(operators, operation):
                raise AssertionError(f"original operator was rebound: {name}.{operation}")
        destroy = copied.DESTROY if name == "alns.solve" else copied.DESTROY_OPERATORS
        for operator_name, operation in destroy:
            if operation is not getattr(operators, f"destroy_{operator_name}"):
                raise AssertionError(f"destroy list captured legacy operator: {name}")
    for method in METHODS:
        original = importlib.import_module(f"{method}.ppo")
        copied = importlib.import_module(f"v2_codex.{method}.ppo")
        _equal(original.PPOConfig().to_dict(), copied.PPOConfig().to_dict(),
               f"{method}/production config")
    print("[PASS namespace] original modules coexist, operator bindings and defaults match", flush=True)


def _compare_observations(left, right, where):
    from torch_geometric.data import Batch

    if len(left) != len(right):
        raise AssertionError(f"{where}: environment count differs")
    for index, (a, b) in enumerate(zip(left, right)):
        if type(a) is not type(b):
            raise AssertionError(f"{where}/{index}: observation class differs")
        _equal(a.to_dict(), b.to_dict(), f"{where}/{index}")
    _equal(Batch.from_data_list(left).to_dict(), Batch.from_data_list(right).to_dict(),
           f"{where}/batch")


def _provider(method, args, split):
    alns = importlib.import_module(f"v2_codex.{method}.alns")
    return alns.DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag, seed=0,
        train_count=2 if split == "train" else None, split=split)


def check_environments(method, args, codec):
    ppo = importlib.import_module(f"v2_codex.{method}.ppo")
    # Explicit smoke-only overrides: exercise terminal auto-reset in five steps.
    cfg = ppo.PPOConfig(device="cpu", n_envs=2, search_iterations=3, train_count=2)
    serial_provider, process_provider = (_provider(method, args, "train") for _ in range(2))
    norms, builder = {}, None
    if method == "gnn_ppo_alns":
        gnn = importlib.import_module(f"v2_codex.{method}.gnn")
        norms = gnn.compute_norms([serial_provider._params(path)
                                  for path in serial_provider.train])
        builder = gnn.GraphBuilder(norms, cfg)
    before = {process.pid for process in multiprocessing.active_children()}
    with ppo.make_envs(cfg, serial_provider, builder, env_backend="serial", norms=norms) as serial:
        with ppo.make_envs(cfg, process_provider, builder, env_backend="process", norms=norms,
                           observation_codec=codec) as parallel:
            if parallel.package != f"v2_codex.{method}":
                raise AssertionError("environment worker selected the wrong package")
            for runtime in parallel.worker_runtime.values():
                for key, filename in (("ppo_module", "ppo.py"), ("alns_module", "alns.py")):
                    _equal(Path(runtime[key]).resolve(),
                           (SRC / "v2_codex" / method / filename).resolve(), "worker module")
                if runtime["cuda_initialized"]:
                    raise AssertionError("CPU worker initialized CUDA")
            _compare_observations(serial.reset(), parallel.reset(), "reset")
            seen = set()
            for step in range(5):
                actions = [(2 * step + index) % cfg.n_actions for index in range(cfg.n_envs)]
                seen.update(actions)
                left, right = serial.step(actions), parallel.step(actions)
                _compare_observations(left[0], right[0], f"step={step}")
                _equal(left[1:], right[1:], f"step={step}/reward,done,info")
                _equal(serial_provider.rng.getstate(), process_provider.rng.getstate(),
                       f"step={step}/instance sampling RNG")
            _equal(seen, set(range(cfg.n_actions)), "joint action coverage")
    remaining = {process.pid for process in multiprocessing.active_children()} - before
    if remaining:
        raise AssertionError(f"worker process leak: {remaining}")
    print(f"[PASS environment] {method}/{codec}: nine actions, auto-reset, exact spawn agreement",
          flush=True)


def _without_timers(record):
    stats = {key: value for key, value in record["stats"].items() if key != "runtime_s"}
    stats["best_trace"] = [(step, cost) for step, _, cost in stats["best_trace"]]
    return {"instance_id": record["instance_id"], "seed": record["seed"],
            "stats": stats, "routes": record["routes"]}


def check_checkpoint(method, args):
    import torch
    from torch_geometric.data import Batch
    from v2_codex.common.policy_evaluation import evaluate_cases

    original = importlib.import_module(f"{method}.ppo")
    copied = importlib.import_module(f"v2_codex.{method}.ppo")
    original_alns = importlib.import_module(f"{method}.alns")
    provider = _provider(method, args, "test")
    baseline_provider = original_alns.DirectoryInstanceProvider(
        args.size, params_path=args.params, tag=args.tag, split="test")
    _equal(provider.checkpoint_metadata, baseline_provider.checkpoint_metadata,
           "checkpoint metadata")
    suffix = "" if args.tag is None else f"_{args.tag}"
    path = args.checkpoint_dir / f"{method}_n{args.size}_reward_magnitude{suffix}.pt"
    baseline_model, baseline_cfg, baseline_norms = original.load_model(
        path, provider.checkpoint_metadata, device="cpu")
    model, cfg, norms = copied.load_model(path, provider.checkpoint_metadata, device="cpu")
    _equal(baseline_cfg.to_dict(), cfg.to_dict(), "checkpoint config")
    _equal(baseline_norms, norms, "checkpoint norms")
    _equal(baseline_model.state_dict(), model.state_dict(), "checkpoint weights")
    before_config = cfg.to_dict()
    before_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
    builder = None
    if method == "gnn_ppo_alns":
        builder = importlib.import_module(f"v2_codex.{method}.gnn").GraphBuilder(norms, cfg)
    params = provider._params(provider.test[0])
    observation = copied.ALNSEnv(None, builder, cfg, seed=0).reset(params)
    batch = Batch.from_data_list([observation])
    with torch.no_grad():
        _equal(baseline_model.action_probs(batch), model.action_probs(batch), "checkpoint inference")
    jobs = [(str(provider.test[0].resolve()), seed) for seed in range(2)]
    records = []
    before_children = {process.pid for process in multiprocessing.active_children()}
    for workers in (1, 2):
        runtime = {}
        result = evaluate_cases(f"v2_codex.{method}", model, cfg, builder, provider, path,
                                jobs, workers=workers, sample=True, runtime_stats=runtime)
        for info in runtime.get("workers", {}).values():
            _equal(Path(info["ppo_module"]).resolve(),
                   (SRC / "v2_codex" / method / "ppo.py").resolve(), "evaluation worker module")
        for record in result:
            _equal(record["stats"]["iters_done"], cfg.search_iterations, "evaluation budget")
            _equal(sum(record["stats"]["action_hist"]), cfg.search_iterations, "action count")
        records.append([_without_timers(record) for record in result])
    _equal(records[0], records[1], "serial/process checkpoint evaluation")
    _equal(before_config, cfg.to_dict(), "evaluation preserved config")
    _equal(before_weights, model.state_dict(), "evaluation preserved weights")
    if {process.pid for process in multiprocessing.active_children()} - before_children:
        raise AssertionError("evaluation worker process leak")
    print(f"[PASS checkpoint] {method}: original weights/inference, metadata, and v2 evaluation workers",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(5, 10, 20), default=5)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--skip-cli", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()
    original_hashes = _source_hashes()
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check_preserved_source()
    if not args.skip_cli:
        check_cli()
    check_bindings()
    for method in METHODS:
        for codec in ("direct", "numpy"):
            check_environments(method, args, codec)
        if not args.skip_checkpoints:
            check_checkpoint(method, args)
    _equal(original_hashes, _source_hashes(), "original source hashes")
    print("[PASS integration] original sources unchanged; no training or output artifacts written",
          flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
