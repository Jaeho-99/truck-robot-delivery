"""Verify identical experiment settings, startup logs, and relocated artifacts.

Executes each training entrypoint through argument parsing, PPOConfig creation,
and artifact reservation. Stops before file creation, CUDA setup, data loading,
or training. Production settings and source files are never changed.
"""

import argparse
from contextlib import redirect_stdout
import importlib
import io
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path[:] = [p for p in sys.path
                   if Path(p).resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.params import DEFAULT_PARAMS_PATH, REPO_ROOT
from v2_codex.verify_integration import check_bindings, check_preserved_source


class _ReservationReached(Exception):
    pass


def _training_start(path, arguments):
    # A fresh private module keeps probes from rebinding imported production
    # modules. main.__globals__ is used because runpy returns a namespace copy.
    namespace = runpy.run_path(str(path), run_name="_v2_contract_probe")
    main = namespace["main"]
    globals_ = main.__globals__
    config_class = globals_["PPOConfig"]
    captured = {}

    def config(*args, **kwargs):
        result = config_class(*args, **kwargs)
        captured["config"] = result.to_dict()
        captured["effective_steps"] = result.n_updates * result.t_rollout * result.n_envs
        return result

    def reserve(paths):
        captured["artifacts"] = [Path(path) for path in paths]
        raise _ReservationReached

    globals_["PPOConfig"] = config
    globals_["reserve_artifacts"] = reserve
    stdout = io.StringIO()
    with patch.object(sys, "argv", [str(path), *arguments]), redirect_stdout(stdout):
        try:
            main()
        except _ReservationReached:
            pass
        else:
            raise AssertionError("training entrypoint bypassed artifact reservation")
    captured["stdout"] = stdout.getvalue()
    assert "artifacts" in captured and "config" in captured
    return captured


def _v2_destination(path):
    if path.is_relative_to(REPO_ROOT / "models"):
        return REPO_ROOT / "models" / "v2_codex" / path.relative_to(REPO_ROOT / "models")
    assert path.is_relative_to(REPO_ROOT / "output"), f"unexpected original artifact: {path}"
    return REPO_ROOT / "output" / "v2_codex" / path.relative_to(REPO_ROOT / "output")


def check_training(sizes):
    checked = 0
    for method in ("ppo_alns", "gnn_ppo_alns"):
        for size in sizes:
            # Default invocation plus each reward mode with labels/custom data
            # and execution options. No files/data for the test tag are needed.
            scenarios = [["--size", str(size)]]
            for reward in ("magnitude", "alns_5310", "new_best_5"):
                scenarios.append([
                    "--size", str(size), "--reward-mode", reward,
                    "--tag", "contract", "--run-label", "contract_01",
                    "--device", "cpu", "--env-backend", "process",
                    "--observation-codec", "numpy",
                ])
            for arguments in scenarios:
                old = _training_start(REPO_ROOT / "src" / method / "train.py", arguments)
                new = _training_start(REPO_ROOT / "src" / "v2_codex" / method / "train.py", arguments)
                assert old["config"] == new["config"], (method, arguments, "PPOConfig")
                assert old["effective_steps"] == new["effective_steps"], "effective rollout budget"
                assert old["stdout"] == new["stdout"], (method, arguments, "startup console")
                expected = [_v2_destination(path) for path in old["artifacts"]]
                assert new["artifacts"] == expected, (method, arguments, "artifact paths")
                checked += 1
                if len(arguments) == 2:
                    cfg = new["config"]
                    print(f"[PASS train] {method} --size {size}: "
                          f"train_count={cfg['train_count']} total_steps={cfg['total_steps']} "
                          f"effective_steps={new['effective_steps']} "
                          f"search_iterations={cfg['search_iterations']} n_envs={cfg['n_envs']}",
                          flush=True)
                    print(f"  checkpoint={new['artifacts'][0].relative_to(REPO_ROOT)}", flush=True)
    print(f"[PASS startup] {checked} training configurations: identical settings/logs; "
          "only output/models roots differ", flush=True)


def check_parsers(sizes):
    from v2_codex import candidates

    modules = ("alns.solve", "ppo_alns.train", "ppo_alns.test",
               "gnn_ppo_alns.train", "gnn_ppo_alns.test")
    for name in modules:
        old = importlib.import_module(name)
        new = importlib.import_module("v2_codex." + name)
        assert old is not new
        assert old.DEFAULT_PARAMS_PATH == new.DEFAULT_PARAMS_PATH == DEFAULT_PARAMS_PATH
        for size in sizes:
            a = vars(old._parser().parse_args(["--size", str(size)]))
            b = vars(new._parser().parse_args(["--size", str(size)]))
            assert a == b, (name, size, "parser defaults")
    for name in ("alns.solve", "ppo_alns.alns", "gnn_ppo_alns.alns"):
        old = importlib.import_module(name)
        new = importlib.import_module("v2_codex." + name)
        assert old.load_problem is new.load_problem, "shared input loader"
        for constant in ("DOD", "W_START", "NOISE_FRAC", "L_RET_EXIST", "W_RET_NEW", "N_PHYS_NEAR"):
            assert getattr(old, constant) == getattr(new, constant), (name, constant)
        for constant in ("L_RET_EXIST", "W_RET_NEW", "N_PHYS_NEAR"):
            assert getattr(old, constant) == getattr(candidates, constant), (name, "active", constant)
    print("[PASS inputs] all five CLI defaults, shared params/loader and ALNS settings match",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, choices=(5, 10, 20, 50, 100),
                        default=[5, 10, 20])
    args = parser.parse_args()
    check_preserved_source()
    check_bindings()
    check_parsers(args.sizes)
    check_training(args.sizes)
    print("[PASS contract] no training started and no artifacts written", flush=True)


if __name__ == "__main__":
    main()
