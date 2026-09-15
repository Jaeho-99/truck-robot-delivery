"""Persistent spawn workers shared by CPU PPO and GNN-PPO."""

import importlib
import math
import multiprocessing
import operator
import os
import statistics
import sys
import time
import traceback
import warnings
from multiprocessing.connection import wait
from pathlib import Path


def _normal_path(path):
    return os.path.normcase(str(Path(path).resolve()))


def _alns_worker_main(
    worker_id, connection, cache_init, cfg, norms, codec, package
):
    """Spawn-safe entry; heavy imports and all mutable ALNS state stay local."""
    request_id = None
    phase = "startup"
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        alns = importlib.import_module(f"{package}.alns")
        ppo = importlib.import_module(f"{package}.ppo")
        transport_package = (
            "gnn_ppo_alns"
            if package == "gnn_ppo_alns" and cfg.use_graph
            else "ppo_alns"
        )
        encode_observation = importlib.import_module(
            f"{transport_package}.ppo"
        ).encode_observation
        builder = None
        if package == "gnn_ppo_alns" and cfg.use_graph:
            builder = importlib.import_module(f"{package}.gnn").GraphBuilder(
                norms, cfg
            )
        cache = alns.InstanceCache(**cache_init)
        env = ppo.ALNSEnv(None, builder, cfg, cfg.seed + worker_id)
        initialized = False
        if torch.cuda.is_initialized():
            raise RuntimeError("ALNS worker unexpectedly initialized CUDA")
        connection.send(
            (
                "ready",
                worker_id,
                os.getpid(),
                {
                    "executable": sys.executable,
                    "python_version": tuple(sys.version_info[:3]),
                    "ppo_module": ppo.__file__,
                    "alns_module": alns.__file__,
                    "torch_version": torch.__version__,
                    "cuda_initialized": torch.cuda.is_initialized(),
                },
            )
        )
        previous_request_id = -1
        while True:
            phase = "receive"
            request_id = None
            try:
                command = connection.recv()
            except EOFError:
                break
            if not isinstance(command, tuple) or len(command) != 3:
                raise RuntimeError("invalid parent command framing")
            phase, request_id, argument = command
            if (
                not isinstance(request_id, int)
                or request_id <= previous_request_id
            ):
                raise RuntimeError("request IDs must strictly increase")
            previous_request_id = request_id
            if phase == "close":
                break
            started = time.perf_counter()
            load_seconds = 0.0
            if phase == "reset":
                load_started = time.perf_counter()
                pr = cache.load_ref(argument)
                load_seconds = time.perf_counter() - load_started
                observation = env.reset(pr)
                initialized = True
                payload = {}
            elif phase == "step":
                if not initialized:
                    raise RuntimeError("STEP received before the first RESET")
                observation, reward, done, info = env.step(argument)
                payload = {
                    "reward": float(reward),
                    "done": bool(done),
                    "info": info,
                }
            else:
                raise RuntimeError(f"unknown command {phase!r}")
            env_seconds = time.perf_counter() - started
            if torch.cuda.is_initialized():
                raise RuntimeError("ALNS worker unexpectedly initialized CUDA")
            encoded_at = time.perf_counter()
            encoded = encode_observation(observation, codec=codec)
            timings = dict(getattr(env, "last_timing", {}))
            timings.update(
                {
                    "load_seconds": load_seconds,
                    "env_seconds": env_seconds,
                    "encode_seconds": time.perf_counter() - encoded_at,
                    "cache_size": len(cache),
                }
            )
            payload.update(observation=encoded, timings=timings)
            connection.send(("ok", request_id, worker_id, payload))
    except KeyboardInterrupt:
        # Never continue from a partially executed ALNS transition.
        pass
    except BaseException:
        try:
            connection.send(
                (
                    "error",
                    request_id,
                    worker_id,
                    phase,
                    traceback.format_exc()[-32768:],
                )
            )
        except (EOFError, OSError, KeyboardInterrupt):
            pass
    finally:
        connection.close()


class ParallelVecALNS:
    """One persistent process per env; auto-reset preserves terminal r/d/info.

    ``last_timings`` measures parent wall time, not the sum of worker times.
    ``workers[id][phase]`` contains worker measurements and current cache size.
    Parent wait and worker execution overlap and must not be added together.
    The class is intentionally single-caller: at most one request per worker is
    outstanding. On any failed operation, all workers are closed, without retry.
    """

    def __init__(
        self,
        cfg,
        provider,
        norms,
        *,
        observation_codec="direct",
        startup_timeout=180.0,
        heartbeat_interval=30.0,
        package="gnn_ppo_alns",
    ):
        if package not in {"gnn_ppo_alns", "ppo_alns"}:
            raise ValueError("unsupported ALNS policy package")
        transport_package = (
            "gnn_ppo_alns"
            if package == "gnn_ppo_alns" and cfg.use_graph
            else "ppo_alns"
        )
        decode_observation = importlib.import_module(
            f"{transport_package}.ppo"
        ).decode_observation

        self.cfg = cfg
        self.package = package
        self.provider = provider
        self.worker_count = operator.index(cfg.n_envs)
        if self.worker_count < 1:
            raise ValueError("cfg.n_envs must be positive")
        if observation_codec not in {"direct", "numpy"}:
            raise ValueError("observation_codec must be 'direct' or 'numpy'")
        for name, value in (
            ("startup_timeout", startup_timeout),
            ("heartbeat_interval", heartbeat_interval),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.observation_codec = observation_codec
        self._decode = decode_observation
        self._heartbeat_interval = float(heartbeat_interval)
        self._closed = False
        self._has_reset = False
        self._next_request_id = 0
        self._context = (None, None)
        self._connections = {}
        self._pending = {}
        self.processes = []
        self.worker_pids = []
        self.worker_runtime = {}
        self.last_timings = {}
        self.startup_seconds = 0.0
        context = multiprocessing.get_context("spawn")
        startup_at = time.perf_counter()
        try:
            cache_init = provider.worker_spec()
            for worker_id in range(self.worker_count):
                parent_connection, child_connection = context.Pipe(duplex=True)
                self._connections[worker_id] = parent_connection
                process = context.Process(
                    target=_alns_worker_main,
                    args=(
                        worker_id,
                        child_connection,
                        cache_init,
                        cfg,
                        norms,
                        observation_codec,
                        package,
                    ),
                    name=f"ALNS-env-{worker_id}",
                    daemon=False,
                )
                self.processes.append(process)
                self._pending[worker_id] = ("startup", None)
                try:
                    process.start()
                    self.worker_pids.append(process.pid)
                finally:
                    # Retaining this duplicate prevents reliable EOF detection.
                    child_connection.close()
            self._collect_ready(startup_at + float(startup_timeout))
            self.startup_seconds = time.perf_counter() - startup_at
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def set_context(self, upd=None, t=None):
        """Attach progress context to long-request heartbeat messages."""
        self._context = (upd, t)

    def _require_open(self):
        if self._closed:
            raise RuntimeError("ParallelVecALNS is closed")

    def _request_id(self):
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _check_workers(self):
        for worker_id, process in enumerate(self.processes):
            if process.exitcode is not None:
                pending = self._pending.get(worker_id)
                # A traceback may arrive just after wait()'s pipe snapshot.
                # Preserve it before falling back to the process-death report.
                if pending is not None and self._connections[worker_id].poll():
                    self._receive(worker_id)
                raise RuntimeError(
                    f"ALNS worker {worker_id} pid={process.pid} died "
                    f"exitcode={process.exitcode}, pending={pending}"
                )

    def _heartbeat(self, pending, phase, started, last_report):
        now = time.perf_counter()
        if now - last_report >= self._heartbeat_interval:
            upd, t = self._context
            print(
                f"[waiting upd={upd} t={t}] phase={phase} "
                f"workers={sorted(pending)} elapsed={now - started:.1f}s",
                flush=True,
            )
            return now
        return last_report

    def _wait_objects(self, pending):
        # Include all sentinels: an already-answered worker can still die while
        # another worker is calculating. Observe that death in this operation.
        return [self._connections[i] for i in pending] + [
            process.sentinel for process in self.processes
        ]

    def _receive(self, worker_id):
        try:
            message = self._connections[worker_id].recv()
        except (EOFError, OSError) as exc:
            process = self.processes[worker_id]
            raise RuntimeError(
                f"ALNS worker {worker_id} pid={process.pid} pipe failed; "
                f"exitcode={process.exitcode}, "
                f"pending={self._pending.get(worker_id)}"
            ) from exc
        if not isinstance(message, tuple) or not message:
            raise RuntimeError(f"invalid reply from ALNS worker {worker_id}")
        if message[0] == "error":
            if len(message) != 5:
                raise RuntimeError(f"malformed ERROR from worker {worker_id}")
            _, request_id, reported_id, phase, detail = message
            if reported_id != worker_id:
                raise RuntimeError("ERROR worker ID does not match its pipe")
            expected = self._pending.get(worker_id)
            # Failures before recv/parsing may legitimately lack a request ID.
            if (
                request_id is not None
                and expected is not None
                and request_id != expected[1]
            ):
                raise RuntimeError(
                    f"worker {worker_id} ERROR request mismatch: "
                    f"expected={expected[1]}, got={request_id}; {detail}"
                )
            raise RuntimeError(
                f"ALNS worker {worker_id} failed in {phase}, "
                f"request_id={request_id}:\n{detail}"
            )
        return message

    def _collect_ready(self, deadline):
        pending = set(range(self.worker_count))
        started = last_report = time.perf_counter()
        package_dir = Path(__file__).resolve().parent.parent / self.package
        expected_ppo = _normal_path(package_dir / "ppo.py")
        expected_alns = _normal_path(package_dir / "alns.py")
        while pending:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"ALNS startup timed out; pending workers={sorted(pending)}"
                )
            ready = wait(
                self._wait_objects(pending), timeout=min(5.0, remaining)
            )
            # Drain an ERROR reply before reporting its process sentinel.
            for worker_id in sorted(pending):
                if self._connections[worker_id] not in ready:
                    continue
                message = self._receive(worker_id)
                if (
                    len(message) != 4
                    or message[0] != "ready"
                    or message[1] != worker_id
                    or message[2] != self.processes[worker_id].pid
                ):
                    raise RuntimeError(
                        f"invalid READY from worker {worker_id}"
                    )
                runtime = message[3]
                if (
                    _normal_path(runtime["executable"])
                    != _normal_path(sys.executable)
                    or tuple(runtime["python_version"])
                    != tuple(sys.version_info[:3])
                    or _normal_path(runtime["ppo_module"]) != expected_ppo
                    or _normal_path(runtime["alns_module"]) != expected_alns
                    or runtime["cuda_initialized"]
                ):
                    raise RuntimeError(
                        f"worker {worker_id} runtime/source mismatch: {runtime}"
                    )
                self.worker_runtime[worker_id] = runtime
                self._pending.pop(worker_id)
                pending.remove(worker_id)
            self._check_workers()
            last_report = self._heartbeat(
                pending, "startup", started, last_report
            )

    def _dispatch(self, phase, arguments):
        """Send every command first, then collect every reply in env-ID order."""
        self._require_open()
        if self._pending:
            raise RuntimeError(
                "cannot dispatch with outstanding ALNS requests"
            )
        self._check_workers()
        started = time.perf_counter()
        metrics = {
            "send_seconds": 0.0,
            "receive_decode_seconds": 0.0,
            "wait_seconds": 0.0,
            "wall_seconds": 0.0,
        }
        pending = set(arguments)
        for worker_id in sorted(arguments):
            if worker_id not in self._connections:
                raise ValueError(f"unknown ALNS worker {worker_id}")
            request_id = self._request_id()
            self._pending[worker_id] = (phase, request_id)
            send_started = time.perf_counter()
            self._connections[worker_id].send(
                (phase, request_id, arguments[worker_id])
            )
            metrics["send_seconds"] += time.perf_counter() - send_started
        results = {}
        last_report = time.perf_counter()
        while pending:
            wait_started = time.perf_counter()
            ready = wait(self._wait_objects(pending), timeout=5.0)
            metrics["wait_seconds"] += time.perf_counter() - wait_started
            for worker_id in sorted(pending):
                if self._connections[worker_id] not in ready:
                    continue
                receive_started = time.perf_counter()
                message = self._receive(worker_id)
                expected = self._pending[worker_id]
                if (
                    len(message) != 4
                    or message[0] != "ok"
                    or message[1] != expected[1]
                    or message[2] != worker_id
                ):
                    raise RuntimeError(
                        f"invalid {phase} reply from worker {worker_id}: "
                        f"expected request={expected[1]}"
                    )
                payload = message[3]
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"invalid payload from worker {worker_id}"
                    )
                payload["observation"] = self._decode(
                    payload["observation"], codec=self.observation_codec
                )
                results[worker_id] = payload
                self._pending.pop(worker_id)
                pending.remove(worker_id)
                metrics["receive_decode_seconds"] += (
                    time.perf_counter() - receive_started
                )
            self._check_workers()
            last_report = self._heartbeat(pending, phase, started, last_report)
        metrics["wall_seconds"] = time.perf_counter() - started
        return results, metrics

    def _record_timings(
        self, started, steps, resets, step_metrics, reset_metrics
    ):
        workers = {}
        for phase, results in (("step", steps), ("reset", resets)):
            for worker_id, payload in results.items():
                workers.setdefault(worker_id, {})[phase] = payload["timings"]
        primary = steps if steps else resets
        durations = {
            i: payload["timings"]["env_seconds"]
            for i, payload in primary.items()
        }
        slowest = max(durations, key=durations.get) if durations else None
        self.last_timings = {
            "env_seconds": time.perf_counter() - started,
            "step_seconds": step_metrics.get("wall_seconds", 0.0),
            "reset_seconds": reset_metrics.get("wall_seconds", 0.0),
            "workers": workers,
            "slowest_worker": slowest,
            "worker_max_seconds": max(durations.values(), default=0.0),
            "worker_mean_seconds": (
                statistics.mean(durations.values()) if durations else 0.0
            ),
        }
        for name in ("send_seconds", "receive_decode_seconds", "wait_seconds"):
            self.last_timings[name] = step_metrics.get(
                name, 0.0
            ) + reset_metrics.get(name, 0.0)

    def reset(self):
        self._require_open()
        started = time.perf_counter()
        try:
            # Keep the only selection RNG and its consumption order in parent.
            refs = {
                i: self.provider.sample_ref() for i in range(self.worker_count)
            }
            results, metrics = self._dispatch("reset", refs)
            observations = [
                results[i]["observation"] for i in range(self.worker_count)
            ]
            self._has_reset = True
            self._record_timings(started, {}, results, {}, metrics)
            return observations
        except BaseException:
            self.close()
            raise

    def step(self, actions):
        self._require_open()
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        actions = list(actions)
        if len(actions) != self.worker_count:
            raise ValueError(
                f"expected {self.worker_count} actions, got {len(actions)}"
            )
        action_map = {}
        for worker_id, action in enumerate(actions):
            if isinstance(action, bool):
                raise ValueError(f"boolean action for env {worker_id}")
            action = operator.index(action)
            if not 0 <= action < self.cfg.n_actions:
                raise ValueError(
                    f"invalid action for env {worker_id}: {action}"
                )
            action_map[worker_id] = action
        started = time.perf_counter()
        try:
            results, step_metrics = self._dispatch("step", action_map)
            done_ids = [
                i for i in range(self.worker_count) if results[i]["done"]
            ]
            resets, reset_metrics = {}, {}
            if done_ids:
                refs = {i: self.provider.sample_ref() for i in done_ids}
                resets, reset_metrics = self._dispatch("reset", refs)
            observations, rewards, dones, infos = [], [], [], []
            for worker_id in range(self.worker_count):
                payload = results[worker_id]
                observation = (
                    resets[worker_id]["observation"]
                    if worker_id in resets
                    else payload["observation"]
                )
                observations.append(observation)
                rewards.append(payload["reward"])
                dones.append(payload["done"])
                infos.append(payload["info"])
            self._record_timings(
                started, results, resets, step_metrics, reset_metrics
            )
            return observations, rewards, dones, infos
        except BaseException:
            self.close()
            raise

    def close(self, timeout=5.0):
        """Best-effort bounded cleanup, including partially started workers.

        Only idle workers receive CLOSE. For an outstanding request, do not
        enqueue another command behind a potentially blocked large response.
        Such workers are terminated after the shared grace period. A tiny CLOSE
        on an idle dedicated pipe avoids ordinary backpressure, but send() is
        still a blocking OS API and is not a general live-hang guarantee.
        """
        if self._closed:
            return
        self._closed = True
        timeout = max(0.0, float(timeout))
        survivors = []
        for worker_id, process in enumerate(self.processes):
            try:
                if (
                    process.pid is not None
                    and process.is_alive()
                    and worker_id not in self._pending
                ):
                    self._connections[worker_id].send(
                        ("close", self._request_id(), None)
                    )
            except (EOFError, OSError, ValueError, KeyboardInterrupt):
                pass
        deadline = time.perf_counter() + timeout
        for process in self.processes:
            try:
                if process.pid is not None:
                    process.join(max(0.0, deadline - time.perf_counter()))
            except (OSError, ValueError, KeyboardInterrupt):
                pass
        for process in self.processes:
            try:
                if process.pid is not None and process.is_alive():
                    process.terminate()
            except (OSError, ValueError, KeyboardInterrupt):
                pass
        deadline = time.perf_counter() + timeout
        for process in self.processes:
            try:
                if process.pid is not None:
                    process.join(max(0.0, deadline - time.perf_counter()))
                    if process.is_alive():
                        survivors.append(process.pid)
                    else:
                        process.close()
            except (OSError, ValueError, KeyboardInterrupt):
                pass
        for connection in self._connections.values():
            try:
                connection.close()
            except OSError:
                pass
        self._pending.clear()
        if survivors:
            warnings.warn(
                f"ALNS cleanup deadline exceeded; inspect worker PIDs "
                f"{survivors}",
                RuntimeWarning,
                stacklevel=2,
            )
