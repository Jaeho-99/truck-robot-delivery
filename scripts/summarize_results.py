"""Build cumulative per-size CSV summaries from experiment artifacts.

The script keeps filenames and rows human-readable: experiments are identified
internally by their model, reward mode, arguments, PPO configuration, problem
parameters, and test protocol. No opaque run ID is written to the tables.

Run from the repository root::

    python scripts/summarize_results.py
    python scripts/summarize_results.py --size 20

Run this script after each new training or test setting. Model-specific
artifact filenames are intentionally simple and are overwritten when exactly
the same model/reward/tag slot is executed again; rows already copied into the
cumulative tables remain available.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output"
MODELS_DIR = REPO_ROOT / "models"

# Row order written to both tables; models outside this list sort last.
METHOD_ORDER = ("alns", "ppo_alns", "gnn_ppo_alns")

TRAINED_METHODS = ("gnn_ppo_alns", "ppo_alns")
COMPARED_METHODS = METHOD_ORDER
SIZE_PATTERN = re.compile(r"n(5|10|20|50|100)\Z")

TRAINING_TABLE = "training_time_n{size}.csv"
COMPARISON_TABLE = "comparison_n{size}.csv"

TRAINING_HEAD = (
    "recorded_at", "model", "tag", "size", "reward_mode",
    "train_time_s", "train_time_hms", "time_source", "episodes",
    "final_step", "checkpoint",
)
COMPARISON_HEAD = (
    "recorded_at", "model", "tag", "size", "reward_mode",
    "selection_mode", "n_instances", "n_seeds", "n_runs",
    "obj_min", "obj_mean", "obj_max", "runtime_s_min",
    "runtime_s_mean", "runtime_s_max", "checkpoint",
)

_TRAINING_VALUE_FIELDS = {
    "recorded_at", "train_time_s", "train_time_hms", "time_source",
    "episodes", "final_step",
}
_COMPARISON_VALUE_FIELDS = {
    "recorded_at", "n_runs", "obj_min", "obj_mean", "obj_max",
    "runtime_s_min", "runtime_s_mean", "runtime_s_max",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing")
        return list(reader)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _artifact_time(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(
        timespec="seconds")


def _hms(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600}:{total // 60 % 60:02d}:{total % 60:02d}"


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested metadata into stable, spreadsheet-friendly columns."""
    if not isinstance(value, Mapping):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten(name, item))
        elif isinstance(item, (list, tuple)):
            result[name] = json.dumps(item, ensure_ascii=False,
                                      separators=(",", ":"))
        else:
            result[name] = item
    return result


def _metadata_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    groups = (
        ("arg", metadata.get("arguments")),
        ("ppo", metadata.get("ppo_config")),
        ("param", metadata.get("parameters")),
        ("data", metadata.get("checkpoint_metadata")),
    )
    for prefix, value in groups:
        if isinstance(value, Mapping):
            fields.update(_flatten(prefix, value))
    return fields


def _validate_metadata(metadata: Mapping[str, Any], source: Path, *,
                       method: str, size: int, tag: str,
                       reward_mode: str | None) -> None:
    """Reject stale or misnamed metadata before joining it to a result."""
    expected = {
        "model": method,
        "size": size,
        "tag": tag,
    }
    if reward_mode is not None:
        expected["reward_mode"] = reward_mode
    for field, value in expected.items():
        stored = metadata.get(field)
        if field == "tag":
            stored = stored or ""
        if stored != value:
            raise ValueError(
                f"metadata {field}={stored!r} does not match {value!r}: "
                f"{source}")


def _method_and_tag(directory_name: str) -> tuple[str, str] | None:
    """Return the canonical method and optional semantic directory tag."""
    for method in ("gnn_ppo_alns", "ppo_alns", "alns", "exact"):
        if directory_name == method:
            return method, ""
        if directory_name.startswith(f"{method}_"):
            return method, directory_name[len(method) + 1:]
    return None


def _runs(methods: tuple[str, ...]) -> list[tuple[str, str, int, Path]]:
    """Discover ``(method, tag, size, directory)`` for selected methods."""
    if not OUTPUT_DIR.is_dir():
        return []
    found = []
    for method_dir in sorted(path for path in OUTPUT_DIR.iterdir()
                             if path.is_dir()):
        parsed = _method_and_tag(method_dir.name)
        if parsed is None or parsed[0] not in methods:
            continue
        method, tag = parsed
        for size_dir in sorted(path for path in method_dir.iterdir()
                               if path.is_dir()):
            match = SIZE_PATTERN.fullmatch(size_dir.name)
            if match is not None:
                found.append((method, tag, int(match.group(1)), size_dir))
    return found


def _checkpoint_path(method: str, tag: str, size: int,
                     reward_token: str) -> Path:
    suffix = f"_{tag}" if tag else ""
    return MODELS_DIR / f"{method}_n{size}_{reward_token}{suffix}.pt"


def _checkpoint_fields(path: Path) -> dict[str, Any]:
    """Read settings from a legacy run that has no metadata snapshot."""
    if not path.is_file():
        print(f"  ! checkpoint missing, settings left blank: {path}")
        return {}
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to read legacy checkpoints; rerun training "
            "to create a JSON metadata snapshot") from exc
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    metadata = {
        "ppo_config": checkpoint.get("config") or {},
        "checkpoint_metadata": checkpoint.get("metadata") or {},
    }
    return _metadata_fields(metadata)


def _finite_float(row: Mapping[str, str], field: str, path: Path) -> float:
    if field not in row or row[field] == "":
        raise ValueError(f"missing {field!r} column value in {path}")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field!r} value in {path}: {value}")
    return value


def _consistent_reward(rows: list[dict[str, str]], path: Path,
                       expected: str | None = None) -> str:
    modes = {row.get("reward_mode", "") for row in rows}
    if len(modes) != 1:
        raise ValueError(f"mixed reward modes in {path}: {sorted(modes)}")
    mode = modes.pop()
    if expected is not None and mode != expected:
        raise ValueError(
            f"reward mode {mode!r} in {path} does not match {expected!r}")
    return mode


def _training_rows(
    now: str,
) -> tuple[dict[int, list[dict[str, Any]]], set[Path]]:
    by_size: dict[int, list[dict[str, Any]]] = {}
    seen_checkpoints: set[Path] = set()
    for method, tag, size, directory in _runs(TRAINED_METHODS):
        for log_path in sorted(directory.glob("train_reward_*.csv")):
            try:
                reward_token = log_path.stem[len("train_"):]
                expected_reward = reward_token.removeprefix("reward_")
                log = _read_csv(log_path)
                if not log:
                    raise ValueError("training log is empty")
                reward_mode = _consistent_reward(
                    log, log_path, expected_reward)
                elapsed_values = [
                    _finite_float(item, "elapsed_s", log_path) for item in log
                ]
                checkpoint = _checkpoint_path(
                    method, tag, size, reward_token).resolve()
                seen_checkpoints.add(checkpoint)
                metadata_path = directory / f"training_metadata_{reward_token}.json"
                metadata = (_read_json(metadata_path)
                            if metadata_path.is_file() else {})
                if metadata:
                    _validate_metadata(
                        metadata, metadata_path, method=method, size=size,
                        tag=tag, reward_mode=reward_mode)
                    elapsed = float(metadata["train_time_s"])
                    if not math.isfinite(elapsed) or elapsed < 0.0:
                        raise ValueError(
                            f"invalid train_time_s in {metadata_path}")
                    recorded_at = str(metadata.get("completed_at") or now)
                    time_source = "training_metadata"
                    checkpoint_label = str(metadata.get("checkpoint") or checkpoint)
                    settings = _metadata_fields(metadata)
                else:
                    elapsed = max(elapsed_values)
                    recorded_at = _artifact_time(log_path)
                    time_source = "episode_log_max (legacy; final update excluded)"
                    checkpoint_label = str(checkpoint)
                    settings = _checkpoint_fields(checkpoint)
                step_values = [int(item["step"]) for item in log]
                row: dict[str, Any] = {
                    "recorded_at": recorded_at,
                    "model": method,
                    "tag": tag,
                    "size": size,
                    "reward_mode": reward_mode,
                    "train_time_s": round(elapsed, 1),
                    "train_time_hms": _hms(elapsed),
                    "time_source": time_source,
                    "episodes": len(log),
                    "final_step": max(step_values),
                    "checkpoint": checkpoint_label,
                }
                row.update({key: value for key, value in settings.items()
                            if key not in row})
                by_size.setdefault(size, []).append(row)
            except (KeyError, TypeError, ValueError, RuntimeError, OSError,
                    json.JSONDecodeError) as exc:
                print(f"  ! invalid training artifact skipped: {log_path}: {exc}")
    return by_size, seen_checkpoints


def _result_files(method: str, directory: Path) -> list[tuple[Path, str]]:
    if method == "alns":
        summary = directory / "summary.csv"
        return [(summary, "")] if summary.is_file() else []
    return [
        (path, path.stem[len("test_"):])
        for path in sorted(directory.glob("test_reward_*.csv"))
    ]


def _test_metadata(method: str, directory: Path,
                   reward_token: str) -> dict[str, Any]:
    name = ("test_metadata.json" if method == "alns"
            else f"test_metadata_{reward_token}.json")
    path = directory / name
    return _read_json(path) if path.is_file() else {}


def _legacy_selection_mode(method: str, directory: Path,
                           reward_token: str,
                           first: Mapping[str, str]) -> str:
    if method == "alns":
        return "roulette"
    instance_id = first.get("instance_id", "")
    seed = first.get("seed", "")
    path = directory / f"{instance_id}_{reward_token}_s{seed}.json"
    if path.is_file():
        data = _read_json(path)
        stats = data.get("stats")
        if isinstance(stats, Mapping):
            return str(stats.get("selection_mode") or "")
    return ""


def _comparison_rows(now: str) -> dict[int, list[dict[str, Any]]]:
    by_size: dict[int, list[dict[str, Any]]] = {}
    for method, tag, size, directory in _runs(COMPARED_METHODS):
        for result_path, reward_token in _result_files(method, directory):
            try:
                results = _read_csv(result_path)
                if not results:
                    raise ValueError("test result is empty")
                expected = (reward_token.removeprefix("reward_")
                            if reward_token else None)
                reward_mode = (_consistent_reward(results, result_path, expected)
                               if method != "alns" else "")
                pairs = [(item.get("instance_id", ""), item.get("seed", ""))
                         for item in results]
                if any(not instance or seed == "" for instance, seed in pairs):
                    raise ValueError("instance_id or seed is missing")
                if len(set(pairs)) != len(pairs):
                    raise ValueError("duplicate instance_id/seed result rows")
                objectives = [
                    _finite_float(item, "obj", result_path) for item in results
                ]
                runtimes = [
                    _finite_float(item, "runtime_s", result_path)
                    for item in results
                ]
                metadata = _test_metadata(method, directory, reward_token)
                checkpoint = ""
                settings: dict[str, Any] = {}
                if metadata:
                    _validate_metadata(
                        metadata, result_path, method=method, size=size,
                        tag=tag,
                        reward_mode=(reward_mode if method != "alns" else None))
                    recorded_at = str(metadata.get("completed_at") or now)
                    checkpoint = str(metadata.get("checkpoint") or "")
                    selection_mode = str(metadata.get("selection_mode") or "")
                    settings = _metadata_fields(metadata)
                else:
                    recorded_at = _artifact_time(result_path)
                    selection_mode = _legacy_selection_mode(
                        method, directory, reward_token, results[0])
                    if method != "alns":
                        checkpoint_path = _checkpoint_path(
                            method, tag, size, reward_token).resolve()
                        checkpoint = str(checkpoint_path)
                        settings = _checkpoint_fields(checkpoint_path)
                instances = {instance for instance, _ in pairs}
                seeds = {int(seed) for _, seed in pairs}
                row: dict[str, Any] = {
                    "recorded_at": recorded_at,
                    "model": method,
                    "tag": tag,
                    "size": size,
                    "reward_mode": reward_mode,
                    "selection_mode": selection_mode,
                    "n_instances": len(instances),
                    "n_seeds": len(seeds),
                    "n_runs": len(results),
                    "obj_min": round(min(objectives), 4),
                    "obj_mean": round(statistics.fmean(objectives), 4),
                    "obj_max": round(max(objectives), 4),
                    "runtime_s_min": round(min(runtimes), 3),
                    "runtime_s_mean": round(statistics.fmean(runtimes), 3),
                    "runtime_s_max": round(max(runtimes), 3),
                    "checkpoint": checkpoint,
                }
                row.update({key: value for key, value in settings.items()
                            if key not in row})
                by_size.setdefault(size, []).append(row)
            except (KeyError, TypeError, ValueError, RuntimeError, OSError,
                    json.JSONDecodeError) as exc:
                print(f"  ! invalid test artifact skipped: {result_path}: {exc}")
    return by_size


def _identity(row: Mapping[str, Any], value_fields: set[str]) -> str:
    """Return an internal setting signature; it is never written to CSV.

    Blank entries are dropped so a row read back from CSV -- which carries
    every union column, blank-filled -- matches the freshly built row that
    simply lacks those keys.
    """
    settings = {}
    for key, value in row.items():
        if key in value_fields:
            continue
        text = "" if value is None else str(value)
        if text != "":
            settings[key] = text
    return json.dumps(settings, ensure_ascii=False, sort_keys=True,
                      default=str, separators=(",", ":"))


def _model_rank(row: Mapping[str, Any]) -> int:
    """Sort key ordering rows by METHOD_ORDER; unknown models come last."""
    model = str(row.get("model", ""))
    return (METHOD_ORDER.index(model) if model in METHOD_ORDER
            else len(METHOD_ORDER))


def _merge_table(path: Path, rows: list[dict[str, Any]],
                 head: tuple[str, ...], value_fields: set[str]) -> None:
    """Update the same setting in place and append genuinely new settings.

    Rows are written grouped by model in METHOD_ORDER. The sort is stable,
    so within one model the accumulated history keeps its order and a new
    setting still lands below that model's earlier rows.
    """
    existing: list[dict[str, Any]] = []
    fieldnames = list(head)
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing.append(row)
            for field in reader.fieldnames or ():
                if field not in fieldnames:
                    fieldnames.append(field)

    positions = {
        _identity(row, value_fields): index
        for index, row in enumerate(existing)
    }
    added = refreshed = 0
    for row in rows:
        signature = _identity(row, value_fields)
        if signature in positions:
            existing[positions[signature]] = row
            refreshed += 1
        else:
            positions[signature] = len(existing)
            existing.append(row)
            added += 1
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, restval="",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(existing, key=_model_rank))
    print(f"{path}  ({added} new, {refreshed} refreshed, "
          f"{len(existing)} total)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize output/ into cumulative per-size training and "
                    "test comparison tables")
    parser.add_argument(
        "--size", type=int, choices=(5, 10, 20, 50, 100), action="append",
        help="only summarize this size (repeatable; default: all found)")
    args = parser.parse_args()

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    training, seen_checkpoints = _training_rows(now)
    comparison = _comparison_rows(now)
    if MODELS_DIR.is_dir():
        for checkpoint in sorted(path.resolve()
                                 for path in MODELS_DIR.glob("*.pt")):
            if checkpoint not in seen_checkpoints:
                print(f"  ! no training log for {checkpoint.name}; "
                      "training time cannot be reported")

    sizes = sorted(set(training) | set(comparison))
    if args.size is not None:
        sizes = [size for size in sizes if size in args.size]
    if not sizes:
        print("no training logs or test results found under output/")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        if size in training:
            _merge_table(
                OUTPUT_DIR / TRAINING_TABLE.format(size=size),
                training[size], TRAINING_HEAD, _TRAINING_VALUE_FIELDS)
        if size in comparison:
            _merge_table(
                OUTPUT_DIR / COMPARISON_TABLE.format(size=size),
                comparison[size], COMPARISON_HEAD, _COMPARISON_VALUE_FIELDS)
            present = {row["model"] for row in comparison[size]}
            missing = [method for method in COMPARED_METHODS
                       if method not in present]
            if missing:
                print(f"  ! n{size} comparison is incomplete; missing: "
                      f"{', '.join(missing)}")


if __name__ == "__main__":
    main()
