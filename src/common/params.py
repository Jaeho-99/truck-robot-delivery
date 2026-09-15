"""Load shared problem parameters and validated, immutable NPZ instances.

All solvers use this single definition of vehicle capacities, travel
coefficients, and input validation. Geometry is precomputed by
``scripts/preprocess.py``; loading an instance never rebuilds its matrix.

Example from a solver or a Python session with ``src`` on the path::

    config = load_params()
    path = REPO_ROOT / "data/processed/test/n5/test_n5_000.npz"
    config, instance = load_problem(path)
    print(config.fleet.n_trucks_for(5))
    print(instance.d.shape)

Returned arrays are read-only. Build a separate solution object when
changing routes; do not mutate the shared problem during a search.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import yaml
from yaml.constructor import ConstructorError

from common.sizes import SUPPORTED_SIZES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = REPO_ROOT / "configs" / "params.yaml"
CELL_COORD_PATH = REPO_ROOT / "data" / "cells_ulsan_namgu.csv"
DONG_POLYGON_PATH = REPO_ROOT / "data" / "ulsan_namgu_dong_boundaries.geojson"

POLYGON_ZONE_METHOD = "point_in_polygon_with_nearest_katec_cell_fallback"
CELL_ZONE_METHOD = "nearest_katec_cell_zone"


def _require_real(
    value: Any, name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result}")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {result}")
    return result


def _require_int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class GeometryParams:
    metric: str
    scaling_factor: float
    path_convention: str
    n_samples: int

    def __post_init__(self) -> None:
        if self.metric != "manhattan":
            raise ValueError("geometry.metric must be 'manhattan'")
        if self.path_convention not in {"x_first", "y_first", "average"}:
            raise ValueError(
                "geometry.path_convention must be x_first, y_first, or average"
            )
        _require_real(
            self.scaling_factor, "geometry.scaling_factor", positive=True
        )
        _require_int(self.n_samples, "geometry.n_samples", positive=True)


@dataclass(frozen=True, slots=True)
class FleetParams:
    n_trucks: Mapping[int, int]
    n_robots_per_truck: int

    def __post_init__(self) -> None:
        if not isinstance(self.n_trucks, Mapping):
            raise TypeError("fleet.n_trucks must be a size-to-count mapping")
        copied = dict(self.n_trucks)
        # Older YAML files remain usable for their configured sizes.
        # A newly requested size must have an explicit fleet entry.
        required_sizes = {5, 10, 20, 50, 100}
        if not required_sizes <= set(copied) <= set(SUPPORTED_SIZES):
            raise ValueError(
                "fleet.n_trucks must contain sizes 5, 10, 20, 50, and 100; "
                f"additional sizes may be configured from {SUPPORTED_SIZES}"
            )
        for size, count in copied.items():
            _require_int(size, "fleet.n_trucks size")
            _require_int(count, f"fleet.n_trucks[{size}]", positive=True)
        _require_int(
            self.n_robots_per_truck, "fleet.n_robots_per_truck", positive=True
        )
        object.__setattr__(self, "n_trucks", MappingProxyType(copied))

    def n_trucks_for(self, size: int) -> int:
        _require_int(size, "size")
        try:
            return self.n_trucks[size]
        except KeyError as exc:
            if size in SUPPORTED_SIZES:
                raise ValueError(
                    f"fleet.n_trucks[{size}] must be configured "
                    f"before running n{size} experiments"
                ) from exc
            choices = ", ".join(
                str(value) for value in sorted(SUPPORTED_SIZES)
            )
            raise ValueError(
                f"unsupported size {size}; choose from {choices}"
            ) from exc


@dataclass(frozen=True, slots=True)
class TruckParams:
    speed_kmh: float
    capacity: int
    range_km: float
    fixed_cost: float
    fuel_cost_per_min: float
    env_cost_per_min: float
    service_time_min: float

    def __post_init__(self) -> None:
        _require_real(self.speed_kmh, "truck.speed_kmh", positive=True)
        _require_int(self.capacity, "truck.capacity", positive=True)
        _require_real(self.range_km, "truck.range_km", positive=True)
        for name in (
            "fixed_cost",
            "fuel_cost_per_min",
            "env_cost_per_min",
            "service_time_min",
        ):
            _require_real(
                getattr(self, name), f"truck.{name}", nonnegative=True
            )


@dataclass(frozen=True, slots=True)
class RobotParams:
    speed_kmh: float
    capacity: int
    range_km: float
    fixed_cost: float
    fuel_cost_per_min: float
    env_cost_per_min: float
    service_time_min: float
    unload_time_min: float
    load_time_min: float

    def __post_init__(self) -> None:
        _require_real(self.speed_kmh, "robot.speed_kmh", positive=True)
        _require_int(self.capacity, "robot.capacity", positive=True)
        _require_real(self.range_km, "robot.range_km", positive=True)
        for name in (
            "fixed_cost",
            "fuel_cost_per_min",
            "env_cost_per_min",
            "service_time_min",
            "unload_time_min",
            "load_time_min",
        ):
            _require_real(
                getattr(self, name), f"robot.{name}", nonnegative=True
            )


@dataclass(frozen=True, slots=True)
class Params:
    version: str
    geometry: GeometryParams
    fleet: FleetParams
    truck: TruckParams
    robot: RobotParams
    lateness_cost_per_min: float
    demand_per_customer: int
    time_window_hours: float

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if not isinstance(self.geometry, GeometryParams):
            raise TypeError("geometry must be GeometryParams")
        if not isinstance(self.fleet, FleetParams):
            raise TypeError("fleet must be FleetParams")
        if not isinstance(self.truck, TruckParams):
            raise TypeError("truck must be TruckParams")
        if not isinstance(self.robot, RobotParams):
            raise TypeError("robot must be RobotParams")
        _require_real(
            self.lateness_cost_per_min,
            "lateness.cost_per_min",
            nonnegative=True,
        )
        demand = _require_int(
            self.demand_per_customer, "demand_per_customer", positive=True
        )
        if demand != 1:
            raise ValueError("demand_per_customer is fixed at 1")
        _require_real(
            self.time_window_hours, "time_window.hours", positive=True
        )

    @property
    def deadline_min(self) -> float:
        return 60.0 * self.time_window_hours

    @property
    def preproc_hash(self) -> str:
        return compute_preproc_hash(self)


@dataclass(frozen=True, slots=True, eq=False)
class Instance:
    source_path: Path
    node_label: np.ndarray
    node_type: np.ndarray
    node_xy: np.ndarray
    node_zone: np.ndarray
    customer_idx: np.ndarray
    parking_idx: np.ndarray
    depot_idx: np.ndarray
    d: np.ndarray
    tau_truck: np.ndarray
    tau_robot: np.ndarray
    alpha_traffic: np.ndarray
    alpha_ped: np.ndarray
    meta: Mapping[str, Any]
    demand: np.ndarray
    e: np.ndarray
    l: np.ndarray  # noqa: E741 - Keep the public time-window notation.

    @property
    def n_nodes(self) -> int:
        return int(self.node_label.size)

    @property
    def depot_index(self) -> int:
        return int(self.depot_idx[0])


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.Node, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _mapping(value: Any, path: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _exact_keys(
    value: Any, expected: set[Any], path: str
) -> Mapping[Any, Any]:
    mapping = _mapping(value, path)
    keys = set(mapping)
    missing = expected - keys
    unknown = keys - expected
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing={sorted(missing, key=str)!r}")
        if unknown:
            parts.append(f"unknown={sorted(unknown, key=str)!r}")
        raise ValueError(f"invalid keys at {path}: {', '.join(parts)}")
    return mapping


def _float_field(
    mapping: Mapping[Any, Any],
    key: str,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    return _require_real(
        mapping[key],
        f"{path}.{key}",
        positive=positive,
        nonnegative=nonnegative,
    )


def _int_field(
    mapping: Mapping[Any, Any], key: str, path: str, *, positive: bool = False
) -> int:
    return _require_int(mapping[key], f"{path}.{key}", positive=positive)


def _parse_params(raw: Any) -> Params:
    root = _exact_keys(
        raw,
        {
            "version",
            "geometry",
            "fleet",
            "truck",
            "robot",
            "lateness",
            "demand_per_customer",
            "time_window",
        },
        "root",
    )

    version = root["version"]
    if not isinstance(version, str) or not version.strip():
        raise TypeError("version must be a non-empty string")

    geometry_raw = _exact_keys(
        root["geometry"],
        {
            "metric",
            "scaling_factor",
            "path_convention",
            "n_samples",
        },
        "geometry",
    )
    metric = geometry_raw["metric"]
    convention = geometry_raw["path_convention"]
    if not isinstance(metric, str):
        raise TypeError("geometry.metric must be a string")
    if not isinstance(convention, str):
        raise TypeError("geometry.path_convention must be a string")
    geometry = GeometryParams(
        metric=metric,
        scaling_factor=_float_field(
            geometry_raw, "scaling_factor", "geometry", positive=True
        ),
        path_convention=convention,
        n_samples=_int_field(
            geometry_raw, "n_samples", "geometry", positive=True
        ),
    )

    fleet_raw = _exact_keys(
        root["fleet"],
        {
            "n_trucks",
            "n_robots_per_truck",
        },
        "fleet",
    )
    truck_counts_raw = _mapping(fleet_raw["n_trucks"], "fleet.n_trucks")
    truck_counts: dict[int, int] = {}
    for size, count in truck_counts_raw.items():
        checked_size = _require_int(size, "fleet.n_trucks size")
        truck_counts[checked_size] = _require_int(
            count, f"fleet.n_trucks[{checked_size}]", positive=True
        )
    fleet = FleetParams(
        n_trucks=truck_counts,
        n_robots_per_truck=_int_field(
            fleet_raw, "n_robots_per_truck", "fleet", positive=True
        ),
    )

    truck_raw = _exact_keys(
        root["truck"],
        {
            "speed_kmh",
            "capacity",
            "range_km",
            "fixed_cost",
            "fuel_cost_per_min",
            "env_cost_per_min",
            "service_time_min",
        },
        "truck",
    )
    truck = TruckParams(
        speed_kmh=_float_field(truck_raw, "speed_kmh", "truck", positive=True),
        capacity=_int_field(truck_raw, "capacity", "truck", positive=True),
        range_km=_float_field(truck_raw, "range_km", "truck", positive=True),
        fixed_cost=_float_field(
            truck_raw, "fixed_cost", "truck", nonnegative=True
        ),
        fuel_cost_per_min=_float_field(
            truck_raw, "fuel_cost_per_min", "truck", nonnegative=True
        ),
        env_cost_per_min=_float_field(
            truck_raw, "env_cost_per_min", "truck", nonnegative=True
        ),
        service_time_min=_float_field(
            truck_raw, "service_time_min", "truck", nonnegative=True
        ),
    )

    robot_raw = _exact_keys(
        root["robot"],
        {
            "speed_kmh",
            "capacity",
            "range_km",
            "fixed_cost",
            "fuel_cost_per_min",
            "env_cost_per_min",
            "service_time_min",
            "unload_time_min",
            "load_time_min",
        },
        "robot",
    )
    robot = RobotParams(
        speed_kmh=_float_field(robot_raw, "speed_kmh", "robot", positive=True),
        capacity=_int_field(robot_raw, "capacity", "robot", positive=True),
        range_km=_float_field(robot_raw, "range_km", "robot", positive=True),
        fixed_cost=_float_field(
            robot_raw, "fixed_cost", "robot", nonnegative=True
        ),
        fuel_cost_per_min=_float_field(
            robot_raw, "fuel_cost_per_min", "robot", nonnegative=True
        ),
        env_cost_per_min=_float_field(
            robot_raw, "env_cost_per_min", "robot", nonnegative=True
        ),
        service_time_min=_float_field(
            robot_raw, "service_time_min", "robot", nonnegative=True
        ),
        unload_time_min=_float_field(
            robot_raw, "unload_time_min", "robot", nonnegative=True
        ),
        load_time_min=_float_field(
            robot_raw, "load_time_min", "robot", nonnegative=True
        ),
    )

    lateness_raw = _exact_keys(root["lateness"], {"cost_per_min"}, "lateness")
    time_window_raw = _exact_keys(
        root["time_window"], {"hours"}, "time_window"
    )
    return Params(
        version=version,
        geometry=geometry,
        fleet=fleet,
        truck=truck,
        robot=robot,
        lateness_cost_per_min=_float_field(
            lateness_raw, "cost_per_min", "lateness", nonnegative=True
        ),
        demand_per_customer=_require_int(
            root["demand_per_customer"], "demand_per_customer", positive=True
        ),
        time_window_hours=_float_field(
            time_window_raw, "hours", "time_window", positive=True
        ),
    )


@lru_cache(maxsize=None)
def _load_params_resolved(path: Path) -> Params:
    if not path.is_file():
        raise FileNotFoundError(f"parameter file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_UniqueKeyLoader)
    return _parse_params(raw)


def load_params(path: str | Path | None = None) -> Params:
    """Load and strictly validate params, cached by the resolved path."""
    resolved = (
        DEFAULT_PARAMS_PATH if path is None else Path(path)
    ).expanduser()
    return _load_params_resolved(resolved.resolve())


load_params.cache_clear = _load_params_resolved.cache_clear  # type: ignore[attr-defined]
load_params.cache_info = _load_params_resolved.cache_info  # type: ignore[attr-defined]


def preproc_signature(params: Params) -> dict[str, Any]:
    """Return exactly the values that determine processed geometry arrays."""
    return {
        "geometry": {
            "metric": params.geometry.metric,
            "scaling_factor": params.geometry.scaling_factor,
            "path_convention": params.geometry.path_convention,
            "n_samples": params.geometry.n_samples,
        },
        "robot": {"speed_kmh": params.robot.speed_kmh},
        "truck": {"speed_kmh": params.truck.speed_kmh},
        "version": params.version,
    }


def preproc_hash_from_signature(signature: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_preproc_hash(params: Params) -> str:
    return preproc_hash_from_signature(preproc_signature(params))


@lru_cache(maxsize=1)
def zone_source_info() -> tuple[str, str]:
    """Return the active zone method and a hash of its immutable source data."""
    if not CELL_COORD_PATH.is_file():
        raise FileNotFoundError(
            f"KATEC cell coordinate file not found: {CELL_COORD_PATH}"
        )
    method = (
        POLYGON_ZONE_METHOD
        if DONG_POLYGON_PATH.is_file()
        else CELL_ZONE_METHOD
    )
    sources = [CELL_COORD_PATH]
    if DONG_POLYGON_PATH.is_file():
        sources.insert(0, DONG_POLYGON_PATH)
    source_digests: dict[str, str] = {}
    for path in sources:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        source_digests[path.relative_to(REPO_ROOT).as_posix()] = (
            digest.hexdigest()
        )
    canonical = json.dumps(
        {"method": method, "sources": source_digests},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return method, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON metadata key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _stored_signature(meta: Mapping[str, Any]) -> dict[str, Any]:
    required = {"version", "geometry", "truck", "robot"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"processed metadata missing {sorted(missing)!r}")
    geometry = _exact_keys(
        meta["geometry"],
        {
            "metric",
            "scaling_factor",
            "path_convention",
            "n_samples",
        },
        "meta.geometry",
    )
    truck = _exact_keys(meta["truck"], {"speed_kmh"}, "meta.truck")
    robot = _exact_keys(meta["robot"], {"speed_kmh"}, "meta.robot")
    GeometryParams(
        metric=geometry["metric"],
        scaling_factor=geometry["scaling_factor"],
        path_convention=geometry["path_convention"],
        n_samples=geometry["n_samples"],
    )
    _require_real(truck["speed_kmh"], "meta.truck.speed_kmh", positive=True)
    _require_real(robot["speed_kmh"], "meta.robot.speed_kmh", positive=True)
    if not isinstance(meta["version"], str) or not meta["version"].strip():
        raise ValueError("meta.version must be a non-empty string")
    return {
        "geometry": dict(geometry),
        "robot": dict(robot),
        "truck": dict(truck),
        "version": meta["version"],
    }


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def _check_preproc_compatibility(
    meta: Mapping[str, Any], params: Params, npz_path: Path
) -> None:
    required = {
        "instance_id",
        "n_customers",
        "n_zones",
        "num_parking_copies",
        "geometry",
        "truck",
        "robot",
        "version",
        "preproc_hash",
        "preprocess_git_commit",
        "timestamp",
        "zone_assignment_method",
        "zone_source_hash",
        "node_zone_assignment_mismatch_count",
    }
    missing = required - set(meta)
    if missing:
        raise ValueError(
            f"{npz_path}: processed metadata missing {sorted(missing)!r}"
        )
    if not isinstance(meta["instance_id"], str) or not meta["instance_id"]:
        raise TypeError(f"{npz_path}: meta.instance_id must be non-empty")
    for field in ("n_customers", "n_zones", "num_parking_copies"):
        _require_int(meta[field], f"meta.{field}", positive=True)
    if (
        not isinstance(meta["preprocess_git_commit"], str)
        or not meta["preprocess_git_commit"]
    ):
        raise TypeError(
            f"{npz_path}: meta.preprocess_git_commit must be non-empty"
        )
    if (
        not isinstance(meta["zone_assignment_method"], str)
        or not meta["zone_assignment_method"]
    ):
        raise TypeError(
            f"{npz_path}: meta.zone_assignment_method must be non-empty"
        )
    zone_source_hash = meta["zone_source_hash"]
    if (
        not isinstance(zone_source_hash, str)
        or len(zone_source_hash) != 64
        or any(char not in "0123456789abcdef" for char in zone_source_hash)
    ):
        raise ValueError(f"{npz_path}: meta.zone_source_hash is invalid")
    mismatch_count = _require_int(
        meta["node_zone_assignment_mismatch_count"],
        "meta.node_zone_assignment_mismatch_count",
    )
    if mismatch_count < 0:
        raise ValueError(
            "meta.node_zone_assignment_mismatch_count must be >= 0"
        )
    if not isinstance(meta["timestamp"], str):
        raise TypeError(f"{npz_path}: meta.timestamp must be an ISO string")
    try:
        datetime.fromisoformat(meta["timestamp"])
    except ValueError as exc:
        raise ValueError(
            f"{npz_path}: meta.timestamp is not valid ISO-8601"
        ) from exc
    current_method, current_zone_hash = zone_source_info()
    if (
        meta["zone_assignment_method"] != current_method
        or zone_source_hash != current_zone_hash
    ):
        raise ValueError(
            f"{npz_path}: zone-assignment source data changed "
            f"(stored method/hash={meta['zone_assignment_method']}/"
            f"{zone_source_hash}, current={current_method}/{current_zone_hash}); "
            "bump params.version and rerun `python scripts/preprocess.py --force`."
        )
    stored_hash = meta.get("preproc_hash")
    if (
        not isinstance(stored_hash, str)
        or len(stored_hash) != 64
        or any(char not in "0123456789abcdef" for char in stored_hash)
    ):
        raise ValueError(
            f"{npz_path}: metadata preproc_hash is missing/invalid"
        )
    stored_signature = _stored_signature(meta)
    derived_hash = preproc_hash_from_signature(stored_signature)
    if derived_hash != stored_hash:
        raise ValueError(
            f"{npz_path}: metadata signature does not match its preproc_hash; "
            "rerun `python scripts/preprocess.py --force`"
        )
    current_signature = preproc_signature(params)
    if stored_hash == params.preproc_hash:
        if stored_signature != current_signature:
            raise ValueError(
                f"{npz_path}: preprocessing hash collision detected"
            )
        return
    stored_flat = _flatten(stored_signature)
    current_flat = _flatten(current_signature)
    differing = []
    for field in sorted(set(stored_flat) | set(current_flat)):
        old = stored_flat.get(field, "<missing>")
        new = current_flat.get(field, "<missing>")
        if old != new:
            differing.append(f"  {field}: stored={old!r}, current={new!r}")
    detail = "\n".join(differing) or "  preproc_hash/format differs"
    raise ValueError(
        f"{npz_path}: processed data is stale for the selected params:\n"
        f"{detail}\nRerun `python scripts/preprocess.py --force`."
    )


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _require_dtype(array: np.ndarray, dtype: np.dtype[Any], name: str) -> None:
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {array.dtype}")


def _validate_instance_arrays(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    params: Params,
    path: Path,
) -> None:
    label = arrays["node_label"]
    node_type = arrays["node_type"]
    xy = arrays["node_xy"]
    zone = arrays["node_zone"]
    customer = arrays["customer_idx"]
    parking = arrays["parking_idx"]
    depot = arrays["depot_idx"]
    matrices = (arrays["d"], arrays["tau_truck"], arrays["tau_robot"])
    alphas = (arrays["alpha_traffic"], arrays["alpha_ped"])

    if label.ndim != 1 or label.dtype.kind != "U":
        raise TypeError("node_label must be a one-dimensional Unicode array")
    if node_type.shape != label.shape or node_type.dtype.kind != "U":
        raise TypeError(
            "node_type must be a Unicode vector matching node_label"
        )
    n_nodes = label.size
    if n_nodes == 0 or len(set(label.tolist())) != n_nodes:
        raise ValueError("node_label must be non-empty and unique")
    _require_dtype(xy, np.dtype(np.float64), "node_xy")
    if xy.shape != (n_nodes, 2) or not np.isfinite(xy).all():
        raise ValueError("node_xy must have finite shape (n_nodes, 2)")
    _require_dtype(zone, np.dtype(np.int64), "node_zone")
    if zone.shape != (n_nodes,):
        raise ValueError("node_zone must have shape (n_nodes,)")

    for name, index in (
        ("customer_idx", customer),
        ("parking_idx", parking),
        ("depot_idx", depot),
    ):
        _require_dtype(index, np.dtype(np.int64), name)
        if index.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if ((index < 0) | (index >= n_nodes)).any():
            raise ValueError(f"{name} contains an out-of-range index")
    if depot.shape != (1,):
        raise ValueError("depot_idx must contain exactly one index")
    partition = np.concatenate((customer, parking, depot))
    if partition.size != n_nodes or not np.array_equal(
        np.sort(partition), np.arange(n_nodes)
    ):
        raise ValueError("customer/parking/depot indices must partition nodes")
    expected_types = {
        "customer": customer,
        "parking": parking,
        "depot": depot,
    }
    if set(node_type.tolist()) != set(expected_types):
        raise ValueError("node_type contains an unsupported or missing type")
    for kind, index in expected_types.items():
        if not np.all(node_type[index] == kind):
            raise ValueError(f"{kind} indices disagree with node_type")

    for name, matrix in zip(("d", "tau_truck", "tau_robot"), matrices):
        _require_dtype(matrix, np.dtype(np.float64), name)
        if matrix.shape != (n_nodes, n_nodes):
            raise ValueError(f"{name} must have shape (n_nodes, n_nodes)")
        if not np.isfinite(matrix).all() or (matrix < 0.0).any():
            raise ValueError(f"{name} must contain finite nonnegative values")
        if np.count_nonzero(np.diag(matrix)):
            raise ValueError(f"{name} diagonal must be exactly zero")

    if meta.get("geometry", {}).get("path_convention") == "average":
        for name, matrix in zip(("d", "tau_truck", "tau_robot"), matrices):
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
                raise ValueError(f"{name} must be symmetric for average paths")

    n_zones = meta.get("n_zones")
    if (
        isinstance(n_zones, bool)
        or not isinstance(n_zones, int)
        or n_zones <= 0
    ):
        raise ValueError("meta.n_zones must be a positive integer")
    for name, alpha in zip(("alpha_traffic", "alpha_ped"), alphas):
        _require_dtype(alpha, np.dtype(np.float64), name)
        if alpha.shape != (n_zones,):
            raise ValueError(f"{name} must have shape (n_zones,)")
        if not np.isfinite(alpha).all() or (alpha < 1.0).any():
            raise ValueError(f"{name} must be finite and >= 1")

    def validate_tau_bounds(
        name: str, tau: np.ndarray, speed: float, alpha: np.ndarray
    ) -> None:
        lower = arrays["d"] / speed * 60.0
        upper = arrays["d"] * float(alpha.max()) / speed * 60.0
        tolerance = 1e-12
        if (tau < lower - tolerance).any():
            raise ValueError(f"{name} is below its alpha>=1 travel-time bound")
        if (tau > upper + tolerance).any():
            raise ValueError(f"{name} exceeds its maximum-alpha bound")

    validate_tau_bounds(
        "tau_truck",
        arrays["tau_truck"],
        params.truck.speed_kmh,
        arrays["alpha_traffic"],
    )
    validate_tau_bounds(
        "tau_robot",
        arrays["tau_robot"],
        params.robot.speed_kmh,
        arrays["alpha_ped"],
    )
    if ((zone < 0) | (zone >= n_zones)).any():
        raise ValueError("node_zone contains an out-of-range zone")
    if meta.get("n_customers") != customer.size:
        raise ValueError("meta.n_customers disagrees with customer_idx")
    copies = meta.get("num_parking_copies")
    if (
        isinstance(copies, bool)
        or not isinstance(copies, int)
        or copies <= 0
        or parking.size != n_zones * copies
    ):
        raise ValueError(
            "parking count disagrees with meta.num_parking_copies"
        )
    if meta.get("instance_id") != path.stem:
        raise ValueError("meta.instance_id must match the npz filename")


def load_instance(
    npz_path: str | Path, params: Params | None = None
) -> Instance:
    """Load a processed instance and reject stale or malformed data."""
    resolved = Path(npz_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"processed instance not found: {resolved}")
    selected_params = load_params() if params is None else params
    expected = {
        "node_label",
        "node_type",
        "node_xy",
        "node_zone",
        "customer_idx",
        "parking_idx",
        "depot_idx",
        "d",
        "tau_truck",
        "tau_robot",
        "alpha_traffic",
        "alpha_ped",
        "meta",
    }
    with np.load(resolved, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != expected:
            raise ValueError(
                f"{resolved}: invalid npz keys; missing={sorted(expected - keys)!r}, "
                f"unknown={sorted(keys - expected)!r}"
            )
        raw_arrays = {key: np.asarray(archive[key]) for key in expected}

    meta_array = raw_arrays.pop("meta")
    if meta_array.shape != () or meta_array.dtype.kind != "U":
        raise TypeError("meta must be a scalar Unicode JSON string")
    meta = json.loads(
        str(meta_array.item()),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(meta, dict):
        raise TypeError("meta JSON must contain an object")
    _check_preproc_compatibility(meta, selected_params, resolved)
    _validate_instance_arrays(raw_arrays, meta, selected_params, resolved)

    arrays = {key: _readonly_copy(value) for key, value in raw_arrays.items()}
    n_customers = arrays["customer_idx"].size
    demand = _readonly_copy(
        np.full(
            n_customers, selected_params.demand_per_customer, dtype=np.int64
        )
    )
    e = _readonly_copy(np.zeros(n_customers, dtype=np.float64))
    deadlines = _readonly_copy(
        np.full(n_customers, selected_params.deadline_min, dtype=np.float64)
    )
    return Instance(
        source_path=resolved,
        node_label=arrays["node_label"],
        node_type=arrays["node_type"],
        node_xy=arrays["node_xy"],
        node_zone=arrays["node_zone"],
        customer_idx=arrays["customer_idx"],
        parking_idx=arrays["parking_idx"],
        depot_idx=arrays["depot_idx"],
        d=arrays["d"],
        tau_truck=arrays["tau_truck"],
        tau_robot=arrays["tau_robot"],
        alpha_traffic=arrays["alpha_traffic"],
        alpha_ped=arrays["alpha_ped"],
        meta=_freeze_json(meta),
        demand=demand,
        e=e,
        l=deadlines,
    )


def load_problem(
    npz_path: str | Path, params_path: str | Path | None = None
) -> tuple[Params, Instance]:
    """Load the shared runtime parameters and one compatible instance."""
    params = load_params(params_path)
    return params, load_instance(npz_path, params)
