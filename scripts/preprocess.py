"""Precompute distance and travel-time matrices from immutable raw JSON.

Zone assignment uses the 14 WGS84 administrative-dong polygons in
``data/ulsan_namgu_dong_boundaries.geojson``.  Polygon vertices are converted
to the legacy KATEC TM128 CRS used to create the raw instances, and every
sample-piece midpoint is assigned by point-in-polygon.  A midpoint in a map
gap or outside the polygon union is assigned from the nearest 50 m KT cell in
``data/cells_ulsan_namgu.csv``; shared-boundary and nearest-cell ties use the
lower raw zone ID.  If the dong polygon file is absent, nearest-cell-zone is
used for every midpoint.  This complete cell table is stable across instances,
unlike using only the randomly sampled nodes of the current instance.

The selected Manhattan convention is applied to ``n_samples`` equal pieces
per nonzero L-leg.  With ``average``, x-first and y-first zone decompositions
are averaged elementwise, giving an exactly symmetric stored decomposition.
Models consume the resulting matrices and never recompute geometry.

Example from the repository root::

    python scripts/preprocess.py --split all --size 5

Processing follows validation -> zone decomposition -> matrix building
-> validation -> atomic NPZ publication. Existing compatible data is
reused; ``--force`` explicitly requests regeneration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

if __package__ in (None, ""):
    # Keep direct execution usable without an editable package installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.params import (
    CELL_COORD_PATH,
    CELL_ZONE_METHOD,
    DEFAULT_PARAMS_PATH,
    DONG_POLYGON_PATH,
    POLYGON_ZONE_METHOD,
    Params,
    load_instance,
    load_params,
    preproc_signature,
    zone_source_info,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
SIZES = (5, 10, 20, 50, 100)
SPLITS = ("train", "test")
TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# Exact projection string from the KT source-table definition used by the
# script that generated data/raw.  EPSG:5178 is a different coordinate frame.
KATEC_CRS = CRS.from_proj4(
    "+proj=tmerc +lat_0=38 +lon_0=128 +k=0.9999 "
    "+x_0=400000 +y_0=600000 +ellps=bessel "
    "+towgs84=-115.8,474.99,674.11,1.16,-2.31,-1.63,6.43 "
    "+units=m +no_defs"
)

RAW_KEYS = {
    "instance_id",
    "region",
    "seed",
    "n_customers",
    "num_parking_copies",
    "n_zones",
    "zones",
    "origin_katec",
    "nodes",
    "alpha_traffic",
    "alpha_ped",
}
NODE_KEYS = {"label", "type", "x", "y", "zone"}
ZONE_KEYS = {"admi_cd", "admi_nm"}
NODE_TYPES = frozenset({"customer", "parking", "depot"})


class _ZoneLocator(Protocol):
    method: str

    def locate_many(self, points: np.ndarray) -> np.ndarray:
        """Return one integer zone for each local-km point."""


@dataclass(frozen=True, slots=True)
class _PolygonPart:
    zone: int
    rings: tuple[np.ndarray, ...]
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _CellTable:
    katec_xy: np.ndarray
    admi_cd: np.ndarray


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant {value!r}")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )


def _mapping(value: Any, path: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    return value


def _exact_keys(
    value: Any, expected: set[Any], path: str
) -> Mapping[Any, Any]:
    mapping = _mapping(value, path)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{path} has invalid keys: missing={sorted(expected - actual)!r}, "
            f"unknown={sorted(actual - expected)!r}"
        )
    return mapping


def _integer(value: Any, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{path} must be > 0, got {value}")
    return value


def _real(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite, got {value!r}")
    return result


def _validate_payload(
    path: Path, payload: Any, expected_size: int
) -> dict[str, Any]:
    raw = _exact_keys(payload, RAW_KEYS, str(path))
    instance_id = raw["instance_id"]
    if not isinstance(instance_id, str) or not instance_id:
        raise TypeError(f"{path}: instance_id must be a non-empty string")
    if instance_id != path.stem:
        raise ValueError(
            f"{path}: instance_id {instance_id!r} does not match filename"
        )
    if not isinstance(raw["region"], str) or not raw["region"]:
        raise TypeError(f"{path}: region must be a non-empty string")
    seed = raw["seed"]
    if (
        not isinstance(seed, list)
        or len(seed) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in seed
        )
    ):
        raise TypeError(f"{path}: seed must contain exactly three integers")

    n_customers = _integer(
        raw["n_customers"], f"{path}: n_customers", positive=True
    )
    if n_customers != expected_size:
        raise ValueError(
            f"{path}: expected n{expected_size}, found {n_customers} customers"
        )
    n_zones = _integer(raw["n_zones"], f"{path}: n_zones", positive=True)
    copies = _integer(
        raw["num_parking_copies"], f"{path}: num_parking_copies", positive=True
    )

    zones_raw = _mapping(raw["zones"], f"{path}: zones")
    try:
        zone_ids = {int(key) for key in zones_raw}
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path}: zone keys must be integer strings") from exc
    expected_zones = set(range(n_zones))
    if zone_ids != expected_zones or {str(zone) for zone in zone_ids} != set(
        zones_raw
    ):
        raise ValueError(
            f"{path}: zones must be contiguous string keys 0..{n_zones - 1}"
        )
    zones: dict[int, dict[str, str]] = {}
    for zone in range(n_zones):
        info = _exact_keys(
            zones_raw[str(zone)], ZONE_KEYS, f"{path}: zones[{zone}]"
        )
        if (
            not isinstance(info["admi_cd"], str)
            or not isinstance(info["admi_nm"], str)
            or not info["admi_cd"]
            or not info["admi_nm"]
        ):
            raise TypeError(f"{path}: invalid metadata for zone {zone}")
        zones[zone] = {"admi_cd": info["admi_cd"], "admi_nm": info["admi_nm"]}
    names = [zones[zone]["admi_nm"] for zone in range(n_zones)]
    if len(set(names)) != n_zones:
        raise ValueError(f"{path}: zone names must be unique")

    origin = raw["origin_katec"]
    if not isinstance(origin, list) or len(origin) != 2:
        raise TypeError(f"{path}: origin_katec must contain [x, y]")
    origin_xy = np.asarray(
        [
            _real(origin[0], f"{path}: origin_katec[0]"),
            _real(origin[1], f"{path}: origin_katec[1]"),
        ],
        dtype=np.float64,
    )

    nodes_raw = raw["nodes"]
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise TypeError(f"{path}: nodes must be a non-empty array")
    labels: list[str] = []
    node_types: list[str] = []
    node_xy: list[tuple[float, float]] = []
    node_zone: list[int] = []
    for index, value in enumerate(nodes_raw):
        node = _exact_keys(value, NODE_KEYS, f"{path}: nodes[{index}]")
        label = node["label"]
        kind = node["type"]
        if not isinstance(label, str) or not label:
            raise TypeError(f"{path}: nodes[{index}].label must be non-empty")
        if kind not in NODE_TYPES:
            raise ValueError(
                f"{path}: nodes[{index}].type is invalid: {kind!r}"
            )
        zone = _integer(node["zone"], f"{path}: nodes[{index}].zone")
        if zone not in expected_zones:
            raise ValueError(f"{path}: nodes[{index}].zone is out of range")
        labels.append(label)
        node_types.append(kind)
        node_xy.append(
            (
                _real(node["x"], f"{path}: nodes[{index}].x"),
                _real(node["y"], f"{path}: nodes[{index}].y"),
            )
        )
        node_zone.append(zone)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path}: node labels must be unique")

    node_label_array = np.asarray(labels, dtype=np.str_)
    node_type_array = np.asarray(node_types, dtype=np.str_)
    node_xy_array = np.asarray(node_xy, dtype=np.float64)
    node_zone_array = np.asarray(node_zone, dtype=np.int64)
    customer_idx = np.flatnonzero(node_type_array == "customer").astype(
        np.int64
    )
    parking_idx = np.flatnonzero(node_type_array == "parking").astype(np.int64)
    depot_idx = np.flatnonzero(node_type_array == "depot").astype(np.int64)
    if customer_idx.size != n_customers:
        raise ValueError(
            f"{path}: customer node count disagrees with n_customers"
        )
    if parking_idx.size != n_zones * copies:
        raise ValueError(
            f"{path}: expected {n_zones * copies} parking copies, "
            f"found {parking_idx.size}"
        )
    if depot_idx.size != 1 or labels[int(depot_idx[0])] != "D":
        raise ValueError(f"{path}: exactly one depot labelled 'D' is required")
    expected_parking_labels = {
        f"P{zone}#{copy}"
        for zone in range(n_zones)
        for copy in range(1, copies + 1)
    }
    actual_parking_labels = {labels[index] for index in parking_idx}
    if actual_parking_labels != expected_parking_labels:
        raise ValueError(
            f"{path}: parking-copy labels do not match P{{zone}}#{{copy}}"
        )
    parking_pattern = re.compile(r"P(\d+)#(\d+)\Z")
    for index in parking_idx:
        match = parking_pattern.fullmatch(labels[int(index)])
        if match is None:
            raise ValueError(
                f"{path}: invalid parking label {labels[int(index)]!r}"
            )
        label_zone, copy = (int(value) for value in match.groups())
        if (
            label_zone != int(node_zone_array[index])
            or not 1 <= copy <= copies
        ):
            raise ValueError(
                f"{path}: parking label {labels[int(index)]!r} disagrees with "
                f"node.zone={int(node_zone_array[index])}"
            )

    def alpha_array(field: str) -> np.ndarray:
        source = _mapping(raw[field], f"{path}: {field}")
        if set(source) != {str(zone) for zone in range(n_zones)}:
            raise ValueError(f"{path}: {field} keys must match zones")
        result = np.asarray(
            [
                _real(source[str(zone)], f"{path}: {field}[{zone}]")
                for zone in range(n_zones)
            ],
            dtype=np.float64,
        )
        if (result < 1.0).any():
            raise ValueError(f"{path}: every {field} value must be >= 1")
        return result

    return {
        "instance_id": instance_id,
        "n_customers": n_customers,
        "n_zones": n_zones,
        "num_parking_copies": copies,
        "zones": zones,
        "origin_katec": origin_xy,
        "node_label": node_label_array,
        "node_type": node_type_array,
        "node_xy": node_xy_array,
        "node_zone": node_zone_array,
        "customer_idx": customer_idx,
        "parking_idx": parking_idx,
        "depot_idx": depot_idx,
        "alpha_traffic": alpha_array("alpha_traffic"),
        "alpha_ped": alpha_array("alpha_ped"),
    }


def _load_cell_table(path: Path) -> _CellTable:
    """Load the complete KT 50 m cells without modifying the source CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"KATEC cell coordinate file not found: {path}"
        )
    expected_columns = ["cell_id", "x_katec", "y_katec", "admi_cd"]
    cell_ids: set[str] = set()
    coordinates: set[tuple[int, int]] = set()
    xy: list[tuple[float, float]] = []
    codes: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"{path}: expected CSV columns {expected_columns!r}, "
                f"got {reader.fieldnames!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(
                row[column] is None or row[column] == ""
                for column in expected_columns
            ):
                raise ValueError(f"{path}:{line_number}: incomplete cell row")
            cell_id = row["cell_id"]
            if cell_id in cell_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate cell_id {cell_id}"
                )
            cell_ids.add(cell_id)
            try:
                east = int(row["x_katec"])
                north = int(row["y_katec"])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: KATEC coordinates must be integers"
                ) from exc
            coordinate = (east, north)
            if coordinate in coordinates:
                raise ValueError(
                    f"{path}:{line_number}: duplicate KATEC coordinate {coordinate}"
                )
            coordinates.add(coordinate)
            xy.append((float(east), float(north)))
            codes.append(row["admi_cd"])
    if not xy:
        raise ValueError(f"{path}: cell table is empty")
    katec_xy = np.asarray(xy, dtype=np.float64)
    admi_cd = np.asarray(codes, dtype=np.str_)
    katec_xy.setflags(write=False)
    admi_cd.setflags(write=False)
    return _CellTable(katec_xy=katec_xy, admi_cd=admi_cd)


def _ring_membership(
    points: np.ndarray, ring: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized even-odd membership and boundary flags for one ring."""
    px = points[:, 0, None]
    py = points[:, 1, None]
    start = ring[:-1]
    end = ring[1:]
    x1 = start[:, 0][None, :]
    y1 = start[:, 1][None, :]
    x2 = end[:, 0][None, :]
    y2 = end[:, 1][None, :]

    dx = x2 - x1
    dy = y2 - y1
    cross = (px - x1) * dy - (py - y1) * dx
    tolerance = 1e-10 * np.maximum(1.0, np.abs(dx) + np.abs(dy))
    boundary = (
        (np.abs(cross) <= tolerance)
        & (px >= np.minimum(x1, x2) - 1e-10)
        & (px <= np.maximum(x1, x2) + 1e-10)
        & (py >= np.minimum(y1, y2) - 1e-10)
        & (py <= np.maximum(y1, y2) + 1e-10)
    ).any(axis=1)

    crosses_y = (y1 > py) != (y2 > py)
    safe_dy = np.where(dy == 0.0, 1.0, dy)
    x_intersection = x1 + (py - y1) * dx / safe_dy
    crossings = crosses_y & (px < x_intersection)
    inside = np.logical_xor.reduce(crossings, axis=1)
    return inside, boundary


def _polygon_covers(
    points: np.ndarray, rings: tuple[np.ndarray, ...]
) -> np.ndarray:
    outer_inside, outer_boundary = _ring_membership(points, rings[0])
    covered = outer_inside | outer_boundary
    for hole in rings[1:]:
        hole_inside, hole_boundary = _ring_membership(points, hole)
        covered &= ~(hole_inside & ~hole_boundary)
    return covered


class _CellZoneLocator:
    method = CELL_ZONE_METHOD

    def __init__(self, table: _CellTable, instance: Mapping[str, Any]):
        code_to_zone = {
            info["admi_cd"]: zone for zone, info in instance["zones"].items()
        }
        table_codes = set(table.admi_cd.tolist())
        if table_codes != set(code_to_zone):
            raise ValueError(
                "cell-table administrative codes do not match raw zones: "
                f"missing={sorted(set(code_to_zone) - table_codes)!r}, "
                f"unknown={sorted(table_codes - set(code_to_zone))!r}"
            )
        origin = instance["origin_katec"]
        local_xy = (table.katec_xy - origin[None, :]) / 1000.0
        cell_zone = np.asarray(
            [code_to_zone[code] for code in table.admi_cd], dtype=np.int64
        )
        self.tree = cKDTree(local_xy)
        self.cell_zone = cell_zone

    def locate_many(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("zone lookup points must have shape (n, 2)")
        if points.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        # A small neighbour set captures regular-grid equidistance; choose the
        # lower zone ID rather than relying on cKDTree's internal tie order.
        k = min(16, self.cell_zone.size)
        distance, index = self.tree.query(points, k=k)
        if k == 1:
            return self.cell_zone[np.asarray(index, dtype=np.int64)]
        distance = np.asarray(distance, dtype=np.float64)
        index = np.asarray(index, dtype=np.int64)
        nearest = distance[:, [0]]
        tied = np.isclose(distance, nearest, rtol=0.0, atol=1e-12)
        candidates = self.cell_zone[index]
        sentinel = int(self.cell_zone.max()) + 1
        return np.min(np.where(tied, candidates, sentinel), axis=1).astype(
            np.int64
        )


class _PolygonZoneLocator:
    method = POLYGON_ZONE_METHOD

    def __init__(
        self,
        parts: tuple[_PolygonPart, ...],
        n_zones: int,
        fallback: _CellZoneLocator,
    ):
        self.parts = parts
        self.n_zones = n_zones
        self.fallback = fallback
        grouped: dict[int, list[_PolygonPart]] = {
            zone: [] for zone in range(n_zones)
        }
        for part in parts:
            grouped[part.zone].append(part)
        if any(not grouped[zone] for zone in range(n_zones)):
            raise ValueError("every raw zone must have at least one polygon")
        self.by_zone = {zone: tuple(grouped[zone]) for zone in grouped}

    def locate_many(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("zone lookup points must have shape (n, 2)")
        result = np.full(points.shape[0], -1, dtype=np.int64)
        for zone in range(self.n_zones):
            covered = np.zeros(points.shape[0], dtype=bool)
            for part in self.by_zone[zone]:
                x0, y0, x1, y1 = part.bbox
                candidates = (
                    (points[:, 0] >= x0 - 1e-10)
                    & (points[:, 0] <= x1 + 1e-10)
                    & (points[:, 1] >= y0 - 1e-10)
                    & (points[:, 1] <= y1 + 1e-10)
                )
                if candidates.any():
                    selected = np.flatnonzero(candidates)
                    covered[selected] |= _polygon_covers(
                        points[selected], part.rings
                    )
            # Iterating in ascending ID makes shared-boundary ties deterministic.
            result[(result < 0) & covered] = zone

        unresolved = np.flatnonzero(result < 0)
        if unresolved.size:
            result[unresolved] = self.fallback.locate_many(points[unresolved])
        return result


def _geojson_polygon_groups(geojson: Any) -> dict[str, list[Any]]:
    root = _mapping(geojson, str(DONG_POLYGON_PATH))
    if root.get("type") != "FeatureCollection" or not isinstance(
        root.get("features"), list
    ):
        raise ValueError(f"{DONG_POLYGON_PATH} must be a FeatureCollection")
    grouped: dict[str, list[Any]] = {}
    for index, feature_value in enumerate(root["features"]):
        feature = _mapping(feature_value, f"polygon feature {index}")
        properties = _mapping(
            feature.get("properties"), f"polygon feature {index}.properties"
        )
        name = properties.get("dong_kr")
        if not isinstance(name, str) or not name:
            raise ValueError(f"polygon feature {index} has no dong_kr")
        geometry = _mapping(
            feature.get("geometry"), f"polygon feature {index}.geometry"
        )
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "Polygon":
            polygons = [coordinates]
        elif geometry_type == "MultiPolygon":
            polygons = coordinates
        else:
            raise ValueError(
                f"polygon feature {index} has unsupported {geometry_type!r}"
            )
        if not isinstance(polygons, list) or not polygons:
            raise ValueError(f"polygon feature {index} has no coordinates")
        grouped.setdefault(name, []).extend(polygons)
    return grouped


def _polygon_locator(
    polygon_groups: Mapping[str, list[Any]],
    instance: Mapping[str, Any],
    fallback: _CellZoneLocator,
) -> _PolygonZoneLocator:
    name_to_zone = {
        info["admi_nm"]: zone for zone, info in instance["zones"].items()
    }
    if set(polygon_groups) != set(name_to_zone):
        raise ValueError(
            "dong polygon names do not match raw zones: "
            f"missing={sorted(set(name_to_zone) - set(polygon_groups))!r}, "
            f"unknown={sorted(set(polygon_groups) - set(name_to_zone))!r}"
        )
    transformer = Transformer.from_crs("EPSG:4326", KATEC_CRS, always_xy=True)
    origin = instance["origin_katec"]
    parts: list[_PolygonPart] = []
    for name, polygons in polygon_groups.items():
        zone = name_to_zone[name]
        for polygon_index, polygon in enumerate(polygons):
            if not isinstance(polygon, list) or not polygon:
                raise ValueError(f"empty polygon for {name}[{polygon_index}]")
            transformed_rings: list[np.ndarray] = []
            for ring_index, ring_value in enumerate(polygon):
                ring = np.asarray(ring_value, dtype=np.float64)
                if (
                    ring.ndim != 2
                    or ring.shape[1] < 2
                    or ring.shape[0] < 4
                    or not np.isfinite(ring[:, :2]).all()
                ):
                    raise ValueError(
                        f"invalid ring for {name}[{polygon_index}][{ring_index}]"
                    )
                east, north = transformer.transform(ring[:, 0], ring[:, 1])
                local = np.column_stack(
                    (
                        (np.asarray(east) - origin[0]) / 1000.0,
                        (np.asarray(north) - origin[1]) / 1000.0,
                    )
                ).astype(np.float64)
                if not np.allclose(local[0], local[-1], rtol=0.0, atol=1e-12):
                    local = np.vstack((local, local[0]))
                transformed_rings.append(local)
            all_points = np.concatenate(transformed_rings, axis=0)
            parts.append(
                _PolygonPart(
                    zone=zone,
                    rings=tuple(transformed_rings),
                    bbox=(
                        float(all_points[:, 0].min()),
                        float(all_points[:, 1].min()),
                        float(all_points[:, 0].max()),
                        float(all_points[:, 1].max()),
                    ),
                )
            )
    return _PolygonZoneLocator(
        tuple(parts), instance["n_zones"], fallback=fallback
    )


def _locator_key(
    instance: Mapping[str, Any], use_polygons: bool
) -> tuple[Any, ...]:
    zone_schema = tuple(
        (zone, info["admi_cd"], info["admi_nm"])
        for zone, info in sorted(instance["zones"].items())
    )
    return (use_polygons, *instance["origin_katec"].tolist(), zone_schema)


def _get_locator(
    instance: Mapping[str, Any],
    cell_table: _CellTable,
    polygon_groups: Mapping[str, list[Any]] | None,
    cache: dict[tuple[Any, ...], _ZoneLocator],
) -> _ZoneLocator:
    key = _locator_key(instance, polygon_groups is not None)
    cached = cache.get(key)
    if cached is not None:
        return cached
    cell_locator = _CellZoneLocator(cell_table, instance)
    locator: _ZoneLocator
    if polygon_groups is None:
        locator = cell_locator
    else:
        locator = _polygon_locator(polygon_groups, instance, cell_locator)
    cache[key] = locator
    return locator


class _LegDecomposer:
    def __init__(
        self,
        locator: _ZoneLocator,
        n_zones: int,
        n_samples: int,
        scaling_factor: float,
    ):
        self.locator = locator
        self.n_zones = n_zones
        self.n_samples = n_samples
        self.scaling_factor = scaling_factor
        self.cache: dict[tuple[str, float, float, float], np.ndarray] = {}

    def decompose(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        if dx != 0.0 and dy != 0.0:
            raise ValueError("a Manhattan leg must be axis-aligned")
        if dx == 0.0 and dy == 0.0:
            return np.zeros(self.n_zones, dtype=np.float64)
        if dy == 0.0:
            low, high = sorted((float(start[0]), float(end[0])))
            fixed = float(start[1])
            key = ("h", fixed, low, high)
            axis = 0
        else:
            low, high = sorted((float(start[1]), float(end[1])))
            fixed = float(start[0])
            key = ("v", fixed, low, high)
            axis = 1
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        step = (high - low) / self.n_samples
        midpoint_axis = low + (np.arange(self.n_samples) + 0.5) * step
        points = np.empty((self.n_samples, 2), dtype=np.float64)
        if axis == 0:
            points[:, 0] = midpoint_axis
            points[:, 1] = fixed
        else:
            points[:, 0] = fixed
            points[:, 1] = midpoint_axis
        zones = self.locator.locate_many(points)
        if (
            zones.shape != (self.n_samples,)
            or ((zones < 0) | (zones >= self.n_zones)).any()
        ):
            raise ValueError("zone locator returned invalid midpoint zones")
        result = np.bincount(zones, minlength=self.n_zones).astype(np.float64)
        result *= step * self.scaling_factor
        result.setflags(write=False)
        self.cache[key] = result
        return result


def _path_decomposition(
    start: np.ndarray, end: np.ndarray, convention: str, legs: _LegDecomposer
) -> np.ndarray:
    corner_x = np.asarray([end[0], start[1]], dtype=np.float64)
    corner_y = np.asarray([start[0], end[1]], dtype=np.float64)
    x_first = legs.decompose(start, corner_x) + legs.decompose(corner_x, end)
    if convention == "x_first":
        return x_first
    y_first = legs.decompose(start, corner_y) + legs.decompose(corner_y, end)
    if convention == "y_first":
        return y_first
    if convention == "average":
        return 0.5 * (x_first + y_first)
    raise ValueError(f"unsupported Manhattan convention {convention!r}")


def _build_geometry(
    instance: Mapping[str, Any], params: Params, locator: _ZoneLocator
) -> tuple[np.ndarray, ...]:
    xy = instance["node_xy"]
    n_nodes = xy.shape[0]
    n_zones = instance["n_zones"]
    convention = params.geometry.path_convention
    scaling = params.geometry.scaling_factor
    legs = _LegDecomposer(locator, n_zones, params.geometry.n_samples, scaling)

    d = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    tau_truck = np.zeros_like(d)
    tau_robot = np.zeros_like(d)
    decomposition_sum = np.zeros_like(d)

    def calculate(i: int, j: int) -> tuple[float, float, float, float]:
        zone_km = _path_decomposition(xy[i], xy[j], convention, legs)
        distance = float(
            (abs(xy[i, 0] - xy[j, 0]) + abs(xy[i, 1] - xy[j, 1])) * scaling
        )
        truck_time = float(
            np.dot(zone_km, instance["alpha_traffic"])
            / params.truck.speed_kmh
            * 60.0
        )
        robot_time = float(
            np.dot(zone_km, instance["alpha_ped"])
            / params.robot.speed_kmh
            * 60.0
        )
        return distance, truck_time, robot_time, float(zone_km.sum())

    if convention == "average":
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                values = calculate(i, j)
                for matrix, value in zip(
                    (d, tau_truck, tau_robot, decomposition_sum), values
                ):
                    matrix[i, j] = value
                    matrix[j, i] = value
    else:
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                values = calculate(i, j)
                for matrix, value in zip(
                    (d, tau_truck, tau_robot, decomposition_sum), values
                ):
                    matrix[i, j] = value

    _validate_geometry(
        d,
        tau_truck,
        tau_robot,
        decomposition_sum,
        instance["alpha_traffic"],
        instance["alpha_ped"],
        params,
    )
    return d, tau_truck, tau_robot


def _validate_geometry(
    d: np.ndarray,
    tau_truck: np.ndarray,
    tau_robot: np.ndarray,
    decomposition_sum: np.ndarray,
    alpha_traffic: np.ndarray,
    alpha_ped: np.ndarray,
    params: Params,
) -> None:
    for name, matrix in (
        ("d", d),
        ("tau_truck", tau_truck),
        ("tau_robot", tau_robot),
    ):
        if np.count_nonzero(np.diag(matrix)):
            raise ValueError(f"{name} diagonal is not exactly zero")
        if not np.isfinite(matrix).all() or (matrix < 0.0).any():
            raise ValueError(f"{name} contains an invalid value")
    decomposition_error = float(np.max(np.abs(decomposition_sum - d)))
    if decomposition_error > 1e-9:
        raise ValueError(
            "per-zone decomposition does not sum to Manhattan distance: "
            f"max error={decomposition_error:.3e}"
        )
    if params.geometry.path_convention == "average":
        for name, matrix in (
            ("d", d),
            ("tau_truck", tau_truck),
            ("tau_robot", tau_robot),
        ):
            if not np.array_equal(matrix, matrix.T):
                raise ValueError(f"{name} is not exactly symmetric")

    def validate_bounds(
        name: str, tau: np.ndarray, speed: float, alpha: np.ndarray
    ) -> None:
        lower = d / speed * 60.0
        upper = d * float(alpha.max()) / speed * 60.0
        tolerance = 1e-12
        if (tau < lower - tolerance).any():
            index = tuple(
                int(value) for value in np.argwhere(tau < lower - tolerance)[0]
            )
            raise ValueError(f"{name}{index} is below its alpha>=1 bound")
        if (tau > upper + tolerance).any():
            index = tuple(
                int(value) for value in np.argwhere(tau > upper + tolerance)[0]
            )
            raise ValueError(f"{name}{index} exceeds its maximum-alpha bound")

    validate_bounds(
        "tau_truck", tau_truck, params.truck.speed_kmh, alpha_traffic
    )
    validate_bounds("tau_robot", tau_robot, params.robot.speed_kmh, alpha_ped)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def _metadata(
    instance: Mapping[str, Any],
    params: Params,
    locator: _ZoneLocator,
    git_commit: str,
    zone_source_hash: str,
    node_zone_mismatch_count: int,
) -> dict[str, Any]:
    signature = preproc_signature(params)
    return {
        "geometry": signature["geometry"],
        "instance_id": instance["instance_id"],
        "n_customers": instance["n_customers"],
        "n_zones": instance["n_zones"],
        "node_zone_assignment_mismatch_count": node_zone_mismatch_count,
        "num_parking_copies": instance["num_parking_copies"],
        "preproc_hash": params.preproc_hash,
        "preprocess_git_commit": git_commit,
        "robot": signature["robot"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "truck": signature["truck"],
        "version": signature["version"],
        "zone_assignment_method": locator.method,
        "zone_source_hash": zone_source_hash,
    }


def _atomic_write(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _existing_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta_array = np.asarray(archive["meta"])
        if meta_array.shape != () or meta_array.dtype.kind != "U":
            return None
        meta = json.loads(
            str(meta_array.item()),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        value = meta.get("preproc_hash") if isinstance(meta, dict) else None
        return value if isinstance(value, str) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _process_one(
    raw_path: Path,
    output_path: Path,
    size: int,
    params: Params,
    polygon_groups: Mapping[str, list[Any]] | None,
    cell_table: _CellTable,
    locator_cache: dict[tuple[Any, ...], _ZoneLocator],
    git_commit: str,
    zone_source_hash: str,
    force: bool,
) -> tuple[str, int, int]:
    if not force and _existing_hash(output_path) == params.preproc_hash:
        try:
            load_instance(output_path, params)
        except Exception as exc:
            raise RuntimeError(
                f"{output_path} has the current hash but is invalid; "
                "rerun with --force"
            ) from exc
        return "skipped", 0, 0

    payload = _read_json(raw_path)
    instance = _validate_payload(raw_path, payload, size)
    locator = _get_locator(instance, cell_table, polygon_groups, locator_cache)
    assigned_node_zone = locator.locate_many(instance["node_xy"])
    node_zone_mismatch_count = int(
        np.count_nonzero(assigned_node_zone != instance["node_zone"])
    )
    d, tau_truck, tau_robot = _build_geometry(instance, params, locator)
    meta = _metadata(
        instance,
        params,
        locator,
        git_commit,
        zone_source_hash,
        node_zone_mismatch_count,
    )
    arrays = {
        "node_label": instance["node_label"],
        "node_type": instance["node_type"],
        "node_xy": instance["node_xy"],
        "node_zone": instance["node_zone"],
        "customer_idx": instance["customer_idx"],
        "parking_idx": instance["parking_idx"],
        "depot_idx": instance["depot_idx"],
        "d": d,
        "tau_truck": tau_truck,
        "tau_robot": tau_robot,
        "alpha_traffic": instance["alpha_traffic"],
        "alpha_ped": instance["alpha_ped"],
        "meta": np.asarray(
            json.dumps(
                meta,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            dtype=np.str_,
        ),
    }
    _atomic_write(output_path, arrays)
    return (
        "written",
        node_zone_mismatch_count,
        int(instance["node_xy"].shape[0]),
    )


def _tag(value: str) -> str:
    if not TAG_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "tag must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute immutable truck-robot instance geometry"
    )
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument(
        "--size", choices=("5", "10", "20", "50", "100", "all"), default="all"
    )
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--tag", type=_tag)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    params = load_params(args.params)
    selected_splits = SPLITS if args.split == "all" else (args.split,)
    selected_sizes = SIZES if args.size == "all" else (int(args.size),)
    output_root = (
        REPO_ROOT
        / "data"
        / ("processed" if args.tag is None else f"processed_{args.tag}")
    )

    cell_table = _load_cell_table(CELL_COORD_PATH)
    zone_method, zone_source_hash = zone_source_info()
    polygon_groups = None
    if DONG_POLYGON_PATH.is_file():
        polygon_groups = _geojson_polygon_groups(_read_json(DONG_POLYGON_PATH))
        if zone_method != POLYGON_ZONE_METHOD:
            raise RuntimeError(
                "zone-source method disagrees with polygon availability"
            )
        zone_note = (
            f"point-in-polygon ({DONG_POLYGON_PATH.name}), nearest KT cell "
            "for gaps/outside"
        )
    else:
        if zone_method != CELL_ZONE_METHOD:
            raise RuntimeError(
                "zone-source method disagrees with polygon availability"
            )
        zone_note = "nearest KT-cell zone (dong polygon missing)"
    git_commit = _git_commit()
    print(f"[preprocess] params={Path(args.params).expanduser().resolve()}")
    print(f"[preprocess] hash={params.preproc_hash}")
    print(f"[preprocess] geometry={params.geometry}")
    print(f"[preprocess] zone_assignment={zone_note}")
    print(f"[preprocess] zone_source_hash={zone_source_hash}")

    written = 0
    skipped = 0
    checked_nodes = 0
    mismatched_nodes = 0
    locator_cache: dict[tuple[Any, ...], _ZoneLocator] = {}
    for split in selected_splits:
        for size in selected_sizes:
            source_dir = RAW_ROOT / split / f"n{size}"
            if not source_dir.is_dir():
                raise FileNotFoundError(
                    f"raw instance directory not found: {source_dir}"
                )
            raw_paths = sorted(source_dir.glob("*.json"))
            if not raw_paths:
                raise FileNotFoundError(
                    f"no JSON instances under {source_dir}"
                )
            output_dir = output_root / split / f"n{size}"
            for raw_path in raw_paths:
                output_path = output_dir / f"{raw_path.stem}.npz"
                status, mismatches, node_count = _process_one(
                    raw_path,
                    output_path,
                    size,
                    params,
                    polygon_groups,
                    cell_table,
                    locator_cache,
                    git_commit,
                    zone_source_hash,
                    args.force,
                )
                if status == "skipped":
                    skipped += 1
                else:
                    written += 1
                    checked_nodes += node_count
                    mismatched_nodes += mismatches
                    print(f"[written] {output_path.relative_to(REPO_ROOT)}")
    if checked_nodes:
        print(
            "[zone-check] raw node.zone vs active assignment: "
            f"{mismatched_nodes}/{checked_nodes} differ "
            f"({100.0 * mismatched_nodes / checked_nodes:.3f}%)"
        )
    print(f"[done] written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
