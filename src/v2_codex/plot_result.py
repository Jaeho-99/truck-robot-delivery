"""Plot one exact/ALNS result JSON on the Ulsan Nam-gu map.

The result JSON stores routes but deliberately does not duplicate instance
geometry.  This script infers the matching ``data/processed/test`` NPZ and
``data/raw/test`` JSON from the result path and instance id, then combines
them only for visualization.

Example (run from the repository root)::

    python src/v2_codex/plot_result.py output/exact/n5/test_n5_000.json

Two files are written next to the result: the static map
``test_n5_000_map.svg`` and ``test_n5_000_map.html``, an interactive
OpenStreetMap (Leaflet) view of the same routes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib
import numpy as np
from pyproj import CRS, Transformer

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

matplotlib.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["svg.fonttype"] = "none"  # editable text in the SVG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOUNDARY = REPO_ROOT / "data" / "ulsan_namgu_dong_boundaries.geojson"
METHODS = ("gnn_ppo_alns", "ppo_alns", "exact", "alns")
SIZE_PATTERN = re.compile(r"n(5|10|20|50|100)\Z")

# This must match the legacy KATEC definition used by preprocess.py.
KATEC_CRS = CRS.from_proj4(
    "+proj=tmerc +lat_0=38 +lon_0=128 +k=0.9999 "
    "+x_0=400000 +y_0=600000 +ellps=bessel "
    "+towgs84=-115.8,474.99,674.11,1.16,-2.31,-1.63,6.43 "
    "+units=m +no_defs"
)
WGS84_TO_KATEC = Transformer.from_crs(
    "EPSG:4326", KATEC_CRS, always_xy=True
)
KATEC_TO_WGS84 = Transformer.from_crs(
    KATEC_CRS, "EPSG:4326", always_xy=True
)

TRUCK_COL, ROBOT_COL = "#d62728", "#2ca02c"
TRUCK_COLORS = ["#d62728", "#1f77b4", "#9467bd", "#8c564b"]  # HTML, per truck
ROBOT_COLORS = ["#2ca02c", "#ff7f0e", "#17becf", "#bcbd22"]  # HTML, per robot
DEPOT_FC, DEPOT_EC = "#d62728", "#7f1d1d"
PARK_FC, PARK_EC = "#d9d9d9", "#333333"
CUST_FC, CUST_EC = "#ffffff", "#222222"
MAP_FACE = "#f2f2f2"
MAP_EDGE = "#5a5a5a"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _size_directory(result_path: Path) -> Path:
    # Preserve rendering for both nSIZE/file.json and nSIZE/runs/LABEL/file.json.
    for parent in result_path.parents:
        if SIZE_PATTERN.fullmatch(parent.name):
            return parent
    raise ValueError(f"result path has no nSIZE directory: {result_path}")


def _method_and_tag(result_path: Path) -> tuple[str, str | None]:
    directory = _size_directory(result_path).parent.name
    for method in METHODS:
        if directory == method:
            return method, None
        prefix = f"{method}_"
        if directory.startswith(prefix):
            return method, directory[len(prefix):]
    expected = ", ".join(METHODS)
    raise ValueError(
        f"result must be under output/{{{expected}}}/nSIZE: {result_path}"
    )


def _size_from_path(result_path: Path) -> int:
    match = SIZE_PATTERN.fullmatch(_size_directory(result_path).name)
    if match is None:
        raise ValueError(
            "result parent directory must be n5, n10, n20, n50, or n100"
        )
    return int(match.group(1))


def _instance_id(result: dict[str, Any]) -> str:
    value = result.get("instance_id", result.get("name"))
    if not isinstance(value, str) or not value:
        raise ValueError("result JSON has no valid 'instance_id' or 'name'")
    return value


def _default_data_paths(
    result_path: Path, result: dict[str, Any]
) -> tuple[Path, Path, str, int]:
    method, tag = _method_and_tag(result_path)
    size = _size_from_path(result_path)
    instance_id = _instance_id(result)
    processed_dir = "processed" if tag is None else f"processed_{tag}"
    processed = (
        REPO_ROOT / "data" / processed_dir / "test" / f"n{size}"
        / f"{instance_id}.npz"
    )
    raw = (
        REPO_ROOT / "data" / "raw" / "test" / f"n{size}"
        / f"{instance_id}.json"
    )
    return processed, raw, method, size


def _load_nodes(processed_path: Path) -> list[dict[str, Any]]:
    if not processed_path.is_file():
        raise FileNotFoundError(f"processed instance not found: {processed_path}")
    with np.load(processed_path, allow_pickle=False) as archive:
        required = {"node_label", "node_type", "node_xy"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{processed_path}: missing arrays {sorted(missing)}"
            )
        labels = archive["node_label"].tolist()
        types = archive["node_type"].tolist()
        xy = np.asarray(archive["node_xy"], dtype=float)
    if len(labels) != len(types) or xy.shape != (len(labels), 2):
        raise ValueError(f"inconsistent node arrays: {processed_path}")
    return [
        {"label": str(label), "type": str(node_type),
         "x": float(point[0]), "y": float(point[1])}
        for label, node_type, point in zip(labels, types, xy, strict=True)
    ]


def _iter_exterior_rings(geometry: dict[str, Any]) -> Iterable[list[list[float]]]:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geom_type == "Polygon":
        if coordinates:
            yield coordinates[0]
    elif geom_type == "MultiPolygon":
        for polygon in coordinates or []:
            if polygon:
                yield polygon[0]
    else:
        raise ValueError(f"unsupported boundary geometry: {geom_type!r}")


def _ring_area_centroid(ring: np.ndarray) -> tuple[float, float, float]:
    """Return (area, cx, cy) of one closed ring, for dong-label placement."""
    xs, ys = ring[:, 0], ring[:, 1]
    cross = xs[:-1] * ys[1:] - xs[1:] * ys[:-1]
    area = 0.5 * float(cross.sum())
    if abs(area) < 1e-9:
        return 0.0, float(xs.mean()), float(ys.mean())
    cx = float(((xs[:-1] + xs[1:]) * cross).sum()) / (6.0 * area)
    cy = float(((ys[:-1] + ys[1:]) * cross).sum()) / (6.0 * area)
    return abs(area), cx, cy


def _origin_katec(raw: dict[str, Any], raw_path: Path) -> np.ndarray:
    origin = raw.get("origin_katec")
    if (not isinstance(origin, list) or len(origin) != 2
            or not all(isinstance(value, (int, float)) for value in origin)):
        raise ValueError(f"invalid origin_katec in {raw_path}")
    return np.asarray(origin, dtype=float)


def _load_boundaries(
    boundary_path: Path, raw_path: Path
) -> list[tuple[list[np.ndarray], str, tuple[float, float]]]:
    if not boundary_path.is_file():
        raise FileNotFoundError(f"boundary file not found: {boundary_path}")
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"raw instance needed for origin_katec was not found: {raw_path}"
        )
    origin_xy = _origin_katec(_load_json(raw_path), raw_path)

    geojson = _load_json(boundary_path)
    features = geojson.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"boundary GeoJSON has no features: {boundary_path}")

    boundaries: list[tuple[list[np.ndarray], str, tuple[float, float]]] = []
    for feature in features:
        if not isinstance(feature, dict) or not isinstance(
                feature.get("geometry"), dict):
            raise ValueError(f"invalid feature in {boundary_path}")
        properties = feature.get("properties") or {}
        name = str(properties.get("dong_en") or properties.get("adm_nm") or "")
        rings: list[np.ndarray] = []
        best = (-1.0, 0.0, 0.0)          # largest part carries the dong name
        for ring in _iter_exterior_rings(feature["geometry"]):
            lonlat = np.asarray(ring, dtype=float)
            if lonlat.ndim != 2 or lonlat.shape[1] < 2:
                raise ValueError(f"invalid polygon ring in {boundary_path}")
            east, north = WGS84_TO_KATEC.transform(
                lonlat[:, 0], lonlat[:, 1]
            )
            local_xy = (
                np.column_stack((east, north)) - origin_xy[None, :]
            ) / 1000.0
            rings.append(local_xy)
            area, cx, cy = _ring_area_centroid(local_xy)
            if area > best[0]:
                best = (area, cx, cy)
        if rings:
            boundaries.append((rings, name, (best[1], best[2])))
    return boundaries


def _place_labels(
    ax: Any, labels: list[tuple[float, float, str]],
    obstacles: list[tuple[float, float]], min_dx: float = 1.3,
    min_dy: float = 0.42, ob_dx: float = 0.62, ob_dy: float = 0.28,
    iters: int = 1000,
) -> None:
    """Place dong names, nudged vertically off each other and off nodes."""
    if not labels:
        return
    pos = np.array([[x, y] for x, y, _ in labels], float)
    obs = np.array(obstacles, float) if len(obstacles) else np.empty((0, 2))
    for _ in range(iters):
        moved = False
        for i in range(len(pos)):                       # label vs. label
            for j in range(i + 1, len(pos)):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                if abs(dx) < min_dx and abs(dy) < min_dy:
                    need = (min_dy - abs(dy)) / 2 + 0.01
                    s = 1.0 if dy >= 0 else -1.0
                    pos[i, 1] += s * need
                    pos[j, 1] -= s * need
                    moved = True
        for i in range(len(pos)):                       # label vs. node icon
            for ox, oy in obs:
                dx = pos[i, 0] - ox
                dy = pos[i, 1] - oy
                if abs(dx) < ob_dx and abs(dy) < ob_dy:
                    s = 1.0 if dy >= 0 else -1.0
                    pos[i, 1] += s * (ob_dy - abs(dy) + 0.01) * 0.6
                    moved = True
        if not moved:
            break
    for (x0, y0, name), (lx, ly) in zip(labels, pos):
        text = ax.text(lx, ly, name, fontsize=7, color="#1a1a1a",
                       ha="center", va="center", zorder=3, clip_on=True)
        text.set_path_effects(
            [pe.withStroke(linewidth=2.4, foreground="white")])


def _internal_labels(nodes: list[dict[str, Any]]) -> list[str]:
    depot = [node["label"] for node in nodes if node["type"] == "depot"]
    customers = [node["label"] for node in nodes
                 if node["type"] == "customer"]
    parkings = [node["label"] for node in nodes
                if node["type"] == "parking"]
    if len(depot) != 1:
        raise ValueError("processed instance must contain exactly one depot")
    return [depot[0], *customers, *parkings, depot[0]]


def _node_label(value: Any, labels: list[str]) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"route node must be an integer, got {value!r}")
    if value < 0 or value >= len(labels):
        raise ValueError(f"route node index out of range: {value}")
    return labels[value]


def _heuristic_arcs(
    result: dict[str, Any], nodes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routes = result.get("routes")
    if not isinstance(routes, dict):
        raise ValueError("ALNS result JSON must contain a 'routes' object")
    labels = _internal_labels(nodes)
    depot = labels[0]
    truck_arcs: list[dict[str, Any]] = []
    robot_arcs: list[dict[str, Any]] = []

    def route_order(item: tuple[str, Any]) -> int:
        try:
            return int(item[0])
        except ValueError as exc:
            raise ValueError(f"invalid truck id in routes: {item[0]!r}") from exc

    for truck_key, route in sorted(routes.items(), key=route_order):
        truck = int(truck_key)
        if not isinstance(route, list):
            raise TypeError(f"route for truck {truck} must be a list")
        previous = depot
        for stop in route:
            if not isinstance(stop, dict):
                raise TypeError(f"route stop for truck {truck} must be an object")
            kind = stop.get("kind")
            if kind == "cust":
                node = _node_label(stop.get("c"), labels)
            elif kind == "park":
                node = _node_label(stop.get("p"), labels)
            else:
                raise ValueError(f"unknown route stop kind: {kind!r}")
            truck_arcs.append({"k": truck, "i": previous, "j": node})
            previous = node

            if kind == "park":
                deploys = stop.get("deploys", [])
                if not isinstance(deploys, list):
                    raise TypeError("park.deploys must be a list")
                for trip in deploys:
                    if not isinstance(trip, dict):
                        raise TypeError("robot trip must be an object")
                    robot = trip.get("r")
                    if isinstance(robot, bool) or not isinstance(robot, int):
                        raise TypeError(f"invalid robot id: {robot!r}")
                    trip_previous = node
                    customers = trip.get("custs", [])
                    if not isinstance(customers, list):
                        raise TypeError("robot trip custs must be a list")
                    for customer in customers:
                        customer_label = _node_label(customer, labels)
                        robot_arcs.append({
                            "k": truck, "r": robot,
                            "i": trip_previous, "j": customer_label,
                        })
                        trip_previous = customer_label
                    return_parking = _node_label(trip.get("ret_p"), labels)
                    robot_arcs.append({
                        "k": truck, "r": robot,
                        "i": trip_previous, "j": return_parking,
                    })
        if route:
            truck_arcs.append({"k": truck, "i": previous, "j": depot})
    return truck_arcs, robot_arcs


def _exact_arcs(
    result: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    truck_arcs = result.get("truck_arcs")
    robot_arcs = result.get("robot_arcs")
    if not isinstance(truck_arcs, list) or not isinstance(robot_arcs, list):
        raise ValueError(
            "exact result JSON must contain truck_arcs and robot_arcs lists"
        )
    return truck_arcs, robot_arcs


def _draw_arrow(
    ax: Any, start: dict[str, Any], end: dict[str, Any], *,
    color: str, linestyle: str, linewidth: float,
) -> None:
    x1, y1 = start["x"], start["y"]
    x2, y2 = end["x"], end["y"]
    if np.hypot(x2 - x1, y2 - y1) < 1e-9:
        return
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=linewidth,
                        linestyle=linestyle, alpha=0.85,
                        shrinkA=3, shrinkB=3),
        zorder=3,
    )


class _HandlerCircle(HandlerPatch):
    """Render a Circle patch (dashed edge included) as a circle in the legend."""

    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        circle = Circle((w * 0.5 - xd, h * 0.5 - yd), h * 0.45,
                        facecolor=orig.get_facecolor(),
                        edgecolor=orig.get_edgecolor(),
                        linestyle=orig.get_linestyle(),
                        linewidth=orig.get_linewidth())
        circle.set_transform(trans)
        return [circle]


def _legend(ax: Any) -> Any:
    handles = [
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor=DEPOT_FC, markeredgecolor=DEPOT_EC,
               markersize=18, linestyle="None", label="Main depot"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=PARK_FC, markeredgecolor=PARK_EC,
               markersize=12, linestyle="None", label="Parking node"),
        Circle((0, 0), 1, facecolor=CUST_FC, edgecolor=CUST_EC,
               linewidth=1.3, linestyle="-",
               label="Customer (truck-served)"),
        Circle((0, 0), 1, facecolor=CUST_FC, edgecolor=CUST_EC,
               linewidth=1.3, linestyle="--",
               label="Customer (robot-served)"),
        Line2D([0], [0], color=TRUCK_COL, lw=2.2, linestyle="-",
               label="Truck route"),
        Line2D([0], [0], color=ROBOT_COL, lw=1.6, linestyle="--",
               label="Robot route"),
    ]
    return ax.legend(handles=handles, loc="center left",
                     bbox_to_anchor=(1.0, 0.5), fontsize=11,
                     framealpha=1.0, borderpad=1.0, labelspacing=1.3,
                     handlelength=2.2, handletextpad=0.8,
                     handler_map={Circle: _HandlerCircle()})


def _fleet_box(
    fig: Any, ax: Any, main_legend: Any, n_trucks: int, n_robots: int
) -> Any:
    """Fleet-count box placed right below the main legend box."""
    ax.add_artist(main_legend)               # keep both legends
    fig.canvas.draw()
    box = main_legend.get_window_extent().transformed(ax.transAxes.inverted())
    dummy = (Line2D([0], [0], color="none", linestyle="None"),
             Line2D([0], [0], color="none", linestyle="None"))
    return ax.legend(dummy,
                     [f"Trucks used: {n_trucks}",
                      f"Robots used: {n_robots}"],
                     loc="upper left", bbox_to_anchor=(1.0, box.y0 - 0.03),
                     fontsize=12, framealpha=1.0, borderpad=1.0,
                     labelspacing=0.7, handlelength=0, handletextpad=0)


def _unique_physical_parkings(
    nodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    physical: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if node["type"] == "parking":
            physical.setdefault(node["label"].rsplit("#", 1)[0], node)
    return list(physical.values())


# Split out so the source line stays inside the file's width.
_DECORATOR_JS = ("https://unpkg.com/leaflet-polylinedecorator@1.6.0"
                 "/dist/leaflet.polylineDecorator.js")

_HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>__TITLE__</title>
<link rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="__DECORATOR_JS__"></script>
<style>
html, body, #map { height: 100%; margin: 0 }
.legend { background: #fff; padding: 8px 12px; border-radius: 6px;
  box-shadow: 0 1px 5px rgba(0, 0, 0, .4); font: 13px sans-serif;
  line-height: 1.6 }
.swatch { display: inline-block; width: 22px; height: 0;
  border-top-width: 3px; border-top-style: solid; margin-right: 6px;
  vertical-align: middle }
</style></head>
<body><div id="map"></div><script>
const D = __DATA__, TCOL = __TRUCK_COLORS__, RCOL = __ROBOT_COLORS__;
const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}
).addTo(map);

// grey mask outside Nam-gu: world rectangle with the dong rings punched out
const world = [[-90, -360], [-90, 360], [90, 360], [90, -360]];
L.polygon([world, ...D.boundary], {color: '#555', weight: 0,
  fillColor: '#000', fillOpacity: 0.35, fillRule: 'evenodd',
  interactive: false}).addTo(map);
D.boundary.forEach(ring => L.polygon(ring, {color: '#5a5a5a', weight: 1.2,
  fill: false, interactive: false}).addTo(map));

const gT = L.featureGroup(), gR = L.featureGroup(), gN = L.featureGroup();
function arrow(line, color, size, group) {          // route direction marker
  L.polylineDecorator(line, {patterns: [{offset: '72%', repeat: 0,
    symbol: L.Symbol.arrowHead({pixelSize: size, polygon: true,
      pathOptions: {color: color, fillOpacity: 1, weight: 0}})}]}
  ).addTo(group);
}
D.truck_seg.forEach(s => {
  const col = TCOL[(s.k - 1) % TCOL.length];
  const line = L.polyline([s.a, s.b], {color: col, weight: 4, opacity: 0.9})
    .bindTooltip('Truck ' + s.k).addTo(gT);
  arrow(line, col, 13, gT);
});
D.robot_seg.forEach(s => {
  const col = RCOL[((s.k - 1) * 2 + (s.r - 1)) % RCOL.length];
  const line = L.polyline([s.a, s.b], {color: col, weight: 2.5,
    opacity: 0.9, dashArray: '5,4'})
    .bindTooltip('Truck ' + s.k + ' - Robot ' + s.r).addTo(gR);
  arrow(line, col, 10, gR);
});

D.customers.forEach(c => L.circleMarker(c.at, {radius: 6, color: '#08306b',
  weight: 1, fillColor: (c.served_by === 'robot' ? '#2ca02c' : '#1f77b4'),
  fillOpacity: 0.9})
  .bindTooltip('<b>' + c.label + '</b> (' + c.served_by + '-served)'
    + (c.admi_nm ? '<br>' + c.admi_nm : '')).addTo(gN));
D.parkings.forEach(p => L.marker(p.at, {icon: L.divIcon({className: '',
  html: '<div style="width:13px;height:13px;background:#7f7f7f;'
    + 'border:2px solid #333"></div>',
  iconSize: [13, 13], iconAnchor: [7, 7]})})
  .bindTooltip('<b>' + p.label + '</b> parking'
    + (p.admi_nm ? '<br>' + p.admi_nm : '')).addTo(gN));
L.marker(D.depot.at, {icon: L.divIcon({className: '',
  html: '<div style="width:20px;height:20px;background:#d62728;'
    + 'border:3px solid #7f1d1d;border-radius:50%"></div>',
  iconSize: [20, 20], iconAnchor: [10, 10]})})
  .bindTooltip('<b>' + D.depot.label + '</b> depot').addTo(gN);

gT.addTo(map); gR.addTo(map); gN.addTo(map);
map.fitBounds(L.latLngBounds(D.boundary.flat()).pad(0.05));
L.control.layers(null, {'Truck routes': gT, 'Robot routes': gR,
  'Nodes': gN}, {collapsed: false}).addTo(map);

const legend = L.control({position: 'bottomleft'});
legend.onAdd = () => {
  const div = L.DomUtil.create('div', 'legend');
  div.innerHTML =
    '<span class="swatch" style="border-top-color:#d62728"></span>'
    + 'Truck route<br>'
    + '<span class="swatch" style="border-top-color:#2ca02c;'
    + 'border-top-style:dashed"></span>Robot route<br>'
    + '<span style="color:#1f77b4">&#9679;</span> Customer (truck-served)'
    + '&nbsp;&nbsp;<span style="color:#2ca02c">&#9679;</span>'
    + ' Customer (robot-served)'
    + '<br><b>Trucks used: ' + D.n_trucks
    + ' &nbsp;|&nbsp; Robots used: ' + D.n_robots + '</b>';
  return div;
};
legend.addTo(map);
</script></body></html>
"""


def _boundary_latlng(boundary_path: Path) -> list[list[list[float]]]:
    """Dong exterior rings as Leaflet [[lat, lon], ...] rings (WGS84)."""
    geojson = _load_json(boundary_path)
    rings: list[list[list[float]]] = []
    for feature in geojson.get("features") or []:
        for ring in _iter_exterior_rings(feature["geometry"]):
            rings.append([[round(lat, 6), round(lon, 6)]
                          for lon, lat, *_ in ring])
    return rings


def _zone_names(raw: dict[str, Any]) -> dict[str, str]:
    """Map node label -> dong name, for the HTML tooltips."""
    zones = raw.get("zones")
    raw_nodes = raw.get("nodes")
    if not isinstance(zones, dict) or not isinstance(raw_nodes, list):
        return {}
    names: dict[str, str] = {}
    for node in raw_nodes:
        zone = zones.get(str(node.get("zone")))
        if isinstance(zone, dict) and zone.get("admi_nm"):
            names[str(node.get("label"))] = str(zone["admi_nm"])
    return names


def _write_html(
    output_path: Path, *, raw_path: Path, boundary_path: Path,
    nodes: list[dict[str, Any]], truck_arcs: list[dict[str, Any]],
    robot_arcs: list[dict[str, Any]], robot_customers: set[str],
) -> Path:
    """Write the interactive OpenStreetMap (Leaflet) view of the routes."""
    raw = _load_json(raw_path)
    origin_xy = _origin_katec(raw, raw_path)
    admi_nm = _zone_names(raw)

    local_xy = np.array([[node["x"], node["y"]] for node in nodes], float)
    east, north = (local_xy * 1000.0 + origin_xy[None, :]).T
    lon, lat = KATEC_TO_WGS84.transform(east, north)
    at = {node["label"]: [round(float(a), 6), round(float(o), 6)]
          for node, a, o in zip(nodes, lat, lon, strict=True)}

    def marker(node: dict[str, Any], label: str) -> dict[str, Any]:
        return {"label": label, "at": at[node["label"]],
                "admi_nm": admi_nm.get(node["label"], "")}

    def segments(
        arcs: list[dict[str, Any]], with_robot: bool
    ) -> list[dict[str, Any]]:
        out = []
        for arc in arcs:
            record = {"a": at[arc["i"]], "b": at[arc["j"]],
                      "k": int(arc["k"])}
            if with_robot:
                record["r"] = int(arc["r"])
            out.append(record)
        return out

    depot = next(node for node in nodes if node["type"] == "depot")
    data = {
        "boundary": _boundary_latlng(boundary_path),
        "depot": marker(depot, depot["label"]),
        "customers": [
            {**marker(node, node["label"]),
             "served_by": ("robot" if node["label"] in robot_customers
                           else "truck")}
            for node in nodes if node["type"] == "customer"
        ],
        "parkings": [marker(node, node["label"].rsplit("#", 1)[0])
                     for node in _unique_physical_parkings(nodes)],
        "truck_seg": segments(truck_arcs, False),
        "robot_seg": segments(robot_arcs, True),
        "n_trucks": len({int(arc["k"]) for arc in truck_arcs}),
        "n_robots": len({(int(arc["k"]), int(arc["r"]))
                         for arc in robot_arcs}),
    }
    html = (
        _HTML_TEMPLATE
        .replace("__DECORATOR_JS__", _DECORATOR_JS)
        .replace("__TITLE__", output_path.stem)
        .replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__TRUCK_COLORS__", json.dumps(TRUCK_COLORS))
        .replace("__ROBOT_COLORS__", json.dumps(ROBOT_COLORS))
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def plot_result(
    result_path: Path, *, processed_path: Path | None = None,
    raw_path: Path | None = None, boundary_path: Path = DEFAULT_BOUNDARY,
    output_path: Path | None = None, image_format: str = "svg",
    dpi: int = 200, show_labels: bool = False,
) -> tuple[Path, Path]:
    """Create the static route map plus its HTML twin, returning both paths."""
    result_path = result_path.expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"result JSON not found: {result_path}")
    result = _load_json(result_path)
    inferred_processed, inferred_raw, method, size = _default_data_paths(
        result_path, result
    )
    processed_path = (processed_path or inferred_processed).expanduser().resolve()
    raw_path = (raw_path or inferred_raw).expanduser().resolve()
    boundary_path = boundary_path.expanduser().resolve()
    nodes = _load_nodes(processed_path)
    boundaries = _load_boundaries(boundary_path, raw_path)
    if method == "exact":
        truck_arcs, robot_arcs = _exact_arcs(result)
    else:
        truck_arcs, robot_arcs = _heuristic_arcs(result, nodes)

    node_by_label = {node["label"]: node for node in nodes}
    for arc in [*truck_arcs, *robot_arcs]:
        if arc.get("i") not in node_by_label or arc.get("j") not in node_by_label:
            raise ValueError(f"route arc references an unknown node: {arc}")

    robot_customers = {
        label for arc in robot_arcs for label in (arc["i"], arc["j"])
        if node_by_label[label]["type"] == "customer"
    }
    customers = [node for node in nodes if node["type"] == "customer"]
    truck_customers = [node for node in customers
                       if node["label"] not in robot_customers]
    robot_customer_nodes = [node for node in customers
                            if node["label"] in robot_customers]
    depots = [node for node in nodes if node["type"] == "depot"]
    if len(depots) != 1:
        raise ValueError("processed instance must contain exactly one depot")

    fig, ax = plt.subplots(figsize=(13, 12))
    obstacles = [(node["x"], node["y"]) for node in nodes]
    all_boundary_xy = []
    dong_labels: list[tuple[float, float, str]] = []
    for rings, name, (cx, cy) in boundaries:
        for ring in rings:
            ax.add_patch(MplPolygon(
                ring, closed=True, facecolor=MAP_FACE, edgecolor=MAP_EDGE,
                lw=0.9, zorder=0,
            ))
            all_boundary_xy.append(ring)
        if name:
            dong_labels.append((cx, cy, name))
    _place_labels(ax, dong_labels, obstacles)

    parkings = _unique_physical_parkings(nodes)
    if parkings:
        ax.scatter(
            [node["x"] for node in parkings],
            [node["y"] for node in parkings],
            marker="s", s=95, facecolors=PARK_FC, edgecolors=PARK_EC,
            linewidths=1.1, zorder=4,
        )
    if truck_customers:
        ax.scatter(
            [node["x"] for node in truck_customers],
            [node["y"] for node in truck_customers],
            marker="o", s=105, facecolors=CUST_FC, edgecolors=CUST_EC,
            linewidths=1.5, linestyle="-", zorder=5,
        )
    if robot_customer_nodes:
        ax.scatter(
            [node["x"] for node in robot_customer_nodes],
            [node["y"] for node in robot_customer_nodes],
            marker="o", s=105, facecolors=CUST_FC, edgecolors=CUST_EC,
            linewidths=1.5, linestyle="--", zorder=5,
        )
    depot = depots[0]
    ax.scatter(
        [depot["x"]], [depot["y"]], marker="*", s=460,
        facecolors=DEPOT_FC, edgecolors=DEPOT_EC, linewidths=1.3, zorder=7,
    )

    for arc in truck_arcs:
        _draw_arrow(
            ax, node_by_label[arc["i"]], node_by_label[arc["j"]],
            color=TRUCK_COL, linestyle="-", linewidth=2.0,
        )
    for arc in robot_arcs:
        _draw_arrow(
            ax, node_by_label[arc["i"]], node_by_label[arc["j"]],
            color=ROBOT_COL, linestyle="--", linewidth=1.3,
        )

    if show_labels:
        for node in [*customers, *parkings, depot]:
            label = (node["label"].rsplit("#", 1)[0]
                     if node["type"] == "parking" else node["label"])
            ax.annotate(
                label, (node["x"], node["y"]), xytext=(4, 4),
                textcoords="offset points", fontsize=6, zorder=7,
            )

    legend = _legend(ax)
    n_trucks = len({int(arc["k"]) for arc in truck_arcs})
    n_robots = len({(int(arc["k"]), int(arc["r"])) for arc in robot_arcs})
    fleet_legend = _fleet_box(fig, ax, legend, n_trucks, n_robots)

    if all_boundary_xy:
        combined = np.vstack(all_boundary_xy)
        margin = 0.3
        ax.set_xlim(float(combined[:, 0].min()) - margin,
                    float(combined[:, 0].max()) + margin)
        ax.set_ylim(float(combined[:, 1].min()) - margin,
                    float(combined[:, 1].max()) + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    if output_path is None:
        output_path = result_path.with_name(
            f"{result_path.stem}_map.{image_format}"
        )
    else:
        output_path = output_path.expanduser().resolve()
        if output_path.suffix.lower() != f".{image_format}":
            raise ValueError(
                f"--output suffix must be .{image_format}: {output_path}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=image_format, dpi=dpi,
                bbox_inches="tight",
                bbox_extra_artists=[legend, fleet_legend])
    plt.close(fig)
    html_path = _write_html(
        output_path.with_suffix(".html"), raw_path=raw_path,
        boundary_path=boundary_path, nodes=nodes, truck_arcs=truck_arcs,
        robot_arcs=robot_arcs, robot_customers=robot_customers,
    )
    return output_path, html_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one result JSON from exact, ALNS, PPO-ALNS, or "
                    "GNN-PPO-ALNS"
    )
    parser.add_argument(
        "result", type=Path,
        help="output/METHOD/nSIZE/INSTANCE.json",
    )
    parser.add_argument(
        "--processed", type=Path,
        help="matching processed NPZ (normally inferred automatically)",
    )
    parser.add_argument(
        "--raw", type=Path,
        help="matching raw JSON containing origin_katec (normally inferred)",
    )
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--format", dest="image_format", choices=("png", "svg", "pdf"),
        default="svg",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--labels", action="store_true",
        help="show node labels (dong names are always drawn)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    image_path, html_path = plot_result(
        args.result, processed_path=args.processed, raw_path=args.raw,
        boundary_path=args.boundary, output_path=args.output,
        image_format=args.image_format, dpi=args.dpi,
        show_labels=args.labels,
    )
    print(image_path)
    print(html_path)


if __name__ == "__main__":
    main()
