"""SVG visualization of instances and solutions.

  make_instance_svg(inst_json, out_svg)
      zone grid + per-zone T/P congestion text + nodes
  make_route_svg(inst_json, sol_json, out_svg)
      grid + nodes + truck/robot routes (+ labels)

Each zone shows its traffic/pedestrian congestion as "T x.xx / P x.xx"
text (no heatmap or colorbar). Coordinates are km. Output is SVG with
editable text.
"""

import json

import matplotlib

matplotlib.use("Agg")  # set backend before importing pyplot (E402 below)

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"

DEPOT_FC, DEPOT_EC = "#d62728", "#7f1d1d"
PARK_FC, PARK_EC = "#d9d9d9", "#333333"
CUST_FC, CUST_EC = "#ffffff", "#222222"
TRUCK_COL, ROBOT_COL = "#d62728", "#2ca02c"


def _draw_grid(ax, inst):
    """Zone rectangles (no fill) plus congestion text per zone."""
    zk = inst["zone_km"]
    gx = inst.get("grid_x", inst["grid"])
    gy = inst.get("grid_y", inst["grid"])
    at, ap = inst["alpha_traffic"], inst["alpha_ped"]
    for zx in range(gx):
        for zy in range(gy):
            z = zx * gy + zy
            ax.add_patch(Rectangle((zx * zk, zy * zk), zk, zk,
                                   facecolor="none", edgecolor="#888888",
                                   lw=1.0, zorder=0))
            t = ax.text(zx * zk + 0.4, (zy + 1) * zk - 0.4,
                        f"T: {at[str(z)]:.2f}\nP: {ap[str(z)]:.2f}",
                        fontsize=8, color="#1a1a1a", ha="left", va="top",
                        linespacing=1.4, zorder=10)
            t.set_path_effects(
                [pe.withStroke(linewidth=2.0, foreground="white")])
    ax.set_xlim(-1, gx * zk + 1)
    ax.set_ylim(-1, gy * zk + 1)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_nodes(ax, nodes):
    depot = next(n for n in nodes if n["type"] == "depot")
    park = [n for n in nodes if n["type"] == "parking"]
    cust = [n for n in nodes if n["type"] == "customer"]
    if park:
        ax.scatter([p["x"] for p in park], [p["y"] for p in park],
                   marker="s", s=110, facecolors=PARK_FC,
                   edgecolors=PARK_EC, linewidths=1.1, zorder=4)
    # Customers: solid outline = truck-served, dashed = robot-served
    # (same fill). Without served_by (instance map) the outline is
    # solid.
    tc = [c for c in cust if c.get("served_by") != "robot"]
    rc = [c for c in cust if c.get("served_by") == "robot"]
    if tc:
        ax.scatter([c["x"] for c in tc], [c["y"] for c in tc],
                   marker="o", s=130, facecolors=CUST_FC,
                   edgecolors=CUST_EC, linewidths=1.4, linestyle="-",
                   zorder=5)
    if rc:
        ax.scatter([c["x"] for c in rc], [c["y"] for c in rc],
                   marker="o", s=130, facecolors=CUST_FC,
                   edgecolors=CUST_EC, linewidths=1.4, linestyle="--",
                   zorder=5)
    ax.scatter(depot["x"], depot["y"], marker="*", s=420,
               facecolors=DEPOT_FC, edgecolors=DEPOT_EC, linewidths=1.2,
               zorder=7)


def _draw_labels(ax, nodes):
    """Label next to each node (customer Cx / parking Px / depot D).

    Parking copies share coordinates, so only one physical label
    (e.g. P4) per location.
    """
    seen = set()
    for n in nodes:
        lab = n["label"].split("#")[0]          # copy 'P4#2' -> 'P4'
        key = (round(n["x"], 2), round(n["y"], 2), lab)
        if key in seen:
            continue
        seen.add(key)
        if n["type"] == "parking":
            col = "#08306b"
        elif n["type"] == "depot":
            col = "#7f1d1d"
        else:
            col = "#222222"
        t = ax.text(n["x"] + 0.35, n["y"] + 0.35, lab, fontsize=7,
                    fontweight="bold", color=col, ha="left", va="bottom",
                    zorder=11)
        t.set_path_effects(
            [pe.withStroke(linewidth=2.2, foreground="white")])


def _arrow(ax, ni, nj, color, lw, ls="-"):
    ax.annotate("", xy=(nj["x"], nj["y"]), xytext=(ni["x"], ni["y"]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, alpha=0.85,
                                shrinkA=4, shrinkB=4),
                zorder=3)


class _HC(HandlerPatch):
    def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
        c = Circle((w * 0.5 - xd, h * 0.5 - yd), h * 0.45,
                   facecolor=orig.get_facecolor(),
                   edgecolor=orig.get_edgecolor(),
                   linestyle=orig.get_linestyle(),
                   linewidth=orig.get_linewidth())
        c.set_transform(trans)
        return [c]


def _legend(ax, with_routes):
    h = [
        Line2D([0], [0], marker="*", color="none",
               markerfacecolor=DEPOT_FC, markeredgecolor=DEPOT_EC,
               markersize=18, linestyle="None", label="Main depot"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=PARK_FC, markeredgecolor=PARK_EC,
               markersize=12, linestyle="None", label="Parking node"),
    ]
    if with_routes:
        h += [
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
    else:
        h += [Circle((0, 0), 1, facecolor=CUST_FC, edgecolor=CUST_EC,
                     linewidth=1.3, linestyle="-", label="Customer")]
    return ax.legend(handles=h, loc="center left",
                     bbox_to_anchor=(1.0, 0.5), fontsize=11,
                     framealpha=1.0, borderpad=1.0, labelspacing=1.3,
                     handlelength=2.2, handletextpad=0.8,
                     handler_map={Circle: _HC()})


def _fleet_box(fig, ax, main_leg, n_trucks, n_robots):
    """Box with the used truck/robot counts, aligned under the legend."""
    ax.add_artist(main_leg)                     # keep both legends
    fig.canvas.draw()
    bb = main_leg.get_window_extent().transformed(
        ax.transAxes.inverted())
    dummy = (Line2D([0], [0], color="none", linestyle="None"),
             Line2D([0], [0], color="none", linestyle="None"))
    return ax.legend(dummy,
                     [f"Trucks used: {n_trucks}",
                      f"Robots used: {n_robots}"],
                     loc="upper left", bbox_to_anchor=(1.0, bb.y0 - 0.03),
                     fontsize=12, framealpha=1.0, borderpad=1.0,
                     labelspacing=0.7, handlelength=0, handletextpad=0)


def make_instance_svg(inst_json, out_svg):
    with open(inst_json) as f:
        inst = json.load(f)
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_grid(ax, inst)
    _draw_nodes(ax, inst["nodes"])
    leg = _legend(ax, with_routes=False)
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                bbox_extra_artists=[leg])
    plt.close(fig)
    return out_svg


def make_route_svg(inst_json, sol_json, out_svg, numbering=True):
    with open(inst_json) as f:
        inst = json.load(f)
    with open(sol_json) as f:
        sol = json.load(f)
    nmap = {n["label"]: n for n in sol["nodes"]}
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_grid(ax, inst)
    _draw_nodes(ax, sol["nodes"])
    for a in sol["truck_arcs"]:
        if a["i"] in nmap and a["j"] in nmap:
            _arrow(ax, nmap[a["i"]], nmap[a["j"]], TRUCK_COL, 2.0)
    for a in sol["robot_arcs"]:
        if a["i"] in nmap and a["j"] in nmap:
            _arrow(ax, nmap[a["i"]], nmap[a["j"]], ROBOT_COL, 1.4,
                   ls="--")
    if numbering:
        _draw_labels(ax, sol["nodes"])
    leg = _legend(ax, with_routes=True)
    n_trucks = len({a["k"] for a in sol["truck_arcs"]})
    n_robots = len({(a["k"], a["r"]) for a in sol["robot_arcs"]})
    fleet_leg = _fleet_box(fig, ax, leg, n_trucks, n_robots)
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                bbox_extra_artists=[leg, fleet_leg])
    plt.close(fig)
    return out_svg
