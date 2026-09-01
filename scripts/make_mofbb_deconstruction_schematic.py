#!/usr/bin/env python3
"""Render a data-grounded schematic of the MOFBB deconstruction workflow.

The figure uses the representative building blocks, exact reconstruction maps,
and periodic topology records bundled with the MOFBB examples.  It intentionally
shows a diverse set of outputs: metal nodes, organic nodes, and two-connected
edges.  All visual rendering and export are performed with Matplotlib.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from itertools import product
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import numpy as np
from ase.geometry import cellpar_to_cell

from mofbb.compose import Reconstructor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "ink": "#252A30",
    "muted": "#66717C",
    "line": "#6F7880",
    "cell": "#B8C0C7",
    "divider": "#DCE2E7",
    "row": "#F7F9FA",
    "sidebar": "#E9F0F5",
    "metal_node": "#5276C5",
    "metal_node_light": "#E9EEF9",
    "organic_node": "#4E9B86",
    "organic_node_light": "#E5F2EE",
    "edge": "#D18B3C",
    "edge_light": "#FAEEDC",
    "x": "#C33C7B",
}

ELEMENT_COLORS = {
    "H": "#D6DADF",
    "C": "#555A60",
    "N": "#466FB3",
    "O": "#E14B4B",
    "F": "#6BBF59",
    "S": "#D4A72C",
    "P": "#E68A3A",
    "Cl": "#55A868",
    "Cu": "#3C78B4",
    "Zn": "#8068B2",
    "Zr": "#35A0A4",
    "Ni": "#4E9A66",
    "X": COLORS["x"],
}

KIND_LABELS = {
    "metal_node": "metal node",
    "organic_node": "organic node",
    "edge": "edge",
}

SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@dataclass(frozen=True)
class ExampleSpec:
    directory: str
    label: str
    azimuth: float
    elevation: float


EXAMPLES = (
    ExampleSpec("HKUST1", "HKUST‑1", 42.0, 25.0),
    ExampleSpec("MOF5", "MOF‑5", 42.0, 23.0),
    ExampleSpec("UiO-67", "UiO‑67", -38.0, 24.0),
    ExampleSpec("CPO-27-Ni", "CPO‑27‑Ni", 25.0, 20.0),
    ExampleSpec("Zn2-TTFB", "Zn₂‑TTFB", -35.0, 22.0),
)


@dataclass
class XYZBlock:
    symbols: list[str]
    positions: np.ndarray
    bonds: np.ndarray


@dataclass
class CGDNet:
    cell: np.ndarray
    nodes: np.ndarray
    coordination: np.ndarray
    edges: list[tuple[np.ndarray, np.ndarray]]


def apply_style() -> None:
    """Set a compact publication-style Matplotlib theme."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_xyz(path: Path) -> XYZBlock:
    """Read the extended XYZ flavour emitted by MOFBB."""

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) < 2:
        raise ValueError(f"Incomplete XYZ file: {path}")
    atom_count = int(lines[0])
    atom_lines = lines[2 : 2 + atom_count]
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in atom_lines:
        tokens = line.split()
        symbols.append(tokens[0])
        positions.append([float(value) for value in tokens[1:4]])

    bonds: list[tuple[int, int]] = []
    for line in lines[2 + atom_count :]:
        tokens = line.split()
        if len(tokens) < 2 or not re.fullmatch(r"[+-]?\d+", tokens[0]):
            continue
        bonds.append((int(tokens[0]), int(tokens[1])))
    return XYZBlock(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        bonds=np.asarray(bonds, dtype=int).reshape(-1, 2),
    )


def parse_cgd(path: Path) -> CGDNet:
    """Read the cell, topology vertices, and explicit periodic edges."""

    cell_parameters: list[float] | None = None
    nodes: list[list[float]] = []
    coordination: list[int] = []
    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        tokens = raw_line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if tokens[0] == "CELL":
            cell_parameters = [float(value) for value in tokens[1:7]]
        elif tokens[0] == "NODE":
            coordination.append(int(tokens[2]))
            nodes.append([float(value) for value in tokens[3:6]])
        elif tokens[0] == "EDGE":
            left = np.asarray([float(value) for value in tokens[1:4]], dtype=float)
            right = np.asarray([float(value) for value in tokens[4:7]], dtype=float)
            edges.append((left, right))
    if cell_parameters is None or not nodes:
        raise ValueError(f"CGD file lacks a cell or vertices: {path}")
    return CGDNet(
        cell=cellpar_to_cell(cell_parameters),
        nodes=np.asarray(nodes, dtype=float),
        coordination=np.asarray(coordination, dtype=int),
        edges=edges,
    )


def view_projection(
    points: np.ndarray,
    azimuth: float,
    elevation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Cartesian coordinates using an orthographic camera."""

    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    right = np.asarray([-np.sin(az), np.cos(az), 0.0])
    up = np.asarray(
        [-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)]
    )
    forward = np.asarray(
        [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
    )
    return np.column_stack((points @ right, points @ up)), points @ forward


def pca_projection(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose a deterministic principal-axis projection for a building block."""

    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if len(vh) == 2:
        vh = np.vstack((vh, np.cross(vh[0], vh[1])))
    projected = centered @ vh[:2].T
    depth = centered @ vh[2]
    if np.ptp(projected[:, 1]) > np.ptp(projected[:, 0]) * 1.25:
        projected = projected[:, ::-1]
    max_index = int(np.argmax(np.linalg.norm(projected, axis=1)))
    if projected[max_index, 0] < 0:
        projected[:, 0] *= -1
    return projected, depth


def fit_to_box(
    points: np.ndarray,
    center: tuple[float, float],
    box: tuple[float, float],
    padding: float = 0.08,
) -> np.ndarray:
    """Scale projected coordinates into a data-coordinate rectangle."""

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    usable = np.asarray(box, dtype=float) * (1.0 - 2.0 * padding)
    scale = float(np.min(usable / span))
    midpoint = (minimum + maximum) / 2.0
    return (points - midpoint) * scale + np.asarray(center, dtype=float)


def make_clip(
    ax: plt.Axes,
    center: tuple[float, float],
    box: tuple[float, float],
) -> Rectangle:
    patch = Rectangle(
        (center[0] - box[0] / 2.0, center[1] - box[1] / 2.0),
        box[0],
        box[1],
        facecolor="none",
        edgecolor="none",
        transform=ax.transData,
    )
    ax.add_patch(patch)
    return patch


def cell_corners(cell: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    bits = list(product((0.0, 1.0), repeat=3))
    corners = np.asarray(bits, dtype=float) @ cell
    edges = [
        (left, right)
        for left, a in enumerate(bits)
        for right, b in enumerate(bits)
        if left < right and sum(x != y for x, y in zip(a, b)) == 1
    ]
    return corners, edges


def central_periodic_segment(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest-image edge copy closest to the unit-cell centre."""

    left_wrapped = np.mod(left, 1.0)
    right_wrapped = np.mod(right, 1.0)
    left_wrapped[np.isclose(np.mod(left, 1.0), 0.0) & (left > 0.5)] = 1.0
    right_wrapped[np.isclose(np.mod(right, 1.0), 0.0) & (right > 0.5)] = 1.0
    delta = right - left
    delta -= np.round(delta)
    candidates = (
        (left_wrapped, left_wrapped + delta),
        (right_wrapped, right_wrapped - delta),
    )
    return min(candidates, key=lambda pair: np.linalg.norm(np.mean(pair, axis=0) - 0.5))


def add_collection(
    ax: plt.Axes,
    segments: Iterable[np.ndarray],
    *,
    color: str,
    linewidth: float,
    alpha: float,
    clip: Rectangle,
    zorder: float,
) -> LineCollection:
    collection = LineCollection(
        list(segments),
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    collection.set_clip_path(clip)
    ax.add_collection(collection)
    return collection


def draw_framework(
    ax: plt.Axes,
    decomposition_dir: Path,
    center: tuple[float, float],
    box: tuple[float, float],
    azimuth: float,
    elevation: float,
) -> None:
    """Draw one exact reconstructed unit cell as a compact ball-and-stick view."""

    framework = Reconstructor(decomposition_dir).reconstruct()
    atoms = framework.atoms
    cell = np.asarray(atoms.cell)
    fractional = atoms.get_scaled_positions(wrap=True)
    cartesian = fractional @ cell
    corners, cell_edge_indices = cell_corners(cell)

    fit_cartesian = np.vstack((cartesian, corners))
    fit_projected, _ = view_projection(fit_cartesian, azimuth, elevation)
    fitted = fit_to_box(fit_projected, center, box, padding=0.06)
    atom_xy = fitted[: len(cartesian)]
    corner_xy = fitted[len(cartesian) :]
    _, atom_depth = view_projection(cartesian, azimuth, elevation)
    clip = make_clip(ax, center, box)

    cell_segments = [corner_xy[[left, right]] for left, right in cell_edge_indices]
    add_collection(
        ax,
        cell_segments,
        color=COLORS["cell"],
        linewidth=0.55,
        alpha=0.65,
        clip=clip,
        zorder=0.5,
    )

    projected_bonds: list[np.ndarray] = []
    bond_depths: list[float] = []
    symbols = atoms.get_chemical_symbols()
    all_segment_cartesian: list[np.ndarray] = []
    for left_index, right_index in np.asarray(framework.bonds, dtype=int):
        if symbols[left_index] == "H" or symbols[right_index] == "H":
            continue
        left_frac, right_frac = central_periodic_segment(
            fractional[left_index], fractional[right_index]
        )
        segment_cartesian = np.vstack((left_frac @ cell, right_frac @ cell))
        all_segment_cartesian.append(segment_cartesian)
        _, depth = view_projection(segment_cartesian, azimuth, elevation)
        bond_depths.append(float(np.mean(depth)))

    if all_segment_cartesian:
        flattened = np.vstack(all_segment_cartesian)
        projected, _ = view_projection(flattened, azimuth, elevation)
        minimum = np.min(fit_projected, axis=0)
        maximum = np.max(fit_projected, axis=0)
        span = np.maximum(maximum - minimum, 1e-9)
        usable = np.asarray(box) * (1.0 - 2.0 * 0.06)
        scale = float(np.min(usable / span))
        midpoint = (minimum + maximum) / 2.0
        projected = (projected - midpoint) * scale + np.asarray(center)
        projected_bonds = list(projected.reshape(-1, 2, 2))
        order = np.argsort(bond_depths)
        add_collection(
            ax,
            [projected_bonds[index] for index in order],
            color="#747A80",
            linewidth=0.42,
            alpha=0.50,
            clip=clip,
            zorder=1.0,
        )

    parent_sizes = {
        "C": 4.0,
        "N": 5.0,
        "O": 5.5,
        "S": 7.5,
        "Cu": 13.0,
        "Zn": 13.0,
        "Zr": 14.0,
        "Ni": 12.0,
    }
    non_hydrogen = [index for index, symbol in enumerate(symbols) if symbol != "H"]
    for symbol in sorted({symbols[index] for index in non_hydrogen}):
        indices = [index for index in non_hydrogen if symbols[index] == symbol]
        indices.sort(key=lambda index: atom_depth[index])
        artist = ax.scatter(
            atom_xy[indices, 0],
            atom_xy[indices, 1],
            s=parent_sizes.get(symbol, 6.0),
            c=ELEMENT_COLORS.get(symbol, "#888888"),
            edgecolors="white",
            linewidths=0.16,
            alpha=0.94,
            zorder=2.0,
        )
        artist.set_clip_path(clip)


def draw_block(
    ax: plt.Axes,
    path: Path,
    center: tuple[float, float],
    box: tuple[float, float],
) -> None:
    """Draw a grouped X-capped building block."""

    block = parse_xyz(path)
    projected, depth = pca_projection(block.positions)
    xy = fit_to_box(projected, center, box, padding=0.07)
    clip = make_clip(ax, center, box)

    normal_segments: list[np.ndarray] = []
    x_segments: list[np.ndarray] = []
    for left, right in block.bonds:
        target = x_segments if "X" in (block.symbols[left], block.symbols[right]) else normal_segments
        target.append(xy[[left, right]])
    add_collection(
        ax,
        normal_segments,
        color="#70757A",
        linewidth=0.75,
        alpha=0.9,
        clip=clip,
        zorder=1.0,
    )
    add_collection(
        ax,
        x_segments,
        color=COLORS["x"],
        linewidth=0.75,
        alpha=0.82,
        clip=clip,
        zorder=1.1,
    )

    sizes = {
        "H": 9.0,
        "C": 17.0,
        "N": 19.0,
        "O": 21.0,
        "S": 25.0,
        "Cu": 34.0,
        "Zn": 34.0,
        "Zr": 36.0,
        "Ni": 32.0,
        "X": 23.0,
    }
    for index in np.argsort(depth):
        symbol = block.symbols[index]
        if symbol == "X":
            artist = ax.scatter(
                [xy[index, 0]],
                [xy[index, 1]],
                s=sizes[symbol],
                facecolors="white",
                edgecolors=COLORS["x"],
                linewidths=1.0,
                zorder=3.0,
            )
        else:
            artist = ax.scatter(
                [xy[index, 0]],
                [xy[index, 1]],
                s=sizes.get(symbol, 17.0),
                c=ELEMENT_COLORS.get(symbol, "#888888"),
                edgecolors="white",
                linewidths=0.35,
                zorder=2.0,
            )
        artist.set_clip_path(clip)


def draw_topology(
    ax: plt.Axes,
    path: Path,
    kinds: list[str],
    center: tuple[float, float],
    box: tuple[float, float],
    azimuth: float,
    elevation: float,
) -> None:
    """Draw a connectivity-coloured periodic CGD net."""

    net = parse_cgd(path)
    node_cartesian = net.nodes @ net.cell
    corners, cell_edge_indices = cell_corners(net.cell)
    fit_cartesian = np.vstack((node_cartesian, corners))
    fit_projected, _ = view_projection(fit_cartesian, azimuth, elevation)
    fitted = fit_to_box(fit_projected, center, box, padding=0.055)
    node_xy = fitted[: len(node_cartesian)]
    corner_xy = fitted[len(node_cartesian) :]
    _, node_depth = view_projection(node_cartesian, azimuth, elevation)
    clip = make_clip(ax, center, box)

    add_collection(
        ax,
        [corner_xy[[left, right]] for left, right in cell_edge_indices],
        color=COLORS["cell"],
        linewidth=0.55,
        alpha=0.75,
        clip=clip,
        zorder=0.5,
    )

    segment_cartesian: list[np.ndarray] = []
    for left, right in net.edges:
        left_frac, right_frac = central_periodic_segment(left, right)
        segment_cartesian.append(np.vstack((left_frac @ net.cell, right_frac @ net.cell)))
    if segment_cartesian:
        flat = np.vstack(segment_cartesian)
        projected, _ = view_projection(flat, azimuth, elevation)
        minimum = np.min(fit_projected, axis=0)
        maximum = np.max(fit_projected, axis=0)
        span = np.maximum(maximum - minimum, 1e-9)
        usable = np.asarray(box) * (1.0 - 2.0 * 0.055)
        scale = float(np.min(usable / span))
        midpoint = (minimum + maximum) / 2.0
        projected = (projected - midpoint) * scale + np.asarray(center)
        add_collection(
            ax,
            projected.reshape(-1, 2, 2),
            color=COLORS["line"],
            linewidth=0.62,
            alpha=0.78,
            clip=clip,
            zorder=1.0,
        )

    if len(kinds) != len(net.nodes):
        kinds = ["metal_node" if cn > 4 else "organic_node" if cn > 2 else "edge" for cn in net.coordination]
    for kind in ("edge", "organic_node", "metal_node"):
        indices = [index for index, value in enumerate(kinds) if value == kind]
        indices.sort(key=lambda index: node_depth[index])
        if not indices:
            continue
        sizes = 9.0 + 1.4 * net.coordination[indices]
        artist = ax.scatter(
            node_xy[indices, 0],
            node_xy[indices, 1],
            s=sizes,
            c=COLORS[kind],
            edgecolors="white",
            linewidths=0.38,
            zorder=2.0,
        )
        artist.set_clip_path(clip)


def category_positions(center: float, column_width: float, count: int) -> tuple[list[float], float]:
    if count == 1:
        return [center], column_width * 0.78
    if count == 2:
        return [center - column_width * 0.235, center + column_width * 0.235], column_width * 0.42
    offsets = np.linspace(-0.31, 0.31, count)
    return [center + column_width * offset for offset in offsets], column_width * 0.285


def down_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=0.85,
            color=COLORS["ink"],
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        )
    )


def formula_text(formula: str) -> str:
    return formula.translate(SUBSCRIPT)


def draw_sidebar(ax: plt.Axes) -> None:
    arrow = Polygon(
        [(0.012, 0.89), (0.098, 0.89), (0.098, 0.105), (0.055, 0.058), (0.012, 0.105)],
        closed=True,
        facecolor=COLORS["sidebar"],
        edgecolor="none",
        zorder=-1,
    )
    ax.add_patch(arrow)
    ax.text(
        0.055,
        0.925,
        "DECONSTRUCTION",
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=COLORS["muted"],
        rotation=90,
    )
    labels = (
        (0.755, "INPUT", "parent\nMOF"),
        (0.475, "OUTPUT 1", "reusable\nbuilding blocks"),
        (0.195, "OUTPUT 2", "periodic\ntopology"),
    )
    for y, kicker, label in labels:
        ax.text(
            0.055,
            y + 0.045,
            kicker,
            ha="center",
            va="center",
            fontsize=5.6,
            fontweight="bold",
            color=COLORS["muted"],
        )
        ax.text(
            0.055,
            y - 0.002,
            label,
            ha="center",
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color=COLORS["ink"],
            linespacing=1.08,
        )


def create_figure(examples_root: Path, output_dir: Path, dpi: int) -> dict[str, Path]:
    apply_style()
    fig = plt.figure(figsize=(12.6, 6.25))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    ax.add_patch(Rectangle((0.105, 0.64), 0.882, 0.245, facecolor=COLORS["row"], edgecolor="none", zorder=-3))
    ax.add_patch(Rectangle((0.105, 0.35), 0.882, 0.245, facecolor="#FBFCFC", edgecolor="none", zorder=-3))
    ax.add_patch(Rectangle((0.105, 0.105), 0.882, 0.185, facecolor=COLORS["row"], edgecolor="none", zorder=-3))
    draw_sidebar(ax)

    ax.text(
        0.112,
        0.959,
        "MOFBB deconstructs periodic frameworks into reusable, connectivity-aware components",
        ha="left",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.112,
        0.923,
        "Exact unit-cell reconstruction  →  X-capped building-block types  +  connectivity-preserving CGD net",
        ha="left",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )
    ax.plot([0.112, 0.987], [0.904, 0.904], color=COLORS["divider"], linewidth=0.8)

    content_left = 0.115
    content_right = 0.985
    gutter = 0.012
    column_width = (content_right - content_left - gutter * (len(EXAMPLES) - 1)) / len(EXAMPLES)
    manifests: list[dict] = []

    for example_index, spec in enumerate(EXAMPLES):
        left = content_left + example_index * (column_width + gutter)
        center_x = left + column_width / 2.0
        example_dir = examples_root / spec.directory
        summary = load_json(example_dir / "decomposition_summary.json")
        reconstruction = load_json(example_dir / "reconstruction_map.json")
        groups = summary["groups"]
        topology_record = summary["topology"]

        if example_index:
            separator_x = left - gutter / 2.0
            ax.plot(
                [separator_x, separator_x],
                [0.105, 0.885],
                color=COLORS["divider"],
                linewidth=0.55,
                linestyle=(0, (2.0, 3.0)),
                zorder=-1,
            )

        ax.text(
            center_x,
            0.872,
            spec.label,
            ha="center",
            va="top",
            fontsize=9.7,
            fontweight="bold",
            color=COLORS["ink"],
        )
        draw_framework(
            ax,
            example_dir,
            center=(center_x, 0.752),
            box=(column_width * 0.94, 0.205),
            azimuth=spec.azimuth,
            elevation=spec.elevation,
        )

        group_x, block_width = category_positions(center_x, column_width, len(groups))
        branch_y = 0.618
        ax.plot([center_x, center_x], [0.642, branch_y], color=COLORS["ink"], linewidth=0.8)
        if len(group_x) > 1:
            ax.plot([min(group_x), max(group_x)], [branch_y, branch_y], color=COLORS["ink"], linewidth=0.8)
        for x_position in group_x:
            down_arrow(ax, (x_position, branch_y), (x_position, 0.594))

        for group_index, (group, x_position) in enumerate(zip(groups, group_x)):
            kind = str(group["kind"])
            label_font = 5.5 if len(groups) > 2 else 6.1
            ax.text(
                x_position,
                0.575,
                KIND_LABELS[kind],
                ha="center",
                va="center",
                fontsize=label_font,
                fontweight="bold",
                color=COLORS[kind],
                bbox={
                    "boxstyle": "round,pad=0.20,rounding_size=0.25",
                    "facecolor": COLORS[f"{kind}_light"],
                    "edgecolor": COLORS[kind],
                    "linewidth": 0.45,
                },
                zorder=6,
            )
            draw_block(
                ax,
                example_dir / group["path"],
                center=(x_position, 0.486),
                box=(block_width, 0.125),
            )
            formula_size = 4.6 if len(groups) > 2 else 5.2
            ax.text(
                x_position,
                0.406,
                formula_text(str(group["formula"])),
                ha="center",
                va="top",
                fontsize=formula_size,
                fontweight="bold",
                color=COLORS["ink"],
            )
            ax.text(
                x_position,
                0.385,
                f"CN {int(group['cn'])}  ·  n={int(group['count'])}",
                ha="center",
                va="top",
                fontsize=formula_size - 0.2,
                color=COLORS["muted"],
            )

        merge_y = 0.342
        for x_position in group_x:
            ax.plot([x_position, x_position], [0.364, merge_y], color=COLORS["ink"], linewidth=0.75)
        if len(group_x) > 1:
            ax.plot([min(group_x), max(group_x)], [merge_y, merge_y], color=COLORS["ink"], linewidth=0.75)
        down_arrow(ax, (center_x, merge_y), (center_x, 0.297))

        slot_kinds = [str(slot["kind"]) for slot in reconstruction["slots"]]
        draw_topology(
            ax,
            example_dir / topology_record["path"],
            kinds=slot_kinds,
            center=(center_x, 0.203),
            box=(column_width * 0.93, 0.152),
            azimuth=spec.azimuth,
            elevation=spec.elevation,
        )
        ax.text(
            center_x,
            0.116,
            f"{int(topology_record['num_vertices'])} vertices  ·  {int(topology_record['num_edges'])} edges",
            ha="center",
            va="center",
            fontsize=5.7,
            color=COLORS["muted"],
        )

        manifests.append(
            {
                "example": spec.directory,
                "display_label": spec.label,
                "groups": [
                    {
                        key: group[key]
                        for key in ("bb_id", "kind", "formula", "cn", "count", "path")
                    }
                    for group in groups
                ],
                "topology": topology_record,
                "reconstructed_atoms": len(Reconstructor(example_dir).reconstruct().atoms),
            }
        )

    legend_y = 0.047
    ax.text(
        0.112,
        legend_y,
        "Encoding key:",
        ha="left",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=COLORS["ink"],
    )
    legend_items = (
        ("metal_node", "metal node"),
        ("organic_node", "organic node"),
        ("edge", "edge"),
    )
    legend_x = 0.205
    for kind, label in legend_items:
        ax.scatter([legend_x], [legend_y], s=23, c=COLORS[kind], edgecolors="white", linewidths=0.4, zorder=3)
        ax.text(legend_x + 0.011, legend_y, label, ha="left", va="center", fontsize=6.2, color=COLORS["muted"])
        legend_x += 0.115
    ax.scatter([legend_x], [legend_y], s=24, facecolors="white", edgecolors=COLORS["x"], linewidths=1.0, zorder=3)
    ax.text(legend_x + 0.011, legend_y, "X connection site", ha="left", va="center", fontsize=6.2, color=COLORS["muted"])
    ax.text(
        0.986,
        legend_y,
        "Representative grouped blocks; multiplicity n refers to one reconstructed unit cell",
        ha="right",
        va="center",
        fontsize=5.7,
        color=COLORS["muted"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "mofbb_deconstruction_schematic"
    outputs = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
    }
    fig.savefig(outputs["svg"], metadata={"Title": "MOFBB deconstruction schematic"})
    fig.savefig(outputs["pdf"], metadata={"Title": "MOFBB deconstruction schematic"})
    fig.savefig(outputs["png"], dpi=dpi, metadata={"Title": "MOFBB deconstruction schematic"})
    plt.close(fig)

    source_dir = output_dir.parent / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "figure_contract": {
            "core_conclusion": (
                "MOFBB deconstructs chemically diverse periodic frameworks into explicit, "
                "X-capped reusable components while retaining a connectivity-preserving periodic net."
            ),
            "archetype": "schematic-led composite",
            "backend": "Python / Matplotlib",
            "final_size_inches": [12.6, 6.25],
            "exports": ["SVG", "PDF", f"PNG ({dpi} dpi)"],
            "reviewer_risk": (
                "The CGD drawings preserve recorded connectivity but are projections, not canonical net-name assignments."
            ),
        },
        "examples": manifests,
    }
    manifest_path = source_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outputs["manifest"] = manifest_path
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=REPOSITORY_ROOT / "examples" / "node-edge-decomposed",
        help="Directory containing the prepared MOFBB decomposition examples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "publication_figures" / "figures",
        help="Destination for SVG, PDF, and PNG files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch. Default: 600.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = create_figure(args.examples_root, args.output_dir, args.dpi)
    for file_type, path in outputs.items():
        print(f"{file_type.upper()}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
