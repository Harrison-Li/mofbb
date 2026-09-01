#!/usr/bin/env python3
"""Create publication figures from the audited MOFBB experiment outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from pymatgen.symmetry.groups import SpaceGroup


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "strict": "#0F4D92",
    "fallback": "#B64342",
    "experimental": "#6F6F6F",
    "edge": "#D88949",
    "metal_node": "#7C6CCF",
    "organic_node": "#33A99A",
    "pass": "#0F4D92",
    "fail": "#E9A6A1",
    "missing": "#E6E6E6",
    "ink": "#272727",
    "neutral": "#A8A8A8",
    "light": "#E7E7E7",
}

KIND_LABELS = {
    "edge": "Edge",
    "metal_node": "Metal node",
    "organic_node": "Organic node",
}


def apply_style() -> None:
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
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(direction="out")


def export_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": figure_dir / f"{stem}.svg",
        "pdf": figure_dir / f"{stem}.pdf",
        "tiff": figure_dir / f"{stem}.tiff",
        "png": figure_dir / f"{stem}.png",
    }
    fig.savefig(paths["svg"])
    fig.savefig(paths["pdf"])
    fig.savefig(paths["tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=300)
    plt.close(fig)
    return {key: str(value) for key, value in paths.items()}


def write_csv(frame: pd.DataFrame, output_dir: Path, name: str) -> Path:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / name
    frame.to_csv(path, index=False)
    return path


def parse_bool(value: object) -> float:
    if pd.isna(value) or value == "":
        return np.nan
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no"}:
        return 0.0
    return np.nan


def canonical_space_group_numbers(symbols: pd.Series) -> tuple[pd.Series, list[str]]:
    mapping: dict[str, float] = {}
    failures: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for symbol in symbols.astype(str).unique():
            try:
                mapping[symbol] = float(SpaceGroup(symbol).int_number)
            except Exception:
                mapping[symbol] = np.nan
                failures.append(symbol)
    return symbols.astype(str).map(mapping), failures


def empirical_cdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy())
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def ccdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(pd.to_numeric(values, errors="coerce").dropna().astype(int))
    x = np.array(sorted(counts), dtype=float)
    cumulative = np.cumsum([counts[int(value)] for value in x])
    y = 1.0 - (cumulative - np.array([counts[int(value)] for value in x])) / sum(counts.values())
    return x, y


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    value: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=color,
        edgecolor="white",
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.62,
        value,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + width / 2,
        y + height * 0.27,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color="white",
        linespacing=1.05,
    )


def draw_arrow(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    arrow = FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        color="#777777",
    )
    ax.add_patch(arrow)


def prepare_data(project_root: Path, output_dir: Path) -> dict:
    paths = {
        "experimental": project_root / "dataset" / "ccdc_valid_primitive_manifest.csv",
        "library_summary": project_root / "lib" / "summary.json",
        "blocks": project_root / "lib" / "building_blocks_manifest.csv",
        "structures": project_root / "lib" / "structures_manifest.csv",
        "roundtrip_summary": project_root
        / "mofbb"
        / "outputs"
        / "assembly_roundtrip_100_combined_20260807"
        / "summary.json",
        "roundtrip_cases": project_root
        / "mofbb"
        / "outputs"
        / "assembly_roundtrip_100_combined_20260807"
        / "comparison.csv",
        "strict_summary": project_root / "10K_mofs" / "generation_summary.json",
        "strict_manifest": project_root / "10K_mofs" / "manifest.csv",
        "fallback_summary": project_root
        / "10k_mofs_diverse_v2"
        / "generation_summary.json",
        "fallback_manifest": project_root / "10k_mofs_diverse_v2" / "manifest.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required experiment outputs: " + ", ".join(missing))

    experimental = pd.read_csv(paths["experimental"])
    blocks = pd.read_csv(paths["blocks"])
    structures = pd.read_csv(paths["structures"])
    roundtrip_cases = pd.read_csv(paths["roundtrip_cases"])
    strict_manifest = pd.read_csv(paths["strict_manifest"])
    fallback_manifest = pd.read_csv(paths["fallback_manifest"])
    library_summary = load_json(paths["library_summary"])
    roundtrip_summary = load_json(paths["roundtrip_summary"])
    strict_summary = load_json(paths["strict_summary"])
    fallback_summary = load_json(paths["fallback_summary"])

    blocks["source_family"] = np.where(
        blocks["source_refcode"].astype(str).str.startswith("PORMAKE:"),
        "Imported curated",
        "Experimental",
    )

    strict_topology_number, strict_parse_failures = canonical_space_group_numbers(
        strict_manifest["topology_space_group"]
    )
    fallback_topology_number, fallback_parse_failures = canonical_space_group_numbers(
        fallback_manifest["topology_space_group"]
    )
    strict_manifest["topology_space_group_number"] = strict_topology_number
    fallback_manifest["topology_space_group_number"] = fallback_topology_number
    strict_manifest["space_group_retained"] = (
        strict_manifest["topology_space_group_number"]
        == strict_manifest["realized_space_group_number"]
    )
    fallback_manifest["space_group_retained"] = (
        fallback_manifest["topology_space_group_number"]
        == fallback_manifest["cif_space_group_number"]
    )

    data = {
        "paths": paths,
        "experimental": experimental,
        "blocks": blocks,
        "structures": structures,
        "library_summary": library_summary,
        "roundtrip_cases": roundtrip_cases,
        "roundtrip_summary": roundtrip_summary,
        "strict_manifest": strict_manifest,
        "strict_summary": strict_summary,
        "fallback_manifest": fallback_manifest,
        "fallback_summary": fallback_summary,
        "space_group_parse_failures": {
            "strict": strict_parse_failures,
            "fallback": fallback_parse_failures,
        },
    }

    source_manifest = pd.DataFrame(
        [
            {
                "source_role": role,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for role, path in paths.items()
        ]
    )
    write_csv(source_manifest, output_dir, "input_source_manifest.csv")
    return data


def make_figure_1(data: dict, output_dir: Path) -> dict:
    summary = data["library_summary"]
    blocks = data["blocks"]
    experimental_blocks = blocks[blocks["source_family"] == "Experimental"].copy()

    flow = pd.DataFrame(
        [
            ("Quality-controlled MOFs", int(summary["records"])),
            ("Successfully decomposed", int(summary["status_counts"]["deconstructed"])),
            ("Fragment instances", int(sum(summary["fragment_instance_counts"].values()))),
            (
                "Unique experimental blocks",
                int(sum(summary["unique_building_block_counts"].values())),
            ),
            ("Available library", int(summary["current_building_block_file_count"])),
        ],
        columns=["stage", "count"],
    )
    write_csv(flow, output_dir, "fig1a_flow_counts.csv")

    rows = []
    for kind in ["edge", "metal_node", "organic_node"]:
        rows.append(
            {
                "kind": kind,
                "fragment_instances": int(summary["fragment_instance_counts"][kind]),
                "unique_experimental_blocks": int(
                    summary["unique_building_block_counts"][kind]
                ),
                "imported_curated_blocks": int(
                    ((blocks["kind"] == kind) & (blocks["source_family"] == "Imported curated")).sum()
                ),
                "available_blocks": int(summary["current_building_block_file_counts"][kind]),
            }
        )
    kind_counts = pd.DataFrame(rows)
    write_csv(kind_counts, output_dir, "fig1b_component_counts.csv")

    def cn_bin(value: int) -> str:
        if value <= 4:
            return f"CN{value}"
        if value <= 6:
            return "CN5–6"
        if value <= 8:
            return "CN7–8"
        if value <= 12:
            return "CN9–12"
        return "CN>12"

    cn_order = ["CN1", "CN2", "CN3", "CN4", "CN5–6", "CN7–8", "CN9–12", "CN>12"]
    experimental_blocks["coordination_bin"] = (
        pd.to_numeric(experimental_blocks["logical_cn"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .map(cn_bin)
    )
    coordination = (
        experimental_blocks.groupby(["kind", "coordination_bin"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    coordination["fraction"] = coordination["count"] / coordination.groupby("kind")[
        "count"
    ].transform("sum")
    coordination["coordination_bin"] = pd.Categorical(
        coordination["coordination_bin"], categories=cn_order, ordered=True
    )
    coordination = coordination.sort_values(["kind", "coordination_bin"])
    write_csv(coordination, output_dir, "fig1c_coordination_distribution.csv")
    write_csv(
        experimental_blocks[
            ["name", "kind", "logical_cn", "instance_count", "structure_count", "formula"]
        ],
        output_dir,
        "fig1d_block_reuse.csv",
    )

    fig = plt.figure(figsize=(7.2, 6.05))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.95, 1.55],
        left=0.105,
        right=0.985,
        top=0.965,
        bottom=0.105,
        hspace=0.60,
        wspace=0.54,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    panel_label(ax_a, "a", x=-0.035, y=1.02)
    ax_a.set_axis_off()
    titles = [
        "quality-controlled\nexperimental MOFs",
        "successfully\ndecomposed",
        "fragment\ninstances",
        "unique experimental\nbuilding blocks",
        "available blocks\nafter curated import",
    ]
    box_colors = ["#6F6F6F", "#526D82", "#4B7F91", "#4D779E", COLORS["strict"]]
    xs = [0.015, 0.215, 0.415, 0.615, 0.815]
    values = [f"{value:,}" for value in flow["count"]]
    for x, title, value, color in zip(xs, titles, values, box_colors, strict=True):
        draw_box(ax_a, x, 0.21, 0.17, 0.59, title, value, color)
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        draw_arrow(ax_a, left + 0.17, right, 0.505)
    success = 100 * summary["status_counts"]["deconstructed"] / summary["records"]
    ax_a.text(
        0.20,
        0.87,
        f"{success:.1f}% decomposition success",
        transform=ax_a.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=COLORS["ink"],
    )
    imported = int((blocks["source_family"] == "Imported curated").sum())
    ax_a.text(
        0.80,
        0.87,
        f"+{imported:,} curated blocks",
        transform=ax_a.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=COLORS["ink"],
    )

    panel_label(ax_b, "b")
    y = np.arange(3)
    order = ["edge", "organic_node", "metal_node"]
    subset = kind_counts.set_index("kind").loc[order]
    unique = subset["unique_experimental_blocks"].to_numpy()
    instances = subset["fragment_instances"].to_numpy()
    for yi, kind, lo, hi in zip(y, order, unique, instances, strict=True):
        ax_b.plot([lo, hi], [yi, yi], color="#C8C8C8", lw=2.0, zorder=1)
        ax_b.scatter(lo, yi, s=34, color=COLORS[kind], edgecolor="white", lw=0.6, zorder=3)
        ax_b.scatter(hi, yi, s=34, facecolor="white", edgecolor=COLORS[kind], lw=1.2, zorder=3)
        ax_b.text(lo, yi + 0.20, f"{lo:,}", ha="center", va="bottom", fontsize=5.7)
        ax_b.text(hi, yi + 0.20, f"{hi:,}", ha="center", va="bottom", fontsize=5.7)
    ax_b.set_xscale("log")
    ax_b.set_xlim(1_500, 350_000)
    ax_b.set_ylim(-0.45, 2.55)
    ax_b.set_yticks(y, [KIND_LABELS[kind] for kind in order])
    ax_b.set_xlabel("Count (log scale)")
    ax_b.set_title("Instances and unique types", loc="left", pad=6)
    ax_b.scatter([], [], s=28, color="#777777", label="Unique blocks")
    ax_b.scatter([], [], s=28, facecolor="white", edgecolor="#777777", label="Instances")
    ax_b.legend(loc="lower right", fontsize=5.7, handletextpad=0.35, labelspacing=0.25)
    style_axis(ax_b)

    panel_label(ax_c, "c")
    cn_colors = [
        "#EDF2F8",
        "#D5E1F0",
        "#B4CBE3",
        "#8FAFD3",
        "#6F92C1",
        "#4F74AA",
        "#35588F",
        "#243D68",
    ]
    y = np.arange(3)
    left = np.zeros(3)
    kind_order = ["edge", "organic_node", "metal_node"]
    pivot = (
        coordination.pivot(index="kind", columns="coordination_bin", values="fraction")
        .reindex(index=kind_order, columns=cn_order)
        .fillna(0.0)
    )
    for label, color in zip(cn_order, cn_colors, strict=True):
        values = pivot[label].to_numpy() * 100
        bars = ax_c.barh(y, values, left=left, color=color, edgecolor="white", lw=0.35, label=label)
        for bar, value in zip(bars, values, strict=True):
            if value >= 15:
                ax_c.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if color in cn_colors[-3:] else COLORS["ink"],
                )
        left += values
    ax_c.set_xlim(0, 100)
    ax_c.set_yticks(y, [KIND_LABELS[kind] for kind in kind_order])
    ax_c.set_xlabel("Share of unique blocks (%)")
    ax_c.set_title("Logical coordination", loc="left", pad=6)
    ax_c.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.25),
        fontsize=5.2,
        handlelength=0.9,
        columnspacing=0.65,
        handletextpad=0.25,
    )
    style_axis(ax_c)

    panel_label(ax_d, "d")
    for kind in ["edge", "organic_node", "metal_node"]:
        x, y_ccdf = ccdf(experimental_blocks.loc[experimental_blocks["kind"] == kind, "instance_count"])
        ax_d.step(x, y_ccdf, where="post", color=COLORS[kind], lw=1.6, label=KIND_LABELS[kind])
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlim(1, 50_000)
    ax_d.set_ylim(5e-5, 1.1)
    ax_d.set_xlabel("Occurrences per unique block")
    ax_d.set_ylabel("Fraction of blocks ≥ x")
    ax_d.set_title("Reuse across structures", loc="left", pad=6)
    ax_d.legend(loc="lower left", fontsize=5.7, handlelength=1.4)
    style_axis(ax_d)

    return export_figure(fig, output_dir, "fig1_building_block_library")


def make_figure_2(data: dict, output_dir: Path) -> dict:
    summary = data["roundtrip_summary"]
    cases = data["roundtrip_cases"].copy()
    metrics = summary["metrics"]

    metric_rows = [
        ("Connectivity recovered", metrics["connectivity_recovery"]),
        ("Zero atomic overlap", metrics["zero_overlap"]),
        ("Strict build accepted", metrics["strict_build_acceptance"]),
        ("Space group matched (0.5 Å)", metrics["symmetry"]["baseline_primary"]),
        ("Space group matched at all tolerances", metrics["symmetry"]["baseline_all_symprecs_per_case"]),
    ]
    metric_frame = pd.DataFrame(
        [
            {
                "metric": label,
                "count": int(record["count"]),
                "denominator": int(record["denominator"]),
                "percent": 100 * float(record["fraction"]),
            }
            for label, record in metric_rows
        ]
    )
    write_csv(metric_frame, output_dir, "fig2a_baseline_metrics.csv")

    tolerances = ["0.05", "0.1", "0.2", "0.5"]
    tolerance_frame = pd.DataFrame(
        {
            "symmetry_tolerance_angstrom": [float(value) for value in tolerances],
            "baseline_percent_intent_to_test": [
                100 * metrics["symmetry"]["baseline_by_symprec"][value]["fraction"]
                for value in tolerances
            ],
            "rotated_percent_intent_to_test": [
                100
                * metrics["symmetry"]["rotated_by_symprec_intent_to_test"][value][
                    "fraction"
                ]
                for value in tolerances
            ],
        }
    )
    write_csv(tolerance_frame, output_dir, "fig2b_symmetry_tolerance.csv")

    matrix_specs = [
        ("Connectivity", cases["baseline_connectivity_passed"].map(parse_bool)),
        ("Strict geometry", cases["baseline_strict_accepted"].map(parse_bool)),
        ("Zero overlap", (pd.to_numeric(cases["baseline_overlap_count"], errors="coerce") == 0).astype(float)),
        ("SG at 0.05 Å", cases["baseline_sg_match_0.05"].map(parse_bool)),
        ("SG at 0.5 Å", cases["baseline_sg_match_0.5"].map(parse_bool)),
        ("SG at all tolerances", cases["baseline_all_symprecs_match"].map(parse_bool)),
        (
            "Rotation robust (5/5)",
            (pd.to_numeric(cases["rotation_invariant_trials"], errors="coerce") == 5).astype(float),
        ),
    ]
    matrix = pd.DataFrame({label: values for label, values in matrix_specs})
    matrix.loc[cases["status"] != "completed", :] = np.nan
    score = matrix.fillna(-0.5).sum(axis=1)
    ordered = (
        pd.DataFrame({"row": np.arange(len(cases)), "score": score, "refcode": cases["refcode"]})
        .sort_values(["score", "refcode"], ascending=[False, True])
        ["row"]
        .to_numpy()
    )
    matrix_ordered = matrix.iloc[ordered].reset_index(drop=True)
    case_source = pd.concat(
        [
            cases.iloc[ordered][["refcode", "cohort", "status"]].reset_index(drop=True),
            matrix_ordered,
        ],
        axis=1,
    )
    case_source.insert(0, "display_order", np.arange(1, len(case_source) + 1))
    write_csv(case_source, output_dir, "fig2c_case_matrix.csv")

    complete_rotation = cases[pd.to_numeric(cases["rotation_trials_completed"], errors="coerce") == 5].copy()
    rotation_columns = {
        "Strict build": "rotated_strict_accepted",
        "Zero overlap": "rotated_zero_overlap",
        "Pose invariant": "rotation_invariant_trials",
        "Structure invariant": "structural_rotation_invariant_trials",
    }
    rotation_rows = []
    for label, column in rotation_columns.items():
        counts = pd.to_numeric(complete_rotation[column], errors="coerce").value_counts().to_dict()
        for passes in range(6):
            rotation_rows.append(
                {
                    "criterion": label,
                    "successful_trials": passes,
                    "case_count": int(counts.get(passes, 0)),
                    "eligible_cases": len(complete_rotation),
                }
            )
    rotation_frame = pd.DataFrame(rotation_rows)
    write_csv(rotation_frame, output_dir, "fig2d_rotation_robustness.csv")

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.20, 1.45, 1.20],
        height_ratios=[1.0, 1.35],
        left=0.165,
        right=0.985,
        top=0.95,
        bottom=0.105,
        hspace=0.66,
        wspace=0.56,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, :2])
    ax_d = fig.add_subplot(gs[1, 2])

    panel_label(ax_a, "a")
    values = metric_frame["percent"].to_numpy()[::-1]
    labels = np.array(
        [
            "Connectivity",
            "Zero overlap",
            "Strict acceptance",
            "SG match (0.5 Å)",
            "SG match (all tol.)",
        ]
    )[::-1]
    bars = ax_a.barh(np.arange(len(values)), values, color=COLORS["strict"], height=0.62)
    ax_a.set_yticks(np.arange(len(values)), labels)
    ax_a.set_xlim(0, 105)
    ax_a.set_xlabel("Intent-to-test success (%)")
    ax_a.set_title("Round-trip benchmark", loc="left", pad=6)
    for bar, value, count, denominator in zip(
        bars,
        values,
        metric_frame["count"].to_numpy()[::-1],
        metric_frame["denominator"].to_numpy()[::-1],
        strict=True,
    ):
        ax_a.text(
            min(value + 2.0, 101.5),
            bar.get_y() + bar.get_height() / 2,
            f"{count}/{denominator}",
            ha="left",
            va="center",
            fontsize=5.8,
        )
    style_axis(ax_a)

    panel_label(ax_b, "b", x=-0.07)
    x = tolerance_frame["symmetry_tolerance_angstrom"].to_numpy()
    ax_b.plot(
        x,
        tolerance_frame["baseline_percent_intent_to_test"],
        marker="o",
        ms=4,
        lw=1.6,
        color=COLORS["strict"],
        label="Baseline (n = 100)",
    )
    ax_b.plot(
        x,
        tolerance_frame["rotated_percent_intent_to_test"],
        marker="s",
        ms=3.6,
        lw=1.4,
        color="#6F92C1",
        label="Rotated trials (n = 500 intended)",
    )
    ax_b.set_xticks(x, [f"{value:g}" for value in x])
    ax_b.set_ylim(56, 70)
    ax_b.set_yticks([58, 62, 66, 70])
    ax_b.set_xlabel("Symmetry tolerance (Å)")
    ax_b.set_ylabel("Space-group recovery (%)")
    ax_b.set_title("Recovery increases with tolerance", loc="left", pad=6)
    ax_b.legend(loc="lower right")
    style_axis(ax_b)

    panel_label(ax_c, "c", x=-0.055)
    plot_matrix = matrix_ordered.T.to_numpy(dtype=float)
    encoded = np.where(np.isnan(plot_matrix), -1, plot_matrix)
    cmap = ListedColormap([COLORS["missing"], COLORS["fail"], COLORS["pass"]])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    ax_c.imshow(encoded, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax_c.set_yticks(np.arange(len(matrix_ordered.columns)), matrix_ordered.columns)
    ax_c.set_xticks([0, 24, 49, 74, 99], ["1", "25", "50", "75", "100"])
    ax_c.set_xlabel("Structures ordered by number of passed criteria")
    ax_c.set_title("Per-structure failure pattern", loc="left", pad=6)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    ax_c.tick_params(length=0)
    handles = [
        mpl.patches.Patch(facecolor=COLORS["pass"], label="Pass"),
        mpl.patches.Patch(facecolor=COLORS["fail"], label="Fail"),
        mpl.patches.Patch(facecolor=COLORS["missing"], label="Not completed"),
    ]
    ax_c.legend(handles=handles, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.18))

    panel_label(ax_d, "d", x=-0.16)
    criterion_order = list(rotation_columns)
    pass_colors = ["#C95E58", "#E9A6A1", "#E8D8D2", "#C5D4E8", "#789AC5", COLORS["strict"]]
    left = np.zeros(len(criterion_order))
    for passes, color in zip(range(6), pass_colors, strict=True):
        vals = (
            rotation_frame[rotation_frame["successful_trials"] == passes]
            .set_index("criterion")
            .loc[criterion_order, "case_count"]
            .to_numpy()
        )
        ax_d.barh(
            np.arange(len(criterion_order)),
            vals,
            left=left,
            height=0.62,
            color=color,
            edgecolor="white",
            lw=0.3,
            label=str(passes),
        )
        left += vals
    ax_d.set_yticks(np.arange(len(criterion_order)), criterion_order)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, len(complete_rotation))
    ax_d.set_xlabel("Cases (5 completed trials)")
    ax_d.set_title("Rotation robustness", loc="left", pad=6)
    ax_d.legend(
        title="Successful trials",
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.26),
        fontsize=5.4,
        title_fontsize=5.7,
        handlelength=0.9,
        columnspacing=0.6,
    )
    style_axis(ax_d)

    return export_figure(fig, output_dir, "fig2_roundtrip_validation")


def make_figure_3(data: dict, output_dir: Path) -> dict:
    experimental = data["experimental"]
    strict = data["strict_manifest"]
    fallback = data["fallback_manifest"]
    strict_summary = data["strict_summary"]
    fallback_summary = data["fallback_summary"]

    regime_summary = pd.DataFrame(
        [
            {
                "regime": "Experimental corpus",
                "structures": len(experimental),
                "p1_count": int((experimental["space_group_number"] == 1).sum()),
                "p1_fraction": float((experimental["space_group_number"] == 1).mean()),
                "space_group_retained_count": np.nan,
                "space_group_retained_fraction": np.nan,
                "productive_topologies": np.nan,
            },
            {
                "regime": "Independent placement",
                "structures": len(fallback),
                "p1_count": int((fallback["cif_space_group_number"] == 1).sum()),
                "p1_fraction": float((fallback["cif_space_group_number"] == 1).mean()),
                "space_group_retained_count": int(fallback["space_group_retained"].sum()),
                "space_group_retained_fraction": float(fallback["space_group_retained"].mean()),
                "productive_topologies": int(fallback_summary["productive_topologies"]),
            },
            {
                "regime": "Symmetry constrained",
                "structures": len(strict),
                "p1_count": int((strict["realized_space_group_number"] == 1).sum()),
                "p1_fraction": float((strict["realized_space_group_number"] == 1).mean()),
                "space_group_retained_count": int(strict["space_group_retained"].sum()),
                "space_group_retained_fraction": float(strict["space_group_retained"].mean()),
                "productive_topologies": int(strict_summary["productive_topologies"]),
            },
        ]
    )
    write_csv(regime_summary, output_dir, "fig3bc_symmetry_summary.csv")

    rank_rows = []
    for regime, frame in [("Independent placement", fallback), ("Symmetry constrained", strict)]:
        counts = frame["topology"].value_counts().sort_values(ascending=False)
        for rank, (topology, count) in enumerate(counts.items(), start=1):
            rank_rows.append(
                {"regime": regime, "rank": rank, "topology": topology, "structures": int(count)}
            )
    rank_frame = pd.DataFrame(rank_rows)
    write_csv(rank_frame, output_dir, "fig3d_topology_rank_abundance.csv")

    fig = plt.figure(figsize=(7.2, 5.65))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.05, 1.40],
        width_ratios=[0.9, 0.9, 1.5],
        left=0.075,
        right=0.985,
        top=0.955,
        bottom=0.105,
        hspace=0.48,
        wspace=0.48,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    panel_label(ax_a, "a", x=-0.035, y=1.02)
    ax_a.set_axis_off()
    lane_y = [0.67, 0.24]
    lane_specs = [
        (
            "Independent placement",
            ["Local port fit", "Place each orbit", "P1 fallback\n(n = 10,000)"],
            COLORS["fallback"],
        ),
        (
            "Symmetry constrained",
            ["Site stabilizer", "Orbit covariance", "Retain parent symmetry\n(n = 10,000)"],
            COLORS["strict"],
        ),
    ]
    for y, (label, steps, color) in zip(lane_y, lane_specs, strict=True):
        ax_a.text(
            0.015,
            y + 0.09,
            label,
            transform=ax_a.transAxes,
            ha="left",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=color,
        )
        xs = [0.25, 0.50, 0.75]
        for x, step in zip(xs, steps, strict=True):
            patch = FancyBboxPatch(
                (x, y),
                0.18,
                0.20,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=mpl.colors.to_rgba(color, 0.13),
                edgecolor=color,
                linewidth=1.0,
                transform=ax_a.transAxes,
            )
            ax_a.add_patch(patch)
            ax_a.text(
                x + 0.09,
                y + 0.10,
                step,
                transform=ax_a.transAxes,
                ha="center",
                va="center",
                fontsize=6.3,
                color=COLORS["ink"],
            )
        draw_arrow(ax_a, 0.43, 0.50, y + 0.10)
        draw_arrow(ax_a, 0.68, 0.75, y + 0.10)
    panel_label(ax_b, "b")
    b_order = ["Experimental corpus", "Independent placement", "Symmetry constrained"]
    b_data = regime_summary.set_index("regime").loc[b_order]
    b_colors = [COLORS["experimental"], COLORS["fallback"], COLORS["strict"]]
    x = np.arange(3)
    bars = ax_b.bar(x, 100 * b_data["p1_fraction"], color=b_colors, width=0.67)
    ax_b.set_xticks(
        x,
        ["Experimental", "Independent", "Constrained"],
        rotation=24,
        ha="right",
    )
    ax_b.set_ylim(0, 108)
    ax_b.set_ylabel("Structures assigned to P1 (%)")
    ax_b.set_title("P1 assignment", loc="left", pad=6)
    for bar, row in zip(bars, b_data.itertuples(), strict=True):
        value = 100 * row.p1_fraction
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.2,
            f"{value:.1f}%\n({int(row.p1_count):,}/{int(row.structures):,})",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    style_axis(ax_b)

    panel_label(ax_c, "c")
    c_order = ["Independent placement", "Symmetry constrained"]
    c_data = regime_summary.set_index("regime").loc[c_order]
    c_colors = [COLORS["fallback"], COLORS["strict"]]
    bars = ax_c.bar(np.arange(2), 100 * c_data["space_group_retained_fraction"], color=c_colors, width=0.66)
    ax_c.set_xticks(
        np.arange(2), ["Independent", "Constrained"], rotation=20, ha="right"
    )
    ax_c.set_ylim(0, 108)
    ax_c.set_ylabel("Parent space group retained (%)")
    ax_c.set_title("Parent-group retention", loc="left", pad=6)
    for bar, row in zip(bars, c_data.itertuples(), strict=True):
        value = 100 * row.space_group_retained_fraction
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.2,
            f"{value:.2f}%\n({int(row.space_group_retained_count):,}/{int(row.structures):,})",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    style_axis(ax_c)

    panel_label(ax_d, "d", x=-0.13)
    for regime, color in [
        ("Independent placement", COLORS["fallback"]),
        ("Symmetry constrained", COLORS["strict"]),
    ]:
        subset = rank_frame[rank_frame["regime"] == regime]
        ax_d.plot(
            subset["rank"],
            subset["structures"],
            lw=1.6,
            color=color,
            label=f"{regime} (n = {len(subset)} topologies)",
        )
        ax_d.fill_between(
            subset["rank"].to_numpy(),
            0,
            subset["structures"].to_numpy(),
            color=color,
            alpha=0.08,
        )
    ax_d.set_xlim(1, max(rank_frame["rank"]))
    ax_d.set_ylim(0, 210)
    ax_d.set_xlabel("Topology abundance rank")
    ax_d.set_ylabel("Generated structures")
    ax_d.set_title("Topology coverage", loc="left", pad=6)
    ax_d.legend(loc="upper right", fontsize=5.6)
    style_axis(ax_d)

    return export_figure(fig, output_dir, "fig3_symmetry_constrained_generation")


def make_supplementary_figure_1(data: dict, output_dir: Path) -> dict:
    strict = data["strict_manifest"]
    fallback = data["fallback_manifest"]
    strict_summary = data["strict_summary"]
    fallback_summary = data["fallback_summary"]

    geometry = pd.concat(
        [
            strict[
                ["sample_id", "topology", "num_atoms", "max_rmsd", "minimum_atom_distance_angstrom"]
            ].assign(regime="Symmetry constrained"),
            fallback[
                ["sample_id", "topology", "num_atoms", "max_rmsd", "minimum_atom_distance_angstrom"]
            ].assign(regime="Independent placement"),
        ],
        ignore_index=True,
    )
    write_csv(geometry, output_dir, "figs1ab_generation_geometry.csv")

    efficiency_rows = []
    for regime, summary in [
        ("Symmetry constrained", strict_summary),
        ("Independent placement", fallback_summary),
    ]:
        names = sorted(
            set(summary["attempted_count_by_topology"])
            | set(summary["final_count_by_topology"])
        )
        for topology in names:
            attempted = int(summary["attempted_count_by_topology"].get(topology, 0))
            accepted = int(summary["final_count_by_topology"].get(topology, 0))
            efficiency_rows.append(
                {
                    "regime": regime,
                    "topology": topology,
                    "attempted": attempted,
                    "accepted": accepted,
                    "acceptance_fraction": accepted / attempted if attempted else np.nan,
                }
            )
    efficiency = pd.DataFrame(efficiency_rows)
    write_csv(efficiency, output_dir, "figs1c_topology_efficiency.csv")

    failure_categories = {
        "Geometry / quality": ["ValueError"],
        "Symmetry compatibility": ["SymmetryCompatibilityError"],
        "Implementation indexing": ["IndexError", "KeyError"],
        "Memory": ["MemoryError"],
    }
    failure_rows = []
    for regime, summary in [
        ("Symmetry constrained", strict_summary),
        ("Independent placement", fallback_summary),
    ]:
        for category, error_types in failure_categories.items():
            failure_rows.append(
                {
                    "regime": regime,
                    "failure_class": category,
                    "count": int(sum(summary["failure_types"].get(key, 0) for key in error_types)),
                    "error_types": ";".join(error_types),
                }
            )
    failures = pd.DataFrame(failure_rows)
    write_csv(failures, output_dir, "figs1d_failure_classes.csv")

    fig = plt.figure(figsize=(7.2, 5.85))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.975,
        top=0.95,
        bottom=0.105,
        hspace=0.50,
        wspace=0.38,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], "abcd", strict=True):
        panel_label(ax, label)

    for regime, frame, color in [
        ("Independent placement", fallback, COLORS["fallback"]),
        ("Symmetry constrained", strict, COLORS["strict"]),
    ]:
        x, y = empirical_cdf(frame["max_rmsd"])
        ax_a.plot(x, 100 * y, color=color, lw=1.6, label=regime)
    ax_a.set_xlim(0, 0.61)
    ax_a.set_ylim(0, 100)
    ax_a.set_xlabel("Maximum port-fit RMSD (Å)")
    ax_a.set_ylabel("Structures ≤ x (%)")
    ax_a.set_title("Port-fit geometry", loc="left", pad=6)
    ax_a.legend(loc="lower right")
    style_axis(ax_a)

    for regime, frame, color in [
        ("Independent placement", fallback, COLORS["fallback"]),
        ("Symmetry constrained", strict, COLORS["strict"]),
    ]:
        x, y = empirical_cdf(frame["minimum_atom_distance_angstrom"])
        ax_b.plot(x, 100 * y, color=color, lw=1.6, label=regime)
    ax_b.axvline(0.5, color=COLORS["neutral"], ls="--", lw=0.9)
    ax_b.set_xlim(0.48, 1.6)
    ax_b.set_ylim(0, 100)
    ax_b.set_xlabel("Minimum interatomic distance (Å)")
    ax_b.set_ylabel("Structures ≤ x (%)")
    ax_b.set_title("Atomic-separation quality", loc="left", pad=6)
    ax_b.legend(loc="lower right")
    style_axis(ax_b)

    for regime, color, marker in [
        ("Independent placement", COLORS["fallback"], "o"),
        ("Symmetry constrained", COLORS["strict"], "s"),
    ]:
        subset = efficiency[(efficiency["regime"] == regime) & (efficiency["attempted"] > 0)]
        ax_c.scatter(
            subset["attempted"],
            subset["accepted"],
            s=13,
            color=color,
            marker=marker,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.35,
            label=regime,
        )
    ax_c.set_xscale("log")
    ax_c.set_xlabel("Attempted candidates per topology")
    ax_c.set_ylabel("Accepted structures per topology")
    ax_c.set_title("Topology-specific generation yield", loc="left", pad=6)
    ax_c.legend(loc="upper left")
    style_axis(ax_c)

    categories = list(failure_categories)
    y = np.arange(len(categories))
    width = 0.34
    for offset, regime, color in [
        (-width / 2, "Independent placement", COLORS["fallback"]),
        (width / 2, "Symmetry constrained", COLORS["strict"]),
    ]:
        vals = (
            failures[failures["regime"] == regime]
            .set_index("failure_class")
            .loc[categories, "count"]
            .to_numpy()
        )
        bars = ax_d.barh(y + offset, vals, height=width, color=color, label=regime)
        for bar, value in zip(bars, vals, strict=True):
            if value > 0:
                ax_d.text(
                    value,
                    bar.get_y() + bar.get_height() / 2,
                    f" {value:,}",
                    ha="left",
                    va="center",
                    fontsize=5.5,
                )
    ax_d.set_xscale("symlog", linthresh=10)
    ax_d.set_yticks(y, categories)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Recorded failed attempts (symlog)")
    ax_d.set_title("Failure diagnostics", loc="left", pad=6)
    ax_d.legend(loc="lower right")
    style_axis(ax_d)

    return export_figure(fig, output_dir, "figs1_generation_diagnostics")


def write_legends(data: dict, output_dir: Path) -> Path:
    summary = data["library_summary"]
    round_metrics = data["roundtrip_summary"]["metrics"]
    experimental = data["experimental"]
    strict = data["strict_manifest"]
    fallback = data["fallback_manifest"]
    text = f"""# Draft figure legends

## Fig. 1 | Experimental structures yield a reusable MOF building-block library

**a,** Audited flow from {summary['records']:,} quality-controlled experimental MOFs to {summary['status_counts']['deconstructed']:,} successful decompositions, {sum(summary['fragment_instance_counts'].values()):,} fragment instances and {sum(summary['unique_building_block_counts'].values()):,} unique data-derived building blocks. The final library contains {summary['current_building_block_file_count']:,} blocks after inclusion of curated external blocks. **b,** Fragment-instance counts and unique data-derived types for edges, organic nodes and metal nodes. **c,** Logical coordination-number composition of each block class. **d,** Complementary cumulative distribution of the number of fragment instances represented by each unique block. Counts are exhaustive for the processed corpus; no sampling error bars are shown. Source data are provided as Source Data files.

## Fig. 2 | Connectivity is robust while geometry and symmetry limit round-trip reconstruction

**a,** Intent-to-test success rates across 100 experimental MOF reconstruction cases. **b,** Baseline and rotated-trial space-group recovery across symmetry tolerances of 0.05–0.5 Å. Rotated-trial percentages use 500 intended trials as the denominator. **c,** Per-structure pass/fail matrix after ordering cases by the number of satisfied criteria. Grey denotes the single case that did not complete. **d,** Distribution of successful outcomes among cases with all five rotation trials completed. The benchmark contains {round_metrics['completed_cases']} completed baselines and {round_metrics['rotation_trial_accounting']['completed']} completed rotation trials. Source data are provided as Source Data files.

## Fig. 3 | Symmetry constraints prevent collapse to P1 during MOF generation

**a,** Conceptual comparison of independent building-block placement with crystallographic-site and orbit-covariance constraints. **b,** Fraction of structures assigned to P1 in the experimental corpus (n = {len(experimental):,}), independent-placement dataset (n = {len(fallback):,}) and symmetry-constrained dataset (n = {len(strict):,}). **c,** Fraction of generated structures retaining the parent topology space-group number. Symbols were canonicalized to International space-group numbers before comparison. **d,** Ranked topology abundance in the two generated datasets. The independent-placement and symmetry-constrained datasets contain {fallback['topology'].nunique()} and {strict['topology'].nunique()} productive topologies, respectively. These regimes differ in both symmetry handling and linker conformer filtering and therefore should not be interpreted as a single-factor ablation. Source data are provided as Source Data files.

## Supplementary Fig. 1 | Geometry and throughput diagnostics of generation regimes

**a,** Empirical cumulative distributions of maximum port-fit RMSD. **b,** Empirical cumulative distributions of minimum interatomic distance; the dashed line marks the 0.5 Å generation threshold. **c,** Attempted and accepted candidates for each productive topology. **d,** Recorded failure classes during generation. Failure counts depend on candidate scheduling and software configuration and are presented as workflow diagnostics rather than direct efficiency estimates. Source data are provided as Source Data files.
"""
    path = output_dir / "figure_legends.md"
    path.write_text(text, encoding="utf-8")
    return path


def validate_outputs(output_dir: Path, exports: dict[str, dict[str, str]], data: dict) -> dict:
    checks: dict[str, object] = {
        "backend": "Python/matplotlib",
        "all_files_nonempty": True,
        "svg_contains_editable_text": True,
        "figures": {},
        "space_group_parse_failures": data["space_group_parse_failures"],
        "notes": [
            "All quantitative panels use exhaustive descriptive counts; no inferential statistics are implied.",
            "The 500,000-structure run was excluded because its current status is incomplete/failed.",
            "The two generation regimes are not a single-factor ablation because their linker conformer filters differ.",
        ],
    }
    for figure, mapping in exports.items():
        figure_checks = {}
        for fmt, path_text in mapping.items():
            path = Path(path_text)
            exists = path.is_file() and path.stat().st_size > 0
            figure_checks[f"{fmt}_nonempty"] = exists
            checks["all_files_nonempty"] = bool(checks["all_files_nonempty"] and exists)
            if fmt == "svg" and exists:
                svg_text = path.read_text(encoding="utf-8")
                editable = "<text" in svg_text
                figure_checks["svg_editable_text"] = editable
                checks["svg_contains_editable_text"] = bool(
                    checks["svg_contains_editable_text"] and editable
                )
        checks["figures"][figure] = figure_checks
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root containing experiment output directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "publication_figures",
        help="Destination for figures, source data and QA files.",
    )
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    data = prepare_data(project_root, output_dir)
    exports = {
        "figure_1": make_figure_1(data, output_dir),
        "figure_2": make_figure_2(data, output_dir),
        "figure_3": make_figure_3(data, output_dir),
        "supplementary_figure_1": make_supplementary_figure_1(data, output_dir),
    }
    legend_path = write_legends(data, output_dir)
    qa = validate_outputs(output_dir, exports, data)
    qa["figure_legends"] = str(legend_path)
    qa_path = output_dir / "qa_report.json"
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "exports": exports, "qa": qa}, indent=2))


if __name__ == "__main__":
    main()
