"""Command-line interface for MOF composition."""

from __future__ import annotations

import argparse
from pathlib import Path

from mofbb.compose.builder import Builder
from mofbb.compose.building_block import BuildingBlock
from mofbb.compose.reconstructor import TopoInput, Reconstructor
from mofbb.compose.topology import Topology

NodeAssignment = tuple[int, Path]
EdgeAssignment = tuple[tuple[int, int], Path]


def _parse_node_assignment(value: str) -> NodeAssignment:
    raw_type, raw_path = value.split("=", maxsplit=1)
    return int(raw_type), Path(raw_path)


def _parse_edge_assignment(value: str) -> EdgeAssignment:
    raw_types, raw_path = value.split("=", maxsplit=1)
    left, right = (int(item) for item in raw_types.split(",", maxsplit=1))
    return tuple(sorted((left, right))), Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the composition command-line parser."""

    parser = argparse.ArgumentParser(
        prog="mofbb-compose",
        description="Compose a periodic framework from a CGD topology and blocks.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Decomposition output directory or input CGD topology file.",
    )
    parser.add_argument(
        "--node-bb",
        action="append",
        default=[],
        type=_parse_node_assignment,
        metavar="TYPE=PATH",
        help="Building block for a topology node type; repeat for each type.",
    )
    parser.add_argument(
        "--edge-bb",
        action="append",
        default=[],
        type=_parse_edge_assignment,
        metavar="LEFT_TYPE,RIGHT_TYPE=PATH",
        help="Optional building block for an edge type; may be repeated.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("composed.cif"),
        help="Output CIF path. Default: composed.cif.",
    )
    parser.add_argument(
        "--n-angles",
        "--accuracy",
        type=int,
        default=6,
        help="Samples per Euler-angle axis (minimum 4 when sampling). Default: 6.",
    )
    parser.add_argument(
        "--port-tol",
        "--x-mid-tolerance",
        type=float,
        default=0.3,
        metavar="ANGSTROM",
        help=(
            "Maximum minimum-image distance between paired X connection "
            "atoms before rejecting the assembly. Default: 0.3 Å."
        ),
    )
    parser.add_argument(
        "--bond-scale",
        "--fused-bond-scale",
        type=float,
        default=1.2,
        help=(
            "Maximum fused-bond distance as a multiplier of summed ASE "
            "covalent radii. Default: 1.2."
        ),
    )
    parser.add_argument(
        "--atomic-overlap-scale",
        type=float,
        default=1.0,
        help=(
            "Minimum allowed atom-pair distance as a multiplier of the "
            "smaller ASE covalent radius. Use 0 to disable. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--allow-low-dimensional",
        action="store_true",
        help="Explicitly allow layers, chains, or finite assemblies. Default: require 3D.",
    )
    parser.add_argument(
        "--no-wrap",
        action="store_true",
        help="Do not wrap composed atoms into the unit cell.",
    )
    parser.add_argument(
        "--abstract-topology",
        action="store_true",
        help=(
            "For a decomposition directory, rebuild from topology.cgd and "
            "grouped BBs without reconstruction_map.json."
        ),
    )
    return parser


def _load_assignments(
    assignments: list[tuple[object, Path]],
) -> dict[object, BuildingBlock]:
    loaded: dict[object, BuildingBlock] = {}
    for key, path in assignments:
        if key in loaded:
            raise ValueError(f"Duplicate building-block assignment for {key!r}.")
        loaded[key] = BuildingBlock(path)
    return loaded


def main(argv: list[str] | None = None) -> int:
    """Compose a framework and write it as CIF."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.source.is_dir():
        if args.node_bb or args.edge_bb:
            raise ValueError(
                "Building-block assignments cannot be used with a "
                "decomposition directory."
            )
        if args.abstract_topology:
            abstract = TopoInput.from_dir(args.source)
            framework = Builder().build(
                abstract.topology,
                abstract.copy_bbs(),
                n_angles=args.n_angles,
                port_tol=args.port_tol,
                bond_scale=args.bond_scale,
                overlap_scale=args.atomic_overlap_scale,
                wrap=not args.no_wrap,
                require_3d=not args.allow_low_dimensional,
            )
        else:
            framework = Reconstructor(args.source).reconstruct()
    else:
        if args.abstract_topology:
            raise ValueError(
                "--abstract-topology requires a decomposition directory."
            )
        if not args.node_bb:
            raise ValueError(
                "At least one --node-bb assignment is required for a CGD file."
            )
        topology = Topology(args.source)
        node_bbs = _load_assignments(args.node_bb)
        edge_bbs = _load_assignments(args.edge_bb)
        framework = Builder().build_by_type(
            topology,
            node_bbs,
            edge_bbs,
            n_angles=args.n_angles,
            port_tol=args.port_tol,
            bond_scale=args.bond_scale,
            overlap_scale=args.atomic_overlap_scale,
            wrap=not args.no_wrap,
            require_3d=not args.allow_low_dimensional,
        )
    output = framework.write_cif(args.output)
    print(f"Wrote composed framework to {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
