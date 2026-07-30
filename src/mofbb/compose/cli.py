"""Command-line interface for MOF composition."""

from __future__ import annotations

import argparse
from pathlib import Path

from .builder import Builder
from .building_block import BuildingBlock
from .reconstructor import Reconstructor
from .topology import Topology

NodeAssignment = tuple[int, Path]
EdgeAssignment = tuple[tuple[int, int], Path]


def _parse_node_assignment(value: str) -> NodeAssignment:
    try:
        raw_type, raw_path = value.split("=", maxsplit=1)
        node_type = int(raw_type)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "node building blocks must use TYPE=PATH"
        ) from exc
    return node_type, Path(raw_path)


def _parse_edge_assignment(value: str) -> EdgeAssignment:
    try:
        raw_types, raw_path = value.split("=", maxsplit=1)
        left, right = (int(item) for item in raw_types.split(",", maxsplit=1))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "edge building blocks must use LEFT_TYPE,RIGHT_TYPE=PATH"
        ) from exc
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
        "--accuracy",
        type=int,
        default=6,
        help="Orientation-search accuracy. Default: 6.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Port permutations tried after a chemistry failure. Default: 4.",
    )
    parser.add_argument(
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
        "--no-wrap",
        action="store_true",
        help="Do not wrap composed atoms into the unit cell.",
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


def _validate_node_assignments(
    topology: Topology,
    node_bbs: dict[object, BuildingBlock],
) -> None:
    required = {int(value) for value in topology.unique_node_types}
    missing = sorted(required - set(node_bbs))
    if missing:
        raise ValueError(f"Missing node building blocks for types {missing}.")

    for node_index in topology.node_indices:
        node_type = int(topology.node_types[node_index])
        expected = int(topology.cn[node_index])
        actual = node_bbs[node_type].n_connection_points
        if actual != expected:
            raise ValueError(
                f"Node type {node_type} requires {expected} connection points, "
                f"but {node_bbs[node_type].name!r} has {actual}."
            )


def main(argv: list[str] | None = None) -> int:
    """Compose a framework and write it as CIF."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.source.is_dir():
            if args.node_bb or args.edge_bb:
                raise ValueError(
                    "Building-block assignments cannot be used with a "
                    "decomposition directory."
                )
            framework = Reconstructor(args.source).reconstruct()
        else:
            if not args.node_bb:
                raise ValueError(
                    "At least one --node-bb assignment is required for a CGD file."
                )
            topology = Topology(args.source)
            node_bbs = _load_assignments(args.node_bb)
            edge_bbs = _load_assignments(args.edge_bb)
            _validate_node_assignments(topology, node_bbs)
            framework = Builder().build_by_type(
                topology,
                node_bbs,
                edge_bbs,
                accuracy=args.accuracy,
                top_k=args.top_k,
                tolerance=args.x_mid_tolerance,
                scale=args.fused_bond_scale,
                overlap_scale=args.atomic_overlap_scale,
                wrap=not args.no_wrap,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        framework.write_cif(args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    output = args.output.with_suffix(".cif")
    if not output.exists():
        parser.error(f"Composition did not produce {output}.")
    print(f"Wrote composed framework to {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
