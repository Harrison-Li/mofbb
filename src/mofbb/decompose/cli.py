"""Command-line interface for MOF decomposition."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from .decomposer import SingleNodeDecomposer
from .edge_decomposer import NodeEdgeDecomposer, NodeEdgeDecompositionResult
from .results import DecompositionResult

CliDecompositionResult = DecompositionResult | NodeEdgeDecompositionResult

_NODE_GROUP_KINDS = frozenset({"node", "metal_node", "organic_node"})
_EDGE_GROUP_KINDS = frozenset({"linker", "edge"})


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        # Previous decomposition command, retained for reference:
        # prog="mofbb",
        prog="mofbb-decompose",
        description=(
            "Decompose a CIF into deduplicated edge/node building blocks "
            "with PORMAKE X caps."
        ),
    )
    parser.add_argument("cif", type=Path, help="Input CIF file readable by ASE.")
    parser.add_argument(
        "--decomposition",
        choices=("single-node", "node-edge"),
        # Previous CLI default, retained for reference:
        # default="single-node",
        default="node-edge",
        help=(
            "Decomposition workflow to run. node-edge is the default; "
            "single-node keeps the existing "
            "node/linker output; node-edge writes metal nodes, organic nodes, "
            "and edges."
        ),
    )
    parser.add_argument(
        "--solvent-mode",
        choices=("none", "free", "all"),
        default="none",
        help=(
            "Solvent removal before decomposition: none keeps legacy cleanup, "
            "free uses the FSR structure, and all uses the ASR structure."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("decomposition_output"),
        help=(
            "Directory where deduplicated bbs/{edge,nodes} and the original "
            "MOF CIF and topology will be written."
        ),
    )
    parser.add_argument(
        "--x-symbol",
        default="X",
        help="Dummy atom symbol used to cap cut bonds. Default: X.",
    )
    parser.add_argument(
        "--bond-scale",
        type=float,
        default=1.2,
        help="Multiplier applied to ASE natural covalent cutoffs. Default: 1.2.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not reduce the input structure to its largest connected component.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the reduced published-output summary JSON to stdout.",
    )
    return parser


def _reduced_summary(
    summary: dict[str, object],
    groups: list[dict[str, object]],
    original_cif_path: Path,
    topology_path: Path,
    solvent_mode: str,
) -> dict[str, object]:
    count_keys = (
        "num_nodes",
        "num_linkers",
        "num_metal_nodes",
        "num_organic_nodes",
        "num_edges",
        "num_connecting_bonds",
    )
    reduced = {
        key: summary[key]
        for key in ("name", "decomposition", *count_keys)
        if key in summary
    }
    reduced.update(
        {
            "solvent_mode": solvent_mode,
            "groups": groups,
            "original_cif": str(original_cif_path),
            "topology": str(topology_path),
            "output_layout": {
                "building_blocks": "bbs/{edge,nodes}",
                "original_mof": "original_mof_cif",
                "topology": "topology",
            },
        }
    )
    return reduced


def write_cli_results(
    result: CliDecompositionResult,
    source_cif: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Publish only grouped BBs and a byte-identical copy of the input CIF."""

    source_cif = Path(source_cif)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".mofbb-decompose-",
        dir=output_dir.parent,
    ) as temporary_directory:
        staging_root = Path(temporary_directory)
        full_output = staging_root / "full"
        published_output = staging_root / "published"

        # Previous CLI writer, retained for reference:
        # summary = result.write(output_dir)
        summary = result.write(full_output)

        edge_dir = published_output / "bbs" / "edge"
        node_dir = published_output / "bbs" / "nodes"
        edge_dir.mkdir(parents=True)
        node_dir.mkdir(parents=True)

        published_groups = []
        raw_groups = summary.get("groups", [])
        if not isinstance(raw_groups, list):
            raise TypeError("Decomposition summary 'groups' must be a list.")
        for raw_group in raw_groups:
            if not isinstance(raw_group, dict):
                raise TypeError("Decomposition group metadata must be an object.")
            kind = str(raw_group["kind"])
            if kind in _NODE_GROUP_KINDS:
                destination_dir = node_dir
                published_role = "nodes"
            elif kind in _EDGE_GROUP_KINDS:
                destination_dir = edge_dir
                published_role = "edge"
            else:
                raise ValueError(f"Unsupported decomposition BB kind: {kind!r}.")

            relative_source = Path(str(raw_group["path"]))
            source_path = full_output / relative_source
            destination_path = destination_dir / source_path.name
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if destination_path.exists():
                raise FileExistsError(
                    f"Duplicate published BB filename: {destination_path.name}"
                )
            shutil.copyfile(source_path, destination_path)

            group = dict(raw_group)
            group["path"] = str(
                Path("bbs") / published_role / destination_path.name
            )
            published_groups.append(group)

        original_dir = published_output / "original_mof_cif"
        original_dir.mkdir()
        original_cif_path = original_dir / source_cif.name
        shutil.copyfile(source_cif, original_cif_path)

        topology_source = full_output / "topology.cgd"
        topology_dir = published_output / "topology"
        topology_dir.mkdir()
        topology_path = topology_dir / f"{source_cif.stem}.cgd"
        if not topology_source.is_file():
            raise FileNotFoundError(topology_source)
        shutil.copyfile(topology_source, topology_path)

        reduced_summary = _reduced_summary(
            summary,
            published_groups,
            Path("original_mof_cif") / source_cif.name,
            Path("topology") / topology_path.name,
            result.solvent_mode,
        )
        os.replace(published_output, output_dir)

    return reduced_summary


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    decomposer_class = (
        NodeEdgeDecomposer
        if args.decomposition == "node-edge"
        else SingleNodeDecomposer
    )
    decomposer = decomposer_class(
        args.cif,
        x_symbol=args.x_symbol,
        bond_scale=args.bond_scale,
        cleanup=not args.no_cleanup,
        solvent_mode=args.solvent_mode,
    )
    result = decomposer.decompose()
    # Previous full-output call, retained for reference:
    # summary = result.write(args.output_dir)
    summary = write_cli_results(result, args.cif, args.output_dir)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Wrote deduplicated decomposition BBs to {args.output_dir}")
        if args.decomposition == "node-edge":
            print(
                "Fragments: "
                f"{summary['num_metal_nodes']} metal nodes, "
                f"{summary['num_organic_nodes']} organic nodes, "
                f"{summary['num_edges']} edges, "
                f"{summary['num_connecting_bonds']} connecting bonds"
            )
        else:
            print(
                "Fragments: "
                f"{summary['num_nodes']} nodes, "
                f"{summary['num_linkers']} linkers, "
                f"{summary['num_connecting_bonds']} connecting bonds"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
