"""Command line interface for MOFid/PORMAKE decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decomposer import SingleNodePorDecomposer


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="mofbb",
        description=(
            "Decompose a CIF into MOFid SingleNode-style node/linker "
            "fragments with PORMAKE X caps."
        ),
    )
    parser.add_argument("cif", type=Path, help="Input CIF file readable by ASE.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("decomposition_output"),
        help="Directory where fragment, topology, and summary files will be written.",
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
        help="Print the full decomposition summary JSON to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    decomposer = SingleNodePorDecomposer(
        args.cif,
        x_symbol=args.x_symbol,
        bond_scale=args.bond_scale,
        cleanup=not args.no_cleanup,
    )
    result = decomposer.decompose()
    summary = result.write(args.output_dir)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Wrote decomposition outputs to {args.output_dir}")
        print(
            "Fragments: "
            f"{summary['num_nodes']} nodes, "
            f"{summary['num_linkers']} linkers, "
            f"{summary['num_connecting_bonds']} connecting bonds"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
