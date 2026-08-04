"""Command-line interface for node-edge decomposition."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .decomposer import NodeEdgeDecomposer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mofbb-decompose",
        description=(
            "Decompose a CIF into X-capped metal nodes, organic nodes, and edges."
        ),
    )
    parser.add_argument("cif", type=Path)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("decomposition_output"),
    )
    parser.add_argument("--x-symbol", default="X")
    parser.add_argument("--bond-scale", type=float, default=1.2)
    parser.add_argument(
        "--solvent-mode",
        choices=("none", "free", "all"),
        default="none",
    )
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--write-topology", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = NodeEdgeDecomposer(
        args.cif,
        x_symbol=args.x_symbol,
        bond_scale=args.bond_scale,
        cleanup=not args.no_cleanup,
        solvent_mode=args.solvent_mode,
    ).decompose()
    summary = result.write(
        args.output_dir,
        write_topology=args.write_topology,
    )
    shutil.copyfile(args.cif, args.output_dir / args.cif.name)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Wrote node-edge decomposition to {args.output_dir}")
        print(
            "Fragments: "
            f"{summary['num_metal_nodes']} metal nodes, "
            f"{summary['num_organic_nodes']} organic nodes, "
            f"{summary['num_edges']} edges"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
