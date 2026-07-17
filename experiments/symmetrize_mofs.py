#!/usr/bin/env python3
"""Analyze the symmetry of generated MOF CIFs without modifying them."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def analyze_symmetry(collection: Path) -> int:
    input_dir = collection / "mofs"
    input_cifs = sorted(input_dir.glob("*.cif"))
    if not input_cifs:
        raise FileNotFoundError(f"No CIFs found under {input_dir}.")
    symmetry_counts: Counter[tuple[int, str]] = Counter()

    for input_cif in input_cifs:
        structure = Structure.from_file(input_cif, primitive=False)
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=0.3,
            angle_tolerance=5,
        )
        symbol = analyzer.get_space_group_symbol()
        number = analyzer.get_space_group_number()
        symmetry_counts[number, symbol] += 1
        print(f"{input_cif.name}: {symbol} ({number})", flush=True)

    print("Symmetry group rates:")
    for (number, symbol), group_count in sorted(symmetry_counts.items()):
        rate = group_count / len(input_cifs) * 100
        print(f"  {symbol} ({number}): {group_count}/{len(input_cifs)} ({rate:.2f}%)")

    return len(input_cifs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "collection",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    count = analyze_symmetry(args.collection)
    print(f"analyzed {count} MOF CIFs from {args.collection / 'mofs'}")


if __name__ == "__main__":
    main()
