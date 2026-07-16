#!/usr/bin/env python3
"""Write pymatgen symmetry-analyzed copies of generated MOF CIFs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import shutil
import warnings

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def write_symmetry_cifs(collection: Path) -> int:
    input_dir = collection / "mofs"
    output_dir = collection / "sym_mofs"
    staging_dir = collection / ".sym_mofs.staging"
    input_cifs = sorted(input_dir.glob("*.cif"))
    if not input_cifs:
        raise FileNotFoundError(f"No CIFs found under {input_dir}.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}.")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    symmetry_counts: Counter[tuple[int, str]] = Counter()
    writer_tolerance_counts: Counter[str] = Counter()
    report_rows = []

    try:
        for input_cif in input_cifs:
            structure = Structure.from_file(input_cif, primitive=False)
            analyzer = SpacegroupAnalyzer(
                structure,
                symprec=0.1,
                angle_tolerance=5,
            )
            symbol = analyzer.get_space_group_symbol()
            number = analyzer.get_space_group_number()
            output_cif = staging_dir / input_cif.name
            selected_tolerance = None
            written = None
            for writer_tolerance in (0.1, 0.05, 0.01, None):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Incorrect stoichiometry.*"
                    )
                    CifWriter(
                        structure,
                        symprec=writer_tolerance,
                        angle_tolerance=5,
                        refine_struct=writer_tolerance is not None,
                        significant_figures=8,
                    ).write_file(output_cif)
                    candidate = Structure.from_file(output_cif, primitive=False)
                composition_matches = (
                    candidate.composition.fractional_composition
                    == structure.composition.fractional_composition
                )
                source_density = len(structure) / structure.volume
                written_density = len(candidate) / candidate.volume
                density_matches = (
                    abs(written_density - source_density) / source_density < 0.01
                )
                if composition_matches and density_matches:
                    selected_tolerance = writer_tolerance
                    written = candidate
                    break
            if written is None:
                raise ValueError(
                    f"Could not write a composition-preserving CIF: {input_cif.name}"
                )
            tolerance_label = (
                "P1_fallback" if selected_tolerance is None else str(selected_tolerance)
            )
            writer_tolerance_counts[tolerance_label] += 1
            report_rows.append(
                {
                    "cif": input_cif.name,
                    "detected_symbol_0.1": symbol,
                    "detected_number_0.1": number,
                    "writer_symprec": tolerance_label,
                    "output_atoms": len(written),
                    "output_formula": written.composition.formula,
                }
            )
            symmetry_counts[number, symbol] += 1
            print(
                f"{input_cif.name}: {symbol} ({number}), "
                f"writer_symprec={tolerance_label}",
                flush=True,
            )
        with (staging_dir / "_generation_report.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
            writer.writeheader()
            writer.writerows(report_rows)
        os.replace(staging_dir, output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    print("Symmetry group rates:")
    for (number, symbol), group_count in sorted(symmetry_counts.items()):
        rate = group_count / len(input_cifs) * 100
        print(f"  {symbol} ({number}): {group_count}/{len(input_cifs)} ({rate:.2f}%)")
    print(f"Writer tolerances: {dict(sorted(writer_tolerance_counts.items()))}")

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
    count = write_symmetry_cifs(args.collection)
    print(f"wrote {count} symmetry CIFs to {args.collection / 'sym_mofs'}")


if __name__ == "__main__":
    main()
