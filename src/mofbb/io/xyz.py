"""Building-block XYZ files with appended bond records."""

from collections.abc import Iterable
from pathlib import Path

import ase
import numpy as np


def read_bb_xyz(bb_file: str | Path) -> ase.Atoms:
    """Parse a building-block ``.xyz`` file into an :class:`ase.Atoms`."""
    path = Path(bb_file)
    lines = path.read_text(encoding="utf-8").splitlines()
    n_atoms = int(lines[0])
    atoms = [line.split() for line in lines[2 : n_atoms + 2]]
    symbols = [record[0] for record in atoms]
    bond_records = [
        record
        for line in lines[n_atoms + 2 :]
        if len(record := line.split()) >= 3
    ]
    has_bonds = len(lines) > n_atoms + 2
    return ase.Atoms(
        symbols=symbols,
        positions=[[float(value) for value in record[1:4]] for record in atoms],
        charges=[float(record[4]) if len(record) == 5 else 0 for record in atoms],
        info={
            "port_idx": [index for index, symbol in enumerate(symbols) if symbol == "X"],
            "name": path.stem,
            "bonds": (
                np.asarray([[int(record[0]), int(record[1])] for record in bond_records])
                if has_bonds
                else None
            ),
            "bond_types": (
                [record[2] for record in bond_records] if has_bonds else None
            ),
        },
    )

def write_bb_xyz(
    path: Path,
    atoms: ase.Atoms,
    bonds: Iterable[tuple[int, int, str]],
) -> Path:
    """Write coordinates, dummy-atom indices, and appended bond records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    symbols = atoms.get_chemical_symbols()
    x_indices = [str(i) for i, symbol in enumerate(symbols) if symbol == "X"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(" ".join(x_indices) + "\n")
        for symbol, coord in zip(symbols, atoms.positions):
            handle.write(
                f"{symbol:<2s} {coord[0]: .10f} {coord[1]: .10f} {coord[2]: .10f}\n"
            )
        for i, j, bond_type in bonds:
            handle.write(f"{i:4d} {j:4d} {bond_type}\n")
    return path
