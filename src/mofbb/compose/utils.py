"""Small composer I/O and geometry helpers."""

from pathlib import Path

import ase
import numpy as np



def wrap_fractional(values, tolerance=1.0e-12):
    """Return canonical fractional coordinates in the interval ``[0, 1)``."""

    wrapped = np.mod(values, 1.0)
    on_boundary = np.isclose(wrapped, 0.0, atol=tolerance, rtol=0.0) | np.isclose(
        wrapped,
        1.0,
        atol=tolerance,
        rtol=0.0,
    )
    return np.where(on_boundary, 0.0, wrapped)


def read_building_block_xyz(bb_file):
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
            "cpi": [index for index, symbol in enumerate(symbols) if symbol == "X"],
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
