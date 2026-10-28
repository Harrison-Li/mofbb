"""Assembled framework model and CIF writer."""

import copy
from pathlib import Path

import numpy as np
from ase.cell import Cell

from mofbb.io.cif import write_bonded_cif

from mofbb.common.connectivity import (
    bond_arrays,
    mic_shifts,
    validate_3d_connectivity,
)


class Framework:
    """Hold a fully assembled periodic framework and write it to CIF."""

    def __init__(self, atoms, bonds, bond_types, info, wrap=True, *, bond_shifts=None):
        """Store copies of the geometry, bonds, and metadata."""
        self.atoms = atoms.copy()
        if bond_shifts is None:
            bond_shifts = mic_shifts(atoms, bonds)
        pairs, shifts = bond_arrays(len(atoms), bonds, bond_shifts)
        self.bonds = pairs.copy()
        self.bond_shifts = shifts.copy()
        self.bond_types = copy.deepcopy(bond_types)
        self.info = copy.deepcopy(info)

        if wrap:
            self.wrap()

    def wrap(self):
        """Fold atoms into ``[0, 1)`` using only lattice translations."""

        scaled = self.atoms.get_scaled_positions(wrap=False)
        images = np.floor(scaled).astype(int)
        self.bond_shifts += images[self.bonds[:, 1]] - images[self.bonds[:, 0]]
        self.atoms.set_scaled_positions(scaled - images)

    def validate_3d(self):
        """Check current coordinates against the intended periodic bond graph."""
        quality = validate_3d_connectivity(
            self.atoms,
            self.bonds,
            self.bond_shifts,
            bond_scale=self.info.get("scale", 1.2),
        )
        self.info.update(quality)
        return quality

    def write_cif(self, filename: str | Path) -> Path:
        """Write the framework to a CIF file."""
        path = Path(filename).resolve()
        if path.suffix != ".cif":
            path = path.with_suffix(".cif")

        # Check the serialized geometry: rounding must not break a bridge.
        cell_parameters = np.round(self.atoms.cell.cellpar(), 3)
        cell = np.asarray(Cell.fromcellpar(cell_parameters))
        rounded = np.round(self.atoms.get_scaled_positions(wrap=False), 10)
        frac_coords = np.mod(rounded, 1.0)
        atom_images = np.rint(rounded - frac_coords).astype(int)
        left, right = self.bonds.T
        shifts = (
            self.bond_shifts
            + atom_images[right]
            - atom_images[left]
        )
        images = shifts + 5  # CIF encodes the zero image as 555.
        if np.any((images < 0) | (images > 9)):
            raise ValueError("Bond image cannot be represented by a CIF 1_abc code.")
        if self.info.get("require_3d", False):
            serialized = self.atoms.copy()
            serialized.set_cell(cell)
            serialized.set_scaled_positions(frac_coords)
            self.info.update(
                validate_3d_connectivity(
                    serialized, self.bonds, shifts,
                    bond_scale=self.info.get("scale", 1.2),
                )
            )

        return write_bonded_cif(
            path, self.atoms, self.bonds, self.bond_types, self.info,
            cell_parameters, cell, frac_coords, shifts,
        )
