'''
Store assembled atoms and bonds, including their periodic images.
'''

import copy

import numpy as np


from mofbb.common.connectivity import (
    bond_arrays,
    mic_shifts,
    validate_3d_connectivity,
)


class Framework:
    '''
    Keep the geometry, periodic bonds, and metadata of an assembled MOF.

    Args:
        atoms (ase.Atoms):
            Atomic coordinates and periodic unit cell.
        bonds (array-like):
            An (n_bonds, 2) array of atom-index pairs.
        bond_types (sequence[str]):
            One bond-order label per bond.
        info (dict):
            Assembly metadata and validation settings.
        wrap (bool):
            Whether to wrap atoms into the unit cell on construction.
        bond_shifts (array-like or None):
            An (n_bonds, 3) array of integer images of the second atom relative to the
            first. If omitted, use minimum-image shifts.
    '''

    def __init__(self, atoms, bonds, bond_types, info, wrap=True, *, bond_shifts=None):
        '''
        Copy the supplied structure and bond data into a framework.

        Args:
            atoms (ase.Atoms):
                Atomic coordinates and periodic unit cell.
            bonds (array-like):
                An (n_bonds, 2) array of atom-index pairs.
            bond_types (sequence[str]):
                One bond-order label per bond.
            info (dict):
                Assembly metadata and validation settings.
            wrap (bool):
                Whether to wrap atoms into the unit cell on construction.
            bond_shifts (array-like or None):
                An (n_bonds, 3) array of integer images of the second atom relative to
                the first. If omitted, use minimum-image shifts.
        '''
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
        '''
        Move atoms into the unit cell and update their bond image shifts.

        Coordinates and bond shifts are changed in place so each bond still
        connects the same periodic images.
        '''

        scaled = self.atoms.get_scaled_positions(wrap=False)
        images = np.floor(scaled).astype(int)
        self.bond_shifts += images[self.bonds[:, 1]] - images[self.bonds[:, 0]]
        self.atoms.set_scaled_positions(scaled - images)

    def validate_3d(self):
        '''
        Check the framework geometry against its periodic bond graph.

        Returns:
            A dictionary of connectivity diagnostics, also added to self.info.
        '''
        quality = validate_3d_connectivity(
            self.atoms,
            self.bonds,
            self.bond_shifts,
            bond_scale=self.info.get("scale", 1.2),
        )
        self.info.update(quality)
        return quality
