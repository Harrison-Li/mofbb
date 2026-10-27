"""Source crystal geometry for locating extracted fragments."""

from __future__ import annotations

from dataclasses import dataclass

import ase
import numpy as np

from mofbb.common.pbc import ComponentPeriodicity, PeriodicBond, analyze_component
from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.object import Fragment


@dataclass
class Crystal:
    """Source atoms, bonds, and periodic records used during decomposition.

    Centroids are unwrapped fractional coordinates and may lie outside [0, 1).
    Source geometry and bond records must describe the same structure.
    """

    atoms: ase.Atoms
    bonds: BondNetwork
    bond_images: tuple[PeriodicBond, ...]

    def get_fragment_frame(
        self,
        fragment: Fragment,
        connections: tuple[PeriodicBond, ...],
    ) -> ComponentPeriodicity:
        """Use stored image offsets, or derive a frame with internal cuts removed."""
        if fragment.source_image_offsets:
            return ComponentPeriodicity(fragment.source_image_offsets, (), 0)
        indices = set(fragment.source_indices)
        internal_cuts = frozenset(
            record for record in connections
            if record.i in indices and record.j in indices
        )
        return analyze_component(indices, self.bond_images, excluded=internal_cuts)

    def get_bb_centroid(
        self, fragment: Fragment, frame: ComponentPeriodicity
    ) -> np.ndarray:
        """Return the real-atom centroid in unwrapped source fractional coordinates."""
        scaled = self.atoms.get_scaled_positions(wrap=False)
        positions = [scaled[i] + frame.offset(i) for i in fragment.source_indices]
        return np.mean(positions, axis=0)

    def get_x_centroid(
        self, fragment: Fragment, frame: ComponentPeriodicity
    ) -> np.ndarray:
        """Return the source fractional X centroid, or real-atom centroid."""
        if not fragment.x_sites:
            return self.get_bb_centroid(fragment, frame)
        scaled = self.atoms.get_scaled_positions(wrap=False)
        inverse_cell = np.linalg.inv(np.asarray(self.atoms.cell.array, dtype=float))
        positions = []
        for site in fragment.x_sites:
            anchor = scaled[site.internal_source_index] + np.asarray(
                frame.offset(site.internal_source_index), dtype=int
            )
            half_bond = (
                0.5
                * self.bonds.vector(site.internal_source_index, site.external_source_index)
                @ inverse_cell
            )
            positions.append(anchor + half_bond)
        return np.mean(positions, axis=0)
