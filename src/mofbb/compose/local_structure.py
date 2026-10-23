"""Normalized local connection geometry."""
import ase
import numpy as np


class LocalStructure:
    """Describe a slot as a set of unit direction vectors."""

    def __init__(self, positions, indices):
        """Build a local structure from a set of positions."""
        positions = np.asarray(positions, dtype=float)
        centered = positions - np.mean(positions, axis=0)
        directions = centered / np.linalg.norm(centered, axis=1)[:, np.newaxis]
        self.atoms = ase.Atoms(positions=directions)
        self.indices = np.array(indices, dtype=np.int32)

    @property
    def positions(self):
        """Normalized direction vectors of the slot."""
        return self.atoms.positions
