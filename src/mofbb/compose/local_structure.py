"""Normalized local connection geometry."""
import ase
import numpy as np


class LocalStructure:
    """Describe a slot as a set of unit direction vectors."""

    def __init__(self, positions, indices, normalization_func=None):
        """Build a local structure from a set of positions."""
        # Normalize before using.
        if normalization_func is not None:
            positions = normalization_func(positions)
        else:
            positions = self.normalize_positions(positions)

        self.atoms = ase.Atoms(positions=positions)
        self.indices = np.array(indices, dtype=np.int32)

    @property
    def positions(self):
        """Normalized direction vectors of the slot."""
        return self.atoms.positions

    def normalize_positions(self, positions):
        """Convert positions to unit directions from their centroid."""
        # Calculate centroid.
        centroid = np.mean(positions, axis=0)

        # Calculate norms of the connection points.
        positions = positions - centroid
        distances = np.linalg.norm(positions, axis=1)

        # Normalize norm of connection points.
        positions = positions / distances[:, np.newaxis]

        # Warning: the centroid of positions are not the zero.
        return positions
