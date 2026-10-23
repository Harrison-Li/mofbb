"""Periodic topology model."""
import copy
from functools import cached_property

import numpy as np

from mofbb.compose.cgd import CgdReader
from mofbb.compose.connectivity import periodic_components
from mofbb.compose.local_structure import LocalStructure
from mofbb.compose.neighbor import NeighborList


class Topology:
    """Represent the slots, cell, types, and connectivity of a CGD net."""

    def __init__(self, cgd_file):
        """Parse ``cgd_file`` and build/validate the net's properties."""
        self.atoms = CgdReader(cgd_file).read()
        self.update_properties()

    def update_properties(self):
        """Parse, validate, and cache all net properties from ``self.atoms``."""
        self.__dict__.pop("component_ranks", None)

        self.name = self.atoms.info["name"]
        self.spacegroup = self.atoms.info["spacegroup"]

        for method in ("distance", "nearest"):
            self.neighbor_list = NeighborList(self.atoms, method=method)
            if self._has_valid_connectivity():
                break
        else:
            raise ValueError(f"Invalid CGD topology: {self.name}")

        self.calculate_properties()

    def _has_valid_connectivity(self):
        """Check declared coordination and the midpoint of each edge."""
        for cn, neighbors in zip(self.atoms.info["cn"], self.neighbor_list):
            if cn != len(neighbors):
                return False

        for edge in np.flatnonzero(self.atoms.get_tags() < 0):
            left, right = self.neighbor_list[edge]
            if np.any(np.abs(left.distance_vector + right.distance_vector) > 1e-3):
                return False
        return True

    @property
    def edge_shifts(self):
        """Integer images of each edge's two nodes, relative to its center."""
        positions = self.atoms.positions
        inverse_cell = np.linalg.inv(self.atoms.cell)
        shifts = {}
        for edge in self.edge_idx:
            vectors = np.array([
                positions[edge] + neighbor.distance_vector - positions[neighbor.index]
                for neighbor in self.neighbor_list[edge]
            ])
            fractional = vectors @ inverse_cell
            images = np.rint(fractional)
            if not np.allclose(fractional, images, atol=1e-5, rtol=0):
                raise ValueError(f"Topology edge {edge} has inconsistent periodic images.")
            shifts[int(edge)] = images.astype(int)
        return shifts

    @cached_property
    def component_ranks(self):
        """Cache dimensionality of the declared net, including image edges."""
        pairs, shifts = [], []
        for edge, images in self.edge_shifts.items():
            pairs.extend((edge, neighbor.index) for neighbor in self.neighbor_list[edge])
            shifts.extend(images)
        return tuple(
            component["rank"]
            for component in periodic_components(self.n_slots, pairs, shifts)
        )

    def validate_3d(self):
        """Require a 3D net before any building-block placement."""
        if not self.component_ranks or any(rank != 3 for rank in self.component_ranks):
            dimensions = ", ".join(f"{rank}D" for rank in self.component_ranks)
            raise ValueError(
                f"3D topology required; {self.name!r} contains {dimensions or 'no components'}."
            )

    def copy(self):
        """Return a deep, independent copy of this topology."""
        return copy.deepcopy(self)

    def calculate_properties(self):
        """Derive cached node/edge indices, types, and counts."""
        self._node_types = self.atoms.get_tags()
        self._node_idx = np.flatnonzero(self._node_types >= 0)
        self._edge_idx = np.flatnonzero(self._node_types < 0)

        self._edge_types = np.full((self.n_slots, 2), -1, dtype=int)
        for edge in self.edge_idx:
            self._edge_types[edge] = sorted(
                self.node_types[neighbor.index] for neighbor in self.neighbor_list[edge]
            )

        self.cn = self.atoms.info["cn"]
        self.unique_cn = np.array([
            self.cn[np.flatnonzero(self.node_types == kind)[0]]
            for kind in self.unique_node_types
        ])

    def local_structure(self, i):
        """Return the normalized neighbor directions around slot ``i``."""
        indices = []
        positions = []
        for n in self.neighbor_list[i]:
            indices.append(n.index)
            positions.append(n.distance_vector)

        return LocalStructure(positions, indices)

    @property
    def unique_node_types(self):
        """Return the distinct node types present in the net."""
        return np.unique(self.node_types[self.node_idx])

    @property
    def unique_edge_types(self):
        """Return the distinct edge types present in the net."""
        return np.unique(self.edge_types[self.edge_idx], axis=0)

    @property
    def node_types(self):
        """Return the per-slot node-type array."""
        return self._node_types

    @property
    def edge_types(self):
        """Return the per-slot edge-type array."""
        return self._edge_types

    @property
    def n_slots(self):
        """Return the number of fillable slots."""
        return len(self.atoms)

    @property
    def node_idx(self):
        """Return the slot indices that are nodes (vertices)."""
        return self._node_idx

    @property
    def edge_idx(self):
        """Return the slot indices that are edge-centers."""
        return self._edge_idx

    @property
    def n_nodes(self):
        """Return the number of node (vertex) slots."""
        return len(self.node_idx)

    @property
    def n_edges(self):
        """Return the number of edge-center slots."""
        return len(self.edge_idx)

    @property
    def n_node_types(self):
        """Return the number of distinct node types."""
        return len(self.unique_node_types)

    @property
    def n_edge_types(self):
        """Return the number of distinct edge types."""
        return len(self.unique_edge_types)

    def __repr__(self):
        cn = ",".join(str(value) for value in self.unique_cn)
        return f"Topology {self.name}, ({cn})-cn, num edge types: {self.n_edge_types}"
