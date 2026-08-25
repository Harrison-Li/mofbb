"""Periodic topology model."""
import copy

import numpy as np

from mofbb.compose.cgd import CgdReader
from mofbb.compose.local_structure import LocalStructure
from mofbb.compose.neighbor import NeighborList

class Topology:
    """Represent the slots, cell, types, and connectivity of a CGD net."""

    def __init__(self, cgd_file, local_structure_func=None):
        """Parse ``cgd_file`` and build/validate the net's properties."""
        self.atoms = CgdReader(cgd_file).read()
        self.update_properties()
        self.local_structure_func = local_structure_func

    def update_properties(self):
        """Parse, validate, and cache all net properties from ``self.atoms``."""

        # Save additional information.
        self.name = self.atoms.info["name"]
        self.spacegroup = self.atoms.info["spacegroup"]

        self.neighbor_list = NeighborList(self.atoms, method="distance")
        if not self.check_validity():
            self.neighbor_list = NeighborList(self.atoms, method="nearest")

        if not self.check_validity():
            raise ValueError(f"Invalid CGD topology: {self.name}")

        # Calculate properties.
        self.calculate_properties()

    def check_coordination_numbers(self):
        """Verify each slot's neighbor count matches its expected CN."""
        for cn, ns in zip(self.atoms.info["cn"], self.neighbor_list):
            if cn != len(ns):
                return False
        return True

    def check_edge_zerosum(self):
        """Verify each edge-center sits midway between its two nodes."""
        eps = 1e-3
        edge_indices = np.argwhere(self.atoms.get_tags() < 0).reshape(-1)
        for e in edge_indices:
            neighbors = self.neighbor_list[e]
            if len(neighbors) != 2:
                return False
            n1, n2 = neighbors

            zerosum = n1.distance_vector + n2.distance_vector
            zerosum = np.abs(zerosum)

            if (zerosum > eps).any():
                return False
        return True

    def check_validity(self):
        """Check that the parsed connectivity is self-consistent."""
        if not self.check_coordination_numbers():
            return False
        if not self.check_edge_zerosum():
            return False
        return True

    def copy(self):
        """Return a deep, independent copy of this topology."""
        return copy.deepcopy(self)

    def calculate_properties(self):
        """Derive cached node/edge indices, types, and counts."""
        # Build indices of nodes and edges.
        self._node_indices = np.argwhere(self.atoms.get_tags() >= 0).reshape(-1)
        self._edge_indices = np.argwhere(self.atoms.get_tags() < 0).reshape(-1)

        # Build node type.
        self._node_types = self.atoms.get_tags()

        # Build edge type.
        self._edge_types = [(-1, -1) for _ in range(self.n_slots)]
        for i in self.edge_indices:
            n = self.neighbor_list[i]

            i0 = n[0].index
            i1 = n[1].index

            t0 = self.get_node_type(i0)
            t1 = self.get_node_type(i1)

            # Sort.
            if t0 > t1:
                t0, t1 = t1, t0

            self._edge_types[i] = (t0, t1)
        self._edge_types = np.array(self._edge_types)

        # Calculate the number of node and edge types.
        self._n_node_types = np.unique(
            self.node_types[self.node_indices]
        ).shape[0]
        self._n_edge_types = np.unique(
            self.edge_types[self.edge_indices], axis=0
        ).shape[0]

        # self.cn = np.array([len(n) for n in self.neighbor_list])
        # self.cn = self.cn[self.node_indices]
        self.cn = self.atoms.info["cn"]
        self.unique_cn = []
        types = np.unique(self.node_types[self.node_indices])
        for t in types:
            i = np.argmax(self.node_types == t)
            self.unique_cn.append(self.cn[i])
        self.unique_cn = np.array(self.unique_cn)

    def local_structure(self, i):
        """Return the normalized neighbor directions around slot ``i``."""
        indices = []
        positions = []
        for n in self.neighbor_list[i]:
            indices.append(n.index)
            positions.append(n.distance_vector)

        return LocalStructure(
            positions, indices, normalization_func=self.local_structure_func
        )

    def get_node_type(self, i):
        """Return the node type of slot ``i``."""
        return self._node_types[i]

    @property
    def unique_node_types(self):
        """Return the distinct node types present in the net."""
        return np.unique(self.node_types[self.node_indices])

    @property
    def unique_edge_types(self):
        """Return the distinct edge types present in the net."""
        return np.unique(self.edge_types[self.edge_indices], axis=0)

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
    def node_indices(self):
        """Return the slot indices that are nodes (vertices)."""
        return self._node_indices

    @property
    def edge_indices(self):
        """Return the slot indices that are edge-centers."""
        return self._edge_indices

    @property
    def n_nodes(self):
        """Return the number of node (vertex) slots."""
        return len(self.node_indices)

    @property
    def n_edges(self):
        """Return the number of edge-center slots."""
        return len(self.edge_indices)

    @property
    def n_node_types(self):
        """Return the number of distinct node types."""
        return self._n_node_types

    @property
    def n_edge_types(self):
        """Return the number of distinct edge types."""
        return self._n_edge_types

    def __repr__(self):
        cn = ",".join(str(value) for value in self.unique_cn)
        return f"Topology {self.name}, ({cn})-cn, num edge types: {self.n_edge_types}"
