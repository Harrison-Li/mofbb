"""Connectivity between topology nodes and edge centers."""
from itertools import product

import ase
import ase.neighborlist
import numpy as np

class Neighbor:
    """Hold a single topology connectivity record."""

    def __init__(self, index, distance_vector):
        """Store the neighbor's slot index and displacement vector."""
        self.index = index
        self.distance_vector = distance_vector

    def __repr__(self):
        """Return a one-line ``index``/``distance vector`` summary."""
        return "index: {}, distance vector: {}".format(
            self.index, self.distance_vector
        )


class NeighborList:
    """Build node-to-edge-center connectivity for a topology."""

    def __init__(self, atoms, method):
        """Build connectivity from ``atoms`` using the chosen ``method``."""
        explicit_endpoints = atoms.info.get("edge_endpoints")
        if explicit_endpoints is not None:
            self.explicit_build(atoms, explicit_endpoints)
            return

        # C for nodes and O for edges.
        if method == "distance":
            self.distance_based_build(atoms)
        elif method == "nearest":
            self.nearest_two_based_build(atoms)
        else:
            raise ValueError(f"Unknown neighbor-list method: {method}.")

    def explicit_build(self, atoms, edge_endpoints):
        """Build connectivity from expanded CGD edge endpoint coordinates."""

        node_indices = np.flatnonzero(atoms.symbols == "C")
        edge_indices = np.flatnonzero(atoms.symbols == "O")
        if len(edge_endpoints) != len(edge_indices):
            raise ValueError(
                "Explicit CGD edge endpoints do not match the edge-center count."
            )

        scaled_positions = atoms.get_scaled_positions(wrap=True)
        cell = np.asarray(atoms.cell, dtype=float)
        self.max_index = len(atoms) - 1
        self._neighbor_list = [[] for _ in atoms]

        for edge_index, raw_endpoints in zip(edge_indices, edge_endpoints):
            endpoints = np.asarray(raw_endpoints, dtype=float)
            center = np.mean(endpoints, axis=0)
            for endpoint in endpoints:
                differences = endpoint - scaled_positions[node_indices]
                image_shifts = np.rint(differences)
                residuals = (
                    scaled_positions[node_indices] + image_shifts - endpoint
                ) @ cell
                distances = np.linalg.norm(residuals, axis=1)
                match = int(np.argmin(distances))
                if distances[match] > 1e-5:
                    raise ValueError(
                        f"CGD edge slot {edge_index} endpoint {endpoint.tolist()} "
                        "does not match any topology node image."
                    )

                node_index = int(node_indices[match])
                vector = (endpoint - center) @ cell
                self._neighbor_list[edge_index].append(
                    Neighbor(node_index, vector)
                )
                self._neighbor_list[node_index].append(
                    Neighbor(int(edge_index), -vector)
                )

    def distance_based_build(self, atoms):
        """Build connectivity by bonding C-O pairs within a fixed cutoff."""
        eps = 1e-3
        cutoffs = {
            ("C", "C"): 0.0,
            ("O", "O"): 0.0,
            ("C", "O"): 0.5 + eps,
        }

        I, J, D = ase.neighborlist.neighbor_list("ijD", atoms, cutoff=cutoffs)

        self.max_index = len(atoms) - 1
        self._neighbor_list = [[] for _ in atoms]

        for i, j, d in zip(I, J, D):
            self._neighbor_list[i].append(Neighbor(j, d))

    def nearest_two_based_build(self, atoms):
        """Build connectivity by keeping each edge-center's two nearest nodes."""
        node_indices = np.flatnonzero(atoms.symbols == "C")
        edge_indices = np.flatnonzero(atoms.symbols == "O")
        if len(node_indices) == 0:
            raise ValueError("A topology must contain at least one node.")

        periodic_images = [
            (-1, 0, 1) if periodic else (0,) for periodic in atoms.pbc
        ]
        shifts = np.asarray(list(product(*periodic_images)), dtype=int)
        cell = np.asarray(atoms.cell, dtype=float)

        self.max_index = len(atoms) - 1
        self._neighbor_list = [[] for _ in atoms]
        for edge_index in edge_indices:
            candidates = []
            edge_position = atoms.positions[edge_index]
            for node_index in node_indices:
                vectors = (
                    atoms.positions[node_index]
                    + shifts @ cell
                    - edge_position
                )
                for shift, vector in zip(shifts, vectors):
                    candidates.append(
                        (
                            float(vector @ vector),
                            int(node_index),
                            tuple(int(value) for value in shift),
                            vector,
                        )
                    )

            candidates.sort(key=lambda item: item[:3])
            if len(candidates) < 2:
                raise ValueError(
                    f"Topology edge slot {edge_index} has fewer than two "
                    "candidate node images."
                )
            for _, node_index, _, vector in candidates[:2]:
                self._neighbor_list[edge_index].append(
                    Neighbor(node_index, vector)
                )
                self._neighbor_list[node_index].append(
                    Neighbor(int(edge_index), -vector)
                )

    def __getitem__(self, i):
        """Return the list of ``Neighbor`` records for slot ``i``."""
        return self._neighbor_list[i]

    def __iter__(self):
        """Iterate over slots, yielding each slot's ``Neighbor`` list."""
        return iter(self._neighbor_list)

    def set_data(self, data):
        """Replace the connectivity with externally supplied records."""
        new_list = []
        for neighbor in data:
            new_list.append([])
            for n in neighbor:
                new_list[-1].append(Neighbor(n[0], n[1]))

        self._neighbor_list = new_list

    def __repr__(self):
        """Return a multi-line per-slot listing of neighbor counts and records."""
        output = ""
        for i, neighbor in enumerate(self):
            line = "{}: {}\n".format(i, len(neighbor))
            for n in neighbor:
                line += "{}\n".format(n)
            output += line
        return output
