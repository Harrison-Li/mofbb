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
        return f"index: {self.index}, distance vector: {self.distance_vector}"


class NeighborList:
    """Build node-to-edge-center connectivity for a topology."""

    def __init__(self, atoms, method):
        """Build connectivity from ``atoms`` using the chosen ``method``."""
        self._neighbor_list = [[] for _ in atoms]
        explicit_endpoints = atoms.info.get("edge_endpoints")
        if explicit_endpoints is not None:
            self._build_explicit(atoms, explicit_endpoints)
        else:
            builders = {
                "distance": self._build_by_distance,
                "nearest": self._build_by_nearest_nodes,
            }
            builders[method](atoms)

    def _build_explicit(self, atoms, edge_endpoints):
        """Build connectivity from expanded CGD edge endpoint coordinates."""

        node_idx = np.flatnonzero(atoms.symbols == "C")
        edge_idx = np.flatnonzero(atoms.symbols == "O")
        if len(edge_endpoints) != len(edge_idx):
            raise ValueError(
                "Explicit CGD edge endpoints do not match the edge-center count."
            )

        scaled_positions = atoms.get_scaled_positions(wrap=True)
        cell = np.asarray(atoms.cell, dtype=float)
        for edge_index, raw_endpoints in zip(edge_idx, edge_endpoints):
            endpoints = np.asarray(raw_endpoints, dtype=float)
            center = np.mean(endpoints, axis=0)
            for endpoint in endpoints:
                differences = endpoint - scaled_positions[node_idx]
                image_shifts = np.rint(differences)
                residuals = (
                    scaled_positions[node_idx] + image_shifts - endpoint
                ) @ cell
                distances = np.linalg.norm(residuals, axis=1)
                match = np.argmin(distances)
                if distances[match] > 1e-5:
                    raise ValueError(
                        f"CGD edge slot {edge_index} endpoint {endpoint.tolist()} "
                        "does not match any topology node image."
                    )

                node_index = node_idx[match]
                vector = (endpoint - center) @ cell
                self._neighbor_list[edge_index].append(
                    Neighbor(node_index, vector)
                )
                self._neighbor_list[node_index].append(
                    Neighbor(edge_index, -vector)
                )

    def _build_by_distance(self, atoms):
        """Build connectivity by bonding C-O pairs within a fixed cutoff."""
        eps = 1e-3
        cutoffs = {
            ("C", "C"): 0.0,
            ("O", "O"): 0.0,
            ("C", "O"): 0.5 + eps,
        }

        first, second, vectors = ase.neighborlist.neighbor_list("ijD", atoms, cutoff=cutoffs)
        for left, right, vector in zip(first, second, vectors):
            self._neighbor_list[left].append(Neighbor(right, vector))

    def _build_by_nearest_nodes(self, atoms):
        """Build connectivity by keeping each edge-center's two nearest nodes."""
        node_idx = np.flatnonzero(atoms.symbols == "C")
        edge_idx = np.flatnonzero(atoms.symbols == "O")
        if len(node_idx) == 0:
            raise ValueError("A topology must contain at least one node.")

        periodic_images = [
            (-1, 0, 1) if periodic else (0,) for periodic in atoms.pbc
        ]
        shifts = np.asarray(list(product(*periodic_images)), dtype=int)
        translations = shifts @ atoms.cell

        for edge_index in edge_idx:
            candidates = []
            edge_position = atoms.positions[edge_index]
            for node_index in node_idx:
                vectors = (
                    atoms.positions[node_index]
                    + translations
                    - edge_position
                )
                for shift, vector in zip(shifts, vectors):
                    candidates.append(
                        (
                            vector @ vector,
                            node_index,
                            tuple(shift),
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
                    Neighbor(edge_index, -vector)
                )

    def __getitem__(self, i):
        """Return the list of ``Neighbor`` records for slot ``i``."""
        return self._neighbor_list[i]

    def __iter__(self):
        """Iterate over slots, yielding each slot's ``Neighbor`` list."""
        return iter(self._neighbor_list)

    def set_data(self, data):
        """Replace the connectivity with externally supplied records."""
        self._neighbor_list = [
            [Neighbor(index, vector) for index, vector in neighbors]
            for neighbors in data
        ]

    def __repr__(self):
        """Return a multi-line per-slot listing of neighbor counts and records."""
        return "".join(
            f"{index}: {len(neighbors)}\n"
            + "".join(f"{neighbor}\n" for neighbor in neighbors)
            for index, neighbors in enumerate(self)
        )
