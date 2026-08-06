"""Symmetry-covariant orientation targets for two-connected linkers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class EdgeOrientationTarget:
    """Orientation target and symmetry metadata for one topology edge."""

    transverse: np.ndarray
    symmetry_edge_type: int
    representative_edge: int
    operation_index: int
    endpoint_reversed: bool
    representative_gauge: str


class Edge:
    """Build symmetry-related transverse directions for topology edges."""

    def __init__(
        self,
        original_topology,
        scaled_topology,
        permutations,
        symmetry,
        tolerance=2e-3,
    ):
        if tolerance <= 0:
            raise ValueError("Edge symmetry matching tolerance must be positive.")

        self.original = original_topology
        self.scaled = scaled_topology
        self.permutations = permutations
        self.symmetry = symmetry
        self.tolerance = tolerance

        self.original_cell = np.asarray(self.original.atoms.cell, dtype=float)
        self.scaled_cell = np.asarray(self.scaled.atoms.cell, dtype=float)
        self.original_cell_inverse = np.linalg.inv(self.original_cell)
        self.scaled_cell_inverse = np.linalg.inv(self.scaled_cell)

        self.original_centers = (
            self.original.atoms.positions @ self.original_cell_inverse
        )
        self.orbit_by_edge = {
            int(member): orbit
            for orbit in self.symmetry.edge_orbits
            for member in orbit.members
        }
        self.edges_by_type = defaultdict(list)
        self.endpoint_order = {}
        self.original_endpoints = {}
        self.scaled_endpoints = {}

        self._prepare_edges()

    def build(self):
        """Return one transverse orientation target per occupied edge."""

        targets = {}

        for _, edge_indices in self.edges_by_type.items():
            representative = min(edge_indices)
            edge_type = int(self.original.node_types[representative])
            transverse, gauge = self._choose_transverse(representative)

            targets[representative] = EdgeOrientationTarget(
                transverse=transverse,
                symmetry_edge_type=edge_type,
                representative_edge=representative,
                operation_index=-1,
                endpoint_reversed=False,
                representative_gauge=gauge,
            )

            for edge_index in edge_indices:
                if edge_index == representative:
                    continue

                operation_index, endpoint_reversed = self._operation_for_edge(
                    representative,
                    edge_index,
                )

                cartesian_transform = (
                    self.scaled_cell_inverse
                    @ self.symmetry.rotations[operation_index].T
                    @ self.scaled_cell
                )

                transformed = transverse @ cartesian_transform
                start, end = self.scaled_endpoints[edge_index]
                transformed = self._perpendicular(transformed, end - start)

                targets[edge_index] = EdgeOrientationTarget(
                    transverse=transformed,
                    symmetry_edge_type=edge_type,
                    representative_edge=representative,
                    operation_index=operation_index,
                    endpoint_reversed=endpoint_reversed,
                    representative_gauge=gauge,
                )

        return targets

    def _prepare_edges(self):
        """Cache ordered endpoints and group occupied edges by symmetry type."""

        for raw_edge_index in self.original.edge_indices:
            edge_index = int(raw_edge_index)
            permutation = self.permutations[edge_index]
            if permutation is None:
                continue

            order = np.argsort(np.asarray(permutation, dtype=int))
            self.endpoint_order[edge_index] = order

            original_center = self.original.atoms.positions[edge_index]
            original_neighbors = self.original.neighbor_list[edge_index]
            original_points = np.array(
                [
                    original_center + neighbor.distance_vector
                    for neighbor in original_neighbors
                ]
            )
            self.original_endpoints[edge_index] = (
                original_points[order] @ self.original_cell_inverse
            )

            scaled_center = self.scaled.atoms.positions[edge_index]
            scaled_neighbors = self.scaled.neighbor_list[edge_index]
            scaled_points = np.array(
                [
                    scaled_center + neighbor.distance_vector
                    for neighbor in scaled_neighbors
                ]
            )
            self.scaled_endpoints[edge_index] = scaled_points[order]

            orbit = self.orbit_by_edge[edge_index]
            self.edges_by_type[orbit.representative].append(edge_index)

    def _choose_transverse(self, edge_index):
        """Choose a stable transverse direction for a representative edge."""

        start, end = self.scaled_endpoints[edge_index]
        edge_axis = end - start
        axis_length = np.linalg.norm(edge_axis)
        if axis_length < EPSILON:
            raise ValueError("Cannot orient a near-zero edge axis.")

        unit_axis = edge_axis / axis_length
        edge_neighbors = self.scaled.neighbor_list[edge_index]

        best_key = None
        best_transverse = None

        for endpoint in self.endpoint_order[edge_index]:
            node_index = edge_neighbors[int(endpoint)].index

            for neighbor in self.scaled.neighbor_list[node_index]:
                if neighbor.index == edge_index:
                    continue

                vector = np.asarray(neighbor.distance_vector, dtype=float)
                perpendicular = vector - np.dot(vector, unit_axis) * unit_axis
                length = float(np.linalg.norm(perpendicular))
                if length < EPSILON:
                    continue

                key = (length, -int(neighbor.index))
                if best_key is None or key > best_key:
                    best_key = key
                    best_transverse = perpendicular / length

        if best_transverse is not None:
            return best_transverse, "local_neighbor"

        axes = np.eye(3)
        fallback = axes[np.argmin(np.abs(axes @ unit_axis))]
        return (
            self._perpendicular(fallback, edge_axis),
            "least_parallel_cartesian_fallback",
        )

    def _operation_for_edge(self, representative, edge_index):
        """Choose a cached operation and its ordered-endpoint direction."""

        representative_center = self.original_centers[representative]
        representative_points = self.original_endpoints[representative]
        target_center = self.original_centers[edge_index]
        target_points = self.original_endpoints[edge_index]
        mapped_candidates = set(
            int(index)
            for index in self.symmetry.operations_between(
                representative,
                edge_index,
            )
        )
        candidates = []
        for raw_index, (operation, operation_index) in enumerate(
            zip(
                self.symmetry.raw_operations,
                self.symmetry.raw_operation_indices,
            )
        ):
            if operation_index not in mapped_candidates:
                continue

            center_difference = (
                operation.operate(representative_center) - target_center
            )
            center_difference -= np.round(center_difference)
            center_error = float(np.linalg.norm(center_difference))
            if center_error >= self.tolerance:
                continue

            transformed_points = np.array(
                [operation.operate(point) for point in representative_points]
            )
            direct_difference = transformed_points - target_points
            direct_difference -= np.round(direct_difference)
            direct_error = float(
                np.max(np.linalg.norm(direct_difference, axis=1))
            )
            reversed_difference = transformed_points - target_points[::-1]
            reversed_difference -= np.round(reversed_difference)
            reversed_error = float(
                np.max(np.linalg.norm(reversed_difference, axis=1))
            )
            endpoint_reversed = reversed_error < direct_error
            endpoint_error = min(direct_error, reversed_error)
            key = (
                endpoint_error,
                endpoint_reversed,
                center_error,
                raw_index,
            )
            candidates.append(
                (key, int(operation_index), endpoint_reversed)
            )

        if not candidates:
            raise ValueError(
                "Could not map a representative topology edge to its "
                "symmetry-equivalent edge."
            )
        best = min(candidates)
        if best[0][0] >= self.tolerance:
            raise ValueError(
                "Could not map a representative topology edge to its "
                "symmetry-equivalent edge."
            )
        _, operation_index, endpoint_reversed = best
        return operation_index, endpoint_reversed

    @staticmethod
    def _perpendicular(vector, axis):
        """Project a vector perpendicular to an axis and normalize it."""

        axis = np.asarray(axis, dtype=float)
        axis_length = np.linalg.norm(axis)
        if axis_length < EPSILON:
            raise ValueError("Cannot define a direction from a near-zero edge axis.")

        unit_axis = axis / axis_length
        perpendicular = np.asarray(vector, dtype=float).copy()
        perpendicular -= np.dot(perpendicular, unit_axis) * unit_axis

        length = np.linalg.norm(perpendicular)
        if length < EPSILON:
            raise ValueError(
                "Cannot define a transverse direction parallel to an edge."
            )

        return perpendicular / length
