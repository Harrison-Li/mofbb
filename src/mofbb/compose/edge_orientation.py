"""Symmetry-covariant target frames for two-connected linkers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pymatgen.symmetry.groups import SpaceGroup


@dataclass(frozen=True)
class EdgeOrientationTarget:
    """Transverse target direction and provenance for one topology edge."""

    transverse: np.ndarray
    symmetry_edge_type: int
    representative_edge: int
    operation_index: int
    endpoint_reversed: bool
    representative_gauge: str


def _unit_transverse(vector, edge_axis, epsilon=1e-12):
    """Return the vector's unit component perpendicular to the edge axis."""

    axis = np.asarray(edge_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    transverse = np.asarray(vector, dtype=float)
    transverse = transverse - np.dot(transverse, axis) * axis
    length = np.linalg.norm(transverse)
    if length < epsilon:
        raise ValueError("Cannot define a transverse direction parallel to an edge.")
    return transverse / length


def symmetry_edge_orientation_targets(
    original_topology,
    scaled_topology,
    permutations,
    tolerance=2e-3,
):
    """Return one symmetry-propagated transverse target per occupied edge.

    Negative topology tags identify crystallographic edge symmetry types. A
    transverse gauge is chosen once from the representative edge's local
    neighborhood and transformed to the other edges of that symmetry type with
    the topology space-group operations. The fractional rotations are converted
    through the scaled cell, so this remains consistent after the builder
    changes the lattice parameters.
    """

    if tolerance <= 0.0:
        raise ValueError("Edge symmetry matching tolerance must be positive.")

    original_cell = np.asarray(original_topology.atoms.cell, dtype=float)
    original_inverse_cell = np.linalg.inv(original_cell)
    scaled_cell = np.asarray(scaled_topology.atoms.cell, dtype=float)
    scaled_inverse_cell = np.linalg.inv(scaled_cell)
    original_centers = original_topology.atoms.positions @ original_inverse_cell

    # Gather the ordered endpoints once. The permutation defines which physical
    # endpoint is treated as the first endpoint throughout the symmetry type.
    edges_by_symmetry_type = defaultdict(list)
    edge_orders = {}
    original_fractional_endpoints = {}
    scaled_endpoints = {}
    for raw_edge_index in original_topology.edge_indices:
        edge_index = int(raw_edge_index)
        if permutations[edge_index] is None:
            continue

        edge_order = np.argsort(np.asarray(permutations[edge_index], dtype=int))
        edge_orders[edge_index] = edge_order

        original_center = original_topology.atoms.positions[edge_index]
        original_neighbors = original_topology.neighbor_list[edge_index]
        original_endpoints = np.array(
            [
                original_center + neighbor.distance_vector
                for neighbor in original_neighbors
            ]
        )
        original_fractional_endpoints[edge_index] = (
            original_endpoints[edge_order] @ original_inverse_cell
        )

        scaled_center = scaled_topology.atoms.positions[edge_index]
        scaled_neighbors = scaled_topology.neighbor_list[edge_index]
        scaled_endpoints[edge_index] = np.array(
            [
                scaled_center + neighbor.distance_vector
                for neighbor in scaled_neighbors
            ]
        )[edge_order]

        symmetry_edge_type = int(original_topology.node_types[edge_index])
        edges_by_symmetry_type[symmetry_edge_type].append(edge_index)

    operations = list(SpaceGroup(original_topology.spacegroup).symmetry_ops)
    targets = {}
    for symmetry_edge_type, edge_indices in edges_by_symmetry_type.items():
        representative_edge = min(edge_indices)

        # Choose a stable transverse direction from the representative edge's
        # local neighborhood. Prefer the longest off-axis neighboring edge and
        # use its index to break ties deterministically.
        edge_axis = (
            scaled_endpoints[representative_edge][1]
            - scaled_endpoints[representative_edge][0]
        )
        unit_axis = edge_axis / np.linalg.norm(edge_axis)
        transverse_candidates = []
        representative_neighbors = scaled_topology.neighbor_list[
            representative_edge
        ]
        for endpoint_index in edge_orders[representative_edge]:
            node_index = representative_neighbors[int(endpoint_index)].index
            for neighbor in scaled_topology.neighbor_list[node_index]:
                if neighbor.index == representative_edge:
                    continue

                neighbor_vector = np.asarray(neighbor.distance_vector, dtype=float)
                projected_vector = neighbor_vector - (
                    np.dot(neighbor_vector, unit_axis) * unit_axis
                )
                try:
                    transverse = _unit_transverse(neighbor_vector, edge_axis)
                except ValueError:
                    continue
                transverse_length = np.linalg.norm(projected_vector)
                transverse_candidates.append(
                    (
                        float(transverse_length),
                        -int(neighbor.index),
                        transverse,
                    )
                )

        if transverse_candidates:
            _, _, representative_transverse = max(
                transverse_candidates,
                key=lambda candidate: candidate[:2],
            )
            representative_gauge = "local_neighbor"
        else:
            # Low-connectivity synthetic nets may have no off-axis neighbor.
            cartesian_axes = np.eye(3)
            seed = cartesian_axes[np.argmin(np.abs(cartesian_axes @ unit_axis))]
            representative_transverse = _unit_transverse(seed, edge_axis)
            representative_gauge = "least_parallel_cartesian_fallback"

        targets[representative_edge] = EdgeOrientationTarget(
            transverse=representative_transverse,
            symmetry_edge_type=symmetry_edge_type,
            representative_edge=representative_edge,
            operation_index=-1,
            endpoint_reversed=False,
            representative_gauge=representative_gauge,
        )

        for edge_index in edge_indices:
            if edge_index == representative_edge:
                continue

            # Find the space-group operation that maps the representative edge
            # to this symmetry-equivalent edge. Fractional differences are
            # wrapped to the nearest periodic image before measuring the error.
            operation_candidates = []
            representative_center = original_centers[representative_edge]
            representative_endpoints = original_fractional_endpoints[
                representative_edge
            ]
            target_center = original_centers[edge_index]
            target_endpoints = original_fractional_endpoints[edge_index]
            for operation_index, operation in enumerate(operations):
                center_difference = (
                    operation.operate(representative_center) - target_center
                )
                center_difference -= np.round(center_difference)
                center_error = np.linalg.norm(center_difference)
                if center_error >= tolerance:
                    continue

                transformed_endpoints = np.array(
                    [
                        operation.operate(endpoint)
                        for endpoint in representative_endpoints
                    ]
                )
                direct_difference = transformed_endpoints - target_endpoints
                direct_difference -= np.round(direct_difference)
                direct_error = np.max(np.linalg.norm(direct_difference, axis=1))

                reversed_difference = transformed_endpoints - target_endpoints[::-1]
                reversed_difference -= np.round(reversed_difference)
                reversed_error = np.max(np.linalg.norm(reversed_difference, axis=1))
                endpoint_reversed = reversed_error < direct_error
                operation_candidates.append(
                    (
                        float(min(direct_error, reversed_error)),
                        endpoint_reversed,
                        float(center_error),
                        operation_index,
                        operation,
                    )
                )

            if not operation_candidates:
                raise ValueError(
                    "Could not map a representative topology edge to its "
                    "symmetry-equivalent edge."
                )

            best_match = min(
                operation_candidates,
                key=lambda candidate: candidate[:4],
            )
            (
                endpoint_error,
                endpoint_reversed,
                _,
                operation_index,
                operation,
            ) = best_match
            if endpoint_error >= tolerance:
                raise ValueError(
                    "Could not map a representative topology edge to its "
                    "symmetry-equivalent edge."
                )

            # Pymatgen applies column-vector fractional rotations. Coordinates
            # in MOFBB are row vectors, hence the transpose below.
            cartesian_transform = (
                scaled_inverse_cell
                @ np.asarray(operation.rotation_matrix, dtype=float).T
                @ scaled_cell
            )
            transverse = representative_transverse @ cartesian_transform
            target_start, target_end = scaled_endpoints[edge_index]
            target_axis = target_end - target_start
            transverse = _unit_transverse(transverse, target_axis)
            targets[edge_index] = EdgeOrientationTarget(
                transverse=transverse,
                symmetry_edge_type=symmetry_edge_type,
                representative_edge=representative_edge,
                operation_index=operation_index,
                endpoint_reversed=endpoint_reversed,
                representative_gauge=representative_gauge,
            )

    return targets
