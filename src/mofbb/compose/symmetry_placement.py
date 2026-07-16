"""Crystallographic orbit maps for atomistic building-block placement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.spatial.transform import Rotation

from .symmetry_scaling import _periodic_difference, _spacegroup_operations


# Expanded edge centers inherit the decimal precision of both CGD endpoints.
# A 1e-4 fractional threshold covers the observed rounding accumulation while
# remaining far below atomistic symmetry tolerances.
_MAPPING_TOLERANCE = 1.0e-4
_ORTHOGONALITY_TOLERANCE = 1.0e-8
_TORSION_GRID_SIZE = 24


@dataclass(frozen=True)
class PlacementOrbit:
    """One crystallographic orbit of topology slots."""

    tag: int
    representative: int
    member_indices: np.ndarray
    operation_indices: np.ndarray
    stabilizer_operation_indices: np.ndarray


class SymmetryPlacement:
    """Map topology slots and their incident ports under a space group."""

    def __init__(self, topology, mapping_tolerance=_MAPPING_TOLERANCE):
        if mapping_tolerance <= 0.0:
            raise ValueError("Symmetry mapping tolerance must be positive.")

        self.topology = topology
        self.mapping_tolerance = float(mapping_tolerance)
        self.operations = _spacegroup_operations(topology.spacegroup)
        self.rotations = np.asarray(
            [operation.rotation_matrix for operation in self.operations],
            dtype=float,
        )
        self.translations = np.asarray(
            [operation.translation_vector for operation in self.operations],
            dtype=float,
        )
        self.positions = np.asarray(
            topology.atoms.get_scaled_positions(wrap=True),
            dtype=float,
        )
        self.tags = np.asarray(topology.atoms.get_tags(), dtype=int)
        inverse_cell = np.linalg.inv(np.asarray(topology.atoms.cell, dtype=float))
        self.neighbor_indices = tuple(
            np.asarray(
                [neighbor.index for neighbor in slot_neighbors],
                dtype=int,
            )
            for slot_neighbors in topology.neighbor_list
        )
        self.neighbor_vectors = tuple(
            np.asarray(
                [neighbor.distance_vector for neighbor in slot_neighbors],
                dtype=float,
            ).reshape(-1, 3)
            @ inverse_cell
            for slot_neighbors in topology.neighbor_list
        )

        self.operation_mappings = self._build_operation_mappings()
        self.identity_operation_index = self._find_identity_operation()
        self._validate_connectivity_mappings()
        self.node_orbits = self._build_orbits(topology.node_indices)
        self.edge_orbits = self._build_orbits(topology.edge_indices)

    def _build_operation_mappings(self):
        mappings = []
        for operation_index, (rotation, translation) in enumerate(
            zip(self.rotations, self.translations)
        ):
            transformed = self.positions @ rotation.T + translation
            mapping = np.empty(len(self.positions), dtype=int)
            for source_index, transformed_position in enumerate(transformed):
                candidates = np.flatnonzero(self.tags == self.tags[source_index])
                differences = _periodic_difference(
                    transformed_position,
                    self.positions[candidates],
                )
                matches = np.all(
                    np.abs(differences) <= self.mapping_tolerance,
                    axis=1,
                )
                matched = candidates[matches]
                if len(matched) == 0:
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} cannot "
                        f"map slot {source_index} with tag "
                        f"{self.tags[source_index]} within fractional tolerance "
                        f"{self.mapping_tolerance:.1e}."
                    )
                if len(matched) > 1:
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} maps "
                        f"slot {source_index} ambiguously to slots "
                        f"{matched.tolist()} within fractional tolerance "
                        f"{self.mapping_tolerance:.1e}."
                    )
                mapping[source_index] = int(matched[0])

            if len(np.unique(mapping)) != len(mapping):
                raise ValueError(
                    f"Declared symmetry operation {operation_index} does not "
                    "map topology slots bijectively."
                )
            mappings.append(mapping)
        return np.asarray(mappings, dtype=int)

    def _find_identity_operation(self):
        for index, (rotation, translation) in enumerate(
            zip(self.rotations, self.translations)
        ):
            if not np.array_equal(
                np.rint(rotation).astype(int),
                np.eye(3, dtype=int),
            ):
                continue
            if np.all(
                np.abs(_periodic_difference(translation, 0.0)) <= 1.0e-12
            ):
                return index
        raise ValueError(
            "The declared space group does not contain an identity operation."
        )

    def _validate_connectivity_mappings(self):
        for operation_index, mapping in enumerate(self.operation_mappings):
            for source_index in range(len(self.positions)):
                target_index = int(mapping[source_index])
                try:
                    self.neighbor_offset_map(
                        source_index,
                        target_index,
                        operation_index,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} maps "
                        f"slot {source_index} to slot {target_index}, but does "
                        "not preserve their topology neighbors."
                    ) from error

    def _build_orbits(self, slot_indices):
        slot_indices = np.asarray(slot_indices, dtype=int)
        orbits = []
        for tag in np.unique(self.tags[slot_indices]):
            members = slot_indices[self.tags[slot_indices] == tag]
            members = np.sort(members)
            representative = int(members[0])
            reached = np.unique(
                self.operation_mappings[:, representative]
            )
            if not np.array_equal(np.sort(reached), members):
                raise ValueError(
                    f"Slot tag {int(tag)} is not one crystallographic orbit: "
                    f"representative slot {representative} reaches "
                    f"{reached.tolist()}, expected {members.tolist()}."
                )

            operation_indices = []
            for member in members:
                if member == representative:
                    operation_indices.append(self.identity_operation_index)
                    continue
                candidates = np.flatnonzero(
                    self.operation_mappings[:, representative] == member
                )
                if len(candidates) == 0:
                    raise ValueError(
                        f"No declared operation maps representative slot "
                        f"{representative} to slot {int(member)}."
                    )
                operation_indices.append(int(candidates[0]))

            stabilizer = np.flatnonzero(
                self.operation_mappings[:, representative] == representative
            )
            orbits.append(
                PlacementOrbit(
                    tag=int(tag),
                    representative=representative,
                    member_indices=members,
                    operation_indices=np.asarray(operation_indices, dtype=int),
                    stabilizer_operation_indices=stabilizer,
                )
            )
        return tuple(orbits)

    @staticmethod
    def operation_for_member(orbit, member_index):
        matches = np.flatnonzero(orbit.member_indices == member_index)
        if len(matches) != 1:
            raise ValueError(
                f"Slot {member_index} is not a unique member of orbit "
                f"tag {orbit.tag}."
            )
        return int(orbit.operation_indices[int(matches[0])])

    def neighbor_offset_map(self, source_slot, target_slot, operation_index):
        """Map source neighbor-list offsets onto target offsets."""

        source_neighbors = self.neighbor_indices[source_slot]
        target_neighbors = self.neighbor_indices[target_slot]
        mapped_neighbors = self.operation_mappings[
            operation_index, source_neighbors
        ]
        mapped_vectors = (
            self.neighbor_vectors[source_slot]
            @ self.rotations[operation_index].T
        )
        target_vectors = self.neighbor_vectors[target_slot]
        offset_map = np.empty(len(source_neighbors), dtype=int)
        for source_offset, (mapped_neighbor, mapped_vector) in enumerate(
            zip(mapped_neighbors, mapped_vectors)
        ):
            matches = np.flatnonzero(
                (target_neighbors == mapped_neighbor)
                & np.all(
                    np.abs(target_vectors - mapped_vector)
                    <= self.mapping_tolerance,
                    axis=1,
                )
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Operation {operation_index} maps neighbor offset "
                    f"{source_offset} of slot {source_slot} to slot "
                    f"{int(mapped_neighbor)}, which is not a unique neighbor "
                    f"of target slot {target_slot}."
                )
            offset_map[source_offset] = int(matches[0])
        if len(np.unique(offset_map)) != len(offset_map):
            raise ValueError(
                f"Operation {operation_index} does not map neighbor offsets "
                f"of slot {source_slot} bijectively onto slot {target_slot}."
            )
        return offset_map

    def member_permutation(self, orbit, member_index, representative_permutation):
        """Derive a member's port permutation from its orbit operation."""

        representative_permutation = np.asarray(
            representative_permutation,
            dtype=int,
        )
        operation_index = self.operation_for_member(orbit, member_index)
        offset_map = self.neighbor_offset_map(
            orbit.representative,
            member_index,
            operation_index,
        )
        if representative_permutation.shape != offset_map.shape:
            raise ValueError(
                f"Representative slot {orbit.representative} has "
                f"{len(offset_map)} neighbors but permutation shape "
                f"{representative_permutation.shape}."
            )
        member_permutation = np.empty_like(representative_permutation)
        member_permutation[offset_map] = representative_permutation
        return member_permutation

    def representative_permutation(self, orbit, member_index, member_permutation):
        """Pull a user-provided member permutation back to the representative."""

        member_permutation = np.asarray(member_permutation, dtype=int)
        operation_index = self.operation_for_member(orbit, member_index)
        offset_map = self.neighbor_offset_map(
            orbit.representative,
            member_index,
            operation_index,
        )
        if member_permutation.shape != offset_map.shape:
            raise ValueError(
                f"Member slot {member_index} has {len(offset_map)} neighbors "
                f"but permutation shape {member_permutation.shape}."
            )
        return member_permutation[offset_map]

    def cartesian_transform(self, operation_index, cell):
        """Return the row-vector Cartesian transform for one operation."""

        cell = np.asarray(cell, dtype=float)
        transform = (
            np.linalg.inv(cell)
            @ self.rotations[operation_index].T
            @ cell
        )
        residual = float(
            np.max(np.abs(transform @ transform.T - np.eye(3)))
        )
        if residual > _ORTHOGONALITY_TOLERANCE:
            raise ValueError(
                f"Operation {operation_index} is not orthogonal in the "
                f"placement cell: residual {residual:.3e}."
            )
        return transform

    def transform_building_block(
        self,
        representative_block,
        operation_index,
        target_centroid,
        cell,
    ):
        """Create one orbit member by transforming a representative block."""

        transform = self.cartesian_transform(operation_index, cell)
        member = representative_block.copy()
        centered = (
            representative_block.atoms.positions
            - representative_block.centroid
        )
        member.atoms.set_positions(
            centered @ transform + np.asarray(target_centroid, dtype=float)
        )
        return member

    def stabilizer_diagnostic(
        self,
        orbit,
        representative_block,
        representative_permutation,
        cell,
        tolerance,
    ):
        """Measure motif compatibility with a representative's stabilizer."""

        if tolerance <= 0.0:
            raise ValueError("Stabilizer tolerance must be positive.")

        positions = (
            representative_block.atoms.positions
            - representative_block.centroid
        )
        symbols = np.asarray(
            representative_block.atoms.get_chemical_symbols()
        )
        connection_indices = np.asarray(
            representative_block.connection_point_indices,
            dtype=int,
        )
        permutation = np.asarray(representative_permutation, dtype=int)

        maximum_atom_residual = 0.0
        maximum_port_residual = 0.0
        for operation_index in orbit.stabilizer_operation_indices:
            transform = self.cartesian_transform(operation_index, cell)
            transformed = positions @ transform
            maximum_atom_residual = max(
                maximum_atom_residual,
                self._species_assignment_residual(
                    transformed,
                    positions,
                    symbols,
                ),
            )

            offset_map = self.neighbor_offset_map(
                orbit.representative,
                orbit.representative,
                int(operation_index),
            )
            if permutation.shape != offset_map.shape:
                raise ValueError(
                    f"Representative slot {orbit.representative} has "
                    f"{len(offset_map)} neighbors but permutation shape "
                    f"{permutation.shape}."
                )
            source_atoms = connection_indices[permutation]
            target_atoms = connection_indices[permutation[offset_map]]
            port_residuals = np.linalg.norm(
                transformed[source_atoms] - positions[target_atoms],
                axis=1,
            )
            maximum_port_residual = max(
                maximum_port_residual,
                float(np.max(port_residuals, initial=0.0)),
            )

        maximum_residual = max(
            maximum_atom_residual,
            maximum_port_residual,
        )
        return {
            "tag": int(orbit.tag),
            "representative": int(orbit.representative),
            "orbit_size": int(len(orbit.member_indices)),
            "stabilizer_order": int(
                len(orbit.stabilizer_operation_indices)
            ),
            "tolerance": float(tolerance),
            "max_atom_residual": float(maximum_atom_residual),
            "max_port_residual": float(maximum_port_residual),
            "compatible": bool(maximum_residual <= tolerance),
        }

    def optimize_edge_stabilizer_torsion(
        self,
        orbit,
        representative_block,
        representative_permutation,
        cell,
        tolerance,
    ):
        """Use the free linker-axis torsion to satisfy a site stabilizer."""

        if not representative_block.is_edge:
            raise ValueError("Stabilizer torsion optimization requires an edge.")

        initial = self.stabilizer_diagnostic(
            orbit,
            representative_block,
            representative_permutation,
            cell,
            tolerance,
        )
        initial_residual = max(
            initial["max_atom_residual"],
            initial["max_port_residual"],
        )
        initial.update(
            {
                "torsion_optimized": False,
                "torsion_angle_degrees": 0.0,
                "best_torsion_residual": float(initial_residual),
            }
        )
        if (
            initial["compatible"]
            or len(orbit.stabilizer_operation_indices) == 1
        ):
            return representative_block, initial

        permutation = np.asarray(representative_permutation, dtype=int)
        ports = representative_block.connection_points[permutation]
        axis = ports[1] - ports[0]
        axis /= np.linalg.norm(axis)
        centroid = representative_block.centroid
        centered = representative_block.atoms.positions - centroid
        candidate = representative_block.copy()

        def evaluate(angle):
            wrapped_angle = float(angle % (2.0 * np.pi))
            rotation = Rotation.from_rotvec(wrapped_angle * axis)
            positions = rotation.apply(centered) + centroid
            candidate.atoms.set_positions(positions)
            diagnostic = self.stabilizer_diagnostic(
                orbit,
                candidate,
                permutation,
                cell,
                tolerance,
            )
            residual = max(
                diagnostic["max_atom_residual"],
                diagnostic["max_port_residual"],
            )
            return float(residual), positions.copy(), diagnostic, wrapped_angle

        step = 2.0 * np.pi / _TORSION_GRID_SIZE
        grid_results = [
            evaluate(index * step) for index in range(_TORSION_GRID_SIZE)
        ]
        best = min(grid_results, key=lambda result: result[0])
        if best[0] > tolerance:
            local = minimize_scalar(
                lambda angle: evaluate(angle)[0],
                bounds=(best[3] - step, best[3] + step),
                method="bounded",
                options={"xatol": 1.0e-5, "maxiter": 24},
            )
            refined = evaluate(local.x)
            if refined[0] < best[0]:
                best = refined

        initial["best_torsion_residual"] = best[0]
        if best[0] > tolerance:
            return representative_block, initial

        optimized = representative_block.copy()
        optimized.atoms.set_positions(best[1])
        diagnostic = best[2]
        diagnostic.update(
            {
                "torsion_optimized": True,
                "torsion_angle_degrees": float(np.degrees(best[3])),
                "best_torsion_residual": best[0],
            }
        )
        return optimized, diagnostic

    @staticmethod
    def _species_assignment_residual(transformed, reference, symbols):
        maximum = 0.0
        for symbol in np.unique(symbols):
            indices = np.flatnonzero(symbols == symbol)
            distances = np.linalg.norm(
                transformed[indices, np.newaxis, :]
                - reference[np.newaxis, indices, :],
                axis=2,
            )
            rows, columns = linear_sum_assignment(distances)
            maximum = max(
                maximum,
                float(np.max(distances[rows, columns], initial=0.0)),
            )
        return maximum
