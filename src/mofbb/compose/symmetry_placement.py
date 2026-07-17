"""Crystallographic orbit maps for atomistic building-block placement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.spatial.transform import Rotation

from .symmetry_scaling import Symmetry


@dataclass(frozen=True, slots=True)
class SlotOrbit:
    """One crystallographic orbit of topology node or edge slots."""

    tag: int
    representative: int
    members: np.ndarray
    operations: np.ndarray
    stabilizers: np.ndarray


class SymmetryPlacement:
    """Map topology slots and their ports under a space group."""

    def __init__(self, topology, mapping_tolerance=1.0e-4):
        if mapping_tolerance <= 0:
            raise ValueError("Symmetry mapping tolerance must be positive.")

        self.topology = topology
        self.mapping_tolerance = float(mapping_tolerance)
        self.orthogonality_tolerance = 1.0e-8
        self.torsion_steps = 24
        self.tau = 2.0 * np.pi
        self.operations = Symmetry.operations(topology.spacegroup)
        self.rotations = np.asarray([op.rotation_matrix for op in self.operations], float)
        self.translations = np.asarray([op.translation_vector for op in self.operations], float)
        self.positions = np.asarray(topology.atoms.get_scaled_positions(wrap=True), float)
        self.tags = np.asarray(topology.atoms.get_tags(), int)
        self.neighbor_indices, self.neighbor_vectors = self.read_neighbors()
        self.operation_mappings = self.map_slots()
        self.identity_operation_index = self.find_identity()
        self.check_links()
        self.node_orbits = self.make_orbits(topology.node_indices)
        self.edge_orbits = self.make_orbits(topology.edge_indices)

    def read_neighbors(self):
        """Return neighbor indices and fractional neighbor vectors."""

        inverse = np.linalg.inv(np.asarray(self.topology.atoms.cell, float))
        groups = self.topology.neighbor_list
        indices = tuple(np.asarray([item.index for item in group], int) for group in groups)
        vectors = tuple(
            np.asarray([item.distance_vector for item in group], float).reshape(-1, 3) @ inverse
            for group in groups
        )
        return indices, vectors

    def map_slots(self):
        """Map every topology slot under every symmetry operation."""

        maps = []
        same_tag = self.tags[:, None] == self.tags[None, :]
        for op_index, (rotation, translation) in enumerate(zip(self.rotations, self.translations)):
            transformed = self.positions @ rotation.T + translation
            differences = Symmetry.diff(transformed[:, None], self.positions[None])
            matches = same_tag & np.all(abs(differences) <= self.mapping_tolerance, axis=2)
            counts = matches.sum(axis=1)
            invalid = np.flatnonzero(counts != 1)
            if len(invalid):
                source = int(invalid[0])
                targets = np.flatnonzero(matches[source])
                if not len(targets):
                    raise ValueError(
                        f"Operation {op_index} cannot map slot {source} with tag "
                        f"{self.tags[source]} within tolerance {self.mapping_tolerance:.1e}."
                    )
                raise ValueError(
                    f"Operation {op_index} maps slot {source} ambiguously to "
                    f"{targets.tolist()} within tolerance {self.mapping_tolerance:.1e}."
                )

            mapping = np.argmax(matches, axis=1)
            if len(np.unique(mapping)) != len(mapping):
                raise ValueError(f"Operation {op_index} does not map slots bijectively.")
            maps.append(mapping)
        return np.asarray(maps, int)

    def find_identity(self):
        """Return the identity operation index."""

        identity = np.eye(3, dtype=int)
        for index, (rotation, translation) in enumerate(zip(self.rotations, self.translations)):
            is_identity = np.array_equal(np.rint(rotation).astype(int), identity)
            has_no_shift = np.all(abs(Symmetry.diff(translation, 0)) <= 1.0e-12)
            if is_identity and has_no_shift:
                return index
        raise ValueError("The space group does not contain an identity operation.")

    def check_links(self):
        """Check that every operation preserves topology connectivity."""

        for op_index, mapping in enumerate(self.operation_mappings):
            for source, target in enumerate(mapping):
                try:
                    self.offset_map(source, int(target), op_index)
                except ValueError as error:
                    raise ValueError(
                        f"Operation {op_index} maps slot {source} to {target}, "
                        "but does not preserve its neighbors."
                    ) from error

    def make_orbits(self, slots):
        """Build crystallographic orbits for topology slots."""

        slots = np.asarray(slots, int)
        tags = self.tags[slots]
        orbits = []
        for tag in np.unique(tags):
            members = np.sort(slots[tags == tag])
            representative = int(members[0])
            reached = np.unique(self.operation_mappings[:, representative])
            if not np.array_equal(np.sort(reached), members):
                raise ValueError(
                    f"Tag {int(tag)} is not one orbit: slot {representative} reaches "
                    f"{reached.tolist()}, expected {members.tolist()}."
                )

            operations = []
            for member in members:
                if member == representative:
                    operations.append(self.identity_operation_index)
                    continue
                candidates = np.flatnonzero(self.operation_mappings[:, representative] == member)
                if not len(candidates):
                    raise ValueError(f"No operation maps slot {representative} to {int(member)}.")
                operations.append(int(candidates[0]))

            orbits.append(
                SlotOrbit(
                    int(tag),
                    representative,
                    members,
                    np.asarray(operations, int),
                    np.flatnonzero(self.operation_mappings[:, representative] == representative),
                )
            )
        return tuple(orbits)

    @staticmethod
    def operation(orbit, member):
        """Return the operation mapping an orbit representative to a member."""

        matches = np.flatnonzero(orbit.members == member)
        if len(matches) != 1:
            raise ValueError(f"Slot {member} is not a unique member of orbit tag {orbit.tag}.")
        return int(orbit.operations[int(matches[0])])

    def offset_map(self, source, target, op_index):
        """Map source neighbor offsets onto target neighbor offsets."""

        source_neighbors = self.neighbor_indices[source]
        target_neighbors = self.neighbor_indices[target]
        mapped_neighbors = self.operation_mappings[op_index, source_neighbors]
        mapped_vectors = self.neighbor_vectors[source] @ self.rotations[op_index].T
        target_vectors = self.neighbor_vectors[target]
        matches = (mapped_neighbors[:, None] == target_neighbors[None]) & np.all(
            abs(mapped_vectors[:, None] - target_vectors[None]) <= self.mapping_tolerance,
            axis=2,
        )
        invalid = np.flatnonzero(matches.sum(axis=1) != 1)
        if len(invalid):
            offset = int(invalid[0])
            raise ValueError(
                f"Operation {op_index} maps neighbor {offset} of slot {source} to "
                f"slot {int(mapped_neighbors[offset])}, not a unique neighbor of {target}."
            )

        result = np.argmax(matches, axis=1)
        if len(np.unique(result)) != len(result):
            raise ValueError(f"Operation {op_index} does not map slot {source} neighbors bijectively.")
        return result

    def to_member(self, orbit, member, permutation):
        """Map a representative port permutation to an orbit member."""

        permutation = np.asarray(permutation, int)
        offsets = self.offset_map(orbit.representative, member, self.operation(orbit, member))
        if permutation.shape != offsets.shape:
            raise ValueError(
                f"Slot {orbit.representative} has {len(offsets)} neighbors, "
                f"but permutation shape is {permutation.shape}."
            )
        result = np.empty_like(permutation)
        result[offsets] = permutation
        return result

    def to_representative(self, orbit, member, permutation):
        """Map a member port permutation back to its representative."""

        permutation = np.asarray(permutation, int)
        offsets = self.offset_map(orbit.representative, member, self.operation(orbit, member))
        if permutation.shape != offsets.shape:
            raise ValueError(
                f"Slot {member} has {len(offsets)} neighbors, "
                f"but permutation shape is {permutation.shape}."
            )
        return permutation[offsets]

    def transform(self, op_index, cell):
        """Return the row-vector Cartesian transform for an operation."""

        cell = np.asarray(cell, float)
        transform = np.linalg.inv(cell) @ self.rotations[op_index].T @ cell
        residual = float(np.max(abs(transform @ transform.T - np.eye(3))))
        if residual > self.orthogonality_tolerance:
            raise ValueError(f"Operation {op_index} is not orthogonal: residual {residual:.3e}.")
        return transform

    def place(self, block, op_index, centroid, cell):
        """Create an orbit member from a representative building block."""

        member = block.copy()
        centered = block.atoms.positions - block.centroid
        member.atoms.set_positions(centered @ self.transform(op_index, cell) + np.asarray(centroid, float))
        return member

    def diagnose(self, orbit, block, permutation, cell, tolerance):
        """Measure motif compatibility with an orbit stabilizer."""

        if tolerance <= 0:
            raise ValueError("Stabilizer tolerance must be positive.")

        positions = block.atoms.positions - block.centroid
        symbols = np.asarray(block.atoms.get_chemical_symbols())
        connections = np.asarray(block.connection_point_indices, int)
        permutation = np.asarray(permutation, int)
        atom_residual = port_residual = 0.0

        for op_index in orbit.stabilizers:
            op_index = int(op_index)
            transformed = positions @ self.transform(op_index, cell)
            atom_residual = max(atom_residual, self.species_residual(transformed, positions, symbols))
            offsets = self.offset_map(orbit.representative, orbit.representative, op_index)
            if permutation.shape != offsets.shape:
                raise ValueError(
                    f"Slot {orbit.representative} has {len(offsets)} neighbors, "
                    f"but permutation shape is {permutation.shape}."
                )
            source_atoms = connections[permutation]
            target_atoms = connections[permutation[offsets]]
            residuals = np.linalg.norm(transformed[source_atoms] - positions[target_atoms], axis=1)
            port_residual = max(port_residual, float(np.max(residuals, initial=0)))

        maximum = max(atom_residual, port_residual)
        return {
            "tag": int(orbit.tag),
            "representative": int(orbit.representative),
            "orbit_size": int(len(orbit.members)),
            "stabilizer_order": int(len(orbit.stabilizers)),
            "tolerance": float(tolerance),
            "max_atom_residual": float(atom_residual),
            "max_port_residual": float(port_residual),
            "compatible": maximum <= tolerance,
        }

    def fit_torsion(self, orbit, block, permutation, cell, tolerance):
        """Fit the free linker-axis torsion to an edge-slot stabilizer."""

        if not block.is_edge:
            raise ValueError("Stabilizer torsion optimization requires an edge.")

        initial = self.diagnose(orbit, block, permutation, cell, tolerance)
        residual = max(initial["max_atom_residual"], initial["max_port_residual"])
        initial.update(
            torsion_optimized=False,
            torsion_angle_degrees=0.0,
            best_torsion_residual=float(residual),
        )
        if initial["compatible"] or len(orbit.stabilizers) == 1:
            return block, initial

        permutation = np.asarray(permutation, int)
        ports = block.connection_points[permutation]
        axis = ports[1] - ports[0]
        axis /= np.linalg.norm(axis)
        centroid = block.centroid
        centered = block.atoms.positions - centroid
        candidate = block.copy()

        def evaluate(angle):
            angle = float(angle % self.tau)
            positions = Rotation.from_rotvec(angle * axis).apply(centered) + centroid
            candidate.atoms.set_positions(positions)
            report = self.diagnose(orbit, candidate, permutation, cell, tolerance)
            residual = max(report["max_atom_residual"], report["max_port_residual"])
            return float(residual), positions, report, angle

        step = self.tau / self.torsion_steps
        best = min(
            (evaluate(i * step) for i in range(self.torsion_steps)),
            key=lambda item: item[0],
        )
        if best[0] > tolerance:
            result = minimize_scalar(
                lambda angle: evaluate(angle)[0],
                bounds=(best[3] - step, best[3] + step),
                method="bounded",
                options={"xatol": 1.0e-5, "maxiter": 24},
            )
            refined = evaluate(result.x)
            best = min(best, refined, key=lambda item: item[0])

        initial["best_torsion_residual"] = best[0]
        if best[0] > tolerance:
            return block, initial

        optimized = block.copy()
        optimized.atoms.set_positions(best[1])
        report = best[2]
        report.update(
            torsion_optimized=True,
            torsion_angle_degrees=float(np.degrees(best[3])),
            best_torsion_residual=best[0],
        )
        return optimized, report

    @staticmethod
    def species_residual(transformed, reference, symbols):
        """Return the largest optimal same-species assignment distance."""

        maximum = 0.0
        for symbol in np.unique(symbols):
            indices = np.flatnonzero(symbols == symbol)
            distances = np.linalg.norm(
                transformed[indices, None] - reference[None, indices],
                axis=2,
            )
            rows, columns = linear_sum_assignment(distances)
            maximum = max(maximum, float(np.max(distances[rows, columns], initial=0)))
        return maximum
