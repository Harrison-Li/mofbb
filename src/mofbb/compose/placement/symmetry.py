"""Crystallographic orbit maps for atomistic building-block placement."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from weakref import WeakKeyDictionary

import numpy as np

from mofbb.compose.symmetry_scaling import Symmetry
from mofbb.compose.placement.stabilizer import StabilizerFit


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

    _cache = WeakKeyDictionary()
    _cache_lock = RLock()

    @staticmethod
    def topology_signature(topology):
        """Return the topology state that determines symmetry mappings."""

        groups = tuple(topology.neighbor_list)
        counts = np.asarray([len(group) for group in groups], dtype=np.int64)
        indices = np.asarray(
            [neighbor.index for group in groups for neighbor in group],
            dtype=np.int64,
        )
        vectors = np.asarray(
            [neighbor.distance_vector for group in groups for neighbor in group],
            dtype=float,
        ).reshape(-1, 3)

        def array_signature(values, dtype):
            array = np.ascontiguousarray(values, dtype=dtype)
            return array.shape, array.tobytes()

        return (
            str(topology.spacegroup),
            repr(topology.atoms.info.get("symmetry_operations")),
            repr(topology.atoms.info.get("slot_mappings")),
            array_signature(topology.atoms.cell, float),
            array_signature(topology.atoms.positions, float),
            array_signature(topology.atoms.get_tags(), np.int64),
            array_signature(counts, np.int64),
            array_signature(indices, np.int64),
            array_signature(vectors, float),
        )

    @classmethod
    def cached(cls, topology, mapping_tolerance=1.0e-4):
        """Return a validated placement cache entry for the topology state."""

        key = (
            cls,
            float(mapping_tolerance),
            cls.topology_signature(topology),
        )
        with cls._cache_lock:
            cached = cls._cache.get(topology)
            if cached is not None and cached[0] == key:
                return cached[1]

        placement = cls(topology, mapping_tolerance=mapping_tolerance)
        with cls._cache_lock:
            cached = cls._cache.get(topology)
            if cached is not None and cached[0] == key:
                return cached[1]
            cls._cache[topology] = (key, placement)
        return placement

    @classmethod
    def clear_cache(cls):
        """Clear cached topology placements, primarily for controlled tests."""

        with cls._cache_lock:
            cls._cache.clear()

    def __init__(self, topology, mapping_tolerance=1.0e-4):
        if mapping_tolerance <= 0:
            raise ValueError("Symmetry mapping tolerance must be positive.")

        self.topology = topology
        self.mapping_tolerance = float(mapping_tolerance)
        self.orthogonality_tolerance = 1.0e-8
        self.torsion_steps = 24
        self.tau = 2.0 * np.pi
        (
            self.raw_operations,
            self.operations,
            self.raw_operation_indices,
        ) = Symmetry.topology_operation_data(topology)
        self.rotations = np.asarray([op.rotation_matrix for op in self.operations], float)
        self.translations = np.asarray([op.translation_vector for op in self.operations], float)
        self.positions = np.asarray(topology.atoms.get_scaled_positions(wrap=True), float)
        self.tags = np.asarray(topology.atoms.get_tags(), int)
        self.neighbor_indices, self.neighbor_vectors = self.read_neighbors(topology)
        self.operation_mappings = self.map_slots()
        self.identity_operation_index = self.find_identity()
        self.check_links()
        self.node_orbits = self.make_orbits(topology.node_indices)
        self.edge_orbits = self.make_orbits(topology.edge_indices)

    @staticmethod
    def read_neighbors(topology):
        """Return neighbor indices and fractional neighbor vectors."""

        inverse = np.linalg.inv(np.asarray(topology.atoms.cell, float))
        groups = topology.neighbor_list
        indices = tuple(np.asarray([item.index for item in group], int) for group in groups)
        vectors = tuple(
            np.asarray([item.distance_vector for item in group], float).reshape(-1, 3) @ inverse
            for group in groups
        )
        return indices, vectors

    def map_slots(self):
        """Map every topology slot under every symmetry operation."""

        declared = self.topology.atoms.info.get("slot_mappings")
        if declared is not None:
            raw_mappings = np.asarray(declared, dtype=int)
            expected = (len(self.operations), len(self.positions))
            if raw_mappings.shape != expected:
                raise ValueError(
                    f"Declared slot mappings have shape {raw_mappings.shape}; "
                    f"expected {expected}."
                )
            mappings = np.empty_like(raw_mappings)
            for raw_index, ordered_index in enumerate(self.raw_operation_indices):
                mappings[ordered_index] = raw_mappings[raw_index]
            for op_index, mapping in enumerate(mappings):
                if sorted(mapping.tolist()) != list(range(len(self.positions))):
                    raise ValueError(
                        f"Operation {op_index} has a non-bijective declared slot map."
                    )
                if np.any(self.tags != self.tags[mapping]):
                    source = int(np.flatnonzero(self.tags != self.tags[mapping])[0])
                    raise ValueError(
                        f"Operation {op_index} changes slot {source} from tag "
                        f"{self.tags[source]} to {self.tags[mapping[source]]}."
                    )
                transformed = (
                    self.positions @ self.rotations[op_index].T
                    + self.translations[op_index]
                )
                differences = Symmetry.diff(transformed, self.positions[mapping])
                if np.any(abs(differences) > self.mapping_tolerance):
                    source = int(
                        np.flatnonzero(
                            np.any(
                                abs(differences) > self.mapping_tolerance,
                                axis=1,
                            )
                        )[0]
                    )
                    raise ValueError(
                        f"Operation {op_index} declared slot map sends slot "
                        f"{source} away from slot {int(mapping[source])}."
                    )
            return mappings

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
            remaining = set(slots[tags == tag])
            while remaining:
                representative = min(remaining)
                members = np.sort(
                    np.unique(self.operation_mappings[:, representative])
                )
                if not set(members) <= remaining:
                    raise ValueError(
                        f"Overlapping crystallographic orbits for slot tag {int(tag)}."
                    )

                operations = []
                for member in members:
                    if member == representative:
                        operations.append(self.identity_operation_index)
                        continue
                    candidates = np.flatnonzero(
                        self.operation_mappings[:, representative] == member
                    )
                    operations.append(int(candidates[0]))

                orbits.append(
                    SlotOrbit(
                        int(tag),
                        representative,
                        members,
                        np.asarray(operations, int),
                        np.flatnonzero(
                            self.operation_mappings[:, representative]
                            == representative
                        ),
                    )
                )
                remaining.difference_update(members)
        return tuple(orbits)

    @staticmethod
    def operation(orbit, member):
        """Return the operation mapping an orbit representative to a member."""

        matches = np.flatnonzero(orbit.members == member)
        if len(matches) != 1:
            raise ValueError(f"Slot {member} is not a unique member of orbit tag {orbit.tag}.")
        return int(orbit.operations[int(matches[0])])

    def operations_between(self, source, target):
        """Return operation indices that map one slot onto another."""

        matches = np.flatnonzero(self.operation_mappings[:, source] == target)
        if not len(matches):
            raise ValueError(f"No operation maps slot {source} to slot {target}.")
        return matches

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
            raise ValueError(
                f"Operation {op_index} does not map slot {source} "
                "neighbors bijectively."
            )
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
        member.atoms.set_positions(
            centered @ self.transform(op_index, cell)
            + np.asarray(centroid, float)
        )
        return member

    def diagnose(self, orbit, block, permutation, cell, tolerance):
        """Measure motif compatibility with an orbit stabilizer."""
        return StabilizerFit(self).diagnose(
            orbit,
            block,
            permutation,
            cell,
            tolerance,
        )

    def fit_torsion(self, orbit, block, permutation, cell, tolerance):
        """Fit the free linker-axis torsion to an edge-slot stabilizer."""
        return StabilizerFit(self).fit(
            orbit,
            block,
            permutation,
            cell,
            tolerance,
        )

    @staticmethod
    def species_residual(transformed, reference, symbols):
        """Return the largest optimal same-species assignment distance."""
        return StabilizerFit.species_error(transformed, reference, symbols)
