'''
Crystallographic orbit maps for atomistic building-block placement.
'''

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from weakref import WeakKeyDictionary

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.spatial.transform import Rotation

from mofbb.compose.scaler import Symmetry


@dataclass(frozen=True, slots=True)
class SlotOrbit:
    '''
    Store symmetry-related topology slots and the operations connecting them.

    Args:
        tag (int):
            Shared topology type.
        representative (int):
            Slot used as the reference for this orbit.
        members (numpy.ndarray):
            Slot indices belonging to the orbit.
        operations (numpy.ndarray):
            Operation taking the representative to each member, in member order.
        stabilizers (numpy.ndarray):
            Operations that leave the representative slot in place.
    '''

    tag: int
    representative: int
    members: np.ndarray
    operations: np.ndarray
    stabilizers: np.ndarray


class StabilizerFit:
    '''
    Measure site-symmetry residuals and optimize edge torsion.
    '''

    def __init__(self, symmetry):
        self.symmetry = symmetry

    def diagnose(self, orbit, block, perm, cell, tolerance):
        '''
        Check whether a block preserves the symmetry of its topology site.

        Args:
            orbit (SlotOrbit):
                Orbit containing the representative slot and its site-symmetry
                operations.
            block (BuildingBlock):
                Building block at the representative slot.
            perm (numpy.ndarray):
                Source port indices in target connection order.
            cell (array-like):
                A (3, 3) cell matrix with lattice vectors as rows, in angstroms.
            tolerance (float):
                Maximum atom or port mismatch in angstroms.

        Returns:
            A dictionary with the largest atom and port residuals in angstroms, orbit
            metadata, and a compatible flag.
        '''
        positions = block.atoms.positions - block.centroid
        symbols = np.asarray(block.atoms.get_chemical_symbols())
        ports = np.asarray(block.port_idx, int)
        perm = np.asarray(perm, int)
        atom_error = port_error = 0.0
        for operation in orbit.stabilizers:
            moved = positions @ self.symmetry.transform(operation, cell)
            atom_error = max(atom_error, self.species_error(moved, positions, symbols))
            offsets = self.symmetry.offset_map(
                orbit.representative, orbit.representative, operation
            )
            source = ports[perm]
            target = ports[perm[offsets]]
            errors = np.linalg.norm(moved[source] - positions[target], axis=1)
            port_error = max(port_error, np.max(errors, initial=0))
        return {
            "tag": int(orbit.tag),
            "representative": int(orbit.representative),
            "orbit_size": len(orbit.members),
            "stabilizer_order": len(orbit.stabilizers),
            "tolerance": float(tolerance),
            "max_atom_residual": float(atom_error),
            "max_port_residual": float(port_error),
            "compatible": bool(max(atom_error, port_error) <= tolerance),
        }

    def fit(self, orbit, block, perm, cell, tolerance):
        '''
        Rotate an edge about its port axis to satisfy site symmetry.

        Args:
            orbit (SlotOrbit):
                Orbit containing the representative slot and its site-symmetry
                operations.
            block (BuildingBlock):
                Building block at the representative slot.
            perm (numpy.ndarray):
                Source port indices in target connection order.
            cell (array-like):
                A (3, 3) cell matrix with lattice vectors as rows, in angstroms.
            tolerance (float):
                Maximum atom or port mismatch in angstroms.

        Returns:
            A fitted copy and diagnostic dictionary when a compatible angle is found.
            Otherwise, the original block and diagnostics are returned.
        '''
        base = self.diagnose(orbit, block, perm, cell, tolerance)
        error = max(base["max_atom_residual"], base["max_port_residual"])
        base.update(
            torsion_optimized=False,
            torsion_angle_degrees=0.0,
            best_torsion_residual=error,
        )
        if base["compatible"] or len(orbit.stabilizers) == 1:
            return block, base

        perm = np.asarray(perm, int)
        ports = block.port_pos[perm]
        axis = ports[1] - ports[0]
        axis /= np.linalg.norm(axis)
        center = block.centroid
        positions = block.atoms.positions - center
        trial = block.copy()

        def evaluate(angle):
            angle %= self.symmetry.tau
            moved = Rotation.from_rotvec(angle * axis).apply(positions) + center
            trial.atoms.set_positions(moved)
            report = self.diagnose(orbit, trial, perm, cell, tolerance)
            residual = max(report["max_atom_residual"], report["max_port_residual"])
            return residual, moved, report, angle

        step = self.symmetry.tau / self.symmetry.torsion_steps
        samples = [evaluate(index * step) for index in range(self.symmetry.torsion_steps)]
        minimum = min(sample[0] for sample in samples)
        best = min(
            (sample for sample in samples if sample[0] <= minimum + 1.0e-10),
            key=lambda sample: sample[3],
        )
        if best[0] > tolerance:
            result = minimize_scalar(
                lambda angle: evaluate(angle)[0],
                bounds=(best[3] - step, best[3] + step),
                method="bounded",
                options={"xatol": 1.0e-5, "maxiter": 24},
            )
            best = min(best, evaluate(result.x), key=lambda sample: sample[0])
        base["best_torsion_residual"] = best[0]
        if best[0] > tolerance:
            return block, base
        fitted = block.copy()
        fitted.atoms.set_positions(best[1])
        report = best[2]
        report.update(
            torsion_optimized=True,
            torsion_angle_degrees=float(np.degrees(best[3])),
            best_torsion_residual=best[0],
        )
        return fitted, report

    @staticmethod
    def species_error(moved, reference, symbols):
        '''
        Match atoms of the same element and measure their largest displacement.

        Args:
            moved (numpy.ndarray):
                An (n, 3) array of transformed atom positions.
            reference (numpy.ndarray):
                An (n, 3) array of reference positions in the same units.
            symbols (array-like):
                Element symbol for each atom.

        Returns:
            The largest matched distance after a minimum-total-distance assignment
            within each element.
        '''
        error = 0.0
        for symbol in np.unique(symbols):
            indices = np.flatnonzero(symbols == symbol)
            distances = np.linalg.norm(
                moved[indices, None] - reference[None, indices], axis=2
            )
            rows, columns = linear_sum_assignment(distances)
            error = max(error, np.max(distances[rows, columns], initial=0))
        return error


class SymPlacement:
    '''
    Map topology slots and connection orders under the declared space group.

    Args:
        topology (Topology):
            Topology with node and edge slots, connectivity, and symmetry metadata.
        map_tol (float):
            Tolerance for matching positions and neighbor vectors in fractional
            coordinates.
    '''

    _cache = WeakKeyDictionary()
    _cache_lock = RLock()

    @staticmethod
    def topology_signature(topology):
        '''
        Return the topology state that determines symmetry mappings.
        '''

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
    def cached(cls, topology, map_tol=1.0e-4):
        '''
        Reuse symmetry maps when the topology and matching tolerance are unchanged.

        Args:
            topology (Topology):
                Topology whose current geometry and connectivity determine the cache
                entry.
            map_tol (float):
                Matching tolerance in fractional coordinates.

        Returns:
            A SymPlacement instance for the current topology state.
        '''

        key = (
            cls,
            map_tol,
            cls.topology_signature(topology),
        )
        with cls._cache_lock:
            cached = cls._cache.get(topology)
            if cached is not None and cached[0] == key:
                return cached[1]

        placement = cls(topology, map_tol=map_tol)
        with cls._cache_lock:
            cached = cls._cache.get(topology)
            if cached is not None and cached[0] == key:
                return cached[1]
            cls._cache[topology] = (key, placement)
        return placement

    @classmethod
    def clear_cache(cls):
        '''
        Clear cached topology placements, primarily for controlled tests.
        '''

        with cls._cache_lock:
            cls._cache.clear()

    def __init__(self, topology, map_tol=1.0e-4):
        self.topology = topology
        self.map_tol = map_tol
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
        self.node_orbits = self.make_orbits(topology.node_idx)
        self.edge_orbits = self.make_orbits(topology.edge_idx)

    @staticmethod
    def read_neighbors(topology):
        '''
        Convert topology neighbor records to fractional-coordinate arrays.

        Args:
            topology (Topology):
                Topology with Cartesian neighbor vectors and a cell.

        Returns:
            Two tuples, indexed by slot: neighbor index arrays and (n_neighbors, 3)
            fractional displacement arrays.
        '''

        inverse = np.linalg.inv(np.asarray(topology.atoms.cell, float))
        groups = topology.neighbor_list
        indices = tuple(np.asarray([item.index for item in group], int) for group in groups)
        vectors = tuple(
            np.asarray([item.distance_vector for item in group], float).reshape(-1, 3) @ inverse
            for group in groups
        )
        return indices, vectors

    def map_slots(self):
        '''
        Find the destination of each slot under every symmetry operation.

        Returns:
            An (n_operations, n_slots) integer array. Declared mappings are checked for
            positions, tags, and one-to-one correspondence.
        '''

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
                if np.any(abs(differences) > self.map_tol):
                    source = int(
                        np.flatnonzero(
                            np.any(
                                abs(differences) > self.map_tol,
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
            matches = same_tag & np.all(abs(differences) <= self.map_tol, axis=2)
            counts = matches.sum(axis=1)
            invalid = np.flatnonzero(counts != 1)
            if len(invalid):
                source = int(invalid[0])
                targets = np.flatnonzero(matches[source])
                if not len(targets):
                    raise ValueError(
                        f"Operation {op_index} cannot map slot {source} with tag "
                        f"{self.tags[source]} within tolerance {self.map_tol:.1e}."
                    )
                raise ValueError(
                    f"Operation {op_index} maps slot {source} ambiguously to "
                    f"{targets.tolist()} within tolerance {self.map_tol:.1e}."
                )

            mapping = np.argmax(matches, axis=1)
            if len(np.unique(mapping)) != len(mapping):
                raise ValueError(f"Operation {op_index} does not map slots bijectively.")
            maps.append(mapping)
        return np.asarray(maps, int)

    def find_identity(self):
        '''
        Return the identity operation index.
        '''

        identity = np.eye(3, dtype=int)
        for index, (rotation, translation) in enumerate(zip(self.rotations, self.translations)):
            is_identity = np.array_equal(np.rint(rotation).astype(int), identity)
            has_no_shift = np.all(abs(Symmetry.diff(translation, 0)) <= 1.0e-12)
            if is_identity and has_no_shift:
                return index
        raise ValueError("The space group does not contain an identity operation.")

    def check_links(self):
        '''
        Check that every operation preserves topology connectivity.
        '''

        for op_index, mapping in enumerate(self.operation_mappings):
            for source, target in enumerate(mapping):
                self.offset_map(source, int(target), op_index)

    def make_orbits(self, slots):
        '''
        Group symmetry-related slots with the same topology tag.

        Args:
            slots (array-like):
                Topology slot indices to group.

        Returns:
            A tuple of SlotOrbit records with representative slots, member operations,
            and stabilizers.
        '''

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
        '''
        Return the operation mapping an orbit representative to a member.
        '''

        matches = np.flatnonzero(orbit.members == member)
        return int(orbit.operations[int(matches[0])])

    def operations_between(self, source, target):
        '''
        Return operation indices that map one slot onto another.
        '''

        matches = np.flatnonzero(self.operation_mappings[:, source] == target)
        if not len(matches):
            raise ValueError(f"No operation maps slot {source} to slot {target}.")
        return matches

    def offset_map(self, source, target, op_index):
        '''
        Match the source connection order to the transformed target connection order.

        Args:
            source (int):
                Source topology slot.
            target (int):
                Target topology slot.
            op_index (int):
                Symmetry operation mapping source to target.

        Returns:
            An integer array mapping each source neighbor offset to its target offset.
            Periodic displacement vectors are included in the match.
        '''

        source_neighbors = self.neighbor_indices[source]
        target_neighbors = self.neighbor_indices[target]
        mapped_neighbors = self.operation_mappings[op_index, source_neighbors]
        mapped_vectors = self.neighbor_vectors[source] @ self.rotations[op_index].T
        target_vectors = self.neighbor_vectors[target]
        matches = (mapped_neighbors[:, None] == target_neighbors[None]) & np.all(
            abs(mapped_vectors[:, None] - target_vectors[None]) <= self.map_tol,
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

    def to_member(self, orbit, member, perm):
        '''
        Carry a representative port mapping to another member of its orbit.

        Args:
            orbit (SlotOrbit):
                Orbit containing the representative slot and its site-symmetry
                operations.
            member (int):
                Member topology slot index.
            perm (numpy.ndarray):
                Source port indices in target connection order.

        Returns:
            Source port indices in the member slot connection order.
        '''

        perm = np.asarray(perm, int)
        offsets = self.offset_map(orbit.representative, member, self.operation(orbit, member))
        result = np.empty_like(perm)
        result[offsets] = perm
        return result

    def to_representative(self, orbit, member, perm):
        '''
        Carry a member port mapping back to its orbit representative.

        Args:
            orbit (SlotOrbit):
                Orbit containing the representative slot and its site-symmetry
                operations.
            member (int):
                Member topology slot index.
            perm (numpy.ndarray):
                Source port indices in target connection order.

        Returns:
            Source port indices in the representative slot connection order.
        '''

        perm = np.asarray(perm, int)
        offsets = self.offset_map(orbit.representative, member, self.operation(orbit, member))
        return perm[offsets]

    def transform(self, op_index, cell):
        '''
        Convert a fractional symmetry rotation to a Cartesian row-vector transform.

        Args:
            op_index (int):
                Index of the symmetry operation.
            cell (array-like):
                A (3, 3) cell matrix with lattice vectors as rows, in angstroms.

        Returns:
            A (3, 3) orthogonal matrix. Raises ValueError if the cell does not preserve
            orthogonality for this operation.
        '''

        cell = np.asarray(cell, float)
        transform = np.linalg.inv(cell) @ self.rotations[op_index].T @ cell
        residual = float(np.max(abs(transform @ transform.T - np.eye(3))))
        if residual > self.orthogonality_tolerance:
            raise ValueError(f"Operation {op_index} is not orthogonal: residual {residual:.3e}.")
        return transform

    def place(self, block, op_index, centroid, cell):
        '''
        Copy a representative block to a symmetry-related slot.

        Args:
            block (BuildingBlock):
                Building block at the representative slot.
            op_index (int):
                Symmetry operation to apply about the block centroid.
            centroid (array-like):
                Desired three-component Cartesian center in angstroms.
            cell (array-like):
                A (3, 3) cell matrix with lattice vectors as rows, in angstroms.

        Returns:
            A transformed BuildingBlock copy centered at the requested position.
        '''

        member = block.copy()
        centered = block.atoms.positions - block.centroid
        member.atoms.set_positions(
            centered @ self.transform(op_index, cell)
            + np.asarray(centroid, float)
        )
        return member

    def diagnose(self, orbit, block, perm, cell, tolerance):
        '''
        Check whether a block preserves the symmetry of its topology site.

        Args:
            orbit (SlotOrbit):
                Orbit containing the representative slot and its site-symmetry
                operations.
            block (BuildingBlock):
                Building block at the representative slot.
            perm (numpy.ndarray):
                Source port indices in target connection order.
            cell (array-like):
                A (3, 3) cell matrix with lattice vectors as rows, in angstroms.
            tolerance (float):
                Maximum atom or port mismatch in angstroms.

        Returns:
            A dictionary with the largest atom and port residuals in angstroms, orbit
            metadata, and a compatible flag.
        '''
        return StabilizerFit(self).diagnose(
            orbit,
            block,
            perm,
            cell,
            tolerance,
        )

    def fit_torsion(self, orbit, block, perm, cell, tolerance):
        '''
        Fit an edge rotation about its port axis to the edge site symmetry.

        Args:
            orbit (SlotOrbit):
                Orbit containing the representative slot and its site-symmetry
                operations.
            block (BuildingBlock):
                Building block at the representative slot.
            perm (numpy.ndarray):
                Source port indices in target connection order.
            cell (array-like):
                A (3, 3) cell matrix with lattice vectors as rows, in angstroms.
            tolerance (float):
                Maximum atom or port mismatch in angstroms.

        Returns:
            The compatible fitted block and diagnostics, or the original block and
            diagnostics if no acceptable angle is found.
        '''
        return StabilizerFit(self).fit(
            orbit,
            block,
            perm,
            cell,
            tolerance,
        )
