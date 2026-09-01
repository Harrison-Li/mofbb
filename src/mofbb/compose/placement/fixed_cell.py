"""Connection-site assignment and fixed-cell framework fitting."""

from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation

from mofbb.compose.placement.ports import DirectEdgeRepair, PortAssignmentSolver


class FixedCellPoseFitter:
    """Fit symmetry-linked rotations and translations in a fixed cell."""

    def __init__(self, framework_fitter):
        self.framework_fitter = framework_fitter

    def params(self):
        job = self.framework_fitter
        ids, frames, count = {}, {}, 0
        for orbits, compatible_orbits in (
            (job.symmetry.node_orbits, job.symmetry_compatible_node_orbits),
            (job.symmetry.edge_orbits, job.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                members = [int(i) for i in orbit.members if int(i) in job.slot_offsets]
                if not members:
                    continue
                if orbit.representative not in compatible_orbits:
                    for member in members:
                        ids[member], frames[member] = count, np.eye(3)
                        count += 1
                    continue
                variable = count
                count += 1
                for member in members:
                    ids[member] = variable
                    frames[member] = (
                        np.eye(3)
                        if member == orbit.representative
                        else job.symmetry.transform(
                            job.symmetry.operation(orbit, member), job.original_cell
                        )
                    )
        return ids, frames, count

    def moves(self):
        job = self.framework_fitter
        ids, count = {}, 0
        expanded = "symmetry_operations" in job.topology.atoms.info
        anchor = job.node_indices[0] if expanded else None
        for orbits, compatible_orbits in (
            (
                job.symmetry.node_orbits if expanded else (),
                job.symmetry_compatible_node_orbits,
            ),
            (job.symmetry.edge_orbits, job.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                members = [int(i) for i in orbit.members if int(i) in job.slot_offsets]
                if not members:
                    continue
                if orbit.representative in compatible_orbits:
                    if anchor not in members:
                        ids.update({member: count for member in members})
                        count += 1
                    continue
                for member in members:
                    if member != anchor:
                        ids[member] = count
                        count += 1
        return ids, count

    def fit(self, perms, centers, blocks):
        job = self.framework_fitter
        atom_pos = [
            np.round(blocks[slot].atoms.positions - blocks[slot].centroid, 10)
            for slot in job.slot_indices
        ]
        points = [
            atom_pos[i][blocks[slot].connection_point_indices]
            for i, slot in enumerate(job.slot_indices)
        ]
        rotation_ids, frames, rotation_count = self.params()
        move_ids, move_count = self.moves()
        rotation_size = 3 * rotation_count

        def rotations(values):
            raw = values[:rotation_size].reshape(-1, 3)
            return Rotation.from_rotvec(
                [
                    round(np.linalg.det(frames[slot]))
                    * frames[slot].T
                    @ raw[rotation_ids[slot]]
                    for slot in job.slot_indices
                ]
            )

        def residual(values):
            fitted = rotations(values)
            moves = np.zeros((len(job.slot_indices), 3))
            raw = values[rotation_size:].reshape(-1, 3)
            for slot, variable in move_ids.items():
                moves[job.slot_offsets[slot]] = frames[slot].T @ raw[variable]
            return np.asarray(
                [
                    centers[job.slot_offsets[right]]
                    + moves[job.slot_offsets[right]]
                    + image
                    + fitted[job.slot_offsets[right]].apply(
                        points[job.slot_offsets[right]][perms[right][right_x]]
                    )
                    - centers[job.slot_offsets[left]]
                    - moves[job.slot_offsets[left]]
                    - fitted[job.slot_offsets[left]].apply(
                        points[job.slot_offsets[left]][perms[left][left_x]]
                    )
                    for left, left_x, right, right_x, image in job.connections
                ]
            ).ravel()

        result = scipy.optimize.least_squares(
            residual,
            np.zeros(rotation_size + 3 * move_count),
            max_nfev=50,
        )
        fitted = rotations(result.x)
        for index, slot in enumerate(job.slot_indices):
            atom_pos[index] = fitted[index].apply(atom_pos[index])
            points[index] = atom_pos[index][blocks[slot].connection_point_indices]
        raw = result.x[rotation_size:].reshape(-1, 3)
        for slot, variable in move_ids.items():
            centers[job.slot_offsets[slot]] += frames[slot].T @ raw[variable]
        gaps = np.linalg.norm(residual(np.zeros_like(result.x)).reshape(-1, 3), axis=1)
        return result, gaps, atom_pos, points


class FixedCellFrameworkFitter:
    """Repair connection-site assignments and fit a framework in its source cell."""

    def __init__(self, assignment, edge_factory, *, tolerance, bond_scale):
        self.topology = assignment.topology
        self.templates = assignment.blocks
        self.blocks = assignment.placed
        self.permutations = assignment.perms
        self.provided_permutations = assignment.fixed
        self.symmetry = assignment.symmetry
        self.locator = assignment.locator
        self.pbc = assignment.pbc
        self.symmetry_compatible_node_orbits = (
            assignment.symmetry_compatible_node_orbits
        )
        self.symmetry_compatible_edge_orbits = (
            assignment.symmetry_compatible_edge_orbits
        )
        self.node_port_compatibility = assignment.node_port_compatibility
        self.edge_factory = edge_factory
        self.original_cell = assignment.cell
        self.tolerance = tolerance
        self.bond_scale = bond_scale
        self.direct_edges = DirectEdgeRepair(assignment, bond_scale)
        self.direct_edge_masks = self.direct_edges.masks
        self.optimization_result = None
        self.fitted = False

    def run(self):
        """Repair assignments, attempt fixed-cell fitting, and return this stage."""
        self.direct_edges.repair_if_needed()

        if self._should_fit_fixed_cell():
            result = self._fit_fixed_cell()
            if result is not None:
                self.blocks, self.permutations, self.optimization_result = result
                self.fitted = True
        return self

    def edge_endpoint_offsets(self, edge_index):
        return self.direct_edges.ends(edge_index)

    def _should_fit_fixed_cell(self):
        has_edges = any(
            self.templates[int(index)] is not None
            for index in self.topology.edge_indices
        )
        eligible = bool(
            self.direct_edge_masks and not has_edges
            or has_edges and "symmetry_operations" in self.topology.atoms.info
        )
        return eligible and not self._has_pose_restricting_stabilizer()

    def _has_pose_restricting_stabilizer(self):
        """Return whether a site symmetry forbids a freely fitted rigid pose."""

        identity = np.eye(3, dtype=int)
        for orbits, compatible_orbits in (
            (self.symmetry.node_orbits, self.symmetry_compatible_node_orbits),
            (self.symmetry.edge_orbits, self.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                if orbit.representative not in compatible_orbits:
                    continue
                if self.templates[orbit.representative] is None:
                    continue
                for operation_index in orbit.stabilizers:
                    rotation = np.rint(
                        self.symmetry.rotations[int(operation_index)]
                    ).astype(int)
                    if not (
                        np.array_equal(rotation, identity)
                        or np.array_equal(rotation, -identity)
                    ):
                        return True
        return False

    def _fit_fixed_cell(self):
        self.node_indices = [int(index) for index in self.topology.node_indices]
        self.edge_indices = [
            int(index) for index in self.topology.edge_indices
            if self.blocks[int(index)] is not None
        ]
        self.slot_indices = self.node_indices + self.edge_indices
        self.slot_offsets = {slot: offset for offset, slot in enumerate(self.slot_indices)}
        self.connections = self._connections()
        self.symmetry_compatible_edge_members = {
            int(member)
            for orbit in self.symmetry.edge_orbits
            if orbit.representative in self.symmetry_compatible_edge_orbits
            for member in orbit.members
        }
        self.fit_tolerance = min(self.tolerance, self.pbc.tolerance)

        candidate = [
            None if value is None else np.asarray(value).copy()
            for value in self.permutations
        ]
        seen = set()
        for _ in range(4):
            self._propagate_symmetry_related_permutations(candidate)
            key = tuple(np.asarray(candidate[index]).tobytes() for index in self.slot_indices)
            if key in seen:
                break
            seen.add(key)

            centers, blocks, edge_targets = self._place_candidate_blocks(candidate)
            result, gaps, atom_points, points = self._fit_candidate_geometry(
                candidate, centers, blocks
            )
            if (
                np.max(gaps) <= self.fit_tolerance
                and self._assignment_is_allowed(candidate)
            ):
                self._place_independently_fitted_edges(
                    candidate, edge_targets, blocks, centers, atom_points, points
                )
                for offset, slot in enumerate(self.slot_indices):
                    blocks[slot].atoms.set_positions(atom_points[offset] + centers[offset])
                return blocks, candidate, scipy.optimize.OptimizeResult(fun=result.cost)
            if "symmetry_operations" not in self.topology.atoms.info:
                break
            candidate = self._solve_port_assignment(centers, points)
            if candidate is None:
                break
        return None

    def _propagate_symmetry_related_permutations(self, candidate):
        for orbits, compatible_orbits in (
            (self.symmetry.node_orbits, self.symmetry_compatible_node_orbits),
            (self.symmetry.edge_orbits, self.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                if orbit.representative not in compatible_orbits:
                    continue
                representative = candidate[orbit.representative]
                for raw_member in orbit.members:
                    member = int(raw_member)
                    candidate[member] = self.symmetry.to_member(orbit, member, representative)

    def _port_pair_is_allowed(self, slot_i, source_i, slot_j, source_j):
        return PortAssignmentSolver(self).pair_ok(
            slot_i,
            source_i,
            slot_j,
            source_j,
        )

    def _assignment_is_allowed(self, candidate):
        return PortAssignmentSolver(self).valid(candidate)

    def _connections(self):
        return PortAssignmentSolver(self).connections()

    def _solve_port_assignment(self, centers, points):
        return PortAssignmentSolver(self).solve(centers, points)

    def _place_candidate_blocks(self, candidate):
        centers = self.topology.atoms.positions[self.slot_indices].copy()
        blocks = self.blocks.copy()
        for slot in self.slot_indices:
            blocks[slot] = self.templates[slot].copy()
            blocks[slot].set_centroid(self.topology.atoms.positions[slot])

        for orbit in self.symmetry.node_orbits:
            representative = orbit.representative
            members = [int(member) for member in orbit.members]
            if representative not in self.symmetry_compatible_node_orbits:
                for node in members:
                    blocks[node], _ = self.locator.locate_with_permutation(
                        self.topology.local_structure(node), blocks[node], candidate[node]
                    )
                    blocks[node].set_centroid(self.topology.atoms.positions[node])
                continue

            blocks[representative], _ = self.locator.locate_with_permutation(
                self.topology.local_structure(representative),
                blocks[representative], candidate[representative],
            )
            blocks[representative].set_centroid(self.topology.atoms.positions[representative])
            for member in members:
                if member != representative:
                    blocks[member] = self.symmetry.place(
                        blocks[representative],
                        self.symmetry.operation(orbit, member),
                        self.topology.atoms.positions[member], self.original_cell,
                    )

        edge_targets = (
            self.edge_factory(self.topology, self.topology, candidate, self.symmetry).build()
            if self.edge_indices
            else {}
        )
        for orbit in self.symmetry.edge_orbits:
            members = [
                int(member) for member in orbit.members
                if int(member) in self.edge_indices
            ]
            if not members:
                continue
            orbit_representative = orbit.representative
            representative = orbit_representative if orbit_representative in members else members[0]
            blocks[representative], _, _ = self.locator.locate_edge_with_frame(
                self._edge_target_points(representative),
                blocks[representative],
                candidate[representative],
                edge_targets[representative].transverse,
            )
            blocks[representative].set_centroid(self.topology.atoms.positions[representative])
            for member in members:
                if member == representative:
                    continue
                if orbit_representative in self.symmetry_compatible_edge_orbits:
                    blocks[member] = self.symmetry.place(
                        blocks[representative],
                        self.symmetry.operation(orbit, member),
                        self.topology.atoms.positions[member], self.original_cell,
                    )
                else:
                    blocks[member], _, _ = self.locator.locate_edge_with_frame(
                        self._edge_target_points(member),
                        blocks[member],
                        candidate[member],
                        edge_targets[member].transverse,
                    )
                    blocks[member].set_centroid(self.topology.atoms.positions[member])
        return centers, blocks, edge_targets

    def _edge_target_points(self, edge_index):
        return self.topology.atoms.positions[edge_index] + np.asarray(
            [neighbor.distance_vector for neighbor in self.topology.neighbor_list[edge_index]]
        )

    def _pose_parameters(self):
        return FixedCellPoseFitter(self).params()

    def _translation_parameters(self):
        return FixedCellPoseFitter(self).moves()

    def _fit_candidate_geometry(self, candidate, centers, blocks):
        return FixedCellPoseFitter(self).fit(candidate, centers, blocks)

    def _place_independently_fitted_edges(
        self, candidate, edge_targets, blocks, centers, atom_points, points
    ):
        for edge_index in self.edge_indices:
            if edge_index in self.symmetry_compatible_edge_members:
                continue
            target_points = [
                self.topology.atoms.positions[edge_index]
                + neighbor.distance_vector
                + points[self.slot_offsets[node]][candidate[node][target]]
                for neighbor, (node, target) in zip(
                    self.topology.neighbor_list[edge_index],
                    self.edge_endpoint_offsets(edge_index), strict=True,
                )
            ]
            blocks[edge_index], _, _ = self.locator.locate_edge_with_frame(
                target_points,
                self.templates[edge_index],
                candidate[edge_index],
                edge_targets[edge_index].transverse,
            )
            offset = self.slot_offsets[edge_index]
            centers[offset] = blocks[edge_index].centroid
            atom_points[offset] = blocks[edge_index].atoms.positions - centers[offset]
