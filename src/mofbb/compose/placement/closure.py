"""Port assignment and fixed-cell closure for framework composition."""

from __future__ import annotations

import numpy as np
import scipy.optimize

from mofbb.compose.placement.direct import DirectEdgeRepair
from mofbb.compose.placement.ports import PortSolver
from mofbb.compose.placement.pose import PoseFit


class CellClosure:
    """Repair port assignments and close eligible structures in their source cell.

    Direct node-to-node edges first receive a joint binary assignment so every
    node uses each port exactly once and forbidden carbon-metal or overlong
    connections are excluded. For eligible expanded topologies, a nonlinear
    rigid-pose fit then attempts to close all assigned ports without changing
    the declared cell. If no candidate closes within tolerance, the caller can
    safely continue through the ordinary cell-scaling path.

    The class mutates the supplied block and permutation lists only when a port
    repair or successful closure is found. :meth:`run` returns this stage with
    the current blocks, optional SciPy fit, and explicit ``closed`` flag.
    """

    def __init__(self, assignment, edge_factory, *, tolerance, bond_scale):
        self.topology = assignment.topology
        self.templates = assignment.blocks
        self.blocks = assignment.placed
        self.permutations = assignment.perms
        self.provided_permutations = assignment.fixed
        self.symmetry = assignment.symmetry
        self.locator = assignment.locator
        self.pbc = assignment.pbc
        self.covariant_node_orbits = assignment.node_cov
        self.covariant_edge_orbits = assignment.edge_cov
        self.node_port_compatibility = assignment.node_port_compatibility
        self.edge_factory = edge_factory
        self.original_cell = assignment.cell
        self.tolerance = tolerance
        self.bond_scale = bond_scale
        self.direct_edges = DirectEdgeRepair(assignment, bond_scale)
        self.direct_edge_masks = self.direct_edges.masks
        self.fit = None
        self.closed = False

    def run(self):
        """Repair assignments, attempt closure, and return this stage."""
        self.direct_edges.repair_if_needed()

        if self._should_close_fixed_cell():
            result = self._close_fixed_cell()
            if result is not None:
                self.blocks, self.permutations, self.fit = result
                self.closed = True
        return self

    def edge_endpoint_offsets(self, edge_index):
        return self.direct_edges.ends(edge_index)

    def _should_close_fixed_cell(self):
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
        for orbits, covariant in (
            (self.symmetry.node_orbits, self.covariant_node_orbits),
            (self.symmetry.edge_orbits, self.covariant_edge_orbits),
        ):
            for orbit in orbits:
                if orbit.representative not in covariant:
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

    def _close_fixed_cell(self):
        self.node_indices = [int(index) for index in self.topology.node_indices]
        self.edge_indices = [
            int(index) for index in self.topology.edge_indices
            if self.blocks[int(index)] is not None
        ]
        self.slot_indices = self.node_indices + self.edge_indices
        self.slot_offsets = {slot: offset for offset, slot in enumerate(self.slot_indices)}
        self.connections = self._connections()
        self.covariant_edge_members = {
            int(member)
            for orbit in self.symmetry.edge_orbits
            if orbit.representative in self.covariant_edge_orbits
            for member in orbit.members
        }
        self.closure_tolerance = min(self.tolerance, self.pbc.tolerance)

        candidate = [
            None if value is None else np.asarray(value).copy()
            for value in self.permutations
        ]
        seen = set()
        for _ in range(4):
            self._propagate_covariant_permutations(candidate)
            key = tuple(np.asarray(candidate[index]).tobytes() for index in self.slot_indices)
            if key in seen:
                break
            seen.add(key)

            centers, blocks, edge_targets = self._place_candidate_blocks(candidate)
            result, gaps, atom_points, points = self._fit_candidate_geometry(
                candidate, centers, blocks
            )
            if (
                np.max(gaps) <= self.closure_tolerance
                and self._assignment_is_allowed(candidate)
            ):
                self._place_non_covariant_edges(
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

    def _propagate_covariant_permutations(self, candidate):
        for orbits, covariant in (
            (self.symmetry.node_orbits, self.covariant_node_orbits),
            (self.symmetry.edge_orbits, self.covariant_edge_orbits),
        ):
            for orbit in orbits:
                if orbit.representative not in covariant:
                    continue
                representative = candidate[orbit.representative]
                for raw_member in orbit.members:
                    member = int(raw_member)
                    candidate[member] = self.symmetry.to_member(orbit, member, representative)

    def _port_pair_is_allowed(self, slot_i, source_i, slot_j, source_j):
        return PortSolver(self).pair_ok(slot_i, source_i, slot_j, source_j)

    def _assignment_is_allowed(self, candidate):
        return PortSolver(self).valid(candidate)

    def _connections(self):
        return PortSolver(self).connections()

    def _solve_port_assignment(self, centers, points):
        return PortSolver(self).solve(centers, points)

    def _place_candidate_blocks(self, candidate):
        centers = self.topology.atoms.positions[self.slot_indices].copy()
        blocks = self.blocks.copy()
        for slot in self.slot_indices:
            blocks[slot] = self.templates[slot].copy()
            blocks[slot].set_centroid(self.topology.atoms.positions[slot])

        for orbit in self.symmetry.node_orbits:
            representative = orbit.representative
            members = [int(member) for member in orbit.members]
            if representative not in self.covariant_node_orbits:
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
                if orbit_representative in self.covariant_edge_orbits:
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
        return PoseFit(self).params()

    def _translation_parameters(self):
        return PoseFit(self).moves()

    def _fit_candidate_geometry(self, candidate, centers, blocks):
        return PoseFit(self).fit(candidate, centers, blocks)

    def _place_non_covariant_edges(
        self, candidate, edge_targets, blocks, centers, atom_points, points
    ):
        for edge_index in self.edge_indices:
            if edge_index in self.covariant_edge_members:
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
