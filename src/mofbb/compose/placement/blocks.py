"""Final symmetry-aware placement of building blocks in a composed cell."""

from __future__ import annotations

import numpy as np


class BlockPlacement:
    """Scale the topology, then place node and edge blocks in the final cell.

    Slot assignment determines which ports correspond before this class runs.
    ``BlockPlacement`` reuses those permutations after scaling, propagates a
    representative pose across symmetry-compatible orbits, and falls back to
    independent placement when a site stabilizer or propagated edge misses its
    target ports. The fallback decisions and edge-frame diagnostics are written
    into the shared diagnostics dictionaries returned to :class:`Builder`.

    Parameters
    ----------
    assignment : SlotAssignment
        Completed slot-assignment stage with symmetry and port information.
    closure : CellClosure
        Completed closure stage with the current blocks and permutations.
    scaler, edge_factory
        Geometry helpers used when the source cell was not closed directly.
    """

    def __init__(self, assignment, closure, scaler, edge_factory):
        self.source_topology = assignment.topology
        self.topology = assignment.topology
        self.blocks = closure.blocks
        self.permutations = closure.permutations
        self.symmetry = assignment.symmetry
        self.locator = assignment.locator
        self.scaler = scaler
        self.pbc = assignment.pbc
        self.edge_factory = edge_factory
        self.covariant_node_orbits = assignment.node_cov
        self.covariant_edge_orbits = assignment.edge_cov
        self.symmetry_diagnostics = assignment.diag
        self.fixed_cell_closed = closure.closed
        self.scaling_result = closure.fit
        self.port_tolerance = closure.tolerance
        self.bond_scale = closure.bond_scale
        self.symmetry_tolerance = assignment.sym_tol
        self.edge_targets = {}
        self.edge_frame_diagnostics = []

    def run(self):
        """Place all blocks in the final cell and return this stage."""
        self._prepare_topology()
        self.rmsds = np.asarray(self._place_nodes())
        self._place_edges()
        return self

    def _prepare_topology(self):
        if self.fixed_cell_closed:
            self.topology = self.topology.copy()
        else:
            self.topology, self.scaling_result = self.scaler.scale(
                topology=self.topology, bbs=self.blocks,
                perms=self.permutations, return_result=True,
            )

        has_edges = any(self.blocks[int(index)] is not None for index in self.topology.edge_indices)
        if has_edges and not self.fixed_cell_closed:
            self.edge_targets = self.edge_factory(
                self.source_topology, self.topology, self.permutations, self.symmetry
            ).build()

    def _place_nodes(self):
        if self.fixed_cell_closed:
            return [0.0] * len(self.topology.node_indices)
        rmsds = []
        cell = np.asarray(self.topology.atoms.cell, dtype=float)
        for orbit in self.symmetry.node_orbits:
            representative = orbit.representative
            if representative not in self.covariant_node_orbits:
                rmsds.extend(self._place_node_orbit_independently(orbit))
                continue

            representative_block, rmsd = self._located_node(representative)
            representative_block.set_centroid(self.topology.atoms.positions[representative])
            diagnostic = self.symmetry.diagnose(
                orbit, representative_block, self.permutations[representative],
                cell, self.symmetry_tolerance,
            )
            diagnostic.update({"kind": "node", "covariant": True})
            self.symmetry_diagnostics["nodes"].append(diagnostic)
            if not diagnostic["compatible"]:
                diagnostic.update(
                    {"covariant": False, "reason": "final_stabilizer_incompatible"}
                )
                rmsds.extend(self._place_node_orbit_independently(orbit, force=True))
                continue

            for raw_member in orbit.members:
                member = int(raw_member)
                operation_index = self.symmetry.operation(orbit, member)
                self.blocks[member] = (
                    representative_block
                    if member == representative
                    else self.symmetry.place(
                        representative_block, operation_index,
                        self.topology.atoms.positions[member], cell,
                    )
                )
                rmsds.append(rmsd)
        return rmsds

    def _located_node(self, node_index, *, force=False):
        block = self.blocks[node_index]
        if self.fixed_cell_closed and not force:
            return block, 0.0
        return self.locator.locate_with_permutation(
            self.topology.local_structure(node_index), block, self.permutations[node_index]
        )

    def _place_node_orbit_independently(self, orbit, *, force=False):
        rmsds = []
        for raw_member in orbit.members:
            member = int(raw_member)
            block, rmsd = self._located_node(member, force=force)
            block.set_centroid(self.topology.atoms.positions[member])
            self.blocks[member] = block
            rmsds.append(rmsd)
        return rmsds

    def _image(self, neighbor_i, neighbor_j, inverse_cell):
        """Return the periodic image offset between two edge neighbors.

        The neighbor distance vectors include the lattice translation used by
        the topology edge. Subtracting the Cartesian displacement of the two
        stored node positions and converting the remainder to fractional
        coordinates recovers that integer cell translation. Edge placement
        uses it to construct an unwrapped target across a periodic boundary.
        """
        displacement = neighbor_j.distance_vector - neighbor_i.distance_vector
        positions = self.topology.atoms.positions
        return (
            displacement
            - positions[neighbor_j.index]
            + positions[neighbor_i.index]
        ) @ inverse_cell

    def _edge_geometry(self, edge_index, cell, inverse_cell):
        neighbor_i, neighbor_j = self.topology.neighbor_list[edge_index]
        block_i = self.blocks[neighbor_i.index]
        block_j = self.blocks[neighbor_j.index]
        x_i, x_j = self.pbc.find_matched_x_sites(
            self.source_topology, self.blocks, self.permutations, edge_index
        )
        point_i = block_i.atoms.positions[x_i]
        point_j = block_j.atoms.positions[x_j]
        image = self._image(neighbor_i, neighbor_j, inverse_cell)
        vector = point_j - point_i + image @ cell
        return np.array([point_i, point_i + vector]), point_i + 0.5 * vector

    def _place_edge_independently(self, edge_index, cell, inverse_cell):
        target_points, centroid = self._edge_geometry(edge_index, cell, inverse_cell)
        target = self.edge_targets[edge_index]
        edge, _, source_frame = self.locator.locate_edge_with_frame(
            target_points,
            self.blocks[edge_index],
            self.permutations[edge_index],
            target.transverse,
        )
        edge.set_centroid(centroid)
        self.blocks[edge_index] = edge
        self.edge_frame_diagnostics.append(
            self._edge_frame_diagnostic(edge_index, target, source_frame)
        )

    @staticmethod
    def _edge_frame_diagnostic(edge_index, target, source_frame, **overrides):
        diagnostic = {
            "edge_index": int(edge_index),
            "symmetry_edge_type": target.symmetry_edge_type,
            "representative_edge": target.representative_edge,
            "operation_index": target.operation_index,
            "endpoint_reversed": target.endpoint_reversed,
            "representative_gauge": target.representative_gauge,
            "pca_gap": source_frame.pca_gap,
            "torsion_degenerate": source_frame.torsion_degenerate,
            "sign_degenerate": source_frame.sign_degenerate,
        }
        diagnostic.update(overrides)
        return diagnostic

    def _place_edges(self):
        if self.fixed_cell_closed or not self.edge_targets:
            return

        cell = np.asarray(self.topology.atoms.cell, dtype=float)
        inverse_cell = np.linalg.inv(cell)
        for orbit in self.symmetry.edge_orbits:
            representative = orbit.representative
            members = [
                int(member) for member in orbit.members
                if self.blocks[int(member)] is not None
            ]
            if not members:
                continue
            if representative not in self.covariant_edge_orbits:
                for member in members:
                    self._place_edge_independently(member, cell, inverse_cell)
                continue
            self._place_covariant_edge_orbit(orbit, members, cell, inverse_cell)

    def _place_covariant_edge_orbit(self, orbit, members, cell, inverse_cell):
        representative = orbit.representative
        target_points, centroid = self._edge_geometry(representative, cell, inverse_cell)
        target = self.edge_targets[representative]
        edge, _, source_frame = self.locator.locate_edge_with_frame(
            target_points,
            self.blocks[representative],
            self.permutations[representative],
            target.transverse,
        )
        edge.set_centroid(centroid)
        edge, diagnostic = self.symmetry.fit_torsion(
            orbit, edge, self.permutations[representative],
            cell, self.symmetry_tolerance,
        )
        diagnostic.update({"kind": "edge", "covariant": True})
        self.symmetry_diagnostics["edges"].append(diagnostic)
        if not diagnostic["compatible"]:
            diagnostic.update({"covariant": False, "reason": "stabilizer_incompatible"})
            self.blocks[representative] = edge
            self.edge_frame_diagnostics.append(
                self._edge_frame_diagnostic(
                    representative,
                    target,
                    source_frame,
                    symmetry_edge_type=orbit.tag,
                    representative_edge=representative,
                    operation_index=self.symmetry.operation(orbit, representative),
                    endpoint_reversed=False,
                )
            )
            for member in members:
                if member != representative:
                    self._place_edge_independently(member, cell, inverse_cell)
            return

        fallback_members = []
        for member in members:
            operation_index = self.symmetry.operation(orbit, member)
            member_points, member_centroid = self._edge_geometry(member, cell, inverse_cell)
            located = (
                edge
                if member == representative
                else self.symmetry.place(edge, operation_index, member_centroid, cell)
            )
            residual = float(
                np.max(
                    np.linalg.norm(
                        located.connection_points[self.permutations[member]]
                        - member_points, axis=1,
                    )
                )
            )
            if member != representative and residual > self.port_tolerance:
                fallback_members.append(member)
                self._place_edge_independently(member, cell, inverse_cell)
                continue

            self.blocks[member] = located
            endpoint_map = self.symmetry.offset_map(representative, member, operation_index)
            self.edge_frame_diagnostics.append(
                self._edge_frame_diagnostic(
                    member,
                    target,
                    source_frame,
                    symmetry_edge_type=orbit.tag,
                    representative_edge=representative,
                    operation_index=operation_index,
                    endpoint_reversed=bool(endpoint_map[0] == 1),
                )
            )

        if fallback_members:
            diagnostic.update(
                {
                    "covariant": False,
                    "reason": "propagated_port_mismatch",
                    "fallback_members": fallback_members,
                }
            )
