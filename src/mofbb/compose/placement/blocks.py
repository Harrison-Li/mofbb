"""Final symmetry-aware placement of building blocks in a composed cell."""

from __future__ import annotations

import numpy as np

from mofbb.compose.placement.errors import SymmetryCompatibilityError


class BlockPlacement:
    """Optimize framework geometry, then place blocks in the final lattice."""

    def __init__(self, assignment, fixed_cell_fit, geometry_optimizer, edge_factory):
        self.source_topology = assignment.topology
        self.topology = assignment.topology
        self.blocks = fixed_cell_fit.blocks
        self.permutations = fixed_cell_fit.permutations
        self.symmetry = assignment.symmetry
        self.locator = assignment.locator
        self.geometry_optimizer = geometry_optimizer
        self.pbc = assignment.pbc
        self.edge_factory = edge_factory
        self.symmetry_compatible_node_orbits = (
            assignment.symmetry_compatible_node_orbits
        )
        self.symmetry_compatible_edge_orbits = (
            assignment.symmetry_compatible_edge_orbits
        )
        self.symmetry_diagnostics = assignment.diag
        self.fixed_cell_fitted = fixed_cell_fit.fitted
        self.geometry_optimization_result = fixed_cell_fit.optimization_result
        self.port_tolerance = fixed_cell_fit.tolerance
        self.bond_scale = fixed_cell_fit.bond_scale
        self.symmetry_tolerance = assignment.sym_tol
        self.strict_symmetry = assignment.strict_symmetry
        self.edge_targets = {}
        self.edge_frame_diagnostics = []

    def run(self):
        """Place all blocks in the final cell and return this stage."""
        self._prepare_topology()
        if self.fixed_cell_fitted and self.strict_symmetry:
            self._validate_fixed_cell_symmetry()
        self.rmsds = np.asarray(self._place_nodes())
        self._place_edges()
        return self

    def _prepare_topology(self):
        if self.fixed_cell_fitted:
            self.topology = self.topology.copy()
        else:
            optimize = getattr(self.geometry_optimizer, "optimize", None)
            if optimize is None:
                optimize = self.geometry_optimizer.scale
            self.topology, self.geometry_optimization_result = optimize(
                topology=self.topology,
                bbs=self.blocks,
                perms=self.permutations,
                return_result=True,
            )

        has_edges = any(self.blocks[int(index)] is not None for index in self.topology.edge_indices)
        if has_edges and not self.fixed_cell_fitted:
            self.edge_targets = self.edge_factory(
                self.source_topology, self.topology, self.permutations, self.symmetry
            ).build()

    def _reject_symmetry(self, reason, kind, orbit, diagnostic=None, member=None):
        raise SymmetryCompatibilityError(
            reason,
            kind=kind,
            representative=orbit.representative,
            member=member,
            diagnostic=diagnostic,
        )

    def _validate_fixed_cell_symmetry(self):
        """Audit fitted fixed-cell poses that otherwise skip placement checks."""
        cell = np.asarray(self.topology.atoms.cell, dtype=float)
        orbit_groups = (
            (
                "node",
                self.symmetry.node_orbits,
                self.symmetry_compatible_node_orbits,
                self.symmetry_diagnostics["nodes"],
            ),
            (
                "edge",
                self.symmetry.edge_orbits,
                self.symmetry_compatible_edge_orbits,
                self.symmetry_diagnostics["edges"],
            ),
        )
        for kind, orbits, compatible_orbits, diagnostics in orbit_groups:
            for orbit in orbits:
                occupied = [
                    int(member)
                    for member in orbit.members
                    if self.blocks[int(member)] is not None
                ]
                if not occupied:
                    continue
                representative = orbit.representative
                if representative not in compatible_orbits:
                    self._reject_symmetry(
                        "orbit_not_symmetry_compatible",
                        kind,
                        orbit,
                    )
                diagnostic = self.symmetry.diagnose(
                    orbit,
                    self.blocks[representative],
                    self.permutations[representative],
                    cell,
                    self.symmetry_tolerance,
                )
                diagnostic.update({"kind": kind, "symmetry_propagated": True})
                diagnostics.append(diagnostic)
                if not diagnostic["compatible"]:
                    diagnostic.update(
                        {
                            "symmetry_propagated": False,
                            "reason": "final_stabilizer_incompatible",
                        }
                    )
                    self._reject_symmetry(
                        diagnostic["reason"],
                        kind,
                        orbit,
                        diagnostic,
                    )

    def _place_nodes(self):
        if self.fixed_cell_fitted:
            return [0.0] * len(self.topology.node_indices)
        rmsds = []
        cell = np.asarray(self.topology.atoms.cell, dtype=float)
        for orbit in self.symmetry.node_orbits:
            representative = orbit.representative
            if representative not in self.symmetry_compatible_node_orbits:
                if self.strict_symmetry:
                    self._reject_symmetry(
                        "orbit_not_symmetry_compatible",
                        "node",
                        orbit,
                    )
                rmsds.extend(self._place_node_orbit_independently(orbit))
                continue

            representative_block, rmsd = self._located_node(representative)
            representative_block.set_centroid(self.topology.atoms.positions[representative])
            diagnostic = self.symmetry.diagnose(
                orbit, representative_block, self.permutations[representative],
                cell, self.symmetry_tolerance,
            )
            diagnostic.update({"kind": "node", "symmetry_propagated": True})
            self.symmetry_diagnostics["nodes"].append(diagnostic)
            if not diagnostic["compatible"]:
                diagnostic.update(
                    {
                        "symmetry_propagated": False,
                        "reason": "final_stabilizer_incompatible",
                    }
                )
                if self.strict_symmetry:
                    self._reject_symmetry(
                        diagnostic["reason"],
                        "node",
                        orbit,
                        diagnostic,
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
        if self.fixed_cell_fitted and not force:
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
        """Return the periodic image offset between two edge neighbors."""
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
        if self.fixed_cell_fitted or not self.edge_targets:
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
            if representative not in self.symmetry_compatible_edge_orbits:
                if self.strict_symmetry:
                    self._reject_symmetry(
                        "orbit_not_symmetry_compatible",
                        "edge",
                        orbit,
                    )
                for member in members:
                    self._place_edge_independently(member, cell, inverse_cell)
                continue
            self._place_edge_orbit_by_symmetry(orbit, members, cell, inverse_cell)

    def _place_edge_orbit_by_symmetry(self, orbit, members, cell, inverse_cell):
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
        diagnostic.update({"kind": "edge", "symmetry_propagated": True})
        self.symmetry_diagnostics["edges"].append(diagnostic)
        if not diagnostic["compatible"]:
            diagnostic.update(
                {
                    "symmetry_propagated": False,
                    "reason": "stabilizer_incompatible",
                }
            )
            if self.strict_symmetry:
                self._reject_symmetry(
                    diagnostic["reason"],
                    "edge",
                    orbit,
                    diagnostic,
                )
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
                if self.strict_symmetry:
                    diagnostic.update(
                        {
                            "symmetry_propagated": False,
                            "reason": "propagated_port_mismatch",
                            "fallback_members": fallback_members,
                            "propagated_port_residual": residual,
                        }
                    )
                    self._reject_symmetry(
                        diagnostic["reason"],
                        "edge",
                        orbit,
                        diagnostic,
                        member,
                    )
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
                    "symmetry_propagated": False,
                    "reason": "propagated_port_mismatch",
                    "fallback_members": fallback_members,
                }
            )
