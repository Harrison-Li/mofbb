"""Final symmetry-aware placement of building blocks in a composed cell."""

from __future__ import annotations

import numpy as np

from mofbb.compose.placement.errors import SymmetryCompatibilityError


class BBPlacement:
    """Store joint-fit state and place linkers in the optimized lattice."""

    def __init__(self, assignment, edge_factory, *, port_tol, bond_scale):
        self.assignment = assignment
        self.source_topology = assignment.topology
        self.topology = assignment.topology
        self.blocks = assignment.placed
        self.perms = assignment.perms
        self.symmetry = assignment.symmetry
        self.locator = assignment.locator
        self.pbc = assignment.pbc
        self.edge_factory = edge_factory
        self.symmetry_compatible_node_orbits = (
            assignment.symmetry_compatible_node_orbits
        )
        self.symmetry_compatible_edge_orbits = (
            assignment.symmetry_compatible_edge_orbits
        )
        self.symmetry_diagnostics = assignment.diag
        self.port_tol = port_tol
        self.bond_scale = bond_scale
        self.sym_tol = assignment.sym_tol
        self.strict_symmetry = assignment.strict_symmetry
        self.edge_targets = {}
        self.edge_frame_diagnostics = []

    def run(self):
        """Place all blocks in the final cell and return this stage."""
        self.edge_shifts = self.topology.edge_shifts
        self.edge_frame_diagnostics = []
        if any(self.blocks[int(i)] is not None for i in self.topology.edge_idx):
            self.edge_targets = self.edge_factory(
                self.source_topology, self.topology, self.perms, self.symmetry
            ).build()
        self._place_edges()
        return self

    def _reject_symmetry(self, reason, kind, orbit, diagnostic=None, member=None):
        raise SymmetryCompatibilityError(
            reason,
            kind=kind,
            representative=orbit.representative,
            member=member,
            diagnostic=diagnostic,
        )

    def _edge_geometry(self, edge_index, cell):
        neighbor_i, neighbor_j = self.topology.neighbor_list[edge_index]
        block_i = self.blocks[neighbor_i.index]
        block_j = self.blocks[neighbor_j.index]
        x_i, x_j = self.pbc.find_matched_x_sites(
            self.source_topology, self.blocks, self.perms, edge_index
        )
        point_i = block_i.atoms.positions[x_i]
        point_j = block_j.atoms.positions[x_j]
        points = np.array([point_i, point_j]) + self.edge_shifts[edge_index] @ cell
        return points, np.mean(points, axis=0)

    def _place_edge_independently(self, edge_index, cell):
        target_points, centroid = self._edge_geometry(edge_index, cell)
        target = self.edge_targets[edge_index]
        edge, _, source_frame = self.locator.locate_edge_with_frame(
            target_points,
            self.blocks[edge_index],
            self.perms[edge_index],
            target.transverse,
        )
        edge.set_centroid(centroid)
        self.blocks[edge_index] = edge
        self.edge_frame_diagnostics.append(
            self._edge_frame_diagnostic(edge_index, target, source_frame)
        )

    @staticmethod
    def _edge_frame_diagnostic(edge_index, target, source_frame, **overrides):
        """Convert NumPy scalars only when preparing serializable diagnostics."""
        diagnostic = {
            "edge_index": edge_index,
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
        return {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in diagnostic.items()
        }

    def _place_edges(self):
        if not self.edge_targets:
            return

        cell = np.asarray(self.topology.atoms.cell, dtype=float)
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
                    self._place_edge_independently(member, cell)
                continue
            self._place_edge_orbit_by_symmetry(orbit, members, cell)

    def _place_edge_orbit_by_symmetry(self, orbit, members, cell):
        representative = orbit.representative
        target_points, centroid = self._edge_geometry(representative, cell)
        target = self.edge_targets[representative]
        edge, _, source_frame = self.locator.locate_edge_with_frame(
            target_points,
            self.blocks[representative],
            self.perms[representative],
            target.transverse,
        )
        edge.set_centroid(centroid)
        edge, diagnostic = self.symmetry.fit_torsion(
            orbit, edge, self.perms[representative],
            cell, self.sym_tol,
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
                    self._place_edge_independently(member, cell)
            return

        fallback_members = []
        for member in members:
            operation_index = self.symmetry.operation(orbit, member)
            member_points, member_centroid = self._edge_geometry(member, cell)
            located = (
                edge
                if member == representative
                else self.symmetry.place(edge, operation_index, member_centroid, cell)
            )
            residual = float(
                np.max(
                    np.linalg.norm(
                        located.port_pos[self.perms[member]]
                        - member_points, axis=1,
                    )
                )
            )
            if member != representative and residual > self.port_tol:
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
                self._place_edge_independently(member, cell)
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
