"""Assign building-block ports and initial poses to topology slots."""

from __future__ import annotations

import numpy as np

from mofbb.compose.placement.chemistry import BlockGroups, PortRules
from mofbb.compose.placement.errors import SymmetryCompatibilityError
from mofbb.compose.placement.symmetry import SymmetryPlacement


class SlotAssignment:
    """Choose compatible slot permutations and symmetry-aware initial poses."""

    def __init__(
        self,
        topology,
        blocks,
        locator,
        pbc,
        fixed=None,
        *,
        accuracy=6,
        sym_tol=0.1,
        strict_symmetry=False,
    ):
        self.topology = topology
        self.blocks = blocks
        self.locator = locator
        self.pbc = pbc
        self.accuracy = accuracy
        self.sym_tol = sym_tol
        self.strict_symmetry = bool(strict_symmetry)
        self.symmetry = SymmetryPlacement.cached(topology)
        self.cell = np.asarray(topology.atoms.cell, dtype=float)
        self.fixed = {
            int(slot): np.asarray(perm, dtype=int)
            for slot, perm in (fixed or {}).items()
        }
        self.perms = [None] * topology.n_slots
        for slot, perm in self.fixed.items():
            self.perms[slot] = perm.copy()
        self.placed = [None] * topology.n_slots
        self.symmetry_compatible_node_orbits = set()
        self.symmetry_compatible_edge_orbits = set()
        self.diag = {
            "mapping_tolerance": self.symmetry.mapping_tolerance,
            "stabilizer_tolerance": sym_tol,
            "strict": self.strict_symmetry,
            "nodes": [],
            "edges": [],
        }
        self.edge_perms = {}
        self.groups = BlockGroups(self.blocks, self.symmetry, self.fixed)
        self.rules = PortRules(
            self.topology,
            self.blocks,
            self.locator,
            self.pbc,
            self.symmetry,
            self.fixed,
            self.groups,
        )

    def run(self):
        """Populate initial blocks and permutations for every occupied slot."""
        if (
            self.strict_symmetry
            and self.topology.atoms.info.get("source_group_compatible") is False
        ):
            raise SymmetryCompatibilityError(
                "source_group_incompatible",
                kind="topology",
                representative=0,
            )
        self.edge_perms = self.rules.choose()
        self._place_node_orbits()
        self._initialize_edge_orbits()
        self._disable_incompatible_source_symmetry()
        return self

    def node_port_compatibility(self, node, edge_perms):
        """Return a target/source feasibility mask for one node slot."""
        return self.rules.node_mask(node, edge_perms)

    def _locate_node(self, node, forced=None, orbit=None):
        block = self.blocks[node]
        target = self.topology.local_structure(node)
        allowed, bad_pair = self.node_port_compatibility(
            node,
            self.edge_perms,
        )
        if forced is not None:
            perm = np.asarray(forced, dtype=int)
            if not np.all(allowed[np.arange(len(perm)), perm]):
                raise ValueError(
                    "Forbidden fused-bond chemistry rejected provided "
                    f"permutation for node slot {node}."
                )
            placed, rmsd = self.locator.locate_with_permutation(
                target,
                block,
                perm,
            )
            return placed, perm, rmsd

        placed, perm, rmsd = self.locator.locate(
            target,
            block,
            max_n_slices=max(4, self.accuracy),
            allowed=allowed,
        )
        if placed is None:
            pair = f" {bad_pair[0]}-{bad_pair[1]}" if bad_pair else ""
            raise ValueError(
                "Forbidden fused-bond chemistry rejected permutation optimization "
                f"for node slot {node}: no compatible ports{pair}."
            )
        if self.strict_symmetry and orbit is not None:
            placed, perm, rmsd = self._prefer_stabilizer_compatible_node(
                orbit,
                target,
                block,
                allowed,
                placed,
                np.asarray(perm, dtype=int),
                rmsd,
            )
        return placed, np.asarray(perm, dtype=int), rmsd

    def _prefer_stabilizer_compatible_node(
        self,
        orbit,
        target,
        block,
        allowed,
        placed,
        permutation,
        rmsd,
    ):
        """Resolve arbitrary port-symmetry ties in favor of site symmetry."""

        def compatible(candidate, candidate_permutation):
            candidate.set_centroid(
                self.topology.atoms.positions[orbit.representative]
            )
            return self.symmetry.diagnose(
                orbit,
                candidate,
                candidate_permutation,
                self.cell,
                self.sym_tol,
            )["compatible"]

        if compatible(placed.copy(), permutation):
            return placed, permutation, rmsd

        best = None
        initial = tuple(int(value) for value in permutation)
        for candidate_permutation in self.locator.candidate_permutations(
            target,
            block,
            max_n_slices=max(4, self.accuracy),
            allowed=allowed,
        ):
            key = tuple(int(value) for value in candidate_permutation)
            if key == initial:
                continue
            candidate, candidate_rmsd = self.locator.locate_with_permutation(
                target,
                block,
                candidate_permutation,
            )
            if not compatible(candidate.copy(), candidate_permutation):
                continue
            if best is None or candidate_rmsd < best[0]:
                best = (
                    float(candidate_rmsd),
                    candidate,
                    np.asarray(candidate_permutation, dtype=int),
                )
                if candidate_rmsd < 1.0e-4:
                    break
        if best is None:
            return placed, permutation, rmsd
        return best[1], best[2], best[0]

    def _place_node_orbits(self):
        for orbit in self.symmetry.node_orbits:
            rep = orbit.representative
            if not self.groups.one(orbit):
                self._place_distinct_node_templates(orbit)
                continue

            rep_perm = self.groups.perm(orbit)
            block, rep_perm, _ = self._locate_node(
                rep,
                rep_perm,
                orbit,
            )
            block.set_centroid(self.topology.atoms.positions[rep])
            diag = self.symmetry.diagnose(
                orbit,
                block,
                rep_perm,
                self.cell,
                self.sym_tol,
            )
            if not diag["compatible"]:
                self._place_incompatible_node_orbit(
                    orbit,
                    block,
                    rep_perm,
                    diag,
                )
                continue

            for raw in orbit.members:
                member = int(raw)
                self.perms[member] = self.symmetry.to_member(
                    orbit,
                    member,
                    rep_perm,
                )
                self.placed[member] = (
                    block
                    if member == rep
                    else self.symmetry.place(
                        block,
                        self.symmetry.operation(orbit, member),
                        self.topology.atoms.positions[member],
                        self.cell,
                    )
                )
            self.symmetry_compatible_node_orbits.add(rep)

    def _place_distinct_node_templates(self, orbit):
        diagnostic = {
            "tag": orbit.tag,
            "representative": orbit.representative,
            "orbit_size": len(orbit.members),
            "symmetry_propagated": False,
            "reason": "different_building_blocks",
        }
        self.diag["nodes"].append(diagnostic)
        if self.strict_symmetry:
            raise SymmetryCompatibilityError(
                diagnostic["reason"],
                kind="node",
                representative=orbit.representative,
                diagnostic=diagnostic,
            )
        self._place_nodes_independently(orbit)

    def _place_nodes_independently(self, orbit, skip=None):
        for raw in orbit.members:
            member = int(raw)
            if member == skip:
                continue
            block, perm, _ = self._locate_node(member, self.perms[member])
            block.set_centroid(self.topology.atoms.positions[member])
            self.placed[member] = block
            self.perms[member] = perm

    def _place_incompatible_node_orbit(
        self,
        orbit,
        block,
        perm,
        diag,
    ):
        rep = orbit.representative
        diag.update(
            {
                "kind": "node",
                "symmetry_propagated": False,
                "reason": "stabilizer_incompatible",
            }
        )
        self.diag["nodes"].append(diag)
        if self.strict_symmetry:
            raise SymmetryCompatibilityError(
                diag["reason"],
                kind="node",
                representative=orbit.representative,
                diagnostic=diag,
            )
        self.placed[rep] = block
        self.perms[rep] = perm
        self._place_nodes_independently(orbit, rep)

    def _initialize_edge_orbits(self):
        for orbit in self.symmetry.edge_orbits:
            rep = orbit.representative
            if all(self.blocks[int(i)] is None for i in orbit.members):
                continue
            if not self.groups.one(orbit):
                self._initialize_distinct_edge_templates(orbit)
                continue

            perm = self.groups.perm(orbit)
            if perm is None:
                perm = self.rules.edge_perm(
                    rep,
                    self.edge_perms,
                    self.placed,
                    self.perms,
                )
            block = self.blocks[rep]
            for raw in orbit.members:
                member = int(raw)
                member_perm = self.symmetry.to_member(
                    orbit,
                    member,
                    perm,
                )
                if not self.rules.edge_ok(
                    member,
                    member_perm,
                    self.placed,
                    self.perms,
                ):
                    raise ValueError(
                        "Forbidden fused-bond chemistry rejected "
                        f"symmetry-derived permutation for edge slot {member}."
                    )
                self.perms[member] = member_perm
                self.placed[member] = block.copy()
            self.symmetry_compatible_edge_orbits.add(rep)

    def _initialize_distinct_edge_templates(self, orbit):
        diagnostic = {
            "tag": orbit.tag,
            "representative": orbit.representative,
            "orbit_size": len(orbit.members),
            "symmetry_propagated": False,
            "reason": "different_building_blocks_or_occupancy",
        }
        self.diag["edges"].append(diagnostic)
        if self.strict_symmetry:
            raise SymmetryCompatibilityError(
                diagnostic["reason"],
                kind="edge",
                representative=orbit.representative,
                diagnostic=diagnostic,
            )
        for raw in orbit.members:
            member = int(raw)
            block = self.blocks[member]
            if block is None:
                continue
            perm = self.perms[member]
            if perm is None:
                perm = self.rules.edge_perm(
                    member,
                    self.edge_perms,
                    self.placed,
                    self.perms,
                )
            elif not self.rules.edge_ok(
                member,
                perm,
                self.placed,
                self.perms,
            ):
                raise ValueError(
                    "Forbidden fused-bond chemistry rejected provided "
                    f"permutation for edge slot {member}."
                )
            self.perms[member] = perm
            self.placed[member] = block

    def _disable_incompatible_source_symmetry(self):
        central = all(
            np.array_equal(rot, np.eye(3)) or np.array_equal(rot, -np.eye(3))
            for rot in self.symmetry.rotations
        )
        if (
            self.topology.atoms.info.get("source_group_compatible") is False
            and not central
        ):
            self.symmetry_compatible_node_orbits.clear()
            self.symmetry_compatible_edge_orbits.clear()

    def validate_connections(self, placed, perms, scale):
        """Reject assigned ports that predict forbidden or overlong bonds."""
        self.rules.validate(placed, perms, scale)
