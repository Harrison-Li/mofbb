"""Assign building-block ports and initial poses to topology slots."""

from __future__ import annotations

import numpy as np

from mofbb.compose.locator import PortAssignmentError
from mofbb.compose.placement.chemistry import PortRules
from mofbb.compose.placement.symmetry import SymmetryPlacement
from mofbb.compose.placement.templates import BlockGroups


class SlotAssignment:
    """Choose compatible slot permutations and symmetry-aware initial poses.

    This is the first stateful stage of composition. It compares templates
    within each crystallographic orbit, chooses edge orientations jointly when
    endpoint chemistry matters, locates node blocks with feasible port masks,
    and propagates representative poses only across symmetry-compatible orbit
    members. Explicit caller permutations remain authoritative and are checked
    against the same chemistry constraints.

    The resulting ``placed`` blocks and ``perms`` are intentionally
    separate from final placement: the topology may still be resized or solved
    in its fixed cell. Orbit covariance sets and diagnostic records are exposed
    for those later stages.
    """

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
    ):
        self.topology = topology
        self.blocks = blocks
        self.locator = locator
        self.pbc = pbc
        self.accuracy = accuracy
        self.sym_tol = sym_tol
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
        self.node_cov = set()
        self.edge_cov = set()
        self.diag = {
            "mapping_tolerance": self.symmetry.mapping_tolerance,
            "stabilizer_tolerance": sym_tol,
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
        self.edge_perms = self.rules.choose()
        self._place_node_orbits()
        self._initialize_edge_orbits()
        self._disable_incompatible_source_symmetry()
        return self

    def node_port_compatibility(self, node, edge_perms):
        """Return a target/source feasibility mask for one node slot."""
        return self.rules.node_mask(node, edge_perms)

    def _locate_node(self, node, forced=None):
        block = self.blocks[node]
        target = self.topology.local_structure(node)
        allowed, bad_pair = self.node_port_compatibility(
            node,
            self.edge_perms,
        )
        pair = f" {bad_pair[0]}-{bad_pair[1]}" if bad_pair else ""
        message = (
            "Forbidden fused-bond chemistry rejected permutation optimization "
            f"for node slot {node}: no compatible ports{pair}."
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

        try:
            placed, perm, rmsd = self.locator.locate(
                target,
                block,
                max_n_slices=max(4, self.accuracy),
                allowed=allowed,
            )
        except PortAssignmentError as exc:
            raise ValueError(message) from exc
        return placed, np.asarray(perm, dtype=int), rmsd

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
            self.node_cov.add(rep)

    def _place_distinct_node_templates(self, orbit):
        self.diag["nodes"].append(
            {
                "tag": orbit.tag,
                "representative": orbit.representative,
                "orbit_size": len(orbit.members),
                "covariant": False,
                "reason": "different_building_blocks",
            }
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
                "covariant": False,
                "reason": "stabilizer_incompatible",
            }
        )
        self.diag["nodes"].append(diag)
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
            self.edge_cov.add(rep)

    def _initialize_distinct_edge_templates(self, orbit):
        self.diag["edges"].append(
            {
                "tag": orbit.tag,
                "representative": orbit.representative,
                "orbit_size": len(orbit.members),
                "covariant": False,
                "reason": "different_building_blocks_or_occupancy",
            }
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
            self.node_cov.clear()
            self.edge_cov.clear()

    def validate_connections(self, placed, perms, scale):
        """Reject assigned ports that predict forbidden or overlong bonds."""
        self.rules.validate(placed, perms, scale)
