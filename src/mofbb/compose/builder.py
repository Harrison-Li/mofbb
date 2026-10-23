"""Framework composition pipeline."""

import numpy as np
from pymatgen.core import Element

from mofbb.compose.chemistry import (
    add_fused_coordination,
    nonmetal_neighbor_counts,
    validate_coordination,
)
from mofbb.compose.edge import Edge
from mofbb.compose.framework import Framework
from mofbb.compose.locator import Locator
from mofbb.compose.pbc import PBC
from mofbb.compose.placement.blocks import BBPlacement
from mofbb.compose.placement.slots import SlotAssignment
from mofbb.compose.scaler import Scaler


class Builder:
    """Assemble building blocks on a topology."""

    def __init__(self, locator=None):
        """Store placement helpers, creating defaults when needed."""
        self.locator = locator if locator is not None else Locator()
        self.scaler = Scaler()

    def expand_bbs(self, topology, node_bbs, edge_bbs=None):
        """Expand type-keyed blocks into independent per-slot copies."""
        bbs = [None] * topology.n_slots

        for i in topology.node_idx:
            t = topology.node_types[i]
            bbs[i] = node_bbs[t].copy()

        edge_bbs = edge_bbs or {}

        for i in topology.edge_idx:
            t = tuple(topology.edge_types[i])
            edge_block = edge_bbs.get(t)
            if edge_block is not None:
                bbs[i] = edge_block.copy()

        return bbs

    def build_by_type(
        self,
        topology,
        node_bbs,
        edge_bbs=None,
        perms=None,
        *,
        n_angles=6,
        wrap=True,
        port_tol=PBC.port_tol,
        bond_scale=PBC.bond_scale,
        overlap_scale=PBC.overlap_scale,
        sym_tol=0.1,
        strict_symmetry=False,
        require_3d=True,
    ):
        """Build from node- and edge-type maps; see :meth:`build` for options."""
        bbs = self.expand_bbs(topology, node_bbs, edge_bbs)
        return self.build(
            topology,
            bbs,
            perms,
            n_angles=n_angles,
            wrap=wrap,
            port_tol=port_tol,
            bond_scale=bond_scale,
            overlap_scale=overlap_scale,
            sym_tol=sym_tol,
            strict_symmetry=strict_symmetry,
            require_3d=require_3d,
        )

    def build(
        self,
        topology,
        bbs,
        perms=None,
        *,
        n_angles=6,
        wrap=True,
        port_tol=PBC.port_tol,
        bond_scale=PBC.bond_scale,
        overlap_scale=PBC.overlap_scale,
        sym_tol=0.1,
        strict_symmetry=False,
        require_3d=True,
    ):
        """Assign ports, jointly fit rigid geometry, and assemble the framework.

        ``perms`` maps slots to port assignments: ``perm[target] = source``.
        ``n_angles`` samples each Euler-angle axis, with a minimum of four;
        blocks with at most five ports use exhaustive permutations instead.
        ``port_tol`` is the maximum paired-port gap in angstroms.
        ``bond_scale`` multiplies summed covalent radii for bond cutoffs;
        ``sym_tol`` is the site-symmetry compatibility tolerance in angstroms.
        """
        if require_3d:
            topology.validate_3d()
        if len(bbs) != topology.n_slots:
            raise ValueError("Provide one building-block entry per topology slot.")
        for slot, block in enumerate(bbs):
            if block is None:
                if slot in topology.node_idx:
                    raise ValueError(f"Topology node {slot} requires a building block.")
                continue
            if block.n_ports != topology.cn[slot]:
                raise ValueError(f"Building block at slot {slot} has the wrong number of X ports.")
            if require_3d:
                block.validate_connectivity(bond_scale)
            else:
                block.validate_chemistry()
        if perms is None:
            perms = topology.atoms.info.get("port_mappings")
        for slot, perm in (perms or {}).items():
            if not np.array_equal(np.sort(perm), np.arange(topology.cn[slot])):
                raise ValueError(f"Port assignment at slot {slot} must use each X port once.")
        assignment = SlotAssignment(
            topology,
            bbs,
            self.locator,
            PBC(),
            perms,
            n_angles=n_angles,
            sym_tol=sym_tol,
            strict_symmetry=strict_symmetry,
        )
        placement = BBPlacement(
            assignment, Edge, port_tol=port_tol, bond_scale=bond_scale,
        )
        self.scaler.optimize(placement)
        symmetry_diagnostics = placement.symmetry_diagnostics
        symmetry_diagnostics["nodes"].sort(
            key=lambda item: item["representative"]
        )
        symmetry_diagnostics["edges"].sort(
            key=lambda item: item["representative"]
        )
        info = {
            "topology": placement.topology,
            "located_bbs": placement.blocks,
            "permutations": placement.perms,
            "geometry_optimization_loss": (
                placement.geometry_optimization_result.fun
            ),
            "max_rmsd": np.max(placement.rmsds),
            "mean_rmsd": np.mean(placement.rmsds),
            "tolerance": port_tol,
            "scale": bond_scale,
            "overlap_scale": overlap_scale,
            "permutation_optimization_steps": placement.permutation_optimization_steps,
            "edge_frame_diagnostics": placement.edge_frame_diagnostics,
            "symmetry_placement_diagnostics": symmetry_diagnostics,
            "strict_symmetry": bool(strict_symmetry),
            "require_3d": bool(require_3d),
        }

        return self._assemble(placement, info, wrap=wrap, overlap_scale=overlap_scale)

    def _assemble(self, placement, info, *, overlap_scale, wrap=True):
        """Return the final framework after removing all dummy X atoms."""
        counts = np.concatenate(
            [
                block.validate_chemistry().nonmetal_counts
                for block in placement.blocks if block is not None
            ]
        )
        sizes = [0 if block is None else block.n_atoms for block in placement.blocks]
        offsets = np.cumsum([0, *sizes[:-1]], dtype=int)
        block_bonds = np.concatenate(
            [
                block.bonds + offset
                for offset, block in zip(offsets, placement.blocks, strict=True)
                if block is not None
            ],
            axis=0,
        )
        block_bond_types = [
            bond_type
            for block in placement.blocks
            if block is not None
            for bond_type in block.bond_types
        ]
        connection_bonds, connection_shifts = self._connection_bonds(placement, offsets)

        block_atoms = [block.atoms for block in placement.blocks if block is not None]
        atoms = sum(block_atoms[1:], block_atoms[0].copy())
        atoms.set_pbc(True)
        atoms.set_cell(placement.topology.atoms.cell)
        self._validate_x_pairs(
            atoms, connection_bonds, connection_shifts,
            pbc=placement.pbc, port_tol=placement.port_tol,
        )
        bonds, bond_types, bond_shifts = self._fuse_x_atoms(
            atoms,
            block_bonds,
            block_bond_types,
            connection_bonds,
            connection_shifts,
            pbc=placement.pbc,
            bond_scale=placement.bond_scale,
            counts=counts,
        )
        self._validate_overlaps(atoms, pbc=placement.pbc, overlap_scale=overlap_scale)
        framework = Framework(
            atoms, bonds, bond_types, info=info, wrap=wrap, bond_shifts=bond_shifts
        )
        if info.get("require_3d", False):
            framework.validate_3d()
        return framework

    @staticmethod
    def _connection_bonds(placement, offsets):
        """Realize every topology edge in its declared periodic images."""
        bonds, shifts = [], []
        edge_shifts = placement.topology.edge_shifts
        for edge_index in placement.topology.edge_idx:
            x_i, x_j = placement.pbc.find_matched_x_sites(
                placement.source_topology,
                placement.blocks,
                placement.perms,
                edge_index,
            )
            neighbor_i, neighbor_j = placement.topology.neighbor_list[edge_index]
            x_i += offsets[neighbor_i.index]
            x_j += offsets[neighbor_j.index]

            edge = placement.blocks[edge_index]
            if edge is None:
                bonds.append((x_i, x_j))
                shifts.append(edge_shifts[edge_index][1] - edge_shifts[edge_index][0])
                continue

            edge_x = (
                edge.port_idx[placement.perms[edge_index]]
                + offsets[edge_index]
            )
            bonds.extend(((edge_x[0], x_i), (edge_x[1], x_j)))
            shifts.extend(edge_shifts[edge_index])

        return bonds, np.asarray(shifts, dtype=int).reshape(-1, 3)

    @staticmethod
    def _validate_x_pairs(atoms, x_pairs, shifts, *, pbc, port_tol):
        failures = pbc.calc_x_gaps(atoms, x_pairs, port_tol, shifts=shifts)
        if failures:
            x_i, x_j, distance = failures[0]
            raise ValueError(
                "Geometric quality gate rejected assembly: "
                f"X atoms {x_i}-{x_j} are {distance:.3f} Å apart."
            )

    @staticmethod
    def _fuse_x_atoms(
        atoms, bonds, bond_types, x_pairs, x_shifts, *, pbc, bond_scale, counts=None
    ):
        symbols = np.asarray(atoms.get_chemical_symbols())
        is_x = symbols == "X"
        if counts is None:
            counts = nonmetal_neighbor_counts(atoms.numbers, bonds)
            validate_coordination(atoms.numbers, counts, context="building blocks")
        new_indices = np.full(len(atoms), -1, dtype=int)
        new_indices[~is_x] = np.arange(np.count_nonzero(~is_x))

        x_neighbors = {}
        fused_bonds = []
        fused_bond_types = []
        for (atom_i, atom_j), bond_type in zip(bonds, bond_types, strict=True):
            if is_x[atom_i] and is_x[atom_j]:
                raise ValueError("Each X port must have exactly one real connection atom.")
            if is_x[atom_i]:
                x_neighbors.setdefault(atom_i, []).append(atom_j)
            elif is_x[atom_j]:
                x_neighbors.setdefault(atom_j, []).append(atom_i)
            else:
                fused_bonds.append((atom_i, atom_j))
                fused_bond_types.append(bond_type)

        # Internal bonds belong to finite, unwrapped building blocks.
        fused_shifts = [(0, 0, 0)] * len(fused_bonds)
        consumed = set()
        affected = set()
        for (x_i, x_j), shift in zip(x_pairs, x_shifts, strict=True):
            if (
                x_i == x_j
                or x_i in consumed
                or x_j in consumed
                or x_i not in x_neighbors
                or x_j not in x_neighbors
            ):
                raise ValueError("Fusion must use each connected X port exactly once.")
            consumed.update((x_i, x_j))
            left, right = x_neighbors[x_i], x_neighbors[x_j]
            if len(left) > 1 or len(right) > 1:
                pair, single = (left, right) if len(left) > 1 else (right, left)
                if (len(pair) != 2 or len(single) != 1
                        or any(symbols[i] != "O" for i in pair)
                        or not Element(symbols[single[0]]).is_metal):
                    raise ValueError("A paired oxygen port must join one metal connection atom.")
            for atom_i in left:
                for atom_j in right:
                    symbol_i, symbol_j = symbols[atom_i], symbols[atom_j]
                    if pbc.is_forbidden_fused_pair(symbol_i, symbol_j):
                        raise ValueError(
                            "Forbidden fused-bond chemistry rejected assembly: "
                            f"new {symbol_i}-{symbol_j} bond."
                        )
                    fused_bonds.append((atom_i, atom_j))
                    fused_bond_types.append("S")
                    fused_shifts.append(shift)
                    add_fused_coordination(atoms.numbers, counts, atom_i, atom_j)
                    affected.update((atom_i, atom_j))

        if consumed != set(np.flatnonzero(is_x)):
            raise ValueError("Fusion must use each connected X port exactly once.")
        validate_coordination(
            atoms.numbers, counts, context="assembly", indices=sorted(affected)
        )

        failures = pbc.calc_fail_connection(
            atoms,
            x_pairs,
            x_neighbors,
            bond_scale,
            shifts=x_shifts,
        )
        if failures:
            atom_i, atom_j, distance, limit = failures[0]
            raise ValueError(
                "Fused-bond quality gate rejected assembly: "
                f"atoms {atom_i}-{atom_j} are {distance:.3f} Å apart; "
                f"limit {limit:.3f} Å."
            )

        fused_bonds = new_indices[np.asarray(fused_bonds, dtype=int).reshape(-1, 2)]
        fused_shifts = np.asarray(fused_shifts, dtype=int).reshape(-1, 3)
        del atoms[is_x.tolist()]
        return fused_bonds, fused_bond_types, fused_shifts

    @staticmethod
    def _validate_overlaps(atoms, *, pbc, overlap_scale):
        overlaps = pbc.calc_atomic_overlaps(atoms, scale=overlap_scale)
        if overlaps:
            atom_i, atom_j, distance, limit = overlaps[0]
            raise ValueError(
                "Atomic-overlap quality gate rejected assembly: "
                f"atoms {atom_i}-{atom_j} are {distance:.3f} Å apart; "
                f"minimum {limit:.3f} Å."
            )
