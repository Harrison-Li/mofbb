"""Create a framework from building blocks that have already been placed."""

from __future__ import annotations

import numpy as np

from mofbb.compose.framework import Framework


class FrameworkAssembler:
    """Merge placed blocks, fuse their connection points, and validate geometry.

    This class owns the final, topology-independent assembly stage. It converts
    per-block atom indices to framework indices, adds the bonds implied by each
    topology edge, validates the paired dummy ``X`` sites, replaces every
    ``X-X`` connection with a bond between the corresponding real atoms, and
    performs the final overlap check.

    Parameters
    ----------
    assignment : SlotAssignment
        Source topology and matched topology ports.
    placement : BlockPlacement
        Final topology, located blocks, and closure thresholds.
    overlap_scale : float
        Minimum allowed atom separation in covalent-radius units.
    """

    def __init__(self, assignment, placement, overlap_scale):
        self.topology = placement.topology
        self.source_topology = assignment.topology
        self.blocks = placement.blocks
        self.permutations = placement.permutations
        self.pbc = assignment.pbc
        self.x_tolerance = placement.port_tolerance
        self.bond_scale = placement.bond_scale
        self.overlap_scale = overlap_scale

    def build(self, info, *, wrap=True):
        """Return the final framework after removing all dummy X atoms."""
        sizes = [0 if block is None else block.n_atoms for block in self.blocks]
        offsets = np.cumsum([0, *sizes[:-1]], dtype=int)
        block_bonds = np.concatenate(
            [
                block.bonds + offset
                for offset, block in zip(offsets, self.blocks, strict=True)
                if block is not None
            ],
            axis=0,
        )
        block_bond_types = [
            bond_type
            for block in self.blocks
            if block is not None
            for bond_type in block.bond_types
        ]
        connection_bonds, connection_bond_types = self._connection_bonds(offsets)
        all_bonds = np.concatenate(
            [block_bonds, np.asarray(connection_bonds, dtype=int)],
            axis=0,
        )
        all_bond_types = block_bond_types + connection_bond_types

        block_atoms = [block.atoms for block in self.blocks if block is not None]
        atoms = sum(block_atoms[1:], block_atoms[0])
        atoms.set_pbc(True)
        atoms.set_cell(self.topology.atoms.cell)
        self._validate_x_pairs(atoms, connection_bonds)
        bonds, bond_types = self._fuse_x_atoms(
            atoms,
            all_bonds,
            all_bond_types,
            connection_bonds,
        )
        self._validate_overlaps(atoms)
        return Framework(atoms, bonds, bond_types, info=info, wrap=wrap)

    def _connection_bonds(self, offsets):
        bonds = []
        bond_types = []
        for raw_edge_index in self.topology.edge_indices:
            edge_index = int(raw_edge_index)
            x_i, x_j = self.pbc._find_matched_x_sites(
                self.source_topology,
                self.blocks,
                self.permutations,
                edge_index,
            )
            neighbor_i, neighbor_j = self.topology.neighbor_list[edge_index]
            x_i += offsets[neighbor_i.index]
            x_j += offsets[neighbor_j.index]

            edge = self.blocks[edge_index]
            if edge is None:
                bonds.append((x_i, x_j))
                bond_types.append("S")
                continue

            edge_x = (
                edge.connection_point_indices[self.permutations[edge_index]]
                + offsets[edge_index]
            )
            bonds.extend(((edge_x[0], x_i), (edge_x[1], x_j)))
            bond_types.extend(("S", "S"))

        return bonds, bond_types

    def _validate_x_pairs(self, atoms, x_pairs):
        failures = self.pbc.calc_x_mid(atoms, x_pairs, self.x_tolerance)
        if failures:
            x_i, x_j, distance = failures[0]
            raise ValueError(
                "Geometric quality gate rejected assembly: "
                f"X atoms {x_i}-{x_j} are {distance:.3f} Å apart."
            )

    def _fuse_x_atoms(self, atoms, bonds, bond_types, x_pairs):
        symbols = np.asarray(atoms.get_chemical_symbols())
        is_x = symbols == "X"
        new_indices = np.full(len(atoms), -1, dtype=int)
        new_indices[~is_x] = np.arange(np.count_nonzero(~is_x))

        x_x_bonds = []
        x_neighbors = {}
        fused_bonds = []
        fused_bond_types = []
        for (atom_i, atom_j), bond_type in zip(bonds, bond_types, strict=True):
            atom_i = int(atom_i)
            atom_j = int(atom_j)
            if is_x[atom_i] and is_x[atom_j]:
                x_x_bonds.append((atom_i, atom_j))
            elif is_x[atom_i]:
                x_neighbors[atom_i] = atom_j
            elif is_x[atom_j]:
                x_neighbors[atom_j] = atom_i
            else:
                fused_bonds.append((atom_i, atom_j))
                fused_bond_types.append(bond_type)

        failures = self.pbc.calc_fail_connection(
            atoms,
            x_pairs,
            x_neighbors,
            self.bond_scale,
        )
        if failures:
            atom_i, atom_j, distance, limit = failures[0]
            raise ValueError(
                "Fused-bond quality gate rejected assembly: "
                f"atoms {atom_i}-{atom_j} are {distance:.3f} Å apart; "
                f"limit {limit:.3f} Å."
            )

        for x_i, x_j in x_x_bonds:
            fused_bonds.append((x_neighbors[x_i], x_neighbors[x_j]))
            fused_bond_types.append("S")

        fused_bonds = np.asarray(
            [(new_indices[i], new_indices[j]) for i, j in fused_bonds],
            dtype=int,
        )
        del atoms[is_x.tolist()]
        return fused_bonds, fused_bond_types

    def _validate_overlaps(self, atoms):
        overlaps = self.pbc.calc_atomic_overlaps(atoms, scale=self.overlap_scale)
        if overlaps:
            atom_i, atom_j, distance, limit = overlaps[0]
            raise ValueError(
                "Atomic-overlap quality gate rejected assembly: "
                f"atoms {atom_i}-{atom_j} are {distance:.3f} Å apart; "
                f"minimum {limit:.3f} Å."
            )
