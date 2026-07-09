"""High-level decomposition orchestration."""

from __future__ import annotations

from pathlib import Path

import ase
from ase.geometry import cell_to_cellpar

from .assignment import SingleNodeAtomAssigner
from .bonding import read_bonded_structure
from .fragments import FragmentBuilder
from .results import DecompositionResult
from .topology import TopologyBuilder


class SingleNodePorDecomposer:
    """Decompose a MOF using MOFid SingleNode-style SBU boundaries.

    The class intentionally acts as a thin coordinator.  The details live in
    focused modules:

    - :mod:`assignment` classifies atoms as node/linker members;
    - :mod:`fragments` creates capped finite fragments;
    - :mod:`topology` collapses fragments into CGD topology records; and
    - :mod:`results` writes output files and JSON metadata.
    """

    def __init__(
        self,
        cif: str | Path,
        x_symbol: str = "X",
        bond_scale: float = 1.2,
        cleanup: bool = True,
    ) -> None:
        self.cif = Path(cif)
        self.x_symbol = x_symbol
        self.bond_scale = bond_scale
        self.cleanup = cleanup
        self.atoms, self.bonds = read_bonded_structure(
            self.cif,
            scale=self.bond_scale,
            cleanup=self.cleanup,
        )
        self.symbols = self.atoms.get_chemical_symbols()

    def decompose(self) -> DecompositionResult:
        """Run decomposition and return fragments plus topology metadata."""

        assignment = SingleNodeAtomAssigner(self.symbols, self.bonds).assign()
        node_indices = set(assignment.node_indices)
        linker_indices = set(range(len(self.atoms))) - node_indices
        connecting_bonds = find_connecting_bonds(self.bonds.pairs, node_indices)

        fragment_builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
        )
        node_fragments = fragment_builder.build_role_fragments(
            "node", node_indices, connecting_bonds
        )
        linker_fragments = fragment_builder.build_role_fragments(
            "linker", linker_indices, connecting_bonds
        )

        topology_vertices, topology_edges = TopologyBuilder(self.atoms, self.bonds).build(
            node_fragments,
            linker_fragments,
            connecting_bonds,
        )

        return DecompositionResult(
            name=self.cif.stem,
            node_indices=tuple(sorted(node_indices)),
            linker_indices=tuple(sorted(linker_indices)),
            points_of_extension=tuple(sorted(assignment.points_of_extension)),
            connecting_bonds=connecting_bonds,
            cell_parameters=tuple(float(value) for value in cell_to_cellpar(self.atoms.cell)),
            topology_vertices=topology_vertices,
            topology_edges=topology_edges,
            node_fragments=node_fragments,
            linker_fragments=linker_fragments,
        )

    def _read_atoms(self) -> ase.Atoms:
        return self.atoms


def find_connecting_bonds(
    bond_pairs: list[tuple[int, int]], node_indices: set[int]
) -> tuple[tuple[int, int], ...]:
    """Return bonds crossing the node/linker boundary."""

    crossing_pairs = (
        pair for pair in bond_pairs if (pair[0] in node_indices) != (pair[1] in node_indices)
    )
    return tuple(sorted(crossing_pairs))
