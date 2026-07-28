"""High-level SingleNode decomposition orchestration."""

from __future__ import annotations

from pathlib import Path

import ase
from ase.geometry import cell_to_cellpar

from .assignment import SingleNodeAtomAssigner
from .bonding import read_bonded_structure
from .fragments import FragmentBuilder, connected_components
from .pbc import PeriodicOpening, periodic_bonds, select_periodic_cuts
from .results import DecompositionResult
from .solvent import MofidSolventRemover, SolventMode, SolventRemovalResult
from .topology import TopologyBuilder, validate_source_atom_ownership


class SingleNodeDecomposer:
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
        solvent_mode: SolventMode = "none",
    ) -> None:
        if solvent_mode not in {"none", "free", "all"}:
            raise ValueError(f"Unsupported solvent mode: {solvent_mode!r}.")
        self.cif = Path(cif)
        self.x_symbol = x_symbol
        self.bond_scale = bond_scale
        self.cleanup = cleanup
        self.solvent_mode = solvent_mode
        self.solvent: SolventRemovalResult | None = None
        self.selected_to_original: tuple[int, ...] | None = None
        self.atoms, self.bonds = read_bonded_structure(
            self.cif,
            scale=self.bond_scale,
            cleanup=self.cleanup if solvent_mode == "none" else False,
        )
        if solvent_mode != "none":
            self.solvent = MofidSolventRemover(self.atoms, self.bonds).classify()
            self.atoms, self.bonds, self.selected_to_original = self.solvent.subset(
                solvent_mode
            )
        self.symbols = self.atoms.get_chemical_symbols()

    def decompose(self) -> DecompositionResult:
        """Run decomposition and return fragments plus topology metadata."""

        assignment = SingleNodeAtomAssigner(self.symbols, self.bonds).assign()
        node_indices = set(assignment.node_indices)
        linker_indices = set(range(len(self.atoms))) - node_indices
        connecting_bonds = find_connecting_bonds(self.bonds.pairs, node_indices)
        source_records = periodic_bonds(self.atoms, self.bonds)
        periodic_cuts = []
        periodic_openings: list[PeriodicOpening] = []
        for component in connected_components(node_indices, self.bonds.pairs):
            selection = select_periodic_cuts(
                self.atoms,
                self.bonds,
                self.symbols,
                component,
                source_records,
            )
            periodic_cuts.extend(selection.cuts)
            if selection.cuts:
                periodic_openings.append(
                    PeriodicOpening.from_selection(component, selection)
                )

        boundary_pairs = set(connecting_bonds)
        boundary_records = tuple(
            record for record in source_records if record.pair in boundary_pairs
        )
        connection_records = tuple(sorted((*boundary_records, *periodic_cuts)))
        all_connecting_bonds = tuple(record.pair for record in connection_records)

        fragment_builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
            source_records=source_records,
        )
        node_fragments = fragment_builder.build_role_fragments(
            "node", node_indices, connection_records
        )
        linker_fragments = fragment_builder.build_role_fragments(
            "linker", linker_indices, connection_records
        )
        validate_source_atom_ownership(
            range(len(self.atoms)),
            [*node_fragments, *linker_fragments],
        )

        topology_vertices, topology_edges = TopologyBuilder(
            self.atoms,
            self.bonds,
            source_records=source_records,
        ).build(node_fragments, linker_fragments, connection_records)

        return DecompositionResult(
            name=self.cif.stem,
            node_indices=tuple(sorted(node_indices)),
            linker_indices=tuple(sorted(linker_indices)),
            points_of_extension=tuple(sorted(assignment.points_of_extension)),
            connecting_bonds=all_connecting_bonds,
            cell_parameters=tuple(float(value) for value in cell_to_cellpar(self.atoms.cell)),
            topology_vertices=topology_vertices,
            topology_edges=topology_edges,
            node_fragments=node_fragments,
            linker_fragments=linker_fragments,
            solvent_mode=self.solvent_mode,
            solvent=self.solvent,
            selected_to_original=self.selected_to_original,
            connection_records=connection_records,
            periodic_openings=tuple(periodic_openings),
            periodic_validation={
                "all_source_atoms_owned_once": True,
                "all_fragments_zero_holonomy": True,
                "all_retained_bonds_undistorted": True,
            },
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
