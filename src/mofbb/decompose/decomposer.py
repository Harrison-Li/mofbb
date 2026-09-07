"""Node-edge decomposition orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.geometry import cell_to_cellpar

from mofbb.decompose.assignment import NodeAtomAssigner
from mofbb.decompose.bonding import read_bonded_structure
from mofbb.decompose.cleanup.terminal import TerminalAbsorber
from mofbb.decompose.fragments import FragmentBuilder, connected_components
from mofbb.decompose.linker import OrganicLinkerSplitter
from mofbb.decompose.object import Fragment
from mofbb.decompose.pbc import (
    PeriodicBond,
    periodic_bonds,
    select_periodic_component_cuts,
)
from mofbb.decompose.ports import LogicalPortAssigner
from mofbb.decompose.results import NodeEdgeDecompositionResult
from mofbb.decompose.solvent import (
    MofidSolventRemover,
    SolventMode,
    SolventRemovalResult,
)
from mofbb.decompose.symmetry import AtomicSymmetry, PeriodicBondOrbitClosure
from mofbb.decompose.topology import NodeEdgeTopologyBuilder


@dataclass
class FragmentGroups:
    """Role-specific fragments carried through topology cleanup."""

    metal: list[Fragment]
    organic: list[Fragment]
    edges: list[Fragment]
    bound: list[Fragment]

    def all(self) -> list[Fragment]:
        return [*self.metal, *self.organic, *self.edges, *self.bound]


def _symmetry_closed_records(
    selected: list[PeriodicBond],
    source_records: tuple[PeriodicBond, ...],
    rotations: np.ndarray,
    atom_maps: np.ndarray,
    atom_images: np.ndarray,
) -> list[PeriodicBond]:
    """Close selected bonds under bond-preserving source symmetry operations."""

    symmetry = AtomicSymmetry(
        "P1",
        rotations,
        np.zeros((len(rotations), 3), dtype=float),
        atom_maps,
        atom_images,
    )
    return PeriodicBondOrbitClosure(source_records, symmetry).close(selected)


class NodeEdgeDecomposer:
    """Decompose a MOF into metal nodes, organic nodes, and edge fragments."""

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
        self.solvent_mode = solvent_mode
        self.solvent: SolventRemovalResult | None = None
        self.selected_to_original: tuple[int, ...] | None = None
        self.atoms, self.bonds = read_bonded_structure(
            self.cif,
            scale=bond_scale,
            cleanup=cleanup if solvent_mode == "none" else False,
        )
        if solvent_mode != "none":
            self.solvent = MofidSolventRemover(self.atoms, self.bonds).classify()
            self.atoms, self.bonds, self.selected_to_original = self.solvent.subset(
                solvent_mode
            )
        self.symbols = self.atoms.get_chemical_symbols()

    def decompose(self) -> NodeEdgeDecompositionResult:
        """Run the node/edge decomposition workflow."""
        metal_ids, organic_ids, boundary_pairs = self._roles()
        source_bonds = periodic_bonds(self.atoms, self.bonds)
        symmetry = AtomicSymmetry.detect(self.atoms)
        cuts, organic_cuts = self._periodic_cuts(
            metal_ids,
            organic_ids,
            source_bonds,
        )
        builder, groups, records = self._build_groups(
            metal_ids,
            organic_ids,
            boundary_pairs,
            source_bonds,
            cuts,
            organic_cuts,
        )
        self._assign_topology(groups.all(), records, source_bonds)
        groups, records = self._absorb_terminal(builder, groups, records)
        vertices, edges = self._assign_topology(
            groups.all(),
            records,
            source_bonds,
        )
        return NodeEdgeDecompositionResult(
            name=self.cif.stem,
            cell_parameters=tuple(
                float(value) for value in cell_to_cellpar(self.atoms.cell)
            ),
            topology_vertices=vertices,
            topology_edges=edges,
            metal_node_fragments=groups.metal,
            organic_node_fragments=groups.organic,
            edge_fragments=groups.edges,
            bound_linker_fragments=groups.bound,
            solvent_mode=self.solvent_mode,
            solvent=self.solvent,
            selected_to_original=self.selected_to_original,
            connection_records=records,
            assembly_spacegroup=symmetry.symbol,
            assembly_rotations=symmetry.rotations,
            assembly_translations=symmetry.translations,
            assembly_atom_maps=symmetry.atom_maps,
            assembly_atom_images=symmetry.atom_images,
        )

    def _roles(self) -> tuple[set[int], set[int], tuple[tuple[int, int], ...]]:
        metal = set(
            NodeAtomAssigner(
                self.symbols,
                self.bonds,
                absorb_one_port_regions=self.solvent_mode != "free",
            ).assign()
        )
        if self.solvent_mode == "free":
            assert self.solvent is not None
            assert self.selected_to_original is not None
            bound = set(self.solvent.bound_solvent_indices)
            metal.difference_update(
                selected
                for selected, original in enumerate(self.selected_to_original)
                if original in bound
            )
        organic = set(range(len(self.atoms))) - metal
        boundary = tuple(
            pair
            for pair in self.bonds.pairs
            if (pair[0] in metal) != (pair[1] in metal)
        )
        return metal, organic, boundary

    def _build_groups(
        self,
        metal: set[int],
        organic: set[int],
        boundary_pairs: tuple[tuple[int, int], ...],
        source: tuple[PeriodicBond, ...],
        cuts: list[PeriodicBond],
        organic_cuts: list[PeriodicBond],
    ) -> tuple[FragmentBuilder, FragmentGroups, tuple[PeriodicBond, ...]]:
        boundary_ids = set(boundary_pairs)
        boundary = tuple(bond for bond in source if bond.pair in boundary_ids)
        initial = tuple(sorted((*boundary, *cuts)))
        builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
            source_records=source,
        )
        metals = builder.build_role_fragments("node", metal, initial)
        linkers = builder.build_role_fragments("linker", organic, initial)
        bound, linkers = self._partition_bound_linkers(linkers)
        for fragment in metals:
            fragment.kind = "metal_node"

        split = OrganicLinkerSplitter(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
            source_records=source,
        ).split(
            linkers,
            tuple(
                sorted(
                    (
                        *boundary_pairs,
                        *(bond.pair for bond in organic_cuts),
                    )
                )
            ),
        )
        organic_pairs = set(split.organic_connecting_bonds)
        organic_bonds = tuple(
            bond for bond in source if bond.pair in organic_pairs
        )
        records = tuple(sorted((*boundary, *organic_bonds, *cuts)))
        groups = FragmentGroups(
            metals,
            split.organic_node_fragments,
            split.edge_fragments,
            bound,
        )
        return builder, groups, records

    @staticmethod
    def _absorb_terminal(
        builder: FragmentBuilder,
        groups: FragmentGroups,
        records: tuple[PeriodicBond, ...],
    ) -> tuple[FragmentGroups, tuple[PeriodicBond, ...]]:
        metal, organic, edges, records = TerminalAbsorber(
            builder,
            groups.metal,
            groups.organic,
            groups.edges,
            groups.bound,
            records,
        ).run()
        return FragmentGroups(metal, organic, edges, groups.bound), records

    def _periodic_cuts(
        self,
        metal_indices: set[int],
        organic_indices: set[int],
        source_records: tuple[PeriodicBond, ...],
    ) -> tuple[list[PeriodicBond], list[PeriodicBond]]:
        periodic_cuts: list[PeriodicBond] = []
        organic_periodic_cuts: list[PeriodicBond] = []
        component_groups = (
            (connected_components(metal_indices, self.bonds.pairs), False),
            (connected_components(organic_indices, self.bonds.pairs), True),
        )
        for components, fast in component_groups:
            selections = select_periodic_component_cuts(
                self.atoms,
                self.bonds,
                self.symbols,
                components,
                source_records,
                fast=fast,
            )
            for _, selection in selections:
                periodic_cuts.extend(selection.cuts)
                if fast:
                    organic_periodic_cuts.extend(selection.cuts)
        # Symmetry-closing this seam can split periodic rods into small pieces.
        return sorted(periodic_cuts), sorted(organic_periodic_cuts)

    def _assign_topology(
        self,
        fragments: list[Fragment],
        connection_records: tuple[PeriodicBond, ...],
        source_records: tuple[PeriodicBond, ...],
    ):
        vertices, edges = NodeEdgeTopologyBuilder(
            self.atoms,
            self.bonds,
            source_records=source_records,
        ).build(fragments, connection_records)
        LogicalPortAssigner(
            fragments,
            connection_records,
            edges,
        ).assign()
        return vertices, edges

    def _partition_bound_linkers(
        self,
        fragments: list[Fragment],
    ) -> tuple[list[Fragment], list[Fragment]]:
        if self.solvent_mode != "free":
            return [], fragments

        assert self.solvent is not None
        assert self.selected_to_original is not None
        bound = set(self.solvent.bound_solvent_indices)
        mapping = self.selected_to_original
        accessory = []
        framework = []
        for fragment in fragments:
            original = {mapping[index] for index in fragment.source_indices}
            (accessory if original <= bound else framework).append(fragment)
        return accessory, framework
