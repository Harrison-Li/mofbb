"""Node-edge decomposition orchestration."""

from __future__ import annotations

from pathlib import Path

from ase.geometry import cell_to_cellpar

from .assignment import NodeAtomAssigner
from .bonding import read_bonded_structure
from .fragments import FragmentBuilder, connected_components
from .linker import OrganicLinkerSplitter
from .object import Fragment
from .pbc import (
    PeriodicBond,
    periodic_bonds,
    select_periodic_component_cuts,
)
from .ports import LogicalPortAssigner
from .results import NodeEdgeDecompositionResult
from .solvent import MofidSolventRemover, SolventMode, SolventRemovalResult
from .topology import NodeEdgeTopologyBuilder


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

        metal_node_indices = set(
            NodeAtomAssigner(
                self.symbols,
                self.bonds,
                absorb_one_port_regions=self.solvent_mode != "free",
            ).assign()
        )
        organic_indices = set(range(len(self.atoms))) - metal_node_indices
        connecting_bonds = tuple(
            pair
            for pair in self.bonds.pairs
            if (pair[0] in metal_node_indices) != (pair[1] in metal_node_indices)
        )
        source_records = periodic_bonds(self.atoms, self.bonds)
        periodic_cuts: list[PeriodicBond] = []
        organic_periodic_cuts: list[PeriodicBond] = []
        metal_components = connected_components(metal_node_indices, self.bonds.pairs)
        organic_components = connected_components(organic_indices, self.bonds.pairs)
        for components, fast in (
            (metal_components, False),
            (organic_components, True),
        ):
            for component, selection in select_periodic_component_cuts(
                self.atoms,
                self.bonds,
                self.symbols,
                components,
                source_records,
                fast=fast,
            ):
                periodic_cuts.extend(selection.cuts)
                if fast:
                    organic_periodic_cuts.extend(selection.cuts)

        boundary_pairs = set(connecting_bonds)
        boundary_records = tuple(
            record for record in source_records if record.pair in boundary_pairs
        )
        initial_connection_records = tuple(sorted((*boundary_records, *periodic_cuts)))

        fragment_builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
            source_records=source_records,
        )
        raw_metal_nodes = fragment_builder.build_role_fragments(
            "node", metal_node_indices, initial_connection_records
        )
        raw_organics = fragment_builder.build_role_fragments(
            "linker", organic_indices, initial_connection_records
        )
        bound_linkers, raw_organics = self._partition_bound_linkers(raw_organics)

        for fragment in raw_metal_nodes:
            fragment.kind = "metal_node"
        metal_node_fragments = raw_metal_nodes
        organic_split = OrganicLinkerSplitter(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
            source_records=source_records,
        ).split(
            raw_organics,
            tuple(
                sorted(
                    (
                        *connecting_bonds,
                        *(record.pair for record in organic_periodic_cuts),
                    )
                )
            ),
        )
        organic_pairs = set(organic_split.organic_connecting_bonds)
        organic_records = tuple(
            record for record in source_records if record.pair in organic_pairs
        )
        connection_records = tuple(
            sorted((*boundary_records, *organic_records, *periodic_cuts))
        )
        fragments = [
            *metal_node_fragments,
            *organic_split.organic_node_fragments,
            *organic_split.edge_fragments,
            *bound_linkers,
        ]
        topology_vertices, topology_edges = NodeEdgeTopologyBuilder(
            self.atoms,
            self.bonds,
            source_records=source_records,
        ).build(
            fragments,
            connection_records,
        )
        LogicalPortAssigner(
            fragments,
            connection_records,
            topology_edges,
        ).assign()

        return NodeEdgeDecompositionResult(
            name=self.cif.stem,
            cell_parameters=tuple(
                float(value) for value in cell_to_cellpar(self.atoms.cell)
            ),
            topology_vertices=topology_vertices,
            topology_edges=topology_edges,
            metal_node_fragments=metal_node_fragments,
            organic_node_fragments=organic_split.organic_node_fragments,
            edge_fragments=organic_split.edge_fragments,
            bound_linker_fragments=bound_linkers,
            solvent_mode=self.solvent_mode,
            solvent=self.solvent,
            selected_to_original=self.selected_to_original,
            connection_records=connection_records,
        )

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
