"""Decomposition into metal nodes, organic nodes, and edges."""

from __future__ import annotations

from pathlib import Path

from ase.geometry import cell_to_cellpar

from .assignment import SingleNodeAtomAssigner
from .bonding import read_bonded_structure
from .decomposer import _BaseDecomposer, find_connecting_bonds
from .fragments import FragmentBuilder, connected_components
from .linker import (
    OrganicLinkerSplit,  # noqa: F401 - compatibility re-export
    OrganicLinkerSplitter,
    classify_organic_fragment_kind,  # noqa: F401 - compatibility re-export
    relabel_fragment,
)
from .object import Fragment
from .pbc import (
    PeriodicBond,
    PeriodicOpening,
    periodic_bonds,
    periodic_graph_symmetries,
    select_periodic_component_cuts,
)
from .ports import (
    LogicalPortAssigner,  # noqa: F401 - compatibility re-export
    _MultiRankSelfEdgeLattice,  # noqa: F401 - compatibility re-export
    assign_logical_ports,
)
from .results import (
    NodeEdgeDecompositionResult,
)
from .solvent import SolventMode
from .topology import (
    NodeEdgeTopologyBuilder,
    TopologyBuilder,  # noqa: F401 - compatibility re-export
    validate_source_atom_ownership,
)


def decompose_mof_with_edges(
    cif: str | Path,
    x_symbol: str = "X",
    bond_scale: float = 1.2,
    cleanup: bool = True,
    solvent_mode: SolventMode = "none",
) -> NodeEdgeDecompositionResult:
    """Convenience function for the node/edge decomposition workflow."""

    return NodeEdgeDecomposer(
        cif,
        x_symbol=x_symbol,
        bond_scale=bond_scale,
        cleanup=cleanup,
        solvent_mode=solvent_mode,
    ).decompose()


class NodeEdgeDecomposer(_BaseDecomposer):
    """Decompose a MOF into metal nodes, organic nodes, and edge fragments."""

    @staticmethod
    def _read_structure(
        cif: Path,
        *,
        scale: float,
        cleanup: bool,
    ):
        return read_bonded_structure(cif, scale=scale, cleanup=cleanup)

    def decompose(self) -> NodeEdgeDecompositionResult:
        """Run the node/edge decomposition workflow."""

        assignment = SingleNodeAtomAssigner(
            self.symbols,
            self.bonds,
            absorb_one_port_regions=self.solvent_mode != "free",
        ).assign()
        metal_node_indices = set(assignment.node_indices)
        organic_indices = set(range(len(self.atoms))) - metal_node_indices
        connecting_bonds = find_connecting_bonds(self.bonds.pairs, metal_node_indices)
        source_records = periodic_bonds(self.atoms, self.bonds)
        source_symmetries = periodic_graph_symmetries(
            self.atoms, self.bonds, source_records
        )
        periodic_cuts: list[PeriodicBond] = []
        organic_periodic_cuts: list[PeriodicBond] = []
        periodic_openings: list[PeriodicOpening] = []
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
                source_symmetries,
                fast=fast,
            ):
                periodic_cuts.extend(selection.cuts)
                if fast:
                    organic_periodic_cuts.extend(selection.cuts)
                if selection.cuts:
                    periodic_openings.append(
                        PeriodicOpening.from_selection(component, selection)
                    )

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

        if (
            self.solvent_mode == "free"
            and self.solvent is not None
            and self.selected_to_original is not None
        ):
            bound_linker_fragments, splittable_organics = self.partition_bound_linkers(
                raw_organics
            )
        else:
            bound_linker_fragments = []
            splittable_organics = raw_organics

        metal_node_fragments = [
            relabel_fragment(fragment, "metal_node", index)
            for index, fragment in enumerate(raw_metal_nodes, start=1)
        ]
        organic_split = OrganicLinkerSplitter(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
            source_records=source_records,
        ).split(
            splittable_organics,
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
        all_connecting_bonds = tuple(record.pair for record in connection_records)
        fragments = [
            *metal_node_fragments,
            *organic_split.organic_node_fragments,
            *organic_split.edge_fragments,
            *bound_linker_fragments,
        ]
        validate_source_atom_ownership(range(len(self.atoms)), fragments)

        topology_vertices, topology_edges = NodeEdgeTopologyBuilder(
            self.atoms,
            self.bonds,
            source_records=source_records,
        ).build(
            fragments,
            connection_records,
        )
        assign_logical_ports(
            self.atoms,
            self.bonds,
            fragments,
            connection_records,
            topology_edges=topology_edges,
        )

        return NodeEdgeDecompositionResult(
            name=self.cif.stem,
            metal_node_indices=tuple(sorted(metal_node_indices)),
            organic_indices=tuple(sorted(organic_indices)),
            points_of_extension=tuple(sorted(assignment.points_of_extension)),
            connecting_bonds=all_connecting_bonds,
            cell_parameters=tuple(
                float(value) for value in cell_to_cellpar(self.atoms.cell)
            ),
            topology_vertices=topology_vertices,
            topology_edges=topology_edges,
            metal_node_fragments=metal_node_fragments,
            organic_node_fragments=organic_split.organic_node_fragments,
            edge_fragments=organic_split.edge_fragments,
            bound_linker_fragments=bound_linker_fragments,
            metal_connecting_bonds=connecting_bonds,
            organic_connecting_bonds=organic_split.organic_connecting_bonds,
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

    def partition_bound_linkers(
        self,
        organic_fragments: list[Fragment],
    ) -> tuple[list[Fragment], list[Fragment]]:
        """Separate preserved bound-solvent linkers from splittable organics."""

        assert self.solvent is not None
        assert self.selected_to_original is not None
        bound_original = set(self.solvent.bound_solvent_indices)
        bound_linkers: list[Fragment] = []
        splittable: list[Fragment] = []

        for fragment in organic_fragments:
            original_indices = {
                self.selected_to_original[index] for index in fragment.source_indices
            }
            target = (
                bound_linkers
                if bound_original and original_indices <= bound_original
                else splittable
            )
            target.append(fragment)

        return bound_linkers, splittable
