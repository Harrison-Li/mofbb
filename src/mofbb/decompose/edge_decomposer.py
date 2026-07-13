"""Decomposition into metal nodes, organic nodes, and edges."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from math import gcd
from pathlib import Path
from typing import Literal

import ase
from ase.geometry import cell_to_cellpar
import networkx as nx

from .assignment import SingleNodeAtomAssigner
from .bonding import BondNetwork, read_bonded_structure
from .decomposer import find_connecting_bonds
from .fragments import FragmentBuilder, connected_components
from .geometry import format_float
from .models import Fragment, LogicalPort, TopologyEdge, TopologyVertex
from .periodicity import (
    PeriodicBond,
    PeriodicOpening,
    periodic_bonds,
    select_periodic_cuts,
)
from .results import (
    fragment_group_key,
    fragment_group_metadata,
    fragment_instance_path,
    write_reconstruction_data,
)
from .solvent import MofidSolventRemover, SolventMode, SolventRemovalResult
from .topology import ConnectionInput, TopologyBuilder, validate_source_atom_ownership

OrganicFragmentKind = Literal["organic_node", "edge"]


@dataclass
class OrganicLinkerSplit:
    """Fragments and internal cut bonds from splitting organic linkers."""

    organic_node_fragments: list[Fragment] = field(default_factory=list)
    edge_fragments: list[Fragment] = field(default_factory=list)
    organic_connecting_bonds: tuple[tuple[int, int], ...] = ()


def classify_organic_fragment_kind(fragment: Fragment) -> OrganicFragmentKind:
    """Classify an organic fragment from its dummy connection-site count."""

    num_connection_sites = len(fragment.x_sites)
    if num_connection_sites > 2:
        return "organic_node"
    if num_connection_sites == 2:
        return "edge"
    raise ValueError(
        f"Organic fragment {fragment.index} has fewer than 2 connection sites "
        f"({num_connection_sites}); it cannot be classified as an edge or node."
    )


def _partition_bound_linkers(
    organic_fragments: list[Fragment],
    selected_to_original: tuple[int, ...],
    bound_solvent_indices: tuple[int, ...],
) -> tuple[list[Fragment], list[Fragment]]:
    """Partition FSR organics using whole-fragment original-index containment.

    MOFid bound-solvent degree counts unique collapsed site connections after
    endpoint/translation deduplication, not atom-level boundary bonds. An
    accessory linker may therefore retain zero, one, or many X sites; its
    preserved ``linker`` role comes from containment, never from X-site count.
    """

    bound_original = set(bound_solvent_indices)
    accessories: list[Fragment] = []
    splittable: list[Fragment] = []
    for fragment in organic_fragments:
        original_indices = {
            selected_to_original[index] for index in fragment.source_indices
        }
        if bound_original and original_indices <= bound_original:
            accessories.append(fragment)
        else:
            splittable.append(fragment)
    return accessories, splittable


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

    def decompose(self) -> NodeEdgeDecompositionResult:
        """Run the node/edge decomposition workflow."""

        assignment = SingleNodeAtomAssigner(self.symbols, self.bonds).assign()
        metal_node_indices = set(assignment.node_indices)
        organic_indices = set(range(len(self.atoms))) - metal_node_indices
        connecting_bonds = find_connecting_bonds(self.bonds.pairs, metal_node_indices)
        source_records = periodic_bonds(self.atoms, self.bonds)
        periodic_cuts = []
        periodic_openings: list[PeriodicOpening] = []
        for component in connected_components(metal_node_indices, self.bonds.pairs):
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
        initial_connection_records = tuple(
            sorted((*boundary_records, *periodic_cuts))
        )

        fragment_builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
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
            bound_linker_fragments, splittable_organics = _partition_bound_linkers(
                raw_organics,
                self.selected_to_original,
                self.solvent.bound_solvent_indices,
            )
        else:
            bound_linker_fragments = []
            splittable_organics = raw_organics

        metal_node_fragments = relabel_fragments(raw_metal_nodes, "metal_node")
        organic_split = OrganicLinkerSplitter(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
        ).split(splittable_organics, connecting_bonds)
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
        assign_logical_ports(
            self.atoms,
            self.bonds,
            fragments,
            connection_records,
        )
        validate_source_atom_ownership(range(len(self.atoms)), fragments)

        topology_vertices, topology_edges = NodeEdgeTopologyBuilder(
            self.atoms,
            self.bonds,
        ).build(
            fragments,
            connection_records,
        )

        return NodeEdgeDecompositionResult(
            name=self.cif.stem,
            metal_node_indices=tuple(sorted(metal_node_indices)),
            organic_indices=tuple(sorted(organic_indices)),
            points_of_extension=tuple(sorted(assignment.points_of_extension)),
            connecting_bonds=all_connecting_bonds,
            cell_parameters=tuple(float(value) for value in cell_to_cellpar(self.atoms.cell)),
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

    def _read_atoms(self) -> ase.Atoms:
        return self.atoms


@dataclass(frozen=True)
class _RoleComponent:
    kind: OrganicFragmentKind
    atom_indices: frozenset[int]


class OrganicLinkerSplitter:
    """Split original organic linkers into PORMAKE-style organic nodes and edges."""

    def __init__(
        self,
        atoms: ase.Atoms,
        bonds: BondNetwork,
        symbols: list[str],
        x_symbol: str = "X",
    ) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.symbols = symbols
        self.x_symbol = x_symbol

    def split(
        self,
        organic_fragments: list[Fragment],
        metal_connecting_bonds: tuple[tuple[int, int], ...],
    ) -> OrganicLinkerSplit:
        """Split each original organic linker into smaller node/edge pieces."""

        role_components: list[_RoleComponent] = []
        organic_cut_bonds: set[tuple[int, int]] = set()

        for fragment in organic_fragments:
            components = self._split_fragment_components(
                fragment,
                metal_connecting_bonds,
            )
            role_components.extend(components)
            organic_cut_bonds.update(
                self._cut_bonds_between_components(
                    [set(component.atom_indices) for component in components]
                )
            )

        all_cut_bonds = tuple(sorted((*metal_connecting_bonds, *organic_cut_bonds)))
        fragment_builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
        )

        organic_nodes: list[Fragment] = []
        edges: list[Fragment] = []
        for component in role_components:
            if component.kind == "organic_node":
                organic_nodes.append(
                    fragment_builder.materialize_fragment(
                        "organic_node",
                        len(organic_nodes) + 1,
                        set(component.atom_indices),
                        all_cut_bonds,
                    )
                )
            else:
                edges.append(
                    fragment_builder.materialize_fragment(
                        "edge",
                        len(edges) + 1,
                        set(component.atom_indices),
                        all_cut_bonds,
                    )
                )

        self._validate_split_fragments(organic_nodes, edges)
        return OrganicLinkerSplit(
            organic_node_fragments=organic_nodes,
            edge_fragments=edges,
            organic_connecting_bonds=tuple(sorted(organic_cut_bonds)),
        )

    def _split_fragment_components(
        self,
        fragment: Fragment,
        metal_connecting_bonds: tuple[tuple[int, int], ...],
    ) -> list[_RoleComponent]:
        source_atoms = set(fragment.source_indices)
        external_cut_bonds = tuple(
            cut_bond
            for cut_bond in metal_connecting_bonds
            if (cut_bond[0] in source_atoms) != (cut_bond[1] in source_atoms)
        )
        heavy_atoms = {
            atom_index
            for atom_index in source_atoms
            if self.symbols[atom_index] != "H"
        }
        heavy_graph = self._heavy_graph(heavy_atoms)
        node_heavy_atoms = self._candidate_node_heavy_atoms(
            heavy_graph,
            external_cut_bonds,
            source_atoms,
        )

        if not node_heavy_atoms:
            return [
                _RoleComponent(
                    classify_organic_fragment_kind(fragment),
                    frozenset(source_atoms),
                )
            ]

        heavy_components = self._classify_heavy_components(
            heavy_graph,
            heavy_atoms,
            node_heavy_atoms,
            external_cut_bonds,
            source_atoms,
        )
        return self._attach_hydrogens(heavy_components, source_atoms)

    def _heavy_graph(self, heavy_atoms: set[int]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(heavy_atoms)
        for i, j in self.bonds.pairs:
            if i in heavy_atoms and j in heavy_atoms:
                graph.add_edge(i, j)
        return graph

    def _candidate_node_heavy_atoms(
        self,
        heavy_graph: nx.Graph,
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> set[int]:
        node_atoms: set[int] = set()
        for block in nx.biconnected_components(heavy_graph):
            if len(block) < 3:
                continue

            port_count = self._external_cut_count(block, external_cut_bonds, source_atoms)
            for atom_index in block:
                for neighbor in heavy_graph.neighbors(atom_index):
                    if neighbor not in block:
                        port_count += 1

            if port_count > 2:
                node_atoms.update(block)
        return node_atoms

    @staticmethod
    def _external_cut_count(
        atom_indices: set[int] | frozenset[int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> int:
        return sum(
            1
            for i, j in external_cut_bonds
            if (i in atom_indices and i in source_atoms)
            or (j in atom_indices and j in source_atoms)
        )

    def _classify_heavy_components(
        self,
        heavy_graph: nx.Graph,
        heavy_atoms: set[int],
        node_heavy_atoms: set[int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> list[_RoleComponent]:
        node_components = [
            set(component)
            for component in nx.connected_components(
                heavy_graph.subgraph(node_heavy_atoms).copy()
            )
        ]
        remaining_heavy_atoms = heavy_atoms - node_heavy_atoms
        edge_components = [
            set(component)
            for component in nx.connected_components(
                heavy_graph.subgraph(remaining_heavy_atoms).copy()
            )
        ]

        absorbed: set[int] = set()
        role_components: list[_RoleComponent] = []
        node_component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(node_components)
            for atom_index in component
        }

        for component in edge_components:
            boundary_node_indices = self._boundary_node_indices(
                component,
                node_component_by_atom,
            )
            port_count = self._component_port_count(
                component,
                external_cut_bonds,
                source_atoms,
                node_component_by_atom,
            )
            is_pendant_donor = self._is_pendant_donor_component(
                component,
                source_atoms,
                node_component_by_atom,
                external_cut_bonds,
            )
            if is_pendant_donor or (
                port_count < 2 and len(boundary_node_indices) == 1
            ):
                node_components[next(iter(boundary_node_indices))].update(component)
                absorbed.update(component)

        node_component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(node_components)
            for atom_index in component
        }
        for component in node_components:
            role_components.append(
                _RoleComponent("organic_node", frozenset(component))
            )

        for component in edge_components:
            if component & absorbed:
                continue
            port_count = self._component_port_count(
                component,
                external_cut_bonds,
                source_atoms,
                node_component_by_atom,
            )
            kind: OrganicFragmentKind = "edge" if port_count == 2 else "organic_node"
            role_components.append(_RoleComponent(kind, frozenset(component)))

        return role_components

    def _boundary_node_indices(
        self,
        component: set[int],
        node_component_by_atom: dict[int, int],
    ) -> set[int]:
        boundary_node_indices: set[int] = set()
        for atom_index in component:
            for neighbor in self.bonds.adjacency[atom_index]:
                node_component_index = node_component_by_atom.get(neighbor)
                if node_component_index is not None:
                    boundary_node_indices.add(node_component_index)
        return boundary_node_indices

    def _is_pendant_donor_component(
        self,
        component: set[int],
        source_atoms: set[int],
        node_component_by_atom: dict[int, int],
        external_cut_bonds: tuple[tuple[int, int], ...],
    ) -> bool:
        """Return whether a donor-only component is pendant to one core bond."""

        donor_elements = {"O", "N", "S"}
        if not component or any(
            self.symbols[atom_index] not in donor_elements
            for atom_index in component
        ):
            return False

        core_boundary_bonds = [
            (atom_index, neighbor)
            for atom_index in component
            for neighbor in self.bonds.adjacency[atom_index]
            if neighbor in node_component_by_atom
        ]
        boundary_node_indices = {
            node_component_by_atom[neighbor]
            for _, neighbor in core_boundary_bonds
        }
        if len(core_boundary_bonds) != 1 or len(boundary_node_indices) != 1:
            return False

        external_cut_set = {
            tuple(sorted(cut_bond)) for cut_bond in external_cut_bonds
        }
        return all(
            neighbor in component
            or neighbor in node_component_by_atom
            or (
                neighbor not in source_atoms
                and tuple(sorted((atom_index, neighbor))) in external_cut_set
            )
            for atom_index in component
            for neighbor in self.bonds.adjacency[atom_index]
        )

    def _component_port_count(
        self,
        component: set[int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
        node_component_by_atom: dict[int, int],
    ) -> int:
        port_count = self._external_cut_count(component, external_cut_bonds, source_atoms)
        for atom_index in component:
            for neighbor in self.bonds.adjacency[atom_index]:
                if neighbor in node_component_by_atom:
                    port_count += 1
        return port_count

    def _attach_hydrogens(
        self,
        heavy_components: list[_RoleComponent],
        source_atoms: set[int],
    ) -> list[_RoleComponent]:
        component_by_heavy_atom = {
            atom_index: component_index
            for component_index, component in enumerate(heavy_components)
            for atom_index in component.atom_indices
        }
        expanded = [set(component.atom_indices) for component in heavy_components]

        for atom_index in sorted(source_atoms):
            if self.symbols[atom_index] != "H":
                continue
            neighboring_component = next(
                (
                    component_by_heavy_atom[neighbor]
                    for neighbor in self.bonds.adjacency[atom_index]
                    if neighbor in component_by_heavy_atom
                ),
                None,
            )
            if neighboring_component is not None:
                expanded[neighboring_component].add(atom_index)

        return [
            _RoleComponent(component.kind, frozenset(atom_indices))
            for component, atom_indices in zip(heavy_components, expanded)
        ]

    def _cut_bonds_between_components(
        self,
        components: list[set[int]],
    ) -> set[tuple[int, int]]:
        component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(components)
            for atom_index in component
        }
        cut_bonds: set[tuple[int, int]] = set()
        for i, j in self.bonds.pairs:
            i_component = component_by_atom.get(i)
            j_component = component_by_atom.get(j)
            if i_component is None or j_component is None or i_component == j_component:
                continue
            cut_bonds.add(tuple(sorted((i, j))))
        return cut_bonds

    @staticmethod
    def _validate_split_fragments(
        organic_nodes: list[Fragment],
        edges: list[Fragment],
    ) -> None:
        invalid_nodes = [
            fragment.index for fragment in organic_nodes if len(fragment.x_sites) <= 2
        ]
        invalid_edges = [
            fragment.index for fragment in edges if len(fragment.x_sites) != 2
        ]
        if invalid_nodes:
            raise ValueError(
                "Generated organic node fragments with fewer than 3 connection "
                f"sites: {invalid_nodes}"
            )
        if invalid_edges:
            raise ValueError(
                "Generated edge fragments without exactly 2 connection sites: "
                f"{invalid_edges}"
            )


def relabel_fragments(fragments: list[Fragment], kind: str) -> list[Fragment]:
    """Return fragments copied with a new kind and stable one-based indices."""

    return [
        relabel_fragment(fragment, kind, index)
        for index, fragment in enumerate(fragments, start=1)
    ]


def relabel_fragment(fragment: Fragment, kind: str, index: int) -> Fragment:
    """Return one fragment copied with a new output role label."""

    return Fragment(
        kind=kind,
        index=index,
        source_indices=fragment.source_indices,
        atoms=fragment.atoms.copy(),
        bonds=fragment.bonds,
        fingerprint=fragment.fingerprint,
        x_sites=fragment.x_sites,
        source_image_offsets=fragment.source_image_offsets,
        logical_kind=fragment.logical_kind,
        logical_ports=fragment.logical_ports,
    )


def _rank_one_self_edge_lattice(
    fragment_key: tuple[str, int],
    translations: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Return the exact rank-one lattice generated by self-edge translations."""

    first = translations[0]
    divisor = 0
    for value in first:
        divisor = gcd(divisor, abs(value))
    if divisor == 0:
        raise ValueError(
            f"Fragment {fragment_key} has a zero self-edge translation."
        )

    primitive = tuple(value // divisor for value in first)
    axis = next(index for index, value in enumerate(primitive) if value)
    if primitive[axis] < 0:
        primitive = tuple(-value for value in primitive)

    coefficients = []
    for translation in translations:
        if translation[axis] % primitive[axis] != 0:
            raise ValueError(
                f"Fragment {fragment_key} has unexpected independent self-edge "
                f"translations {sorted(translations)}."
            )
        coefficient = translation[axis] // primitive[axis]
        if tuple(coefficient * value for value in primitive) != translation:
            raise ValueError(
                f"Fragment {fragment_key} has unexpected independent self-edge "
                f"translations {sorted(translations)}."
            )
        coefficients.append(coefficient)

    period = 0
    for coefficient in coefficients:
        period = gcd(period, abs(coefficient))
    return tuple(period * value for value in primitive)


def _effective_slot_translation(
    translation: tuple[int, int, int],
    neighbor_lattice: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    """Return a stable representative modulo a neighbor's self-edge lattice."""

    if neighbor_lattice is None:
        return translation
    axis = next(index for index, value in enumerate(neighbor_lattice) if value)
    quotient = translation[axis] // neighbor_lattice[axis]
    return tuple(
        value - quotient * lattice_value
        for value, lattice_value in zip(translation, neighbor_lattice)
    )


def assign_logical_ports(
    atoms: ase.Atoms,
    bonds: BondNetwork,
    fragments: list[Fragment],
    connection_records: tuple[PeriodicBond, ...],
) -> None:
    """Assign stable logical ports using exact physical slot translations."""

    fragment_by_key: dict[tuple[str, int], Fragment] = {}
    atom_to_fragment: dict[int, tuple[str, int]] = {}
    for fragment in fragments:
        fragment_key = (fragment.kind, fragment.index)
        if fragment_key in fragment_by_key:
            raise ValueError(f"Duplicate physical fragment key {fragment_key}.")
        fragment_by_key[fragment_key] = fragment
        for atom_index in fragment.source_indices:
            if not 0 <= atom_index < len(atoms):
                raise ValueError(
                    f"Fragment {fragment_key} owns out-of-range source atom "
                    f"{atom_index}."
                )
            previous = atom_to_fragment.get(atom_index)
            if previous is not None:
                raise ValueError(
                    f"Source atom {atom_index} is owned by both {previous} and "
                    f"{fragment_key}."
                )
            atom_to_fragment[atom_index] = fragment_key

    record_by_pair: dict[tuple[int, int], PeriodicBond] = {}
    for record in connection_records:
        if record.pair in record_by_pair:
            raise ValueError(
                f"Duplicate connection record for source bond {record.pair}."
            )
        record_by_pair[record.pair] = record

    _, topology_edges = TopologyBuilder(atoms, bonds).build_fragments(
        fragments,
        connection_records,
    )
    edge_by_pair: dict[tuple[int, int], TopologyEdge] = {}
    for edge in topology_edges:
        if edge.source_cut_bond is None:
            raise ValueError("Logical-port topology edge lacks a source cut bond.")
        if edge.source_cut_bond in edge_by_pair:
            raise ValueError(
                "Logical-port topology produced duplicate edges for source bond "
                f"{edge.source_cut_bond}."
            )
        edge_by_pair[edge.source_cut_bond] = edge
    if set(edge_by_pair) != set(record_by_pair):
        raise ValueError(
            "Logical-port topology records do not match the supplied connection "
            f"records: topology={sorted(edge_by_pair)}, "
            f"supplied={sorted(record_by_pair)}."
        )

    self_translations: dict[
        tuple[str, int], list[tuple[int, int, int]]
    ] = defaultdict(list)
    for edge in topology_edges:
        begin_key = (edge.begin_kind, edge.begin_index)
        end_key = (edge.end_kind, edge.end_index)
        if begin_key == end_key:
            self_translations[begin_key].append(edge.slot_translation)
    self_lattices = {
        fragment_key: _rank_one_self_edge_lattice(fragment_key, translations)
        for fragment_key, translations in self_translations.items()
    }

    observed_sites: Counter[tuple[tuple[int, int], int, int]] = Counter()
    ports_by_fragment: dict[tuple[str, int], tuple[LogicalPort, ...]] = {}
    for fragment in fragments:
        fragment_key = (fragment.kind, fragment.index)
        grouped_x_indices: dict[
            tuple[str, int, tuple[int, int, int]], list[int]
        ] = defaultdict(list)
        seen_x_indices: set[int] = set()
        for site in fragment.x_sites:
            if site.x_index in seen_x_indices:
                raise ValueError(
                    f"Fragment {fragment_key} repeats X site index {site.x_index}."
                )
            seen_x_indices.add(site.x_index)
            if not 0 <= site.x_index < len(fragment.atoms):
                raise ValueError(
                    f"Fragment {fragment_key} X site {site.x_index} is out of range."
                )

            try:
                record = record_by_pair[site.source_cut_bond]
            except KeyError as exc:
                raise ValueError(
                    f"X site {site.x_index} on fragment {fragment_key} has no "
                    "matching connection record for source bond "
                    f"{site.source_cut_bond}."
                ) from exc
            internal = site.internal_source_index
            external = site.external_source_index
            if tuple(sorted((internal, external))) != record.pair:
                raise ValueError(
                    f"X site {site.x_index} on fragment {fragment_key} has "
                    f"endpoints {(internal, external)} inconsistent with "
                    f"connection record {record.pair}."
                )
            if site.source_cut_shift != record.shift:
                raise ValueError(
                    f"X site {site.x_index} on fragment {fragment_key} has source "
                    f"shift {site.source_cut_shift}, expected {record.shift}."
                )
            expected_image_shift = record.oriented_shift(internal, external)
            if site.external_image_shift != expected_image_shift:
                raise ValueError(
                    f"X site {site.x_index} on fragment {fragment_key} has external "
                    f"image shift {site.external_image_shift}, expected "
                    f"{expected_image_shift}."
                )

            try:
                internal_key = atom_to_fragment[internal]
                neighbor_key = atom_to_fragment[external]
            except KeyError as exc:
                raise ValueError(
                    f"X site {site.x_index} on fragment {fragment_key} references "
                    f"unowned source atom {exc.args[0]}."
                ) from exc
            if internal_key != fragment_key:
                raise ValueError(
                    f"X site {site.x_index} on fragment {fragment_key} uses "
                    f"internal source atom {internal} owned by {internal_key}."
                )
            expected_anchor = fragment.source_indices.index(internal)
            if site.anchor_index != expected_anchor:
                raise ValueError(
                    f"X site {site.x_index} on fragment {fragment_key} has anchor "
                    f"{site.anchor_index}, expected {expected_anchor}."
                )

            edge = edge_by_pair[record.pair]
            begin_key = (edge.begin_kind, edge.begin_index)
            end_key = (edge.end_kind, edge.end_index)
            if internal == record.i:
                expected_fragment_key = begin_key
                expected_neighbor_key = end_key
                translation = edge.slot_translation
            else:
                expected_fragment_key = end_key
                expected_neighbor_key = begin_key
                translation = tuple(-value for value in edge.slot_translation)
            if (
                fragment_key != expected_fragment_key
                or neighbor_key != expected_neighbor_key
            ):
                raise ValueError(
                    f"X site {site.x_index} on fragment {fragment_key} resolves to "
                    f"neighbor {neighbor_key}, inconsistent with topology edge "
                    f"{begin_key}->{end_key} for record {record.pair}."
                )

            if neighbor_key != fragment_key:
                translation = _effective_slot_translation(
                    translation,
                    self_lattices.get(neighbor_key),
                )
            group_key = (neighbor_key[0], neighbor_key[1], translation)
            grouped_x_indices[group_key].append(site.x_index)
            observed_sites[(record.pair, internal, external)] += 1

        ports_by_fragment[fragment_key] = tuple(
            LogicalPort(
                logical_port_id=logical_port_id,
                x_indices=tuple(sorted(grouped_x_indices[group_key])),
                neighbor_kind=group_key[0],
                neighbor_index=group_key[1],
                translation=group_key[2],
            )
            for logical_port_id, group_key in enumerate(
                sorted(grouped_x_indices), start=0
            )
        )

    expected_sites: Counter[tuple[tuple[int, int], int, int]] = Counter()
    for record in connection_records:
        expected_sites[(record.pair, record.i, record.j)] += 1
        expected_sites[(record.pair, record.j, record.i)] += 1
    if observed_sites != expected_sites:
        missing = sorted((expected_sites - observed_sites).elements())
        unexpected = sorted((observed_sites - expected_sites).elements())
        raise ValueError(
            "Connection records and physical X sites must match exactly once in "
            f"each direction; missing={missing}, unexpected={unexpected}."
        )

    for fragment in fragments:
        logical_ports = ports_by_fragment[(fragment.kind, fragment.index)]
        fragment.logical_ports = logical_ports
        fragment.logical_kind = fragment.kind
        if fragment.kind not in {"organic_node", "edge"}:
            continue
        if len(logical_ports) == 2:
            fragment.logical_kind = "edge"
        elif len(logical_ports) > 2:
            fragment.logical_kind = "organic_node"


class NodeEdgeTopologyBuilder:
    """Build node/edge topology through the shared exact frame path."""

    def __init__(self, atoms: ase.Atoms, bonds: BondNetwork) -> None:
        self.atoms = atoms
        self.bonds = bonds

    def build(
        self,
        fragments: list[Fragment],
        connecting_bonds: tuple[ConnectionInput, ...],
    ) -> tuple[list[TopologyVertex], list[TopologyEdge]]:
        """Return topology vertices and source cut-bond edges."""

        return TopologyBuilder(self.atoms, self.bonds).build_fragments(
            fragments,
            connecting_bonds,
            preferred_begin_kind="metal_node",
        )


@dataclass
class NodeEdgeDecompositionResult:
    """Complete decomposition into metal nodes, organic nodes, and edges."""

    name: str
    metal_node_indices: tuple[int, ...]
    organic_indices: tuple[int, ...]
    points_of_extension: tuple[int, ...]
    connecting_bonds: tuple[tuple[int, int], ...]
    cell_parameters: tuple[float, float, float, float, float, float]
    topology_vertices: list[TopologyVertex]
    topology_edges: list[TopologyEdge]
    metal_node_fragments: list[Fragment] = field(default_factory=list)
    organic_node_fragments: list[Fragment] = field(default_factory=list)
    edge_fragments: list[Fragment] = field(default_factory=list)
    metal_connecting_bonds: tuple[tuple[int, int], ...] = ()
    organic_connecting_bonds: tuple[tuple[int, int], ...] = ()
    bound_linker_fragments: list[Fragment] = field(default_factory=list)
    solvent_mode: SolventMode = "none"
    solvent: SolventRemovalResult | None = None
    selected_to_original: tuple[int, ...] | None = None
    connection_records: tuple[PeriodicBond, ...] = ()
    periodic_openings: tuple[PeriodicOpening, ...] = ()
    periodic_validation: dict[str, bool] = field(default_factory=dict)

    @property
    def fragments(self) -> list[Fragment]:
        """Return all fragments in output slot order."""

        return [
            *self.metal_node_fragments,
            *self.organic_node_fragments,
            *self.edge_fragments,
            *self.bound_linker_fragments,
        ]

    def write(self, output_dir: Path) -> dict[str, object]:
        """Write all node/edge decomposition outputs and return a summary."""

        validate_source_atom_ownership(
            (*self.metal_node_indices, *self.organic_indices), self.fragments
        )
        if self.solvent_mode == "none":
            if self.solvent is not None or self.selected_to_original is not None:
                raise ValueError(
                    "Inconsistent solvent metadata: mode 'none' requires both "
                    "the solvent result and selected-to-original mapping to be absent."
                )
        elif self.solvent_mode in {"free", "all"}:
            if self.solvent is None or self.selected_to_original is None:
                raise ValueError(
                    f"Inconsistent solvent metadata: active mode {self.solvent_mode!r} "
                    "requires both a solvent result and selected-to-original mapping."
                )
            self.solvent.validate_write_request(
                self.solvent_mode,
                self.selected_to_original,
            )
        else:
            raise ValueError(f"Unsupported solvent mode: {self.solvent_mode!r}.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for fragment in self.fragments:
            fragment.write_xyz(output_dir / fragment_instance_path(fragment))
            fragment.write_xyz(
                output_dir
                / "all_fragments"
                / f"{fragment.kind}_{fragment.index:03d}.xyz"
            )

        group_records, fragment_to_bb_id = self._write_group_representatives(output_dir)

        topology_path = output_dir / "topology.cgd"
        topology_summary = self.write_topology_cgd(topology_path)
        topology_summary["path"] = str(topology_path.relative_to(output_dir))

        reconstruction_path = output_dir / "reconstruction_map.json"
        self.write_reconstruction_map(
            reconstruction_path,
            group_records,
            fragment_to_bb_id,
        )

        summary: dict[str, object] = {
            "name": self.name,
            "decomposition": "node_edge",
            "num_metal_nodes": len(self.metal_node_fragments),
            "num_organic_nodes": len(self.organic_node_fragments),
            "num_edges": len(self.edge_fragments),
            "num_connecting_bonds": len(self.connecting_bonds),
            "num_metal_connecting_bonds": len(self.metal_connecting_bonds),
            "num_organic_connecting_bonds": len(self.organic_connecting_bonds),
            "metal_node_indices": list(self.metal_node_indices),
            "organic_indices": list(self.organic_indices),
            "points_of_extension": list(self.points_of_extension),
            "connecting_bonds": [list(pair) for pair in self.connecting_bonds],
            "metal_connecting_bonds": [
                list(pair) for pair in self.metal_connecting_bonds
            ],
            "organic_connecting_bonds": [
                list(pair) for pair in self.organic_connecting_bonds
            ],
            "topology": topology_summary,
            "reconstruction_map": str(reconstruction_path.relative_to(output_dir)),
            "groups": group_records,
            "metal_nodes": [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.metal_node_fragments
            ],
            "organic_nodes": [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.organic_node_fragments
            ],
            "edges": [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.edge_fragments
            ],
        }
        if self.connection_records:
            legacy_pairs = {
                *self.metal_connecting_bonds,
                *self.organic_connecting_bonds,
            }
            periodic_records = tuple(
                record
                for record in self.connection_records
                if record.pair not in legacy_pairs
            )
            summary["connecting_bond_records"] = [
                record.as_dict() for record in self.connection_records
            ]
            summary["periodic_components"] = [
                opening.as_dict() for opening in self.periodic_openings
            ]
            summary["periodic_validation"] = dict(self.periodic_validation)
            summary["num_periodic_connecting_bonds"] = len(periodic_records)
            summary["periodic_connecting_bonds"] = [
                list(record.pair) for record in periodic_records
            ]
            summary["periodic_connecting_bond_records"] = [
                record.as_dict() for record in periodic_records
            ]
        if self.bound_linker_fragments:
            summary["num_bound_linkers"] = len(self.bound_linker_fragments)
            summary["bound_linkers"] = [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.bound_linker_fragments
            ]
        if self.solvent is not None and self.selected_to_original is not None:
            summary["solvent"] = self.solvent.write(
                output_dir,
                self.solvent_mode,
                self.selected_to_original,
            )
        with (output_dir / "decomposition_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary

    def _write_group_representatives(
        self,
        output_dir: Path,
    ) -> tuple[list[dict[str, object]], dict[tuple[str, int], str]]:
        group_records: list[dict[str, object]] = []
        fragment_to_bb_id: dict[tuple[str, int], str] = {}
        for kind, fragments in (
            ("metal_node", self.metal_node_fragments),
            ("organic_node", self.organic_node_fragments),
            ("edge", self.edge_fragments),
            ("linker", self.bound_linker_fragments),
        ):
            grouped: dict[tuple[object, ...], list[Fragment]] = defaultdict(list)
            for fragment in fragments:
                grouped[fragment_group_key(fragment)].append(fragment)

            for group_index, (_, group) in enumerate(sorted(grouped.items()), start=1):
                representative = group[0]
                bb_id = f"{kind}_group_{group_index:02d}"
                group_path = output_dir / "bbs" / f"{kind}_group_{group_index:02d}.xyz"
                representative.write_xyz(group_path)
                for fragment in group:
                    fragment_to_bb_id[(fragment.kind, fragment.index)] = bb_id
                group_records.append(
                    {
                        "bb_id": bb_id,
                        "kind": kind,
                        "group": group_index,
                        "count": len(group),
                        "cn": len(representative.x_sites),
                        "formula": representative.formula,
                        "num_atoms": representative.num_atoms,
                        "num_x": representative.num_x,
                        **fragment_group_metadata(representative),
                        "x_sites": [
                            {"site_id": site_id, "x_index": site.x_index}
                            for site_id, site in enumerate(representative.x_sites)
                        ],
                        "fingerprint": representative.fingerprint,
                        "fragment_indices": [fragment.index for fragment in group],
                        "path": str(group_path.relative_to(output_dir)),
                    }
                )
        return group_records, fragment_to_bb_id

    def write_reconstruction_map(
        self,
        path: Path,
        group_records: list[dict[str, object]],
        fragment_to_bb_id: dict[tuple[str, int], str],
    ) -> dict[str, object]:
        """Write group-first metadata for exact reconstruction experiments."""

        return write_reconstruction_data(
            path,
            topology_kind="node_edge_slots",
            cell_parameters=self.cell_parameters,
            fragments=self.fragments,
            expected_source_atoms=(*self.metal_node_indices, *self.organic_indices),
            connecting_bonds=self.connecting_bonds,
            connection_records=self.connection_records,
            periodic_openings=self.periodic_openings,
            periodic_validation=self.periodic_validation,
            topology_edges=self.topology_edges,
            group_records=group_records,
            fragment_to_bb_id=fragment_to_bb_id,
        )

    def write_topology_cgd(self, path: Path) -> dict[str, int]:
        """Write a P1 CGD topology file for the node/edge result."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("CRYSTAL\n")
            handle.write(f"  NAME {self.name}\n")
            handle.write("  GROUP P1\n")
            cell = " ".join(format_float(value) for value in self.cell_parameters)
            handle.write(f"  CELL {cell}\n")
            for vertex_number, vertex in enumerate(self.topology_vertices, start=1):
                frac = " ".join(format_float(value) for value in vertex.frac_position)
                handle.write(f"  NODE {vertex_number} {vertex.coordination} {frac}\n")
            for edge in self.topology_edges:
                begin = " ".join(format_float(value) for value in edge.begin_frac)
                end = " ".join(format_float(value) for value in edge.end_frac)
                handle.write(f"  EDGE  {begin}   {end}\n")
            for edge in self.topology_edges:
                center = " ".join(format_float(value) for value in edge.center_frac)
                handle.write(f"  # EDGE_CENTER  {center}\n")
            handle.write("END\n")
        return {
            "num_vertices": len(self.topology_vertices),
            "num_edges": len(self.topology_edges),
        }
