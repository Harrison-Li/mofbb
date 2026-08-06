"""Organic linker splitting for node-edge decomposition."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Literal

import ase
import networkx as nx

from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.fragments import FragmentBuilder
from mofbb.decompose.object import Fragment
from mofbb.decompose.pbc import PeriodicBond

OrganicFragmentKind = Literal["organic_node", "edge"]


@dataclass
class OrganicLinkerSplit:
    """Fragments and internal cut bonds from splitting organic linkers."""

    organic_node_fragments: list[Fragment] = field(default_factory=list)
    edge_fragments: list[Fragment] = field(default_factory=list)
    organic_connecting_bonds: tuple[tuple[int, int], ...] = ()


def classify_organic_fragment_kind(fragment: Fragment) -> OrganicFragmentKind:
    return "edge" if len(fragment.x_sites) == 2 else "organic_node"


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
        source_records: tuple[PeriodicBond, ...],
        x_symbol: str = "X",
    ) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.symbols = symbols
        self.x_symbol = x_symbol
        self.source_records = source_records

    def split(
        self,
        organic_fragments: list[Fragment],
        metal_connecting_bonds: tuple[tuple[int, int], ...],
    ) -> OrganicLinkerSplit:
        role_components: list[_RoleComponent] = []
        organic_cut_bonds: set[tuple[int, int]] = set()
        external_cut_set = {
            tuple(sorted(cut_bond)) for cut_bond in metal_connecting_bonds
        }

        for fragment in organic_fragments:
            components = self.split_fragment(
                fragment,
                metal_connecting_bonds,
            )
            role_components.extend(components)
            organic_cut_bonds.update(
                self.cut_bonds_between_components(
                    [set(component.atom_indices) for component in components],
                    excluded_bonds=external_cut_set,
                )
            )

        all_cut_pairs = {*metal_connecting_bonds, *organic_cut_bonds}
        all_cut_records = tuple(
            record for record in self.source_records if record.pair in all_cut_pairs
        )
        fragment_builder = FragmentBuilder(
            self.atoms,
            self.bonds,
            self.symbols,
            x_symbol=self.x_symbol,
            source_records=self.source_records,
        )

        fragments: dict[OrganicFragmentKind, list[Fragment]] = {
            "organic_node": [],
            "edge": [],
        }
        for component in role_components:
            group = fragments[component.kind]
            group.append(
                fragment_builder.materialize_fragment(
                    component.kind,
                    len(group) + 1,
                    set(component.atom_indices),
                    all_cut_records,
                )
            )

        return OrganicLinkerSplit(
            organic_node_fragments=fragments["organic_node"],
            edge_fragments=fragments["edge"],
            organic_connecting_bonds=tuple(sorted(organic_cut_bonds)),
        )

    def split_fragment(
        self,
        fragment: Fragment,
        metal_connecting_bonds: tuple[tuple[int, int], ...],
    ) -> list[_RoleComponent]:
        source_atoms = set(fragment.source_indices)
        if fragment.num_x == 2:
            return [_RoleComponent("edge", frozenset(source_atoms))]

        external_cut_bonds = tuple(
            cut_bond
            for cut_bond in metal_connecting_bonds
            if cut_bond[0] in source_atoms or cut_bond[1] in source_atoms
        )
        heavy_atoms = {
            atom_index for atom_index in source_atoms if self.symbols[atom_index] != "H"
        }
        heavy_graph, node_heavy_atoms = self.identify_node_heavy_atoms(
            heavy_atoms,
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

        heavy_components = self.classify_heavy_components(
            heavy_graph,
            heavy_atoms,
            node_heavy_atoms,
            external_cut_bonds,
            source_atoms,
        )
        return self.attach_hydrogens(heavy_components, source_atoms)

    def identify_node_heavy_atoms(
        self,
        heavy_atoms: set[int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> tuple[nx.Graph, set[int]]:
        excluded_bonds = {tuple(sorted(cut_bond)) for cut_bond in external_cut_bonds}
        graph = nx.Graph()
        graph.add_nodes_from(heavy_atoms)
        for i, j in self.bonds.pairs:
            if (
                i in heavy_atoms
                and j in heavy_atoms
                and tuple(sorted((i, j))) not in excluded_bonds
            ):
                graph.add_edge(i, j)

        blocks = [set(block) for block in nx.biconnected_components(graph)]
        node_atoms: set[int] = set()
        for block in blocks:
            if len(block) < 3:
                continue
            port_count = self.external_cut_count(
                block,
                external_cut_bonds,
                source_atoms,
            )
            port_count += sum(
                neighbor not in block
                for atom_index in block
                for neighbor in graph.neighbors(atom_index)
            )
            if port_count > 2:
                node_atoms.update(block)

        if (
            not node_atoms
            and self.external_cut_count(
                heavy_atoms,
                external_cut_bonds,
                source_atoms,
            )
            > 2
        ):
            node_atoms = self.identify_acyclic_node_atoms(
                graph,
                blocks,
                external_cut_bonds,
                source_atoms,
            )

        return graph, node_atoms

    def identify_acyclic_node_atoms(
        self,
        graph: nx.Graph,
        blocks: list[set[int]],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> set[int]:
        bridges = {tuple(sorted(edge)) for edge in nx.bridges(graph)}
        bridge_free_graph = graph.copy()
        bridge_free_graph.remove_edges_from(bridges)
        regions = [
            set(component) for component in nx.connected_components(bridge_free_graph)
        ]
        region_by_atom = {
            atom: region_index
            for region_index, region in enumerate(regions)
            for atom in region
        }

        bridge_tree = nx.Graph()
        bridge_tree.add_nodes_from(range(len(regions)))
        bridge_tree.add_edges_from(
            (region_by_atom[i], region_by_atom[j]) for i, j in bridges
        )
        port_count: dict[int, int] = defaultdict(int)
        for i, j in external_cut_bonds:
            if i in source_atoms and i in region_by_atom:
                port_count[region_by_atom[i]] += 1
            if j in source_atoms and j in region_by_atom:
                port_count[region_by_atom[j]] += 1

        active_regions = set(bridge_tree)
        active_degree = dict(bridge_tree.degree)
        queue = deque(
            region
            for region in bridge_tree
            if active_degree[region] <= 1 and port_count[region] == 0
        )
        while queue:
            region = queue.popleft()
            if region not in active_regions or port_count[region] != 0:
                continue
            active_regions.remove(region)
            for neighbor in bridge_tree[region]:
                if neighbor not in active_regions:
                    continue
                active_degree[neighbor] -= 1
                if active_degree[neighbor] <= 1 and port_count[neighbor] == 0:
                    queue.append(neighbor)

        seed_regions = {
            region
            for region in active_regions
            if sum(neighbor in active_regions for neighbor in bridge_tree[region])
            + port_count[region]
            >= 3
        }
        if not seed_regions:
            return set()

        seed_atoms = {atom for region in seed_regions for atom in regions[region]}
        active_atoms = {atom for region in active_regions for atom in regions[region]}
        cyclic_atoms = {
            atom
            for block in blocks
            if len(block) >= 3
            for atom in block
            if atom in active_atoms and atom not in seed_atoms
        }
        if not cyclic_atoms:
            return set()

        node_distances = nx.multi_source_dijkstra_path_length(
            graph,
            seed_atoms,
        )
        cycle_distances = nx.multi_source_dijkstra_path_length(
            graph,
            cyclic_atoms,
        )
        return {
            atom
            for atom in active_atoms
            if atom in node_distances
            and atom in cycle_distances
            and node_distances[atom] <= cycle_distances[atom]
        }

    @staticmethod
    def external_cut_count(
        atom_indices: set[int] | frozenset[int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> int:
        return sum(
            int(i in atom_indices and i in source_atoms)
            + int(j in atom_indices and j in source_atoms)
            for i, j in external_cut_bonds
        )

    def classify_heavy_components(
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
            boundary_node_indices, port_count = self.component_connections(
                component,
                node_component_by_atom,
                external_cut_bonds,
                source_atoms,
            )
            is_pendant_donor = self.is_pendant_donor(
                component,
                node_component_by_atom,
                external_cut_bonds,
            )
            if is_pendant_donor or (port_count < 2 and len(boundary_node_indices) == 1):
                node_components[next(iter(boundary_node_indices))].update(component)
                absorbed.update(component)

        node_component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(node_components)
            for atom_index in component
        }
        for component in node_components:
            role_components.append(_RoleComponent("organic_node", frozenset(component)))

        for component in edge_components:
            if component & absorbed:
                continue
            _, port_count = self.component_connections(
                component,
                node_component_by_atom,
                external_cut_bonds,
                source_atoms,
            )
            kind: OrganicFragmentKind = "edge" if port_count == 2 else "organic_node"
            role_components.append(_RoleComponent(kind, frozenset(component)))

        return self.repair_components(
            role_components,
            external_cut_bonds,
            source_atoms,
        )

    def repair_components(
        self,
        role_components: list[_RoleComponent],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> list[_RoleComponent]:
        components = [set(component.atom_indices) for component in role_components]
        kinds = [component.kind for component in role_components]

        while True:
            neighbors, port_counts = self.partition_state(
                components,
                external_cut_bonds,
                source_atoms,
            )
            absorbed = next(
                (
                    index
                    for index, port_count in enumerate(port_counts)
                    if port_count <= 1 and len(neighbors[index]) == 1
                ),
                None,
            )
            if absorbed is not None:
                parent = next(iter(neighbors[absorbed]))
                self.merge_component(components, kinds, parent, absorbed)
                continue

            repaired_kinds: list[OrganicFragmentKind] = [
                "organic_node" if port_count >= 3 else "edge"
                for port_count in port_counts
            ]
            if repaired_kinds != kinds:
                kinds = repaired_kinds
                continue
            kinds = repaired_kinds

            adjacent_edges = next(
                (
                    (index, adjacent)
                    for index, neighbors_for_component in enumerate(neighbors)
                    if kinds[index] == "edge"
                    for adjacent in sorted(neighbors_for_component)
                    if kinds[adjacent] == "edge"
                ),
                None,
            )
            if adjacent_edges is None:
                return [
                    _RoleComponent(kind, frozenset(component))
                    for kind, component in zip(kinds, components)
                ]
            left, right = sorted(adjacent_edges)
            self.merge_component(components, kinds, left, right)

    @staticmethod
    def merge_component(
        components: list[set[int]],
        kinds: list[OrganicFragmentKind],
        target: int,
        source: int,
    ) -> None:
        components[target].update(components[source])
        del components[source]
        del kinds[source]

    def partition_state(
        self,
        components: list[set[int]],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> tuple[list[set[int]], list[int]]:
        component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(components)
            for atom_index in component
        }
        neighbors = [set() for _ in components]
        external_cut_set = {tuple(sorted(cut_bond)) for cut_bond in external_cut_bonds}
        port_counts = [
            self.external_cut_count(
                component,
                external_cut_bonds,
                source_atoms,
            )
            for component in components
        ]

        for i, j in self.bonds.pairs:
            if tuple(sorted((i, j))) in external_cut_set:
                continue
            i_component = component_by_atom.get(i)
            j_component = component_by_atom.get(j)
            if i_component is None or j_component is None or i_component == j_component:
                continue
            neighbors[i_component].add(j_component)
            neighbors[j_component].add(i_component)
            port_counts[i_component] += 1
            port_counts[j_component] += 1

        return neighbors, port_counts

    def component_connections(
        self,
        component: set[int],
        node_component_by_atom: dict[int, int],
        external_cut_bonds: tuple[tuple[int, int], ...],
        source_atoms: set[int],
    ) -> tuple[set[int], int]:
        excluded_bonds = {tuple(sorted(cut_bond)) for cut_bond in external_cut_bonds}
        boundary_nodes: set[int] = set()
        port_count = self.external_cut_count(
            component,
            external_cut_bonds,
            source_atoms,
        )
        for atom_index in component:
            for neighbor in self.bonds.adjacency[atom_index]:
                if tuple(sorted((atom_index, neighbor))) in excluded_bonds:
                    continue
                node_component = node_component_by_atom.get(neighbor)
                if node_component is not None:
                    boundary_nodes.add(node_component)
                    port_count += 1
        return boundary_nodes, port_count

    def is_pendant_donor(
        self,
        component: set[int],
        node_component_by_atom: dict[int, int],
        external_cut_bonds: tuple[tuple[int, int], ...],
    ) -> bool:
        donor_elements = {"O", "N", "S"}
        if not component or any(
            self.symbols[atom_index] not in donor_elements for atom_index in component
        ):
            return False

        external_cut_set = {tuple(sorted(cut_bond)) for cut_bond in external_cut_bonds}
        core_boundary_bonds = [
            (atom_index, neighbor)
            for atom_index in component
            for neighbor in self.bonds.adjacency[atom_index]
            if neighbor in node_component_by_atom
            and tuple(sorted((atom_index, neighbor))) not in external_cut_set
        ]
        boundary_node_indices = {
            node_component_by_atom[neighbor] for _, neighbor in core_boundary_bonds
        }
        if len(core_boundary_bonds) != 1 or len(boundary_node_indices) != 1:
            return False

        return all(
            neighbor in component
            or neighbor in node_component_by_atom
            or tuple(sorted((atom_index, neighbor))) in external_cut_set
            for atom_index in component
            for neighbor in self.bonds.adjacency[atom_index]
        )

    def attach_hydrogens(
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

    def cut_bonds_between_components(
        self,
        components: list[set[int]],
        excluded_bonds: set[tuple[int, int]] | frozenset[tuple[int, int]] = frozenset(),
    ) -> set[tuple[int, int]]:
        component_by_atom = {
            atom_index: component_index
            for component_index, component in enumerate(components)
            for atom_index in component
        }
        cut_bonds: set[tuple[int, int]] = set()
        for i, j in self.bonds.pairs:
            if tuple(sorted((i, j))) in excluded_bonds:
                continue
            i_component = component_by_atom.get(i)
            j_component = component_by_atom.get(j)
            if i_component is None or j_component is None or i_component == j_component:
                continue
            cut_bonds.add(tuple(sorted((i, j))))
        return cut_bonds
