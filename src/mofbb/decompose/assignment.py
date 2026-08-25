"""Metal-node atom assignment."""

from __future__ import annotations

from collections import deque

import networkx as nx

from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.constants import is_metal


class NodeAtomAssigner:
    """Classify source atoms using MOFid-inspired rules."""

    def __init__(
        self,
        symbols: list[str],
        bonds: BondNetwork,
        *,
        absorb_one_port_regions: bool = False,
    ) -> None:
        self.symbols = symbols
        self.bonds = bonds
        self.absorb_one_port_regions = absorb_one_port_regions
        self.node_core_by_atom: dict[int, int] = {}
        self.coordination_seed_atoms: frozenset[int] = frozenset()

    def assign(self) -> frozenset[int]:
        nodes = self._seed_node_atoms()
        visited_bridges: set[int] = set()
        seed_atoms = tuple(sorted(nodes))

        for atom_index in seed_atoms:
            if atom_index in visited_bridges or self.symbols[atom_index] != "O":
                continue
            self._classify_oxygen_neighbor(
                atom_index,
                nodes,
                visited_bridges,
            )

        for atom_index in seed_atoms:
            if (
                self.symbols[atom_index] == "N"
                and self._paired_node_nitrogen(atom_index, nodes) is None
            ):
                nodes.discard(atom_index)

        self.coordination_seed_atoms = frozenset(nodes.intersection(seed_atoms))
        self._index_node_cores(nodes)

        for atom_index in seed_atoms:
            if (
                atom_index not in nodes
                or atom_index in visited_bridges
                or self.symbols[atom_index] != "N"
            ):
                continue
            self._classify_nitrogen_neighbor(
                atom_index,
                nodes,
                visited_bridges,
            )

        if self.absorb_one_port_regions:
            self._absorb_one_port_regions(nodes)

        return frozenset(nodes)

    def _seed_node_atoms(self) -> set[int]:
        nodes = {i for i, symbol in enumerate(self.symbols) if is_metal(symbol)}
        for metal in list(nodes):
            nodes.update(self.bonds.adjacency[metal])
        return nodes

    def _index_node_cores(self, nodes: set[int]) -> None:
        node_graph = nx.Graph()
        node_graph.add_nodes_from(nodes)
        node_graph.add_edges_from(
            (i, j) for i, j in self.bonds.pairs if i in nodes and j in nodes
        )
        self.node_core_by_atom = {
            atom: core
            for core, component in enumerate(nx.connected_components(node_graph))
            for atom in component
        }

    def _absorb_one_port_regions(self, nodes: set[int]) -> None:
        remaining = set(range(len(self.symbols))) - nodes
        while remaining:
            start = remaining.pop()
            component = {start}
            boundary_count = 0
            queue: deque[int] = deque([start])

            while queue:
                atom = queue.popleft()
                for neighbor in self.bonds.adjacency[atom]:
                    if neighbor in nodes:
                        boundary_count += 1
                    elif neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            if boundary_count == 1:
                nodes.update(component)

    def _classify_oxygen_neighbor(
        self,
        oxygen: int,
        nodes: set[int],
        visited_bridges: set[int],
    ) -> None:
        attached_hydrogens: set[int] = set()
        attached_carbon: int | None = None
        metal_oxide_or_carboxylate = True

        for neighbor in self.bonds.adjacency[oxygen]:
            if neighbor in nodes:
                continue

            neighbor_symbol = self.symbols[neighbor]
            if neighbor_symbol == "H":
                attached_hydrogens.add(neighbor)
            elif neighbor_symbol == "O":
                attached_hydrogens.add(neighbor)
                for second_neighbor in self.bonds.adjacency[neighbor]:
                    if second_neighbor == oxygen or second_neighbor in nodes:
                        continue
                    if self.symbols[second_neighbor] == "H":
                        attached_hydrogens.add(second_neighbor)
                    else:
                        metal_oxide_or_carboxylate = False
            elif neighbor_symbol == "C":
                attached_carbon = neighbor
            else:
                metal_oxide_or_carboxylate = False

        if not metal_oxide_or_carboxylate:
            nodes.discard(oxygen)
            return

        if attached_carbon is None:
            nodes.update(attached_hydrogens)
            return

        self._classify_carboxylate_oxygen(
            oxygen,
            attached_carbon,
            attached_hydrogens,
            nodes,
            visited_bridges,
        )

    def _classify_carboxylate_oxygen(
        self,
        oxygen: int,
        carbon: int,
        attached_hydrogens: set[int],
        nodes: set[int],
        visited_bridges: set[int],
    ) -> None:
        carbon_neighbors = self.bonds.adjacency[carbon]
        carbon_c_neighbors = [i for i in carbon_neighbors if self.symbols[i] == "C"]
        carbon_o_neighbors = [i for i in carbon_neighbors if self.symbols[i] == "O"]
        unexpected_neighbors = [
            i for i in carbon_neighbors if self.symbols[i] not in {"C", "O"}
        ]

        if (
            len(carbon_c_neighbors) != 1
            or len(carbon_o_neighbors) != 2
            or unexpected_neighbors
        ):
            nodes.discard(oxygen)
            return

        other_oxygen = next((i for i in carbon_o_neighbors if i != oxygen), None)
        if other_oxygen is None:
            nodes.discard(oxygen)
            return

        carboxylate = {oxygen, carbon, other_oxygen}
        visited_bridges.update(carboxylate)
        nodes.update(carboxylate)

        for neighbor in self.bonds.adjacency[other_oxygen]:
            if self.symbols[neighbor] == "H":
                attached_hydrogens.add(neighbor)
        nodes.update(attached_hydrogens)

    def _classify_nitrogen_neighbor(
        self,
        nitrogen: int,
        nodes: set[int],
        visited_bridges: set[int],
    ) -> None:
        bridge_nitrogen = self._paired_node_nitrogen(nitrogen, nodes)
        if bridge_nitrogen is None:
            nodes.discard(nitrogen)
            return

        ring = self._calculate_nonmetal_ring(nitrogen, bridge_nitrogen)
        if not ring:
            return

        nodes.update(ring)
        visited_bridges.update(ring)

    def _paired_node_nitrogen(self, nitrogen: int, nodes: set[int]) -> int | None:
        return next(
            (
                neighbor
                for neighbor in self.bonds.adjacency[nitrogen]
                if self.symbols[neighbor] == "N" and neighbor in nodes
            ),
            None,
        )

    def _calculate_nonmetal_ring(self, a: int, b: int) -> set[int]:
        graph = nx.Graph()
        for i, j in self.bonds.pairs:
            if is_metal(self.symbols[i]) or is_metal(self.symbols[j]):
                continue
            if {i, j} == {a, b}:
                continue
            graph.add_edge(i, j)

        if a not in graph or b not in graph or not nx.has_path(graph, a, b):
            return set()
        path = nx.shortest_path(graph, a, b)

        if len(path) > 6:
            return set()

        ring = set(path)
        ring.add(a)
        ring.add(b)
        node_core = self.node_core_by_atom.get(a)

        for atom in list(ring):
            substituents: set[int] = set()
            reaches_other_core = False
            for neighbor in self.bonds.adjacency[atom]:
                if neighbor in ring:
                    continue
                substituent, reaches_other_core = self._trace_external_region(
                    neighbor, blocked=ring, node_core=node_core
                )
                substituents.update(substituent)
                if reaches_other_core:
                    break
            if not reaches_other_core:
                ring.update(substituents)

        return ring

    def _trace_external_region(
        self, start: int, blocked: set[int], node_core: int | None
    ) -> tuple[set[int], bool]:
        substituent: set[int] = set()
        visited: set[int] = set()
        queue: deque[int] = deque([start])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if current in blocked:
                continue
            reached_core = self.node_core_by_atom.get(current)
            if reached_core is not None:
                if reached_core != node_core or (
                    substituent and current in self.coordination_seed_atoms
                ):
                    return substituent, True
                continue

            substituent.add(current)
            for neighbor in self.bonds.adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        return substituent, False
