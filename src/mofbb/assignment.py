"""SingleNode-style atom-role assignment.

This module decides which input atoms belong to node SBUs and which atoms are
points of extension.  It deliberately does not create fragments, cap bonds, or
write files; those concerns live in separate modules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx

from .bonding import BondNetwork
from .constants import is_mofid_metal


@dataclass(frozen=True)
class AtomAssignment:
    """Result of assigning source atoms to SingleNode-inspired roles."""

    node_indices: frozenset[int]
    points_of_extension: frozenset[int]


class SingleNodeAtomAssigner:
    """Classify source atoms using MOFid SingleNode-inspired rules."""

    def __init__(self, symbols: list[str], bonds: BondNetwork) -> None:
        self.symbols = symbols
        self.bonds = bonds

    def assign(self) -> AtomAssignment:
        """Return node atoms and points of extension for the source structure."""

        nodes = self._seed_node_atoms()
        points_of_extension: set[int] = set()
        visited_bridges: set[int] = set()

        for atom_index in list(nodes):
            if atom_index in visited_bridges:
                continue

            symbol = self.symbols[atom_index]
            if symbol == "O":
                self._classify_oxygen_neighbor(
                    atom_index, nodes, points_of_extension, visited_bridges
                )
            elif symbol == "N":
                self._classify_nitrogen_neighbor(
                    atom_index, nodes, points_of_extension, visited_bridges
                )

        return AtomAssignment(
            node_indices=frozenset(nodes),
            points_of_extension=frozenset(points_of_extension),
        )

    def _seed_node_atoms(self) -> set[int]:
        """Start nodes with MOFid metals and their direct neighbors."""

        nodes = {i for i, symbol in enumerate(self.symbols) if is_mofid_metal(symbol)}
        for metal in list(nodes):
            nodes.update(self.bonds.adjacency[metal])
        return nodes

    def _classify_oxygen_neighbor(
        self,
        oxygen: int,
        nodes: set[int],
        points_of_extension: set[int],
        visited_bridges: set[int],
    ) -> None:
        """Apply SingleNode-style rules for oxygen atoms adjacent to metals."""

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
            points_of_extension,
            visited_bridges,
        )

    def _classify_carboxylate_oxygen(
        self,
        oxygen: int,
        carbon: int,
        attached_hydrogens: set[int],
        nodes: set[int],
        points_of_extension: set[int],
        visited_bridges: set[int],
    ) -> None:
        """Promote a valid carboxylate bridge into the node assignment."""

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
        points_of_extension.add(carbon)
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
        points_of_extension: set[int],
        visited_bridges: set[int],
    ) -> None:
        """Apply SingleNode-style rules for nitrogen atoms adjacent to metals."""

        bridge_nitrogen = self._paired_node_nitrogen(nitrogen, nodes)
        if bridge_nitrogen is None:
            nodes.discard(nitrogen)
            return

        ring, ring_poe = self._calculate_nonmetal_ring(nitrogen, bridge_nitrogen)
        if not ring:
            return

        nodes.update(ring)
        visited_bridges.update(ring)
        points_of_extension.update(ring_poe - nodes)

    def _paired_node_nitrogen(self, nitrogen: int, nodes: set[int]) -> int | None:
        return next(
            (
                neighbor
                for neighbor in self.bonds.adjacency[nitrogen]
                if self.symbols[neighbor] == "N" and neighbor in nodes
            ),
            None,
        )

    def _calculate_nonmetal_ring(self, a: int, b: int) -> tuple[set[int], set[int]]:
        graph = nx.Graph()
        for i, j in self.bonds.pairs:
            if is_mofid_metal(self.symbols[i]) or is_mofid_metal(self.symbols[j]):
                continue
            if {i, j} == {a, b}:
                continue
            graph.add_edge(i, j)

        try:
            path = nx.shortest_path(graph, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return set(), set()

        if len(path) > 6:
            return set(), set()

        ring = set(path)
        ring.add(a)
        ring.add(b)
        ring_poe: set[int] = set()

        for atom in list(ring):
            for neighbor in self.bonds.adjacency[atom]:
                if neighbor in ring or is_mofid_metal(self.symbols[neighbor]):
                    continue
                substituent = self._nonmetal_substituent(neighbor, blocked=ring)
                if len(substituent) == 1:
                    ring_poe.add(atom)
                else:
                    ring.update(substituent)

        return ring, ring_poe

    def _nonmetal_substituent(self, start: int, blocked: set[int]) -> set[int]:
        substituent: set[int] = set()
        queue: deque[int] = deque([start])

        while queue:
            current = queue.popleft()
            if current in substituent or current in blocked:
                continue
            if is_mofid_metal(self.symbols[current]):
                continue

            substituent.add(current)
            for neighbor in self.bonds.adjacency[current]:
                if neighbor not in substituent and neighbor not in blocked:
                    queue.append(neighbor)

        return substituent
