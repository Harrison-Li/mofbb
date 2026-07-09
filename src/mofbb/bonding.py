"""Distance-based bond graph construction for ASE structures."""

from __future__ import annotations

from dataclasses import dataclass

import ase
import networkx as nx
import numpy as np

try:  # ASE has moved this helper across versions.
    from ase.neighborlist import natural_cutoffs, neighbor_list
except ImportError:  # pragma: no cover - compatibility for unusual ASE layouts
    from ase.utils import natural_cutoffs  # type: ignore[attr-defined]
    from ase.neighborlist import neighbor_list


@dataclass(frozen=True)
class BondNetwork:
    """Undirected minimum-image bond graph for an ASE structure.

    Attributes
    ----------
    adjacency
        Atom-neighbor sets indexed by source atom index.
    vectors
        Minimum-image vectors keyed by sorted atom-index pairs.
    """

    adjacency: tuple[frozenset[int], ...]
    vectors: dict[tuple[int, int], np.ndarray]

    @property
    def pairs(self) -> list[tuple[int, int]]:
        """Return sorted bonded atom-index pairs."""

        return sorted(self.vectors)

    def vector(self, i: int, j: int) -> np.ndarray:
        """Return the minimum-image vector from atom ``i`` to atom ``j``."""

        if i < j:
            return self.vectors[(i, j)]
        return -self.vectors[(j, i)]


def build_bond_network(atoms: ase.Atoms, scale: float = 1.2) -> BondNetwork:
    """Build an undirected minimum-image bond graph from covalent cutoffs.

    Parameters
    ----------
    atoms
        Structure to analyze.
    scale
        Multiplier applied to ASE natural covalent cutoffs.

    Returns
    -------
    BondNetwork
        Bond adjacency and minimum-image vectors.
    """

    cutoffs = [scale * cutoff for cutoff in natural_cutoffs(atoms)]
    i_list, j_list, d_list = neighbor_list(
        "ijD", atoms, cutoff=cutoffs, self_interaction=False
    )

    vector_by_pair: dict[tuple[int, int], np.ndarray] = {}
    for i, j, vector in zip(i_list, j_list, d_list):
        atom_i = int(i)
        atom_j = int(j)
        if atom_i == atom_j:
            continue
        a, b = sorted((atom_i, atom_j))
        pair_vector = np.array(vector if atom_i == a else -vector, dtype=float)
        previous = vector_by_pair.get((a, b))
        if previous is None or np.linalg.norm(pair_vector) < np.linalg.norm(previous):
            vector_by_pair[(a, b)] = pair_vector

    adjacency = [set() for _ in atoms]
    for i, j in vector_by_pair:
        adjacency[i].add(j)
        adjacency[j].add(i)

    return BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vector_by_pair,
    )


def largest_connected_component(atoms: ase.Atoms, scale: float = 1.2) -> ase.Atoms:
    """Return a copy containing only the largest bonded component.

    Structures from databases sometimes include solvents, counterions, or other
    disconnected species.  Keeping this helper in the bonding module makes that
    cleanup behavior explicit and reusable.
    """

    bonds = build_bond_network(atoms, scale=scale)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from(bonds.pairs)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if len(components) <= 1:
        return atoms
    return atoms[sorted(components[0])]

