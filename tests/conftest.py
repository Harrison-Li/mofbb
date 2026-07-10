import numpy as np
import pytest
from ase import Atoms

from mofbb.bonding import BondNetwork


@pytest.fixture
def framework_with_free_and_bound_solvent() -> tuple[Atoms, BondNetwork]:
    symbols = ["Zn", "Zn", "N", "C", "N", "N", "C", "N", "O", "N", "C"]
    positions = np.array(
        [(float(index), float(index % 2), 0.0) for index in range(len(symbols))]
    )
    pairs = [
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 1),
        (0, 9),
        (9, 10),
    ]
    atoms = Atoms(symbols, positions=positions, cell=[30.0, 30.0, 30.0], pbc=True)
    adjacency = [set() for _ in atoms]
    vectors = {}
    for i, j in pairs:
        a, b = sorted((i, j))
        adjacency[a].add(b)
        adjacency[b].add(a)
        vectors[(a, b)] = positions[b] - positions[a]
    return atoms, BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )
