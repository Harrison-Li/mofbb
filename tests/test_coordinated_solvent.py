import numpy as np
import pytest
from ase import Atoms

import mofbb.decompose.decomposer as decomposer_module
from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.decomposer import NodeEdgeDecomposer
from mofbb.decompose.solvent import MofidSolventRemover


def make_atoms_and_bonds(symbols, pairs, *, positions=None, shifts=None):
    if positions is None:
        positions = [(float(index), 0.0, 0.0) for index in range(len(symbols))]
    positions = np.asarray(positions, dtype=float)
    atoms = Atoms(symbols, positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)
    adjacency = [set() for _ in symbols]
    vectors = {}
    shifts = shifts or {}
    for i, j in pairs:
        left, right = sorted((i, j))
        adjacency[left].add(right)
        adjacency[right].add(left)
        shift = np.asarray(shifts.get((left, right), (0, 0, 0)), dtype=float)
        vectors[(left, right)] = (
            positions[right] + shift @ atoms.cell.array - positions[left]
        )
    return atoms, BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )


def make_stable_framework(extra_symbols, extra_pairs, *, positions=None, shifts=None):
    symbols = ["Zn", "Zn", "N", "C", "N", "N", "C", "N", *extra_symbols]
    framework_pairs = [
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 1),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 1),
    ]
    return make_atoms_and_bonds(
        symbols,
        [*framework_pairs, *extra_pairs],
        positions=positions,
        shifts=shifts,
    )


def test_terminal_aqua_is_classified_as_bound_solvent() -> None:
    atoms, bonds = make_stable_framework(
        ["O", "H", "H"],
        [(0, 8), (8, 9), (8, 10)],
    )

    result = MofidSolventRemover(atoms, bonds).classify()

    assert result.free_solvent_components == ()
    assert result.bound_solvent_components == ((8, 9, 10),)
    assert result.fsr_indices == tuple(range(11))
    assert result.asr_indices == tuple(range(8))


def test_terminal_aqua_tolerates_inferred_metal_hydrogen_bond() -> None:
    atoms, bonds = make_stable_framework(
        ["O", "H", "H"],
        [(0, 8), (8, 9), (8, 10), (0, 10)],
    )

    result = MofidSolventRemover(atoms, bonds).classify()

    assert result.bound_solvent_components == ((8, 9, 10),)
    assert result.asr_indices == tuple(range(8))


def test_free_mode_keeps_terminal_aqua_out_of_metal_node(monkeypatch) -> None:
    atoms, bonds = make_stable_framework(
        ["O", "H", "H"],
        [(0, 8), (8, 9), (8, 10)],
    )
    monkeypatch.setattr(
        decomposer_module,
        "read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )

    decomposer = NodeEdgeDecomposer("fixture.cif", solvent_mode="free")
    metal, organic, _ = decomposer._roles()

    assert {8, 9, 10} <= organic
    assert {8, 9, 10}.isdisjoint(metal)

    result = decomposer.decompose()
    assert len(result.bound_linker_fragments) == 1
    assert result.bound_linker_fragments[0].source_indices == (8, 9, 10)


@pytest.mark.parametrize("second_site_shift", [(0, 0, 0), (1, 0, 0)])
def test_split_aqua_oxygen_sites_are_classified_as_bound_solvent(
    second_site_shift,
) -> None:
    positions = [(float(index), 0.0, 0.0) for index in range(10)]
    positions[8] = (0.0, 2.0, 0.0)
    positions[9] = (0.8, 2.0, 0.0)
    atoms, bonds = make_stable_framework(
        ["O", "O"],
        [(0, 8), (0, 9), (8, 9)],
        positions=positions,
        shifts={(0, 9): second_site_shift},
    )

    result = MofidSolventRemover(atoms, bonds).classify()

    assert result.free_solvent_components == ()
    assert result.bound_solvent_components == ((8, 9),)
    assert result.fsr_indices == tuple(range(10))
    assert result.asr_indices == tuple(range(8))


@pytest.mark.parametrize(
    ("extra_symbols", "extra_pairs"),
    [
        (["O"], [(0, 8)]),
        (["O", "H"], [(0, 8), (8, 9)]),
    ],
    ids=["terminal-oxo", "terminal-hydroxide"],
)
def test_terminal_framework_oxygen_is_not_removed(extra_symbols, extra_pairs) -> None:
    atoms, bonds = make_stable_framework(extra_symbols, extra_pairs)

    result = MofidSolventRemover(atoms, bonds).classify()

    assert result.bound_solvent_components == ()
    assert result.asr_indices == tuple(range(len(atoms)))


def test_bridging_water_is_not_removed() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "Zn", "O", "H", "H", "N", "C", "N"],
        [
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (1, 7),
        ],
        shifts={(1, 7): (1, 0, 0)},
    )

    result = MofidSolventRemover(atoms, bonds).classify()

    assert result.free_solvent_components == ()
    assert result.bound_solvent_components == ()
    assert result.asr_indices == tuple(range(len(atoms)))


def test_realistic_oxygen_oxygen_bond_is_not_treated_as_split_disorder() -> None:
    positions = [(float(index), 0.0, 0.0) for index in range(10)]
    positions[8] = (0.0, 2.0, 0.0)
    positions[9] = (1.2, 2.0, 0.0)
    atoms, bonds = make_stable_framework(
        ["O", "O"],
        [(0, 8), (0, 9), (8, 9)],
        positions=positions,
    )

    result = MofidSolventRemover(atoms, bonds).classify()

    assert result.bound_solvent_components == ()
    assert result.asr_indices == tuple(range(len(atoms)))
