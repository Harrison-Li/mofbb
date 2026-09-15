"""Finite terminal contraction, periodic safeguards, and atom preservation."""

import numpy as np
import pytest
from ase import Atoms

from mofbb.compose import Reconstructor
from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.cleanup.terminal import TerminalAbsorber
from mofbb.decompose.decomposer import NodeEdgeDecomposer
from mofbb.decompose.fragments import FragmentBuilder
from mofbb.decompose.results import NodeEdgeDecompositionResult


def make_fragments(symbols, pairs, components, cut_pairs, shifts=None, positions=None):
    symbols = Atoms(symbols).get_chemical_symbols()
    if positions is None:
        positions = [(i, i % 2, 0) for i in range(len(symbols))]
    atoms = Atoms(symbols, positions=positions, cell=[10] * 3, pbc=True)
    adjacency = [set() for _ in atoms]
    vectors = {}
    for i, j in pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)
        vectors[(i, j)] = (
            atoms.positions[j]
            - atoms.positions[i]
            + np.array((shifts or {}).get((i, j), (0, 0, 0))) @ atoms.cell
        )
    bonds = BondNetwork(tuple(map(frozenset, adjacency)), vectors)
    builder = FragmentBuilder(atoms, bonds, atoms.get_chemical_symbols())
    cuts = tuple(r for r in builder.source_records if r.pair in cut_pairs)
    fragments = [
        builder.materialize_fragment(kind, index, set(ids), cuts)
        for kind, index, ids in components
    ]
    decomposer = object.__new__(NodeEdgeDecomposer)
    decomposer.atoms, decomposer.bonds = atoms, bonds
    decomposer._assign_topology(fragments, cuts, builder.source_records)
    return builder, decomposer, fragments, cuts


def absorb(builder, fragments, cuts):
    return TerminalAbsorber(
        builder,
        *[
            [f for f in fragments if f.kind == kind]
            for kind in ("metal_node", "organic_node", "edge", "linker")
        ],
        cuts,
    ).run()


def test_terminal_metal_merges_into_cage_and_roundtrip_keeps_every_source_bond(
    tmp_path,
):
    builder, decomposer, fragments, cuts = make_fragments(
        "ZnNCC",
        [(0, 1), (0, 2), (1, 2), (2, 3), (1, 3)],
        [("metal_node", 1, [0]), ("organic_node", 1, [1, 2, 3])],
        {(0, 1), (0, 2), (1, 3)},
        {(1, 3): (-1, 0, 0)},
        positions=[(2, 1, 0), (1, 0, 0), (4, 0, 0), (8, 0, 0)],
    )
    assert len(fragments[0].logical_ports) == 1
    metal, organic, edges, remaining = absorb(builder, fragments, cuts)
    assert not organic and not edges and len(metal) == 1
    merged = metal[0]
    assert merged.source_indices == (0, 1, 2, 3)
    assert merged.num_x == 2
    assert [r.pair for r in remaining] == [(1, 3)]
    assert {(i, j) for i, j, _ in merged.bonds if i < 4 and j < 4} == {
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
    }
    assert {t for _, _, t in merged.bonds} == {"S"}
    vertices, top_edges = decomposer._assign_topology(
        metal, remaining, builder.source_records
    )
    result = NodeEdgeDecompositionResult(
        "terminal",
        tuple(builder.atoms.cell.cellpar()),
        vertices,
        top_edges,
        metal_node_fragments=metal,
        connection_records=remaining,
    )
    result.write(tmp_path)
    rebuilt = Reconstructor(tmp_path).reconstruct()
    assert rebuilt.atoms.get_chemical_symbols() == builder.atoms.get_chemical_symbols()
    delta = rebuilt.atoms.get_scaled_positions(
        wrap=False
    ) - builder.atoms.get_scaled_positions(wrap=False)
    np.testing.assert_allclose(delta - np.rint(delta), 0, atol=1e-10)
    assert set(map(tuple, rebuilt.bonds)) == set(builder.bonds.pairs)
    vectors = (
        rebuilt.atoms.positions[rebuilt.bonds[:, 1]]
        - rebuilt.atoms.positions[rebuilt.bonds[:, 0]]
        + rebuilt.bond_shifts @ rebuilt.atoms.cell
    )
    for pair, vector in zip(rebuilt.bonds, vectors):
        np.testing.assert_allclose(vector, builder.bonds.vector(*pair), atol=1e-10)


@pytest.mark.parametrize("periodic_host", [False, True])
def test_distinct_images_are_not_absorbed_even_when_logical_ports_quotient_them(
    periodic_host,
):
    pairs = [(0, 1), (0, 3), (1, 2), (2, 3)]
    cuts = {(0, 1), (0, 3)}
    shifts = {(0, 3): (1, 0, 0)}
    if periodic_host:
        pairs.append((1, 3))
        cuts.add((1, 3))
        shifts[(1, 3)] = (1, 0, 0)
    builder, _, fragments, cuts = make_fragments(
        "NZnCC",
        pairs,
        [("edge", 1, [0]), ("metal_node", 1, [1, 2, 3])],
        cuts,
        shifts,
    )
    assert len(fragments[0].logical_ports) == (1 if periodic_host else 2)
    metal, _, edges, remaining = absorb(builder, fragments, cuts)
    assert metal == [fragments[1]] and edges == [fragments[0]]
    assert remaining == cuts


def test_terminal_chain_contracts_without_mutual_absorption_or_duplicate_atoms():
    builder, _, fragments, cuts = make_fragments(
        "ZnNCC",
        [(0, 1), (1, 2), (2, 3)],
        [
            ("metal_node", 1, [0]),
            ("organic_node", 1, [1]),
            ("edge", 1, [2]),
            ("edge", 2, [3]),
        ],
        {(0, 1), (1, 2), (2, 3)},
    )
    metal, organic, edges, remaining = absorb(builder, fragments, cuts)
    assert not organic and not edges and not remaining
    assert len(metal) == 1 and metal[0].source_indices == (0, 1, 2, 3)
    assert metal[0].num_x == 0


def test_periodic_self_ports_are_preserved():
    builder, _, fragments, cuts = make_fragments(
        "ZnCC",
        [(0, 1), (1, 2), (0, 2)],
        [("metal_node", 1, [0, 1, 2])],
        {(0, 2)},
        {(0, 2): (1, 0, 0)},
    )
    metal, _, _, remaining = absorb(builder, fragments, cuts)
    assert metal == fragments and remaining == cuts


def test_bound_solvent_is_kept_separate_even_for_a_one_neighbor_metal():
    builder, _, fragments, cuts = make_fragments(
        "ZnO",
        [(0, 1)],
        [("metal_node", 1, [0]), ("linker", 1, [1])],
        {(0, 1)},
    )
    metal, organic, edges, remaining = absorb(builder, fragments, cuts)
    assert metal == [fragments[0]] and not organic and not edges
    assert remaining == cuts and fragments[1].source_indices == (1,)
