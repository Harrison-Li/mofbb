import numpy as np
import pytest
from ase import Atoms

import mofbb.decompose.periodicity as periodicity_module
from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.periodicity import (
    ComponentPeriodicity,
    PeriodicBond,
    PeriodicOpening,
    analyze_component,
    connection_shift,
    periodic_bonds,
    select_periodic_cuts,
)


def make_bonds(atoms, pairs_with_shifts):
    adjacency = [set() for _ in atoms]
    vectors = {}
    for i, j, shift in pairs_with_shifts:
        a, b = sorted((i, j))
        oriented = np.asarray(shift if i == a else tuple(-v for v in shift))
        adjacency[a].add(b)
        adjacency[b].add(a)
        vectors[(a, b)] = (
            atoms.positions[b]
            + oriented @ atoms.cell.array
            - atoms.positions[a]
        )
    return BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )


def make_rank_one_records():
    atoms, bonds = make_rank_one_triangle()
    return periodic_bonds(atoms, bonds)


def make_rank_one_triangle():
    atoms = Atoms(
        "CCC",
        positions=[(1, 0, 0), (4, 0, 0), (8, 0, 0)],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = make_bonds(
        atoms,
        [(0, 1, (0, 0, 0)), (1, 2, (0, 0, 0)), (0, 2, (-1, 0, 0))],
    )
    return atoms, bonds


def make_two_cycle_rod():
    atoms = Atoms(
        ["Zn", "O", "C", "O", "C"],
        positions=[
            (1, 1, 1),
            (3, 1, 1),
            (5, 1, 1),
            (1, 3, 1),
            (1, 5, 1),
        ],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = make_bonds(
        atoms,
        [
            (0, 1, (0, 0, 0)),
            (1, 2, (0, 0, 0)),
            (0, 2, (-1, 0, 0)),
            (0, 3, (0, 0, 0)),
            (3, 4, (0, 0, 0)),
            (0, 4, (-1, 0, 0)),
        ],
    )
    return atoms, bonds


def permute_bonded_structure(atoms, bonds, old_indices_by_new_index):
    order = tuple(old_indices_by_new_index)
    old_to_new = {old: new for new, old in enumerate(order)}
    permuted_atoms = atoms[list(order)]
    adjacency = [set() for _ in permuted_atoms]
    vectors = {}
    bond_types = {}
    for old_i, old_j in bonds.pairs:
        new_i = old_to_new[old_i]
        new_j = old_to_new[old_j]
        a, b = sorted((new_i, new_j))
        vector = bonds.vector(old_i, old_j)
        vectors[(a, b)] = np.array(vector if new_i == a else -vector)
        adjacency[a].add(b)
        adjacency[b].add(a)
        bond_types[(a, b)] = bonds.bond_type(old_i, old_j)
    return permuted_atoms, BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
        bond_types=bond_types,
    )


def test_periodic_bond_shift_is_canonical_and_reversible():
    atoms = Atoms("CC", positions=[(1, 0, 0), (9, 0, 0)], cell=[10, 10, 10], pbc=True)
    bonds = make_bonds(atoms, [(0, 1, (-1, 0, 0))])

    record = periodic_bonds(atoms, bonds)[0]

    assert record == PeriodicBond(0, 1, (-1, 0, 0))
    assert record.oriented_shift(0, 1) == (-1, 0, 0)
    assert record.oriented_shift(1, 0) == (1, 0, 0)


def test_component_analysis_detects_nonzero_cycle_rank():
    records = make_rank_one_records()
    excluded_record = next(
        record for record in records if record.pair == (1, 2)
    )

    periodic = analyze_component({0, 1, 2}, records)
    finite = analyze_component({0, 1, 2}, records, excluded={excluded_record})

    assert periodic.rank == 1
    assert periodic.generators == ((-1, 0, 0),)
    assert finite.rank == 0
    assert finite.generators == ()


def test_component_analysis_returns_sorted_offsets():
    analysis = analyze_component({0, 1, 2}, make_rank_one_records())

    assert analysis.offsets == (
        (0, (0, 0, 0)),
        (1, (0, 0, 0)),
        (2, (-1, 0, 0)),
    )
    assert analysis.offset(2) == (-1, 0, 0)


def test_connection_shift_reverses_with_endpoints_and_components():
    record = PeriodicBond(1, 4, (1, -1, 0))
    begin_offsets = ComponentPeriodicity(
        offsets=((1, (2, 0, 0)),),
        generators=(),
        rank=0,
    )
    end_offsets = ComponentPeriodicity(
        offsets=((4, (-1, 3, 0)),),
        generators=(),
        rank=0,
    )

    forward = connection_shift(record, 1, begin_offsets, 4, end_offsets)
    reverse = connection_shift(record, 4, end_offsets, 1, begin_offsets)

    assert forward == (4, -4, 0)
    assert reverse == tuple(-value for value in forward)


def test_component_rank_counts_independent_generators():
    records = (
        PeriodicBond(0, 1, (0, 0, 0)),
        PeriodicBond(0, 2, (0, 0, 0)),
        PeriodicBond(0, 3, (0, 0, 0)),
        PeriodicBond(1, 2, (1, 0, 0)),
        PeriodicBond(1, 3, (0, 1, 0)),
        PeriodicBond(2, 3, (1, 1, 0)),
    )

    analysis = analyze_component({0, 1, 2, 3}, records)

    assert analysis.generators == (
        (-1, -1, 0),
        (-1, 0, 0),
        (0, -1, 0),
    )
    assert analysis.generators[0] == tuple(
        left + right
        for left, right in zip(analysis.generators[1], analysis.generators[2])
    )
    assert analysis.rank == 2
    assert analysis.rank != len(analysis.generators)


def test_selector_finds_two_cuts_for_two_edge_disjoint_periodic_cycles():
    atoms, bonds = make_two_cycle_rod()
    records = periodic_bonds(atoms, bonds)

    selection = select_periodic_cuts(
        atoms, bonds, atoms.get_chemical_symbols(), set(range(5)), records
    )

    assert len(selection.cuts) == 2
    residual = analyze_component(set(range(5)), records, set(selection.cuts))
    assert residual.rank == 0
    assert selection.residual == residual


def test_selector_rejects_genuine_rank_two_component_with_full_diagnostic():
    atoms = Atoms(
        "CCCC",
        positions=[(1, 1, 1), (3, 1, 1), (1, 3, 1), (3, 3, 1)],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = make_bonds(
        atoms,
        [
            (0, 1, (0, 0, 0)),
            (0, 2, (0, 0, 0)),
            (0, 3, (0, 0, 0)),
            (1, 2, (1, 0, 0)),
            (1, 3, (0, 1, 0)),
            (2, 3, (1, 1, 0)),
        ],
    )
    records = periodic_bonds(atoms, bonds)

    analysis = analyze_component(set(range(4)), records)
    assert analysis.rank == 2

    with pytest.raises(ValueError) as caught:
        select_periodic_cuts(
            atoms,
            bonds,
            atoms.get_chemical_symbols(),
            set(range(4)),
            records,
        )

    message = str(caught.value)
    assert "C4" in message
    assert "rank 2" in message
    assert "generators=" in message


def test_selector_reports_private_candidate_bound_and_component_context(monkeypatch):
    atoms, bonds = make_rank_one_triangle()
    records = periodic_bonds(atoms, bonds)
    monkeypatch.setattr(periodicity_module, "_MAX_CANDIDATES", 0, raising=False)

    with pytest.raises(ValueError) as caught:
        select_periodic_cuts(
            atoms,
            bonds,
            atoms.get_chemical_symbols(),
            set(range(3)),
            records,
        )

    message = str(caught.value)
    assert "bound=0" in message
    assert "formula=C3" in message
    assert "component_size=3" in message


def test_selector_reports_when_private_cut_limit_precludes_an_opening(monkeypatch):
    atoms, bonds = make_two_cycle_rod()
    records = periodic_bonds(atoms, bonds)
    monkeypatch.setattr(periodicity_module, "_MAX_CUT_SIZE", 1, raising=False)

    with pytest.raises(ValueError) as caught:
        select_periodic_cuts(
            atoms,
            bonds,
            atoms.get_chemical_symbols(),
            set(range(5)),
            records,
        )

    message = str(caught.value)
    assert "max_cut_size=1" in message
    assert "formula=C2O2Zn" in message
    assert "component_size=5" in message


def test_marked_graph_preserves_oriented_shift_and_cut_state():
    atoms, bonds = make_rank_one_triangle()
    records = periodic_bonds(atoms, bonds)
    cut = next(record for record in records if record.pair == (0, 2))

    graph = periodicity_module._marked_graph(
        atoms.get_chemical_symbols(), set(range(3)), records, (cut,), bonds
    )

    assert graph.is_directed()
    assert graph[0][2]["label"] == ("S", True, (-1, 0, 0), 3.0)
    assert graph[2][0]["label"] == ("S", True, (1, 0, 0), 3.0)
    assert graph[0][1]["label"][1] is False


def test_canonical_selector_uses_supplied_component_records(monkeypatch):
    atoms, bonds = make_rank_one_triangle()
    records = periodic_bonds(atoms, bonds)
    cut = next(record for record in records if record.pair == (0, 2))
    residual = analyze_component(set(range(3)), records, excluded={cut})

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("periodic records must not be recomputed")

    monkeypatch.setattr(periodicity_module, "periodic_bonds", fail_if_recomputed)

    selected = periodicity_module._select_canonical_candidate(
        atoms,
        bonds,
        atoms.get_chemical_symbols(),
        set(range(3)),
        records,
        [((cut,), residual)],
    )

    assert selected == ((cut,), residual)


def test_canonical_selector_rejects_inequivalent_ties_with_cut_details(monkeypatch):
    atoms = Atoms(
        ["C", "O", "N"],
        positions=[(0, 0, 0), (1, 0, 0), (3, 0, 0)],
        cell=[10, 10, 10],
        pbc=False,
    )
    bonds = make_bonds(
        atoms,
        [(0, 1, (0, 0, 0)), (1, 2, (0, 0, 0))],
    )
    records = periodic_bonds(atoms, bonds)
    finite = ComponentPeriodicity(
        offsets=((0, (0, 0, 0)), (1, (0, 0, 0)), (2, (0, 0, 0))),
        generators=(),
        rank=0,
    )
    monkeypatch.setattr(
        periodicity_module,
        "_candidate_score",
        lambda *args, **kwargs: (0,),
        raising=False,
    )

    with pytest.raises(ValueError) as caught:
        periodicity_module._select_canonical_candidate(
            atoms,
            bonds,
            atoms.get_chemical_symbols(),
            set(range(3)),
            records,
            [((records[0],), finite), ((records[1],), finite)],
        )

    message = str(caught.value)
    assert "ambiguous periodic cut" in message
    assert "nonmetal-covalent" in message
    assert "C-O" in message
    assert "N-O" in message
    assert "1.000000" in message
    assert "2.000000" in message


def test_selector_is_invariant_to_atom_permutation_when_chemistry_is_unique():
    atoms, bonds = make_two_cycle_rod()
    records = periodic_bonds(atoms, bonds)
    selection = select_periodic_cuts(
        atoms, bonds, atoms.get_chemical_symbols(), set(range(5)), records
    )

    old_indices_by_new_index = (2, 4, 1, 3, 0)
    permuted_atoms, permuted_bonds = permute_bonded_structure(
        atoms, bonds, old_indices_by_new_index
    )
    permuted_records = periodic_bonds(permuted_atoms, permuted_bonds)
    permuted_selection = select_periodic_cuts(
        permuted_atoms,
        permuted_bonds,
        permuted_atoms.get_chemical_symbols(),
        set(range(5)),
        permuted_records,
    )

    original_pairs = {record.pair for record in selection.cuts}
    permuted_pairs_in_original_numbering = {
        tuple(
            sorted(
                (
                    old_indices_by_new_index[record.i],
                    old_indices_by_new_index[record.j],
                )
            )
        )
        for record in permuted_selection.cuts
    }
    assert original_pairs == {(0, 1), (0, 3)}
    assert permuted_pairs_in_original_numbering == original_pairs


def test_periodic_opening_serializes_auditable_cut_metadata():
    atoms = Atoms(
        ["Zn", "O", "C"],
        positions=[(1, 0, 0), (4, 0, 0), (8, 0, 0)],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = make_bonds(
        atoms,
        [(0, 1, (0, 0, 0)), (1, 2, (0, 0, 0)), (0, 2, (-1, 0, 0))],
    )
    records = periodic_bonds(atoms, bonds)
    selection = select_periodic_cuts(
        atoms, bonds, atoms.get_chemical_symbols(), set(range(3)), records
    )

    opening = PeriodicOpening.from_selection(set(range(3)), selection)
    payload = opening.as_dict()

    assert payload["source_indices"] == [0, 1, 2]
    assert payload["rank"] == 1
    assert payload["generators"] == [[-1, 0, 0]]
    assert payload["candidates_considered"] == selection.candidates_considered
    assert payload["cuts"] == [
        {
            "source_cut_bond": [0, 1],
            "shift": [0, 0, 0],
            "element_pair": ["O", "Zn"],
            "bond_class": "metal-coordination",
            "bond_class_penalty": 1,
            "bond_type": "S",
            "length": 3.0,
        }
    ]
