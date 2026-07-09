from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")

import ase.io
from ase import Atoms

from mofbb.bonding import (
    BondNetwork,
    build_bond_network,
    build_bond_network_from_cif,
    largest_connected_component_with_bonds,
)
from mofbb.fragments import FragmentBuilder


def test_build_bond_network_from_cif_preserves_periodic_bonds_and_types(
    tmp_path: Path,
) -> None:
    cif = tmp_path / "explicit_bonds.cif"
    cif.write_text(
        "\n".join(
            [
                "data_test",
                "_symmetry_space_group_name_H-M    P1",
                "_symmetry_Int_Tables_number       1",
                "loop_",
                "_symmetry_equiv_pos_as_xyz",
                "'x, y, z'",
                "_cell_length_a     10.0",
                "_cell_length_b     10.0",
                "_cell_length_c     10.0",
                "_cell_angle_alpha  90.0",
                "_cell_angle_beta   90.0",
                "_cell_angle_gamma  90.0",
                "loop_",
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_type_partial_charge",
                "C0 C 0.95 0.50 0.50 0.0",
                "O1 O 0.05 0.50 0.50 0.0",
                "loop_",
                "_geom_bond_atom_site_label_1",
                "_geom_bond_atom_site_label_2",
                "_geom_bond_distance",
                "_geom_bond_site_symmetry_2",
                "_ccdc_geom_bond_type",
                "C0 O1 1.000 1_655 A",
                "",
            ]
        ),
        encoding="utf-8",
    )
    atoms = ase.io.read(cif)

    bonds = build_bond_network_from_cif(cif, atoms)

    assert bonds is not None
    assert bonds.pairs == [(0, 1)]
    np.testing.assert_allclose(bonds.vector(0, 1), np.array([1.0, 0.0, 0.0]))
    assert bonds.bond_type(0, 1) == "A"


def test_fragment_builder_preserves_source_bond_types() -> None:
    atoms = Atoms(
        symbols=["C", "O"],
        positions=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0)],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    bonds = BondNetwork(
        adjacency=(frozenset({1}), frozenset({0})),
        vectors={(0, 1): np.array([1.2, 0.0, 0.0])},
        bond_types={(0, 1): "A"},
    )

    fragment = FragmentBuilder(
        atoms,
        bonds,
        atoms.get_chemical_symbols(),
    ).materialize_fragment("linker", 1, {0, 1}, ())

    assert fragment.bonds == ((0, 1, "A"),)


def test_build_bond_network_prunes_extra_carboxylate_carbon_neighbors() -> None:
    atoms = Atoms(
        symbols=["Cd", "O", "O", "C", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.2, 1.2, 0.0),
            (3.2, 0.0, 0.0),
            (4.6, 0.0, 0.0),
            (3.2, -1.65, 0.0),
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )

    bonds = build_bond_network(atoms)

    carbon_neighbors = bonds.adjacency[3]
    assert 4 in carbon_neighbors
    assert 5 not in carbon_neighbors


def test_build_bond_network_prunes_extra_coordinating_oxygen_neighbors() -> None:
    atoms = Atoms(
        symbols=["Cd", "O", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.2, 0.0, 0.0),
            (2.0, 1.65, 0.0),
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )

    bonds = build_bond_network(atoms)

    oxygen_neighbors = bonds.adjacency[1]
    assert 2 in oxygen_neighbors
    assert 3 not in oxygen_neighbors


def test_build_bond_network_prunes_extra_carboxylate_oxygen_carbon_neighbors() -> None:
    atoms = Atoms(
        symbols=["O", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.24, 0.0, 0.0),
            (0.0, 1.62, 0.0),
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )

    bonds = build_bond_network(atoms)

    oxygen_neighbors = bonds.adjacency[0]
    assert 1 in oxygen_neighbors
    assert 2 not in oxygen_neighbors


def test_build_bond_network_prunes_spurious_carboxylate_oxygen_hydrogen() -> None:
    atoms = Atoms(
        symbols=["Cd", "O", "C", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (2.30, 0.0, 0.0),
            (3.56, 0.0, 0.0),
            (2.30, 1.08, 0.0),
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )

    bonds = build_bond_network(atoms)

    oxygen_neighbors = bonds.adjacency[1]
    assert 2 in oxygen_neighbors
    assert 3 not in oxygen_neighbors


def test_build_bond_network_reconnects_nearby_terminal_organic_components() -> None:
    atoms = Atoms(
        symbols=["C", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (2.35, 0.0, 0.0),
            (3.75, 0.0, 0.0),
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )

    bonds = build_bond_network(atoms)

    assert (0, 1) in bonds.pairs
    assert (1, 2) in bonds.pairs


def test_build_bond_network_reconnects_both_terminals_to_same_component() -> None:
    atoms = Atoms(
        symbols=["C", "C", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (0.0, 1.4, 0.0),
            (2.35, 0.0, 0.0),
            (2.35, 1.4, 0.0),
        ],
        cell=[12.0, 12.0, 12.0],
        pbc=False,
    )

    bonds = build_bond_network(atoms)

    assert (0, 2) in bonds.pairs
    assert (1, 3) in bonds.pairs


def test_cleanup_warns_before_dropping_nontrivial_components() -> None:
    atoms = Atoms(
        symbols=["C", "C", "C", "C"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.4, 0.0, 0.0),
            (8.0, 0.0, 0.0),
            (9.4, 0.0, 0.0),
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=False,
    )
    bonds = BondNetwork(
        adjacency=(
            frozenset({1}),
            frozenset({0}),
            frozenset({3}),
            frozenset({2}),
        ),
        vectors={
            (0, 1): np.array([1.4, 0.0, 0.0]),
            (2, 3): np.array([1.4, 0.0, 0.0]),
        },
    )

    with pytest.warns(RuntimeWarning, match="Cleanup dropped disconnected"):
        kept_atoms, kept_bonds = largest_connected_component_with_bonds(atoms, bonds)

    assert len(kept_atoms) == 2
    assert kept_bonds.pairs == [(0, 1)]
