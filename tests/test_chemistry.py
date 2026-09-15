"""Coordination screening at library validation and X substitution."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from mofbb.compose import Builder, BuildingBlock
from mofbb.compose.framework import FrameworkAssembler
from mofbb.compose.pbc import PBC


def write_block(path, symbols, bonds):
    positions = np.zeros((len(symbols), 3))
    angles = np.arange(1, len(symbols)) * 2 * np.pi / max(1, len(symbols) - 1)
    positions[1:, :2] = np.column_stack((np.cos(angles), np.sin(angles)))
    path.write_text(
        f"{len(symbols)}\n\n"
        + "".join(f"{s} {x} {y} {z}\n" for s, (x, y, z) in zip(symbols, positions))
        + "".join(f"{i} {j} S\n" for i, j in bonds)
    )
    return BuildingBlock(path)


@pytest.mark.parametrize(
    "symbol,limit", [("B", 3), ("N", 3), ("O", 2), ("C", 4), ("H", 1), ("F", 1)]
)
def test_known_nonmetal_coordination_is_screened_before_placement(
    tmp_path, symbol, limit
):
    block = write_block(
        tmp_path / "overcoordinated.xyz",
        [symbol] + ["C"] * (limit + 1),
        [(0, i) for i in range(1, limit + 2)],
    )
    with pytest.raises(ValueError, match=rf"atom 0 \({symbol}\).*{limit + 1} nonmetal"):
        block.validate_connectivity()


def test_boron_case_is_detected_even_when_boron_has_no_x_ports(tmp_path):
    # The local B(C)(O)3 environment in o_node2389; its X ports are elsewhere.
    block = write_block(
        tmp_path / "o_node2389_environment.xyz",
        ["B", "C", "O", "O", "O", "X"],
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)],
    )
    topology = SimpleNamespace(n_slots=1, node_indices=[0], cn=[1])
    with (
        patch("mofbb.compose.builder.SlotAssignment") as placement,
        pytest.raises(ValueError, match=r"B\).*4 nonmetal"),
    ):
        Builder().build(topology, [block], require_3d=False)
    placement.assert_not_called()
    assert set(block.bond_types) == {"S"}


@pytest.mark.parametrize("symbol,count", [("B", 3), ("N", 3), ("O", 2)])
def test_metal_neighbors_and_unassigned_x_are_not_counted_as_nonmetals(
    tmp_path, symbol, count
):
    symbols = [symbol] + ["C"] * count + ["Zn", "Zn", "X"]
    block = write_block(
        tmp_path / "donor.xyz", symbols, [(0, i) for i in range(1, len(symbols))]
    )
    assert block.validate_chemistry().nonmetal_counts[0] == count


def test_cached_screen_tracks_graph_changes_but_reuses_rigid_copies(tmp_path):
    block = write_block(
        tmp_path / "block.xyz", ["N", "C", "C", "C", "X"], [(0, i) for i in range(1, 5)]
    )
    profile = block.validate_chemistry()
    copied = block.copy()
    copied.atoms.rotate(37, "z")
    assert copied.validate_chemistry() is profile
    assert not profile.nonmetal_counts.flags.writeable
    copied.atoms[4].symbol = "H"
    with pytest.raises(ValueError, match="4 nonmetal neighbors"):
        copied.validate_chemistry()
    assert block.validate_chemistry() is profile
    block._bonds = np.array([(0, 1), (0, 2), (1, 3), (0, 4)])
    assert block.validate_chemistry().nonmetal_counts[0] == 2


@pytest.mark.parametrize("bonds", [[(0, 1), (1, 2)], [], [(0, 1), (0, 1)]])
def test_invalid_dummy_connectivity_is_rejected(tmp_path, bonds):
    block = write_block(tmp_path / "invalid.xyz", ["C", "X", "X"], bonds)
    # Supply an explicit empty bond graph when the XYZ contains no bond lines.
    block._bonds = np.asarray(bonds, dtype=int).reshape(-1, 2)
    with pytest.raises(ValueError, match="X port|duplicate bonds"):
        block.validate_chemistry()


def fuse(atoms, bonds, pairs, shifts=None):
    assembler = object.__new__(FrameworkAssembler)
    assembler.pbc = PBC()
    assembler.bond_scale = 1.2
    shifts = np.zeros((len(pairs), 3), dtype=int) if shifts is None else shifts
    return assembler._fuse_x_atoms(
        atoms, np.array(bonds), ["S"] * len(bonds), pairs, shifts
    )


def donor_with_partner(symbol, internal_count, partner):
    atoms = Atoms(
        [symbol] + ["C"] * internal_count + [partner, "X", "X"],
        positions=[[0, 0, 0]]
        + [[-1, i, 0] for i in range(internal_count)]
        + [[1.4, 0, 0], [0.7, 0, 0], [0.7, 0, 0]],
        cell=[10] * 3,
        pbc=True,
    )
    n = internal_count
    bonds = [(0, i) for i in range(1, n + 1)] + [(0, n + 2), (n + 1, n + 3)]
    return atoms, bonds, [(n + 2, n + 3)]


@pytest.mark.parametrize(
    "symbol,count,partner", [("B", 3, "O"), ("N", 3, "C"), ("O", 2, "H")]
)
def test_fusion_checks_actual_nonmetal_partner(tmp_path, symbol, count, partner):
    atoms, bonds, pairs = donor_with_partner(symbol, count, partner)
    with pytest.raises(ValueError, match=rf"assembly: atom 0 \({symbol}\)"):
        fuse(atoms, bonds, pairs)
    assert (
        "X" in atoms.get_chemical_symbols()
    )  # A rejected fusion does not delete atoms.


@pytest.mark.parametrize(
    "symbol,count,partner", [("B", 2, "O"), ("N", 3, "Zn"), ("O", 2, "Zn")]
)
def test_fusion_allows_ordinary_coordination_and_keeps_s_bonds(symbol, count, partner):
    atoms, bonds, pairs = donor_with_partner(symbol, count, partner)
    fused, types, shifts = fuse(atoms, bonds, pairs)
    assert "X" not in atoms.get_chemical_symbols()
    assert set(types) == {"S"}
    assert (0, count + 1) in map(tuple, fused)
    assert not shifts.any()


def test_multiple_ports_on_one_anchor_are_checked_together():
    atoms = Atoms("NCCCCXXXX", positions=np.zeros((9, 3)), cell=[10] * 3, pbc=True)
    bonds = [(0, 1), (0, 2), (0, 5), (0, 6), (3, 7), (4, 8)]
    with pytest.raises(ValueError, match=r"N\).*4 nonmetal"):
        fuse(atoms, bonds, [(5, 7), (6, 8)])


def test_periodic_self_bonds_count_both_neighbor_images():
    atoms = Atoms("CXXXXXX", positions=np.zeros((7, 3)), cell=[1.4] * 3, pbc=True)
    with pytest.raises(ValueError, match=r"C\).*6 nonmetal"):
        fuse(
            atoms,
            [(0, i) for i in range(1, 7)],
            [(1, 2), (3, 4), (5, 6)],
            np.eye(3, dtype=int),
        )


@pytest.mark.parametrize("pairs", [[(2, 3), (2, 3)], [], [(2, 2)]])
def test_fusion_requires_each_x_port_exactly_once(pairs):
    atoms = Atoms(
        "CCXX",
        positions=[[0, 0, 0], [1.4, 0, 0], [0.7, 0, 0], [0.7, 0, 0]],
        cell=[10] * 3,
        pbc=True,
    )
    with pytest.raises(ValueError, match="each anchored X port exactly once"):
        fuse(atoms, [(0, 2), (1, 3)], pairs)
