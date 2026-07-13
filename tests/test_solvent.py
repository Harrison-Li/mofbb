from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from ase import Atoms
from ase.cell import Cell
from ase.io import read
from ase.io.cif import parse_cif

import mofbb.decompose.solvent as solvent_module
from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.decomposer import SingleNodeDecomposer
from mofbb.decompose.solvent import (
    MofidSolventRemover,
    SolventRemovalResult,
    _simplify_topology,
    _TopologyConnection,
    _TopologySite,
)

SOLVENT_ARTIFACT_NAMES = {
    "free_solvent.cif",
    "bound_solvent.cif",
    "mof_fsr.cif",
    "mof_asr.cif",
}
CELL_TAGS = {
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
}


def make_atoms_and_bonds(symbols, pairs, shifts=None):
    positions = np.array(
        [(float(index), 0.0, 0.0) for index in range(len(symbols))]
    )
    atoms = Atoms(symbols, positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)
    adjacency = [set() for _ in symbols]
    vectors = {}
    shifts = shifts or {}
    for i, j in pairs:
        a, b = sorted((i, j))
        adjacency[a].add(b)
        adjacency[b].add(a)
        shift = np.array(shifts.get((a, b), (0, 0, 0)), dtype=float)
        vectors[(a, b)] = positions[b] + shift @ atoms.cell.array - positions[a]
    return atoms, BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )


def make_two_connection_fixture(second_shift):
    return make_atoms_and_bonds(
        ["Zn", "N", "N", "C"],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        shifts={(0, 2): second_shift},
    )


def connection(left: int, right: int, shift=(0, 0, 0)) -> _TopologyConnection:
    return _TopologyConnection(left, right, shift)


def assert_empty_cif(path, *, has_cell_metadata: bool) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"data_{path.stem}")
    assert "loop_" not in text
    assert "_mofbb_atom_count 0" in text

    blocks = list(parse_cif(str(path)))
    assert len(blocks) == 1
    block = blocks[0]
    assert block.name == path.stem
    assert block["_mofbb_atom_count"] == 0
    for tag in CELL_TAGS:
        assert (tag in block) is has_cell_metadata


def test_simplify_topology_classifies_zero_and_one_connected_sites() -> None:
    sites = {
        0: _TopologySite(0, "node", {0}),
        1: _TopologySite(1, "linker", {1, 2}),
        2: _TopologySite(2, "linker", {3}),
    }
    free, bound = _simplify_topology(sites, {connection(0, 1)})
    assert free == ((0,), (3,))
    assert bound == ((1, 2),)


def test_simplify_topology_merges_terminal_node_then_repeats() -> None:
    sites = {
        0: _TopologySite(0, "node", {0}),
        1: _TopologySite(1, "linker", {1, 2, 3}),
        2: _TopologySite(2, "node", {4}),
    }
    free, bound = _simplify_topology(
        sites,
        {connection(0, 1), connection(1, 2)},
    )
    assert free == ((4,),)
    assert bound == ((0, 1, 2, 3),)


def test_periodic_self_connection_prevents_free_solvent_classification() -> None:
    sites = {0: _TopologySite(0, "linker", {0, 1})}
    free, bound = _simplify_topology(
        sites,
        {connection(0, 0, (1, 0, 0))},
    )
    assert free == ()
    assert bound == ()


def test_classifier_repeats_node_merge_and_solvent_deletion() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "N", "C", "N", "Zn"],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.bound_solvent_components == ((0, 1, 2, 3),)
    assert result.free_solvent_components == ((4,),)
    assert result.fsr_indices == (0, 1, 2, 3)
    assert result.asr_indices == ()


def test_distinct_periodic_images_prevent_bound_classification() -> None:
    atoms, bonds = make_two_connection_fixture(second_shift=(1, 0, 0))
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.free_solvent_components == ()
    assert result.bound_solvent_components == ()


def test_solvent_remover_reuses_cached_periodic_bonds(monkeypatch) -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "N"],
        [(0, 1)],
        shifts={(0, 1): (1, 0, 0)},
    )
    calls = []
    original_periodic_bonds = solvent_module.periodic_bonds

    def counting_periodic_bonds(candidate_atoms, candidate_bonds):
        calls.append((candidate_atoms, candidate_bonds))
        return original_periodic_bonds(candidate_atoms, candidate_bonds)

    monkeypatch.setattr(solvent_module, "periodic_bonds", counting_periodic_bonds)

    remover = MofidSolventRemover(atoms, bonds)

    assert len(calls) == 1
    assert calls[0][0] is atoms
    assert calls[0][1] is bonds
    for _ in range(2):
        np.testing.assert_array_equal(remover._bond_shift(0, 1), (1, 0, 0))
        np.testing.assert_array_equal(remover._bond_shift(1, 0), (-1, 0, 0))
    assert len(calls) == 1


def test_redundant_nonperiodic_connections_collapse_to_one() -> None:
    atoms, bonds = make_two_connection_fixture(second_shift=(0, 0, 0))
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.bound_solvent_components == ((1, 2, 3),)
    assert result.free_solvent_components == ((0,),)


def test_subset_maps_clean_indices_to_original_indices(
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    result = MofidSolventRemover(atoms, bonds).classify()
    selected_atoms, selected_bonds, mapping = result.subset("all")
    assert mapping == result.asr_indices
    assert len(selected_atoms) == len(result.asr_indices)
    assert len(selected_bonds.adjacency) == len(selected_atoms)


@pytest.mark.parametrize(
    ("mode", "expected_mapping"),
    [("free", "fsr_indices"), ("all", "asr_indices")],
)
def test_single_node_decomposer_selects_solvent_mode(
    monkeypatch,
    mode,
    expected_mapping,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    cleanup_values = []
    classify_calls = []
    original_classify = MofidSolventRemover.classify

    def fake_read(*args, cleanup, **kwargs):
        cleanup_values.append(cleanup)
        return atoms.copy(), bonds

    def counting_classify(remover):
        classify_calls.append(remover)
        return original_classify(remover)

    monkeypatch.setattr("mofbb.decompose.decomposer.read_bonded_structure", fake_read)
    monkeypatch.setattr(
        "mofbb.decompose.decomposer.MofidSolventRemover.classify", counting_classify
    )

    result = SingleNodeDecomposer("fixture.cif", solvent_mode=mode).decompose()

    assert cleanup_values == [False]
    assert len(classify_calls) == 1
    assert result.solvent_mode == mode
    assert result.solvent is not None
    assert result.selected_to_original == getattr(result.solvent, expected_mapping)


def test_single_node_rejects_invalid_solvent_mode_before_reading_cif(
    monkeypatch,
) -> None:
    read_called = False

    def fake_read(*args, **kwargs):
        nonlocal read_called
        read_called = True
        raise AssertionError("CIF read should not be attempted")

    monkeypatch.setattr("mofbb.decompose.decomposer.read_bonded_structure", fake_read)

    with pytest.raises(ValueError, match="solvent mode"):
        SingleNodeDecomposer("fixture.cif", solvent_mode="invalid")

    assert read_called is False


@pytest.mark.parametrize("cleanup", [True, False])
def test_single_node_default_mode_preserves_cleanup_and_legacy_state(
    monkeypatch,
    cleanup,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    cleanup_values = []

    def fake_read(*args, cleanup, **kwargs):
        cleanup_values.append(cleanup)
        return atoms.copy(), bonds

    monkeypatch.setattr("mofbb.decompose.decomposer.read_bonded_structure", fake_read)

    result = SingleNodeDecomposer("fixture.cif", cleanup=cleanup).decompose()

    assert cleanup_values == [cleanup]
    assert result.solvent_mode == "none"
    assert result.solvent is None
    assert result.selected_to_original is None


def test_result_write_includes_solvent_block_when_enabled(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )

    result = SingleNodeDecomposer("fixture.cif", solvent_mode="all").decompose()
    assert result.selected_to_original is not None
    selected_indices = set(range(len(result.selected_to_original)))

    assert set(result.node_indices) | set(result.linker_indices) == selected_indices
    assert set(result.node_indices).isdisjoint(result.linker_indices)
    assert all(
        set(fragment.source_indices) <= selected_indices
        for fragment in result.fragments
    )

    output_dir = tmp_path / "enabled"
    summary = result.write(output_dir)

    assert summary["solvent"]["mode"] == "all"
    assert summary["solvent"]["selected_to_original"] == list(
        result.selected_to_original
    )
    assert {path.name for path in output_dir.glob("*.cif")} == SOLVENT_ARTIFACT_NAMES


@pytest.mark.parametrize(
    "inconsistent_state",
    ["missing-solvent", "missing-mapping", "metadata-with-none-mode"],
)
def test_result_write_rejects_inconsistent_solvent_metadata(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
    inconsistent_state,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = SingleNodeDecomposer("fixture.cif", solvent_mode="all").decompose()
    if inconsistent_state == "missing-solvent":
        result.solvent = None
    elif inconsistent_state == "missing-mapping":
        result.selected_to_original = None
    else:
        result.solvent_mode = "none"
    output_dir = tmp_path / inconsistent_state

    with pytest.raises(ValueError, match="Inconsistent solvent metadata"):
        result.write(output_dir)

    assert not output_dir.exists()


def test_result_write_preflights_wrong_mapping_before_output(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = SingleNodeDecomposer("fixture.cif", solvent_mode="all").decompose()
    assert result.selected_to_original is not None
    result.selected_to_original = tuple(reversed(result.selected_to_original))
    output_dir = tmp_path / "wrong-mapping"

    with pytest.raises(
        ValueError,
        match="selected-to-original mapping is inconsistent with mode 'all'",
    ):
        result.write(output_dir)

    assert not output_dir.exists()


def test_result_write_preflights_unrepresentable_solvent_cell_before_output(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = SingleNodeDecomposer("fixture.cif", solvent_mode="all").decompose()
    assert result.solvent is not None
    result.solvent.original_atoms.set_pbc(False)
    output_dir = tmp_path / "unrepresentable-cell"

    with pytest.raises(
        ValueError,
        match="CIF output requires either a rank-3 cell with full periodicity",
    ):
        result.write(output_dir)

    assert not output_dir.exists()


def test_internal_offsets_collapse_equivalent_periodic_connections() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "N", "N", "C"],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        shifts={(0, 2): (1, 0, 0), (2, 3): (-1, 0, 0)},
    )
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.bound_solvent_components == ((1, 2, 3),)
    assert result.free_solvent_components == ((0,),)


def test_reversed_site_endpoints_invert_periodic_translation() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "N", "C", "C"],
        [(0, 1), (0, 3), (1, 2), (2, 3)],
        shifts={(0, 1): (1, 0, 0), (2, 3): (-1, 0, 0)},
    )
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.bound_solvent_components == ((1, 2),)
    assert result.free_solvent_components == ((0, 3),)


def test_internal_periodic_cycle_prevents_free_classification() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["C", "C", "C"],
        [(0, 1), (0, 2), (1, 2)],
        shifts={(1, 2): (1, 0, 0)},
    )
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.free_solvent_components == ()
    assert result.bound_solvent_components == ()


def test_solvent_products_conserve_original_atoms(
    framework_with_free_and_bound_solvent,
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    original = set(range(len(result.original_atoms)))
    assert set(result.fsr_indices) | set(result.free_solvent_indices) == original
    assert (
        set(result.asr_indices)
        | set(result.free_solvent_indices)
        | set(result.bound_solvent_indices)
    ) == original
    assert set(result.free_solvent_indices).isdisjoint(
        result.bound_solvent_indices
    )


def test_free_subset_maps_clean_indices_to_original_indices(
    framework_with_free_and_bound_solvent,
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    selected_atoms, selected_bonds, mapping = result.subset("free")
    assert mapping == result.fsr_indices
    assert len(selected_atoms) == len(result.fsr_indices)
    assert len(selected_bonds.adjacency) == len(selected_atoms)


def test_solvent_removal_result_is_immutable(
    framework_with_free_and_bound_solvent,
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    with pytest.raises(FrozenInstanceError):
        result.free_solvent_components = ()


def test_subset_rejects_invalid_mode(framework_with_free_and_bound_solvent) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    with pytest.raises(ValueError, match="Unsupported active solvent mode"):
        result.subset("none")


@pytest.mark.parametrize("mode", ["free", "all"])
def test_subset_rejects_entirely_deleted_structure(mode) -> None:
    atoms, bonds = make_atoms_and_bonds(["C"], [])
    result = MofidSolventRemover(atoms, bonds).classify()
    with pytest.raises(ValueError, match="deleted the entire structure"):
        result.subset(mode)


def test_classifier_rejects_mismatched_bond_network_size() -> None:
    atoms = Atoms(["Zn", "C"], positions=np.zeros((2, 3)))
    bonds = BondNetwork(adjacency=(frozenset(),), vectors={})
    with pytest.raises(ValueError, match="does not match the atom count"):
        MofidSolventRemover(atoms, bonds)


@pytest.mark.parametrize(
    ("neighbor", "message"),
    [
        ("1", "integer atom index"),
        (0, "self-neighbor"),
        (2, "out of range"),
    ],
)
def test_classifier_rejects_invalid_adjacency_neighbor(neighbor, message) -> None:
    atoms = Atoms(["Zn", "C"], positions=np.zeros((2, 3)))
    bonds = BondNetwork(
        adjacency=(frozenset({neighbor}), frozenset()),
        vectors={},
    )

    with pytest.raises(ValueError, match=message):
        MofidSolventRemover(atoms, bonds)


def test_classifier_rejects_asymmetric_adjacency() -> None:
    atoms = Atoms(["Zn", "C"], positions=np.zeros((2, 3)))
    bonds = BondNetwork(
        adjacency=(frozenset({1}), frozenset()),
        vectors={(0, 1): np.zeros(3)},
    )

    with pytest.raises(ValueError, match=r"Adjacency.*not symmetric.*\(0, 1\)"):
        MofidSolventRemover(atoms, bonds)


@pytest.mark.parametrize("key", [(0,), (0, "1"), (1, 0), (0, 0)])
def test_classifier_rejects_invalid_vector_key(key) -> None:
    atoms = Atoms(["Zn", "C"], positions=np.zeros((2, 3)))
    bonds = BondNetwork(
        adjacency=(frozenset(), frozenset()),
        vectors={key: np.zeros(3)},
    )

    with pytest.raises(ValueError, match="Vector key.*canonical"):
        MofidSolventRemover(atoms, bonds)


def test_classifier_rejects_vector_endpoint_out_of_range() -> None:
    atoms = Atoms(["Zn", "C"], positions=np.zeros((2, 3)))
    bonds = BondNetwork(
        adjacency=(frozenset(), frozenset()),
        vectors={(0, 2): np.zeros(3)},
    )

    with pytest.raises(ValueError, match="Vector key.*out of range"):
        MofidSolventRemover(atoms, bonds)


def test_classifier_rejects_adjacency_edge_without_vector() -> None:
    atoms = Atoms(["Zn", "C"], positions=np.zeros((2, 3)))
    bonds = BondNetwork(
        adjacency=(frozenset({1}), frozenset({0})),
        vectors={},
    )

    with pytest.raises(ValueError, match=r"Adjacency bond.*missing.*vector.*\(0, 1\)"):
        MofidSolventRemover(atoms, bonds)


def test_classifier_rejects_vector_edge_without_adjacency() -> None:
    atoms = Atoms(["Zn", "C"], positions=np.zeros((2, 3)))
    bonds = BondNetwork(
        adjacency=(frozenset(), frozenset()),
        vectors={(0, 1): np.zeros(3)},
    )

    with pytest.raises(ValueError, match=r"Vector bond.*missing.*adjacency.*\(0, 1\)"):
        MofidSolventRemover(atoms, bonds)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("invalid", "Bond type key.*canonical"),
        ((1, 0), "Bond type key.*canonical"),
        ((0, 2), "Bond type key.*out of range"),
        ((0, 1), "Bond type bond.*missing.*vector"),
    ],
)
def test_classifier_rejects_invalid_bond_type_key(key, message) -> None:
    atoms = Atoms(["Zn", "C"], positions=np.zeros((2, 3)))
    bonds = BondNetwork(
        adjacency=(frozenset(), frozenset()),
        vectors={},
        bond_types={key: "S"},
    )

    with pytest.raises(ValueError, match=message):
        MofidSolventRemover(atoms, bonds)


def test_periodic_classifier_rejects_singular_cell() -> None:
    atoms, bonds = make_atoms_and_bonds(["Zn", "N"], [(0, 1)])
    atoms.set_cell(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="periodic cell"):
        MofidSolventRemover(atoms, bonds).classify()


def test_periodic_classifier_rejects_singular_cell_without_bonds() -> None:
    atoms, bonds = make_atoms_and_bonds(["C"], [])
    atoms.set_cell(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="periodic cell"):
        MofidSolventRemover(atoms, bonds).classify()


def test_periodic_classifier_rejects_non_integral_shift() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "C"], [(0, 1)], shifts={(0, 1): (0.5, 0, 0)}
    )
    with pytest.raises(ValueError, match="non-integral periodic translation"):
        MofidSolventRemover(atoms, bonds).classify()


def test_periodic_classifier_rejects_nonperiodic_axis_shift() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "C"], [(0, 1)], shifts={(0, 1): (0, 1, 0)}
    )
    atoms.set_pbc((True, False, True))
    with pytest.raises(ValueError, match="non-periodic axis"):
        MofidSolventRemover(atoms, bonds).classify()


def test_nonperiodic_classifier_rejects_vector_displacement_mismatch() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "C"], [(0, 1)], shifts={(0, 1): (1, 0, 0)}
    )
    atoms.set_pbc(False)
    with pytest.raises(ValueError, match="non-periodic axis"):
        MofidSolventRemover(atoms, bonds).classify()


def test_write_emits_mofid_named_outputs_and_exact_summary(
    tmp_path, framework_with_free_and_bound_solvent
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    _, _, mapping = result.subset("all")

    summary = result.write(tmp_path, "all", mapping)

    assert {path.name for path in tmp_path.iterdir()} == SOLVENT_ARTIFACT_NAMES
    assert summary == {
        "mode": "all",
        "free": {
            "num_components": len(result.free_solvent_components),
            "num_atoms": len(result.free_solvent_indices),
            "components": [list(part) for part in result.free_solvent_components],
            "path": "free_solvent.cif",
        },
        "bound": {
            "num_components": len(result.bound_solvent_components),
            "num_atoms": len(result.bound_solvent_indices),
            "components": [list(part) for part in result.bound_solvent_components],
            "path": "bound_solvent.cif",
        },
        "fsr": {
            "num_atoms": len(result.fsr_indices),
            "path": "mof_fsr.cif",
        },
        "asr": {
            "num_atoms": len(result.asr_indices),
            "path": "mof_asr.cif",
        },
        "selected_to_original": list(mapping),
    }

    selections = {
        "free_solvent.cif": result.free_solvent_indices,
        "bound_solvent.cif": result.bound_solvent_indices,
        "mof_fsr.cif": result.fsr_indices,
        "mof_asr.cif": result.asr_indices,
    }
    for filename, indices in selections.items():
        assert indices
        written = read(tmp_path / filename)
        expected = result.original_atoms[list(indices)]
        assert len(written) == len(expected)
        assert written.get_chemical_formula() == expected.get_chemical_formula()
        assert written.get_chemical_symbols() == expected.get_chemical_symbols()
        assert np.allclose(written.cell.cellpar(), expected.cell.cellpar())
        translation = (
            written.get_scaled_positions(wrap=False)
            - expected.get_scaled_positions(wrap=False)
        )
        assert np.allclose(translation, np.rint(translation))
        assert np.array_equal(written.pbc, expected.pbc)


def test_empty_solvent_files_are_metadata_only_valid_cif(
    tmp_path, framework_with_free_and_bound_solvent
) -> None:
    classified = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    atoms, bonds, _ = classified.subset("all")
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.free_solvent_components == ()
    assert result.bound_solvent_components == ()

    mapping = tuple(range(len(result.original_atoms)))
    summary = result.write(tmp_path, "all", mapping)

    assert summary["free"]["num_components"] == 0
    assert summary["free"]["num_atoms"] == 0
    assert summary["free"]["components"] == []
    assert summary["bound"]["num_components"] == 0
    assert summary["bound"]["num_atoms"] == 0
    assert summary["bound"]["components"] == []
    for stem in ("free_solvent", "bound_solvent"):
        assert_empty_cif(tmp_path / f"{stem}.cif", has_cell_metadata=True)

    for filename in ("mof_fsr.cif", "mof_asr.cif"):
        written = read(tmp_path / filename)
        assert len(written) == len(result.original_atoms)
        assert (
            written.get_chemical_formula()
            == result.original_atoms.get_chemical_formula()
        )


def test_write_uses_valid_rank_zero_placeholders_for_empty_fsr_and_asr(
    tmp_path,
) -> None:
    atoms, bonds = make_atoms_and_bonds(["C"], [])
    atoms.set_cell(np.zeros((3, 3)))
    atoms.set_pbc(False)
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.free_solvent_indices == (0,)
    assert result.fsr_indices == ()
    assert result.asr_indices == ()

    summary = result.write(tmp_path, "all", ())

    assert summary["fsr"]["num_atoms"] == 0
    assert summary["asr"]["num_atoms"] == 0
    assert {path.name for path in tmp_path.iterdir()} == SOLVENT_ARTIFACT_NAMES
    for stem in ("bound_solvent", "mof_fsr", "mof_asr"):
        assert_empty_cif(tmp_path / f"{stem}.cif", has_cell_metadata=False)
    written_free = read(tmp_path / "free_solvent.cif")
    assert written_free.get_chemical_symbols() == ["C"]
    assert written_free.cell.rank == 0
    assert not np.any(written_free.pbc)


def test_write_preserves_unwrapped_fractional_coordinates_in_rotated_cell(
    tmp_path,
) -> None:
    canonical = Cell.fromcellpar([4.0, 5.0, 6.0, 80.0, 75.0, 70.0])
    theta = np.deg2rad(31.0)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_cell = Cell(canonical.array @ rotation.T)
    scaled_positions = np.array([[1.25, -0.2, 0.4], [-0.1, 0.3, 1.7]])
    atoms = Atoms(
        ["C", "N"],
        scaled_positions=scaled_positions,
        cell=rotated_cell,
        pbc=True,
    )
    bonds = BondNetwork(
        adjacency=(frozenset(), frozenset()),
        vectors={},
    )
    result = SolventRemovalResult(atoms, bonds, (), ())

    result.write(tmp_path, "all", (0, 1))

    path = tmp_path / "mof_asr.cif"
    block = list(parse_cif(str(path)))[0]
    raw_scaled = np.column_stack(
        [block[f"_atom_site_fract_{axis}"] for axis in "xyz"]
    )
    assert np.allclose(raw_scaled, scaled_positions)

    written = read(path)
    assert written.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert np.allclose(written.cell.cellpar(), atoms.cell.cellpar())
    translation = written.get_scaled_positions(wrap=False) - scaled_positions
    assert np.allclose(translation, np.rint(translation))


@pytest.mark.parametrize(
    ("cell", "pbc"),
    [
        pytest.param(np.eye(3) * 10.0, False, id="rank-3-nonperiodic"),
        pytest.param(
            np.eye(3) * 10.0,
            (True, False, True),
            id="rank-3-mixed-periodicity",
        ),
        pytest.param(
            [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            (True, False, False),
            id="rank-1",
        ),
        pytest.param(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]],
            (True, True, False),
            id="rank-2",
        ),
        pytest.param(np.zeros((3, 3)), True, id="rank-0-periodic"),
    ],
)
def test_write_rejects_cif_unrepresentable_cell_and_pbc_before_output(
    tmp_path, cell, pbc
) -> None:
    atoms = Atoms(["C"], positions=[(0.0, 0.0, 0.0)], cell=cell, pbc=pbc)
    bonds = BondNetwork(adjacency=(frozenset(),), vectors={})
    result = SolventRemovalResult(atoms, bonds, (), ())
    output_dir = tmp_path / "solvent"

    with pytest.raises(
        ValueError,
        match=(
            "CIF output requires either a rank-3 cell with full periodicity "
            "or a rank-0 cell with no periodicity"
        ),
    ):
        result.write(output_dir, "all", (0,))

    assert not output_dir.exists()


def test_write_accepts_free_mode_mapping_and_emits_only_fixed_artifacts(
    tmp_path, framework_with_free_and_bound_solvent
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    _, _, mapping = result.subset("free")

    summary = result.write(tmp_path, "free", mapping)

    assert summary["mode"] == "free"
    assert summary["selected_to_original"] == list(result.fsr_indices)
    assert {path.name for path in tmp_path.iterdir()} == SOLVENT_ARTIFACT_NAMES


def test_write_rejects_invalid_active_mode_before_emitting_files(
    tmp_path, framework_with_free_and_bound_solvent
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()

    with pytest.raises(ValueError, match="Unsupported active solvent mode"):
        result.write(tmp_path, "none", result.asr_indices)

    assert list(tmp_path.glob("*.cif")) == []


def test_write_rejects_mapping_inconsistent_with_active_mode(
    tmp_path, framework_with_free_and_bound_solvent
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()

    with pytest.raises(ValueError, match="selected-to-original mapping.*mode 'all'"):
        result.write(tmp_path, "all", result.fsr_indices)

    assert list(tmp_path.glob("*.cif")) == []
