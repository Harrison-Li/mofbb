"""Whole-linker regressions, including mixed linkers hidden in a metal block."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from ase import Atoms
from ase.io import read, write
import numpy as np
import pytest

from mofbb.compose.linkers import (
    SINGLE_LINKER_POLICY_VERSION,
    analyze_linker_types,
    validate_single_linker,
)
from mofbb.compose.validate import main as validate_main


MIXED_CDQ = Path(__file__).parent / "data" / "cdq_mixed_linkers.cif"


def cubic_single_linker():
    # Three identical C bridges, each connecting two images of the same Zn.
    return Atoms(
        "ZnCCC", positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
        cell=[4, 4, 4], pbc=True,
    )


def test_reported_cdq_contains_two_complete_linker_types():
    atoms = read(MIXED_CDQ)
    quality = analyze_linker_types(atoms)
    assert quality["num_linker_types"] == 2
    assert quality["num_linkers"] == 4
    assert {item["formula"]: item["count"] for item in quality["linker_types"]} == {
        "C25H16N4O4": 2, "C2N8": 2,
    }
    with pytest.raises(ValueError, match="2 linker types:.*C25H16N4O4.*C2N8"):
        validate_single_linker(atoms)


def test_metal_images_are_distinct_even_with_one_metal_in_the_cell():
    quality = validate_single_linker(cubic_single_linker())
    assert quality["num_linker_types"] == 1
    assert quality["num_linkers"] == 3
    assert quality["single_linker_policy_version"] == SINGLE_LINKER_POLICY_VERSION


def test_identity_survives_atom_reordering_wrapping_and_supercell_expansion():
    atoms = read(MIXED_CDQ)
    expected = analyze_linker_types(atoms)["linker_types"]
    atoms = atoms.repeat((2, 1, 1))
    rng = np.random.default_rng(42)
    atoms = atoms[rng.permutation(len(atoms))]
    atoms.positions += rng.integers(-2, 3, size=(len(atoms), 3)) @ atoms.cell
    atoms.wrap()
    actual = analyze_linker_types(atoms)["linker_types"]
    assert actual == [dict(item, count=2 * item["count"]) for item in expected]


@pytest.mark.parametrize("same_connectivity", [False, True])
def test_molecular_identity_uses_connectivity_not_just_formula(same_connectivity):
    atoms = Atoms("ZnZnNCONCO", cell=[20] * 3, pbc=True)
    # The second NCO has C in the middle, or N in the middle (same formula).
    bonds = [(0, 2), (2, 3), (3, 4), (4, 1), (0, 5), (5, 6), (7, 1)]
    bonds.append((6, 7) if same_connectivity else (5, 7))
    shifts = np.zeros((len(bonds), 3), dtype=int)
    quality = analyze_linker_types(atoms, bonds, shifts)
    assert quality["num_linker_types"] == (1 if same_connectivity else 2)
    assert {item["formula"] for item in quality["linker_types"]} == {"CNO"}


@pytest.mark.parametrize("crosses_cell", [False, True])
def test_unwrap_the_whole_ligand_before_counting_metal_contacts(crosses_cell):
    atoms = Atoms("ZnNCN", cell=[10] * 3, pbc=True)
    bonds = [(0, 1), (1, 2), (2, 3), (3, 0)]
    shifts = np.zeros((4, 3), dtype=int)
    shifts[2, 0] = int(crosses_cell)
    quality = analyze_linker_types(atoms, bonds, shifts)
    # A chelate to one metal image is terminal. Unwrapping the NCN molecule
    # reveals a second metal image only in the boundary-spanning case.
    assert quality["num_linkers"] == int(crosses_cell)


def test_free_guests_single_metal_ligands_and_inorganic_bridges_do_not_count():
    atoms = Atoms("ZnZnCCCO", cell=[20] * 3, pbc=True)
    bonds = [(0, 2), (2, 1), (0, 3), (0, 5), (5, 1)]
    quality = validate_single_linker(atoms, bonds, np.zeros((len(bonds), 3)))
    assert quality["num_linkers"] == 1
    assert quality["ignored_nonbridging_organic_components"] == 2


def test_periodic_organic_network_is_not_a_finite_molecular_linker():
    atoms = Atoms("ZnC", cell=[10] * 3, pbc=True)
    with pytest.raises(ValueError, match="finite organic molecules.*1D"):
        validate_single_linker(atoms, [(0, 1), (1, 1)], [[0, 0, 0], [1, 0, 0]])


@pytest.mark.parametrize("symbols, message", [
    ("C", "requires metal atoms"),
    ("Zn", "0 linker types"),
])
def test_missing_metal_or_linker_cannot_pass(symbols, message):
    with pytest.raises(ValueError, match=message):
        validate_single_linker(Atoms(symbols, cell=[20] * 3, pbc=True))


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("single_linker_mof_generator")


@pytest.mark.parametrize("require_3d", [False, True])
def test_new_build_rejects_mixed_linkers_before_writing(
    tmp_path, monkeypatch, generator, require_3d,
):
    framework = SimpleNamespace(
        atoms=read(MIXED_CDQ), info={"max_rmsd": 0.0}, write_cif=Mock(),
    )
    monkeypatch.setattr(
        generator, "_GENERATION_WORKER_BUILDER",
        SimpleNamespace(build_by_type=lambda **kwargs: framework),
    )
    config = generator.GenerationConfig(
        tmp_path / "manifest", tmp_path, tmp_path,
        require_3d_connectivity=require_3d,
    )
    output = tmp_path / "mixed.cif"
    with pytest.raises(ValueError, match="2 linker types"):
        generator.build_and_validate(config, None, {}, {}, output, topology=object())
    framework.write_cif.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("require_3d", [False, True])
def test_staging_recovery_also_rejects_mixed_linkers(tmp_path, generator, require_3d):
    config = generator.GenerationConfig(
        tmp_path / "manifest", tmp_path, tmp_path,
        require_3d_connectivity=require_3d,
    )
    with pytest.raises(ValueError, match="2 linker types"):
        generator.validate_staged_cif(config, MIXED_CDQ)


def test_actual_generation_and_recovery_accept_one_linker_type(tmp_path, generator):
    topology_path = tmp_path / "pcu.cgd"
    topology_path.write_text(
        "CRYSTAL\n NAME pcu-test\n GROUP P1\n CELL 4 4 4 90 90 90\n"
        " NODE 1 6 0 0 0\n EDGE 0 0 0 1 0 0\n EDGE 0 0 0 0 1 0\n"
        " EDGE 0 0 0 0 0 1\nEND\n"
    )
    metal_path = tmp_path / "metal.xyz"
    metal_path.write_text(
        "7\n1 2 3 4 5 6\nZn 0 0 0\nX 1 0 0\nX -1 0 0\n"
        "X 0 1 0\nX 0 -1 0\nX 0 0 1\nX 0 0 -1\n"
        + "".join(f"0 {i} S\n" for i in range(1, 7))
    )
    linker_path = tmp_path / "linker.xyz"
    linker_path.write_text(
        "5\n3 4\nN -1.2 0 0\nC 0 0 0\nN 1.2 0 0\n"
        "X -2.2 0 0\nX 2.2 0 0\n0 1 S\n1 2 S\n0 3 S\n2 4 S\n"
    )
    topology = generator.Topology(topology_path)
    record = generator.topology_record_from_object(topology, topology_path)
    config = generator.GenerationConfig(tmp_path / "manifest", tmp_path, tmp_path)
    output = tmp_path / "single.cif"
    quality = generator.build_and_validate(
        config, record, {0: metal_path}, {(0, 0): linker_path}, output,
        topology=topology,
    )
    assert quality["num_linker_types"] == 1
    assert quality["num_linkers"] == 3
    recovered = generator.validate_staged_cif(config, output)
    assert recovered["linker_types"] == quality["linker_types"]


@pytest.mark.parametrize("require_3d", [False, True])
def test_changed_linker_policy_invalidates_bulk_runs_and_checkpoints(
    tmp_path, generator, monkeypatch, require_3d,
):
    bulk = importlib.import_module("generate_500k_single_linker_mofs")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("test manifest\n")
    config = generator.GenerationConfig(
        manifest, tmp_path, tmp_path, require_3d_connectivity=require_3d,
    )
    checkpoint_fingerprint = generator.validation_fingerprint(config)
    old_fingerprint = bulk.fingerprint(config, [])
    connection = bulk.connect(tmp_path / "generation.sqlite3")
    try:
        bulk.initialize(connection, old_fingerprint, 1, False, "strict")
        monkeypatch.setattr(generator, "SINGLE_LINKER_POLICY_VERSION", -1)
        assert generator.validation_fingerprint(config) != checkpoint_fingerprint
        current_fingerprint = bulk.fingerprint(config, [])
        assert current_fingerprint != old_fingerprint
        with pytest.raises(RuntimeError, match="fresh output folder"):
            bulk.initialize(connection, current_fingerprint, 1, False, "strict")
    finally:
        connection.close()


def test_cli_can_audit_mixed_linkers_without_altering_the_cif(capsys):
    before = MIXED_CDQ.read_bytes()
    assert validate_main([str(MIXED_CDQ), "--single-linker"]) == 1
    assert "2 linker types" in capsys.readouterr().out
    assert MIXED_CDQ.read_bytes() == before


def test_valid_single_linker_cif_passes_the_cli(tmp_path, capsys):
    path = tmp_path / "single.cif"
    write(path, cubic_single_linker())
    assert validate_main([str(path), "--single-linker"]) == 0
    assert '"num_linker_types": 1' in capsys.readouterr().out
