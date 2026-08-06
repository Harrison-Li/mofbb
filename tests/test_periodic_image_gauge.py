"""Regression tests for deterministic periodic atom-image bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
import numpy as np
import pytest

from mofbb.compose import Framework, Reconstructor
from mofbb.decompose.bonding import read_bonded_structure
from mofbb.decompose.pbc import periodic_bonds


def _write_boundary_cif(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "data_test",
                "_symmetry_space_group_name_H-M    P1",
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
                "C0 C 0.9999999999999999 0.5 0.5",
                "O1 O 0.1 0.5 0.5",
                "loop_",
                "_geom_bond_atom_site_label_1",
                "_geom_bond_atom_site_label_2",
                "_geom_bond_distance",
                "_geom_bond_site_symmetry_2",
                "_ccdc_geom_bond_type",
                "C0 O1 1.000 . S",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_cif_ingestion_canonicalizes_boundary_atom_images(tmp_path: Path) -> None:
    atoms, bonds = read_bonded_structure(
        _write_boundary_cif(tmp_path / "boundary-image.cif"),
        cleanup=False,
    )

    assert atoms.get_scaled_positions(wrap=False)[0, 0] == pytest.approx(0.0)
    assert periodic_bonds(atoms, bonds)[0].shift == (0, 0, 0)


def test_reconstructor_canonicalizes_boundary_roundoff(tmp_path: Path) -> None:
    decomposition_dir = tmp_path / "decomposition"
    fragment_dir = decomposition_dir / "metal_nodes"
    fragment_dir.mkdir(parents=True)
    (fragment_dir / "metal_node_001.xyz").write_text(
        "1\n\nC 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    (decomposition_dir / "reconstruction_map.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "topology": {
                    "kind": "node_edge_slots",
                    "cell": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
                },
                "bb_groups": [],
                "slots": [
                    {
                        "slot_id": 0,
                        "kind": "metal_node",
                        "exact_instance_path": "metal_nodes/metal_node_001.xyz",
                        "source_indices": [0],
                        "exact_frac_position": [0.25, 0.25, -1.0e-12],
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )

    framework = Reconstructor(decomposition_dir).reconstruct()

    assert framework.atoms.get_scaled_positions(wrap=False)[0, 2] == pytest.approx(
        0.0
    )


@pytest.mark.filterwarnings(
    "ignore:crystal system 'triclinic' is not interpreted.*"
)
def test_cif_writer_preserves_near_boundary_bond_image(tmp_path: Path) -> None:
    atoms = Atoms(
        symbols=["C", "O"],
        scaled_positions=[[0.9, 0.5, 0.5], [0.999998, 0.5, 0.5]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    framework = Framework(
        atoms,
        bonds=np.asarray([(0, 1)], dtype=int),
        bond_types=["S"],
        info={},
    )

    output = tmp_path / "near-boundary-bond.cif"
    framework.write_cif(output)
    written_atoms, written_bonds = read_bonded_structure(output, cleanup=False)

    assert written_atoms.get_scaled_positions(wrap=False)[1, 0] == pytest.approx(
        0.999998
    )
    assert periodic_bonds(written_atoms, written_bonds)[0].shift == (0, 0, 0)
