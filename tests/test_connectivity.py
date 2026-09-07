"""Regression coverage for dimension, image preservation, and chemistry gates."""

from collections import namedtuple
import importlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

from mofbb.compose import Builder, BuildingBlock, Framework, Topology
from mofbb.compose.cli import main as compose_main
from mofbb.compose.connectivity import periodic_components, validate_3d_connectivity
from mofbb.compose.pbc import PBC
from mofbb.compose.validate import main as validate_main


def cubic_atoms():
    return Atoms("C", positions=[[0, 0, 0]], cell=[1.4] * 3, pbc=True)


@pytest.mark.parametrize("dimension", [0, 1, 2, 3])
def test_single_atom_periodic_graph_counts_self_image_bonds(dimension):
    components = periodic_components(
        1, [(0, 0)] * dimension, np.eye(3, dtype=int)[:dimension]
    )
    assert components[0]["rank"] == dimension


def test_parallel_edges_to_different_images_preserve_rank():
    components = periodic_components(
        2, [(0, 1)] * 4, [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    )
    assert components[0]["rank"] == 3


def test_differently_oriented_layers_do_not_combine_into_3d():
    atoms = Atoms("CC", positions=[[0, 0, 0], [.4, .4, .4]], cell=[1.4]*3, pbc=True)
    with pytest.raises(ValueError, match="3D connectivity required.*2D"):
        validate_3d_connectivity(
            atoms, [(0, 0), (0, 0), (1, 1), (1, 1)],
            [(1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)],
        )


def test_interpenetrated_3d_components_are_allowed():
    atoms = Atoms("CC", positions=[[0, 0, 0], [.7, .7, .7]], cell=[1.4]*3, pbc=True)
    result = validate_3d_connectivity(
        atoms, [(0, 0)]*3 + [(1, 1)]*3, np.tile(np.eye(3, dtype=int), (2, 1))
    )
    assert result["periodic_component_ranks"] == [3, 3]


def test_a_3d_component_cannot_hide_a_disconnected_layer():
    atoms = Atoms("CC", positions=[[0, 0, 0], [.7, .7, .7]], cell=[1.4]*3, pbc=True)
    with pytest.raises(ValueError, match="2D"):
        validate_3d_connectivity(
            atoms, [(0, 0)]*3 + [(1, 1)]*2,
            np.concatenate((np.eye(3, dtype=int), np.eye(3, dtype=int)[:2])),
        )


def test_inferred_metal_contacts_do_not_supply_framework_connectivity():
    atoms = Atoms("Zn", positions=[[0, 0, 0]], cell=[2.243]*3, pbc=True)
    with pytest.raises(ValueError, match="0D"):
        validate_3d_connectivity(atoms)


@pytest.mark.parametrize("cutoff, closest", [(1.0, None), (1.4, None), (3.0, 1.4)])
def test_combined_search_reports_minimum_distance_without_changing_rank(cutoff, closest):
    atoms = cubic_atoms()
    expected = validate_3d_connectivity(atoms)
    with patch("mofbb.compose.connectivity.neighbor_list", wraps=neighbor_list) as search:
        quality = validate_3d_connectivity(atoms, minimum_distance_cutoff=cutoff)
    assert search.call_count == 1
    assert quality.pop("minimum_atom_distance_angstrom") == closest
    assert quality == expected


def test_wider_minimum_distance_search_cannot_connect_a_layer():
    atoms = Atoms("C", cell=[1.4, 1.4, 2.5], pbc=True)
    with pytest.raises(ValueError, match="3D connectivity required.*2D"):
        validate_3d_connectivity(atoms, minimum_distance_cutoff=3.0)


def test_combined_search_keeps_valid_bonds_beyond_distance_reporting_cutoff():
    atoms = Atoms("CsI", scaled_positions=[[0, 0, 0], [.5, .5, .5]],
                  cell=[4.0]*3, pbc=True)
    quality = validate_3d_connectivity(atoms, minimum_distance_cutoff=3.0)
    assert quality["periodic_component_ranks"] == [3]
    assert quality["minimum_atom_distance_angstrom"] is None


def test_minimum_distance_includes_metal_contacts_excluded_from_bond_inference():
    atoms = cubic_atoms() + Atoms("ZnZn", positions=[[.5, .5, .5], [.5, .5, .55]])
    quality = validate_3d_connectivity(atoms, minimum_distance_cutoff=3.0)
    assert quality["periodic_component_ranks"] == [3]
    assert quality["minimum_atom_distance_angstrom"] == pytest.approx(.05)


def test_supported_bonds_inside_metal_blocks_are_retained():
    # Explicit source bonds can retain a metal connection; the coordinate-only
    # screen deliberately cannot infer that support from proximity alone.
    atoms = Atoms("Zn", positions=[[0, 0, 0]], cell=[2.243]*3, pbc=True)
    assert validate_3d_connectivity(
        atoms, [(0, 0)]*3, np.eye(3, dtype=int)
    )["periodic_component_ranks"] == [3]


def test_cell_expansion_breaks_a_declared_3d_network():
    atoms = cubic_atoms()
    bonds, shifts = [(0, 0)]*3, np.eye(3, dtype=int)
    assert validate_3d_connectivity(atoms, bonds, shifts)["periodic_component_ranks"] == [3]
    atoms.cell[2] *= 3
    with pytest.raises(ValueError, match="2D"):
        validate_3d_connectivity(atoms, bonds, shifts)


def test_cif_rounding_cannot_silently_remove_a_bridge(tmp_path):
    atoms = Atoms(
        "CC", scaled_positions=[[0, 0, 0], [0, 0, .62]],
        cell=[1.4, 1.4, 2.9419], pbc=True,
    )
    framework = Framework(
        atoms, np.array([(0, 0), (0, 0), (0, 1), (1, 0)]), ["S"]*4,
        {"require_3d": True},
        bond_shifts=[[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
    )
    assert framework.validate_3d()["periodic_component_ranks"] == [3]
    path = tmp_path / "rounded_layer.cif"
    with pytest.raises(ValueError, match="2D"):
        framework.write_cif(path)
    assert not path.exists()


def test_periodic_self_bond_gate_measures_the_neighbor_cell():
    atoms = Atoms("CXX", positions=[[0, 0, 0], [0, 0, -2], [0, 0, 2]],
                  cell=[10, 10, 4], pbc=True)
    failures = PBC.calc_fail_connection(atoms, [(1, 2)], {1: 0, 2: 0})
    assert failures[0][2] == pytest.approx(4.0)


def test_cif_writer_keeps_all_three_periodic_self_images(tmp_path):
    framework = Framework(
        cubic_atoms(), np.array([(0, 0)]*3), ["S"]*3, {},
        bond_shifts=np.eye(3, dtype=int),
    )
    path = framework.write_cif(tmp_path / "self.cif")
    atoms = read(path, store_tags=True)
    assert set(atoms.info["_geom_bond_site_symmetry_2"]) == {"1_655", "1_565", "1_556"}
    assert atoms.info["_geom_bond_distance"] == pytest.approx([1.4]*3)


def test_wrap_and_serialization_preserve_nonminimum_image_bond(tmp_path):
    atoms = Atoms("CC", scaled_positions=[[1.9, .2, .2], [-.9, .2, .2]],
                  cell=[10]*3, pbc=True)
    framework = Framework(atoms, np.array([(0, 1)]), ["S"], {}, bond_shifts=[[4, 0, 0]])
    np.testing.assert_array_equal(framework.bond_shifts, [[2, 0, 0]])
    path = framework.write_cif(tmp_path / "nonminimum.cif")
    reread = read(path, store_tags=True)
    assert reread.info["_geom_bond_distance"] == pytest.approx([12.0])
    assert reread.info["_geom_bond_site_symmetry_2"] == ["1_755"]


def make_carbon_net(tmp_path, dimension):
    # A primitive cell has one atom and bonds to its own images. This exposes
    # the same image-loss bug as hms without depending on the external library.
    lengths = [1.4 if k < dimension else 5 for k in range(3)]
    vectors = np.eye(3, dtype=int)[:dimension]
    topology_path = tmp_path / "net.cgd"
    topology_path.write_text(
        "CRYSTAL\n NAME synthetic\n GROUP P1\n CELL "
        + " ".join(map(str, lengths)) + " 90 90 90\n"
        + f" NODE 1 {2*dimension} 0 0 0\n"
        + "".join(" EDGE 0 0 0 " + " ".join(map(str, v)) + "\n" for v in vectors)
        + "END\n"
    )
    cap_positions = np.concatenate((.7*vectors, -.7*vectors))
    block_path = tmp_path / "carbon.xyz"
    count = len(cap_positions)
    block_path.write_text(
        f"{count+1}\n" + " ".join(str(i) for i in range(1, count+1))
        + "\nC 0 0 0\n"
        + "".join("X " + " ".join(map(str, v)) + "\n" for v in cap_positions)
        + "".join(f"0 {i} S\n" for i in range(1, count+1))
    )
    return topology_path, block_path


@pytest.mark.parametrize("dimension", [1, 2])
def test_builder_rejects_low_dimensional_net_by_default(tmp_path, dimension):
    topology_path, block_path = make_carbon_net(tmp_path, dimension)
    with patch("mofbb.compose.builder.SlotAssignment") as placement:
        with pytest.raises(ValueError, match=f"{dimension}D"):
            Builder().build_by_type(Topology(topology_path), {0: BuildingBlock(block_path)})
    placement.assert_not_called()


def test_topology_rank_is_cached_and_refreshed_on_reload(tmp_path):
    topology_path, _ = make_carbon_net(tmp_path, 3)
    topology = Topology(topology_path)
    with patch("mofbb.compose.topology.periodic_components", wraps=periodic_components) as analysis:
        topology.validate_3d()
        topology.validate_3d()
        assert analysis.call_count == 1
        topology_path, _ = make_carbon_net(tmp_path, 2)
        topology.atoms = Topology(topology_path).atoms
        topology.update_properties()
        with pytest.raises(ValueError, match="2D"):
            topology.validate_3d()
        assert analysis.call_count == 2


def test_disconnected_building_block_is_rejected_before_placement(tmp_path):
    topology_path, block_path = make_carbon_net(tmp_path, 3)
    block = BuildingBlock(block_path)
    block.atoms += Atoms("C", positions=[[.5, .5, .5]])
    with patch("mofbb.compose.builder.SlotAssignment") as placement:
        with pytest.raises(ValueError, match="connected real-atom component"):
            Builder().build_by_type(Topology(topology_path), {0: block})
    placement.assert_not_called()


def test_a_dummy_port_cannot_connect_two_real_fragments(tmp_path):
    block_path = tmp_path / "disconnected.xyz"
    block_path.write_text("3\n\nC -1 0 0\nC 1 0 0\nX 0 0 0\n0 2 S\n1 2 S\n")
    with pytest.raises(ValueError, match="connected real-atom component"):
        BuildingBlock(block_path).validate_connectivity()


def test_provided_assignment_cannot_reuse_a_port(tmp_path):
    topology_path, block_path = make_carbon_net(tmp_path, 3)
    with pytest.raises(ValueError, match="use each X port once"):
        Builder().build_by_type(
            Topology(topology_path), {0: BuildingBlock(block_path)},
            permutations={0: [0, 1, 2, 3, 4, 4]},
        )


@pytest.mark.parametrize("insert_linker", [False, True])
def test_assembly_uses_topology_images_even_when_slot_coordinates_are_unwrapped(
    tmp_path, insert_linker,
):
    topology_path, block_path = make_carbon_net(tmp_path, 3)
    if insert_linker:
        topology_path.write_text(topology_path.read_text().replace("1.4", "2.8"))
    topology = Topology(topology_path)
    # Changing which cells represent slots must not change the net or which
    # images supply its bonds. This also exercises linker-center image choices.
    images = np.array([[1, -1, 0], [0, 2, -1], [-1, 0, 1], [2, -1, 0]])
    topology.atoms.positions += images @ topology.atoms.cell
    edges = {}
    if insert_linker:
        edge_path = tmp_path / "linker.xyz"
        edge_path.write_text("3\n\nC 0 0 0\nX -.7 0 0\nX .7 0 0\n0 1 S\n0 2 S\n")
        edges[(0, 0)] = BuildingBlock(edge_path)
    with patch(
        "mofbb.compose.framework.minimum_image_shifts",
        side_effect=AssertionError("Assembly must use declared topology images"),
    ):
        framework = Builder().build_by_type(topology, {0: BuildingBlock(block_path)}, edges)
    assert len(framework.bonds) == (6 if insert_linker else 3)
    assert validate_3d_connectivity(framework.atoms)["periodic_component_ranks"] == [3]
    path = framework.write_cif(tmp_path / "unwrapped.cif")
    assert validate_main([str(path)]) == 0


def test_generator_excludes_layer_topologies_before_candidate_matching(tmp_path, generator):
    make_carbon_net(tmp_path, 2)
    config = generator.GenerationConfig(tmp_path / "manifest", tmp_path, tmp_path / "out")
    with patch.object(generator, "topology_record_from_object") as classify:
        records, skipped = generator.load_eligible_topologies(config, {}, topology_mode="uninodal")
    assert not records
    assert skipped["non_3d_topology"] == 1
    classify.assert_not_called()


def test_builder_preserves_real_3d_periodic_connections_and_rechecks_write(tmp_path):
    topology_path, block_path = make_carbon_net(tmp_path, 3)
    framework = Builder().build_by_type(Topology(topology_path), {0: BuildingBlock(block_path)})
    assert framework.info["periodic_component_ranks"] == [3]
    assert len(framework.bonds) == 3
    output = framework.write_cif(tmp_path / "good.cif")
    assert validate_main([str(output)]) == 0
    framework.atoms.cell[2] *= 3
    rejected = tmp_path / "bad.cif"
    with pytest.raises(ValueError, match="2D"):
        framework.write_cif(rejected)
    assert not rejected.exists()


def test_cli_does_not_write_layer_by_default(tmp_path):
    topology_path, block_path = make_carbon_net(tmp_path, 2)
    output = tmp_path / "layer.cif"
    args = [str(topology_path), "--node-bb", f"0={block_path}", "-o", str(output)]
    with pytest.raises(ValueError, match="2D"):
        compose_main(args)
    assert not output.exists()
    assert compose_main(args + ["--allow-low-dimensional"]) == 0
    assert validate_main([str(output)]) == 1


def test_new_metal_metal_fusion_is_rejected_without_blacklisting_a_topology(tmp_path):
    topology_path, block_path = make_carbon_net(tmp_path, 3)
    block_path.write_text(block_path.read_text().replace("C 0 0 0", "Zn 0 0 0"))
    with pytest.raises(ValueError, match="Forbidden fused-bond chemistry"):
        Builder().build_by_type(Topology(topology_path), {0: BuildingBlock(block_path)})


def test_stale_cif_bond_table_cannot_mask_lost_connectivity(tmp_path):
    atoms = cubic_atoms()
    atoms.cell[2] *= 3
    path = tmp_path / "relaxed.cif"
    write(path, atoms)
    with path.open("a") as handle:
        handle.write("\nloop_\n_geom_bond_atom_site_label_1\n"
                     "_geom_bond_atom_site_label_2\n_geom_bond_distance\n"
                     "_geom_bond_site_symmetry_2\nC1 C1 1.4 1_556\n")
    assert validate_main([str(path)]) == 1


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("single_linker_mof_generator")


def test_staged_layer_cannot_be_recovered_as_an_accepted_sample(tmp_path, generator):
    path = tmp_path / "layer.cif"
    atoms = cubic_atoms()
    atoms.cell[2] *= 3
    write(path, atoms)
    path.write_text("# Max RMSD: 0.000\n" + path.read_text())
    config = generator.GenerationConfig(tmp_path / "manifest", tmp_path, tmp_path)
    with pytest.raises(ValueError, match="3D connectivity required.*2D"):
        generator.validate_staged_cif(config, path)


def test_staged_3d_sample_records_current_connectivity_policy(tmp_path, generator):
    path = tmp_path / "three_dimensional.cif"
    write(path, cubic_atoms())
    path.write_text("# Max RMSD: 0.000\n" + path.read_text())
    config = generator.GenerationConfig(tmp_path / "manifest", tmp_path, tmp_path)
    quality = generator.validate_staged_cif(config, path)
    assert quality["periodic_component_ranks"] == [3]
    assert quality["connectivity_policy_version"] == generator.CONNECTIVITY_POLICY_VERSION
    assert quality["minimum_atom_distance_angstrom"] == pytest.approx(1.4)


@pytest.mark.parametrize("require_3d", [False, True])
def test_staged_overlap_is_still_rejected(tmp_path, generator, require_3d):
    path = tmp_path / "overlap.cif"
    atoms = cubic_atoms() + Atoms("C", positions=[[.1, 0, 0]])
    write(path, atoms)
    path.write_text("# Max RMSD: 0.000\n" + path.read_text())
    config = generator.GenerationConfig(
        tmp_path / "manifest", tmp_path, tmp_path,
        require_3d_connectivity=require_3d,
    )
    with pytest.raises(ValueError, match="minimum-distance validation"):
        generator.validate_staged_cif(config, path)


def test_generator_port_screen_rejects_new_metal_metal_connections(generator):
    Ports = namedtuple("Ports", "connection_anchor_symbols connection_anchor_lengths")
    metal = Ports(("Zn",)*3, (1.12,)*3)
    assert not any(any(row) for row in generator.connection_port_compatibility(metal, metal))
