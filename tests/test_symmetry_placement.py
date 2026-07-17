"""Regression tests for symmetry-covariant atomistic placement."""

from pathlib import Path

from ase import Atoms
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import mofbb.compose.builder as builder_module
from mofbb.compose import Builder, BuildingBlock, Framework, Locator, Topology
from mofbb.compose.edge import Edge
from mofbb.compose.symmetry_placement import SymmetryPlacement


class _CountingLocator(Locator):
    def __init__(self):
        self.node_relocations = 0
        self.edge_placements = 0

    def locate_with_permutation(self, target, bb, permutation):
        self.node_relocations += 1
        return super().locate_with_permutation(target, bb, permutation)

    def locate_edge_with_frame(
        self,
        target_points,
        bb,
        permutation,
        target_transverse,
    ):
        self.edge_placements += 1
        return super().locate_edge_with_frame(
            target_points,
            bb,
            permutation,
            target_transverse,
        )


def _write_p4mm_cycle_topology(path: Path) -> Path:
    path.write_text(
        """CRYSTAL
  NAME p4mm-cycle
  GROUP P4mm
  CELL 10 10 10 90 90 90
  NODE 1 2 0.25 0 0.5
  EDGE 0.25 0 0.5 0 0.25 0.5
END
""",
        encoding="utf-8",
    )
    return path


def _write_linear_node(path: Path) -> Path:
    path.write_text(
        """4
linear-node
C -0.5 0 0
C  0.5 0 0
X -1.0 0 0
X  1.0 0 0
0 1 S
0 2 S
1 3 S
""",
        encoding="utf-8",
    )
    return path


def _write_asymmetric_node(path: Path) -> Path:
    path.write_text(
        """6
asymmetric-node
C -0.5 0 0
C  0.5 0 0
N  0.1 0.3 0.2
O -0.2 0.1 0.4
X -1.0 0 0
X  1.0 0 0
0 1 S
0 2 S
0 3 S
0 4 S
1 5 S
""",
        encoding="utf-8",
    )
    return path


def _write_asymmetric_edge(path: Path) -> Path:
    path.write_text(
        """5
asymmetric-edge
C -0.3 0 0
C  0.3 0 0
N  0.1 0.2 0.15
X -0.5 0 0
X  0.5 0 0
0 1 S
0 2 S
0 3 S
1 4 S
""",
        encoding="utf-8",
    )
    return path


def _write_pmmm_edge_topology(path: Path) -> Path:
    path.write_text(
        """CRYSTAL
  NAME pmmm-edge
  GROUP Pmmm
  CELL 0.99999 1.61270 2.18289 90 90 90
  NODE 1 5 0 0 0.27095
  NODE 2 6 0 0.5 0
  EDGE 0 0 0.27095 0 0 0.72905
  EDGE 0 0.5 0 1 0.5 0
  EDGE 0 0 0.27095 1 0 0.27095
  EDGE 0 0 0.27095 0 0.5 0
END
""",
        encoding="utf-8",
    )
    return path


def _write_planar_edge(path: Path) -> Path:
    path.write_text(
        """6
planar-edge
C -0.5  0.2 0
C -0.5 -0.2 0
C  0.5  0.2 0
C  0.5 -0.2 0
X -1.0  0.0 0
X  1.0  0.0 0
""",
        encoding="utf-8",
    )
    return path


def _maximum_operation_residual(atoms, spacegroup):
    fractional = atoms.get_scaled_positions(wrap=True)
    cell = np.asarray(atoms.cell, dtype=float)
    symbols = np.asarray(atoms.get_chemical_symbols())
    maximum = 0.0
    for operation in SpaceGroup(spacegroup).symmetry_ops:
        transformed = np.asarray(
            [operation.operate(position) for position in fractional]
        )
        for symbol in np.unique(symbols):
            indices = np.flatnonzero(symbols == symbol)
            differences = (
                transformed[indices, np.newaxis, :]
                - fractional[np.newaxis, indices, :]
            )
            differences -= np.rint(differences)
            distances = np.linalg.norm(differences @ cell, axis=2)
            rows, columns = linear_sum_assignment(distances)
            maximum = max(
                maximum,
                float(np.max(distances[rows, columns], initial=0.0)),
            )
    return maximum


def test_port_permutations_follow_operation_induced_neighbors(
    tmp_path: Path,
) -> None:
    topology = Topology(
        _write_p4mm_cycle_topology(tmp_path / "p4mm-cycle.cgd")
    )
    placement = SymmetryPlacement(topology)

    for orbit in (*placement.node_orbits, *placement.edge_orbits):
        representative_permutation = np.array([0, 1])
        for raw_member in orbit.members:
            member = int(raw_member)
            operation_index = placement.operation(orbit, member)
            offset_map = placement.offset_map(
                orbit.representative,
                member,
                operation_index,
            )
            member_permutation = placement.to_member(
                orbit,
                member,
                representative_permutation,
            )
            np.testing.assert_array_equal(
                member_permutation[offset_map],
                representative_permutation,
            )


def test_cached_placement_invalidates_after_topology_mutation(
    tmp_path: Path,
) -> None:
    topology = Topology(
        _write_p4mm_cycle_topology(tmp_path / "p4mm-cycle.cgd")
    )
    SymmetryPlacement.clear_cache()

    first = SymmetryPlacement.cached(topology)
    assert SymmetryPlacement.cached(topology) is first

    topology.atoms.positions += np.asarray(topology.atoms.cell)[0]
    replacement = SymmetryPlacement.cached(topology)
    assert replacement is not first
    np.testing.assert_array_equal(
        replacement.operation_mappings,
        first.operation_mappings,
    )
    SymmetryPlacement.clear_cache()


def test_builder_skips_edge_orientation_without_edge_blocks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    topology = Topology(
        _write_p4mm_cycle_topology(tmp_path / "p4mm-cycle.cgd")
    )
    node = BuildingBlock(_write_linear_node(tmp_path / "linear-node.xyz"))

    class UnexpectedEdge:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Edge orientation should not be constructed")

    monkeypatch.setattr(builder_module, "Edge", UnexpectedEdge)
    framework = Builder().build_by_type(
        topology,
        {0: node},
        accuracy=3,
        tolerance=1.0,
        scale=2.0,
    )

    assert len(framework.atoms) > 0
    assert framework.info["edge_frame_diagnostics"] == []


def test_edge_targets_use_cached_slot_operations(tmp_path: Path) -> None:
    topology = Topology(
        _write_p4mm_cycle_topology(tmp_path / "p4mm-cycle.cgd")
    )
    node = BuildingBlock(_write_linear_node(tmp_path / "linear-node.xyz"))
    edge = BuildingBlock(_write_linear_node(tmp_path / "linear-edge.xyz"))
    edge_type = tuple(topology.unique_edge_types[0])
    framework = Builder().build_by_type(
        topology,
        {0: node},
        {edge_type: edge},
        accuracy=3,
        tolerance=1.0,
        scale=2.0,
    )

    placement = SymmetryPlacement.cached(topology)
    permutations = framework.info["permutations"]
    targets = Edge(
        topology,
        framework.info["topology"],
        permutations,
        placement,
    ).build()
    for edge_index, target in targets.items():
        if edge_index == target.representative_edge:
            assert target.operation_index == -1
            continue

        assert (
            placement.operation_mappings[
                target.operation_index,
                target.representative_edge,
            ]
            == edge_index
        )
        offsets = placement.offset_map(
            target.representative_edge,
            edge_index,
            target.operation_index,
        )
        source_order = np.argsort(permutations[target.representative_edge])
        target_order = np.argsort(permutations[edge_index])
        expected_reversed = np.array_equal(
            offsets[source_order],
            target_order[::-1],
        )
        assert target.endpoint_reversed == expected_reversed


def test_edge_torsion_is_chosen_from_site_stabilizer(tmp_path: Path) -> None:
    topology = Topology(
        _write_pmmm_edge_topology(tmp_path / "pmmm-edge.cgd")
    )
    placement = SymmetryPlacement(topology)
    orbit = next(orbit for orbit in placement.edge_orbits if orbit.tag == -3)
    edge = BuildingBlock(_write_planar_edge(tmp_path / "planar-edge.xyz"))
    centered = edge.atoms.positions - edge.centroid
    tilted = Rotation.from_rotvec(
        np.deg2rad(30.0) * np.array([1.0, 0.0, 0.0])
    ).apply(centered)
    edge.atoms.set_positions(
        tilted + topology.atoms.positions[orbit.representative]
    )
    permutation = np.array([0, 1])
    initial_ports = edge.connection_points.copy()

    initial = placement.diagnose(
        orbit,
        edge,
        permutation,
        topology.atoms.cell,
        1.0e-6,
    )
    optimized, diagnostic = placement.fit_torsion(
        orbit,
        edge,
        permutation,
        topology.atoms.cell,
        1.0e-6,
    )

    assert not initial["compatible"]
    assert diagnostic["compatible"]
    assert diagnostic["torsion_optimized"]
    assert diagnostic["best_torsion_residual"] < 1.0e-12
    np.testing.assert_allclose(
        optimized.connection_points,
        initial_ports,
        atol=1.0e-12,
    )


def test_compatible_orbits_preserve_declared_operations(
    tmp_path: Path,
) -> None:
    topology = Topology(
        _write_p4mm_cycle_topology(tmp_path / "p4mm-cycle.cgd")
    )
    node = BuildingBlock(_write_linear_node(tmp_path / "linear-node.xyz"))
    edge = BuildingBlock(_write_linear_node(tmp_path / "linear-edge.xyz"))
    edge_type = tuple(topology.unique_edge_types[0])

    locator = _CountingLocator()
    framework = Builder(locator=locator).build_by_type(
        topology,
        {0: node},
        {edge_type: edge},
        accuracy=3,
        tolerance=1.0,
        scale=2.0,
    )

    node_diagnostic = framework.info["symmetry_placement_diagnostics"][
        "nodes"
    ][0]
    assert node_diagnostic["covariant"]
    assert node_diagnostic["compatible"]
    edge_diagnostic = framework.info["symmetry_placement_diagnostics"][
        "edges"
    ][0]
    assert edge_diagnostic["covariant"]
    assert edge_diagnostic["compatible"]
    assert locator.node_relocations == 1
    assert locator.edge_placements == 1

    placement = SymmetryPlacement(topology)
    scaled_cell = np.asarray(framework.atoms.cell, dtype=float)
    located_blocks = framework.info["located_bbs"]
    for orbit in (*placement.node_orbits, *placement.edge_orbits):
        representative = located_blocks[orbit.representative]
        for raw_member in orbit.members:
            member = int(raw_member)
            expected = placement.place(
                representative,
                placement.operation(orbit, member),
                located_blocks[member].centroid,
                scaled_cell,
            )
            np.testing.assert_allclose(
                located_blocks[member].atoms.positions,
                expected.atoms.positions,
                atol=1.0e-12,
            )
    assert _maximum_operation_residual(
        framework.atoms,
        topology.spacegroup,
    ) < 1.0e-10


def test_incompatible_motifs_are_retained_and_not_rejected(
    tmp_path: Path,
) -> None:
    topology = Topology(
        _write_p4mm_cycle_topology(tmp_path / "p4mm-cycle.cgd")
    )
    node = BuildingBlock(
        _write_asymmetric_node(tmp_path / "asymmetric-node.xyz")
    )
    edge = BuildingBlock(
        _write_asymmetric_edge(tmp_path / "asymmetric-edge.xyz")
    )
    edge_type = tuple(topology.unique_edge_types[0])

    locator = _CountingLocator()
    framework = Builder(locator=locator).build_by_type(
        topology,
        {0: node},
        {edge_type: edge},
        accuracy=3,
        tolerance=1.0,
        scale=2.0,
    )

    diagnostics = framework.info["symmetry_placement_diagnostics"]
    assert not diagnostics["nodes"][0]["covariant"]
    assert not diagnostics["nodes"][0]["compatible"]
    assert diagnostics["nodes"][0]["reason"] == "stabilizer_incompatible"
    assert not diagnostics["edges"][0]["covariant"]
    assert not diagnostics["edges"][0]["compatible"]
    assert diagnostics["edges"][0]["reason"] == "stabilizer_incompatible"
    assert locator.node_relocations == topology.n_nodes
    assert locator.edge_placements == topology.n_edges
    assert len(framework.atoms) > 0


def test_framework_wrap_uses_only_integer_lattice_translations() -> None:
    initial_scaled = np.array(
        [
            [0.0, 1.0, -0.25],
            [2.5, -1.0, 0.999999999],
        ]
    )
    atoms = Atoms(
        "CN",
        scaled_positions=initial_scaled,
        cell=np.diag([5.0, 6.0, 7.0]),
        pbc=True,
    )

    framework = Framework(
        atoms,
        bonds=np.empty((0, 2), dtype=int),
        bond_types=[],
        info={},
        wrap=True,
    )

    wrapped = framework.atoms.get_scaled_positions(wrap=False)
    np.testing.assert_allclose(
        initial_scaled - wrapped,
        np.floor(initial_scaled),
        atol=1.0e-15,
    )
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < 1.0)
    assert wrapped[0, 0] == 0.0
    assert wrapped[0, 1] == 0.0
