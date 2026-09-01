import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "evaluate_assembly.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mofbb_evaluate_assembly",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)

from mofbb.compose import Builder, BuildingBlock, Locator, Topology
from mofbb.compose.locator import find_best_permutation
from mofbb.compose.pbc import PBC
from mofbb.compose.placement.fixed_cell import FixedCellFrameworkFitter
from mofbb.decompose.object import TopologyEdge, TopologyVertex
from mofbb.decompose.results import _write_topology_cgd


def _write_block(path: Path) -> Path:
    path.write_text(
        """5
test-block
C -0.5 0.0 0.0
N  0.0 0.2 0.3
C  0.5 0.0 0.0
X -1.0 0.0 0.0
X  1.0 0.0 0.0
0 1 S
1 2 S
0 3 S
2 4 S
""",
        encoding="utf-8",
    )
    return path


def _write_chemistry_topology(path: Path) -> Path:
    path.write_text(
        """CRYSTAL
  NAME chemistry-mask
  GROUP P1
  CELL 20 20 20 90 90 90
  NODE 1 2 0.25 0.5 0.5
  NODE 2 2 0.75 0.5 0.5
  EDGE 0.25 0.5 0.5 0.75 0.5 0.5
  EDGE 0.25 0.5 0.5 -0.25 0.5 0.5
END
""",
        encoding="utf-8",
    )
    return path


def _write_carbon_node(path: Path) -> Path:
    path.write_text(
        """4
carbon-node
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


def _write_cadmium_node(path: Path) -> Path:
    path.write_text(
        """3
cadmium-node
Cd 0.0 0 0
X -0.5 0 0
X  0.5 0 0
0 1 S
0 2 S
""",
        encoding="utf-8",
    )
    return path


def _write_mixed_edge(path: Path) -> Path:
    path.write_text(
        """4
mixed-edge
Cd -0.5 0 0
O   0.5 0 0
X  -1.0 0 0
X   1.0 0 0
0 1 S
0 2 S
1 3 S
""",
        encoding="utf-8",
    )
    return path


def _write_mixed_node(path: Path, symbols: tuple[str, str]) -> Path:
    path.write_text(
        f"""4
mixed-node
{symbols[0]} -0.5 0 0
{symbols[1]}  0.5 0 0
X -1.0 0 0
X  1.0 0 0
0 1 S
0 2 S
1 3 S
""",
        encoding="utf-8",
    )
    return path


def _write_nitrogen_carbon_edge(path: Path) -> Path:
    path.write_text(
        """5
mixed-edge
N -0.5 0.0 0.0
C  0.5 0.0 0.0
H  0.0 0.5 0.0
X -1.0 0.0 0.0
X  1.0 0.0 0.0
0 1 S
0 2 S
0 3 S
1 4 S
""",
        encoding="utf-8",
    )
    return path


def _write_high_coordination_block(path: Path) -> Path:
    path.write_text(
        """7
high-coordination
C  0.0  0.0  0.0
X  1.0  0.0  0.0
X -0.8  0.2  0.1
X  0.0  1.1  0.2
X  0.0 -0.9  0.3
X  0.2  0.1  1.2
X -0.1 -0.2 -0.7
0 1 S
0 2 S
0 3 S
0 4 S
0 5 S
0 6 S
""",
        encoding="utf-8",
    )
    return path


def test_rigid_bb_rotation_preserves_centroid_and_internal_distances(
    tmp_path: Path,
) -> None:
    block = BuildingBlock(_write_block(tmp_path / "block.xyz"))
    initial_centroid = block.centroid.copy()
    initial_distances = block.atoms.get_all_distances()

    matrix = evaluation.rotate_building_block(
        block,
        Rotation.from_euler("xyz", [31.0, -17.0, 73.0], degrees=True),
    )

    np.testing.assert_allclose(block.centroid, initial_centroid, atol=1.0e-12)
    np.testing.assert_allclose(
        block.atoms.get_all_distances(),
        initial_distances,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1.0e-12)


def test_high_coordination_locator_is_rotation_invariant(tmp_path: Path) -> None:
    block = BuildingBlock(_write_high_coordination_block(tmp_path / "high.xyz"))
    target = block.local_structure()
    reference, reference_permutation, _ = Locator().locate(target, block, 3)
    rotated = block.copy()
    evaluation.rotate_building_block(
        rotated,
        Rotation.from_euler("xyz", [41.0, -23.0, 67.0], degrees=True),
    )

    rebuilt, permutation, _ = Locator().locate(target, rotated, 3)

    np.testing.assert_array_equal(permutation, reference_permutation)
    np.testing.assert_allclose(
        rebuilt.atoms.positions, reference.atoms.positions, atol=1.0e-12
    )


def test_carbon_metal_mask_has_no_unconstrained_node_fallback(
    tmp_path: Path,
) -> None:
    class RecordingLocator(Locator):
        def __init__(self):
            self.allowed_masks = []

        def locate(self, target, bb, max_n_slices=4, allowed=None):
            self.allowed_masks.append(allowed)
            return super().locate(
                target,
                bb,
                max_n_slices=max_n_slices,
                allowed=allowed,
            )

    topology = Topology(_write_chemistry_topology(tmp_path / "topology.cgd"))
    carbon = BuildingBlock(_write_carbon_node(tmp_path / "carbon.xyz"))
    cadmium = BuildingBlock(_write_cadmium_node(tmp_path / "cadmium.xyz"))
    locator = RecordingLocator()

    with pytest.raises(
        ValueError,
        match=r"Forbidden fused-bond chemistry.*permutation optimization",
    ):
        Builder(locator=locator).build_by_type(
            topology,
            {0: cadmium, 1: carbon},
            accuracy=3,
        )

    assert locator.allowed_masks
    assert all(mask is not None for mask in locator.allowed_masks)


def test_permutation_cost_matrix_excludes_masked_assignments() -> None:
    target = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    source = target[::-1]
    allowed = np.array([[True, False], [False, True]])

    np.testing.assert_array_equal(
        find_best_permutation(target, source, allowed=allowed),
        [0, 1],
    )


def test_edge_permutation_rejects_two_forbidden_endpoint_orders(
    tmp_path: Path,
) -> None:
    topology = Topology(_write_chemistry_topology(tmp_path / "topology.cgd"))
    carbon = BuildingBlock(_write_carbon_node(tmp_path / "carbon.xyz"))
    edge = BuildingBlock(_write_mixed_edge(tmp_path / "edge.xyz"))

    with pytest.raises(
        ValueError,
        match=r"Forbidden fused-bond chemistry.*edge slot",
    ):
        Builder().build_by_type(
            topology,
            {0: carbon, 1: carbon},
            {(0, 1): edge},
            accuracy=3,
        )


def test_joint_edge_orientation_finds_topology_only_port_assignment(
    tmp_path: Path,
) -> None:
    topology_path = _write_chemistry_topology(tmp_path / "topology.cgd")
    topology_path.write_text(
        topology_path.read_text()
        .replace(
            "GROUP P1",
            "GROUP P1\n  # EXPANDED\n"
            "  # SYMOP 1 0 0 0 1 0 0 0 1 0 0 0",
        )
        .replace("CELL 20 20 20", "CELL 8 8 8"),
        encoding="utf-8",
    )
    topology = Topology(topology_path)
    left = BuildingBlock(
        _write_mixed_node(tmp_path / "left.xyz", ("Cu", "C"))
    )
    right = BuildingBlock(
        _write_mixed_node(tmp_path / "right.xyz", ("C", "Cu"))
    )
    edge = BuildingBlock(_write_nitrogen_carbon_edge(tmp_path / "edge.xyz"))

    baseline = Builder().build_by_type(
        topology,
        {0: left, 1: right},
        {(0, 1): edge},
        accuracy=3,
        overlap_scale=0.0,
    )
    for index, block in enumerate((left, right, edge), start=1):
        evaluation.rotate_building_block(
            block,
            Rotation.from_rotvec(np.array([0.2, -0.3, 0.4]) * index),
        )
    rebuilt = Builder().build_by_type(
        topology,
        {0: left, 1: right},
        {(0, 1): edge},
        accuracy=3,
        overlap_scale=0.0,
    )

    assert baseline.info["fixed_cell_fitted"]
    assert rebuilt.info["fixed_cell_fitted"]
    np.testing.assert_allclose(
        rebuilt.atoms.positions,
        baseline.atoms.positions,
        atol=1.0e-12,
    )


def test_joint_direct_edge_assignment_avoids_carbon_metal(
    tmp_path: Path,
) -> None:
    topology = Topology(_write_chemistry_topology(tmp_path / "topology.cgd"))
    left = BuildingBlock(
        _write_mixed_node(tmp_path / "left.xyz", ("O", "Mn"))
    )
    right = BuildingBlock(
        _write_mixed_node(tmp_path / "right.xyz", ("C", "O"))
    )

    framework = Builder().build_by_type(
        topology,
        {0: left, 1: right},
        accuracy=3,
        tolerance=100.0,
        scale=100.0,
        overlap_scale=0.0,
    )

    assert len(framework.atoms) > 0


def test_port_milp_uses_current_global_gap_costs() -> None:
    fit_stage = object.__new__(FixedCellFrameworkFitter)
    fit_stage.slot_indices = [0, 1]
    fit_stage.slot_offsets = {0: 0, 1: 1}
    fit_stage.connections = [(0, 0, 1, 0, np.zeros(3))]
    fit_stage.permutations = [np.arange(2), np.arange(2)]
    fit_stage.provided_permutations = {}
    fit_stage.templates = [
        SimpleNamespace(port_descriptors=(("O", 0.1), ("O", 0.1))),
        SimpleNamespace(port_descriptors=(("O", 0.1), ("O", 0.1))),
    ]
    fit_stage.pbc = PBC()
    fit_stage.bond_scale = PBC.scale
    fit_stage.symmetry = SimpleNamespace(node_orbits=(), edge_orbits=())
    fit_stage.symmetry_compatible_node_orbits = set()
    fit_stage.symmetry_compatible_edge_orbits = set()

    candidate = fit_stage._solve_port_assignment(
        np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        [
            np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ],
    )

    np.testing.assert_array_equal(candidate[0], [1, 0])
    np.testing.assert_array_equal(candidate[1], [0, 1])


def test_fixed_cell_pose_fit_jointly_rotates_and_translates_nodes() -> None:
    fit_stage = object.__new__(FixedCellFrameworkFitter)
    fit_stage.slot_indices = fit_stage.node_indices = [0, 1]
    fit_stage.edge_indices = []
    fit_stage.slot_offsets = {0: 0, 1: 1}
    fit_stage.connections = [
        (0, 0, 1, 0, np.zeros(3)),
        (0, 1, 1, 1, np.zeros(3)),
    ]
    fit_stage.topology = SimpleNamespace(
        atoms=SimpleNamespace(info={"symmetry_operations": True})
    )
    fit_stage.symmetry = SimpleNamespace(
        node_orbits=(
            SimpleNamespace(representative=0, members=(0,)),
            SimpleNamespace(representative=1, members=(1,)),
        ),
        edge_orbits=(),
    )
    fit_stage.symmetry_compatible_node_orbits = set()
    fit_stage.symmetry_compatible_edge_orbits = set()
    fit_stage.original_cell = np.eye(3)
    source_points = (
        np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    blocks = [
        SimpleNamespace(
            atoms=Atoms("XX", positions=points),
            centroid=np.zeros(3),
            connection_point_indices=np.arange(2),
        )
        for points in source_points
    ]
    centers = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    _, gaps, _, fitted_points = fit_stage._fit_candidate_geometry(
        [np.arange(2), np.arange(2)], centers, blocks
    )

    assert np.max(gaps) < 1.0e-7
    assert np.linalg.norm(centers[1] - [5.0, 0.0, 0.0]) > 1.0
    assert not np.allclose(fitted_points[1], source_points[1])


def test_element_graph_isomorphism_detects_rewiring_with_equal_counts() -> None:
    atoms = Atoms(
        ["C", "C", "N", "O"],
        positions=np.eye(4, 3),
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    reference = evaluation.bond_network_from_pairs(
        atoms,
        [(0, 1), (1, 2), (2, 3)],
        ["S", "S", "S"],
    )
    rewired = evaluation.bond_network_from_pairs(
        atoms,
        [(0, 2), (2, 1), (1, 3)],
        ["S", "S", "S"],
    )

    assert len(reference.pairs) == len(rewired.pairs)
    assert not evaluation.graphs_are_isomorphic(atoms, reference, atoms, rewired)
    assert not evaluation.graphs_are_isomorphic(
        atoms, reference, atoms, rewired, include_bond_types=True
    )

    report = evaluation.connectivity_report(atoms, reference, atoms, rewired)
    assert report["atom_count_match"]
    assert report["bond_count_match"]
    assert not report["element_graph_isomorphic"]
    assert not report["structurally_complete"]


def test_periodic_rank_profile_distinguishes_3d_from_finite_network() -> None:
    atoms = Atoms(
        "CCCC",
        positions=[
            (1.0, 1.0, 1.0),
            (3.0, 1.0, 1.0),
            (1.0, 3.0, 1.0),
            (1.0, 1.0, 3.0),
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    finite = evaluation.bond_network_from_pairs(
        atoms,
        [(0, 1), (0, 2), (0, 3)],
        ["S", "S", "S"],
    )

    adjacency = [set(neighbors) for neighbors in finite.adjacency]
    vectors = dict(finite.vectors)
    bond_types = dict(finite.bond_types)
    for pair, shift in (
        ((1, 2), (1, 0, 0)),
        ((1, 3), (0, 1, 0)),
        ((2, 3), (0, 0, 1)),
    ):
        i, j = pair
        adjacency[i].add(j)
        adjacency[j].add(i)
        vectors[pair] = (
            atoms.positions[j]
            + np.asarray(shift) @ atoms.cell.array
            - atoms.positions[i]
        )
        bond_types[pair] = "S"
    periodic = type(finite)(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
        bond_types=bond_types,
    )

    assert evaluation.component_rank_profile(atoms, finite) == [0]
    assert evaluation.component_rank_profile(atoms, periodic) == [3]


def test_writer_emits_source_subgroup_and_retains_net_group(tmp_path: Path):
    vertices = [
        TopologyVertex("node", 1, 1, (0.25, 0.5, 0.5)),
        TopologyVertex("node", 2, 1, (0.75, 0.5, 0.5)),
    ]
    edges = [
        TopologyEdge(
            "node",
            1,
            "node",
            2,
            (0.25, 0.5, 0.5),
            (0.75, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        )
    ]
    path = tmp_path / "subgroup.cgd"
    summary = _write_topology_cgd(
        path,
        "subgroup",
        (10, 10, 10, 90, 90, 90),
        vertices,
        edges,
        {("node", 1): "node", ("node", 2): "node"},
        assembly_symmetry=(
            "P1",
            np.eye(3, dtype=int)[None],
            np.zeros((1, 3)),
        ),
    )

    text = path.read_text()
    assert summary["spacegroup"] == "P1"
    assert summary["net_spacegroup"] != "P1"
    assert text.count("  # SYMOP ") == 1
    assert f"  # NET_GROUP {summary['net_spacegroup']}" in text
    assert "  # SOURCE_GROUP P1" in text
    assert "  # SOURCE_GROUP_COMPATIBLE 1" in text
    assert Topology(path).atoms.info["source_group_compatible"] is True

    incompatible = tmp_path / "incompatible.cgd"
    fallback = _write_topology_cgd(
        incompatible,
        "incompatible",
        (10, 10, 10, 90, 90, 90),
        vertices,
        edges,
        {("node", 1): "node", ("node", 2): "node"},
        assembly_symmetry=(
            "P1",
            np.eye(3, dtype=int)[None],
            np.asarray([[0.123, 0, 0]]),
        ),
    )
    assert fallback["spacegroup"] == fallback["net_spacegroup"]
    assert Topology(incompatible).atoms.info["source_group_compatible"] is False


def test_fixed_cell_solver_keeps_declared_orbits_symmetry_compatible(
    tmp_path: Path,
) -> None:
    topology_path = tmp_path / "inversion.cgd"
    topology_path.write_text(
        """CRYSTAL
  NAME inversion
  GROUP P-1
  CELL 6 6 6 90 90 90
  # EXPANDED
  # SYMOP 1 0 0 0 1 0 0 0 1 0 0 0
  # SYMOP -1 0 0 0 -1 0 0 0 -1 0 0 0
  NODE 1 2 0.25 0.5 0.5
  NODE 1 2 0.75 0.5 0.5
  # EDGE_TYPE 1 1
  EDGE 0.25 0.5 0.5 0.75 0.5 0.5
  # EDGE_TYPE 2 2
  EDGE 0.25 0.5 0.5 -0.25 0.5 0.5
END
""",
        encoding="utf-8",
    )
    node_path = _write_mixed_node(tmp_path / "node.xyz", ("C", "C"))
    edge_path = tmp_path / "symmetric-edge.xyz"
    edge_path.write_text(
        """4
2 3
C -0.25 0 0
C  0.25 0 0
X -0.50 0 0
X  0.50 0 0
0 1 S
0 2 S
1 3 S
""",
        encoding="utf-8",
    )
    topology = Topology(topology_path)
    framework = Builder().build_by_type(
        topology,
        {0: BuildingBlock(node_path)},
        {tuple(topology.unique_edge_types[0]): BuildingBlock(edge_path)},
        accuracy=3,
        overlap_scale=0.0,
        strict_symmetry=True,
    )

    assert framework.info["fixed_cell_fitted"]
    diagnostics = framework.info["symmetry_placement_diagnostics"]
    assert all(
        item["symmetry_propagated"] and item["compatible"]
        for kind in ("nodes", "edges")
        for item in diagnostics[kind]
    )
    fractional = framework.atoms.get_scaled_positions(wrap=True)
    symbols = np.asarray(framework.atoms.get_chemical_symbols())
    for symbol in np.unique(symbols):
        indices = np.flatnonzero(symbols == symbol)
        difference = (
            -fractional[indices, None, :] - fractional[None, indices, :]
        )
        difference -= np.rint(difference)
        distances = np.linalg.norm(difference @ framework.atoms.cell, axis=2)
        rows, columns = linear_sum_assignment(distances)
        assert np.max(distances[rows, columns], initial=0.0) < 1.0e-8
