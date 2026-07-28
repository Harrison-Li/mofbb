import importlib.util
from pathlib import Path
import sys

from ase import Atoms
import numpy as np
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

bond_network_from_pairs = evaluation.bond_network_from_pairs
component_rank_profile = evaluation.component_rank_profile
connectivity_report = evaluation.connectivity_report
graphs_are_isomorphic = evaluation.graphs_are_isomorphic
rotate_building_block = evaluation.rotate_building_block
from mofbb.compose import BuildingBlock


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


def test_rigid_bb_rotation_preserves_centroid_and_internal_distances(
    tmp_path: Path,
) -> None:
    block = BuildingBlock(_write_block(tmp_path / "block.xyz"))
    initial_centroid = block.centroid.copy()
    initial_distances = block.atoms.get_all_distances()

    matrix = rotate_building_block(
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


def test_element_graph_isomorphism_detects_rewiring_with_equal_counts() -> None:
    atoms = Atoms(
        ["C", "C", "N", "O"],
        positions=np.eye(4, 3),
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    reference = bond_network_from_pairs(
        atoms,
        [(0, 1), (1, 2), (2, 3)],
        ["S", "S", "S"],
    )
    rewired = bond_network_from_pairs(
        atoms,
        [(0, 2), (2, 1), (1, 3)],
        ["S", "S", "S"],
    )

    assert len(reference.pairs) == len(rewired.pairs)
    assert not graphs_are_isomorphic(atoms, reference, atoms, rewired)

    report = connectivity_report(atoms, reference, atoms, rewired)
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
    finite = bond_network_from_pairs(
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

    assert component_rank_profile(atoms, finite) == [0]
    assert component_rank_profile(atoms, periodic) == [3]
