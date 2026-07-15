"""Composition package and executable integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import mofbb.compose.scaler as scaler_module
from mofbb.compose import (
    Builder,
    BuildingBlock,
    Locator,
    Reconstructor,
    Scaler,
    Topology,
)
from mofbb.compose.cli import build_parser, main
from mofbb.compose.pbc import PBC
from mofbb.compose.scaler import _objective_value_and_gradient


def _write_linear_topology(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            CRYSTAL
              NAME linear
              GROUP P1
              CELL 20 20 20 90 90 90
              NODE 1 2 0.25 0.5 0.5
              NODE 2 2 0.75 0.5 0.5
              EDGE 0.25 0.5 0.5 0.75 0.5 0.5
              EDGE 0.25 0.5 0.5 -0.25 0.5 0.5
            END
            """
        ),
        encoding="utf-8",
    )
    return path


def _write_linear_block(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            4
            2 3
            C  -0.5 0.0 0.0
            C   0.5 0.0 0.0
            X  -1.0 0.0 0.0
            X   1.0 0.0 0.0
            0 1 S
            0 2 S
            1 3 S
            """
        ),
        encoding="utf-8",
    )
    return path


def _write_long_cap_block(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            4
            2 3
            C  -0.5 0.0 0.0
            C   0.5 0.0 0.0
            X  -1.5 0.0 0.0
            X   1.5 0.0 0.0
            0 1 S
            0 2 S
            1 3 S
            """
        ),
        encoding="utf-8",
    )
    return path


def _write_pca_edge_block(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            7
            5 6
            C   0.0  0.0 -1.5
            C   0.0  0.0  1.5
            C   2.0  0.0 -0.7
            N   0.8  0.4  0.2
            O  -0.2 -0.3  0.8
            X   0.0  0.0 -2.0
            X   0.0  0.0  2.0
            0 5 S
            1 6 S
            0 2 S
            2 3 S
            3 4 S
            4 1 S
            """
        ),
        encoding="utf-8",
    )
    return path


def _write_parallel_topology(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            CRYSTAL
              NAME parallel
              GROUP P1
              CELL 20 20 20 90 90 90
              NODE 1 2 0.25 0.5 0.5
              NODE 2 2 0.75 0.5 0.5
              EDGE 0.25 0.5 0.5 0.75 0.5 0.5
              EDGE 0.25 0.5 0.5 0.75 0.5 0.5
            END
            """
        ),
        encoding="utf-8",
    )
    return path


def test_compose_public_api_does_not_load_jax() -> None:
    script = textwrap.dedent(
        """
        import sys
        from mofbb.compose import Scaler

        assert Scaler.__name__ == "Scaler"
        assert "jax" not in sys.modules
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_composer_builds_and_writes_framework(tmp_path: Path) -> None:
    topology_path = _write_linear_topology(tmp_path / "linear.cgd")
    block_path = _write_linear_block(tmp_path / "linear.xyz")
    topology = Topology(topology_path)
    block = BuildingBlock(block_path)

    framework = Builder().build_by_type(
        topology,
        {0: block, 1: block},
        accuracy=3,
    )

    assert len(framework.atoms) == 4
    assert "X" not in framework.atoms.get_chemical_symbols()
    assert len(framework.bonds) == 4

    output = tmp_path / "framework.cif"
    framework.write_cif(output)
    assert output.is_file()


def test_edge_frame_uses_transverse_pca(tmp_path: Path) -> None:
    block = BuildingBlock(_write_pca_edge_block(tmp_path / "pca-edge.xyz"))

    frame = block.edge_frame()

    np.testing.assert_allclose(frame.axes @ frame.axes.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(frame.z, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.linalg.det(frame.axes) == pytest.approx(1.0)
    assert frame.pca_gap > 0.1
    assert not frame.torsion_degenerate
    assert not frame.sign_degenerate


def test_edge_frame_is_rigid_transform_equivariant(tmp_path: Path) -> None:
    block = BuildingBlock(_write_pca_edge_block(tmp_path / "pca-edge.xyz"))
    frame = block.edge_frame()

    rotation = Rotation.from_euler("xyz", [31.0, -17.0, 73.0], degrees=True)
    rotation_matrix = rotation.as_matrix()
    moved = block.copy()
    moved.atoms.set_positions(
        rotation.apply(block.atoms.positions) + np.array([4.0, -2.0, 7.0])
    )

    moved_frame = moved.edge_frame()

    np.testing.assert_allclose(
        moved_frame.axes,
        frame.axes @ rotation_matrix.T,
        atol=1e-12,
    )


def test_edge_frame_centers_transverse_coordinates(tmp_path: Path) -> None:
    block = BuildingBlock(_write_pca_edge_block(tmp_path / "pca-edge.xyz"))
    frame = block.edge_frame()

    shifted = block.copy()
    real_atoms = np.asarray(shifted.atoms.get_chemical_symbols()) != "X"
    shifted.atoms.positions[real_atoms] += np.array([2.0, -1.0, 0.0])

    shifted_frame = shifted.edge_frame()

    np.testing.assert_allclose(shifted_frame.x, frame.x, atol=1e-12)
    assert shifted_frame.pca_gap == pytest.approx(frame.pca_gap)


def test_transversely_isotropic_edge_frame_is_degenerate(
    tmp_path: Path,
) -> None:
    block = BuildingBlock(_write_pca_edge_block(tmp_path / "pca-edge.xyz"))
    root_three_over_two = np.sqrt(3.0) / 2.0
    block.atoms.positions[2:5] = [
        [1.0, 0.0, -0.7],
        [-0.5, root_three_over_two, 0.2],
        [-0.5, -root_three_over_two, 0.8],
    ]

    frame = block.edge_frame()

    assert frame.pca_gap < 1e-12
    assert frame.torsion_degenerate


def test_complete_edge_placement_is_input_orientation_invariant(
    tmp_path: Path,
) -> None:
    block = BuildingBlock(_write_pca_edge_block(tmp_path / "pca-edge.xyz"))
    rotation = Rotation.from_euler("zyx", [23.0, 41.0, -29.0], degrees=True)
    moved = block.copy()
    moved.atoms.set_positions(
        rotation.apply(block.atoms.positions) + np.array([-3.0, 5.0, 2.0])
    )

    target_points = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 6.0, 3.0],
        ]
    )
    target_transverse = np.array([1.0, 0.0, 1.0])
    original_target_transverse = target_transverse.copy()
    locator = Locator()

    located, rmsd, _ = locator.locate_edge_with_frame(
        target_points,
        block,
        permutation=[0, 1],
        target_transverse=target_transverse,
    )
    located_moved, moved_rmsd, _ = locator.locate_edge_with_frame(
        target_points,
        moved,
        permutation=[0, 1],
        target_transverse=target_transverse,
    )

    np.testing.assert_allclose(
        located_moved.atoms.positions,
        located.atoms.positions,
        atol=1e-12,
    )
    assert rmsd < 1e-12
    assert moved_rmsd < 1e-12
    np.testing.assert_array_equal(
        target_transverse,
        original_target_transverse,
    )


def test_complete_edge_placement_respects_endpoint_reversal(
    tmp_path: Path,
) -> None:
    block = BuildingBlock(_write_pca_edge_block(tmp_path / "pca-edge.xyz"))
    target_points = np.array(
        [
            [2.0, -1.0, 0.5],
            [2.0, 3.0, 0.5],
        ]
    )

    located, rmsd, _ = Locator().locate_edge_with_frame(
        target_points,
        block,
        permutation=[1, 0],
        target_transverse=[0.0, 0.0, 1.0],
    )

    np.testing.assert_allclose(
        located.connection_points[[1, 0]],
        target_points,
        atol=1e-12,
    )
    assert rmsd < 1e-12


def test_complete_edge_placement_rejects_parallel_transverse_direction(
    tmp_path: Path,
) -> None:
    block = BuildingBlock(_write_pca_edge_block(tmp_path / "pca-edge.xyz"))

    with pytest.raises(ValueError, match="must not be parallel"):
        Locator().locate_edge_with_frame(
            target_points=[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
            bb=block,
            permutation=[0, 1],
            target_transverse=[0.0, 0.0, 1.0],
        )


def test_linear_edge_frame_is_torsion_degenerate(tmp_path: Path) -> None:
    block = BuildingBlock(_write_linear_block(tmp_path / "linear.xyz"))

    frame = block.edge_frame()

    np.testing.assert_allclose(frame.axes @ frame.axes.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame.axes) == pytest.approx(1.0)
    assert frame.pca_gap == 0.0
    assert frame.torsion_degenerate
    assert frame.sign_degenerate


def test_scaler_analytic_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(9)
    positions = rng.uniform(0.1, 0.9, size=(4, 3))
    cell = rng.normal(size=(3, 3))
    ij = np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
    ik = np.array([[0, 2], [0, 3], [1, 2], [2, 1]])
    ij_image = rng.integers(-1, 2, size=(4, 3))
    ik_image = rng.integers(-1, 2, size=(4, 3))
    target_dots = rng.normal(size=4)
    weights = np.array([2.0, 1.0, 1.0, 2.0])
    x = np.concatenate([positions.reshape(-1), cell.reshape(-1)])

    def evaluate(values):
        value, position_gradient, cell_gradient = (
            _objective_value_and_gradient(
                values[:-9].reshape(4, 3),
                values[-9:].reshape(3, 3),
                ij,
                ik,
                ij_image,
                ik_image,
                target_dots,
                weights,
            )
        )
        return value, np.concatenate(
            [position_gradient.reshape(-1), cell_gradient.reshape(-1)]
        )

    _, analytic = evaluate(x)
    step = 1e-6
    numerical = np.empty_like(x)
    for index in range(len(x)):
        offset = np.zeros_like(x)
        offset[index] = step
        numerical[index] = (
            evaluate(x + offset)[0] - evaluate(x - offset)[0]
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-6)


def test_scaler_optimizes_only_node_coordinates(tmp_path: Path, monkeypatch) -> None:
    topology = Topology(_write_linear_topology(tmp_path / "linear.cgd"))
    block = BuildingBlock(_write_linear_block(tmp_path / "linear.xyz"))
    real_minimize = scaler_module.sp.optimize.minimize
    captured = {}

    def capture_minimize(*args, **kwargs):
        captured["variables"] = len(kwargs["x0"])
        captured["jac"] = kwargs["jac"]
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(scaler_module.sp.optimize, "minimize", capture_minimize)
    framework = Builder(scaler=Scaler()).build_by_type(
        topology,
        {0: block, 1: block},
        accuracy=3,
    )

    assert captured == {
        "variables": 3 * topology.n_nodes + 9,
        "jac": True,
    }
    assert "X" not in framework.atoms.get_chemical_symbols()


def test_composer_rejects_misaligned_x_pairs(tmp_path: Path) -> None:
    class NoOpScaler:
        def scale(self, topology, bbs, perms, return_result=False):
            result = SimpleNamespace(fun=0.0)
            return (topology.copy(), result) if return_result else topology.copy()

    topology_path = _write_linear_topology(tmp_path / "linear.cgd")
    block_path = _write_linear_block(tmp_path / "linear.xyz")
    topology = Topology(topology_path)
    block = BuildingBlock(block_path)
    builder = Builder(scaler=NoOpScaler())

    with pytest.raises(ValueError, match="Geometric quality gate rejected"):
        builder.build_by_type(topology, {0: block, 1: block}, accuracy=3)

    framework = builder.build_by_type(
        topology,
        {0: block, 1: block},
        accuracy=3,
        tolerance=10.0,
        scale=10.0,
    )
    assert "X" not in framework.atoms.get_chemical_symbols()
    assert framework.info["tolerance"] == 10.0


def test_fused_bond_gate_uses_scaled_covalent_radii() -> None:
    cases = [
        ("CNXX", 1.90, True),
        ("CCXX", 1.55, False),
        ("ZnNXX", 2.14, False),
    ]
    for symbols, distance, should_fail in cases:
        atoms = Atoms(
            symbols,
            positions=[
                [0.0, 0.0, 0.0],
                [distance, 0.0, 0.0],
                [0.5 * distance, 0.0, 0.0],
                [0.5 * distance, 0.0, 0.0],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        x_pairs = [(2, 3)]
        x_neighbors = {2: 0, 3: 1}

        assert PBC.calc_x_mid(atoms, x_pairs) == []
        assert bool(
            PBC.calc_fail_connection(atoms, x_pairs, x_neighbors)
        ) is should_fail


def test_fused_bond_gate_uses_minimum_image_distance() -> None:
    atoms = Atoms(
        "CCXX",
        positions=[
            [9.9, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [9.95, 0.0, 0.0],
            [0.05, 0.0, 0.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    assert PBC.calc_x_mid(atoms, [(2, 3)], tolerance=0.2) == []
    assert PBC.calc_fail_connection(
        atoms,
        [(2, 3)],
        {2: 0, 3: 1},
    ) == []


def test_predicted_fused_bond_failure_prevents_scaling(tmp_path: Path) -> None:
    class CountingScaler:
        calls = 0

        def scale(self, topology, bbs, perms, return_result=False):
            self.calls += 1
            result = SimpleNamespace(fun=0.0)
            return (topology.copy(), result) if return_result else topology.copy()

    topology = Topology(_write_linear_topology(tmp_path / "linear.cgd"))
    block = BuildingBlock(_write_long_cap_block(tmp_path / "long.xyz"))
    scaler = CountingScaler()

    with pytest.raises(ValueError, match="Predicted fused-bond gate rejected"):
        Builder(scaler=scaler).build_by_type(
            topology,
            {0: block, 1: block},
            accuracy=3,
        )

    assert scaler.calls == 0


def test_connection_point_requires_one_real_anchor(tmp_path: Path) -> None:
    block_path = tmp_path / "bad-anchor.xyz"
    block_path.write_text(
        textwrap.dedent(
            """\
            3
            0 1
            C 0.0 0.0 0.0
            C 1.0 0.0 0.0
            X 0.5 0.0 0.0
            0 2 S
            1 2 S
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one real-atom anchor"):
        BuildingBlock(block_path).connection_point_anchor_indices


def test_topology_preserves_parallel_edges_with_same_center(tmp_path: Path) -> None:
    topology = Topology(_write_parallel_topology(tmp_path / "parallel.cgd"))

    assert topology.n_nodes == 2
    assert topology.n_edges == 2
    assert [len(topology.neighbor_list[index]) for index in topology.node_indices] == [
        2,
        2,
    ]


def test_compose_cli_is_executable(tmp_path: Path) -> None:
    topology_path = _write_linear_topology(tmp_path / "linear.cgd")
    block_path = _write_linear_block(tmp_path / "linear.xyz")
    output = tmp_path / "cli-framework.cif"

    exit_code = main(
        [
            str(topology_path),
            "--node-bb",
            f"0={block_path}",
            "--node-bb",
            f"1={block_path}",
            "--accuracy",
            "3",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.is_file()


def test_reconstructor_composes_decomposer_output(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    decomposition_dir = (
        project_root / "examples" / "singlenode-decomposed" / "UiO-67"
    )

    framework = Reconstructor(decomposition_dir).reconstruct()
    reconstruction_map = json.loads(
        (decomposition_dir / "reconstruction_map.json").read_text(encoding="utf-8")
    )
    expected_source_indices = {
        source_index
        for slot in reconstruction_map["slots"]
        for source_index in slot["source_indices"]
    }

    assert len(framework.atoms) == len(expected_source_indices)
    assert "X" not in framework.atoms.get_chemical_symbols()
    assert len(framework.bonds) > len(framework.atoms)

    output = tmp_path / "reconstructed.cif"
    exit_code = main([str(decomposition_dir), "--output", str(output)])

    assert exit_code == 0
    assert output.is_file()


def test_compose_parser_has_stable_program_name() -> None:
    assert build_parser().prog == "mofbb-compose"
