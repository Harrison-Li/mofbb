"""Composition package and executable integration tests."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from mofbb.compose import Builder, BuildingBlock, Reconstructor, Topology
from mofbb.compose.cli import build_parser, main


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
