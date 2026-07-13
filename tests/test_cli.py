"""Command-line and public-export tests."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from mofbb.decompose.cli import build_parser, main


def test_decomposers_use_project_owned_public_names() -> None:
    from mofbb import NodeEdgeDecomposer as top_level_node_edge
    from mofbb import SingleNodeDecomposer as top_level_single_node
    from mofbb.decompose import NodeEdgeDecomposer, SingleNodeDecomposer

    assert top_level_single_node is SingleNodeDecomposer
    assert top_level_node_edge is NodeEdgeDecomposer
    assert SingleNodeDecomposer.__name__ == "SingleNodeDecomposer"
    assert NodeEdgeDecomposer.__name__ == "NodeEdgeDecomposer"


def test_root_cli_compatibility_entry_point() -> None:
    from mofbb.cli import main as compatibility_main

    assert compatibility_main is main


def test_cli_parser_defaults_to_no_solvent_removal() -> None:
    args = build_parser().parse_args(["input.cif"])

    assert args.solvent_mode == "none"


@pytest.mark.parametrize("mode", ["none", "free", "all"])
def test_cli_parser_accepts_every_solvent_mode(mode: str) -> None:
    args = build_parser().parse_args(["input.cif", "--solvent-mode", mode])

    assert args.solvent_mode == mode


def test_cli_parser_rejects_invalid_solvent_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["input.cif", "--solvent-mode", "invalid"])

    assert error.value.code == 2
    assert "invalid choice: 'invalid'" in capsys.readouterr().err


def test_cli_help_explains_fsr_and_asr_modes() -> None:
    help_text = " ".join(build_parser().format_help().split())

    assert "--solvent-mode {none,free,all}" in help_text
    assert "free uses the FSR structure" in help_text
    assert "all uses the ASR structure" in help_text


def test_solvent_types_are_public_lazy_exports() -> None:
    script = textwrap.dedent(
        """
        import sys

        import mofbb

        expected = {
            "MofidSolventRemover",
            "SolventMode",
            "SolventRemovalResult",
        }
        assert "mofbb.decompose.solvent" not in sys.modules
        assert expected <= set(mofbb.__all__)

        from mofbb import MofidSolventRemover, SolventMode, SolventRemovalResult

        assert "mofbb.decompose.solvent" in sys.modules
        assert MofidSolventRemover.__name__ == "MofidSolventRemover"
        assert SolventRemovalResult.__name__ == "SolventRemovalResult"
        assert set(SolventMode.__args__) == {"none", "free", "all"}
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("decomposition", "decomposer_name", "summary"),
    [
        pytest.param(
            "single-node",
            "SingleNodeDecomposer",
            {"num_nodes": 0, "num_linkers": 0, "num_connecting_bonds": 0},
            id="single-node",
        ),
        pytest.param(
            "node-edge",
            "NodeEdgeDecomposer",
            {
                "num_metal_nodes": 0,
                "num_organic_nodes": 0,
                "num_edges": 0,
                "num_connecting_bonds": 0,
            },
            id="node-edge",
        ),
    ],
)
@pytest.mark.parametrize(
    ("mode_arguments", "expected_mode"),
    [
        pytest.param([], "none", id="default-none"),
        pytest.param(["--solvent-mode", "none"], "none", id="explicit-none"),
        pytest.param(["--solvent-mode", "free"], "free", id="free"),
        pytest.param(["--solvent-mode", "all"], "all", id="all"),
    ],
)
def test_cli_forwards_solvent_mode_to_each_decomposition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    decomposition: str,
    decomposer_name: str,
    summary: dict[str, int],
    mode_arguments: list[str],
    expected_mode: str,
) -> None:
    captured: dict[str, object] = {}

    class FakeResult:
        def write(self, output_dir: Path) -> dict[str, int]:
            captured["output_dir"] = output_dir
            return summary

    class FakeDecomposer:
        def __init__(self, cif: Path, **kwargs: object) -> None:
            captured["cif"] = cif
            captured["kwargs"] = kwargs

        def decompose(self) -> FakeResult:
            return FakeResult()

    monkeypatch.setattr(f"mofbb.decompose.cli.{decomposer_name}", FakeDecomposer)
    output_dir = tmp_path / "outputs"

    exit_code = main(
        [
            "input.cif",
            "--decomposition",
            decomposition,
            "--output-dir",
            str(output_dir),
            *mode_arguments,
        ]
    )

    assert exit_code == 0
    assert captured == {
        "cif": Path("input.cif"),
        "kwargs": {
            "x_symbol": "X",
            "bond_scale": 1.2,
            "cleanup": True,
            "solvent_mode": expected_mode,
        },
        "output_dir": output_dir,
    }
