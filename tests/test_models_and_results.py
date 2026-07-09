import json

import pytest

ase = pytest.importorskip("ase")

from ase import Atoms

from mofbb.models import Fragment, TopologyEdge, TopologyVertex, XSite
from mofbb.results import DecompositionResult


def make_fragment(kind: str, index: int, source_indices: tuple[int, ...]) -> Fragment:
    atoms = Atoms(
        symbols=["C", "X"],
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    return Fragment(
        kind=kind,
        index=index,
        source_indices=source_indices,
        atoms=atoms,
        bonds=((0, 1, "S"),),
        fingerprint=f"{kind}-fingerprint",
        x_sites=(
            XSite(
                x_index=1,
                anchor_index=0,
                source_cut_bond=(0, 1),
                internal_source_index=source_indices[0],
                external_source_index=1 - source_indices[0],
            ),
        ),
    )


def test_fragment_write_xyz_records_x_sites(tmp_path) -> None:
    fragment = make_fragment("node", 1, (0,))
    path = fragment.write_xyz(tmp_path / "node.xyz")

    lines = path.read_text().splitlines()

    assert lines[0] == "2"
    assert lines[1] == "1"
    assert lines[-1].strip() == "0    1 S"


def test_result_write_uses_stable_instance_paths(tmp_path) -> None:
    node = make_fragment("node", 1, (0,))
    linker = make_fragment("linker", 1, (1,))
    result = DecompositionResult(
        name="example",
        node_indices=(0,),
        linker_indices=(1,),
        points_of_extension=(),
        connecting_bonds=((0, 1),),
        cell_parameters=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        topology_vertices=[
            TopologyVertex("node", 1, 1, (0.25, 0.25, 0.25)),
            TopologyVertex("linker", 1, 1, (0.75, 0.25, 0.25)),
        ],
        topology_edges=[
            TopologyEdge(
                "node",
                1,
                "linker",
                1,
                (0.25, 0.25, 0.25),
                (0.75, 0.25, 0.25),
                (0.5, 0.25, 0.25),
            )
        ],
        node_fragments=[node],
        linker_fragments=[linker],
    )

    summary = result.write(tmp_path)

    assert summary["nodes"][0]["path"] == "nodes/node_001.xyz"
    assert summary["linkers"][0]["path"] == "linkers/linker_001.xyz"
    assert (tmp_path / "bbs" / "node_group_01.xyz").exists()
    assert (tmp_path / "topology.cgd").exists()

    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    assert reconstruction["validation"]["all_edges_mapped"] is True
