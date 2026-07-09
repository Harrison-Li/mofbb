import json
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms

from mofbb.bonding import BondNetwork
from mofbb.edge_decomposer import (
    NodeEdgePorDecomposer,
    NodeEdgeDecompositionResult,
    NodeEdgeTopologyBuilder,
    OrganicLinkerSplitter,
    classify_organic_fragment_kind,
)
from mofbb.fragments import FragmentBuilder
from mofbb.models import Fragment, TopologyEdge, TopologyVertex, XSite


def make_fragment(
    kind: str,
    index: int,
    source_indices: tuple[int, ...],
    cut_bonds: tuple[tuple[int, int], ...],
    x_symbol: str = "X",
) -> Fragment:
    symbols = ["C"] * len(source_indices) + [x_symbol] * len(cut_bonds)
    positions = [(float(i), 0.0, 0.0) for i in range(len(symbols))]
    bonds = []
    x_sites = []

    for offset, cut_bond in enumerate(cut_bonds):
        x_index = len(source_indices) + offset
        internal = next(atom for atom in cut_bond if atom in source_indices)
        external = next(atom for atom in cut_bond if atom not in source_indices)
        anchor_index = source_indices.index(internal)
        bonds.append((anchor_index, x_index, "S"))
        x_sites.append(
            XSite(
                x_index=x_index,
                anchor_index=anchor_index,
                source_cut_bond=tuple(sorted(cut_bond)),
                internal_source_index=internal,
                external_source_index=external,
            )
        )

    return Fragment(
        kind=kind,
        index=index,
        source_indices=source_indices,
        atoms=Atoms(
            symbols=symbols,
            positions=positions,
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        bonds=tuple(bonds),
        fingerprint=f"{kind}-{len(cut_bonds)}-fingerprint",
        x_sites=tuple(x_sites),
    )


def test_organic_fragment_classification_uses_connection_sites() -> None:
    edge = make_fragment("linker", 1, (1,), ((0, 1), (1, 2)), x_symbol="He")
    organic_node = make_fragment(
        "linker",
        2,
        (3,),
        ((0, 3), (2, 3), (3, 4)),
        x_symbol="He",
    )

    assert classify_organic_fragment_kind(edge) == "edge"
    assert classify_organic_fragment_kind(organic_node) == "organic_node"


def test_organic_fragment_classification_rejects_terminal_fragments() -> None:
    terminal = make_fragment("linker", 1, (1,), ((0, 1),))

    with pytest.raises(ValueError, match="fewer than 2 connection sites"):
        classify_organic_fragment_kind(terminal)


def make_bond_network(
    positions: list[tuple[float, float, float]],
    pairs: list[tuple[int, int]],
) -> BondNetwork:
    adjacency = [set() for _ in positions]
    vectors = {}
    for i, j in pairs:
        a, b = sorted((i, j))
        adjacency[a].add(b)
        adjacency[b].add(a)
        vectors[(a, b)] = np.array(positions[b]) - np.array(positions[a])
    return BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )


def test_organic_linker_splitter_creates_internal_nodes_and_edges() -> None:
    symbols = ["Zn", "Zn", "Zn", "Zn", "C", "C", "C", "C", "C", "C", "C"]
    positions = [
        (-2.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, -2.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.6, 0.0, 0.0),
        (0.3, 0.5, 0.0),
        (-1.0, 0.0, 0.0),
        (1.4, 0.0, 0.0),
        (0.3, 1.3, 0.0),
        (0.3, -1.0, 0.0),
    ]
    bond_pairs = [
        (4, 5),
        (5, 6),
        (4, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (4, 10),
        (0, 7),
        (1, 8),
        (2, 9),
        (3, 10),
    ]
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    bonds = make_bond_network(positions, bond_pairs)
    metal_cut_bonds = ((0, 7), (1, 8), (2, 9), (3, 10))
    raw_linker = FragmentBuilder(atoms, bonds, symbols).materialize_fragment(
        "linker",
        1,
        {4, 5, 6, 7, 8, 9, 10},
        metal_cut_bonds,
    )

    split = OrganicLinkerSplitter(atoms, bonds, symbols).split(
        [raw_linker],
        metal_cut_bonds,
    )

    assert len(split.organic_node_fragments) == 1
    assert len(split.edge_fragments) == 4
    assert split.organic_connecting_bonds == ((4, 7), (4, 10), (5, 8), (6, 9))
    assert split.organic_node_fragments[0].source_indices == (4, 5, 6)
    assert split.organic_node_fragments[0].num_x == 4
    assert sorted(fragment.source_indices for fragment in split.edge_fragments) == [
        (7,),
        (8,),
        (9,),
        (10,),
    ]
    assert all(fragment.num_x == 2 for fragment in split.edge_fragments)


def test_node_edge_topology_uses_connection_site_directions_for_parallel_edges() -> None:
    atoms = Atoms(
        symbols=["Zn", "Zn", "C", "C"],
        positions=[
            (5.0, 4.0, 5.0),
            (5.0, 6.0, 5.0),
            (9.0, 4.0, 5.0),
            (9.0, 6.0, 5.0),
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    bonds = make_bond_network(
        [tuple(position) for position in atoms.positions],
        [(0, 2), (1, 3)],
    )
    metal_node = make_fragment("metal_node", 1, (0, 1), ((0, 2), (1, 3)))
    metal_node.atoms.positions[2] = (6.0, 5.0, 5.0)
    metal_node.atoms.positions[3] = (4.0, 5.0, 5.0)
    edge = make_fragment("edge", 1, (2, 3), ((0, 2), (1, 3)))
    edge.atoms.positions[2] = (4.0, 5.0, 5.0)
    edge.atoms.positions[3] = (6.0, 5.0, 5.0)

    _, topology_edges = NodeEdgeTopologyBuilder(atoms, bonds).build(
        [metal_node, edge],
        ((0, 2), (1, 3)),
    )

    assert len(topology_edges) == 2
    assert topology_edges[0].end_frac != topology_edges[1].end_frac


def test_node_edge_topology_keeps_non_edge_connections_centroid_based() -> None:
    atoms = Atoms(
        symbols=["Zn", "Zn", "C", "C"],
        positions=[
            (5.0, 4.0, 5.0),
            (5.0, 6.0, 5.0),
            (9.0, 4.0, 5.0),
            (9.0, 6.0, 5.0),
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    bonds = make_bond_network(
        [tuple(position) for position in atoms.positions],
        [(0, 2), (1, 3)],
    )
    metal_node = make_fragment("metal_node", 1, (0, 1), ((0, 2), (1, 3)))
    metal_node.atoms.positions[2] = (6.0, 5.0, 5.0)
    metal_node.atoms.positions[3] = (4.0, 5.0, 5.0)
    organic_node = make_fragment("organic_node", 1, (2, 3), ((0, 2), (1, 3)))
    organic_node.atoms.positions[2] = (4.0, 5.0, 5.0)
    organic_node.atoms.positions[3] = (6.0, 5.0, 5.0)

    _, topology_edges = NodeEdgeTopologyBuilder(atoms, bonds).build(
        [metal_node, organic_node],
        ((0, 2), (1, 3)),
    )

    assert len(topology_edges) == 2
    assert topology_edges[0].end_frac == topology_edges[1].end_frac


def test_node_edge_result_write_uses_three_output_roles(tmp_path) -> None:
    connecting_bonds = ((0, 1), (1, 2), (0, 3), (2, 3), (3, 4))
    metal_node = make_fragment("metal_node", 1, (0, 2, 4), connecting_bonds)
    edge = make_fragment("edge", 1, (1,), ((0, 1), (1, 2)))
    organic_node = make_fragment("organic_node", 1, (3,), ((0, 3), (2, 3), (3, 4)))

    result = NodeEdgeDecompositionResult(
        name="example",
        metal_node_indices=(0, 2, 4),
        organic_indices=(1, 3),
        points_of_extension=(),
        connecting_bonds=connecting_bonds,
        cell_parameters=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        topology_vertices=[
            TopologyVertex("metal_node", 1, 5, (0.25, 0.25, 0.25)),
            TopologyVertex("edge", 1, 2, (0.50, 0.25, 0.25)),
            TopologyVertex("organic_node", 1, 3, (0.75, 0.25, 0.25)),
        ],
        topology_edges=[
            TopologyEdge(
                "metal_node",
                1,
                "edge",
                1,
                (0.25, 0.25, 0.25),
                (0.50, 0.25, 0.25),
                (0.375, 0.25, 0.25),
            )
        ],
        metal_node_fragments=[metal_node],
        organic_node_fragments=[organic_node],
        edge_fragments=[edge],
    )

    summary = result.write(tmp_path)

    assert summary["num_metal_nodes"] == 1
    assert summary["num_organic_nodes"] == 1
    assert summary["num_edges"] == 1
    assert summary["metal_nodes"][0]["path"] == "metal_nodes/metal_node_001.xyz"
    assert summary["organic_nodes"][0]["path"] == "organic_nodes/organic_node_001.xyz"
    assert summary["edges"][0]["path"] == "edges/edge_001.xyz"
    assert (tmp_path / "bbs" / "organic_node_group_01.xyz").exists()
    assert (tmp_path / "edges" / "edge_001.xyz").exists()

    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    assert reconstruction["validation"]["all_edges_mapped"] is True


def _pormake_acs_h_edge_sample() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "test_decom"
        / "pormake_stratified_manual_examples"
        / "decomposed_no_source_match"
        / "0014_acs-h_edge"
        / "generated.cif"
    )


def _strip_geom_bond_loop(cif_text: str) -> str:
    lines = cif_text.splitlines()
    stripped_lines: list[str] = []
    i = 0
    while i < len(lines):
        if (
            lines[i].strip() == "loop_"
            and i + 1 < len(lines)
            and lines[i + 1].strip() == "_geom_bond_atom_site_label_1"
        ):
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped == "loop_" or stripped.startswith("data_"):
                    break
                i += 1
            continue
        stripped_lines.append(lines[i])
        i += 1
    return "\n".join(stripped_lines) + "\n"


def _assert_pormake_acs_h_edge_summary(summary: dict) -> None:
    assert summary["num_metal_nodes"] == 12
    assert summary["num_organic_nodes"] == 0
    assert summary["num_edges"] == 30
    assert summary["num_metal_connecting_bonds"] == 60
    assert summary["num_organic_connecting_bonds"] == 0
    assert summary["groups"] == [
        {
            "bb_id": "metal_node_group_01",
            "kind": "metal_node",
            "group": 1,
            "count": 12,
            "cn": 5,
            "formula": "C5Cd2O10X5",
            "num_atoms": 22,
            "num_x": 5,
            "x_sites": [
                {"site_id": 0, "x_index": 17},
                {"site_id": 1, "x_index": 18},
                {"site_id": 2, "x_index": 19},
                {"site_id": 3, "x_index": 20},
                {"site_id": 4, "x_index": 21},
            ],
            "fingerprint": summary["groups"][0]["fingerprint"],
            "fragment_indices": list(range(1, 13)),
            "path": "bbs/metal_node_group_01.xyz",
        },
        {
            "bb_id": "edge_group_01",
            "kind": "edge",
            "group": 1,
            "count": 30,
            "cn": 2,
            "formula": "C10H4S2X2",
            "num_atoms": 18,
            "num_x": 2,
            "x_sites": [
                {"site_id": 0, "x_index": 16},
                {"site_id": 1, "x_index": 17},
            ],
            "fingerprint": summary["groups"][1]["fingerprint"],
            "fragment_indices": list(range(1, 31)),
            "path": "bbs/edge_group_01.xyz",
        },
    ]


def test_node_edge_decomposer_uses_explicit_pormake_cif_bonds(tmp_path) -> None:
    sample = _pormake_acs_h_edge_sample()
    if not sample.exists():
        pytest.skip("PORMAKE stratified benchmark sample is not available")

    result = NodeEdgePorDecomposer(sample).decompose()
    summary = result.write(tmp_path)

    _assert_pormake_acs_h_edge_summary(summary)


def test_node_edge_decomposer_recovers_pormake_cif_without_geom_bonds(
    tmp_path,
) -> None:
    sample = _pormake_acs_h_edge_sample()
    if not sample.exists():
        pytest.skip("PORMAKE stratified benchmark sample is not available")
    stripped_sample = tmp_path / "generated_without_geom_bond.cif"
    stripped_sample.write_text(
        _strip_geom_bond_loop(sample.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    result = NodeEdgePorDecomposer(stripped_sample).decompose()
    summary = result.write(tmp_path / "decomposed")

    _assert_pormake_acs_h_edge_summary(summary)
