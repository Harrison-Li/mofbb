import json

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms

import mofbb.topology as topology
from mofbb.bonding import BondNetwork
from mofbb.fragments import FragmentBuilder
from mofbb.models import Fragment, LogicalPort, TopologyEdge, TopologyVertex, XSite
from mofbb.periodicity import PeriodicBond
from mofbb.results import (
    DecompositionResult,
    fragment_group_key,
    fragment_group_metadata,
)


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


def test_fragment_models_preserve_legacy_positional_construction() -> None:
    atoms = Atoms(symbols=["C"], positions=[(0.0, 0.0, 0.0)])

    site = XSite(1, 0, (0, 1), 0, 1)
    fragment = Fragment("node", 1, (0,), atoms, (), "legacy-fingerprint")

    assert site.source_cut_shift == (0, 0, 0)
    assert site.external_image_shift == (0, 0, 0)
    assert site.as_dict()["source_cut_shift"] == [0, 0, 0]
    assert site.as_dict()["external_image_shift"] == [0, 0, 0]
    assert fragment.x_sites == ()
    assert fragment.source_image_offsets == ()


def test_group_metadata_uses_port_sizes_not_instance_local_x_indices() -> None:
    first = make_fragment("node", 1, (0,))
    second = make_fragment("node", 2, (1,))
    for fragment in (first, second):
        fragment.atoms = Atoms(
            symbols=["C", "X", "X", "X", "X", "X", "X"],
            positions=[(0.0, 0.0, 0.0)] * 7,
        )
        internal = fragment.source_indices[0]
        fragment.x_sites = tuple(
            XSite(
                x_index=x_index,
                anchor_index=0,
                source_cut_bond=tuple(sorted((internal, 10 + x_index))),
                internal_source_index=internal,
                external_source_index=10 + x_index,
            )
            for x_index in range(1, 7)
        )
    first.logical_kind = second.logical_kind = "edge"
    first.logical_ports = (
        LogicalPort(0, (1, 2, 3), "node", 2, (0, 0, 0)),
        LogicalPort(1, (4, 5, 6), "node", 3, (0, 0, 0)),
    )
    second.logical_ports = (
        LogicalPort(0, (1, 3, 5), "node", 4, (0, 0, 0)),
        LogicalPort(1, (2, 4, 6), "node", 5, (0, 0, 0)),
    )

    assert fragment_group_key(first) == fragment_group_key(second)
    assert fragment_group_metadata(first) == {
        "physical_cn": 6,
        "logical_cn": 2,
        "logical_kind": "edge",
        "logical_port_sizes": [3, 3],
        "logical_ports": [
            {"logical_port_id": 0, "size": 3},
            {"logical_port_id": 1, "size": 3},
        ],
    }


def test_task4_models_preserve_legacy_positional_construction() -> None:
    edge = TopologyEdge(
        "node",
        1,
        "linker",
        2,
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
        (0.25, 0.35, 0.45),
    )
    result = DecompositionResult(
        "legacy",
        (),
        (),
        (),
        (),
        (10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        [],
        [],
    )

    assert edge.source_cut_bond is None
    assert edge.source_cut_shift == (0, 0, 0)
    assert edge.slot_translation == (0, 0, 0)
    assert result.connection_records == ()
    assert result.periodic_openings == ()
    assert result.periodic_validation == {}


def _topology_fixture(
    symbols: str,
    positions: list[tuple[float, float, float]],
    shifts: dict[tuple[int, int], tuple[int, int, int]],
) -> tuple[Atoms, BondNetwork]:
    atoms = Atoms(symbols, positions=positions, cell=[10.0, 10.0, 10.0], pbc=True)
    adjacency = [set() for _ in atoms]
    vectors = {}
    for pair, shift in shifts.items():
        i, j = pair
        adjacency[i].add(j)
        adjacency[j].add(i)
        vectors[pair] = (
            np.asarray(positions[j])
            + np.asarray(shift) @ atoms.cell.array
            - np.asarray(positions[i])
        )
    return atoms, BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )


def test_topology_retains_translated_periodic_self_edge_and_wraps_vertex() -> None:
    atoms, bonds = _topology_fixture(
        "CCC",
        [(11.0, 0.0, 0.0), (14.0, 0.0, 0.0), (18.0, 0.0, 0.0)],
        {(0, 1): (0, 0, 0), (1, 2): (0, 0, 0), (0, 2): (-1, 0, 0)},
    )
    cut = PeriodicBond(0, 2, (-1, 0, 0))
    fragment = FragmentBuilder(atoms, bonds, ["C", "C", "C"]).materialize_fragment(
        "node", 1, {0, 1, 2}, (cut,)
    )

    vertices, edges = topology.TopologyBuilder(atoms, bonds).build(
        [fragment], [], (cut,)
    )

    assert len(edges) == 1
    assert edges[0].begin_kind == edges[0].end_kind == "node"
    assert edges[0].begin_index == edges[0].end_index == 1
    assert edges[0].source_cut_bond == (0, 2)
    assert edges[0].source_cut_shift == (-1, 0, 0)
    assert edges[0].slot_translation == (-1, 0, 0)
    assert vertices[0].coordination == 2
    assert all(0.0 <= value < 1.0 for value in vertices[0].frac_position)
    np.testing.assert_allclose(vertices[0].frac_position, (13.0 / 30.0, 0.0, 0.0))
    np.testing.assert_allclose(
        np.asarray(edges[0].end_frac) - edges[0].begin_frac,
        (-1, 0, 0),
    )


def test_topology_swaps_source_endpoints_with_node_role_order() -> None:
    atoms, bonds = _topology_fixture(
        "CCZnZn",
        [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (6.0, 0.0, 0.0), (7.0, 0.0, 0.0)],
        {(0, 1): (0, 0, 0), (2, 3): (0, 0, 0), (0, 2): (1, 0, 0)},
    )
    connection = PeriodicBond(0, 2, (1, 0, 0))
    builder = FragmentBuilder(atoms, bonds, atoms.get_chemical_symbols())
    linker = builder.materialize_fragment("linker", 1, {0, 1}, (connection,))
    node = builder.materialize_fragment("node", 1, {2, 3}, (connection,))

    _, edges = topology.TopologyBuilder(atoms, bonds).build(
        [node], [linker], (connection,)
    )

    assert len(edges) == 1
    assert (edges[0].begin_kind, edges[0].end_kind) == ("node", "linker")
    assert edges[0].source_cut_bond == (0, 2)
    assert edges[0].source_cut_shift == (1, 0, 0)
    assert edges[0].slot_translation == (-1, 0, 0)


def test_topology_rejects_zero_translation_same_slot_connection() -> None:
    atoms, bonds = _topology_fixture(
        "CCC",
        [(1.0, 0.0, 0.0), (4.0, 0.0, 0.0), (8.0, 0.0, 0.0)],
        {(0, 1): (0, 0, 0), (1, 2): (0, 0, 0), (0, 2): (0, 0, 0)},
    )
    cut = PeriodicBond(0, 2, (0, 0, 0))
    fragment = FragmentBuilder(atoms, bonds, ["C", "C", "C"]).materialize_fragment(
        "node", 1, {0, 1, 2}, (cut,)
    )

    with pytest.raises(ValueError, match=r"same-slot.*zero translation"):
        topology.TopologyBuilder(atoms, bonds).build([fragment], [], (cut,))


def test_topology_validates_stored_fragment_frame() -> None:
    atoms, bonds = _topology_fixture(
        "CCC",
        [(1.0, 0.0, 0.0), (4.0, 0.0, 0.0), (8.0, 0.0, 0.0)],
        {(0, 1): (0, 0, 0), (1, 2): (0, 0, 0), (0, 2): (-1, 0, 0)},
    )
    cut = PeriodicBond(0, 2, (-1, 0, 0))
    fragment = FragmentBuilder(atoms, bonds, ["C", "C", "C"]).materialize_fragment(
        "node", 1, {0, 1, 2}, (cut,)
    )
    fragment.source_image_offsets = (
        (0, (0, 0, 0)),
        (1, (1, 0, 0)),
        (2, (1, 0, 0)),
    )

    with pytest.raises(ValueError, match=r"Fragment node 1.*stored.*frame"):
        topology.TopologyBuilder(atoms, bonds).build([fragment], [], (cut,))


@pytest.mark.parametrize(
    ("expected", "fragments", "message"),
    [
        (
            {0, 1},
            [make_fragment("node", 1, (0,))],
            r"missing source atoms \[1\]",
        ),
        (
            {0},
            [
                make_fragment("node", 1, (0,)),
                make_fragment("linker", 1, (0,)),
            ],
            r"duplicate source atoms \{0: 2\}",
        ),
    ],
)
def test_source_atom_ownership_requires_exactly_once(expected, fragments, message) -> None:
    with pytest.raises(ValueError, match=message):
        topology.validate_source_atom_ownership(expected, fragments)


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

    assert "solvent" not in summary
    assert summary["nodes"][0]["path"] == "nodes/node_001.xyz"
    assert summary["linkers"][0]["path"] == "linkers/linker_001.xyz"
    assert (tmp_path / "bbs" / "node_group_01.xyz").exists()
    assert (tmp_path / "topology.cgd").exists()
    assert list(tmp_path.glob("*.cif")) == []

    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    assert reconstruction["validation"]["all_edges_mapped"] is True
    assert reconstruction["geometry_contract"] == {
        "exact_slot_geometry_field": "exact_instance_path",
        "group_representatives_are_exact_slot_geometry": False,
        "representative_atom_permutations_emitted": False,
    }
    assert reconstruction["slots"][0]["exact_instance_path"] == (
        reconstruction["slots"][0]["debug_instance"]
    )


def _legacy_connection_result(topology_count: int = 1) -> DecompositionResult:
    node = make_fragment("node", 1, (0,))
    linker = make_fragment("linker", 1, (1,))
    topology_edges = [
        TopologyEdge(
            "node",
            1,
            "linker",
            1,
            (0.25, 0.25, 0.25),
            (0.75, 0.25, 0.25),
            (0.5, 0.25, 0.25),
        )
        for _ in range(topology_count)
    ]
    return DecompositionResult(
        name="legacy",
        node_indices=(0,),
        linker_indices=(1,),
        points_of_extension=(),
        connecting_bonds=((0, 1),),
        cell_parameters=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        topology_vertices=[
            TopologyVertex("node", 1, 1, (0.25, 0.25, 0.25)),
            TopologyVertex("linker", 1, 1, (0.75, 0.25, 0.25)),
        ],
        topology_edges=topology_edges,
        node_fragments=[node],
        linker_fragments=[linker],
    )


@pytest.mark.parametrize("topology_count", [0, 2])
def test_legacy_reconstruction_detects_missing_or_extra_topology_edges(
    tmp_path, topology_count
) -> None:
    result = _legacy_connection_result(topology_count)

    result.write(tmp_path)
    validation = json.loads(
        (tmp_path / "reconstruction_map.json").read_text()
    )["validation"]

    assert validation["all_topology_edges_mapped"] is False
    assert validation["all_edges_mapped"] is False


def test_reconstruction_uses_separate_legacy_topology_cursor(tmp_path) -> None:
    node_1 = make_fragment("node", 1, (0,))
    linker_1 = make_fragment("linker", 1, (1,))
    node_2 = make_fragment("node", 2, (2,))
    linker_2 = make_fragment("linker", 2, (3,))
    node_2.x_sites = (XSite(1, 0, (2, 3), 2, 3),)
    linker_2.x_sites = (XSite(1, 0, (2, 3), 3, 2),)
    result = DecompositionResult(
        name="mixed-topology",
        node_indices=(0, 2),
        linker_indices=(1, 3),
        points_of_extension=(),
        connecting_bonds=((0, 1), (2, 3)),
        cell_parameters=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        topology_vertices=[
            TopologyVertex("node", 1, 1, (0.1, 0.0, 0.0)),
            TopologyVertex("node", 2, 1, (0.2, 0.0, 0.0)),
            TopologyVertex("linker", 1, 1, (0.6, 0.0, 0.0)),
            TopologyVertex("linker", 2, 1, (0.7, 0.0, 0.0)),
        ],
        topology_edges=[
            TopologyEdge(
                "node",
                1,
                "linker",
                1,
                (0.1, 0.0, 0.0),
                (0.6, 0.0, 0.0),
                (0.35, 0.0, 0.0),
                (0, 1),
            ),
            TopologyEdge(
                "node",
                2,
                "linker",
                2,
                (0.2, 0.0, 0.0),
                (0.7, 0.0, 0.0),
                (0.45, 0.0, 0.0),
            ),
        ],
        node_fragments=[node_1, node_2],
        linker_fragments=[linker_1, linker_2],
        connection_records=(
            PeriodicBond(0, 1, (0, 0, 0)),
            PeriodicBond(2, 3, (0, 0, 0)),
        ),
    )

    result.write(tmp_path)
    validation = json.loads(
        (tmp_path / "reconstruction_map.json").read_text()
    )["validation"]

    assert validation["all_topology_edges_mapped"] is True
    assert validation["all_edges_mapped"] is True


def test_reconstruction_detects_unrepresented_cap_atom(tmp_path) -> None:
    result = _legacy_connection_result()
    result.node_fragments[0].atoms += Atoms(
        symbols=["X"], positions=[(2.0, 0.0, 0.0)]
    )

    result.write(tmp_path)
    validation = json.loads(
        (tmp_path / "reconstruction_map.json").read_text()
    )["validation"]

    assert validation["all_x_site_metadata_valid"] is False
    assert validation["all_x_sites_used_once"] is False


def _periodic_self_edge_result(*, duplicate: bool = False) -> DecompositionResult:
    cut = PeriodicBond(0, 2, (-1, 0, 0))
    atoms = Atoms(
        symbols=["C", "C", "C", "X", "X"],
        positions=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
            (2.5, 0.0, 0.0),
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    fragment = Fragment(
        kind="node",
        index=1,
        source_indices=(0, 1, 2),
        atoms=atoms,
        bonds=((0, 1, "S"), (1, 2, "S"), (0, 3, "S"), (2, 4, "S")),
        fingerprint="periodic-node",
        x_sites=(
            XSite(3, 0, (0, 2), 0, 2, (-1, 0, 0), (-1, 0, 0)),
            XSite(4, 2, (0, 2), 2, 0, (-1, 0, 0), (1, 0, 0)),
        ),
        source_image_offsets=(
            (0, (0, 0, 0)),
            (1, (0, 0, 0)),
            (2, (0, 0, 0)),
        ),
    )
    topology_edge = TopologyEdge(
        "node",
        1,
        "node",
        1,
        (0.2, 0.0, 0.0),
        (-0.8, 0.0, 0.0),
        (-0.3, 0.0, 0.0),
        (0, 2),
        (-1, 0, 0),
        (-1, 0, 0),
    )
    multiplier = 2 if duplicate else 1
    return DecompositionResult(
        name="periodic",
        node_indices=(0, 1, 2),
        linker_indices=(),
        points_of_extension=(),
        connecting_bonds=((0, 2),) * multiplier,
        cell_parameters=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        topology_vertices=[TopologyVertex("node", 1, 2 * multiplier, (0.2, 0.0, 0.0))],
        topology_edges=[topology_edge] + (
            [
                TopologyEdge(
                    *(
                        topology_edge.begin_kind,
                        topology_edge.begin_index,
                        topology_edge.end_kind,
                        topology_edge.end_index,
                        topology_edge.begin_frac,
                        topology_edge.end_frac,
                        topology_edge.center_frac,
                        topology_edge.source_cut_bond,
                        topology_edge.source_cut_shift,
                        topology_edge.slot_translation,
                    )
                )
            ]
            if duplicate
            else []
        ),
        node_fragments=[fragment],
        connection_records=(cut,) * multiplier,
    )


def test_reconstruction_maps_both_sides_of_same_slot_periodic_cut(tmp_path) -> None:
    result = _periodic_self_edge_result()

    result.write(tmp_path)
    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    edge = reconstruction["edges"][0]

    assert reconstruction["schema_version"] == 2
    assert edge["endpoint_slots"] == [0, 0]
    assert edge["source_cut_bond"] == [0, 2]
    assert edge["source_cut_shift"] == [-1, 0, 0]
    assert edge["slot_translation"] == [-1, 0, 0]
    assert edge["endpoint_slot_translations"] == [[-1, 0, 0], [1, 0, 0]]
    assert [site["x_index"] for site in edge["endpoint_x_sites"]] == [3, 4]
    assert reconstruction["validation"]["all_edges_mapped"] is True
    assert reconstruction["validation"]["all_x_sites_used_once"] is True
    assert reconstruction["validation"]["all_self_edge_shifts_inverse"] is True
    assert reconstruction["validation"]["all_slot_translations_exact"] is True


def test_reconstruction_detects_wrong_periodic_slot_translation(tmp_path) -> None:
    result = _periodic_self_edge_result()
    edge = result.topology_edges[0]
    result.topology_edges = [
        TopologyEdge(
            edge.begin_kind,
            edge.begin_index,
            edge.end_kind,
            edge.end_index,
            edge.begin_frac,
            edge.end_frac,
            edge.center_frac,
            edge.source_cut_bond,
            edge.source_cut_shift,
            (1, 0, 0),
        )
    ]

    result.write(tmp_path)
    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    mapped = reconstruction["edges"][0]

    assert mapped["slot_translation"] == [1, 0, 0]
    assert mapped["expected_slot_translation"] == [-1, 0, 0]
    assert reconstruction["validation"]["all_slot_translations_exact"] is False
    assert reconstruction["validation"]["all_edges_mapped"] is False


def test_reconstruction_counter_detects_reused_x_sites(tmp_path) -> None:
    result = _periodic_self_edge_result(duplicate=True)

    result.write(tmp_path)
    validation = json.loads(
        (tmp_path / "reconstruction_map.json").read_text()
    )["validation"]

    assert validation["all_x_sites_used_once"] is False
    assert validation["duplicate_edge_warnings"] == [
        {
            "edge_count": 2,
            "source_cut_bond": [0, 2],
            "source_cut_shift": [-1, 0, 0],
        }
    ]


def test_result_write_rejects_missing_source_ownership_before_output(tmp_path) -> None:
    result = DecompositionResult(
        name="invalid",
        node_indices=(0, 1),
        linker_indices=(),
        points_of_extension=(),
        connecting_bonds=(),
        cell_parameters=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        topology_vertices=[],
        topology_edges=[],
        node_fragments=[make_fragment("node", 1, (0,))],
    )

    with pytest.raises(ValueError, match=r"missing source atoms \[1\]"):
        result.write(tmp_path)

    assert list(tmp_path.iterdir()) == []
