import json
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms

import mofbb.decompose.edge_decomposer as edge_decomposer  # noqa: E402
from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.edge_decomposer import (
    NodeEdgeDecomposer,
    NodeEdgeDecompositionResult,
    NodeEdgeTopologyBuilder,
    OrganicLinkerSplitter,
    classify_organic_fragment_kind,
    decompose_mof_with_edges,
)
from mofbb.decompose.decomposer import SingleNodeDecomposer
from mofbb.decompose.fragments import FragmentBuilder
from mofbb.decompose.models import Fragment, TopologyEdge, TopologyVertex, XSite
from mofbb.decompose.periodicity import PeriodicBond, periodic_bonds
from mofbb.decompose.solvent import MofidSolventRemover  # noqa: E402

SOLVENT_ARTIFACT_NAMES = {
    "free_solvent.cif",
    "bound_solvent.cif",
    "mof_fsr.cif",
    "mof_asr.cif",
}
LEGACY_NODE_EDGE_SUMMARY_KEYS = {
    "name",
    "decomposition",
    "num_metal_nodes",
    "num_organic_nodes",
    "num_edges",
    "num_connecting_bonds",
    "num_metal_connecting_bonds",
    "num_organic_connecting_bonds",
    "metal_node_indices",
    "organic_indices",
    "points_of_extension",
    "connecting_bonds",
    "metal_connecting_bonds",
    "organic_connecting_bonds",
    "topology",
    "reconstruction_map",
    "groups",
    "metal_nodes",
    "organic_nodes",
    "edges",
}


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
    assert edge.num_x == 2
    assert organic_node.num_x == 3


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


def make_periodic_atoms_and_bonds(
    symbols: list[str],
    pairs: list[tuple[int, int]],
    shifts: dict[tuple[int, int], tuple[int, int, int]] | None = None,
) -> tuple[Atoms, BondNetwork]:
    positions = np.array(
        [(float(index), 0.0, 0.0) for index in range(len(symbols))]
    )
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    adjacency = [set() for _ in symbols]
    vectors = {}
    shifts = shifts or {}
    for i, j in pairs:
        a, b = sorted((i, j))
        adjacency[a].add(b)
        adjacency[b].add(a)
        shift = np.array(shifts.get((a, b), (0, 0, 0)), dtype=float)
        vectors[(a, b)] = positions[b] + shift @ atoms.cell.array - positions[a]
    return atoms, BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )


def test_fragment_builder_caps_both_sides_of_periodic_internal_cut() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["C", "C", "C"],
        [(0, 1), (1, 2), (0, 2)],
        shifts={(0, 2): (-1, 0, 0)},
    )
    cut = PeriodicBond(0, 2, (-1, 0, 0))

    fragment = FragmentBuilder(
        atoms, bonds, ["C", "C", "C"]
    ).materialize_fragment("node", 1, {0, 1, 2}, (cut,))

    assert fragment.source_indices == (0, 1, 2)
    assert fragment.source_image_offsets == (
        (0, (0, 0, 0)),
        (1, (0, 0, 0)),
        (2, (0, 0, 0)),
    )
    assert fragment.num_x == 2
    assert [
        (site.internal_source_index, site.external_image_shift)
        for site in fragment.x_sites
    ] == [
        (0, (-1, 0, 0)),
        (2, (1, 0, 0)),
    ]
    assert [site.source_cut_shift for site in fragment.x_sites] == [
        (-1, 0, 0),
        (-1, 0, 0),
    ]
    assert not any({i, j} == {0, 2} for i, j, _ in fragment.bonds)

    for site in fragment.x_sites:
        actual = (
            fragment.atoms.positions[site.x_index]
            - fragment.atoms.positions[site.anchor_index]
        )
        expected = 0.5 * bonds.vector(
            site.internal_source_index, site.external_source_index
        )
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_fragment_builder_retains_explicit_non_nearest_images_in_skew_cell() -> None:
    cell = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 0.0, 8.0],
        ]
    )
    positions = np.array(
        [
            [0.2, 0.1, 0.3],
            [0.8, 0.2, 0.4],
            [1.4, 0.9, 0.5],
        ]
    )
    atoms = Atoms(
        symbols=["C", "N", "C"],
        positions=positions,
        cell=cell,
        pbc=(True, True, False),
    )
    shifts = {
        (0, 1): (2, 0, 0),
        (1, 2): (0, 1, 0),
        (0, 2): (-1, 0, 0),
    }
    adjacency = (frozenset({1, 2}), frozenset({0, 2}), frozenset({0, 1}))
    vectors = {
        pair: positions[pair[1]] + np.asarray(shift) @ cell - positions[pair[0]]
        for pair, shift in shifts.items()
    }
    bonds = BondNetwork(adjacency=adjacency, vectors=vectors)

    fragment = FragmentBuilder(
        atoms, bonds, atoms.get_chemical_symbols()
    ).materialize_fragment(
        "node",
        7,
        {0, 1, 2},
        (PeriodicBond(0, 2, (-1, 0, 0)),),
    )

    assert fragment.source_image_offsets == (
        (0, (0, 0, 0)),
        (1, (2, 0, 0)),
        (2, (2, 1, 0)),
    )
    np.testing.assert_allclose(
        fragment.atoms.positions[1] - fragment.atoms.positions[0],
        bonds.vector(0, 1),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        fragment.atoms.positions[2] - fragment.atoms.positions[1],
        bonds.vector(1, 2),
        atol=1.0e-12,
    )


def test_fragment_builder_rejects_disconnected_internal_cut_residual() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["C", "C", "C"],
        [(0, 1), (1, 2)],
    )

    with pytest.raises(
        ValueError,
        match=r"Fragment node 3.*disconnected.*internal cut",
    ):
        FragmentBuilder(atoms, bonds, ["C", "C", "C"]).materialize_fragment(
            "node",
            3,
            {0, 1, 2},
            (PeriodicBond(1, 2, (0, 0, 0)),),
        )


def test_fragment_builder_rejects_nonzero_residual_holonomy() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["C", "C", "C", "C"],
        [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)],
        shifts={(0, 1): (1, 0, 0)},
    )

    with pytest.raises(
        ValueError,
        match=r"Fragment linker 4.*non-zero holonomy.*rank 1",
    ):
        FragmentBuilder(atoms, bonds, ["C", "C", "C", "C"]).materialize_fragment(
            "linker",
            4,
            {0, 1, 2, 3},
            (PeriodicBond(0, 3, (0, 0, 0)),),
        )


def test_organic_linker_splitter_rejects_terminal_fragments_directly() -> None:
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    atoms = Atoms(
        symbols=["Zn", "C"],
        positions=positions,
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    bonds = make_bond_network(positions, [(0, 1)])
    terminal = make_fragment("linker", 1, (1,), ((0, 1),))

    with pytest.raises(ValueError, match="fewer than 2 connection sites"):
        OrganicLinkerSplitter(atoms, bonds, ["Zn", "C"]).split(
            [terminal],
            ((0, 1),),
        )


def mapped_fragment_indices(result: NodeEdgeDecompositionResult) -> list[int]:
    assert result.selected_to_original is not None
    return [
        result.selected_to_original[index]
        for fragment in result.fragments
        for index in fragment.source_indices
    ]


def test_node_edge_result_preserves_legacy_positional_field_order() -> None:
    metal_connecting_bonds = ((0, 1),)
    organic_connecting_bonds = ((1, 2),)

    result = NodeEdgeDecompositionResult(
        "legacy",
        (),
        (),
        (),
        (),
        (10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
        [],
        [],
        [],
        [],
        [],
        metal_connecting_bonds,
        organic_connecting_bonds,
    )

    assert result.metal_connecting_bonds == metal_connecting_bonds
    assert result.organic_connecting_bonds == organic_connecting_bonds
    assert result.bound_linker_fragments == []
    assert result.connection_records == ()
    assert result.periodic_openings == ()
    assert result.periodic_validation == {}


def test_relabel_fragment_preserves_source_image_offsets() -> None:
    fragment = make_fragment("node", 4, (0,), ((0, 1),))
    fragment.source_image_offsets = ((0, (2, -1, 0)),)

    relabeled = edge_decomposer.relabel_fragment(fragment, "metal_node", 2)

    assert relabeled.source_image_offsets == ((0, (2, -1, 0)),)


def test_logical_port_model_is_additive_public_and_survives_relabeling() -> None:
    from mofbb import LogicalPort

    atoms = Atoms(symbols=["C"], positions=[(0.0, 0.0, 0.0)])
    legacy = Fragment("node", 1, (0,), atoms, (), "legacy-fingerprint")
    port = LogicalPort(1, (2, 3), "metal_node", 4, (1, -1, 0))

    assert legacy.logical_kind is None
    assert legacy.logical_ports == ()
    assert port.as_dict() == {
        "logical_port_id": 1,
        "x_indices": [2, 3],
        "neighbor_kind": "metal_node",
        "neighbor_index": 4,
        "translation": [1, -1, 0],
    }

    legacy.logical_kind = "organic_node"
    legacy.logical_ports = (port,)
    relabeled = edge_decomposer.relabel_fragment(legacy, "metal_node", 2)

    assert relabeled.logical_kind == "organic_node"
    assert relabeled.logical_ports == (port,)


def test_decomposers_open_periodic_metal_components_and_publish_validation(
    monkeypatch,
) -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "Zn", "Zn"],
        [(0, 1), (1, 2), (0, 2)],
        shifts={(0, 2): (-1, 0, 0)},
    )
    monkeypatch.setattr(
        "mofbb.decompose.decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )

    single = SingleNodeDecomposer("periodic-metal.cif").decompose()
    node_edge = NodeEdgeDecomposer("periodic-metal.cif").decompose()

    expected_validation = {
        "all_source_atoms_owned_once": True,
        "all_fragments_zero_holonomy": True,
        "all_retained_bonds_undistorted": True,
    }
    for result in (single, node_edge):
        assert len(result.connection_records) == 1
        assert result.connecting_bonds == (result.connection_records[0].pair,)
        assert len(result.periodic_openings) == 1
        assert result.periodic_openings[0].cuts == result.connection_records
        assert result.periodic_validation == expected_validation
        assert len(result.topology_edges) == 1
        assert result.topology_edges[0].slot_translation != (0, 0, 0)
        assert result.topology_vertices[0].coordination == 2
    assert node_edge.metal_connecting_bonds == ()
    assert node_edge.organic_connecting_bonds == ()


def test_partition_bound_linkers_requires_whole_original_containment() -> None:
    contained = make_fragment("linker", 1, (1, 2), ())
    partial_overlap = make_fragment("linker", 2, (0, 1), ())
    disjoint = make_fragment("linker", 3, (3,), ())

    accessories, splittable = edge_decomposer._partition_bound_linkers(
        [contained, partial_overlap, disjoint],
        selected_to_original=(10, 11, 12, 13),
        bound_solvent_indices=(11, 12),
    )

    assert accessories == [contained]
    assert splittable == [partial_overlap, disjoint]


@pytest.mark.parametrize(
    ("mode", "mapping_name"),
    [("free", "fsr_indices"), ("all", "asr_indices")],
)
def test_node_edge_active_modes_disable_cleanup_and_classify_once(
    monkeypatch,
    framework_with_free_and_bound_solvent,
    mode,
    mapping_name,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    cleanup_values = []
    classify_calls = []
    original_classify = MofidSolventRemover.classify

    def fake_read(*args, cleanup, **kwargs):
        cleanup_values.append(cleanup)
        return atoms.copy(), bonds

    def counting_classify(remover):
        classify_calls.append(remover)
        return original_classify(remover)

    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure", fake_read
    )
    monkeypatch.setattr(MofidSolventRemover, "classify", counting_classify)

    result = NodeEdgeDecomposer("fixture.cif", solvent_mode=mode).decompose()

    assert cleanup_values == [False]
    assert len(classify_calls) == 1
    assert result.solvent is not None
    assert result.selected_to_original == getattr(result.solvent, mapping_name)


def test_node_edge_free_mode_preserves_bound_solvent_as_accessory_linker(
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    cleanup_values = []
    classify_calls = []
    splitter_inputs = []
    original_classify = MofidSolventRemover.classify
    original_split = OrganicLinkerSplitter.split

    def fake_read(*args, cleanup, **kwargs):
        cleanup_values.append(cleanup)
        return atoms.copy(), bonds

    def counting_classify(remover):
        classify_calls.append(remover)
        return original_classify(remover)

    def inspecting_split(splitter, organic_fragments, metal_connecting_bonds):
        splitter_inputs.extend(organic_fragments)
        return original_split(splitter, organic_fragments, metal_connecting_bonds)

    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure", fake_read
    )
    monkeypatch.setattr(MofidSolventRemover, "classify", counting_classify)
    monkeypatch.setattr(OrganicLinkerSplitter, "split", inspecting_split)

    result = NodeEdgeDecomposer("fixture.cif", solvent_mode="free").decompose()

    assert cleanup_values == [False]
    assert len(classify_calls) == 1
    assert result.solvent_mode == "free"
    assert result.solvent is not None
    assert result.selected_to_original == result.solvent.fsr_indices
    assert len(result.bound_linker_fragments) == 1
    bound_linker = result.bound_linker_fragments[0]
    assert bound_linker.kind == "linker"
    assert bound_linker.index == 3
    assert bound_linker.source_indices == (8, 9)
    assert bound_linker.num_x == 1
    assert [site.source_cut_bond for site in bound_linker.x_sites] == [(0, 8)]
    assert result.fragments[-1] is bound_linker

    mapped_indices = mapped_fragment_indices(result)
    assert len(mapped_indices) == len(set(mapped_indices))
    assert set(mapped_indices) == set(result.solvent.fsr_indices)
    bound_original = {
        result.selected_to_original[index]
        for index in bound_linker.source_indices
    }
    assert bound_original <= set(result.solvent.bound_solvent_indices)
    assert all(
        not {
            result.selected_to_original[index]
            for index in fragment.source_indices
        }
        <= set(result.solvent.bound_solvent_indices)
        for fragment in splitter_inputs
    )
    assert any(
        vertex.kind == "linker"
        and vertex.index == bound_linker.index
        and vertex.coordination == 1
        for vertex in result.topology_vertices
    )
    assert any(
        (edge.begin_kind, edge.begin_index) == ("linker", bound_linker.index)
        or (edge.end_kind, edge.end_index) == ("linker", bound_linker.index)
        for edge in result.topology_edges
    )


def test_node_edge_free_mode_preserves_zero_cut_bound_accessory(
    tmp_path,
    monkeypatch,
) -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "N", "N", "C"],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
    )
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )

    result = NodeEdgeDecomposer("zero_cut.cif", solvent_mode="free").decompose()

    assert result.solvent is not None
    assert result.solvent.free_solvent_components == ((0,),)
    assert result.solvent.bound_solvent_components == ((1, 2, 3),)
    assert result.selected_to_original == (1, 2, 3)
    assert result.connecting_bonds == ()
    assert len(result.bound_linker_fragments) == 1
    accessory = result.bound_linker_fragments[0]
    assert (accessory.kind, accessory.index) == ("linker", 1)
    assert accessory.source_indices == (0, 1, 2)
    assert accessory.num_x == 0
    assert accessory.x_sites == ()
    assert [
        (vertex.kind, vertex.index, vertex.coordination)
        for vertex in result.topology_vertices
    ] == [("linker", 1, 0)]
    mapped_indices = mapped_fragment_indices(result)
    assert len(mapped_indices) == len(set(mapped_indices))
    assert set(mapped_indices) == set(result.solvent.fsr_indices)

    summary = result.write(tmp_path)

    assert summary["num_bound_linkers"] == 1
    linker_group = next(group for group in summary["groups"] if group["kind"] == "linker")
    assert linker_group["bb_id"] == "linker_group_01"
    assert linker_group["fragment_indices"] == [1]
    assert linker_group["cn"] == 0
    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    assert len(reconstruction["slots"]) == 1
    slot = reconstruction["slots"][0]
    assert {
        key: slot[key]
        for key in (
            "bb_id",
            "cn",
            "debug_instance",
            "formula",
            "kind",
            "slot_id",
            "source_indices",
        )
    } == {
        "bb_id": "linker_group_01",
        "cn": 0,
        "debug_instance": "linkers/linker_001.xyz",
        "formula": "CN2",
        "kind": "linker",
        "slot_id": 0,
        "source_indices": [0, 1, 2],
    }
    assert slot["physical_cn"] == slot["logical_cn"] == 0
    assert slot["logical_kind"] == "linker"
    assert reconstruction["edges"] == []
    assert reconstruction["validation"]["all_edges_mapped"] is True
    assert reconstruction["validation"]["all_x_sites_used_once"] is True


def test_node_edge_free_mode_preserves_multi_cut_bound_accessory(
    tmp_path,
    monkeypatch,
) -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "N", "N", "C", "N", "N", "C"],
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (0, 4),
            (0, 5),
            (4, 6),
            (5, 6),
        ],
        shifts={(0, 2): (1, 0, 0)},
    )
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )

    result = NodeEdgeDecomposer("multi_cut.cif", solvent_mode="free").decompose()

    assert result.solvent is not None
    assert result.solvent.free_solvent_components == ()
    assert result.solvent.bound_solvent_components == ((4, 5, 6),)
    assert result.selected_to_original == tuple(range(7))
    assert result.connecting_bonds == ((0, 1), (0, 2), (0, 4), (0, 5))
    assert len(result.bound_linker_fragments) == 1
    accessory = result.bound_linker_fragments[0]
    assert (accessory.kind, accessory.index) == ("linker", 2)
    assert accessory.source_indices == (4, 5, 6)
    assert accessory.num_x == 2
    assert [site.source_cut_bond for site in accessory.x_sites] == [(0, 4), (0, 5)]
    assert any(
        vertex.kind == "linker"
        and vertex.index == 2
        and vertex.coordination == 2
        for vertex in result.topology_vertices
    )
    mapped_indices = mapped_fragment_indices(result)
    assert len(mapped_indices) == len(set(mapped_indices))
    assert set(mapped_indices) == set(result.solvent.fsr_indices)

    summary = result.write(tmp_path)

    linker_group = next(group for group in summary["groups"] if group["kind"] == "linker")
    assert linker_group["bb_id"] == "linker_group_01"
    assert linker_group["fragment_indices"] == [2]
    assert linker_group["cn"] == 2
    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    assert reconstruction["slots"][2]["source_indices"] == [4, 5, 6]
    assert reconstruction["slots"][2]["debug_instance"] == "linkers/linker_002.xyz"
    accessory_edges = [
        edge for edge in reconstruction["edges"] if 2 in edge["endpoint_slots"]
    ]
    assert [
        {
            "edge_id": edge["edge_id"],
            "endpoint_slots": edge["endpoint_slots"],
            "endpoint_x_sites": [
                {"slot_id": site["slot_id"], "x_index": site["x_index"]}
                for site in edge["endpoint_x_sites"]
            ],
            "source_cut_bond": edge["source_cut_bond"],
        }
        for edge in accessory_edges
    ] == [
        {
            "edge_id": 2,
            "endpoint_slots": [0, 2],
            "endpoint_x_sites": [
                {"slot_id": 0, "x_index": 3},
                {"slot_id": 2, "x_index": 3},
            ],
            "source_cut_bond": [0, 4],
        },
        {
            "edge_id": 3,
            "endpoint_slots": [0, 2],
            "endpoint_x_sites": [
                {"slot_id": 0, "x_index": 4},
                {"slot_id": 2, "x_index": 4},
            ],
            "source_cut_bond": [0, 5],
        },
    ]
    assert all(edge["source_cut_shift"] == [0, 0, 0] for edge in accessory_edges)
    assert all(edge["slot_translation"] == [0, 0, 0] for edge in accessory_edges)
    assert reconstruction["validation"]["all_edges_mapped"] is True
    assert reconstruction["validation"]["all_x_sites_used_once"] is True


def test_node_edge_all_mode_removes_bound_linker(
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )

    result = NodeEdgeDecomposer("fixture.cif", solvent_mode="all").decompose()

    assert result.solvent_mode == "all"
    assert result.solvent is not None
    assert result.bound_linker_fragments == []
    assert result.selected_to_original == result.solvent.asr_indices
    mapped_indices = mapped_fragment_indices(result)
    assert len(mapped_indices) == len(set(mapped_indices))
    assert set(mapped_indices) == set(result.solvent.asr_indices)
    assert all(fragment.kind != "linker" for fragment in result.fragments)


def test_node_edge_rejects_invalid_solvent_mode_before_reading_cif(
    monkeypatch,
) -> None:
    read_called = False

    def fake_read(*args, **kwargs):
        nonlocal read_called
        read_called = True
        raise AssertionError("CIF read should not be attempted")

    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure", fake_read
    )

    with pytest.raises(ValueError, match="solvent mode"):
        NodeEdgeDecomposer("fixture.cif", solvent_mode="invalid")

    assert read_called is False


@pytest.mark.parametrize("cleanup", [True, False])
def test_node_edge_default_mode_preserves_cleanup_and_legacy_state(
    monkeypatch,
    cleanup,
) -> None:
    symbols = ["Zn", "Zn", "N", "C", "N"]
    positions = [(float(index), 0.0, 0.0) for index in range(len(symbols))]
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    bonds = make_bond_network(positions, [(0, 2), (2, 3), (3, 4), (4, 1)])
    cleanup_values = []

    def fake_read(*args, cleanup, **kwargs):
        cleanup_values.append(cleanup)
        return atoms.copy(), bonds

    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure", fake_read
    )

    result = NodeEdgeDecomposer("fixture.cif", cleanup=cleanup).decompose()

    assert cleanup_values == [cleanup]
    assert result.solvent_mode == "none"
    assert result.solvent is None
    assert result.selected_to_original is None
    assert result.bound_linker_fragments == []
    covered = [
        index for fragment in result.fragments for index in fragment.source_indices
    ]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(range(len(atoms)))


@pytest.mark.parametrize(
    ("solvent_mode", "call_with_mode"),
    [("none", False), ("free", True), ("all", True)],
)
def test_decompose_mof_with_edges_forwards_solvent_mode(
    monkeypatch,
    solvent_mode,
    call_with_mode,
) -> None:
    sentinel = object()
    calls = []

    class FakeDecomposer:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def decompose(self):
            return sentinel

    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.NodeEdgeDecomposer", FakeDecomposer
    )
    kwargs = {
        "x_symbol": "He",
        "bond_scale": 1.3,
        "cleanup": False,
    }
    if call_with_mode:
        kwargs["solvent_mode"] = solvent_mode

    result = decompose_mof_with_edges("fixture.cif", **kwargs)

    assert result is sentinel
    assert calls == [
        (
            ("fixture.cif",),
            {
                "x_symbol": "He",
                "bond_scale": 1.3,
                "cleanup": False,
                "solvent_mode": solvent_mode,
            },
        )
    ]


def test_node_edge_free_result_writes_accessory_and_solvent_outputs(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = NodeEdgeDecomposer("fixture.cif", solvent_mode="free").decompose()
    bound_linker = result.bound_linker_fragments[0]
    assert bound_linker.index == 3
    assert bound_linker.source_indices == (8, 9)

    summary = result.write(tmp_path)

    assert summary["num_bound_linkers"] == 1
    assert summary["bound_linkers"] == [
        bound_linker.as_dict(f"linkers/linker_{bound_linker.index:03d}.xyz")
    ]
    assert summary["solvent"]["mode"] == "free"
    assert summary["solvent"]["selected_to_original"] == list(
        result.selected_to_original
    )
    assert {path.name for path in tmp_path.glob("*.cif")} == SOLVENT_ARTIFACT_NAMES
    assert (tmp_path / "linkers" / f"linker_{bound_linker.index:03d}.xyz").exists()
    assert (
        tmp_path / "all_fragments" / f"linker_{bound_linker.index:03d}.xyz"
    ).exists()
    assert (tmp_path / "bbs" / "linker_group_01.xyz").exists()
    linker_group = next(group for group in summary["groups"] if group["kind"] == "linker")
    assert linker_group["bb_id"] == "linker_group_01"
    assert linker_group["fragment_indices"] == [3]
    assert linker_group["path"] == "bbs/linker_group_01.xyz"

    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    slot = reconstruction["slots"][-1]
    assert {
        key: slot[key]
        for key in (
            "bb_id",
            "cn",
            "debug_instance",
            "formula",
            "kind",
            "slot_id",
            "source_indices",
        )
    } == {
        "bb_id": "linker_group_01",
        "cn": 1,
        "debug_instance": "linkers/linker_003.xyz",
        "formula": "CNX",
        "kind": "linker",
        "slot_id": 4,
        "source_indices": [8, 9],
    }
    assert slot["physical_cn"] == slot["logical_cn"] == 1
    assert slot["logical_kind"] == "linker"
    edge = reconstruction["edges"][2]
    assert {
        "edge_id": edge["edge_id"],
        "endpoint_slots": edge["endpoint_slots"],
        "endpoint_x_sites": [
            {"slot_id": site["slot_id"], "x_index": site["x_index"]}
            for site in edge["endpoint_x_sites"]
        ],
        "source_cut_bond": edge["source_cut_bond"],
    } == {
        "edge_id": 2,
        "endpoint_slots": [0, 4],
        "endpoint_x_sites": [
            {"slot_id": 0, "x_index": 3},
            {"slot_id": 4, "x_index": 2},
        ],
        "source_cut_bond": [0, 8],
    }
    assert edge["source_cut_shift"] == [0, 0, 0]
    assert edge["slot_translation"] == [0, 0, 0]
    assert reconstruction["validation"]["all_edges_mapped"] is True
    assert reconstruction["validation"]["all_x_sites_used_once"] is True


def test_node_edge_all_result_writes_solvent_without_bound_linker_keys(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = NodeEdgeDecomposer("fixture.cif", solvent_mode="all").decompose()

    summary = result.write(tmp_path)

    assert summary["solvent"]["mode"] == "all"
    assert {path.name for path in tmp_path.glob("*.cif")} == SOLVENT_ARTIFACT_NAMES
    assert "num_bound_linkers" not in summary
    assert "bound_linkers" not in summary


def test_node_edge_free_mode_without_bound_components_omits_bound_linker_keys(
    tmp_path,
    monkeypatch,
) -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "N", "N", "C"],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        shifts={(0, 2): (1, 0, 0)},
    )
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )

    result = NodeEdgeDecomposer("no_bound.cif", solvent_mode="free").decompose()
    summary = result.write(tmp_path)

    assert result.solvent is not None
    assert result.solvent.bound_solvent_components == ()
    assert result.bound_linker_fragments == []
    assert summary["solvent"]["mode"] == "free"
    assert summary["solvent"]["bound"]["num_components"] == 0
    assert "num_bound_linkers" not in summary
    assert "bound_linkers" not in summary


@pytest.mark.parametrize(
    "inconsistent_state",
    ["missing-solvent", "missing-mapping", "metadata-with-none-mode"],
)
def test_node_edge_write_rejects_inconsistent_solvent_metadata_before_output(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
    inconsistent_state,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = NodeEdgeDecomposer("fixture.cif", solvent_mode="all").decompose()
    if inconsistent_state == "missing-solvent":
        result.solvent = None
    elif inconsistent_state == "missing-mapping":
        result.selected_to_original = None
    else:
        result.solvent_mode = "none"
    output_dir = tmp_path / inconsistent_state

    with pytest.raises(ValueError, match="Inconsistent solvent metadata"):
        result.write(output_dir)

    assert not output_dir.exists()


def test_node_edge_write_preflights_solvent_mapping_before_output(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = NodeEdgeDecomposer("fixture.cif", solvent_mode="all").decompose()
    assert result.selected_to_original is not None
    result.selected_to_original = tuple(reversed(result.selected_to_original))
    output_dir = tmp_path / "wrong-mapping"

    with pytest.raises(
        ValueError,
        match="selected-to-original mapping is inconsistent with mode 'all'",
    ):
        result.write(output_dir)

    assert not output_dir.exists()


def test_node_edge_write_preflights_solvent_cell_before_output(
    tmp_path,
    monkeypatch,
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decompose.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = NodeEdgeDecomposer("fixture.cif", solvent_mode="all").decompose()
    assert result.solvent is not None
    result.solvent.original_atoms.set_pbc(False)
    output_dir = tmp_path / "unrepresentable-cell"

    with pytest.raises(
        ValueError,
        match="CIF output requires either a rank-3 cell with full periodicity",
    ):
        result.write(output_dir)

    assert not output_dir.exists()


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


def test_organic_splitter_absorbs_multidentate_pendant_oxygen_into_core() -> None:
    symbols = ["Zn", "Zn", "Zn", "Zn", "C", "C", "C", "O"]
    positions = [
        (-2.0, 0.0, 0.0),
        (-2.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (0.4, 0.7, 0.0),
        (-0.8, 0.0, 0.0),
    ]
    pairs = [
        (4, 5),
        (5, 6),
        (4, 6),
        (4, 7),
        (0, 7),
        (1, 7),
        (2, 5),
        (3, 6),
    ]
    atoms = Atoms(symbols=symbols, positions=positions, cell=[10, 10, 10], pbc=True)
    bonds = make_bond_network(positions, pairs)
    metal_cuts = ((0, 7), (1, 7), (2, 5), (3, 6))
    raw = FragmentBuilder(atoms, bonds, symbols).materialize_fragment(
        "linker", 1, {4, 5, 6, 7}, metal_cuts
    )

    split = OrganicLinkerSplitter(atoms, bonds, symbols).split([raw], metal_cuts)

    assert len(split.organic_node_fragments) == 1
    assert split.organic_node_fragments[0].source_indices == (4, 5, 6, 7)
    assert split.edge_fragments == []
    assert (4, 7) not in split.organic_connecting_bonds


def test_pendant_donor_rejects_two_bonds_into_one_core() -> None:
    symbols = ["C", "C", "O", "Zn"]
    positions = [(float(index), 0.0, 0.0) for index in range(len(symbols))]
    bonds = make_bond_network(positions, [(0, 1), (0, 2), (1, 2), (2, 3)])
    splitter = OrganicLinkerSplitter(
        Atoms(symbols=symbols, positions=positions), bonds, symbols
    )

    assert not splitter._is_pendant_donor_component(
        {2},
        {0, 1, 2},
        {0: 0, 1: 0},
        ((2, 3),),
    )


def test_pendant_donor_rejects_bridge_between_two_cores() -> None:
    symbols = ["C", "C", "S", "Zn"]
    positions = [(float(index), 0.0, 0.0) for index in range(len(symbols))]
    bonds = make_bond_network(positions, [(0, 2), (1, 2), (2, 3)])
    splitter = OrganicLinkerSplitter(
        Atoms(symbols=symbols, positions=positions), bonds, symbols
    )

    assert not splitter._is_pendant_donor_component(
        {2},
        {0, 1, 2},
        {0: 0, 1: 1},
        ((2, 3),),
    )


def test_absorbed_donor_core_groups_six_physical_sites_into_two_logical_ports() -> None:
    symbols = ["Zn"] * 6 + ["C", "C", "C", "O", "C", "Zn"]
    positions = [(float(index), 0.0, 0.0) for index in range(len(symbols))]
    metal_cuts = (
        (0, 9),
        (1, 9),
        (2, 9),
        (3, 7),
        (4, 8),
        (5, 8),
    )
    pairs = [
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 7),
        (7, 8),
        (6, 8),
        (6, 9),
        *metal_cuts,
    ]
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[30.0, 30.0, 30.0],
        pbc=True,
    )
    bonds = make_bond_network(positions, pairs)
    builder = FragmentBuilder(atoms, bonds, symbols)
    raw_organic = builder.materialize_fragment(
        "linker", 1, {6, 7, 8, 9}, metal_cuts
    )

    split = OrganicLinkerSplitter(atoms, bonds, symbols).split(
        [raw_organic], metal_cuts
    )

    assert len(split.organic_node_fragments) == 1
    assert split.edge_fragments == []
    assert split.organic_connecting_bonds == ()
    organic = split.organic_node_fragments[0]
    assert organic.kind == "organic_node"
    assert organic.source_indices == (6, 7, 8, 9)
    assert organic.num_x == 6

    left = builder.materialize_fragment("metal_node", 1, {0, 1, 2}, metal_cuts)
    right = builder.materialize_fragment("metal_node", 2, {3, 4, 5}, metal_cuts)
    accessory = builder.materialize_fragment("linker", 1, {10}, ())
    generic_node = builder.materialize_fragment("node", 1, {11}, ())
    cut_set = set(metal_cuts)
    records = tuple(
        record for record in periodic_bonds(atoms, bonds) if record.pair in cut_set
    )

    edge_decomposer.assign_logical_ports(
        atoms,
        bonds,
        [generic_node, accessory, organic, right, left],
        tuple(reversed(records)),
    )

    assert organic.logical_kind == "edge"
    assert [port.logical_port_id for port in organic.logical_ports] == [0, 1]
    assert [port.x_indices for port in organic.logical_ports] == [
        (4, 5, 6),
        (7, 8, 9),
    ]
    assert [
        (port.neighbor_kind, port.neighbor_index, port.translation)
        for port in organic.logical_ports
    ] == [
        ("metal_node", 1, (0, 0, 0)),
        ("metal_node", 2, (0, 0, 0)),
    ]
    assert left.logical_kind == "metal_node"
    assert right.logical_kind == "metal_node"
    assert accessory.logical_kind == "linker"
    assert generic_node.logical_kind == "node"


def test_logical_ports_separate_effective_slot_translations() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "Zn", "C", "C"],
        [(0, 1), (2, 3), (0, 2), (1, 3)],
        shifts={(0, 2): (1, 0, 0), (1, 3): (-1, 0, 0)},
    )
    records = (
        PeriodicBond(0, 2, (1, 0, 0)),
        PeriodicBond(1, 3, (-1, 0, 0)),
    )
    builder = FragmentBuilder(atoms, bonds, atoms.get_chemical_symbols())
    metal = builder.materialize_fragment("metal_node", 1, {0, 1}, records)
    organic = builder.materialize_fragment("organic_node", 1, {2, 3}, records)

    edge_decomposer.assign_logical_ports(atoms, bonds, [metal, organic], records)

    assert organic.logical_kind == "edge"
    assert [port.x_indices for port in organic.logical_ports] == [(2,), (3,)]
    assert [port.translation for port in organic.logical_ports] == [
        (-1, 0, 0),
        (1, 0, 0),
    ]


def test_logical_ports_quotient_neighbor_periodic_self_translation() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "Zn", "Zn", "C", "C"],
        [(0, 1), (1, 2), (0, 2), (3, 4), (0, 3), (2, 4)],
        shifts={(0, 2): (-2, 0, 0), (2, 4): (-2, 0, 0)},
    )
    self_cut = PeriodicBond(0, 2, (-2, 0, 0))
    connections = (
        self_cut,
        PeriodicBond(0, 3, (0, 0, 0)),
        PeriodicBond(2, 4, (-2, 0, 0)),
    )
    builder = FragmentBuilder(atoms, bonds, atoms.get_chemical_symbols())
    metal = builder.materialize_fragment("metal_node", 1, {0, 1, 2}, connections)
    organic = builder.materialize_fragment("organic_node", 1, {3, 4}, connections)

    edge_decomposer.assign_logical_ports(
        atoms, bonds, [metal, organic], connections
    )

    assert len(organic.logical_ports) == 1
    assert organic.logical_ports[0].x_indices == (2, 3)
    assert organic.logical_ports[0].translation == (0, 0, 0)


def test_logical_ports_keep_two_sides_of_own_periodic_self_edge() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "Zn", "Zn"],
        [(0, 1), (1, 2), (0, 2)],
        shifts={(0, 2): (-2, 0, 0)},
    )
    self_cut = PeriodicBond(0, 2, (-2, 0, 0))
    metal = FragmentBuilder(
        atoms, bonds, atoms.get_chemical_symbols()
    ).materialize_fragment("metal_node", 1, {0, 1, 2}, (self_cut,))

    edge_decomposer.assign_logical_ports(atoms, bonds, [metal], (self_cut,))

    assert metal.logical_kind == "metal_node"
    assert [port.logical_port_id for port in metal.logical_ports] == [0, 1]
    assert [port.x_indices for port in metal.logical_ports] == [(3,), (4,)]
    assert [port.translation for port in metal.logical_ports] == [
        (-2, 0, 0),
        (2, 0, 0),
    ]


def test_logical_ports_reject_independent_self_edge_generators() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn"] * 4,
        [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)],
        shifts={(0, 2): (1, 0, 0), (1, 3): (0, 1, 0)},
    )
    self_cuts = (
        PeriodicBond(0, 2, (1, 0, 0)),
        PeriodicBond(1, 3, (0, 1, 0)),
    )
    metal = FragmentBuilder(
        atoms, bonds, atoms.get_chemical_symbols()
    ).materialize_fragment("metal_node", 1, set(range(4)), self_cuts)

    with pytest.raises(ValueError, match="independent self-edge translations"):
        edge_decomposer.assign_logical_ports(atoms, bonds, [metal], self_cuts)


def test_logical_port_assignment_rejects_site_without_connection_record() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "C"],
        [(0, 1)],
    )
    record = PeriodicBond(0, 1, (0, 0, 0))
    builder = FragmentBuilder(atoms, bonds, atoms.get_chemical_symbols())
    metal = builder.materialize_fragment("metal_node", 1, {0}, (record,))
    organic = builder.materialize_fragment("edge", 1, {1}, (record,))

    with pytest.raises(ValueError, match="X site.*connection record"):
        edge_decomposer.assign_logical_ports(atoms, bonds, [metal, organic], ())


def test_node_edge_topology_uses_exact_shifts_for_parallel_edges() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "Zn", "C", "C"],
        [(0, 1), (2, 3), (0, 2), (1, 3)],
        shifts={(0, 2): (1, 0, 0), (1, 3): (-1, 0, 0)},
    )
    metal_node = make_fragment("metal_node", 1, (0, 1), ((0, 2), (1, 3)))
    edge = make_fragment("edge", 1, (2, 3), ((0, 2), (1, 3)))
    connections = (
        PeriodicBond(0, 2, (1, 0, 0)),
        PeriodicBond(1, 3, (-1, 0, 0)),
    )

    _, topology_edges = NodeEdgeTopologyBuilder(atoms, bonds).build(
        [metal_node, edge],
        connections,
    )

    assert len(topology_edges) == 2
    assert topology_edges[0].end_frac != topology_edges[1].end_frac
    assert [edge.source_cut_bond for edge in topology_edges] == [(0, 2), (1, 3)]
    assert [edge.slot_translation for edge in topology_edges] == [
        (1, 0, 0),
        (-1, 0, 0),
    ]


def test_node_edge_topology_keeps_non_edge_connections_centroid_based() -> None:
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["Zn", "Zn", "C", "C"],
        [(0, 1), (2, 3), (0, 2), (1, 3)],
    )
    metal_node = make_fragment("metal_node", 1, (0, 1), ((0, 2), (1, 3)))
    organic_node = make_fragment("organic_node", 1, (2, 3), ((0, 2), (1, 3)))
    connections = (
        PeriodicBond(0, 2, (0, 0, 0)),
        PeriodicBond(1, 3, (0, 0, 0)),
    )

    _, topology_edges = NodeEdgeTopologyBuilder(atoms, bonds).build(
        [metal_node, organic_node],
        connections,
    )

    assert len(topology_edges) == 2
    assert topology_edges[0].end_frac == topology_edges[1].end_frac
    assert [edge.source_cut_bond for edge in topology_edges] == [(0, 2), (1, 3)]


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
            ),
            TopologyEdge(
                "edge",
                1,
                "metal_node",
                1,
                (0.50, 0.25, 0.25),
                (0.25, 0.25, 0.25),
                (0.375, 0.25, 0.25),
            ),
            TopologyEdge(
                "metal_node",
                1,
                "organic_node",
                1,
                (0.25, 0.25, 0.25),
                (0.75, 0.25, 0.25),
                (0.50, 0.25, 0.25),
            ),
            TopologyEdge(
                "metal_node",
                1,
                "organic_node",
                1,
                (0.25, 0.25, 0.25),
                (0.75, 0.25, 0.25),
                (0.50, 0.25, 0.25),
            ),
            TopologyEdge(
                "organic_node",
                1,
                "metal_node",
                1,
                (0.75, 0.25, 0.25),
                (0.25, 0.25, 0.25),
                (0.50, 0.25, 0.25),
            ),
        ],
        metal_node_fragments=[metal_node],
        organic_node_fragments=[organic_node],
        edge_fragments=[edge],
    )

    summary = result.write(tmp_path)

    assert set(summary) == LEGACY_NODE_EDGE_SUMMARY_KEYS
    assert summary["num_metal_nodes"] == 1
    assert summary["num_organic_nodes"] == 1
    assert summary["num_edges"] == 1
    assert summary["metal_nodes"][0]["path"] == "metal_nodes/metal_node_001.xyz"
    assert summary["organic_nodes"][0]["path"] == "organic_nodes/organic_node_001.xyz"
    assert summary["edges"][0]["path"] == "edges/edge_001.xyz"
    assert (tmp_path / "bbs" / "organic_node_group_01.xyz").exists()
    assert (tmp_path / "edges" / "edge_001.xyz").exists()
    assert not ({path.name for path in tmp_path.glob("*.cif")} & SOLVENT_ARTIFACT_NAMES)

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

    result = NodeEdgeDecomposer(sample).decompose()
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

    result = NodeEdgeDecomposer(stripped_sample).decompose()
    summary = result.write(tmp_path / "decomposed")

    _assert_pormake_acs_h_edge_summary(summary)
