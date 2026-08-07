"""Node-edge decomposition output."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mofbb.decompose.export.cgd import CgdWriter, edge_types
from mofbb.decompose.export.legacy import LegacyMap, fragment_instance_path
from mofbb.decompose.export.ports import PortMapper
from mofbb.decompose.export.symmetry import SymmetryBuilder
from mofbb.decompose.object import Fragment, TopologyEdge, TopologyVertex
from mofbb.decompose.pbc import PeriodicBond
from mofbb.decompose.solvent import SolventMode, SolventRemovalResult
from mofbb.decompose.symmetry import AtomicSymmetry, SpaceGroupOperations, TopologySymmetry
from mofbb.decompose.topology import EdgeTopologyContractor


def logical_port_partition(fragment: Fragment) -> tuple[tuple[int, ...], ...]:
    return tuple(port.x_indices for port in fragment.logical_ports)


def fragment_group_key(fragment: Fragment) -> tuple[object, ...]:
    port_sizes = tuple(sorted(map(len, logical_port_partition(fragment))))
    return (
        fragment.fingerprint,
        fragment.logical_kind or fragment.kind,
        port_sizes,
    )


def fragment_group_metadata(fragment: Fragment) -> dict[str, object]:
    port_sizes = tuple(sorted(map(len, logical_port_partition(fragment))))
    return {
        "physical_cn": fragment.num_x,
        "logical_cn": len(port_sizes) if port_sizes else fragment.num_x,
        "logical_kind": fragment.logical_kind or fragment.kind,
        "logical_port_sizes": list(port_sizes),
        "logical_ports": [
            {"logical_port_id": index, "size": size}
            for index, size in enumerate(port_sizes)
        ],
    }


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_topology_cgd(
    path: Path,
    name: str,
    cell_parameters: tuple[float, float, float, float, float, float],
    vertices: list[TopologyVertex],
    edges: list[TopologyEdge],
    fragment_to_bb_id: dict[tuple[str, int], str],
    edge_bb_ids: dict[int, str] | None = None,
    assembly_symmetry: (
        tuple[str, np.ndarray, np.ndarray] | SpaceGroupOperations | None
    ) = None,
    decorated_symmetry: TopologySymmetry | None = None,
    port_maps: dict[int, list[int]] | None = None,
) -> dict[str, object]:
    """Write the explicit topology with its graph-derived symmetry."""

    source = assembly_symmetry
    if source is not None and not isinstance(source, SpaceGroupOperations):
        source = SpaceGroupOperations(*source)
    return CgdWriter(
        path=Path(path),
        name=name,
        cell=cell_parameters,
        nodes=vertices,
        edges=edges,
        bb_ids=fragment_to_bb_id,
        edge_bbs=edge_bb_ids,
        src_sym=source,
        slot_sym=decorated_symmetry,
        ports=port_maps,
    ).write()


@dataclass
class NodeEdgeDecompositionResult:
    """Metal-node, organic-node, and edge fragments for one MOF."""

    name: str
    cell_parameters: tuple[float, float, float, float, float, float]
    topology_vertices: list[TopologyVertex]
    topology_edges: list[TopologyEdge]
    metal_node_fragments: list[Fragment] = field(default_factory=list)
    organic_node_fragments: list[Fragment] = field(default_factory=list)
    edge_fragments: list[Fragment] = field(default_factory=list)
    bound_linker_fragments: list[Fragment] = field(default_factory=list)
    solvent_mode: SolventMode = "none"
    solvent: SolventRemovalResult | None = None
    selected_to_original: tuple[int, ...] | None = None
    connection_records: tuple[PeriodicBond, ...] = ()
    assembly_spacegroup: str | None = None
    assembly_rotations: np.ndarray | None = None
    assembly_translations: np.ndarray | None = None
    assembly_atom_maps: np.ndarray | None = None
    assembly_atom_images: np.ndarray | None = None

    @property
    def fragments(self) -> list[Fragment]:
        return [
            *self.metal_node_fragments,
            *self.organic_node_fragments,
            *self.edge_fragments,
            *self.bound_linker_fragments,
        ]

    @property
    def fragment_groups(self) -> tuple[tuple[str, list[Fragment]], ...]:
        return (
            ("metal_node", self.metal_node_fragments),
            ("organic_node", self.organic_node_fragments),
            ("edge", self.edge_fragments),
            ("linker", self.bound_linker_fragments),
        )

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                atom_index
                for fragment in self.fragments
                for atom_index in fragment.source_indices
            )
        )

    @property
    def connecting_bonds(self) -> tuple[tuple[int, int], ...]:
        return tuple(record.pair for record in self.connection_records)

    def write(
        self,
        output_dir: Path,
        *,
        write_topology: bool = False,
    ) -> dict[str, object]:
        active = self.solvent_mode != "none"
        if active != (self.solvent is not None) or active != (
            self.selected_to_original is not None
        ):
            raise ValueError(
                f"Inconsistent solvent metadata for mode {self.solvent_mode!r}."
            )
        if active:
            self.solvent.validate_write_request(
                self.solvent_mode,
                self.selected_to_original,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for fragment in self.fragments:
            fragment.write_xyz(output_dir / fragment_instance_path(fragment))

        groups, fragment_to_bb_id = self._write_groups(output_dir)
        topology = None
        if write_topology:
            topology = self.write_topology_cgd(
                output_dir / "topology.cgd",
                fragment_to_bb_id,
            )
            topology["path"] = "topology.cgd"
        self.write_reconstruction_map(
            output_dir / "reconstruction_map.json",
            groups,
            fragment_to_bb_id,
            topology_path="topology.cgd" if write_topology else None,
        )
        summary = self._summary(topology, groups)
        if active:
            summary["solvent"] = self.solvent.write(
                output_dir,
                self.solvent_mode,
                self.selected_to_original,
            )
        _write_json(output_dir / "decomposition_summary.json", summary)
        return summary

    def _write_groups(
        self,
        output_dir: Path,
    ) -> tuple[list[dict[str, object]], dict[tuple[str, int], str]]:
        records: list[dict[str, object]] = []
        fragment_to_bb_id: dict[tuple[str, int], str] = {}
        for kind, fragments in self.fragment_groups:
            grouped: dict[tuple[object, ...], list[Fragment]] = defaultdict(list)
            for fragment in fragments:
                grouped[fragment_group_key(fragment)].append(fragment)
            for group_index, group in enumerate(
                (group for _, group in sorted(grouped.items())),
                start=1,
            ):
                representative = group[0]
                bb_id = f"{kind}_group_{group_index:02d}"
                group_path = Path("bbs") / f"{bb_id}.xyz"
                representative.write_xyz(output_dir / group_path)
                for fragment in group:
                    fragment_to_bb_id[(fragment.kind, fragment.index)] = bb_id
                records.append(
                    {
                        "bb_id": bb_id,
                        "kind": kind,
                        "group": group_index,
                        "count": len(group),
                        "cn": len(representative.x_sites),
                        "formula": representative.formula,
                        "num_atoms": representative.num_atoms,
                        "num_x": representative.num_x,
                        **fragment_group_metadata(representative),
                        "x_sites": [
                            {"site_id": index, "x_index": site.x_index}
                            for index, site in enumerate(representative.x_sites)
                        ],
                        "fingerprint": representative.fingerprint,
                        "fragment_indices": [fragment.index for fragment in group],
                        "path": str(group_path),
                    }
                )
        return records, fragment_to_bb_id

    def _contracted_topology(self):
        return EdgeTopologyContractor(
            self.topology_vertices,
            self.topology_edges,
            self.edge_fragments,
        ).contract()

    def _decorated_topology_symmetry(
        self,
        vertices: list[TopologyVertex],
        edges: list[TopologyEdge],
        topology_slots: dict[tuple[str, int], tuple[str, int]],
        fragment_to_bb_id: dict[tuple[str, int], str],
        edge_colors: list[int],
    ) -> TopologySymmetry | None:
        """Return the source subgroup preserving BB slots and selected cuts."""

        source = self._atomic_symmetry()
        if source is None:
            return None
        return SymmetryBuilder(
            cell=self.cell_parameters,
            nodes=vertices,
            edges=edges,
            slot_map=topology_slots,
            frags=self.fragments,
            bonds=self.connection_records,
            bb_ids=fragment_to_bb_id,
            colors=edge_colors,
            src=source,
        ).build()

    def _topology_port_maps(
        self,
        vertices: list[TopologyVertex],
        edges: list[TopologyEdge],
        topology_slots: dict[tuple[str, int], tuple[str, int]],
    ) -> dict[int, list[int]]:
        """Map topology incidences to each emitted BB's connection ports."""

        return PortMapper(
            src_nodes=self.topology_vertices,
            src_edges=self.topology_edges,
            nodes=vertices,
            edges=edges,
            slot_map=topology_slots,
            frags=self.fragments,
        ).build()

    def _atomic_symmetry(self) -> AtomicSymmetry | None:
        values = (
            self.assembly_spacegroup,
            self.assembly_rotations,
            self.assembly_translations,
            self.assembly_atom_maps,
            self.assembly_atom_images,
        )
        if any(value is None for value in values):
            return None
        return AtomicSymmetry(*values)

    def _space_group_operations(self) -> SpaceGroupOperations | None:
        source = self._atomic_symmetry()
        if source is None:
            return None
        return SpaceGroupOperations(
            source.symbol,
            source.rotations,
            source.translations,
        )

    def write_topology_cgd(
        self,
        path: Path,
        fragment_to_bb_id: dict[tuple[str, int], str] | None = None,
    ) -> dict[str, object]:
        if fragment_to_bb_id is None:
            fragment_to_bb_id = {
                (fragment.kind, fragment.index): repr(fragment_group_key(fragment))
                for fragment in self.fragments
            }
        vertices, edges, topology_slots = self._contracted_topology()
        edge_bb_ids = {
            index: fragment_to_bb_id[key]
            for key, (kind, index) in topology_slots.items()
            if kind == "edge"
        }
        edge_colors, _ = edge_types(len(edges), edge_bb_ids)
        decorated_symmetry = self._decorated_topology_symmetry(
            vertices,
            edges,
            topology_slots,
            fragment_to_bb_id,
            edge_colors,
        )
        port_maps = self._topology_port_maps(vertices, edges, topology_slots)
        return _write_topology_cgd(
            path=Path(path),
            name=self.name,
            cell_parameters=self.cell_parameters,
            vertices=vertices,
            edges=edges,
            fragment_to_bb_id=fragment_to_bb_id,
            edge_bb_ids=edge_bb_ids,
            assembly_symmetry=self._space_group_operations(),
            decorated_symmetry=decorated_symmetry,
            port_maps=port_maps,
        )

    def write_reconstruction_map(
        self,
        path: Path,
        group_records: list[dict[str, object]],
        fragment_to_bb_id: dict[tuple[str, int], str],
        topology_path: str | None = None,
    ) -> dict[str, object]:
        _, _, topology_slots = self._contracted_topology()
        data = LegacyMap(
            self,
            fragment_to_bb_id,
            topology_slots,
        ).build(group_records, topology_path)
        _write_json(Path(path), data)
        return data

    def _reconstruction_slots(
        self,
        fragment_to_bb_id: dict[tuple[str, int], str],
        topology_slots: dict[tuple[str, int], tuple[str, int]],
    ) -> list[dict[str, object]]:
        return LegacyMap(self, fragment_to_bb_id, topology_slots).slots()

    def _reconstruction_edges(self) -> list[dict[str, object]]:
        return LegacyMap(self, {}, {}).edges()

    def _summary(
        self,
        topology: dict[str, object] | None,
        groups: list[dict[str, object]],
    ) -> dict[str, object]:
        summary = {
            "name": self.name,
            "decomposition": "node_edge",
            "num_metal_nodes": len(self.metal_node_fragments),
            "num_organic_nodes": len(self.organic_node_fragments),
            "num_edges": len(self.edge_fragments),
            "num_connecting_bonds": len(self.connecting_bonds),
            "reconstruction_map": "reconstruction_map.json",
            "groups": groups,
        }
        if topology is not None:
            summary["topology"] = topology
        if self.bound_linker_fragments:
            summary["num_bound_linkers"] = len(self.bound_linker_fragments)
        return summary
