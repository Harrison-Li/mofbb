"""Node-edge decomposition output."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import format_float
from .object import Fragment, TopologyEdge, TopologyVertex
from .pbc import PeriodicBond
from .solvent import SolventMode, SolventRemovalResult
from .symmetry import detect_topology_symmetry
from .topology import EdgeTopologyContractor


def fragment_instance_path(fragment: Fragment) -> str:
    return f"{fragment.kind}s/{fragment.kind}_{fragment.index:03d}.xyz"


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
) -> dict[str, object]:
    """Write the explicit topology with its graph-derived symmetry."""

    bb_types: dict[str, int] = {}
    vertex_types = [
        bb_types.setdefault(
            fragment_to_bb_id[(vertex.kind, vertex.index)],
            len(bb_types) + 1,
        )
        for vertex in vertices
    ]
    edge_color_ids: dict[str | None, int] = {}
    edge_colors = [
        edge_color_ids.setdefault(
            None if edge_bb_ids is None else edge_bb_ids.get(index),
            len(edge_color_ids) + 1,
        )
        for index in range(len(edges))
    ]
    symmetry = detect_topology_symmetry(
        cell_parameters,
        vertices,
        edges,
        vertex_types,
        edge_colors,
    )
    positions, endpoints = symmetry.project(vertices, edges)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("CRYSTAL\n")
        handle.write(f"  NAME {name}\n")
        handle.write(f"  GROUP {symmetry.symbol}\n")
        cell = " ".join(map(format_float, cell_parameters))
        handle.write(f"  CELL {cell}\n")
        handle.write("  # EXPANDED\n")
        for rotation, translation in zip(
            symmetry.rotations,
            symmetry.translations,
        ):
            values = [
                *(str(int(value)) for value in rotation.ravel()),
                *(format_float(value) for value in translation),
            ]
            handle.write(f"  # SYMOP {' '.join(values)}\n")
        for vertex_type, vertex, position in zip(
            vertex_types,
            vertices,
            positions,
        ):
            position = " ".join(map(format_float, position))
            handle.write(f"  NODE {vertex_type} {vertex.coordination} {position}\n")
        for index, (edge_type, (begin, end)) in enumerate(
            zip(symmetry.edge_types, endpoints),
            start=1,
        ):
            handle.write(f"  # EDGE_TYPE {index} {edge_type}\n")
            begin = " ".join(map(format_float, begin))
            end = " ".join(map(format_float, end))
            handle.write(f"  EDGE  {begin}   {end}\n")
        for begin, end in endpoints:
            center = " ".join(map(format_float, (begin + end) / 2))
            handle.write(f"  # EDGE_CENTER  {center}\n")
        handle.write("END\n")
    return {
        "num_vertices": len(vertices),
        "num_edges": len(edges),
        "num_node_types": len(bb_types),
        "spacegroup": symmetry.symbol,
        "num_symmetry_operations": len(symmetry.rotations),
    }


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
        return _write_topology_cgd(
            Path(path),
            self.name,
            self.cell_parameters,
            vertices,
            edges,
            fragment_to_bb_id,
            edge_bb_ids,
        )

    def write_reconstruction_map(
        self,
        path: Path,
        group_records: list[dict[str, object]],
        fragment_to_bb_id: dict[tuple[str, int], str],
        topology_path: str | None = None,
    ) -> dict[str, object]:
        _, _, topology_slots = self._contracted_topology()
        slots = self._reconstruction_slots(fragment_to_bb_id, topology_slots)
        topology: dict[str, object] = {
            "kind": "node_edge_slots",
            "cell": list(self.cell_parameters),
        }
        if topology_path:
            topology["path"] = topology_path
        data = {
            "schema_version": 2,
            "topology": topology,
            "bb_groups": group_records,
            "slots": slots,
            "edges": self._reconstruction_edges(),
        }
        _write_json(Path(path), data)
        return data

    def _reconstruction_slots(
        self,
        fragment_to_bb_id: dict[tuple[str, int], str],
        topology_slots: dict[tuple[str, int], tuple[str, int]],
    ) -> list[dict[str, object]]:
        topology_positions = {
            (vertex.kind, vertex.index): vertex.frac_position
            for vertex in self.topology_vertices
        }
        slots = []
        for slot_id, fragment in enumerate(self.fragments):
            key = (fragment.kind, fragment.index)
            position = fragment.real_atom_centroid_frac or topology_positions[key]
            slot_kind, slot_index = topology_slots[key]
            slots.append(
                {
                    "slot_id": slot_id,
                    "kind": fragment.kind,
                    "cn": fragment.num_x,
                    "logical_kind": fragment.logical_kind or fragment.kind,
                    "logical_ports": [
                        port.as_dict() for port in fragment.logical_ports
                    ],
                    "bb_id": fragment_to_bb_id[key],
                    "exact_instance_path": fragment_instance_path(fragment),
                    "source_indices": list(fragment.source_indices),
                    "source_image_offsets": [
                        [atom_index, list(offset)]
                        for atom_index, offset in fragment.source_image_offsets
                    ],
                    "exact_frac_position": list(position),
                    "topology_slot": {"kind": slot_kind, "index": slot_index},
                }
            )
        return slots

    def _reconstruction_edges(self) -> list[dict[str, object]]:
        fragments = self.fragments
        slot_by_key = {
            (fragment.kind, fragment.index): slot_id
            for slot_id, fragment in enumerate(fragments)
        }
        slot_by_atom = {
            atom_index: slot_id
            for slot_id, fragment in enumerate(fragments)
            for atom_index in fragment.source_indices
        }
        x_sites = {
            (
                slot_id,
                site.source_cut_bond,
                site.source_cut_shift,
                site.internal_source_index,
            ): site
            for slot_id, fragment in enumerate(fragments)
            for site in fragment.x_sites
        }
        topology_by_record: dict[
            tuple[tuple[int, int], tuple[int, int, int]], list[TopologyEdge]
        ] = defaultdict(list)
        for edge in self.topology_edges:
            topology_by_record[(edge.source_cut_bond, edge.source_cut_shift)].append(
                edge
            )

        edges = []
        for edge_id, record in enumerate(self.connection_records):
            endpoint_slots = [slot_by_atom[record.i], slot_by_atom[record.j]]
            endpoint_sites = [
                x_sites[(slot_id, record.pair, record.shift, atom_index)]
                for atom_index, slot_id in zip(record.pair, endpoint_slots)
            ]
            topology_edge = topology_by_record[(record.pair, record.shift)].pop(0)
            topology_endpoints = (
                slot_by_key[(topology_edge.begin_kind, topology_edge.begin_index)],
                slot_by_key[(topology_edge.end_kind, topology_edge.end_index)],
            )
            translation = topology_edge.slot_translation
            if topology_endpoints == tuple(reversed(endpoint_slots)):
                translation = tuple(-value for value in translation)
            edges.append(
                {
                    "edge_id": edge_id,
                    "endpoint_slots": endpoint_slots,
                    "endpoint_x_sites": [
                        {
                            "slot_id": slot_id,
                            "x_index": site.x_index,
                            "internal_source_index": site.internal_source_index,
                            "external_source_index": site.external_source_index,
                        }
                        for slot_id, site in zip(endpoint_slots, endpoint_sites)
                    ],
                    "source_cut_bond": list(record.pair),
                    "source_cut_shift": list(record.shift),
                    "slot_translation": list(translation),
                }
            )
        return edges

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
