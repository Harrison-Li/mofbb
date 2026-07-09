"""Output writers for SingleNode-style decomposition results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path

from .geometry import format_float
from .models import Fragment, TopologyEdge, TopologyVertex


def fragment_instance_path(fragment: Fragment) -> str:
    """Return the stable per-instance debug path for a fragment."""

    if fragment.kind == "node":
        return f"nodes/node_{fragment.index:03d}.xyz"
    if fragment.kind == "linker":
        return f"linkers/linker_{fragment.index:03d}.xyz"
    return f"{fragment.kind}s/{fragment.kind}_{fragment.index:03d}.xyz"


@dataclass
class DecompositionResult:
    """Complete SingleNode-style decomposition for one MOF."""

    name: str
    node_indices: tuple[int, ...]
    linker_indices: tuple[int, ...]
    points_of_extension: tuple[int, ...]
    connecting_bonds: tuple[tuple[int, int], ...]
    cell_parameters: tuple[float, float, float, float, float, float]
    topology_vertices: list[TopologyVertex]
    topology_edges: list[TopologyEdge]
    node_fragments: list[Fragment] = field(default_factory=list)
    linker_fragments: list[Fragment] = field(default_factory=list)

    @property
    def fragments(self) -> list[Fragment]:
        """Return all node and linker fragments in output slot order."""

        return [*self.node_fragments, *self.linker_fragments]

    def write(self, output_dir: Path) -> dict[str, object]:
        """Write all output files and return a JSON-serializable summary."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for fragment in self.node_fragments:
            fragment.write_xyz(output_dir / fragment_instance_path(fragment))
            fragment.write_xyz(
                output_dir / "all_fragments" / f"node_{fragment.index:03d}.xyz"
            )
        for fragment in self.linker_fragments:
            fragment.write_xyz(output_dir / fragment_instance_path(fragment))
            fragment.write_xyz(
                output_dir / "all_fragments" / f"linker_{fragment.index:03d}.xyz"
            )

        group_records, fragment_to_bb_id = self._write_group_representatives(output_dir)

        topology_path = output_dir / "topology.cgd"
        topology_summary = self.write_topology_cgd(topology_path)
        topology_summary["path"] = str(topology_path.relative_to(output_dir))

        reconstruction_path = output_dir / "reconstruction_map.json"
        self.write_reconstruction_map(
            reconstruction_path, group_records, fragment_to_bb_id
        )

        summary: dict[str, object] = {
            "name": self.name,
            "num_nodes": len(self.node_fragments),
            "num_linkers": len(self.linker_fragments),
            "num_connecting_bonds": len(self.connecting_bonds),
            "node_indices": list(self.node_indices),
            "linker_indices": list(self.linker_indices),
            "points_of_extension": list(self.points_of_extension),
            "connecting_bonds": [list(pair) for pair in self.connecting_bonds],
            "topology": topology_summary,
            "reconstruction_map": str(reconstruction_path.relative_to(output_dir)),
            "groups": group_records,
            "nodes": [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.node_fragments
            ],
            "linkers": [
                fragment.as_dict(fragment_instance_path(fragment))
                for fragment in self.linker_fragments
            ],
        }
        with (output_dir / "decomposition_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return summary

    def _write_group_representatives(
        self, output_dir: Path
    ) -> tuple[list[dict[str, object]], dict[tuple[str, int], str]]:
        group_records: list[dict[str, object]] = []
        fragment_to_bb_id: dict[tuple[str, int], str] = {}
        for kind, fragments in (
            ("node", self.node_fragments),
            ("linker", self.linker_fragments),
        ):
            grouped: dict[str, list[Fragment]] = defaultdict(list)
            for fragment in fragments:
                grouped[fragment.fingerprint].append(fragment)

            for group_index, (_, group) in enumerate(sorted(grouped.items()), start=1):
                representative = group[0]
                bb_id = f"{kind}_group_{group_index:02d}"
                group_path = output_dir / "bbs" / f"{kind}_group_{group_index:02d}.xyz"
                representative.write_xyz(group_path)
                for fragment in group:
                    fragment_to_bb_id[(fragment.kind, fragment.index)] = bb_id
                group_records.append(
                    {
                        "bb_id": bb_id,
                        "kind": kind,
                        "group": group_index,
                        "count": len(group),
                        "cn": representative.num_x,
                        "formula": representative.formula,
                        "num_atoms": representative.num_atoms,
                        "num_x": representative.num_x,
                        "x_sites": [
                            {"site_id": site_id, "x_index": site.x_index}
                            for site_id, site in enumerate(representative.x_sites)
                        ],
                        "fingerprint": representative.fingerprint,
                        "fragment_indices": [fragment.index for fragment in group],
                        "path": str(group_path.relative_to(output_dir)),
                    }
                )
        return group_records, fragment_to_bb_id

    def write_reconstruction_map(
        self,
        path: Path,
        group_records: list[dict[str, object]],
        fragment_to_bb_id: dict[tuple[str, int], str],
    ) -> dict[str, object]:
        """Write group-first metadata for exact reconstruction experiments."""

        path.parent.mkdir(parents=True, exist_ok=True)
        all_fragments = self.fragments
        slot_by_key = {
            (fragment.kind, fragment.index): slot_id
            for slot_id, fragment in enumerate(all_fragments)
        }
        slot_by_source_atom: dict[int, int] = {}
        for fragment in all_fragments:
            slot_id = slot_by_key[(fragment.kind, fragment.index)]
            for atom_index in fragment.source_indices:
                slot_by_source_atom[atom_index] = slot_id

        slots = []
        for slot_id, fragment in enumerate(all_fragments):
            key = (fragment.kind, fragment.index)
            slots.append(
                {
                    "slot_id": slot_id,
                    "kind": fragment.kind,
                    "cn": fragment.num_x,
                    "bb_id": fragment_to_bb_id[key],
                    "debug_instance": fragment_instance_path(fragment),
                    "source_indices": list(fragment.source_indices),
                    "formula": fragment.formula,
                }
            )

        x_site_by_slot_and_cut: dict[tuple[int, tuple[int, int]], int] = {}
        all_x_site_keys: set[tuple[int, int]] = set()
        for slot_id, fragment in enumerate(all_fragments):
            for site in fragment.x_sites:
                key = (slot_id, site.x_index)
                all_x_site_keys.add(key)
                x_site_by_slot_and_cut[(slot_id, site.source_cut_bond)] = site.x_index

        edges = []
        used_x_site_keys: set[tuple[int, int]] = set()
        all_edges_mapped = True
        slot_pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        for edge_id, cut_bond in enumerate(self.connecting_bonds):
            source_cut_bond = tuple(sorted(cut_bond))
            endpoint_slots = [
                slot_by_source_atom[source_cut_bond[0]],
                slot_by_source_atom[source_cut_bond[1]],
            ]
            endpoint_x_sites = []
            for slot_id in endpoint_slots:
                x_index = x_site_by_slot_and_cut.get((slot_id, source_cut_bond))
                if x_index is None:
                    all_edges_mapped = False
                    continue
                used_x_site_keys.add((slot_id, x_index))
                endpoint_x_sites.append({"slot_id": slot_id, "x_index": x_index})

            if len(endpoint_x_sites) != 2:
                all_edges_mapped = False

            slot_pair_counts[tuple(sorted(endpoint_slots))] += 1
            edges.append(
                {
                    "edge_id": edge_id,
                    "endpoint_slots": endpoint_slots,
                    "endpoint_x_sites": endpoint_x_sites,
                    "source_cut_bond": list(source_cut_bond),
                }
            )

        duplicate_edge_warnings = [
            {
                "endpoint_slots": list(slot_pair),
                "edge_count": count,
            }
            for slot_pair, count in sorted(slot_pair_counts.items())
            if count > 1
        ]

        data: dict[str, object] = {
            "schema_version": 1,
            "topology": {
                "kind": "original_cell_slots",
                "path": "topology.cgd",
                "stock_pormake_reader_safe": False,
                "cell": list(self.cell_parameters),
            },
            "bb_groups": [
                {
                    "bb_id": group["bb_id"],
                    "kind": group["kind"],
                    "path": group["path"],
                    "cn": group["cn"],
                    "formula": group["formula"],
                    "x_sites": group["x_sites"],
                }
                for group in group_records
            ],
            "slots": slots,
            "edges": edges,
            "validation": {
                "all_edges_mapped": all_edges_mapped,
                "all_x_sites_used_once": all_x_site_keys == used_x_site_keys,
                "duplicate_edge_warnings": duplicate_edge_warnings,
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return data

    def write_topology_cgd(self, path: Path) -> dict[str, int]:
        """Write a MOFid-style P1 CGD topology file."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("CRYSTAL\n")
            handle.write(f"  NAME {self.name}\n")
            handle.write("  GROUP P1\n")
            cell = " ".join(format_float(value) for value in self.cell_parameters)
            handle.write(f"  CELL {cell}\n")
            for vertex_number, vertex in enumerate(self.topology_vertices, start=1):
                frac = " ".join(format_float(value) for value in vertex.frac_position)
                handle.write(f"  NODE {vertex_number} {vertex.coordination} {frac}\n")
            for edge in self.topology_edges:
                begin = " ".join(format_float(value) for value in edge.begin_frac)
                end = " ".join(format_float(value) for value in edge.end_frac)
                handle.write(f"  EDGE  {begin}   {end}\n")
            for edge in self.topology_edges:
                center = " ".join(format_float(value) for value in edge.center_frac)
                handle.write(f"  # EDGE_CENTER  {center}\n")
            handle.write("END\n")
        return {
            "num_vertices": len(self.topology_vertices),
            "num_edges": len(self.topology_edges),
        }
