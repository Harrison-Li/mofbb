"""Legacy exact-reconstruction map serialization."""

from __future__ import annotations

from collections import defaultdict

from mofbb.decompose.object import Fragment, TopologyEdge


def fragment_instance_path(fragment: Fragment) -> str:
    """Return the legacy per-instance XYZ path for a fragment."""
    return f"{fragment.kind}s/{fragment.kind}_{fragment.index:03d}.xyz"


class LegacyMap:
    """Build the coordinate-bearing map used by exact reconstruction."""

    def __init__(self, result, bb_ids, top_slots):
        self.result = result
        self.bb_ids = bb_ids
        self.top_slots = top_slots

    def build(self, groups, topology_path=None) -> dict[str, object]:
        topology: dict[str, object] = {
            "kind": "node_edge_slots",
            "cell": list(self.result.cell_parameters),
        }
        if topology_path:
            topology["path"] = topology_path
        return {
            "schema_version": 2,
            "topology": topology,
            "bb_groups": groups,
            "slots": self.slots(),
            "edges": self.edges(),
        }

    def slots(self) -> list[dict[str, object]]:
        positions = {
            (vertex.kind, vertex.index): vertex.frac_position
            for vertex in self.result.topology_vertices
        }
        slots = []
        for slot, fragment in enumerate(self.result.fragments):
            key = fragment.kind, fragment.index
            pos = fragment.real_atom_centroid_frac or positions[key]
            kind, index = self.top_slots[key]
            slots.append(
                {
                    "slot_id": slot,
                    "kind": fragment.kind,
                    "cn": fragment.num_x,
                    "logical_kind": fragment.logical_kind or fragment.kind,
                    "logical_ports": [
                        port.as_dict() for port in fragment.logical_ports
                    ],
                    "bb_id": self.bb_ids[key],
                    "exact_instance_path": fragment_instance_path(fragment),
                    "source_indices": list(fragment.source_indices),
                    "source_image_offsets": [
                        [atom, list(offset)]
                        for atom, offset in fragment.source_image_offsets
                    ],
                    "exact_frac_position": list(pos),
                    "topology_slot": {"kind": kind, "index": index},
                }
            )
        return slots

    def edges(self) -> list[dict[str, object]]:
        fragments = self.result.fragments
        slot_by_key = {
            (fragment.kind, fragment.index): slot
            for slot, fragment in enumerate(fragments)
        }
        slot_by_atom = {
            atom: slot
            for slot, fragment in enumerate(fragments)
            for atom in fragment.source_indices
        }
        x_sites = {
            (
                slot,
                site.source_cut_bond,
                site.source_cut_shift,
                site.internal_source_index,
            ): site
            for slot, fragment in enumerate(fragments)
            for site in fragment.x_sites
        }
        by_record: dict[
            tuple[tuple[int, int], tuple[int, int, int]],
            list[TopologyEdge],
        ] = defaultdict(list)
        for edge in self.result.topology_edges:
            by_record[(edge.source_cut_bond, edge.source_cut_shift)].append(edge)

        edges = []
        for edge_id, bond in enumerate(self.result.connection_records):
            end_slots = [slot_by_atom[bond.i], slot_by_atom[bond.j]]
            end_sites = [
                x_sites[(slot, bond.pair, bond.shift, atom)]
                for atom, slot in zip(bond.pair, end_slots)
            ]
            top_edge = by_record[(bond.pair, bond.shift)].pop(0)
            top_ends = (
                slot_by_key[(top_edge.begin_kind, top_edge.begin_index)],
                slot_by_key[(top_edge.end_kind, top_edge.end_index)],
            )
            shift = top_edge.slot_translation
            if top_ends == tuple(reversed(end_slots)):
                shift = tuple(-value for value in shift)
            edges.append(
                {
                    "edge_id": edge_id,
                    "endpoint_slots": end_slots,
                    "endpoint_x_sites": [
                        {
                            "slot_id": slot,
                            "x_index": site.x_index,
                            "internal_source_index": site.internal_source_index,
                            "external_source_index": site.external_source_index,
                        }
                        for slot, site in zip(end_slots, end_sites)
                    ],
                    "source_cut_bond": list(bond.pair),
                    "source_cut_shift": list(bond.shift),
                    "slot_translation": list(shift),
                }
            )
        return edges
