"""Output writers for MOF decomposition results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
import json
from pathlib import Path

from ase.geometry import cellpar_to_cell
import numpy as np
import spglib

from .geometry import format_float
from .object import Fragment, TopologyEdge, TopologyVertex, XSite
from .pbc import PeriodicBond, PeriodicOpening
from .solvent import SolventMode, SolventRemovalResult
from .topology import validate_source_atom_ownership


def _canonical_edge(
    begin: int,
    end: int,
    shift: np.ndarray,
) -> tuple[int, int, tuple[int, int, int]]:
    direct = (begin, end, tuple(int(value) for value in shift))
    reverse = (end, begin, tuple(int(-value) for value in shift))
    return min(direct, reverse)


def _topology_symmetry(
    cell_parameters: tuple[float, float, float, float, float, float],
    vertices: list[TopologyVertex],
    edges: list[TopologyEdge],
    vertex_types: list[int],
    edge_colors: list[int] | None = None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Detect operations of the colored periodic topology graph."""

    identity = (
        "P1",
        np.eye(3, dtype=int)[None, ...],
        np.zeros((1, 3)),
        np.arange(1, len(edges) + 1),
    )
    if not vertices:
        return identity

    positions = np.asarray([vertex.frac_position for vertex in vertices], float)
    types = np.asarray(vertex_types, int)
    if edge_colors is None:
        edge_colors = [0] * len(edges)
    if len(edge_colors) != len(edges):
        raise ValueError("Topology edge colors must align one-for-one with edges.")
    index_by_key = {
        (vertex.kind, vertex.index): index
        for index, vertex in enumerate(vertices)
    }
    periodic_edges = []
    for edge in edges:
        begin = index_by_key[(edge.begin_kind, edge.begin_index)]
        end = index_by_key[(edge.end_kind, edge.end_index)]
        begin_image = np.rint(np.asarray(edge.begin_frac) - positions[begin])
        end_image = np.rint(np.asarray(edge.end_frac) - positions[end])
        periodic_edges.append(_canonical_edge(begin, end, end_image - begin_image))

    pair_types = {
        pair: int(types.max()) + offset
        for offset, pair in enumerate(
            sorted(
                {
                    (tuple(sorted((types[i], types[j]))), edge_colors[index])
                    for index, (i, j, _) in enumerate(periodic_edges)
                }
            ),
            start=1,
        )
    }
    decorated_positions = list(np.mod(positions, 1.0))
    decorated_types = list(types)
    seen = {
        (int(site_type), *np.round(position, 8))
        for position, site_type in zip(decorated_positions, decorated_types)
    }
    for edge_index, (begin, end, shift) in enumerate(periodic_edges):
        delta = positions[end] + np.asarray(shift) - positions[begin]
        marker_type = pair_types[
            (tuple(sorted((types[begin], types[end]))), edge_colors[edge_index])
        ]
        for fraction in (1.0 / 3.0, 2.0 / 3.0):
            marker = np.mod(positions[begin] + fraction * delta, 1.0)
            key = (int(marker_type), *np.round(marker, 8))
            if key not in seen:
                seen.add(key)
                decorated_positions.append(marker)
                decorated_types.append(marker_type)

    dataset = spglib.get_symmetry_dataset(
        (
            np.asarray(cellpar_to_cell(cell_parameters)),
            np.asarray(decorated_positions),
            np.asarray(decorated_types),
        ),
        symprec=0.05,
        angle_tolerance=5.0,
    )
    if dataset is None:
        return identity

    rotations = np.asarray(dataset.rotations, int)
    translations = np.asarray(dataset.translations, float)
    same_type = types[:, None] == types[None, :]
    node_maps, node_images = [], []
    for rotation, translation in zip(rotations, translations):
        transformed = positions @ rotation.T + translation
        differences = transformed[:, None] - positions[None]
        images = np.rint(differences)
        matches = same_type & np.all(abs(differences - images) <= 2.0e-4, axis=2)
        if np.any(matches.sum(axis=1) != 1):
            return identity
        mapping = np.argmax(matches, axis=1)
        node_maps.append(mapping)
        node_images.append(images[np.arange(len(vertices)), mapping].astype(int))

    indices_by_edge: dict[
        tuple[tuple[int, int, tuple[int, int, int]], int], list[int]
    ] = defaultdict(list)
    for index, edge in enumerate(periodic_edges):
        indices_by_edge[(edge, edge_colors[index])].append(index)
    edge_rank = {
        index: rank
        for indices in indices_by_edge.values()
        for rank, index in enumerate(indices)
    }
    edge_maps = []
    for rotation, mapping, images in zip(rotations, node_maps, node_images):
        mapped = []
        for index, (begin, end, shift) in enumerate(periodic_edges):
            target = _canonical_edge(
                int(mapping[begin]),
                int(mapping[end]),
                images[end] + rotation @ np.asarray(shift) - images[begin],
            )
            candidates = indices_by_edge.get((target, edge_colors[index]), ())
            if edge_rank[index] >= len(candidates):
                return identity
            mapped.append(candidates[edge_rank[index]])
        if len(set(mapped)) != len(edges):
            return identity
        edge_maps.append(mapped)

    edge_types = np.zeros(len(edges), dtype=int)
    for edge_index in range(len(edges)):
        if edge_types[edge_index]:
            continue
        orbit = np.unique(np.asarray(edge_maps)[:, edge_index])
        edge_types[orbit] = int(edge_types.max()) + 1
    return str(dataset.international), rotations, translations, edge_types


def _write_topology_cgd(
    path: Path,
    *,
    name: str,
    cell_parameters: tuple[float, float, float, float, float, float],
    vertices: list[TopologyVertex],
    edges: list[TopologyEdge],
    fragment_to_bb_id: dict[tuple[str, int], str] | None,
    edge_bb_ids: dict[int, str] | None = None,
) -> dict[str, object]:
    """Write an expanded CGD with graph-derived symmetry and BB types."""

    bb_types: dict[str, int] = {}
    vertex_types = []
    for vertex in vertices:
        bb_id = (
            fragment_to_bb_id[(vertex.kind, vertex.index)]
            if fragment_to_bb_id is not None
            else f"{vertex.kind}:{vertex.index}"
        )
        vertex_types.append(bb_types.setdefault(bb_id, len(bb_types) + 1))
    edge_color_ids: dict[str | None, int] = {}
    edge_colors = [
        edge_color_ids.setdefault(
            None if edge_bb_ids is None else edge_bb_ids.get(index),
            len(edge_color_ids) + 1,
        )
        for index in range(len(edges))
    ]
    symbol, rotations, translations, edge_types = _topology_symmetry(
        cell_parameters, vertices, edges, vertex_types, edge_colors
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("CRYSTAL\n")
        handle.write(f"  NAME {name}\n")
        handle.write(f"  GROUP {symbol}\n")
        cell = " ".join(format_float(value) for value in cell_parameters)
        handle.write(f"  CELL {cell}\n")
        handle.write("  # EXPANDED\n")
        for rotation, translation in zip(rotations, translations):
            values = [
                *(str(int(value)) for value in rotation.ravel()),
                *(format_float(value) for value in translation),
            ]
            handle.write(f"  # SYMOP {' '.join(values)}\n")
        for vertex_type, vertex in zip(vertex_types, vertices):
            frac = " ".join(format_float(value) for value in vertex.frac_position)
            handle.write(f"  NODE {vertex_type} {vertex.coordination} {frac}\n")
        for edge_number, (edge_type, edge) in enumerate(
            zip(edge_types, edges), start=1
        ):
            handle.write(f"  # EDGE_TYPE {edge_number} {edge_type}\n")
            begin = " ".join(format_float(value) for value in edge.begin_frac)
            end = " ".join(format_float(value) for value in edge.end_frac)
            handle.write(f"  EDGE  {begin}   {end}\n")
        for edge in edges:
            center = " ".join(format_float(value) for value in edge.center_frac)
            handle.write(f"  # EDGE_CENTER  {center}\n")
        handle.write("END\n")
    return {
        "num_vertices": len(vertices),
        "num_edges": len(edges),
        "num_node_types": len(bb_types),
        "spacegroup": symbol,
        "num_symmetry_operations": len(rotations),
    }


def fragment_instance_path(fragment: Fragment) -> str:
    """Return the stable per-instance debug path for a fragment."""

    if fragment.kind == "node":
        return f"nodes/node_{fragment.index:03d}.xyz"
    if fragment.kind == "linker":
        return f"linkers/linker_{fragment.index:03d}.xyz"
    return f"{fragment.kind}s/{fragment.kind}_{fragment.index:03d}.xyz"


def logical_port_partition(fragment: Fragment) -> tuple[tuple[int, ...], ...]:
    """Return the instance-local groups of physical X indices."""

    return tuple(port.x_indices for port in fragment.logical_ports)


def logical_port_size_signature(fragment: Fragment) -> tuple[int, ...]:
    """Return the representative-independent logical-port size signature."""

    return tuple(sorted(len(indices) for indices in logical_port_partition(fragment)))


def fragment_group_key(fragment: Fragment) -> tuple[object, ...]:
    """Group fragments with the same chemistry and logical-port cardinalities."""

    return (
        fragment.fingerprint,
        fragment.logical_kind or fragment.kind,
        logical_port_size_signature(fragment),
    )


def fragment_group_metadata(fragment: Fragment) -> dict[str, object]:
    """Return representative-independent physical/logical group metadata."""

    port_sizes = logical_port_size_signature(fragment)
    logical_cn = len(port_sizes) if port_sizes else fragment.num_x
    return {
        "physical_cn": fragment.num_x,
        "logical_cn": logical_cn,
        "logical_kind": fragment.logical_kind or fragment.kind,
        "logical_port_sizes": list(port_sizes),
        "logical_ports": [
            {"logical_port_id": port_id, "size": size}
            for port_id, size in enumerate(port_sizes)
        ],
    }


def _legacy_connection_records(
    connecting_bonds: tuple[tuple[int, int], ...],
) -> tuple[PeriodicBond, ...]:
    return tuple(
        PeriodicBond(*tuple(sorted(pair)), (0, 0, 0))
        for pair in connecting_bonds
    )


def _negate_shift(shift: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(-value for value in shift)


def _normalized_undirected_translation(
    endpoint_slots: list[int],
    translation: tuple[int, int, int],
) -> tuple[int, int, int]:
    if endpoint_slots[0] < endpoint_slots[1]:
        return translation
    if endpoint_slots[0] > endpoint_slots[1]:
        return _negate_shift(translation)
    return min(translation, _negate_shift(translation))


def write_reconstruction_data(
    path: Path,
    *,
    topology_kind: str,
    cell_parameters: tuple[float, float, float, float, float, float],
    fragments: list[Fragment],
    expected_source_atoms: Iterable[int],
    connecting_bonds: tuple[tuple[int, int], ...],
    connection_records: tuple[PeriodicBond, ...],
    periodic_openings: tuple[PeriodicOpening, ...],
    periodic_validation: dict[str, bool],
    topology_edges: list[TopologyEdge],
    group_records: list[dict[str, object]],
    fragment_to_bb_id: dict[tuple[str, int], str],
    slot_positions: dict[tuple[str, int], tuple[float, float, float]] | None = None,
    topology_slots: dict[tuple[str, int], tuple[str, int]] | None = None,
) -> dict[str, object]:
    """Write the shared schema-v2, shift-aware reconstruction map."""

    path.parent.mkdir(parents=True, exist_ok=True)
    validate_source_atom_ownership(expected_source_atoms, fragments)

    fragment_keys = [(fragment.kind, fragment.index) for fragment in fragments]
    if len(set(fragment_keys)) != len(fragment_keys):
        raise ValueError("Reconstruction fragment keys must be unique.")
    slot_by_key = {key: slot_id for slot_id, key in enumerate(fragment_keys)}
    slot_by_source_atom = {
        atom_index: slot_by_key[(fragment.kind, fragment.index)]
        for fragment in fragments
        for atom_index in fragment.source_indices
    }

    slots = []
    for slot_id, fragment in enumerate(fragments):
        key = (fragment.kind, fragment.index)
        partition = logical_port_partition(fragment)
        instance_path = fragment_instance_path(fragment)
        slot = {
            "slot_id": slot_id,
            "kind": fragment.kind,
            "cn": fragment.num_x,
            "physical_cn": fragment.num_x,
            "logical_cn": len(partition) if partition else fragment.num_x,
            "logical_kind": fragment.logical_kind or fragment.kind,
            "logical_ports": [port.as_dict() for port in fragment.logical_ports],
            "bb_id": fragment_to_bb_id[key],
            "debug_instance": instance_path,
            "exact_instance_path": instance_path,
            "source_indices": list(fragment.source_indices),
            "source_image_offsets": [
                [atom_index, list(offset)]
                for atom_index, offset in fragment.source_image_offsets
            ],
            "formula": fragment.formula,
        }
        if slot_positions is not None:
            slot["exact_frac_position"] = list(slot_positions[key])
        if topology_slots is not None:
            slot_kind, slot_index = topology_slots[key]
            slot["topology_slot"] = {"kind": slot_kind, "index": slot_index}
        slots.append(slot)

    records = connection_records or _legacy_connection_records(connecting_bonds)
    if connection_records and tuple(record.pair for record in records) != connecting_bonds:
        raise ValueError(
            "Connection records must align one-for-one with connecting_bonds."
        )

    x_site_lookup: dict[
        tuple[int, tuple[int, int], tuple[int, int, int], int],
        tuple[Fragment, XSite],
    ] = {}
    known_x_sites: Counter[tuple[int, int]] = Counter()
    all_x_site_metadata_valid = True
    for slot_id, fragment in enumerate(fragments):
        source_count = len(fragment.source_indices)
        expected_cap_indices = set(range(source_count, len(fragment.atoms)))
        metadata_cap_indices = {site.x_index for site in fragment.x_sites}
        if (
            metadata_cap_indices != expected_cap_indices
            or len(metadata_cap_indices) != len(fragment.x_sites)
        ):
            all_x_site_metadata_valid = False
        for site in fragment.x_sites:
            if not (
                0 <= site.anchor_index < source_count
                and site.internal_source_index
                == fragment.source_indices[site.anchor_index]
                and site.source_cut_bond
                == tuple(
                    sorted(
                        (site.internal_source_index, site.external_source_index)
                    )
                )
            ):
                all_x_site_metadata_valid = False
            site_key = (slot_id, site.x_index)
            known_x_sites[site_key] += 1
            lookup_key = (
                slot_id,
                site.source_cut_bond,
                site.source_cut_shift,
                site.internal_source_index,
            )
            if lookup_key in x_site_lookup:
                raise ValueError(f"Duplicate endpoint-aware X-site key {lookup_key}.")
            x_site_lookup[lookup_key] = (fragment, site)

    topology_by_record: dict[
        tuple[tuple[int, int], tuple[int, int, int]], list[TopologyEdge]
    ] = defaultdict(list)
    legacy_topology_edges: list[TopologyEdge] = []
    for topology_edge in topology_edges:
        if topology_edge.source_cut_bond is None:
            legacy_topology_edges.append(topology_edge)
            continue
        topology_by_record[
            (topology_edge.source_cut_bond, topology_edge.source_cut_shift)
        ].append(topology_edge)
    legacy_topology_cursor = 0

    edges: list[dict[str, object]] = []
    used_x_sites: Counter[tuple[int, int]] = Counter()
    used_topology_edges: Counter[int] = Counter()
    all_edges_mapped = True
    all_self_edge_shifts_inverse = True
    all_slot_translations_exact = True
    physical_identity_counts: Counter[
        tuple[tuple[int, int], tuple[int, int, int]]
    ] = Counter()
    parallel_groups: dict[
        tuple[tuple[int, int], tuple[int, int, int]], list[dict[str, object]]
    ] = defaultdict(list)

    for edge_id, record in enumerate(records):
        endpoint_slots = [
            slot_by_source_atom[record.i],
            slot_by_source_atom[record.j],
        ]
        endpoint_x_sites = []
        endpoint_site_keys: list[tuple[int, int]] = []
        endpoint_sites = []
        for atom_index, slot_id in zip(record.pair, endpoint_slots):
            lookup_key = (slot_id, record.pair, record.shift, atom_index)
            found = x_site_lookup.get(lookup_key)
            if found is None:
                all_edges_mapped = False
                continue
            _, site = found
            other_atom = record.j if atom_index == record.i else record.i
            if site.external_image_shift != record.oriented_shift(
                atom_index, other_atom
            ):
                all_x_site_metadata_valid = False
            site_key = (slot_id, site.x_index)
            used_x_sites[site_key] += 1
            endpoint_site_keys.append(site_key)
            endpoint_sites.append(site)
            endpoint_x_sites.append(
                {
                    "slot_id": slot_id,
                    "x_index": site.x_index,
                    "internal_source_index": site.internal_source_index,
                    "external_source_index": site.external_source_index,
                    "external_image_shift": list(site.external_image_shift),
                }
            )
        if len(endpoint_x_sites) != 2 or len(set(endpoint_site_keys)) != 2:
            all_edges_mapped = False

        topology_edge = None
        candidates = topology_by_record.get((record.pair, record.shift), [])
        if candidates:
            topology_edge = candidates.pop(0)
        elif legacy_topology_cursor < len(legacy_topology_edges):
            topology_edge = legacy_topology_edges[legacy_topology_cursor]
            legacy_topology_cursor += 1
        if topology_edge is None:
            all_edges_mapped = False
            canonical_translation = (0, 0, 0)
        else:
            used_topology_edges[id(topology_edge)] += 1
            begin_slot = slot_by_key[
                (topology_edge.begin_kind, topology_edge.begin_index)
            ]
            end_slot = slot_by_key[(topology_edge.end_kind, topology_edge.end_index)]
            if endpoint_slots[0] == endpoint_slots[1]:
                canonical_translation = topology_edge.slot_translation
            elif (begin_slot, end_slot) == tuple(endpoint_slots):
                canonical_translation = topology_edge.slot_translation
            elif (begin_slot, end_slot) == tuple(reversed(endpoint_slots)):
                canonical_translation = _negate_shift(topology_edge.slot_translation)
            else:
                all_edges_mapped = False
                canonical_translation = topology_edge.slot_translation

        endpoint_offsets = []
        for atom_index, slot_id in zip(record.pair, endpoint_slots):
            fragment = fragments[slot_id]
            offsets = dict(fragment.source_image_offsets)
            endpoint_offsets.append(offsets.get(atom_index, (0, 0, 0)))
        expected_translation = tuple(
            record.shift[axis]
            + endpoint_offsets[0][axis]
            - endpoint_offsets[1][axis]
            for axis in range(3)
        )
        if canonical_translation != expected_translation:
            all_slot_translations_exact = False
            all_edges_mapped = False

        endpoint_translations = [
            canonical_translation,
            _negate_shift(canonical_translation),
        ]
        if endpoint_slots[0] == endpoint_slots[1]:
            if len(endpoint_sites) != 2:
                all_self_edge_shifts_inverse = False
            else:
                left = endpoint_sites[0].external_image_shift
                right = endpoint_sites[1].external_image_shift
                if tuple(left) != _negate_shift(tuple(right)):
                    all_self_edge_shifts_inverse = False

        edge_payload: dict[str, object] = {
            "edge_id": edge_id,
            "endpoint_slots": endpoint_slots,
            "endpoint_x_sites": endpoint_x_sites,
            "source_cut_bond": list(record.pair),
            "source_cut_shift": list(record.shift),
            "slot_translation": list(canonical_translation),
            "expected_slot_translation": list(expected_translation),
            "endpoint_slot_translations": [
                list(translation) for translation in endpoint_translations
            ],
        }
        edges.append(edge_payload)
        physical_identity = (record.pair, record.shift)
        physical_identity_counts[physical_identity] += 1
        normalized_translation = _normalized_undirected_translation(
            endpoint_slots, canonical_translation
        )
        parallel_groups[(tuple(sorted(endpoint_slots)), normalized_translation)].append(
            edge_payload
        )

    duplicate_edge_warnings = [
        {
            "source_cut_bond": list(pair),
            "source_cut_shift": list(shift),
            "edge_count": count,
        }
        for (pair, shift), count in sorted(physical_identity_counts.items())
        if count > 1
    ]
    physical_parallel_edges = [
        {
            "endpoint_slots": list(slot_pair),
            "slot_translation": list(translation),
            "edge_ids": [int(edge["edge_id"]) for edge in group],
            "source_cut_bonds": [edge["source_cut_bond"] for edge in group],
        }
        for (slot_pair, translation), group in sorted(parallel_groups.items())
        if len({tuple(edge["source_cut_bond"]) for edge in group}) > 1
    ]
    translations_by_slot_pair: dict[
        tuple[int, int], set[tuple[int, int, int]]
    ] = defaultdict(set)
    for slot_pair, translation in parallel_groups:
        translations_by_slot_pair[slot_pair].add(translation)
    translated_parallel_edges = [
        {
            "endpoint_slots": list(slot_pair),
            "slot_translations": [list(item) for item in sorted(translations)],
        }
        for slot_pair, translations in sorted(translations_by_slot_pair.items())
        if len(translations) > 1
    ]

    all_x_sites_used_once = (
        all_x_site_metadata_valid
        and known_x_sites == used_x_sites
        and all(count == 1 for count in known_x_sites.values())
        and all(count == 1 for count in used_x_sites.values())
    )
    all_topology_edges_mapped = (
        len(topology_edges) == len(records)
        and len(used_topology_edges) == len(topology_edges)
        and all(count == 1 for count in used_topology_edges.values())
    )
    all_edges_mapped = all_edges_mapped and all_topology_edges_mapped
    legacy_periodic_default = not periodic_openings
    validation = {
        "all_edges_mapped": all_edges_mapped,
        "all_x_site_metadata_valid": all_x_site_metadata_valid,
        "all_x_sites_used_once": all_x_sites_used_once,
        "all_self_edge_shifts_inverse": all_self_edge_shifts_inverse,
        "all_slot_translations_exact": all_slot_translations_exact,
        "all_source_atoms_owned_once": True,
        "all_fragments_zero_holonomy": periodic_validation.get(
            "all_fragments_zero_holonomy", legacy_periodic_default
        ),
        "all_retained_bonds_undistorted": periodic_validation.get(
            "all_retained_bonds_undistorted", legacy_periodic_default
        ),
        "all_topology_edges_mapped": all_topology_edges_mapped,
        "duplicate_edge_warnings": duplicate_edge_warnings,
        "physical_parallel_edges": physical_parallel_edges,
        "translated_parallel_edges": translated_parallel_edges,
    }
    data: dict[str, object] = {
        "schema_version": 2,
        "topology": {
            "kind": topology_kind,
            "path": "topology.cgd",
            "stock_pormake_reader_safe": False,
            "cell": list(cell_parameters),
        },
        "geometry_contract": {
            "exact_slot_geometry_field": "exact_instance_path",
            "group_representatives_are_exact_slot_geometry": False,
            "representative_atom_permutations_emitted": False,
        },
        "bb_groups": [
            {
                key: group[key]
                for key in (
                    "bb_id",
                    "kind",
                    "path",
                    "cn",
                    "physical_cn",
                    "logical_cn",
                    "logical_kind",
                    "logical_port_sizes",
                    "logical_ports",
                    "formula",
                    "x_sites",
                )
            }
            for group in group_records
        ],
        "slots": slots,
        "edges": edges,
        "validation": validation,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return data


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
    solvent_mode: SolventMode = "none"
    solvent: SolventRemovalResult | None = None
    selected_to_original: tuple[int, ...] | None = None
    connection_records: tuple[PeriodicBond, ...] = ()
    periodic_openings: tuple[PeriodicOpening, ...] = ()
    periodic_validation: dict[str, bool] = field(default_factory=dict)

    @property
    def fragments(self) -> list[Fragment]:
        """Return all node and linker fragments in output slot order."""

        return [*self.node_fragments, *self.linker_fragments]

    def write(self, output_dir: Path) -> dict[str, object]:
        """Write all output files and return a JSON-serializable summary."""

        validate_source_atom_ownership(
            (*self.node_indices, *self.linker_indices), self.fragments
        )
        if self.solvent_mode == "none":
            if self.solvent is not None or self.selected_to_original is not None:
                raise ValueError(
                    "Inconsistent solvent metadata: mode 'none' requires both "
                    "the solvent result and selected-to-original mapping to be absent."
                )
        elif self.solvent_mode in {"free", "all"}:
            if self.solvent is None or self.selected_to_original is None:
                raise ValueError(
                    f"Inconsistent solvent metadata: active mode {self.solvent_mode!r} "
                    "requires both a solvent result and selected-to-original mapping."
                )
            self.solvent.validate_write_request(
                self.solvent_mode,
                self.selected_to_original,
            )
        else:
            raise ValueError(f"Unsupported solvent mode: {self.solvent_mode!r}.")

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
        topology_summary = self.write_topology_cgd(
            topology_path, fragment_to_bb_id
        )
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
        if self.connection_records:
            summary["connecting_bond_records"] = [
                record.as_dict() for record in self.connection_records
            ]
            summary["periodic_components"] = [
                opening.as_dict() for opening in self.periodic_openings
            ]
            summary["periodic_validation"] = dict(self.periodic_validation)
        if self.solvent is not None and self.selected_to_original is not None:
            summary["solvent"] = self.solvent.write(
                output_dir,
                self.solvent_mode,
                self.selected_to_original,
            )
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
            grouped: dict[tuple[object, ...], list[Fragment]] = defaultdict(list)
            for fragment in fragments:
                grouped[fragment_group_key(fragment)].append(fragment)

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
                        **fragment_group_metadata(representative),
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

        return write_reconstruction_data(
            path,
            topology_kind="original_cell_slots",
            cell_parameters=self.cell_parameters,
            fragments=self.fragments,
            expected_source_atoms=(*self.node_indices, *self.linker_indices),
            connecting_bonds=self.connecting_bonds,
            connection_records=self.connection_records,
            periodic_openings=self.periodic_openings,
            periodic_validation=self.periodic_validation,
            topology_edges=self.topology_edges,
            group_records=group_records,
            fragment_to_bb_id=fragment_to_bb_id,
        )

    def write_topology_cgd(
        self,
        path: Path,
        fragment_to_bb_id: dict[tuple[str, int], str] | None = None,
    ) -> dict[str, object]:
        """Write the colored topology with its validated graph symmetry."""

        if fragment_to_bb_id is None:
            fragment_to_bb_id = {
                (fragment.kind, fragment.index): repr(
                    (fragment.kind, fragment_group_key(fragment))
                )
                for fragment in self.fragments
            }
        return _write_topology_cgd(
            path,
            name=self.name,
            cell_parameters=self.cell_parameters,
            vertices=self.topology_vertices,
            edges=self.topology_edges,
            fragment_to_bb_id=fragment_to_bb_id,
        )
