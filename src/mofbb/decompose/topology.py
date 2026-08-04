"""Collapsed topology construction for MOF decomposition fragments."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import ase
import numpy as np

from .bonding import BondNetwork
from .object import Fragment, TopologyEdge, TopologyVertex
from .pbc import (
    ComponentPeriodicity,
    PeriodicBond,
    analyze_component,
    connection_shift,
    periodic_bonds,
)

ConnectionInput = tuple[int, int] | PeriodicBond
FragmentKey = tuple[str, int]


def validate_source_atom_ownership(
    expected_source_atoms: Iterable[int],
    fragments: list[Fragment],
) -> None:
    """Require every expected source atom to occur in exactly one fragment."""

    expected = set(expected_source_atoms)
    counts = Counter(
        atom_index for fragment in fragments for atom_index in fragment.source_indices
    )
    missing = sorted(expected - set(counts))
    duplicates = {
        atom_index: count for atom_index, count in sorted(counts.items()) if count > 1
    }
    unexpected = sorted(set(counts) - expected)
    if not (missing or duplicates or unexpected):
        return

    details = []
    if missing:
        details.append(f"missing source atoms {missing}")
    if duplicates:
        details.append(f"duplicate source atoms {duplicates}")
    if unexpected:
        details.append(f"unexpected source atoms {unexpected}")
    raise ValueError(
        "Source atom ownership must be exactly once; " + "; ".join(details) + "."
    )


class TopologyBuilder:
    """Build CGD vertices and edges from exact periodic fragment frames."""

    def __init__(
        self,
        atoms: ase.Atoms,
        bonds: BondNetwork,
        source_records: tuple[PeriodicBond, ...] | None = None,
    ) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.source_records = (
            periodic_bonds(atoms, bonds) if source_records is None else source_records
        )
        self.source_by_pair = {record.pair: record for record in self.source_records}

    def build(
        self,
        node_fragments: list[Fragment],
        linker_fragments: list[Fragment],
        connecting_bonds: tuple[ConnectionInput, ...],
    ) -> tuple[list[TopologyVertex], list[TopologyEdge]]:
        """Return collapsed topology records for a node/linker decomposition."""

        return self.build_fragments(
            [*node_fragments, *linker_fragments],
            connecting_bonds,
            preferred_begin_kind="node",
        )

    def build_fragments(
        self,
        fragments: list[Fragment],
        connections: tuple[ConnectionInput, ...],
        preferred_begin_kind: str | None = None,
    ) -> tuple[list[TopologyVertex], list[TopologyEdge]]:
        """Build topology records for arbitrary fragment role labels."""

        source_records, connection_records = self._normalize_connections(connections)
        supplied_source_atoms = {
            atom_index
            for fragment in fragments
            for atom_index in fragment.source_indices
        }
        validate_source_atom_ownership(supplied_source_atoms, fragments)
        atom_to_vertex = self._atom_to_vertex(fragments)
        frames = {
            (fragment.kind, fragment.index): self._fragment_frame(
                fragment,
                source_records,
                connection_records,
            )
            for fragment in fragments
        }
        real_centroids = self._raw_real_centroids(fragments, frames)
        for fragment in fragments:
            key = (fragment.kind, fragment.index)
            fragment.real_atom_centroid_frac = tuple(
                float(value) for value in np.mod(real_centroids[key], 1.0)
            )
        raw_centroids = self._raw_centroids(
            fragments,
            frames,
            real_centroids,
        )
        coordination: dict[FragmentKey, int] = defaultdict(int)
        edges = self._build_edges(
            connection_records,
            atom_to_vertex,
            frames,
            raw_centroids,
            coordination,
            preferred_begin_kind,
        )
        vertices = self._build_vertices(fragments, raw_centroids, coordination)
        return vertices, edges

    def _normalize_connections(
        self,
        connections: tuple[ConnectionInput, ...],
    ) -> tuple[tuple[PeriodicBond, ...], tuple[PeriodicBond, ...]]:
        normalized = []
        seen = set()
        for connection in connections:
            if isinstance(connection, PeriodicBond):
                supplied = connection
                pair = supplied.pair
            else:
                supplied = None
                pair = tuple(sorted(connection))
            try:
                source = self.source_by_pair[pair]
            except KeyError as exc:
                raise ValueError(
                    f"Topology connection {pair} is not present in the source "
                    "bond network."
                ) from exc
            if supplied is not None and supplied.shift != source.shift:
                raise ValueError(
                    f"Topology connection {pair} supplies periodic shift "
                    f"{supplied.shift}, but the source bond vector requires "
                    f"{source.shift}."
                )
            if source not in seen:
                normalized.append(source)
                seen.add(source)
        return self.source_records, tuple(normalized)

    @staticmethod
    def _atom_to_vertex(fragments: list[Fragment]) -> dict[int, FragmentKey]:
        return {
            atom_index: (fragment.kind, fragment.index)
            for fragment in fragments
            for atom_index in fragment.source_indices
        }

    def _fragment_frame(
        self,
        fragment: Fragment,
        source_records: tuple[PeriodicBond, ...],
        connection_records: tuple[PeriodicBond, ...],
    ) -> ComponentPeriodicity:
        source_indices = set(fragment.source_indices)
        internal_cuts = frozenset(
            record
            for record in connection_records
            if record.i in source_indices and record.j in source_indices
        )
        try:
            derived = analyze_component(
                source_indices,
                source_records,
                excluded=internal_cuts,
            )
        except ValueError as exc:
            raise ValueError(
                f"Fragment {fragment.kind} {fragment.index} does not have a "
                "connected residual source graph after excluding connection "
                f"records {[record.pair for record in sorted(internal_cuts)]}: {exc}"
            ) from exc
        if derived.rank != 0:
            raise ValueError(
                f"Fragment {fragment.kind} {fragment.index} residual source graph "
                f"has rank {derived.rank} with generators {derived.generators}."
            )
        if not fragment.source_image_offsets:
            return derived
        return self._validated_stored_frame(fragment, derived)

    def _validated_stored_frame(
        self,
        fragment: Fragment,
        derived: ComponentPeriodicity,
    ) -> ComponentPeriodicity:
        expected_indices = tuple(sorted(fragment.source_indices))
        stored_indices = tuple(index for index, _ in fragment.source_image_offsets)
        if stored_indices != expected_indices or len(set(stored_indices)) != len(
            stored_indices
        ):
            raise ValueError(
                f"Fragment {fragment.kind} {fragment.index} stored image frame "
                f"indices {stored_indices} do not exactly match source indices "
                f"{expected_indices}."
            )

        stored: dict[int, tuple[int, int, int]] = {}
        periodic_axes = np.asarray(self.atoms.pbc, dtype=bool)
        for atom_index, raw_offset in fragment.source_image_offsets:
            values = np.asarray(raw_offset, dtype=float)
            rounded = np.rint(values)
            if values.shape != (3,) or not np.allclose(
                values, rounded, rtol=0.0, atol=0.0
            ):
                raise ValueError(
                    f"Fragment {fragment.kind} {fragment.index} stored image frame "
                    f"offset for source atom {atom_index} is not an integer cell "
                    f"shift: {raw_offset}."
                )
            offset = tuple(int(value) for value in rounded)
            if np.any(np.asarray(offset)[~periodic_axes]):
                raise ValueError(
                    f"Fragment {fragment.kind} {fragment.index} stored image frame "
                    f"offset {offset} shifts source atom {atom_index} along a "
                    "non-periodic axis."
                )
            stored[atom_index] = offset

        root = expected_indices[0]
        stored_root = np.asarray(stored[root], dtype=int)
        derived_root = np.asarray(derived.offset(root), dtype=int)
        for atom_index in expected_indices:
            stored_relative = np.asarray(stored[atom_index], dtype=int) - stored_root
            derived_relative = (
                np.asarray(derived.offset(atom_index), dtype=int) - derived_root
            )
            if not np.array_equal(stored_relative, derived_relative):
                raise ValueError(
                    f"Fragment {fragment.kind} {fragment.index} stored image frame "
                    "is inconsistent with its exact connected rank-zero residual "
                    f"frame at source atom {atom_index}: "
                    f"{tuple(stored_relative)} != {tuple(derived_relative)}."
                )
        return ComponentPeriodicity(
            offsets=tuple((index, stored[index]) for index in expected_indices),
            generators=(),
            rank=0,
        )

    def _raw_real_centroids(
        self,
        fragments: list[Fragment],
        frames: dict[FragmentKey, ComponentPeriodicity],
    ) -> dict[FragmentKey, np.ndarray]:
        """Return source-frame centroids of each fragment's real atoms."""

        scaled = np.asarray(self.atoms.get_scaled_positions(wrap=False), dtype=float)
        return {
            (fragment.kind, fragment.index): np.mean(
                [
                    scaled[atom_index]
                    + np.asarray(
                        frames[(fragment.kind, fragment.index)].offset(atom_index),
                        dtype=int,
                    )
                    for atom_index in fragment.source_indices
                ],
                axis=0,
            )
            for fragment in fragments
        }

    def _raw_centroids(
        self,
        fragments: list[Fragment],
        frames: dict[FragmentKey, ComponentPeriodicity],
        real_centroids: dict[FragmentKey, np.ndarray],
    ) -> dict[FragmentKey, np.ndarray]:
        """Return source-frame centroids of the fragments' dummy X sites."""

        scaled = np.asarray(self.atoms.get_scaled_positions(wrap=False), dtype=float)
        inverse_cell = np.linalg.inv(np.asarray(self.atoms.cell.array, dtype=float))
        centroids = {}
        for fragment in fragments:
            key = (fragment.kind, fragment.index)
            frame = frames[key]
            if not fragment.x_sites:
                # No dummy centroid exists for a zero-cut accessory fragment;
                # retain the previous real-atom centroid as its fallback anchor.
                centroids[key] = real_centroids[key]
                continue

            # The former topology anchor is retained separately in
            # ``real_atom_centroid_frac`` for reconstruction and provenance.
            x_positions = []
            for site in fragment.x_sites:
                anchor = scaled[site.internal_source_index] + np.asarray(
                    frame.offset(site.internal_source_index), dtype=int
                )
                half_bond = (
                    0.5
                    * self.bonds.vector(
                        site.internal_source_index,
                        site.external_source_index,
                    )
                    @ inverse_cell
                )
                x_positions.append(anchor + half_bond)
            centroids[key] = np.mean(x_positions, axis=0)
        return centroids

    def _build_edges(
        self,
        connection_records: tuple[PeriodicBond, ...],
        atom_to_vertex: dict[int, FragmentKey],
        frames: dict[FragmentKey, ComponentPeriodicity],
        raw_centroids: dict[FragmentKey, np.ndarray],
        coordination: dict[FragmentKey, int],
        preferred_begin_kind: str | None,
    ) -> list[TopologyEdge]:
        edges = []
        for record in connection_records:
            begin_atom, end_atom = record.pair
            try:
                begin_key = atom_to_vertex[begin_atom]
                end_key = atom_to_vertex[end_atom]
            except KeyError as exc:
                raise ValueError(
                    f"Topology connection {record.pair} has an endpoint that is "
                    "not owned by any fragment."
                ) from exc

            if (
                preferred_begin_kind is not None
                and begin_key[0] != preferred_begin_kind
                and end_key[0] == preferred_begin_kind
            ):
                begin_atom, end_atom = end_atom, begin_atom
                begin_key, end_key = end_key, begin_key

            slot_translation = connection_shift(
                record,
                begin_atom,
                frames[begin_key],
                end_atom,
                frames[end_key],
            )
            if begin_key == end_key and slot_translation == (0, 0, 0):
                raise ValueError(
                    f"Connection {record.pair} is an invalid same-slot internal "
                    "connection with zero translation."
                )

            begin_raw = raw_centroids[begin_key]
            end_raw = raw_centroids[end_key]
            origin = np.floor(begin_raw)
            begin_frac = begin_raw - origin
            end_frac = end_raw + np.asarray(slot_translation, dtype=int) - origin
            center_frac = 0.5 * (begin_frac + end_frac)
            edges.append(
                TopologyEdge(
                    begin_kind=begin_key[0],
                    begin_index=begin_key[1],
                    end_kind=end_key[0],
                    end_index=end_key[1],
                    begin_frac=tuple(float(value) for value in begin_frac),
                    end_frac=tuple(float(value) for value in end_frac),
                    center_frac=tuple(float(value) for value in center_frac),
                    source_cut_bond=record.pair,
                    source_cut_shift=record.shift,
                    slot_translation=slot_translation,
                )
            )
            coordination[begin_key] += 1
            coordination[end_key] += 1
        return edges

    @staticmethod
    def _build_vertices(
        fragments: list[Fragment],
        raw_centroids: dict[FragmentKey, np.ndarray],
        coordination: dict[FragmentKey, int],
    ) -> list[TopologyVertex]:
        vertices = []
        for fragment in fragments:
            key = (fragment.kind, fragment.index)
            wrapped = np.mod(raw_centroids[key], 1.0)
            vertices.append(
                TopologyVertex(
                    kind=fragment.kind,
                    index=fragment.index,
                    coordination=coordination[key],
                    frac_position=tuple(float(value) for value in wrapped),
                )
            )
        return vertices


class NodeEdgeTopologyBuilder(TopologyBuilder):
    """Build topology for metal nodes, organic nodes, and edge fragments."""

    def build(
        self,
        fragments: list[Fragment],
        connecting_bonds: tuple[ConnectionInput, ...],
    ) -> tuple[list[TopologyVertex], list[TopologyEdge]]:
        return self.build_fragments(
            fragments,
            connecting_bonds,
            preferred_begin_kind="metal_node",
        )


@dataclass
class EdgeTopologyContractor:
    """Contract two-port fragments into composer edge slots."""

    vertices: list[TopologyVertex]
    edges: list[TopologyEdge]
    edge_fragments: list[Fragment]

    def contract(self):
        edge_keys = {
            (fragment.kind, fragment.index) for fragment in self.edge_fragments
        }
        vertices = [
            vertex
            for vertex in self.vertices
            if (vertex.kind, vertex.index) not in edge_keys
        ]
        positions = {
            (vertex.kind, vertex.index): np.asarray(vertex.frac_position)
            for vertex in self.vertices
        }
        order = {
            (vertex.kind, vertex.index): index for index, vertex in enumerate(vertices)
        }
        topology_slots = {key: ("node", index) for key, index in order.items()}
        incidents = defaultdict(list)
        contracted_edges: list[TopologyEdge] = []

        for edge in self.edges:
            begin = (edge.begin_kind, edge.begin_index)
            end = (edge.end_kind, edge.end_index)
            begin_is_edge = begin in edge_keys
            end_is_edge = end in edge_keys
            if not (begin_is_edge or end_is_edge):
                contracted_edges.append(edge)
                continue

            begin_image = np.rint(
                np.asarray(edge.begin_frac) - positions[begin]
            ).astype(int)
            end_image = np.rint(np.asarray(edge.end_frac) - positions[end]).astype(int)
            shift = end_image - begin_image
            if begin_is_edge:
                incidents[begin].append((end, shift))
            else:
                incidents[end].append((begin, -shift))

        for fragment in self.edge_fragments:
            key = (fragment.kind, fragment.index)
            endpoints = incidents[key]
            (begin, begin_from_edge), (end, end_from_edge) = endpoints
            shift = end_from_edge - begin_from_edge
            if order[begin] > order[end] or (
                begin == end and tuple(shift) > tuple(-shift)
            ):
                begin, end = end, begin
                shift = -shift
            begin_frac = positions[begin]
            end_frac = positions[end] + shift
            edge_index = len(contracted_edges)
            contracted_edges.append(
                TopologyEdge(
                    begin_kind=begin[0],
                    begin_index=begin[1],
                    end_kind=end[0],
                    end_index=end[1],
                    begin_frac=tuple(float(value) for value in begin_frac),
                    end_frac=tuple(float(value) for value in end_frac),
                    center_frac=tuple(
                        float(value) for value in 0.5 * (begin_frac + end_frac)
                    ),
                    slot_translation=tuple(int(value) for value in shift),
                )
            )
            topology_slots[key] = ("edge", edge_index)

        coordination = Counter()
        for edge in contracted_edges:
            coordination[(edge.begin_kind, edge.begin_index)] += 1
            coordination[(edge.end_kind, edge.end_index)] += 1
        vertices = [
            TopologyVertex(
                vertex.kind,
                vertex.index,
                coordination[(vertex.kind, vertex.index)],
                vertex.frac_position,
            )
            for vertex in vertices
        ]
        return vertices, contracted_edges, topology_slots
