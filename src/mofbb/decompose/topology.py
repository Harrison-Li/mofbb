"""Collapsed topology construction for MOF decomposition fragments."""

from __future__ import annotations

from collections import Counter, defaultdict
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
)

FragmentKey = tuple[str, int]


class NodeEdgeTopologyBuilder:
    """Build CGD vertices and edges from exact periodic fragment frames."""

    def __init__(
        self,
        atoms: ase.Atoms,
        bonds: BondNetwork,
        source_records: tuple[PeriodicBond, ...],
    ) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.source_records = source_records

    def build(
        self,
        fragments: list[Fragment],
        connections: tuple[PeriodicBond, ...],
    ) -> tuple[list[TopologyVertex], list[TopologyEdge]]:
        atom_to_vertex = self._atom_to_vertex(fragments)
        frames = {
            (fragment.kind, fragment.index): self._fragment_frame(
                fragment,
                self.source_records,
                connections,
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
            connections,
            atom_to_vertex,
            frames,
            raw_centroids,
            coordination,
        )
        vertices = self._build_vertices(fragments, raw_centroids, coordination)
        return vertices, edges

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
        if fragment.source_image_offsets:
            return ComponentPeriodicity(fragment.source_image_offsets, (), 0)
        return analyze_component(
            source_indices,
            source_records,
            excluded=internal_cuts,
        )

    def _raw_real_centroids(
        self,
        fragments: list[Fragment],
        frames: dict[FragmentKey, ComponentPeriodicity],
    ) -> dict[FragmentKey, np.ndarray]:
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
    ) -> list[TopologyEdge]:
        edges = []
        for record in connection_records:
            begin_atom, end_atom = record.pair
            begin_key = atom_to_vertex[begin_atom]
            end_key = atom_to_vertex[end_atom]

            if begin_key[0] != "metal_node" and end_key[0] == "metal_node":
                begin_atom, end_atom = end_atom, begin_atom
                begin_key, end_key = end_key, begin_key

            slot_translation = connection_shift(
                record,
                begin_atom,
                frames[begin_key],
                end_atom,
                frames[end_key],
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
