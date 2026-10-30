"""Collapsed topology construction for MOF decomposition fragments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import ase
import numpy as np

from mofbb.common.pbc import (
    ComponentPeriodicity,
    PeriodicBond,
    connection_shift,
)
from mofbb.decompose.bonding import BondNetwork
from mofbb.decompose.crystal import Crystal
from mofbb.decompose.fragments import get_atom_map
from mofbb.decompose.object import Fragment, TopologyEdge, TopologyVertex


class TopologyBuilder:
    """Build CGD vertices and edges from exact periodic fragment frames."""

    def __init__(
        self,
        atoms: ase.Atoms,
        bonds: BondNetwork,
        bond_images: tuple[PeriodicBond, ...],
    ) -> None:
        self.crystal = Crystal(atoms, bonds, bond_images)

    def build(
        self,
        fragments: list[Fragment],
        connections: tuple[PeriodicBond, ...],
    ) -> tuple[list[TopologyVertex], list[TopologyEdge]]:
        '''
        Collapse fragments to vertices and turn cut bonds into periodic edges.

        Args:
            fragments (list[Fragment]):
                Fragments covering the atoms at both ends of each connection. Their
                real-atom fractional centroids are updated in place.
            connections (tuple[PeriodicBond, ...]):
                Cut bonds connecting fragments, including cell translations.

        Returns:
            Lists of topology vertices and edges with fractional positions.
        '''
        atom_to_vertex = get_atom_map(fragments)
        frames = {
            (fragment.kind, fragment.index): self.crystal.get_fragment_frame(
                fragment, connections
            )
            for fragment in fragments
        }
        real_centroids = {
            (fragment.kind, fragment.index): self.crystal.get_bb_centroid(
                fragment, frames[(fragment.kind, fragment.index)]
            )
            for fragment in fragments
        }
        for fragment in fragments:
            key = (fragment.kind, fragment.index)
            fragment.real_atom_centroid_frac = tuple(
                float(value) for value in np.mod(real_centroids[key], 1.0)
            )
        raw_centroids = {
            (fragment.kind, fragment.index): self.crystal.get_x_centroid(
                fragment,
                frames[(fragment.kind, fragment.index)],
            )
            for fragment in fragments
        }
        edges = self._build_edges(
            connections,
            atom_to_vertex,
            frames,
            raw_centroids,
        )
        coordination = Counter()
        for edge in edges:
            coordination[(edge.begin_kind, edge.begin_index)] += 1
            coordination[(edge.end_kind, edge.end_index)] += 1
        vertices = self._build_vertices(fragments, raw_centroids, coordination)
        return vertices, edges

    def _build_edges(
        self,
        connection_records: tuple[PeriodicBond, ...],
        atom_to_vertex: dict[int, tuple[str, int]],
        frames: dict[tuple[str, int], ComponentPeriodicity],
        raw_centroids: dict[tuple[str, int], np.ndarray],
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
        return edges

    @staticmethod
    def _build_vertices(
        fragments: list[Fragment],
        raw_centroids: dict[tuple[str, int], np.ndarray],
        coordination: dict[tuple[str, int], int],
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
class EdgeContractor:
    """Contract two-port fragments into composer edge slots."""

    vertices: list[TopologyVertex]
    edges: list[TopologyEdge]
    edge_fragments: list[Fragment]

    def contract(
        self,
    ) -> tuple[
        list[TopologyVertex],
        list[TopologyEdge],
        dict[tuple[str, int], tuple[str, int]],
    ]:
        '''
        Replace two-port fragments with edges between their neighboring vertices.

        Zero-translation loops remain explicit fragments.

        Returns:
            The remaining vertices, the contracted edges, and a mapping from each
            fragment key to its node or edge slot.
        '''
        edge_keys = {
            (fragment.kind, fragment.index) for fragment in self.edge_fragments
        }
        positions = {
            (vertex.kind, vertex.index): np.asarray(vertex.frac_position)
            for vertex in self.vertices
        }
        incidents = defaultdict(list)
        for edge in self.edges:
            begin = (edge.begin_kind, edge.begin_index)
            end = (edge.end_kind, edge.end_index)
            begin_is_edge = begin in edge_keys
            end_is_edge = end in edge_keys
            if not (begin_is_edge or end_is_edge):
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

        contractions = {}
        for key in edge_keys:
            (begin, begin_shift), (end, end_shift) = incidents[key]
            shift = end_shift - begin_shift
            # Keep zero-shift loops as explicit fragments.
            if begin != end or np.any(shift):
                contractions[key] = (begin, end, shift)

        vertices = [
            vertex
            for vertex in self.vertices
            if (vertex.kind, vertex.index) not in contractions
        ]
        order = {
            (vertex.kind, vertex.index): index for index, vertex in enumerate(vertices)
        }
        topology_slots = {key: ("node", index) for key, index in order.items()}
        contracted_edges = [
            edge
            for edge in self.edges
            if (edge.begin_kind, edge.begin_index) not in contractions
            and (edge.end_kind, edge.end_index) not in contractions
        ]

        for fragment in self.edge_fragments:
            key = (fragment.kind, fragment.index)
            if key not in contractions:
                continue
            begin, end, shift = contractions[key]
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
