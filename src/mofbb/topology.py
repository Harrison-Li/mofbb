"""Collapsed topology construction for node/linker fragments."""

from __future__ import annotations

from collections import defaultdict

import ase
import numpy as np

from .bonding import BondNetwork
from .geometry import cart_to_frac, unwrap_fractional_near, unwrap_positions
from .models import Fragment, TopologyEdge, TopologyVertex


class TopologyBuilder:
    """Build P1 CGD vertices and edges from capped fragments."""

    def __init__(self, atoms: ase.Atoms, bonds: BondNetwork) -> None:
        self.atoms = atoms
        self.bonds = bonds

    def build(
        self,
        node_fragments: list[Fragment],
        linker_fragments: list[Fragment],
        connecting_bonds: tuple[tuple[int, int], ...],
    ) -> tuple[list[TopologyVertex], list[TopologyEdge]]:
        """Return collapsed topology records for the decomposition result."""

        fragments = [*node_fragments, *linker_fragments]
        atom_to_vertex = self._atom_to_vertex(fragments)
        centroids = {
            (fragment.kind, fragment.index): self.fragment_centroid_fractional(
                fragment.source_indices
            )
            for fragment in fragments
        }

        coordination: dict[tuple[str, int], int] = defaultdict(int)
        edges = self._build_edges(connecting_bonds, atom_to_vertex, centroids, coordination)
        vertices = self._build_vertices(fragments, centroids, coordination)
        return vertices, edges

    @staticmethod
    def _atom_to_vertex(fragments: list[Fragment]) -> dict[int, tuple[str, int]]:
        atom_to_vertex: dict[int, tuple[str, int]] = {}
        for fragment in fragments:
            for atom_index in fragment.source_indices:
                atom_to_vertex[atom_index] = (fragment.kind, fragment.index)
        return atom_to_vertex

    def _build_edges(
        self,
        connecting_bonds: tuple[tuple[int, int], ...],
        atom_to_vertex: dict[int, tuple[str, int]],
        centroids: dict[tuple[str, int], tuple[float, float, float]],
        coordination: dict[tuple[str, int], int],
    ) -> list[TopologyEdge]:
        edges: list[TopologyEdge] = []

        for i, j in connecting_bonds:
            begin_key = atom_to_vertex[i]
            end_key = atom_to_vertex[j]
            if begin_key == end_key:
                continue
            if begin_key[0] != "node":
                begin_key, end_key = end_key, begin_key

            begin_frac = centroids[begin_key]
            end_frac = unwrap_fractional_near(centroids[end_key], begin_frac)
            center_frac = tuple(0.5 * (np.array(begin_frac) + np.array(end_frac)))
            edges.append(
                TopologyEdge(
                    begin_kind=begin_key[0],
                    begin_index=begin_key[1],
                    end_kind=end_key[0],
                    end_index=end_key[1],
                    begin_frac=tuple(float(value) for value in begin_frac),
                    end_frac=tuple(float(value) for value in end_frac),
                    center_frac=tuple(float(value) for value in center_frac),
                )
            )
            coordination[begin_key] += 1
            coordination[end_key] += 1

        return edges

    @staticmethod
    def _build_vertices(
        fragments: list[Fragment],
        centroids: dict[tuple[str, int], tuple[float, float, float]],
        coordination: dict[tuple[str, int], int],
    ) -> list[TopologyVertex]:
        vertices = []
        for fragment in fragments:
            key = (fragment.kind, fragment.index)
            vertices.append(
                TopologyVertex(
                    kind=fragment.kind,
                    index=fragment.index,
                    coordination=coordination[key],
                    frac_position=tuple(float(value) for value in centroids[key]),
                )
            )
        return vertices

    def fragment_centroid_fractional(
        self, source_indices: tuple[int, ...]
    ) -> tuple[float, float, float]:
        """Return the wrapped fractional centroid of a finite source fragment."""

        source_list = list(source_indices)
        local_index = {global_index: i for i, global_index in enumerate(source_list)}
        positions = np.array([self.atoms.positions[i] for i in source_list], dtype=float)
        local_edges = [
            (local_index[i], local_index[j])
            for i, j in self.bonds.pairs
            if i in local_index and j in local_index
        ]
        unwrapped = unwrap_positions(positions, self.atoms.cell.array, local_edges)
        centroid_cart = unwrapped.mean(axis=0)
        return tuple(
            float(value) for value in cart_to_frac(self.atoms.cell.array, centroid_cart) % 1.0
        )
