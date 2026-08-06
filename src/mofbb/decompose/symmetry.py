"""Symmetry of colored periodic topology graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import spglib
from ase.geometry import cellpar_to_cell
from scipy.optimize import linear_sum_assignment

from mofbb.decompose.object import TopologyEdge, TopologyVertex
from mofbb.decompose.pbc import PeriodicBond

PeriodicEdge = tuple[int, int, tuple[int, int, int]]


@dataclass(frozen=True)
class SpaceGroupOperations:
    """A named set of fractional-coordinate symmetry operations."""

    symbol: str
    rotations: np.ndarray
    translations: np.ndarray


@dataclass(frozen=True)
class AtomicSymmetry(SpaceGroupOperations):
    """Space-group operations and their action on source atoms."""

    atom_maps: np.ndarray
    atom_images: np.ndarray

    @classmethod
    def detect(cls, atoms) -> AtomicSymmetry:
        symbol, rotations, translations = detect_atomic_symmetry(atoms)
        atom_maps, atom_images = map_atoms_under_symmetry(
            atoms,
            rotations,
            translations,
        )
        return cls(
            symbol,
            rotations,
            translations,
            atom_maps,
            atom_images,
        )


def map_periodic_bond(
    record: PeriodicBond,
    rotation: np.ndarray,
    atom_map: np.ndarray,
    atom_images: np.ndarray,
) -> PeriodicBond:
    """Map a canonical periodic bond through one atomic symmetry operation."""

    shift = (
        atom_images[record.j]
        + rotation @ np.asarray(record.shift, dtype=int)
        - atom_images[record.i]
    )
    i = int(atom_map[record.i])
    j = int(atom_map[record.j])
    values = tuple(int(value) for value in shift)
    if i < j:
        return PeriodicBond(i, j, values)
    return PeriodicBond(j, i, tuple(-value for value in values))


class PeriodicBondOrbitClosure:
    """Close selected bonds under bond-preserving source symmetry."""

    def __init__(
        self,
        source_records: tuple[PeriodicBond, ...],
        symmetry: AtomicSymmetry,
    ) -> None:
        self.source_records = source_records
        self.symmetry = symmetry
        self.operations = tuple(
            zip(
                symmetry.rotations,
                symmetry.atom_maps,
                symmetry.atom_images,
                strict=True,
            )
        )
        self.preserves_source = self._preserves_source_connectivity()

    def close(self, selected: list[PeriodicBond]) -> list[PeriodicBond]:
        """Return every symmetry image of the selected periodic bonds."""

        if not self.preserves_source:
            return selected
        return sorted(
            {
                map_periodic_bond(record, rotation, atom_map, atom_images)
                for record in selected
                for rotation, atom_map, atom_images in self.operations
            }
        )

    def _preserves_source_connectivity(self) -> bool:
        source = set(self.source_records)
        return all(
            map_periodic_bond(record, rotation, atom_map, atom_images) in source
            for rotation, atom_map, atom_images in self.operations
            for record in self.source_records
        )


def _canonical_edge(begin: int, end: int, shift: np.ndarray) -> PeriodicEdge:
    direct = (begin, end, tuple(int(value) for value in shift))
    reverse = (end, begin, tuple(int(-value) for value in shift))
    return min(direct, reverse)


def detect_atomic_symmetry(atoms):
    """Return the conservative source space-group operations in its own cell."""

    cell = np.asarray(atoms.cell)
    structure = (
        cell,
        atoms.get_scaled_positions(wrap=True),
        atoms.get_atomic_numbers(),
    )
    for tolerance in (0.05, 0.10):
        dataset = spglib.get_symmetry_dataset(
            structure,
            symprec=tolerance,
            angle_tolerance=5.0,
        )
        if dataset is not None:
            return (
                str(dataset.international),
                np.asarray(dataset.rotations, dtype=int),
                np.asarray(dataset.translations, dtype=float),
            )
    return "P1", np.eye(3, dtype=int)[None], np.zeros((1, 3))


def map_atoms_under_symmetry(
    atoms,
    rotations: np.ndarray,
    translations: np.ndarray,
    tolerance: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return atom permutations and lattice images for symmetry operations."""

    fractional = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    mappings = []
    images = []
    for operation_index, (rotation, translation) in enumerate(
        zip(rotations, translations, strict=True)
    ):
        transformed = fractional @ np.asarray(rotation, dtype=int).T + translation
        mapping = np.empty(len(atoms), dtype=int)
        operation_images = np.empty((len(atoms), 3), dtype=int)
        maximum = 0.0
        for number in np.unique(numbers):
            indices = np.flatnonzero(numbers == number)
            differences = (
                transformed[indices, None, :] - fractional[None, indices, :]
            )
            candidate_images = np.rint(differences)
            distances = np.linalg.norm(
                (differences - candidate_images) @ cell,
                axis=2,
            )
            rows, columns = linear_sum_assignment(distances)
            source = indices[rows]
            mapping[source] = indices[columns]
            operation_images[source] = candidate_images[rows, columns].astype(int)
            maximum = max(
                maximum,
                float(np.max(distances[rows, columns], initial=0.0)),
            )
        if maximum > tolerance:
            raise ValueError(
                f"Atomic symmetry operation {operation_index} has mapping "
                f"residual {maximum:.3f} Å, above {tolerance:.3f} Å."
            )
        mappings.append(mapping)
        images.append(operation_images)
    return np.asarray(mappings, dtype=int), np.asarray(images, dtype=int)


@dataclass(frozen=True)
class TopologySymmetry:
    symbol: str
    rotations: np.ndarray
    translations: np.ndarray
    edge_types: np.ndarray
    node_maps: np.ndarray
    node_images: np.ndarray
    edge_maps: np.ndarray

    def project(
        self,
        vertices: list[TopologyVertex],
        edges: list[TopologyEdge],
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        positions = np.asarray(
            [vertex.frac_position for vertex in vertices],
            dtype=float,
        ).reshape(-1, 3)
        projected = np.zeros_like(positions)
        counts = np.zeros(len(positions), dtype=int)
        for rotation, translation, mapping, images in zip(
            self.rotations,
            self.translations,
            self.node_maps,
            self.node_images,
        ):
            transformed = positions @ rotation.T + translation - images
            np.add.at(projected, mapping, transformed)
            np.add.at(counts, mapping, 1)
        projected /= counts[:, None]

        indices = {
            (vertex.kind, vertex.index): index for index, vertex in enumerate(vertices)
        }
        endpoints = []
        for edge in edges:
            begin = indices[(edge.begin_kind, edge.begin_index)]
            end = indices[(edge.end_kind, edge.end_index)]
            endpoints.append(
                (
                    projected[begin]
                    + np.rint(np.asarray(edge.begin_frac) - positions[begin]),
                    projected[end]
                    + np.rint(np.asarray(edge.end_frac) - positions[end]),
                )
            )
        return projected, endpoints


def detect_topology_symmetry(
    cell_parameters: tuple[float, float, float, float, float, float],
    vertices: list[TopologyVertex],
    edges: list[TopologyEdge],
    vertex_types: list[int],
    edge_colors: list[int],
) -> TopologySymmetry:
    """Return space-group operations that preserve nodes, edges, and colors."""

    identity = TopologySymmetry(
        "P1",
        np.eye(3, dtype=int)[None],
        np.zeros((1, 3)),
        np.arange(1, len(edges) + 1),
        np.arange(len(vertices), dtype=int)[None],
        np.zeros((1, len(vertices), 3), dtype=int),
        np.arange(len(edges), dtype=int)[None],
    )
    if not vertices:
        return identity

    positions = np.asarray([vertex.frac_position for vertex in vertices])
    types = np.asarray(vertex_types)
    indices = {
        (vertex.kind, vertex.index): index for index, vertex in enumerate(vertices)
    }
    periodic_edges = []
    for edge in edges:
        begin = indices[(edge.begin_kind, edge.begin_index)]
        end = indices[(edge.end_kind, edge.end_index)]
        begin_image = np.rint(np.asarray(edge.begin_frac) - positions[begin])
        end_image = np.rint(np.asarray(edge.end_frac) - positions[end])
        periodic_edges.append(_canonical_edge(begin, end, end_image - begin_image))

    marker_types = {
        pair: int(types.max()) + index
        for index, pair in enumerate(
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
    for index, (begin, end, shift) in enumerate(periodic_edges):
        delta = positions[end] + shift - positions[begin]
        marker_type = marker_types[
            (tuple(sorted((types[begin], types[end]))), edge_colors[index])
        ]
        for fraction in (1 / 3, 2 / 3):
            marker = np.mod(positions[begin] + fraction * delta, 1.0)
            key = (marker_type, *np.round(marker, 8))
            if key not in seen:
                seen.add(key)
                decorated_positions.append(marker)
                decorated_types.append(marker_type)

    edge_indices: dict[tuple[PeriodicEdge, int], list[int]] = defaultdict(list)
    for index, edge in enumerate(periodic_edges):
        edge_indices[(edge, edge_colors[index])].append(index)
    edge_rank = {
        index: rank
        for indices in edge_indices.values()
        for rank, index in enumerate(indices)
    }
    cell = np.asarray(cellpar_to_cell(cell_parameters))
    same_type = types[:, None] == types[None]

    for tolerance in (0.05, 0.10):
        dataset = spglib.get_symmetry_dataset(
            (cell, np.asarray(decorated_positions), np.asarray(decorated_types)),
            symprec=tolerance,
            angle_tolerance=5.0,
        )
        if dataset is None:
            continue
        rotations, translations = [], []
        node_maps, node_images, edge_maps = [], [], []
        for rotation, translation in zip(dataset.rotations, dataset.translations):
            transformed = positions @ rotation.T + translation
            differences = transformed[:, None] - positions[None]
            images = np.rint(differences)
            distances = np.linalg.norm((differences - images) @ cell, axis=2)
            matches = same_type & (distances <= tolerance)
            if np.any(matches.sum(axis=1) != 1):
                continue
            mapping = np.argmax(matches, axis=1)
            mapped_images = images[np.arange(len(vertices)), mapping].astype(int)

            mapped_edges = []
            for index, (begin, end, shift) in enumerate(periodic_edges):
                target = _canonical_edge(
                    int(mapping[begin]),
                    int(mapping[end]),
                    mapped_images[end]
                    + rotation @ np.asarray(shift)
                    - mapped_images[begin],
                )
                candidates = edge_indices.get((target, edge_colors[index]), ())
                if edge_rank[index] >= len(candidates):
                    break
                mapped_edges.append(candidates[edge_rank[index]])
            if len(mapped_edges) != len(edges) or len(set(mapped_edges)) != len(edges):
                continue
            rotations.append(rotation)
            translations.append(translation)
            node_maps.append(mapping)
            node_images.append(mapped_images)
            edge_maps.append(mapped_edges)

        if len(rotations) <= 1:
            continue
        rotations = np.asarray(rotations)
        translations = np.asarray(translations)
        group = spglib.get_spacegroup_type_from_symmetry(
            rotations,
            translations,
            cell,
            symprec=tolerance,
        )
        if group is None:
            continue
        edge_types = np.zeros(len(edges), dtype=int)
        for edge_index in range(len(edges)):
            if edge_types[edge_index] == 0:
                orbit = np.unique(np.asarray(edge_maps)[:, edge_index])
                edge_types[orbit] = int(edge_types.max()) + 1
        return TopologySymmetry(
            str(group.international_short),
            rotations,
            translations,
            edge_types,
            np.asarray(node_maps),
            np.asarray(node_images),
            np.asarray(edge_maps),
        )
    return identity
