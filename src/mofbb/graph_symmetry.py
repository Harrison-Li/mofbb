"""Exact chemical-graph symmetry acting on building-block connection ports.

Coordinate point groups describe a particular conformer.  This module instead
labels a molecular graph by element and bond type, computes its exact graph
automorphisms, and restricts those automorphisms to the ordered ``X`` ports.
Consequently, a distorted conformer can retain the chemical symmetry needed by
the assembly machinery.

Bond labels are represented by coloured subdivision vertices.  This lets the
BLISS implementation in :mod:`igraph` preserve both vertex and edge labels,
while still returning a compact generating set rather than enumerating every
automorphism.  :mod:`sympy` then provides exact permutation-group membership,
order, and orbit calculations for the induced port action.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np
from scipy.optimize import linear_sum_assignment
from sympy.combinatorics import Permutation, PermutationGroup


PortPermutation = tuple[int, ...]


def _normalize_permutation(
    permutation: Sequence[int],
    degree: int,
) -> PortPermutation:
    normalized = tuple(int(value) for value in permutation)
    if len(normalized) != degree:
        raise ValueError(
            f"Expected a permutation of {degree} ports, got {len(normalized)}."
        )
    if tuple(sorted(normalized)) != tuple(range(degree)):
        raise ValueError(
            f"Port permutation must contain every index from 0 to {degree - 1}."
        )
    return normalized


def permutation_support(permutation: Sequence[int]) -> tuple[int, ...]:
    """Return the port indices moved by ``permutation``.

    The input is checked to be a valid zero-based permutation before its
    support is returned.
    """

    normalized = _normalize_permutation(permutation, len(permutation))
    return tuple(index for index, image in enumerate(normalized) if image != index)


def _identity_group(degree: int) -> PermutationGroup:
    identity = Permutation(list(range(degree)), size=degree)
    return PermutationGroup([identity])


def _permutation_group(
    generators: Iterable[PortPermutation],
    degree: int,
) -> PermutationGroup:
    normalized = tuple(generators)
    if not normalized:
        return _identity_group(degree)
    return PermutationGroup(
        [Permutation(list(generator), size=degree) for generator in normalized]
    )


@dataclass(frozen=True)
class BuildingBlockGraph:
    """Element- and bond-type-labelled graph with ordered connection ports."""

    symbols: tuple[str, ...]
    bonds: tuple[tuple[int, int], ...]
    bond_types: tuple[str, ...]
    port_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        symbols = tuple(str(symbol) for symbol in self.symbols)
        if not symbols:
            raise ValueError("A building-block graph must contain at least one atom.")

        raw_bonds = tuple(tuple(int(index) for index in bond) for bond in self.bonds)
        bond_types = tuple(str(bond_type) for bond_type in self.bond_types)
        if len(raw_bonds) != len(bond_types):
            raise ValueError("bonds and bond_types must have the same length.")

        labelled_bonds: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int]] = set()
        for bond, bond_type in zip(raw_bonds, bond_types, strict=True):
            if len(bond) != 2:
                raise ValueError("Each bond must contain exactly two atom indices.")
            left, right = bond
            if left == right:
                raise ValueError(f"Self-bond at atom {left} is not supported.")
            if not (0 <= left < len(symbols) and 0 <= right < len(symbols)):
                raise ValueError(
                    f"Bond {(left, right)} contains an invalid atom index."
                )
            pair = tuple(sorted((left, right)))
            if pair in seen:
                raise ValueError(f"Duplicate bond {pair} is not supported.")
            seen.add(pair)
            labelled_bonds.append((pair[0], pair[1], bond_type))
        labelled_bonds.sort()

        ports = tuple(int(index) for index in self.port_indices)
        if len(set(ports)) != len(ports):
            raise ValueError("port_indices must be unique.")
        if any(index < 0 or index >= len(symbols) for index in ports):
            raise ValueError("port_indices contains an invalid atom index.")

        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(
            self,
            "bonds",
            tuple((left, right) for left, right, _ in labelled_bonds),
        )
        object.__setattr__(
            self,
            "bond_types",
            tuple(bond_type for _, _, bond_type in labelled_bonds),
        )
        object.__setattr__(self, "port_indices", ports)

    @classmethod
    def from_data(
        cls,
        symbols: Sequence[str],
        bonds: Iterable[Sequence[int]],
        bond_types: Iterable[str] | None = None,
        port_indices: Iterable[int] | None = None,
    ) -> BuildingBlockGraph:
        """Build a normalized graph from arrays or other simple iterables."""

        normalized_symbols = tuple(str(symbol) for symbol in symbols)
        normalized_bonds = tuple(tuple(int(index) for index in bond) for bond in bonds)
        normalized_bond_types = (
            tuple("S" for _ in normalized_bonds)
            if bond_types is None
            else tuple(str(bond_type) for bond_type in bond_types)
        )
        normalized_ports = (
            tuple(
                index
                for index, symbol in enumerate(normalized_symbols)
                if symbol == "X"
            )
            if port_indices is None
            else tuple(int(index) for index in port_indices)
        )
        return cls(
            symbols=normalized_symbols,
            bonds=normalized_bonds,
            bond_types=normalized_bond_types,
            port_indices=normalized_ports,
        )

    @classmethod
    def from_building_block(cls, block: Any) -> BuildingBlockGraph:
        """Extract graph data from a :class:`mofbb.compose.BuildingBlock`."""

        return cls.from_data(
            symbols=block.atoms.get_chemical_symbols(),
            bonds=block.bonds,
            bond_types=block.bond_types,
            port_indices=block.connection_point_indices,
        )


@dataclass(frozen=True)
class PortGraphSymmetry:
    """Exact automorphism-group action induced on ordered ``X`` ports.

    ``generators`` act on *port positions*, not atom indices.  For example,
    ``(1, 0)`` exchanges the first and second entries of ``port_indices``.
    """

    port_indices: tuple[int, ...]
    generators: tuple[PortPermutation, ...]
    order: int
    orbits: tuple[tuple[int, ...], ...]
    connected: bool
    graph_automorphism_order: int
    _group: PermutationGroup = field(repr=False, compare=False)

    @property
    def degree(self) -> int:
        """Number of ordered ports on which the group acts."""

        return len(self.port_indices)

    @property
    def endpoint_equivalent(self) -> bool:
        """Whether the two endpoints of a two-connected block are equivalent."""

        return self.degree == 2 and self.equivalent(0, 1)

    def contains(self, permutation: Sequence[int]) -> bool:
        """Return whether ``permutation`` belongs to the induced port group."""

        normalized = _normalize_permutation(permutation, self.degree)
        candidate = Permutation(list(normalized), size=self.degree)
        return bool(self._group.contains(candidate, strict=True))

    def supports(self, permutation: Sequence[int]) -> bool:
        """Alias for :meth:`contains`, phrased for topology requirements."""

        return self.contains(permutation)

    def supports_all(self, permutations: Iterable[Sequence[int]]) -> bool:
        """Return whether every required port permutation is supported."""

        return all(self.contains(permutation) for permutation in permutations)

    def equivalent(self, left: int, right: int) -> bool:
        """Return whether two port positions lie in the same group orbit."""

        left = int(left)
        right = int(right)
        if not (0 <= left < self.degree and 0 <= right < self.degree):
            raise IndexError("Port position is out of range.")
        return any(left in orbit and right in orbit for orbit in self.orbits)

    def moves(self, port: int) -> bool:
        """Return whether any graph automorphism moves a port position."""

        port = int(port)
        if not 0 <= port < self.degree:
            raise IndexError("Port position is out of range.")
        return any(len(orbit) > 1 and port in orbit for orbit in self.orbits)


@dataclass(frozen=True)
class CoordinatePortAction:
    """Unique port permutations induced by coordinate symmetry operations."""

    permutations: tuple[PortPermutation, ...]
    max_residual: float
    operation_count: int
    port_count: int
    tolerance: float

    @property
    def degree(self) -> int:
        """Number of ports in each permutation."""

        return self.port_count

    def contains(self, permutation: Sequence[int]) -> bool:
        """Return whether a coordinate operation induces ``permutation``."""

        normalized = _normalize_permutation(permutation, self.degree)
        return normalized in self.permutations

    def supports(self, permutation: Sequence[int]) -> bool:
        """Alias for :meth:`contains`."""

        return self.contains(permutation)

    def supports_all(self, permutations: Iterable[Sequence[int]]) -> bool:
        """Return whether all requested permutations occur geometrically."""

        return all(self.contains(permutation) for permutation in permutations)


def _coerce_building_block_graph(
    source: BuildingBlockGraph | Any,
    bonds: Iterable[Sequence[int]] | None,
    bond_types: Iterable[str] | None,
    port_indices: Iterable[int] | None,
) -> BuildingBlockGraph:
    if isinstance(source, BuildingBlockGraph):
        if bonds is not None or bond_types is not None or port_indices is not None:
            raise ValueError(
                "Do not supply bonds, bond_types, or port_indices with "
                "BuildingBlockGraph."
            )
        return source
    if hasattr(source, "atoms") and hasattr(source, "connection_point_indices"):
        if bonds is not None or bond_types is not None or port_indices is not None:
            raise ValueError(
                "Do not supply graph arrays when source is a BuildingBlock."
            )
        return BuildingBlockGraph.from_building_block(source)
    if bonds is None:
        raise TypeError("Raw element symbols require a bonds argument.")
    return BuildingBlockGraph.from_data(
        symbols=source,
        bonds=bonds,
        bond_types=bond_types,
        port_indices=port_indices,
    )


def analyze_port_graph_symmetry(
    source: BuildingBlockGraph | Any,
    bonds: Iterable[Sequence[int]] | None = None,
    bond_types: Iterable[str] | None = None,
    port_indices: Iterable[int] | None = None,
) -> PortGraphSymmetry:
    """Compute the exact labelled-graph automorphism action on ``X`` ports.

    Parameters
    ----------
    source
        A :class:`BuildingBlockGraph`, a ``BuildingBlock``, or a sequence of
        element symbols.  Raw symbols must be accompanied by ``bonds``.
    bonds, bond_types, port_indices
        Raw graph arrays.  Missing bond labels default to ``"S"`` and missing
        port indices are inferred from atoms whose symbol is ``"X"``.
    """

    block_graph = _coerce_building_block_graph(
        source,
        bonds=bonds,
        bond_types=bond_types,
        port_indices=port_indices,
    )
    atom_count = len(block_graph.symbols)
    bond_count = len(block_graph.bonds)

    port_atom_indices = set(block_graph.port_indices)
    vertex_labels: list[tuple[str, str]] = [
        ("port" if index in port_atom_indices else "atom", symbol)
        for index, symbol in enumerate(block_graph.symbols)
    ]
    vertex_labels.extend(
        ("bond", bond_type) for bond_type in block_graph.bond_types
    )
    palette = {
        label: color for color, label in enumerate(sorted(set(vertex_labels)))
    }
    colors = [palette[label] for label in vertex_labels]

    subdivided_edges: list[tuple[int, int]] = []
    for bond_index, (left, right) in enumerate(block_graph.bonds):
        bond_vertex = atom_count + bond_index
        subdivided_edges.extend(((left, bond_vertex), (right, bond_vertex)))
    graph = ig.Graph(
        n=atom_count + bond_count,
        edges=subdivided_edges,
        directed=False,
    )

    full_generators = graph.automorphism_group(color=colors)
    port_position = {
        atom_index: position
        for position, atom_index in enumerate(block_graph.port_indices)
    }
    identity = tuple(range(len(block_graph.port_indices)))
    induced_candidates: set[PortPermutation] = set()
    for generator in full_generators:
        induced = tuple(
            port_position[int(generator[atom_index])]
            for atom_index in block_graph.port_indices
        )
        if induced != identity:
            induced_candidates.add(induced)

    # BLISS is deterministic for a fixed labelled graph, but sorting and
    # removing generators already implied by earlier candidates gives a small,
    # stable public representation of its induced action.
    selected: list[PortPermutation] = []
    induced_group = _identity_group(len(block_graph.port_indices))
    for candidate in sorted(induced_candidates):
        sympy_candidate = Permutation(
            list(candidate), size=len(block_graph.port_indices)
        )
        if induced_group.contains(sympy_candidate, strict=True):
            continue
        selected.append(candidate)
        induced_group = _permutation_group(
            selected,
            degree=len(block_graph.port_indices),
        )

    orbits = tuple(
        sorted(
            tuple(sorted(int(index) for index in orbit))
            for orbit in induced_group.orbits()
        )
    )
    return PortGraphSymmetry(
        port_indices=block_graph.port_indices,
        generators=tuple(selected),
        order=int(induced_group.order()),
        orbits=orbits,
        connected=bool(graph.is_connected()),
        graph_automorphism_order=int(graph.count_automorphisms(color=colors)),
        _group=induced_group,
    )


def port_permutations_from_operations(
    centered_port_coordinates: Sequence[Sequence[float]],
    operations: Iterable[Any],
    tolerance: float,
) -> CoordinatePortAction:
    """Match centered coordinates after symmetry operations to port indices.

    Each operation is assigned to the original coordinate set with an exact
    one-to-one Hungarian matching.  An operation whose largest assigned
    distance exceeds ``tolerance`` is rejected instead of silently emitting an
    invalid permutation.
    """

    coordinates = np.asarray(centered_port_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError("centered_port_coordinates must have shape (n_ports, 3).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("centered_port_coordinates must contain finite values.")
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")

    operation_list = tuple(operations)
    unique: set[PortPermutation] = set()
    residuals: list[float] = []
    for operation_index, operation in enumerate(operation_list):
        if hasattr(operation, "operate_multi"):
            transformed = np.asarray(operation.operate_multi(coordinates), dtype=float)
        elif hasattr(operation, "rotation_matrix"):
            rotation = np.asarray(operation.rotation_matrix, dtype=float)
            translation = np.asarray(
                getattr(operation, "translation_vector", np.zeros(3)),
                dtype=float,
            )
            transformed = coordinates @ rotation.T + translation
        else:
            rotation = np.asarray(operation, dtype=float)
            if rotation.shape != (3, 3):
                raise TypeError(
                    "Each operation must be a pymatgen SymmOp or a 3x3 matrix."
                )
            transformed = coordinates @ rotation.T

        distances = np.linalg.norm(
            transformed[:, np.newaxis, :] - coordinates[np.newaxis, :, :],
            axis=2,
        )
        row_indices, column_indices = linear_sum_assignment(distances)
        assignment = np.empty(len(coordinates), dtype=int)
        assignment[row_indices] = column_indices
        residual = (
            float(np.max(distances[row_indices, column_indices]))
            if len(coordinates)
            else 0.0
        )
        if residual > tolerance:
            raise ValueError(
                f"Point-group operation {operation_index} does not map the ports "
                f"within tolerance: residual {residual:.6g} > {tolerance:.6g}."
            )
        unique.add(tuple(int(index) for index in assignment))
        residuals.append(residual)

    return CoordinatePortAction(
        permutations=tuple(sorted(unique)),
        max_residual=max(residuals, default=0.0),
        operation_count=len(operation_list),
        port_count=len(coordinates),
        tolerance=tolerance,
    )


def point_group_port_action(
    analyzer: Any,
    centered_port_coordinates: Sequence[Sequence[float]] | None = None,
    tolerance: float | None = None,
) -> CoordinatePortAction:
    """Return unique port permutations induced by a PointGroupAnalyzer.

    When coordinates are omitted, the analyzer's own centered molecule is
    used.  This is the safest pairing because the returned symmetry operations
    are defined in that coordinate frame.
    """

    coordinates = (
        np.asarray(analyzer.centered_mol.cart_coords, dtype=float)
        if centered_port_coordinates is None
        else np.asarray(centered_port_coordinates, dtype=float)
    )
    operation_tolerance = (
        float(getattr(analyzer, "tol", 0.3))
        if tolerance is None
        else float(tolerance)
    )
    return port_permutations_from_operations(
        centered_port_coordinates=coordinates,
        operations=analyzer.get_symmetry_operations(),
        tolerance=operation_tolerance,
    )


__all__ = [
    "BuildingBlockGraph",
    "CoordinatePortAction",
    "PortGraphSymmetry",
    "PortPermutation",
    "analyze_port_graph_symmetry",
    "permutation_support",
    "point_group_port_action",
    "port_permutations_from_operations",
]
