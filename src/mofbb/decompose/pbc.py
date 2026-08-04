"""Periodic-boundary connectivity analysis for MOF decomposition."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import TypeAlias

import numpy as np
from ase import Atoms

from .bonding import BondNetwork
from .constants import is_mofid_metal

CellShift: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True, order=True)
class PeriodicBond:
    """One bond and its integer unit-cell translation."""

    i: int
    j: int
    shift: CellShift

    @property
    def pair(self) -> tuple[int, int]:
        return (self.i, self.j)

    def oriented_shift(self, start: int, end: int) -> CellShift:
        return self.shift if start == self.i else tuple(-v for v in self.shift)


@dataclass(frozen=True)
class ComponentPeriodicity:
    """Image offsets and periodic cycle generators for one component."""

    offsets: tuple[tuple[int, CellShift], ...]
    generators: tuple[CellShift, ...]
    rank: int

    @cached_property
    def _offset_by_atom(self) -> dict[int, CellShift]:
        return dict(self.offsets)

    def offset(self, atom_index: int) -> CellShift:
        return self._offset_by_atom[atom_index]


@dataclass(frozen=True)
class PeriodicCutSelection:
    cuts: tuple[PeriodicBond, ...]


class _DisconnectedComponentError(ValueError):
    """Raised when excluded bonds split a requested component."""


def periodic_bonds(atoms: Atoms, bonds: BondNetwork) -> tuple[PeriodicBond, ...]:
    """Return source bonds annotated with integer cell translations."""

    periodic_axes = np.asarray(atoms.pbc, dtype=bool)
    inverse_cell = (
        np.linalg.inv(np.asarray(atoms.cell.array, dtype=float))
        if np.any(periodic_axes)
        else None
    )

    records = []
    for i, j in bonds.pairs:
        direct = np.asarray(atoms.positions[j] - atoms.positions[i], dtype=float)
        shift_cart = bonds.vector(i, j) - direct
        if inverse_cell is None:
            shift = np.zeros(3, dtype=int)
        else:
            shift = np.rint(shift_cart @ inverse_cell).astype(int)
            shift[~periodic_axes] = 0
        records.append(PeriodicBond(i, j, tuple(int(value) for value in shift)))
    return tuple(records)


def analyze_component(
    atom_indices: set[int] | frozenset[int],
    records: tuple[PeriodicBond, ...],
    excluded: set[PeriodicBond] | frozenset[PeriodicBond] = frozenset(),
) -> ComponentPeriodicity:
    """Return image offsets and periodic rank for one connected component."""

    selected = set(atom_indices)
    if not selected:
        raise ValueError("Cannot analyze an empty periodic component.")

    adjacency: dict[int, list[tuple[int, PeriodicBond]]] = {
        index: [] for index in selected
    }
    for record in records:
        if record in excluded or record.i not in selected or record.j not in selected:
            continue
        adjacency[record.i].append((record.j, record))
        adjacency[record.j].append((record.i, record))

    root = min(selected)
    offsets: dict[int, CellShift] = {root: (0, 0, 0)}
    generators: set[CellShift] = set()
    queue = deque([root])
    while queue:
        current = queue.popleft()
        current_offset = offsets[current]
        for neighbor, record in sorted(adjacency[current]):
            shift = record.oriented_shift(current, neighbor)
            expected = tuple(a + b for a, b in zip(current_offset, shift))
            neighbor_offset = offsets.get(neighbor)
            if neighbor_offset is None:
                offsets[neighbor] = expected
                queue.append(neighbor)
                continue
            residual = tuple(a - b for a, b in zip(expected, neighbor_offset))
            if residual != (0, 0, 0):
                reverse = tuple(-value for value in residual)
                generators.add(min(residual, reverse))

    if set(offsets) != selected:
        missing = sorted(selected - set(offsets))
        raise _DisconnectedComponentError(
            f"Periodic component is disconnected; missing atoms {missing}."
        )

    normalized = tuple(sorted(generators))
    return ComponentPeriodicity(
        offsets=tuple((index, offsets[index]) for index in sorted(offsets)),
        generators=normalized,
        rank=_cell_shift_rank(normalized),
    )


def _cell_shift_rank(vectors: tuple[CellShift, ...]) -> int:
    """Return the rank of integer cell-shift generators."""

    if not vectors:
        return 0
    first = vectors[0]
    for other in vectors[1:]:
        cross = (
            first[1] * other[2] - first[2] * other[1],
            first[2] * other[0] - first[0] * other[2],
            first[0] * other[1] - first[1] * other[0],
        )
        if cross == (0, 0, 0):
            continue
        if any(sum(a * b for a, b in zip(vector, cross)) for vector in vectors):
            return 3
        return 2
    return 1


def connection_shift(
    record: PeriodicBond,
    begin_atom: int,
    begin_offsets: ComponentPeriodicity,
    end_atom: int,
    end_offsets: ComponentPeriodicity,
) -> CellShift:
    """Return the translation from the begin fragment slot to the end slot."""

    raw = np.asarray(record.oriented_shift(begin_atom, end_atom), dtype=int)
    begin = np.asarray(begin_offsets.offset(begin_atom), dtype=int)
    end = np.asarray(end_offsets.offset(end_atom), dtype=int)
    return tuple(int(value) for value in raw + begin - end)


class PeriodicCutSelector:
    """Select a canonical zero-holonomy opening for a periodic component."""

    MAX_CUT_SIZE = 6
    MAX_CANDIDATES = 250_000

    def __init__(
        self,
        atoms: Atoms,
        bonds: BondNetwork,
        symbols: list[str],
        records: tuple[PeriodicBond, ...],
    ) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.symbols = symbols
        self.records = records

    def select(
        self,
        atom_indices: set[int],
        fast: bool = False,
    ) -> PeriodicCutSelection:
        component_records = tuple(
            record
            for record in self.records
            if record.i in atom_indices and record.j in atom_indices
        )
        original = analyze_component(atom_indices, component_records)
        if original.rank == 0:
            return PeriodicCutSelection(())
        if fast or original.rank > 1:
            return self.select_fast(atom_indices, component_records)

        considered = 0
        for cut_size in range(1, min(self.MAX_CUT_SIZE, len(component_records)) + 1):
            valid = []
            for candidate in combinations(component_records, cut_size):
                considered += 1
                if considered > self.MAX_CANDIDATES:
                    raise ValueError(
                        "Periodic cut search exceeded its candidate bound."
                    )
                try:
                    residual = analyze_component(
                        atom_indices, component_records, frozenset(candidate)
                    )
                except _DisconnectedComponentError:
                    continue
                if residual.rank == 0:
                    valid.append((tuple(candidate), residual))
            if valid:
                cuts, residual = self.choose_canonical(
                    atom_indices, component_records, valid
                )
                return PeriodicCutSelection(cuts)

        raise ValueError("No connected zero-holonomy opening found.")

    def select_fast(
        self,
        atom_indices: set[int],
        component_records: tuple[PeriodicBond, ...],
    ) -> PeriodicCutSelection:
        parent = {index: index for index in atom_indices}
        sizes = {index: 1 for index in atom_indices}
        to_parent: dict[int, CellShift] = {index: (0, 0, 0) for index in atom_indices}

        def find(index: int) -> tuple[int, CellShift]:
            direct_parent = parent[index]
            if direct_parent == index:
                return index, (0, 0, 0)
            root, parent_offset = find(direct_parent)
            offset = tuple(a + b for a, b in zip(to_parent[index], parent_offset))
            parent[index] = root
            to_parent[index] = offset
            return root, offset

        def retention_key(record: PeriodicBond) -> tuple[object, ...]:
            length = round(float(np.linalg.norm(self.bonds.vector(*record.pair))), 8)
            reverse_shift = tuple(-value for value in record.shift)
            return (
                -self.bond_class(record),
                length,
                tuple(sorted((self.symbols[record.i], self.symbols[record.j]))),
                self.bonds.bond_type(*record.pair),
                min(record.shift, reverse_shift),
                record.pair,
            )

        cuts = []
        for record in sorted(component_records, key=retention_key):
            i_root, i_offset = find(record.i)
            j_root, j_offset = find(record.j)
            relative = tuple(b - a for a, b in zip(i_offset, j_offset))
            if i_root == j_root:
                if relative != record.shift:
                    cuts.append(record)
                continue

            if sizes[i_root] >= sizes[j_root]:
                parent[j_root] = i_root
                to_parent[j_root] = tuple(
                    shift + left - right
                    for shift, left, right in zip(record.shift, i_offset, j_offset)
                )
                sizes[i_root] += sizes[j_root]
            else:
                parent[i_root] = j_root
                to_parent[i_root] = tuple(
                    right - left - shift
                    for shift, left, right in zip(record.shift, i_offset, j_offset)
                )
                sizes[j_root] += sizes[i_root]

        selected = tuple(sorted(cuts))
        return PeriodicCutSelection(selected)

    def choose_canonical(
        self,
        atom_indices: set[int],
        records: tuple[PeriodicBond, ...],
        valid: list[tuple[tuple[PeriodicBond, ...], ComponentPeriodicity]],
    ) -> tuple[tuple[PeriodicBond, ...], ComponentPeriodicity]:
        return min(
            valid,
            key=lambda item: (
                self.candidate_score(atom_indices, records, *item),
                tuple(record.pair for record in item[0]),
            ),
        )

    def candidate_score(
        self,
        atom_indices: set[int],
        records: tuple[PeriodicBond, ...],
        cuts: tuple[PeriodicBond, ...],
        residual: ComponentPeriodicity,
    ) -> tuple[object, ...]:
        positions = np.asarray(
            [
                self.atoms.positions[index]
                + np.asarray(residual.offset(index)) @ self.atoms.cell.array
                for index in sorted(atom_indices)
            ]
        )
        centered = positions - positions.mean(axis=0)
        radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))

        cut_set = set(cuts)
        degree = dict.fromkeys(atom_indices, 0)
        for record in records:
            if record not in cut_set:
                degree[record.i] += 1
                degree[record.j] += 1
        signatures = tuple(
            sorted(
                (
                    tuple(sorted((self.symbols[record.i], self.symbols[record.j]))),
                    tuple(sorted((degree[record.i], degree[record.j]))),
                    self.bonds.bond_type(*record.pair),
                    round(float(np.linalg.norm(self.bonds.vector(*record.pair))), 6),
                )
                for record in cuts
            )
        )
        penalties = tuple(sorted(self.bond_class(record) for record in cuts))
        return (
            len(cuts),
            max(penalties),
            sum(penalties),
            penalties,
            round(radius, 8),
            signatures,
        )

    def bond_class(self, record: PeriodicBond) -> int:
        left, right = self.symbols[record.i], self.symbols[record.j]
        if "H" in {left, right}:
            return 4
        left_metal = is_mofid_metal(left)
        right_metal = is_mofid_metal(right)
        if left_metal and right_metal:
            return 0
        if left_metal != right_metal and ({left, right} & {"O", "N", "S"}):
            return 1
        if left_metal or right_metal:
            return 2
        return 3


def select_periodic_component_cuts(
    atoms: Atoms,
    bonds: BondNetwork,
    symbols: list[str],
    components: list[set[int]],
    records: tuple[PeriodicBond, ...],
    *,
    fast: bool = False,
) -> tuple[tuple[set[int], PeriodicCutSelection], ...]:
    selector = PeriodicCutSelector(atoms, bonds, symbols, records)
    return tuple(
        (component, selector.select(component, fast=fast)) for component in components
    )
