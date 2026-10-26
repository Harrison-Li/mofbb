"""Shared periodic-bond types, component rank, and image-offset analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
from typing import TypeAlias

import numpy as np

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
        raise ValueError(
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
