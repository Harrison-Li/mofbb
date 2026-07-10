from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

import ase
import networkx as nx
import numpy as np

from .bonding import BondNetwork
from .constants import is_mofid_metal

CellShift: TypeAlias = tuple[int, int, int]

_MAX_CUT_SIZE = 6
_MAX_CANDIDATES = 250_000
_BOND_CLASS_NAMES = (
    "metal-metal",
    "metal-coordination",
    "other-metal",
    "nonmetal-covalent",
    "hydrogen",
)


@dataclass(frozen=True, order=True)
class PeriodicBond:
    i: int
    j: int
    shift: CellShift

    def __post_init__(self) -> None:
        if self.i >= self.j:
            raise ValueError("PeriodicBond endpoints must satisfy i < j.")

    @property
    def pair(self) -> tuple[int, int]:
        return (self.i, self.j)

    def oriented_shift(self, start: int, end: int) -> CellShift:
        if (start, end) == (self.i, self.j):
            return self.shift
        if (start, end) == (self.j, self.i):
            return tuple(-value for value in self.shift)
        raise ValueError(f"Bond {self.pair} does not connect {(start, end)}.")

    def as_dict(self) -> dict[str, object]:
        return {"source_cut_bond": list(self.pair), "shift": list(self.shift)}


@dataclass(frozen=True)
class ComponentPeriodicity:
    offsets: tuple[tuple[int, CellShift], ...]
    generators: tuple[CellShift, ...]
    rank: int

    def offset(self, atom_index: int) -> CellShift:
        return dict(self.offsets)[atom_index]


@dataclass(frozen=True)
class PeriodicCutDetail:
    record: PeriodicBond
    element_pair: tuple[str, str]
    bond_class: str
    bond_class_penalty: int
    bond_type: str
    length: float

    def as_dict(self) -> dict[str, object]:
        return {
            **self.record.as_dict(),
            "element_pair": list(self.element_pair),
            "bond_class": self.bond_class,
            "bond_class_penalty": self.bond_class_penalty,
            "bond_type": self.bond_type,
            "length": self.length,
        }


@dataclass(frozen=True)
class PeriodicCutSelection:
    cuts: tuple[PeriodicBond, ...]
    original: ComponentPeriodicity
    residual: ComponentPeriodicity
    candidates_considered: int
    cut_details: tuple[PeriodicCutDetail, ...] = ()


@dataclass(frozen=True)
class PeriodicOpening:
    source_indices: tuple[int, ...]
    rank: int
    generators: tuple[CellShift, ...]
    cuts: tuple[PeriodicBond, ...]
    candidates_considered: int
    cut_details: tuple[PeriodicCutDetail, ...] = ()

    @classmethod
    def from_selection(
        cls,
        atom_indices: set[int],
        selection: PeriodicCutSelection,
    ) -> "PeriodicOpening":
        return cls(
            source_indices=tuple(sorted(atom_indices)),
            rank=selection.original.rank,
            generators=selection.original.generators,
            cuts=selection.cuts,
            candidates_considered=selection.candidates_considered,
            cut_details=selection.cut_details,
        )

    def as_dict(self) -> dict[str, object]:
        detail_by_record = {detail.record: detail for detail in self.cut_details}
        return {
            "source_indices": list(self.source_indices),
            "rank": self.rank,
            "generators": [list(generator) for generator in self.generators],
            "cuts": [
                detail_by_record[record].as_dict()
                if record in detail_by_record
                else record.as_dict()
                for record in self.cuts
            ],
            "candidates_considered": self.candidates_considered,
        }


class _DisconnectedComponentError(ValueError):
    """Raised when excluded bonds split a requested component."""


def _component_formula(symbols: list[str], atom_indices: set[int]) -> str:
    return ase.Atoms(
        symbols=[symbols[index] for index in sorted(atom_indices)]
    ).get_chemical_formula()


def _bond_class(symbols: list[str], record: PeriodicBond) -> int:
    left, right = symbols[record.i], symbols[record.j]
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


def _cut_detail(
    symbols: list[str],
    bonds: BondNetwork,
    record: PeriodicBond,
) -> PeriodicCutDetail:
    class_index = _bond_class(symbols, record)
    return PeriodicCutDetail(
        record=record,
        element_pair=tuple(sorted((symbols[record.i], symbols[record.j]))),
        bond_class=_BOND_CLASS_NAMES[class_index],
        bond_class_penalty=class_index,
        bond_type=bonds.bond_type(*record.pair),
        length=float(np.linalg.norm(bonds.vector(*record.pair))),
    )


def _marked_graph(
    symbols: list[str],
    atom_indices: set[int],
    records: tuple[PeriodicBond, ...],
    cuts: tuple[PeriodicBond, ...],
    bonds: BondNetwork,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for index in atom_indices:
        graph.add_node(index, symbol=symbols[index])
    cut_set = set(cuts)
    for record in records:
        if record.i not in atom_indices or record.j not in atom_indices:
            continue
        bond_type = bonds.bond_type(record.i, record.j)
        length = round(float(np.linalg.norm(bonds.vector(*record.pair))), 6)
        is_cut = record in cut_set
        for start, end in (record.pair, tuple(reversed(record.pair))):
            label = (
                bond_type,
                is_cut,
                record.oriented_shift(start, end),
                length,
            )
            graph.add_edge(start, end, label=label)
    return graph


def _candidate_score(
    atoms: ase.Atoms,
    bonds: BondNetwork,
    symbols: list[str],
    atom_indices: set[int],
    records: tuple[PeriodicBond, ...],
    cuts: tuple[PeriodicBond, ...],
    residual: ComponentPeriodicity,
) -> tuple[object, ...]:
    positions = np.asarray(
        [
            atoms.positions[index]
            + np.asarray(residual.offset(index)) @ atoms.cell.array
            for index in sorted(atom_indices)
        ]
    )
    centered = positions - positions.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    cut_set = set(cuts)
    residual_degree = {
        index: sum(
            1
            for record in records
            if record not in cut_set and index in record.pair
        )
        for index in atom_indices
    }
    signatures = tuple(
        sorted(
            (
                tuple(sorted((symbols[record.i], symbols[record.j]))),
                tuple(sorted((residual_degree[record.i], residual_degree[record.j]))),
                bonds.bond_type(record.i, record.j),
                round(float(np.linalg.norm(bonds.vector(*record.pair))), 6),
            )
            for record in cuts
        )
    )
    marked = _marked_graph(symbols, atom_indices, records, cuts, bonds)
    graph_hash = nx.weisfeiler_lehman_graph_hash(
        marked, node_attr="symbol", edge_attr="label"
    )
    penalties = tuple(sorted(_bond_class(symbols, record) for record in cuts))
    return (
        len(cuts),
        max(penalties),
        sum(penalties),
        penalties,
        round(radius, 8),
        signatures,
        graph_hash,
    )


def _candidate_description(
    symbols: list[str],
    bonds: BondNetwork,
    cuts: tuple[PeriodicBond, ...],
) -> str:
    descriptions = []
    for record in cuts:
        class_index = _bond_class(symbols, record)
        element_pair = "-".join(sorted((symbols[record.i], symbols[record.j])))
        length = float(np.linalg.norm(bonds.vector(*record.pair)))
        descriptions.append(
            f"class={_BOND_CLASS_NAMES[class_index]}, "
            f"elements={element_pair}, length={length:.6f}, "
            f"bond_type={bonds.bond_type(*record.pair)}, shift={record.shift}"
        )
    return "[" + "; ".join(descriptions) + "]"


def _select_canonical_candidate(
    atoms: ase.Atoms,
    bonds: BondNetwork,
    symbols: list[str],
    atom_indices: set[int],
    records: tuple[PeriodicBond, ...],
    valid: list[tuple[tuple[PeriodicBond, ...], ComponentPeriodicity]],
) -> tuple[tuple[PeriodicBond, ...], ComponentPeriodicity]:
    scored = [
        (
            _candidate_score(
                atoms, bonds, symbols, atom_indices, records, cuts, residual
            ),
            cuts,
            residual,
        )
        for cuts, residual in valid
    ]
    best_score = min(score for score, _, _ in scored)
    best = [(cuts, residual) for score, cuts, residual in scored if score == best_score]
    if len(best) > 1:
        reference = _marked_graph(
            symbols, atom_indices, records, best[0][0], bonds
        )
        node_match = nx.algorithms.isomorphism.categorical_node_match("symbol", None)
        edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
        candidates_are_equivalent = all(
            nx.is_isomorphic(
                reference,
                _marked_graph(symbols, atom_indices, records, cuts, bonds),
                node_match=node_match,
                edge_match=edge_match,
            )
            for cuts, _ in best[1:]
        )
        if not candidates_are_equivalent:
            details = "; ".join(
                f"candidate {index}: {_candidate_description(symbols, bonds, cuts)}"
                for index, (cuts, _) in enumerate(best, start=1)
            )
            formula = _component_formula(symbols, atom_indices)
            raise ValueError(
                f"ambiguous periodic cut for component {formula}: top-ranked "
                "candidates are not chemically/geometrically equivalent; "
                f"candidate bond details: {details}"
            )
    return min(best, key=lambda item: tuple(record.pair for record in item[0]))


def periodic_bonds(atoms: ase.Atoms, bonds: BondNetwork) -> tuple[PeriodicBond, ...]:
    periodic = np.asarray(atoms.pbc, dtype=bool)
    inverse = None
    if np.any(periodic):
        cell = np.asarray(atoms.cell.array, dtype=float)
        if np.linalg.matrix_rank(cell) < 3:
            raise ValueError(
                "Cannot resolve periodic translations from a singular periodic cell."
            )
        inverse = np.linalg.inv(cell)
    records = []
    for i, j in bonds.pairs:
        direct = np.asarray(atoms.positions[j] - atoms.positions[i], dtype=float)
        shift_cart = bonds.vector(i, j) - direct
        if inverse is None:
            if not np.allclose(shift_cart, 0.0, atol=1.0e-6):
                raise ValueError(
                    f"Bond {(i, j)} shifts along a non-periodic axis in a "
                    "non-periodic structure."
                )
            shift = np.zeros(3, dtype=int)
        else:
            fractional = shift_cart @ inverse
            rounded = np.rint(fractional)
            if not np.allclose(fractional, rounded, atol=1.0e-6):
                raise ValueError(
                    f"Bond {(i, j)} has a non-integral periodic translation: "
                    f"{fractional.tolist()}."
                )
            shift = rounded.astype(int)
        if np.any(shift[~periodic]):
            raise ValueError(f"Bond {(i, j)} shifts along a non-periodic axis.")
        records.append(PeriodicBond(i, j, tuple(int(value) for value in shift)))
    return tuple(records)


def analyze_component(
    atom_indices: set[int] | frozenset[int],
    records: tuple[PeriodicBond, ...],
    excluded: set[PeriodicBond] | frozenset[PeriodicBond] = frozenset(),
) -> ComponentPeriodicity:
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
    offsets: dict[int, np.ndarray] = {root: np.zeros(3, dtype=int)}
    generators: set[CellShift] = set()
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor, record in sorted(adjacency[current]):
            expected = offsets[current] + np.asarray(
                record.oriented_shift(current, neighbor), dtype=int
            )
            if neighbor not in offsets:
                offsets[neighbor] = expected
                queue.append(neighbor)
                continue
            residual = expected - offsets[neighbor]
            if np.any(residual):
                forward = tuple(int(value) for value in residual)
                reverse = tuple(-value for value in forward)
                generators.add(min(forward, reverse))

    if set(offsets) != selected:
        missing = sorted(selected - set(offsets))
        raise _DisconnectedComponentError(
            f"Periodic component is disconnected; missing atoms {missing}."
        )
    normalized = tuple(sorted(generators))
    rank = 0 if not normalized else int(np.linalg.matrix_rank(np.asarray(normalized)))
    return ComponentPeriodicity(
        offsets=tuple(
            (index, tuple(int(value) for value in offsets[index]))
            for index in sorted(offsets)
        ),
        generators=normalized,
        rank=rank,
    )


def connection_shift(
    record: PeriodicBond,
    begin_atom: int,
    begin_offsets: ComponentPeriodicity,
    end_atom: int,
    end_offsets: ComponentPeriodicity,
) -> CellShift:
    raw = np.asarray(record.oriented_shift(begin_atom, end_atom), dtype=int)
    begin = np.asarray(begin_offsets.offset(begin_atom), dtype=int)
    end = np.asarray(end_offsets.offset(end_atom), dtype=int)
    return tuple(int(value) for value in raw + begin - end)


def select_periodic_cuts(
    atoms: ase.Atoms,
    bonds: BondNetwork,
    symbols: list[str],
    atom_indices: set[int],
    records: tuple[PeriodicBond, ...],
) -> PeriodicCutSelection:
    component_records = tuple(
        record
        for record in records
        if record.i in atom_indices and record.j in atom_indices
    )
    original = analyze_component(atom_indices, component_records)
    if original.rank == 0:
        return PeriodicCutSelection((), original, original, 0)
    if original.rank != 1:
        formula = _component_formula(symbols, atom_indices)
        raise ValueError(
            f"Periodic metal component {formula} has unsupported rank "
            f"{original.rank}: generators={original.generators}."
        )

    formula = _component_formula(symbols, atom_indices)
    considered = 0
    for cut_size in range(1, min(_MAX_CUT_SIZE, len(component_records)) + 1):
        valid = []
        for candidate in combinations(component_records, cut_size):
            considered += 1
            if considered > _MAX_CANDIDATES:
                raise ValueError(
                    "Periodic cut search exceeded its candidate bound "
                    f"(bound={_MAX_CANDIDATES}, formula={formula}, "
                    f"component_size={len(atom_indices)}, "
                    f"bond_count={len(component_records)}, "
                    f"candidates_considered={considered})."
                )
            try:
                residual = analyze_component(
                    atom_indices, component_records, excluded=frozenset(candidate)
                )
            except _DisconnectedComponentError:
                continue
            if residual.rank == 0:
                valid.append((tuple(candidate), residual))
        if valid:
            cuts, residual = _select_canonical_candidate(
                atoms,
                bonds,
                symbols,
                atom_indices,
                component_records,
                valid,
            )
            details = tuple(_cut_detail(symbols, bonds, record) for record in cuts)
            return PeriodicCutSelection(
                cuts, original, residual, considered, cut_details=details
            )

    raise ValueError(
        "No connected zero-holonomy opening found "
        f"(max_cut_size={_MAX_CUT_SIZE}, formula={formula}, "
        f"component_size={len(atom_indices)}, "
        f"bond_count={len(component_records)})."
    )
