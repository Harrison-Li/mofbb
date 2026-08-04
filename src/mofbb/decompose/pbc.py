"""Periodic-boundary connectivity analysis for MOF decomposition."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import TypeAlias

from ase import Atoms
import networkx as nx
import numpy as np
import spglib

from .bonding import BondNetwork
from .constants import is_mofid_metal

CellShift: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True, order=True)
class PeriodicBond:
    """One bond and its integer unit-cell translation."""

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
class PeriodicGraphSymmetry:
    """Atom and periodic-bond permutations for one source symmetry operation."""

    atom_map: tuple[int, ...]
    bond_map: tuple[int, ...]


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
        cls, atom_indices: set[int], selection: PeriodicCutSelection
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
        details = {detail.record: detail for detail in self.cut_details}
        return {
            "source_indices": list(self.source_indices),
            "rank": self.rank,
            "generators": [list(generator) for generator in self.generators],
            "cuts": [
                details[record].as_dict() if record in details else record.as_dict()
                for record in self.cuts
            ],
            "candidates_considered": self.candidates_considered,
        }


class _DisconnectedComponentError(ValueError):
    """Raised when excluded bonds split a requested component."""


def periodic_bonds(atoms: Atoms, bonds: BondNetwork) -> tuple[PeriodicBond, ...]:
    """Return source bonds annotated with integer cell translations."""

    periodic_axes = np.asarray(atoms.pbc, dtype=bool)
    inverse_cell = None
    if np.any(periodic_axes):
        cell = np.asarray(atoms.cell.array, dtype=float)
        if np.linalg.matrix_rank(cell) < 3:
            raise ValueError(
                "Cannot resolve periodic translations from a singular periodic cell."
            )
        inverse_cell = np.linalg.inv(cell)

    records = []
    for i, j in bonds.pairs:
        direct = np.asarray(atoms.positions[j] - atoms.positions[i], dtype=float)
        shift_cart = bonds.vector(i, j) - direct
        if inverse_cell is None:
            if not np.allclose(shift_cart, 0.0, atol=1.0e-6):
                raise ValueError(
                    f"Bond {(i, j)} shifts along a non-periodic axis in a "
                    "non-periodic structure."
                )
            shift = np.zeros(3, dtype=int)
        else:
            fractional = shift_cart @ inverse_cell
            rounded = np.rint(fractional)
            if not np.allclose(fractional, rounded, atol=1.0e-6):
                raise ValueError(
                    f"Bond {(i, j)} has a non-integral periodic translation: "
                    f"{fractional.tolist()}."
                )
            shift = rounded.astype(int)
        if np.any(shift[~periodic_axes]):
            raise ValueError(f"Bond {(i, j)} shifts along a non-periodic axis.")
        records.append(PeriodicBond(i, j, tuple(int(value) for value in shift)))
    return tuple(records)


def periodic_graph_symmetries(
    atoms: Atoms,
    bonds: BondNetwork,
    records: tuple[PeriodicBond, ...],
    symprec: float = 0.05,
) -> tuple[PeriodicGraphSymmetry, ...]:
    """Return source-crystal operations that also preserve the periodic bond graph."""

    cell = np.asarray(atoms.cell.array, dtype=float)
    raw_positions = atoms.get_scaled_positions(wrap=False)
    wrapped_positions = atoms.get_scaled_positions(wrap=True)
    numbers = np.asarray(atoms.numbers)
    dataset = spglib.get_symmetry_dataset(
        (cell, wrapped_positions, numbers),
        symprec=symprec,
        angle_tolerance=5.0,
    )
    identity = PeriodicGraphSymmetry(
        tuple(range(len(atoms))),
        tuple(range(len(records))),
    )
    if dataset is None:
        return (identity,)

    record_indices = {record: index for index, record in enumerate(records)}
    symmetries = []
    seen = set()
    for rotation, translation in zip(dataset.rotations, dataset.translations):
        transformed = raw_positions @ rotation.T + translation
        atom_map = np.empty(len(atoms), dtype=int)
        atom_images = np.empty((len(atoms), 3), dtype=int)
        valid = True
        for number in np.unique(numbers):
            indices = np.flatnonzero(numbers == number)
            differences = transformed[indices, None] - raw_positions[indices]
            images = np.rint(differences)
            distances = np.linalg.norm((differences - images) @ cell, axis=2)
            nearest = np.argmin(distances, axis=1)
            if (
                len(set(nearest.tolist())) != len(indices)
                or np.max(distances[np.arange(len(indices)), nearest]) > symprec
            ):
                valid = False
                break
            atom_map[indices] = indices[nearest]
            atom_images[indices] = images[np.arange(len(indices)), nearest]
        if not valid:
            continue

        bond_map = []
        for record in records:
            left, right = atom_map[record.i], atom_map[record.j]
            shift = (
                rotation @ np.asarray(record.shift)
                + atom_images[record.j]
                - atom_images[record.i]
            )
            if left > right:
                left, right = right, left
                shift = -shift
            target = PeriodicBond(
                int(left),
                int(right),
                tuple(int(value) for value in shift),
            )
            if (
                target not in record_indices
                or bonds.bond_type(*target.pair)
                != bonds.bond_type(*record.pair)
            ):
                valid = False
                break
            bond_map.append(record_indices[target])
        key = (tuple(atom_map), tuple(bond_map))
        if valid and key not in seen:
            seen.add(key)
            symmetries.append(PeriodicGraphSymmetry(*key))
    return tuple(symmetries) or (identity,)


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
    BOND_CLASS_NAMES = (
        "metal-metal",
        "metal-coordination",
        "other-metal",
        "nonmetal-covalent",
        "hydrogen",
    )

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

    def select(self, atom_indices: set[int], fast: bool = False) -> PeriodicCutSelection:
        component_records = tuple(
            record
            for record in self.records
            if record.i in atom_indices and record.j in atom_indices
        )
        original = analyze_component(atom_indices, component_records)
        if original.rank == 0:
            return PeriodicCutSelection((), original, original, 0)
        if fast or original.rank > 1:
            return self.select_fast(atom_indices, component_records, original)

        formula = self.component_formula(atom_indices)
        considered = 0
        for cut_size in range(1, min(self.MAX_CUT_SIZE, len(component_records)) + 1):
            valid = []
            for candidate in combinations(component_records, cut_size):
                considered += 1
                if considered > self.MAX_CANDIDATES:
                    raise ValueError(
                        "Periodic cut search exceeded its candidate bound "
                        f"(bound={self.MAX_CANDIDATES}, formula={formula}, "
                        f"component_size={len(atom_indices)}, "
                        f"bond_count={len(component_records)}, "
                        f"candidates_considered={considered})."
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
                return PeriodicCutSelection(
                    cuts,
                    original,
                    residual,
                    considered,
                    tuple(self.cut_detail(record) for record in cuts),
                )

        raise ValueError(
            "No connected zero-holonomy opening found "
            f"(max_cut_size={self.MAX_CUT_SIZE}, formula={formula}, "
            f"component_size={len(atom_indices)}, "
            f"bond_count={len(component_records)})."
        )

    def select_symmetry_preserving(
        self,
        atom_indices: set[int],
        bond_maps: tuple[tuple[int, ...], ...],
        fast: bool = False,
        baseline: PeriodicCutSelection | None = None,
    ) -> PeriodicCutSelection:
        """Select an opening invariant under the component stabilizer."""

        if baseline is None:
            baseline = self.select(atom_indices, fast=fast)
        if not baseline.cuts:
            return baseline

        record_indices = {record: index for index, record in enumerate(self.records)}
        component_records = tuple(
            record
            for record in self.records
            if record.i in atom_indices and record.j in atom_indices
        )
        component_indices = {record_indices[record] for record in component_records}
        cut_indices = {record_indices[record] for record in baseline.cuts}
        if all(
            {mapping[index] for index in cut_indices} == cut_indices
            for mapping in bond_maps
        ):
            return baseline

        unseen = set(component_indices)
        orbits = []
        while unseen:
            orbit = {min(unseen)}
            while True:
                expanded = orbit | {
                    mapping[index] for mapping in bond_maps for index in orbit
                }
                if expanded == orbit:
                    break
                orbit = expanded
            if not orbit <= component_indices:
                raise ValueError(
                    "A component stabilizer maps a bond outside its component."
                )
            orbits.append(orbit)
            unseen -= orbit

        considered = 0
        valid = []
        cut_size = len(baseline.cuts)
        for orbit_count in range(1, min(cut_size, len(orbits)) + 1):
            for selected_orbits in combinations(orbits, orbit_count):
                candidate_indices = set().union(*selected_orbits)
                if len(candidate_indices) != cut_size:
                    continue
                considered += 1
                if considered > self.MAX_CANDIDATES:
                    return baseline
                cuts = tuple(
                    self.records[index] for index in sorted(candidate_indices)
                )
                try:
                    residual = analyze_component(
                        atom_indices, component_records, frozenset(cuts)
                    )
                except _DisconnectedComponentError:
                    continue
                if residual.rank == 0:
                    valid.append((cuts, residual))
        if valid:
            cuts, residual = self.choose_canonical(
                atom_indices, component_records, valid
            )
            return PeriodicCutSelection(
                cuts,
                baseline.original,
                residual,
                considered,
                tuple(self.cut_detail(record) for record in cuts),
            )

        return baseline

    def selection_from_cuts(
        self,
        atom_indices: set[int],
        cuts: tuple[PeriodicBond, ...],
        candidates_considered: int,
    ) -> PeriodicCutSelection:
        """Build and validate a selection mapped from an equivalent component."""

        component_records = tuple(
            record
            for record in self.records
            if record.i in atom_indices and record.j in atom_indices
        )
        original = analyze_component(atom_indices, component_records)
        residual = analyze_component(
            atom_indices, component_records, frozenset(cuts)
        )
        if residual.rank != 0:
            raise ValueError("A symmetry-mapped periodic opening has nonzero holonomy.")
        return PeriodicCutSelection(
            cuts,
            original,
            residual,
            candidates_considered,
            tuple(self.cut_detail(record) for record in cuts),
        )

    def select_fast(
        self,
        atom_indices: set[int],
        component_records: tuple[PeriodicBond, ...],
        original: ComponentPeriodicity,
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
        residual = analyze_component(atom_indices, component_records, frozenset(selected))
        if residual.rank != 0:
            raise ValueError(
                "Fast periodic opening failed to produce a zero-holonomy residual "
                f"(formula={self.component_formula(atom_indices)}, "
                f"rank={residual.rank}, generators={residual.generators})."
            )
        return PeriodicCutSelection(
            selected,
            original,
            residual,
            len(component_records),
            tuple(self.cut_detail(record) for record in selected),
        )

    def choose_canonical(
        self,
        atom_indices: set[int],
        records: tuple[PeriodicBond, ...],
        valid: list[tuple[tuple[PeriodicBond, ...], ComponentPeriodicity]],
    ) -> tuple[tuple[PeriodicBond, ...], ComponentPeriodicity]:
        scored = [
            (self.candidate_score(atom_indices, records, cuts, residual), cuts, residual)
            for cuts, residual in valid
        ]
        best_score = min(score for score, _, _ in scored)
        best = [(cuts, residual) for score, cuts, residual in scored if score == best_score]
        if len(best) == 1:
            return best[0]

        reference = self.marked_graph(atom_indices, records, best[0][0])
        node_match = nx.algorithms.isomorphism.categorical_node_match("symbol", None)
        edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
        equivalent = all(
            nx.is_isomorphic(
                reference,
                self.marked_graph(atom_indices, records, cuts),
                node_match=node_match,
                edge_match=edge_match,
            )
            for cuts, _ in best[1:]
        )
        if not equivalent:
            candidates = []
            for index, (cuts, _) in enumerate(best, start=1):
                details = []
                for record in cuts:
                    class_index = self.bond_class(record)
                    pair = "-".join(
                        sorted((self.symbols[record.i], self.symbols[record.j]))
                    )
                    length = np.linalg.norm(self.bonds.vector(*record.pair))
                    details.append(
                        f"class={self.BOND_CLASS_NAMES[class_index]}, "
                        f"elements={pair}, length={length:.6f}, "
                        f"bond_type={self.bonds.bond_type(*record.pair)}, "
                        f"shift={record.shift}"
                    )
                candidates.append(f"candidate {index}: [" + "; ".join(details) + "]")
            raise ValueError(
                f"ambiguous periodic cut for component "
                f"{self.component_formula(atom_indices)}: top-ranked candidates "
                "are not chemically/geometrically equivalent; candidate bond "
                f"details: {'; '.join(candidates)}"
            )
        return min(best, key=lambda item: tuple(record.pair for record in item[0]))

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
        graph_hash = nx.weisfeiler_lehman_graph_hash(
            self.marked_graph(atom_indices, records, cuts),
            node_attr="symbol",
            edge_attr="label",
        )
        penalties = tuple(sorted(self.bond_class(record) for record in cuts))
        return (
            len(cuts),
            max(penalties),
            sum(penalties),
            penalties,
            round(radius, 8),
            signatures,
            graph_hash,
        )

    def marked_graph(
        self,
        atom_indices: set[int],
        records: tuple[PeriodicBond, ...],
        cuts: tuple[PeriodicBond, ...],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        for index in atom_indices:
            graph.add_node(index, symbol=self.symbols[index])
        cut_set = set(cuts)
        for record in records:
            length = round(float(np.linalg.norm(self.bonds.vector(*record.pair))), 6)
            for start, end in (record.pair, tuple(reversed(record.pair))):
                graph.add_edge(
                    start,
                    end,
                    label=(
                        self.bonds.bond_type(*record.pair),
                        record in cut_set,
                        record.oriented_shift(start, end),
                        length,
                    ),
                )
        return graph

    def component_formula(self, atom_indices: set[int]) -> str:
        return Atoms(
            symbols=[self.symbols[index] for index in sorted(atom_indices)]
        ).get_chemical_formula()

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

    def cut_detail(self, record: PeriodicBond) -> PeriodicCutDetail:
        class_index = self.bond_class(record)
        return PeriodicCutDetail(
            record=record,
            element_pair=tuple(
                sorted((self.symbols[record.i], self.symbols[record.j]))
            ),
            bond_class=self.BOND_CLASS_NAMES[class_index],
            bond_class_penalty=class_index,
            bond_type=self.bonds.bond_type(*record.pair),
            length=float(np.linalg.norm(self.bonds.vector(*record.pair))),
        )


def select_periodic_cuts(
    atoms: Atoms,
    bonds: BondNetwork,
    symbols: list[str],
    atom_indices: set[int],
    records: tuple[PeriodicBond, ...],
    *,
    fast: bool = False,
) -> PeriodicCutSelection:
    """Compatibility wrapper around :class:`PeriodicCutSelector`."""

    return PeriodicCutSelector(atoms, bonds, symbols, records).select(
        atom_indices, fast=fast
    )


def select_periodic_component_cuts(
    atoms: Atoms,
    bonds: BondNetwork,
    symbols: list[str],
    components: list[set[int]],
    records: tuple[PeriodicBond, ...],
    symmetries: tuple[PeriodicGraphSymmetry, ...],
    *,
    fast: bool = False,
) -> tuple[tuple[set[int], PeriodicCutSelection], ...]:
    """Select covariant cuts jointly for symmetry-equivalent components."""

    selector = PeriodicCutSelector(atoms, bonds, symbols, records)
    component_keys = {frozenset(component) for component in components}
    record_indices = {record: index for index, record in enumerate(records)}
    selected: dict[frozenset[int], PeriodicCutSelection] = {}

    for component in components:
        key = frozenset(component)
        if key in selected:
            continue
        stabilizer = tuple(
            symmetry.bond_map
            for symmetry in symmetries
            if frozenset(symmetry.atom_map[index] for index in component) == key
        )
        baseline = selector.select(component, fast=fast)
        # A rank-one periodic chain need not admit a connected invariant cut.
        selection = (
            selector.select_symmetry_preserving(
                component,
                stabilizer,
                fast=fast,
                baseline=baseline,
            )
            if baseline.original.rank > 1
            else baseline
        )
        selected[key] = selection
        cut_indices = {record_indices[record] for record in selection.cuts}
        if selection.original.rank < 2 or any(
            {mapping[index] for index in cut_indices} != cut_indices
            for mapping in stabilizer
        ):
            continue
        for symmetry in symmetries:
            target = frozenset(symmetry.atom_map[index] for index in component)
            if target == key or target not in component_keys or target in selected:
                continue
            mapped_cuts = tuple(
                records[symmetry.bond_map[record_indices[record]]]
                for record in selection.cuts
            )
            selected[target] = selector.selection_from_cuts(
                set(target),
                tuple(sorted(mapped_cuts)),
                selection.candidates_considered,
            )

    return tuple(
        (component, selected[frozenset(component)]) for component in components
    )
