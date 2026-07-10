# Periodic Node/Edge Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit finite, zero-holonomy rod-repeat building blocks with explicit translated self-connections, eliminate CPO-27 singleton phenolate fragments, and retain both physical X-site reconstruction and logical multidentate port metadata.

**Architecture:** Add a focused periodic-voltage-graph module that reconstructs integer bond shifts, detects component rank, and selects a bounded minimum cut set for rank-one nodes. Feed canonical `PeriodicBond` records through fragment construction, topology, results, and reconstruction; keep pair-only fields as compatibility views. Extend only the organic splitter with a narrow pendant-donor absorption rule and compute logical ports after all physical fragments exist.

**Tech Stack:** Python 3.10+, ASE, NumPy, NetworkX, pytest, existing MOFBB dataclasses and JSON/CGD writers.

## Global Constraints

- Rank-zero HKUST1, MOF5, and UiO-67 formulas, counts, physical X sites, and reconstruction validity must remain unchanged.
- Preserve one physical X site per cut-bond side; never collapse multiple coordination bonds into one reconstructable X atom.
- Keep existing `connecting_bonds` and `source_cut_bond` pair fields; all shift-bearing fields are additive.
- Automatically open only rank-one periodic metal components. Reject rank-two/rank-three components with a descriptive error.
- Do not infer screw-related Zn2 units or reduce hidden primitive translations in this change.
- The selected periodic cut set must leave one connected, zero-holonomy residual component and must not depend on raw atom indices as a chemical tie-breaker.
- Every selected source atom has exactly one owning fragment; every physical cut has exactly two X sides; every X side is consumed once.
- Implement every behavior test-first and keep edits scoped to periodic decomposition, topology/reconstruction, and CPO pendant-donor absorption.

---

### Task 1: Canonical periodic bonds and component-rank analysis

**Files:**
- Create: `src/mofbb/periodicity.py`
- Create: `tests/test_periodicity.py`
- Modify: `src/mofbb/solvent.py:400-485`

**Interfaces:**
- Consumes: `ase.Atoms`, `mofbb.bonding.BondNetwork`.
- Produces: `CellShift`, `PeriodicBond`, `ComponentPeriodicity`, `periodic_bonds()`, `analyze_component()`, and `connection_shift()`.

- [ ] **Step 1: Write failing tests for canonical shifts, reverse traversal, rank zero, and rank one**

Add this fixture and tests to `tests/test_periodicity.py`:

```python
import numpy as np
from ase import Atoms

from mofbb.bonding import BondNetwork
from mofbb.periodicity import (
    PeriodicBond,
    analyze_component,
    periodic_bonds,
)


def make_bonds(atoms, pairs_with_shifts):
    adjacency = [set() for _ in atoms]
    vectors = {}
    for i, j, shift in pairs_with_shifts:
        a, b = sorted((i, j))
        oriented = np.asarray(shift if i == a else tuple(-v for v in shift))
        adjacency[a].add(b)
        adjacency[b].add(a)
        vectors[(a, b)] = (
            atoms.positions[b]
            + oriented @ atoms.cell.array
            - atoms.positions[a]
        )
    return BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )


def test_periodic_bond_shift_is_canonical_and_reversible():
    atoms = Atoms("CC", positions=[(1, 0, 0), (9, 0, 0)], cell=[10, 10, 10], pbc=True)
    bonds = make_bonds(atoms, [(0, 1, (-1, 0, 0))])

    record = periodic_bonds(atoms, bonds)[0]

    assert record == PeriodicBond(0, 1, (-1, 0, 0))
    assert record.oriented_shift(0, 1) == (-1, 0, 0)
    assert record.oriented_shift(1, 0) == (1, 0, 0)


def test_component_analysis_detects_nonzero_cycle_rank():
    atoms = Atoms("CCC", positions=[(1, 0, 0), (4, 0, 0), (8, 0, 0)], cell=[10, 10, 10], pbc=True)
    bonds = make_bonds(
        atoms,
        [(0, 1, (0, 0, 0)), (1, 2, (0, 0, 0)), (0, 2, (-1, 0, 0))],
    )
    records = periodic_bonds(atoms, bonds)

    periodic = analyze_component({0, 1, 2}, records)
    finite = analyze_component({0, 1, 2}, records, excluded={records[2]})

    assert periodic.rank == 1
    assert periodic.generators == ((-1, 0, 0),)
    assert finite.rank == 0
    assert finite.generators == ()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_periodicity.py
```

Expected: collection fails because `mofbb.periodicity` does not exist.

- [ ] **Step 3: Implement the periodic graph primitives**

Create `src/mofbb/periodicity.py` with these public types and signatures:

```python
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias

import ase
import numpy as np

from .bonding import BondNetwork

CellShift: TypeAlias = tuple[int, int, int]


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


def periodic_bonds(atoms: ase.Atoms, bonds: BondNetwork) -> tuple[PeriodicBond, ...]:
    periodic = np.asarray(atoms.pbc, dtype=bool)
    inverse = None
    if np.any(periodic):
        cell = np.asarray(atoms.cell.array, dtype=float)
        if np.linalg.matrix_rank(cell) < 3:
            raise ValueError("Cannot resolve periodic translations from a singular cell.")
        inverse = np.linalg.inv(cell)
    records = []
    for i, j in bonds.pairs:
        direct = np.asarray(atoms.positions[j] - atoms.positions[i], dtype=float)
        shift_cart = bonds.vector(i, j) - direct
        if inverse is None:
            if not np.allclose(shift_cart, 0.0, atol=1.0e-6):
                raise ValueError(f"Bond {(i, j)} shifts in a non-periodic structure.")
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
        raise ValueError(f"Periodic component is disconnected; missing atoms {missing}.")
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
```

Replace the ellipsis in `analyze_component` with the BFS described in the
docstring comment. Normalize each residual generator by choosing the
lexicographically smaller of `shift` and `-shift`; calculate rank with
`np.linalg.matrix_rank`, and return offsets sorted by atom index.

- [ ] **Step 4: Reuse `periodic_bonds()` in solvent translation analysis**

In `MofidSolventRemover.__init__`, cache records by pair:

```python
self._periodic_bond_by_pair = {
    record.pair: record for record in periodic_bonds(atoms, bonds)
}
```

Replace `_bond_shift()` arithmetic with:

```python
def _bond_shift(self, i: int, j: int) -> np.ndarray:
    pair = tuple(sorted((i, j)))
    return np.asarray(
        self._periodic_bond_by_pair[pair].oriented_shift(i, j),
        dtype=int,
    )
```

- [ ] **Step 5: Run focused and solvent tests and verify GREEN**

Run:

```bash
pytest -q tests/test_periodicity.py tests/test_solvent.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/mofbb/periodicity.py src/mofbb/solvent.py tests/test_periodicity.py
git commit -m "feat: analyze periodic bond graph rank"
```

---

### Task 2: Bounded minimum cut selection for rank-one components

**Files:**
- Modify: `src/mofbb/periodicity.py`
- Modify: `tests/test_periodicity.py`

**Interfaces:**
- Consumes: `PeriodicBond`, `ComponentPeriodicity`, atom symbols and source bond types.
- Produces: `PeriodicCutSelection`, `PeriodicOpening`, and `select_periodic_cuts()`.

- [ ] **Step 1: Add failing tests for a two-cut rod, rank-two rejection, and cut determinism**

Append:

```python
import pytest

import mofbb.periodicity as periodicity_module
from mofbb.periodicity import ComponentPeriodicity
from mofbb.periodicity import select_periodic_cuts


def test_selector_finds_two_cuts_for_two_edge_disjoint_periodic_cycles():
    atoms = Atoms(
        ["Zn", "O", "C", "O", "C"],
        positions=[(1, 1, 1), (3, 1, 1), (5, 1, 1), (1, 3, 1), (1, 5, 1)],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = make_bonds(
        atoms,
        [
            (0, 1, (0, 0, 0)),
            (1, 2, (0, 0, 0)),
            (0, 2, (-1, 0, 0)),
            (0, 3, (0, 0, 0)),
            (3, 4, (0, 0, 0)),
            (0, 4, (-1, 0, 0)),
        ],
    )
    records = periodic_bonds(atoms, bonds)

    selection = select_periodic_cuts(
        atoms, bonds, atoms.get_chemical_symbols(), set(range(5)), records
    )

    assert len(selection.cuts) == 2
    residual = analyze_component(set(range(5)), records, set(selection.cuts))
    assert residual.rank == 0


def test_selector_rejects_rank_two_component():
    atoms = Atoms("CCCC", positions=[(1, 1, 1), (3, 1, 1), (1, 3, 1), (3, 3, 1)], cell=[10, 10, 10], pbc=True)
    bonds = make_bonds(
        atoms,
        [
            (0, 1, (0, 0, 0)),
            (1, 3, (0, 0, 0)),
            (0, 3, (-1, 0, 0)),
            (0, 2, (0, 0, 0)),
            (2, 3, (0, 0, 0)),
            (0, 3, (0, -1, 0)),
        ],
    )

    with pytest.raises(ValueError, match="rank 2"):
        select_periodic_cuts(
            atoms,
            bonds,
            atoms.get_chemical_symbols(),
            set(range(4)),
            periodic_bonds(atoms, bonds),
        )


def test_canonical_selector_rejects_inequivalent_tied_cuts(monkeypatch):
    atoms = Atoms(
        ["C", "O", "N"],
        positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        cell=[10, 10, 10],
        pbc=False,
    )
    bonds = make_bonds(
        atoms,
        [(0, 1, (0, 0, 0)), (1, 2, (0, 0, 0))],
    )
    records = periodic_bonds(atoms, bonds)
    finite = ComponentPeriodicity(
        offsets=((0, (0, 0, 0)), (1, (0, 0, 0)), (2, (0, 0, 0))),
        generators=(),
        rank=0,
    )
    monkeypatch.setattr(
        periodicity_module,
        "_candidate_score",
        lambda *args, **kwargs: (0,),
    )

    with pytest.raises(ValueError, match="ambiguous periodic cut"):
        periodicity_module._select_canonical_candidate(
            atoms,
            bonds,
            atoms.get_chemical_symbols(),
            {0, 1, 2},
            [((records[0],), finite), ((records[1],), finite)],
        )
```

- [ ] **Step 2: Run tests and verify RED**

Run `pytest -q tests/test_periodicity.py`.

Expected: import fails because `select_periodic_cuts` is missing.

- [ ] **Step 3: Implement bounded selection and explicit diagnostics**

Add:

```python
from itertools import combinations

from .constants import is_mofid_metal


@dataclass(frozen=True)
class PeriodicCutSelection:
    cuts: tuple[PeriodicBond, ...]
    original: ComponentPeriodicity
    residual: ComponentPeriodicity
    candidates_considered: int


@dataclass(frozen=True)
class PeriodicOpening:
    source_indices: tuple[int, ...]
    rank: int
    generators: tuple[CellShift, ...]
    cuts: tuple[PeriodicBond, ...]
    candidates_considered: int

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
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_indices": list(self.source_indices),
            "rank": self.rank,
            "generators": [list(generator) for generator in self.generators],
            "cuts": [record.as_dict() for record in self.cuts],
            "candidates_considered": self.candidates_considered,
        }


def select_periodic_cuts(
    atoms: ase.Atoms,
    bonds: BondNetwork,
    symbols: list[str],
    atom_indices: set[int],
    records: tuple[PeriodicBond, ...],
    *,
    max_cut_size: int = 6,
    max_candidates: int = 250_000,
) -> PeriodicCutSelection:
    original = analyze_component(atom_indices, records)
    if original.rank == 0:
        return PeriodicCutSelection((), original, original, 0)
    if original.rank != 1:
        raise ValueError(
            f"Periodic metal component has unsupported rank {original.rank}: "
            f"generators={original.generators}."
        )

    component_records = tuple(
        record for record in records
        if record.i in atom_indices and record.j in atom_indices
    )
    considered = 0
    for cut_size in range(1, min(max_cut_size, len(component_records)) + 1):
        valid = []
        for candidate in combinations(component_records, cut_size):
            considered += 1
            if considered > max_candidates:
                raise ValueError(
                    "Periodic cut search exceeded 250000 candidates for "
                    f"component rooted at atom {min(atom_indices)}."
                )
            try:
                residual = analyze_component(
                    atom_indices, records, excluded=frozenset(candidate)
                )
            except ValueError as error:
                if "disconnected" in str(error):
                    continue
                raise
            if residual.rank == 0:
                valid.append((tuple(candidate), residual))
        if valid:
            cuts, residual = _select_canonical_candidate(
                atoms, bonds, symbols, atom_indices, valid
            )
            return PeriodicCutSelection(cuts, original, residual, considered)

    raise ValueError(
        f"No connected zero-holonomy opening found with at most {max_cut_size} "
        f"cuts for component rooted at atom {min(atom_indices)}."
    )
```

Import `networkx as nx` and add the canonical scorer below. The final pair-order
choice is used only after marked-graph isomorphism proves all remaining choices
chemically equivalent.

```python
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


def _marked_graph(
    symbols: list[str],
    atom_indices: set[int],
    records: tuple[PeriodicBond, ...],
    cuts: tuple[PeriodicBond, ...],
    bonds: BondNetwork,
) -> nx.Graph:
    graph = nx.Graph()
    for index in atom_indices:
        graph.add_node(index, symbol=symbols[index])
    cut_set = set(cuts)
    for record in records:
        if record.i not in atom_indices or record.j not in atom_indices:
            continue
        normalized_shift = min(record.shift, tuple(-value for value in record.shift))
        graph.add_edge(
            record.i,
            record.j,
            bond_type=bonds.bond_type(record.i, record.j),
            cut=record in cut_set,
            shift=normalized_shift,
        )
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
    residual_degree = {
        index: sum(
            1
            for record in records
            if record not in cuts
            and record.i in atom_indices
            and record.j in atom_indices
            and index in record.pair
        )
        for index in atom_indices
    }
    signatures = tuple(
        sorted(
            (
                tuple(sorted((symbols[record.i], symbols[record.j]))),
                tuple(sorted((residual_degree[record.i], residual_degree[record.j]))),
                bonds.bond_type(record.i, record.j),
                round(float(np.linalg.norm(bonds.vector(record.i, record.j))), 6),
            )
            for record in cuts
        )
    )
    marked = _marked_graph(symbols, atom_indices, records, cuts, bonds)
    graph_hash = nx.weisfeiler_lehman_graph_hash(
        marked, node_attr="symbol", edge_attr="bond_type"
    )
    penalties = tuple(sorted(_bond_class(symbols, record) for record in cuts))
    return (
        max(penalties),
        sum(penalties),
        penalties,
        round(radius, 8),
        signatures,
        graph_hash,
    )


def _select_canonical_candidate(
    atoms: ase.Atoms,
    bonds: BondNetwork,
    symbols: list[str],
    atom_indices: set[int],
    valid: list[tuple[tuple[PeriodicBond, ...], ComponentPeriodicity]],
) -> tuple[tuple[PeriodicBond, ...], ComponentPeriodicity]:
    component_records = periodic_bonds(atoms, bonds)
    scored = [
        (
            _candidate_score(
                atoms, bonds, symbols, atom_indices, component_records, cuts, residual
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
            symbols, atom_indices, component_records, best[0][0], bonds
        )
        node_match = nx.algorithms.isomorphism.categorical_node_match("symbol", None)
        edge_match = nx.algorithms.isomorphism.categorical_edge_match(
            ["bond_type", "cut", "shift"], [None, False, (0, 0, 0)]
        )
        for cuts, _ in best[1:]:
            candidate = _marked_graph(
                symbols, atom_indices, component_records, cuts, bonds
            )
            if not nx.is_isomorphic(
                reference, candidate, node_match=node_match, edge_match=edge_match
            ):
                raise ValueError(
                    "ambiguous periodic cut: top-ranked candidates are not "
                    "chemically equivalent"
                )
    return min(best, key=lambda item: tuple(record.pair for record in item[0]))
```

- [ ] **Step 4: Verify selector tests and all periodicity tests pass**

Run `pytest -q tests/test_periodicity.py`.

Expected: all tests pass, including the two-cut and rank-two cases.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/mofbb/periodicity.py tests/test_periodicity.py
git commit -m "feat: select finite cuts for periodic rods"
```

---

### Task 3: Two-sided X capping for periodic internal cuts

**Files:**
- Modify: `src/mofbb/models.py:10-70`
- Modify: `src/mofbb/fragments.py:20-155`
- Modify: `tests/test_models_and_results.py`
- Modify: `tests/test_edge_decomposer.py`

**Interfaces:**
- Consumes: pair tuples or `PeriodicBond` instances as fragment connection inputs.
- Produces: shift-aware `XSite` records and zero-holonomy finite `Fragment` objects.

- [ ] **Step 1: Write a failing fragment test for an internal periodic cut**

Add to `tests/test_edge_decomposer.py`:

```python
from mofbb.periodicity import PeriodicBond


def test_fragment_builder_caps_both_sides_of_periodic_internal_cut():
    atoms, bonds = make_periodic_atoms_and_bonds(
        ["C", "C", "C"],
        [(0, 1), (1, 2), (0, 2)],
        shifts={(0, 2): (-1, 0, 0)},
    )
    cut = PeriodicBond(0, 2, (-1, 0, 0))

    fragment = FragmentBuilder(atoms, bonds, ["C", "C", "C"]).materialize_fragment(
        "node", 1, {0, 1, 2}, (cut,)
    )

    assert fragment.source_indices == (0, 1, 2)
    assert fragment.num_x == 2
    assert [(site.internal_source_index, site.external_image_shift) for site in fragment.x_sites] == [
        (0, (-1, 0, 0)),
        (2, (1, 0, 0)),
    ]
    assert not any({i, j} == {0, 2} for i, j, _ in fragment.bonds)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tests/test_edge_decomposer.py::test_fragment_builder_caps_both_sides_of_periodic_internal_cut
```

Expected: `FragmentBuilder` cannot interpret `PeriodicBond` and does not create
two X sites.

- [ ] **Step 3: Extend `XSite` without breaking positional construction**

Append defaulted fields after the current required fields:

```python
source_cut_shift: tuple[int, int, int] = (0, 0, 0)
external_image_shift: tuple[int, int, int] = (0, 0, 0)
```

Add both lists to `XSite.as_dict()`.

- [ ] **Step 4: Normalize fragment connection inputs and materialize internal cuts**

In `FragmentBuilder`, add:

```python
def _connection_record(
    self, connection: tuple[int, int] | PeriodicBond
) -> PeriodicBond:
    if isinstance(connection, PeriodicBond):
        return connection
    i, j = sorted(connection)
    return next(
        record for record in periodic_bonds(self.atoms, self.bonds)
        if record.pair == (i, j)
    )
```

Normalize once per `build_role_fragments()` or `materialize_fragment()` call.
Exclude records whose two endpoints are in the component from `_internal_bonds`.
For each such record, append one cap anchored at `i` using
`self.bonds.vector(i, j)` and one anchored at `j` using
`self.bonds.vector(j, i)`. Store canonical `source_cut_shift` on both sites and
inverse oriented `external_image_shift` values.

- [ ] **Step 5: Add a retained-bond geometry assertion**

After unwrapping and before constructing `Fragment`, verify every retained real
internal bond:

```python
actual = unwrapped[b] - unwrapped[a]
expected = self.bonds.vector(source_i, source_j)
if not np.isclose(np.linalg.norm(actual), np.linalg.norm(expected), atol=1.0e-6):
    raise ValueError(
        f"Fragment {kind} {index} retained a distorted periodic bond "
        f"{(source_i, source_j)}."
    )
```

- [ ] **Step 6: Run fragment/model tests and verify GREEN**

Run:

```bash
pytest -q tests/test_edge_decomposer.py::test_fragment_builder_caps_both_sides_of_periodic_internal_cut tests/test_models_and_results.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/mofbb/models.py src/mofbb/fragments.py tests/test_models_and_results.py tests/test_edge_decomposer.py
git commit -m "feat: cap periodic cuts on both fragment sides"
```

---

### Task 4: Integrate periodic openings and translated topology self-edges

**Files:**
- Modify: `src/mofbb/decomposer.py:35-105`
- Modify: `src/mofbb/edge_decomposer.py:85-190,540-760`
- Modify: `src/mofbb/topology.py`
- Modify: `src/mofbb/results.py:20-55`
- Modify: `tests/test_edge_decomposer.py`
- Modify: `tests/test_models_and_results.py`

**Interfaces:**
- Consumes: `select_periodic_cuts()`, all physical metal/organic cut records, and built fragments.
- Produces: aligned `connection_records` on both result types and `TopologyEdge` records that include translated same-slot edges.

- [ ] **Step 1: Write a failing topology test for translated self-edges and exact component frames**

Add to `tests/test_models_and_results.py`:

```python
import numpy as np

from mofbb.bonding import BondNetwork
from mofbb.fragments import FragmentBuilder
from mofbb.periodicity import PeriodicBond
from mofbb.topology import TopologyBuilder


def test_topology_retains_translated_periodic_self_edge():
    atoms = Atoms(
        "CCC",
        positions=[(1, 0, 0), (4, 0, 0), (8, 0, 0)],
        cell=[10, 10, 10],
        pbc=True,
    )
    adjacency = (frozenset({1, 2}), frozenset({0, 2}), frozenset({0, 1}))
    vectors = {
        (0, 1): np.array((3.0, 0.0, 0.0)),
        (1, 2): np.array((4.0, 0.0, 0.0)),
        (0, 2): np.array((-3.0, 0.0, 0.0)),
    }
    bonds = BondNetwork(adjacency=adjacency, vectors=vectors)
    cut = PeriodicBond(0, 2, (-1, 0, 0))
    fragment = FragmentBuilder(atoms, bonds, ["C", "C", "C"]).materialize_fragment(
        "node", 1, {0, 1, 2}, (cut,)
    )

    vertices, edges = TopologyBuilder(atoms, bonds).build(
        [fragment], [], (cut,)
    )

    assert len(edges) == 1
    assert edges[0].begin_kind == edges[0].end_kind == "node"
    assert edges[0].begin_index == edges[0].end_index == 1
    assert edges[0].source_cut_shift == (-1, 0, 0)
    assert vertices[0].coordination == 2
    assert np.allclose(
        np.asarray(edges[0].end_frac) - edges[0].begin_frac,
        (-1, 0, 0),
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/test_models_and_results.py::test_topology_retains_translated_periodic_self_edge
```

Expected: the topology builder skips `begin_key == end_key`.

- [ ] **Step 3: Add defaulted source metadata to `TopologyEdge`**

Append:

```python
source_cut_bond: tuple[int, int] | None = None
source_cut_shift: tuple[int, int, int] = (0, 0, 0)
slot_translation: tuple[int, int, int] = (0, 0, 0)
```

Existing seven-argument positional construction remains valid.

- [ ] **Step 4: Give both result types an aligned compatibility field**

Add this final defaulted field to `DecompositionResult` and
`NodeEdgeDecompositionResult`:

```python
connection_records: tuple[PeriodicBond, ...] = ()
periodic_openings: tuple[PeriodicOpening, ...] = ()
periodic_validation: dict[str, bool] = field(default_factory=dict)
```

When empty, writers and topology builders derive records from legacy
`connecting_bonds`. New decomposer runs always populate it.

- [ ] **Step 5: Open periodic metal components before materialization**

In both decomposers:

```python
all_periodic_bonds = periodic_bonds(self.atoms, self.bonds)
periodic_cuts = []
periodic_openings = []
for component in connected_components(metal_node_indices, self.bonds.pairs):
    selection = select_periodic_cuts(
        self.atoms,
        self.bonds,
        self.symbols,
        component,
        all_periodic_bonds,
    )
    periodic_cuts.extend(selection.cuts)
    if selection.cuts:
        periodic_openings.append(PeriodicOpening.from_selection(component, selection))

boundary_records = tuple(
    record for record in all_periodic_bonds
    if record.pair in set(connecting_bonds)
)
```

For `SingleNodePorDecomposer`, set:

```python
connection_records = tuple(sorted((*boundary_records, *periodic_cuts)))
```

For `NodeEdgePorDecomposer`, let `OrganicLinkerSplitter` finish first, then add
its physical internal cuts:

```python
organic_pair_set = set(organic_split.organic_connecting_bonds)
organic_records = tuple(
    record for record in all_periodic_bonds if record.pair in organic_pair_set
)
connection_records = tuple(
    sorted((*boundary_records, *organic_records, *periodic_cuts))
)
```

Pass the records relevant to each materialization stage. Keep
`connecting_bonds=tuple(record.pair for record in connection_records)` in the
result and attach `connection_records`, `tuple(periodic_openings)`, and:

```python
periodic_validation={
    "all_fragments_zero_holonomy": True,
    "all_retained_bonds_undistorted": True,
}
```

These booleans are set only after the selector and fragment geometry checks
complete without error. Preserve
`metal_connecting_bonds` and `organic_connecting_bonds` as their existing
boundary categories; periodic metal-metal cuts are derived as the remaining
records and reported in a new additive category.

- [ ] **Step 6: Replace heuristic topology unwrapping with fragment frames**

In `topology.py`, create one `ComponentPeriodicity` for each fragment after
excluding all connection records. Calculate its unwrapped fractional centroid:

```python
scaled = self.atoms.get_scaled_positions(wrap=False)
raw_centroid = np.mean(
    [scaled[i] + np.asarray(frame.offset(i)) for i in fragment.source_indices],
    axis=0,
)
```

For a connection oriented from `begin_atom` to `end_atom`, calculate
`slot_translation = connection_shift(record, begin_atom, begin_frame,
end_atom, end_frame)`. Normalize both endpoints by the integer floor of the
begin raw centroid:

```python
origin = np.floor(begin_raw_centroid)
begin_frac = begin_raw_centroid - origin
end_frac = end_raw_centroid + np.asarray(slot_translation) - origin
```

Do not skip equal fragment keys. Increment coordination once per endpoint, so a
self-edge increments the same vertex twice. Use this shared frame path from
`NodeEdgeTopologyBuilder` and remove the X-direction image heuristic after its
existing parallel-edge tests pass with exact shifts.

- [ ] **Step 7: Run topology and decomposer tests and verify GREEN**

Run:

```bash
pytest -q tests/test_edge_decomposer.py tests/test_models_and_results.py tests/test_geometry.py
```

Expected: all tests pass, including existing periodic parallel-edge tests and
the new self-edge test.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/mofbb/decomposer.py src/mofbb/edge_decomposer.py src/mofbb/topology.py src/mofbb/results.py src/mofbb/models.py tests/test_edge_decomposer.py tests/test_models_and_results.py
git commit -m "feat: retain periodic rod continuations in topology"
```

---

### Task 5: Preserve pendant donors and assign logical multidentate ports

**Files:**
- Modify: `src/mofbb/models.py`
- Modify: `src/mofbb/edge_decomposer.py:340-520`
- Modify: `tests/test_edge_decomposer.py`

**Interfaces:**
- Consumes: physical fragments, X sites, connection records, and fragment component frames.
- Produces: `LogicalPort`, `Fragment.logical_ports`, `Fragment.logical_kind`, and intact CPO-like organic cores.

- [ ] **Step 1: Write a failing pendant-donor organic test**

Add to `tests/test_edge_decomposer.py`:

```python
def test_organic_splitter_absorbs_multidentate_pendant_oxygen_into_core():
    symbols = ["Zn", "Zn", "Zn", "Zn", "C", "C", "C", "O"]
    positions = [
        (-2.0, 0.0, 0.0),
        (-2.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (0.4, 0.7, 0.0),
        (-0.8, 0.0, 0.0),
    ]
    pairs = [
        (4, 5),
        (5, 6),
        (4, 6),
        (4, 7),
        (0, 7),
        (1, 7),
        (2, 5),
        (3, 6),
    ]
    atoms = Atoms(symbols=symbols, positions=positions, cell=[10, 10, 10], pbc=True)
    bonds = make_bond_network(positions, pairs)
    metal_cuts = ((0, 7), (1, 7), (2, 5), (3, 6))
    raw = FragmentBuilder(atoms, bonds, symbols).materialize_fragment(
        "linker", 1, {4, 5, 6, 7}, metal_cuts
    )

    split = OrganicLinkerSplitter(atoms, bonds, symbols).split([raw], metal_cuts)

    assert len(split.organic_node_fragments) == 1
    assert split.organic_node_fragments[0].source_indices == (4, 5, 6, 7)
    assert split.edge_fragments == []
    assert (4, 7) not in split.organic_connecting_bonds
```

In `test_organic_linker_splitter_creates_internal_nodes_and_edges`, retain and
re-run these existing carbon-path assertions:

```python
assert sorted(fragment.source_indices for fragment in split.edge_fragments) == [
    (7,), (8,), (9,), (10,)
]
assert all(fragment.num_x == 2 for fragment in split.edge_fragments)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest -q \
  tests/test_edge_decomposer.py::test_organic_splitter_absorbs_multidentate_pendant_oxygen_into_core \
  tests/test_edge_decomposer.py::test_organic_linker_splitter_creates_internal_nodes_and_edges
```

Expected: the oxygen is emitted as a separate organic node.

- [ ] **Step 3: Implement the narrow absorption predicate**

Add:

```python
def _is_pendant_donor_component(
    self,
    component: set[int],
    source_atoms: set[int],
    node_component_by_atom: dict[int, int],
) -> bool:
    component_symbols = {self.symbols[index] for index in component}
    if "C" in component_symbols or not (component_symbols & {"O", "N", "S"}):
        return False
    boundary_nodes = self._boundary_node_indices(component, node_component_by_atom)
    if len(boundary_nodes) != 1:
        return False
    return all(
        neighbor in component
        or neighbor in node_component_by_atom
        or neighbor not in source_atoms
        for atom_index in component
        for neighbor in self.bonds.adjacency[atom_index]
    )
```

In the absorption pass, absorb a component when this predicate is true, before
the existing `port_count < 2` rule. Keep the existing rule for non-donor stubs.

- [ ] **Step 4: Add logical port models**

In `models.py` add:

```python
@dataclass(frozen=True)
class LogicalPort:
    logical_port_id: int
    x_indices: tuple[int, ...]
    neighbor_kind: str
    neighbor_index: int
    translation: tuple[int, int, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "logical_port_id": self.logical_port_id,
            "x_indices": list(self.x_indices),
            "neighbor_kind": self.neighbor_kind,
            "neighbor_index": self.neighbor_index,
            "translation": list(self.translation),
        }
```

Append defaulted `logical_kind: str | None = None` and
`logical_ports: tuple[LogicalPort, ...] = ()` fields to `Fragment`. Ensure
`relabel_fragment()` copies them.

- [ ] **Step 5: Assign logical ports after physical fragments exist**

Add `assign_logical_ports(atoms, bonds, fragments, connection_records)` to
`edge_decomposer.py`. Build `atom_to_fragment`, calculate residual frames, and
group each fragment's X sites by:

```python
(neighbor_fragment.kind, neighbor_fragment.index, slot_translation)
```

where `slot_translation` comes from `connection_shift()` oriented from the
X-site's internal source atom to its external source atom. Sort groups by that
key, build stable `LogicalPort` identifiers, and set:

```python
fragment.logical_kind = (
    "edge" if len(fragment.logical_ports) == 2
    else "organic_node" if len(fragment.logical_ports) > 2
    else fragment.kind
)
```

Call it only after metal, organic-node, edge, and accessory-linker fragments are
assembled.

- [ ] **Step 6: Test logical 3+3 grouping**

Add this test using six physical cuts split across two neighboring metal
fragments:

```python
from mofbb.edge_decomposer import assign_logical_ports
from mofbb.periodicity import periodic_bonds


def test_logical_ports_group_three_physical_sites_per_neighbor():
    symbols = ["Zn"] * 6 + ["C"] * 6
    positions = [(float(i), 0.0, 0.0) for i in range(12)]
    cuts = ((0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11))
    pairs = [
        (0, 1), (1, 2),
        (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (6, 11),
        *cuts,
    ]
    atoms = Atoms(symbols=symbols, positions=positions, cell=[30, 30, 30], pbc=True)
    bonds = make_bond_network(positions, pairs)
    builder = FragmentBuilder(atoms, bonds, symbols)
    left = builder.materialize_fragment("metal_node", 1, {0, 1, 2}, cuts)
    right = builder.materialize_fragment("metal_node", 2, {3, 4, 5}, cuts)
    organic = builder.materialize_fragment(
        "organic_node", 1, set(range(6, 12)), cuts
    )
    cut_set = set(cuts)
    records = tuple(
        record for record in periodic_bonds(atoms, bonds) if record.pair in cut_set
    )

    assign_logical_ports(atoms, bonds, [left, right, organic], records)

    assert organic.num_x == 6
    assert organic.logical_kind == "edge"
    assert [port.x_indices for port in organic.logical_ports] == [
        (6, 7, 8),
        (9, 10, 11),
    ]
```

- [ ] **Step 7: Run organic and full edge-decomposer tests**

Run `pytest -q tests/test_edge_decomposer.py`.

Expected: all tests pass.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/mofbb/models.py src/mofbb/edge_decomposer.py tests/test_edge_decomposer.py
git commit -m "fix: preserve multidentate organic donor groups"
```

---

### Task 6: Shift-aware summaries and exact self-edge reconstruction

**Files:**
- Modify: `src/mofbb/models.py`
- Modify: `src/mofbb/results.py:45-360`
- Modify: `src/mofbb/edge_decomposer.py:760-1040`
- Modify: `tests/test_models_and_results.py`
- Modify: `tests/test_edge_decomposer.py`

**Interfaces:**
- Consumes: `Fragment.logical_ports`, `connection_records`, two-sided periodic X sites, and topology edges.
- Produces: schema-version-2 reconstruction maps with shift-aware physical edges and logical-port metadata.

- [ ] **Step 1: Write failing result tests for same-slot two-sided mapping**

Add to `tests/test_models_and_results.py`:

```python
from mofbb.periodicity import PeriodicBond


def test_reconstruction_maps_both_sides_of_same_slot_periodic_cut(tmp_path):
    cut = PeriodicBond(0, 2, (-1, 0, 0))
    atoms = Atoms(
        symbols=["C", "C", "C", "X", "X"],
        positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (-0.5, 0, 0), (2.5, 0, 0)],
        cell=[10, 10, 10],
        pbc=True,
    )
    fragment = Fragment(
        kind="node",
        index=1,
        source_indices=(0, 1, 2),
        atoms=atoms,
        bonds=((0, 1, "S"), (1, 2, "S"), (0, 3, "S"), (2, 4, "S")),
        fingerprint="periodic-node",
        x_sites=(
            XSite(3, 0, (0, 2), 0, 2, (-1, 0, 0), (-1, 0, 0)),
            XSite(4, 2, (0, 2), 2, 0, (-1, 0, 0), (1, 0, 0)),
        ),
    )
    result = DecompositionResult(
        name="periodic",
        node_indices=(0, 1, 2),
        linker_indices=(),
        points_of_extension=(),
        connecting_bonds=((0, 2),),
        cell_parameters=(10, 10, 10, 90, 90, 90),
        topology_vertices=[TopologyVertex("node", 1, 2, (0.2, 0.0, 0.0))],
        topology_edges=[
            TopologyEdge(
                "node", 1, "node", 1,
                (0.2, 0.0, 0.0), (-0.8, 0.0, 0.0), (-0.3, 0.0, 0.0),
                (0, 2), (-1, 0, 0), (-1, 0, 0),
            )
        ],
        node_fragments=[fragment],
        connection_records=(cut,),
    )

    result.write(tmp_path)
    reconstruction = json.loads((tmp_path / "reconstruction_map.json").read_text())
    edge = reconstruction["edges"][0]

    assert edge["endpoint_slots"] == [0, 0]
    assert edge["source_cut_bond"] == [0, 2]
    assert edge["source_cut_shift"] == [-1, 0, 0]
    assert [site["x_index"] for site in edge["endpoint_x_sites"]] == [3, 4]
    assert reconstruction["validation"]["all_edges_mapped"] is True
    assert reconstruction["validation"]["all_x_sites_used_once"] is True
```

- [ ] **Step 2: Run the focused result tests and verify RED**

Expected: the current `(slot_id, source_cut_bond)` lookup overwrites one side
and fails mapping.

- [ ] **Step 3: Use endpoint-aware X lookup in both result writers**

Index sites by:

```python
(
    slot_id,
    site.source_cut_bond,
    site.source_cut_shift,
    site.internal_source_index,
)
```

For each `PeriodicBond`, obtain the endpoint slot for `record.i` and `record.j`
and look up the X side anchored to that exact source endpoint. Include
`source_cut_shift` and each side's `external_image_shift` in JSON.

- [ ] **Step 4: Add physical/logical fragment metadata conditionally**

Extend `Fragment.as_dict()` and group records with:

```python
"physical_cn": fragment.num_x,
"logical_cn": len(fragment.logical_ports) if fragment.logical_ports else fragment.num_x,
"logical_kind": fragment.logical_kind or fragment.kind,
"logical_ports": [port.as_dict() for port in fragment.logical_ports],
```

Keep every legacy field. Update the exact summary-key and exact group-record
tests to include these additive fields.

- [ ] **Step 5: Emit aligned connection records and periodic validation**

When `result.connection_records` is populated, add:

```python
"connecting_bond_records": [record.as_dict() for record in result.connection_records],
"periodic_components": [opening.as_dict() for opening in result.periodic_openings]
```

For node-edge summaries, derive periodic records by subtracting the legacy
metal-boundary and organic-internal pair sets, then add:

```python
"num_periodic_connecting_bonds": len(periodic_records),
"periodic_connecting_bonds": [list(record.pair) for record in periodic_records],
"periodic_connecting_bond_records": [record.as_dict() for record in periodic_records],
```

Do not fold these records into `num_metal_connecting_bonds`; that field retains
its original metal/organic-boundary meaning.

Set reconstruction `schema_version` to `2` and add:

```python
"all_self_edge_shifts_inverse": all_self_edge_shifts_inverse,
"all_fragments_zero_holonomy": all_fragments_zero_holonomy,
"all_retained_bonds_undistorted": all_retained_bonds_undistorted,
```

Take the last two values from `result.periodic_validation`, defaulting to
`True` only for manually constructed legacy results with no periodic openings.
Compute `all_self_edge_shifts_inverse` directly from the two mapped X sides of
every same-slot edge.

Classify duplicate slot pairs with distinct nonzero translations as
`translated_parallel_edges`; reserve `duplicate_edge_warnings` for identical
slot pair plus identical translation duplicates.

- [ ] **Step 6: Run result and edge-decomposer tests and verify GREEN**

Run:

```bash
pytest -q tests/test_models_and_results.py tests/test_edge_decomposer.py
```

Expected: all tests pass and the periodic self-edge uses both X sides exactly
once.

- [ ] **Step 7: Commit Task 6**

```bash
git add src/mofbb/models.py src/mofbb/results.py src/mofbb/edge_decomposer.py tests/test_models_and_results.py tests/test_edge_decomposer.py
git commit -m "feat: write logical ports and periodic reconstruction edges"
```

---

### Task 7: Sample regressions, documentation, and regenerated inspection outputs

**Files:**
- Modify: `README.md`
- Modify: `/nanolab/libo/mof_builder/test_decom/node-edge-decomposed/VALIDATION_REPORT.md`
- Regenerate: `/nanolab/libo/mof_builder/test_decom/node-edge-decomposed/{CPO-27-Ni,HKUST1,MOF5,UiO-67,Zn2-TTFB}`

**Interfaces:**
- Consumes: the completed CLI and all sample CIFs.
- Produces: verified persistent manual-inspection artifacts and corrected reasonableness report.

- [ ] **Step 1: Run the complete unit suite before sample regeneration**

Run:

```bash
pytest -q
```

Expected: every test passes; only pre-existing optional-dependency skips remain.

- [ ] **Step 2: Regenerate all five node-edge outputs into a fresh staging directory**

Run:

```bash
rm -rf /tmp/mofbb-periodic-validation
for name in CPO-27-Ni HKUST1 MOF5 UiO-67 Zn2-TTFB; do
  python -m mofbb.cli "/nanolab/libo/mof_builder/test_decom/${name}.cif" \
    --decomposition node-edge \
    --output-dir "/tmp/mofbb-periodic-validation/${name}"
done
```

Expected: all commands exit zero.

- [ ] **Step 3: Run machine invariants over staged outputs**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/tmp/mofbb-periodic-validation")
stable = {
    "HKUST1": (6, 8, 0, 24, {"C4Cu2O8X4", "C6H3X3"}),
    "MOF5": (8, 0, 24, 48, {"C6O13X6Zn4", "C6H4X2"}),
    "UiO-67": (1, 0, 6, 12, {"C12H4O32X12Zr6", "C12H8X2"}),
}
for name in ("CPO-27-Ni", "HKUST1", "MOF5", "UiO-67", "Zn2-TTFB"):
    summary = json.loads((root / name / "decomposition_summary.json").read_text())
    reconstruction = json.loads((root / name / "reconstruction_map.json").read_text())
    validation = reconstruction["validation"]
    for key in (
        "all_edges_mapped",
        "all_x_sites_used_once",
        "all_self_edge_shifts_inverse",
        "all_fragments_zero_holonomy",
        "all_retained_bonds_undistorted",
    ):
        assert validation[key] is True, (name, key, validation[key])
    if name in stable:
        expected = stable[name]
        actual = (
            summary["num_metal_nodes"],
            summary["num_organic_nodes"],
            summary["num_edges"],
            summary["num_connecting_bonds"],
            {group["formula"] for group in summary["groups"]},
        )
        assert actual == expected, (name, actual, expected)

cpo = json.loads((root / "CPO-27-Ni" / "decomposition_summary.json").read_text())
assert cpo["num_organic_nodes"] == 9
assert cpo["num_edges"] == 0
assert all(group["formula"] != "OX3" for group in cpo["groups"])
cpo_organic = next(
    group for group in cpo["groups"] if group["formula"] == "C6H2O2X6"
)
assert cpo_organic["logical_cn"] == 2

zn = json.loads((root / "Zn2-TTFB" / "decomposition_summary.json").read_text())
zn_metal = next(group for group in zn["groups"] if group["kind"] == "metal_node")
assert zn_metal["formula"] == "C12H12O30X16Zn6"
assert zn["num_connecting_bonds"] == 52
assert (zn["num_organic_nodes"], zn["num_edges"]) == (6, 24)
PY
```

- [ ] **Step 4: Inspect representative XYZ bond lengths**

Run this source-aware check; local real-atom indices precede all X indices in
every fragment:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from mofbb.edge_decomposer import NodeEdgePorDecomposer

inputs = Path("/nanolab/libo/mof_builder/test_decom")
for name in ("CPO-27-Ni", "HKUST1", "MOF5", "UiO-67", "Zn2-TTFB"):
    decomposer = NodeEdgePorDecomposer(inputs / f"{name}.cif")
    result = decomposer.decompose()
    for fragment in result.fragments:
        real_count = len(fragment.source_indices)
        for local_i, local_j, _ in fragment.bonds:
            if local_i >= real_count or local_j >= real_count:
                continue
            source_i = fragment.source_indices[local_i]
            source_j = fragment.source_indices[local_j]
            actual = np.linalg.norm(
                fragment.atoms.positions[local_j] - fragment.atoms.positions[local_i]
            )
            expected = np.linalg.norm(decomposer.bonds.vector(source_i, source_j))
            assert np.isclose(actual, expected, atol=1.0e-6), (
                name, fragment.kind, fragment.index, source_i, source_j, actual, expected
            )
PY
```

Specifically inspect that the Zn2-TTFB representative has no 20-angstrom
internal bonds and CPO-27 has no standalone oxygen BB.

- [ ] **Step 5: Replace persistent outputs only after staged validation passes**

Run:

```bash
target=/nanolab/libo/mof_builder/test_decom/node-edge-decomposed
backup=/nanolab/libo/mof_builder/test_decom/node-edge-decomposed.pre-periodic-fix
test ! -e "$backup"
mv "$target" "$backup"
mkdir -p "$target"
cp -a /tmp/mofbb-periodic-validation/. "$target/"
cp "$backup/VALIDATION_REPORT.md" "$target/VALIDATION_REPORT.md"
```

Keep the backup until the user has inspected the regenerated outputs.

- [ ] **Step 6: Document the corrected semantics**

Update `README.md` to explain physical X sites, logical ports, rank-one rod
opening, translated self-edges, rank-greater-than-one rejection, and the
non-goal of screw-axis Zn2 reduction. Update `VALIDATION_REPORT.md` with actual
post-fix formulas/counts and mark CPO-27 and Zn2-TTFB as finite crystallographic
rod-repeat representations rather than discrete clusters.

- [ ] **Step 7: Run final verification from a clean output directory**

Run:

```bash
pytest -q
python -m mofbb.cli /nanolab/libo/mof_builder/test_decom/CPO-27-Ni.cif \
  --decomposition node-edge --output-dir /tmp/final-cpo-check --json >/tmp/final-cpo-summary.json
python -m mofbb.cli /nanolab/libo/mof_builder/test_decom/Zn2-TTFB.cif \
  --decomposition node-edge --output-dir /tmp/final-zn-check --json >/tmp/final-zn-summary.json
git diff --check
git status --short
```

Expected: full suite passes; both CLI runs exit zero; JSON parses; no whitespace
errors; only intended source, test, README, and ignored manual-output changes
remain.

- [ ] **Step 8: Commit tracked documentation changes**

```bash
git add README.md
git commit -m "docs: explain periodic rod decomposition"
```

The manual inspection directory is outside this repository and is not included
in the git commit.
