# MOFid-Style Solvent Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a verified pure-Python MOFid-style free/bound solvent classifier and opt-in FSR/ASR preparation to both MOFBB decomposition workflows.

**Architecture:** Collapse MOFBB's existing SingleNode node/linker components into a periodic topology graph, simplify that graph to a fixed point using MOFid's zero/one-connected rules, and map deleted sites back to original atom indices. Keep classification, periodic accounting, bonded subsetting, and artifact writing in focused reusable units so both decomposers share one implementation.

**Tech Stack:** Python 3.10+, ASE, NumPy, NetworkX, pytest, ruff.

## Global Constraints

- Preserve `solvent_mode="none"` behavior and the existing `cleanup` semantics.
- `free` and `all` analyze the complete bonded structure with largest-component cleanup disabled.
- Use MOFBB's current `SingleNodeAtomAssigner`; do not port the complete Open Babel/MOFid deconstructor.
- Preserve original atom order, cell, PBC flags, bond vectors, and source bond types.
- Distinguish redundant nonperiodic connections from connections through distinct periodic images.
- Emit `free_solvent.cif`, `bound_solvent.cif`, `mof_fsr.cif`, and `mof_asr.cif` whenever solvent analysis is enabled.
- Treat one-port terminal/capping linkers as bound solvent, matching MOFid's broad rule.
- Do not add dependencies beyond ASE, NumPy, and NetworkX.
- Implement every behavior test-first and keep the complete existing test suite green.

---

## File Map

- Create `src/mofbb/solvent.py`: topology records, periodic normalization, fixed-point classification, atom products, bonded selection, CIF writing, and summary metadata.
- Create `tests/conftest.py`: reusable bonded framework fixture with retained framework, free solvent, and bound solvent.
- Create `tests/test_solvent.py`: focused topology, periodicity, conservation, failure, and artifact tests.
- Modify `src/mofbb/bonding.py`: reusable ordered bonded-structure subset helper.
- Modify `tests/test_bonding.py`: vector/type/mapping tests for bonded subsetting.
- Modify `src/mofbb/decomposer.py`: `solvent_mode` preparation and SingleNode result metadata.
- Modify `src/mofbb/results.py`: optional solvent result fields and shared output writing.
- Modify `src/mofbb/edge_decomposer.py`: solvent preparation, FSR accessory linkers, result metadata, and output grouping.
- Modify `src/mofbb/cli.py`: `--solvent-mode` parsing and forwarding.
- Modify `src/mofbb/__init__.py`: public lazy exports.
- Create `tests/test_cli.py`: parser and constructor-forwarding tests.
- Modify `tests/test_models_and_results.py`: SingleNode solvent output integration.
- Modify `tests/test_edge_decomposer.py`: NodeEdge `free`/`all` behavior and accessory-linker coverage.
- Modify `README.md`: mode semantics, examples, outputs, and limitations.
- Update `docs/superpowers/specs/2026-07-10-mofid-solvent-removal-design.md`: record ASE's zero-atom CIF limitation and node-edge FSR accessory behavior discovered during planning.

---

### Task 1: Ordered Bonded-Structure Subsetting

**Files:**
- Modify: `src/mofbb/bonding.py:210-251`
- Test: `tests/test_bonding.py`

**Interfaces:**
- Consumes: `ase.Atoms`, `BondNetwork`, and `Iterable[int]` original indices.
- Produces: `subset_bonded_structure(atoms, bonds, indices) -> tuple[ase.Atoms, BondNetwork, tuple[int, ...]]`.

- [ ] **Step 1: Write the failing mapping/vector test**

```python
from mofbb.bonding import subset_bonded_structure


def test_subset_bonded_structure_preserves_order_vectors_and_types() -> None:
    atoms = Atoms(
        symbols=["C", "N", "O"],
        positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = BondNetwork(
        adjacency=(frozenset({1}), frozenset({0, 2}), frozenset({1})),
        vectors={(0, 1): np.array([1.0, 0, 0]), (1, 2): np.array([1.0, 0, 0])},
        bond_types={(0, 1): "A", (1, 2): "D"},
    )

    kept_atoms, kept_bonds, selected_to_original = subset_bonded_structure(
        atoms, bonds, (2, 1)
    )

    assert selected_to_original == (2, 1)
    assert kept_atoms.get_chemical_symbols() == ["O", "N"]
    assert kept_bonds.pairs == [(0, 1)]
    np.testing.assert_allclose(kept_bonds.vector(0, 1), [-1.0, 0, 0])
    assert kept_bonds.bond_type(0, 1) == "D"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/test_bonding.py::test_subset_bonded_structure_preserves_order_vectors_and_types -v`

Expected: collection fails because `subset_bonded_structure` does not exist.

- [ ] **Step 3: Implement the ordered subset helper**

```python
from collections.abc import Iterable


def subset_bonded_structure(
    atoms: ase.Atoms,
    bonds: BondNetwork,
    indices: Iterable[int],
) -> tuple[ase.Atoms, BondNetwork, tuple[int, ...]]:
    selected_to_original = tuple(int(index) for index in indices)
    if len(set(selected_to_original)) != len(selected_to_original):
        raise ValueError("Bonded-structure subset indices must be unique.")
    if any(index < 0 or index >= len(atoms) for index in selected_to_original):
        raise ValueError("Bonded-structure subset index is out of range.")

    reindex = {old: new for new, old in enumerate(selected_to_original)}
    selected = set(selected_to_original)
    vectors: dict[tuple[int, int], np.ndarray] = {}
    bond_types: dict[tuple[int, int], str] = {}
    for i, j in bonds.pairs:
        if i not in selected or j not in selected:
            continue
        new_i = reindex[i]
        new_j = reindex[j]
        pair = tuple(sorted((new_i, new_j)))
        vector = bonds.vector(i, j)
        vectors[pair] = np.array(vector if new_i < new_j else -vector, dtype=float)
        bond_types[pair] = bonds.bond_type(i, j)

    adjacency = [set() for _ in selected_to_original]
    for i, j in vectors:
        adjacency[i].add(j)
        adjacency[j].add(i)
    subset_bonds = BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
        bond_types=bond_types,
    )
    return atoms[list(selected_to_original)], subset_bonds, selected_to_original
```

Refactor `largest_connected_component_with_bonds` to call this helper after its warning, returning only the first two tuple elements.

- [ ] **Step 4: Add duplicate/out-of-range tests and verify GREEN**

```python
def test_subset_bonded_structure_rejects_invalid_indices() -> None:
    atoms = Atoms("CO", positions=[(0, 0, 0), (1, 0, 0)])
    bonds = BondNetwork(
        adjacency=(frozenset({1}), frozenset({0})),
        vectors={(0, 1): np.array([1.0, 0.0, 0.0])},
    )
    with pytest.raises(ValueError, match="unique"):
        subset_bonded_structure(atoms, bonds, (0, 0))
    with pytest.raises(ValueError, match="out of range"):
        subset_bonded_structure(atoms, bonds, (0, 2))
```

Run: `pytest tests/test_bonding.py -v`

Expected: all bonding tests pass.

- [ ] **Step 5: Commit the bonded subset**

```bash
git add src/mofbb/bonding.py tests/test_bonding.py
git commit -m "feat: add bonded structure subsetting"
```

---

### Task 2: MOFid Fixed-Point Topology Rules

**Files:**
- Create: `src/mofbb/solvent.py`
- Create: `tests/test_solvent.py`

**Interfaces:**
- Consumes: internal `_TopologySite` and `_TopologyConnection` collections.
- Produces: `_simplify_topology(source_sites, source_connections) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]`, ordered as free then bound components.

- [ ] **Step 1: Write fixed-point rule tests**

```python
from mofbb.solvent import (
    _TopologyConnection,
    _TopologySite,
    _simplify_topology,
)


def connection(left: int, right: int, shift=(0, 0, 0)) -> _TopologyConnection:
    return _TopologyConnection(left, right, shift)


def test_simplify_topology_classifies_zero_and_one_connected_sites() -> None:
    sites = {
        0: _TopologySite(0, "node", {0}),
        1: _TopologySite(1, "linker", {1, 2}),
        2: _TopologySite(2, "linker", {3}),
    }
    free, bound = _simplify_topology(sites, {connection(0, 1)})
    assert free == ((0,), (3,))
    assert bound == ((1, 2),)


def test_simplify_topology_merges_terminal_node_then_repeats() -> None:
    sites = {
        0: _TopologySite(0, "node", {0}),
        1: _TopologySite(1, "linker", {1, 2, 3}),
        2: _TopologySite(2, "node", {4}),
    }
    free, bound = _simplify_topology(
        sites,
        {connection(0, 1), connection(1, 2)},
    )
    assert free == ((4,),)
    assert bound == ((0, 1, 2, 3),)


def test_periodic_self_connection_prevents_free_solvent_classification() -> None:
    sites = {0: _TopologySite(0, "linker", {0, 1})}
    free, bound = _simplify_topology(
        sites,
        {connection(0, 0, (1, 0, 0))},
    )
    assert free == ()
    assert bound == ()
```

- [ ] **Step 2: Run the topology tests and verify RED**

Run: `pytest tests/test_solvent.py -k simplify_topology -v`

Expected: collection fails because `mofbb.solvent` does not exist.

- [ ] **Step 3: Implement topology records and deterministic simplification**

```python
from dataclasses import dataclass
from typing import Literal, TypeAlias

SiteRole: TypeAlias = Literal["node", "linker"]
CellShift: TypeAlias = tuple[int, int, int]


@dataclass
class _TopologySite:
    site_id: int
    role: SiteRole
    atom_indices: set[int]


@dataclass(frozen=True, order=True)
class _TopologyConnection:
    left: int
    right: int
    translation: CellShift


def _incident_connections(
    site_id: int, connections: set[_TopologyConnection]
) -> list[_TopologyConnection]:
    return sorted(
        connection
        for connection in connections
        if connection.left == site_id or connection.right == site_id
    )


def _site_degree(site_id: int, connections: set[_TopologyConnection]) -> int:
    return sum(
        2 if connection.left == connection.right else 1
        for connection in _incident_connections(site_id, connections)
    )


def _other_site(connection: _TopologyConnection, site_id: int) -> int:
    if connection.left == connection.right:
        raise ValueError("A periodic self-connection has no distinct neighbor.")
    return connection.right if connection.left == site_id else connection.left


def _simplify_topology(
    source_sites: dict[int, _TopologySite],
    source_connections: set[_TopologyConnection],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    sites = {
        site_id: _TopologySite(site.site_id, site.role, set(site.atom_indices))
        for site_id, site in source_sites.items()
    }
    connections = set(source_connections)
    free: list[tuple[int, ...]] = []
    bound: list[tuple[int, ...]] = []

    while True:
        changed = False
        for site_id in sorted(tuple(sites)):
            site = sites.get(site_id)
            if site is None:
                continue
            degree = _site_degree(site_id, connections)
            if degree == 1:
                incident = _incident_connections(site_id, connections)[0]
                neighbor_id = _other_site(incident, site_id)
                if site.role == "node":
                    if _site_degree(neighbor_id, connections) == 1:
                        continue
                    sites[neighbor_id].atom_indices.update(site.atom_indices)
                else:
                    bound.append(tuple(sorted(site.atom_indices)))
                connections.remove(incident)
                del sites[site_id]
                changed = True
            elif degree == 0:
                free.append(tuple(sorted(site.atom_indices)))
                del sites[site_id]
                changed = True
        if not changed:
            break

    return (
        tuple(sorted(free, key=lambda component: component[0])),
        tuple(sorted(bound, key=lambda component: component[0])),
    )
```

- [ ] **Step 4: Run and verify GREEN**

Run: `pytest tests/test_solvent.py -k simplify_topology -v`

Expected: all fixed-point tests pass.

- [ ] **Step 5: Commit the topology engine**

```bash
git add src/mofbb/solvent.py tests/test_solvent.py
git commit -m "feat: add MOFid solvent topology rules"
```

---

### Task 3: Periodic Component Collapse and Public Classification

**Files:**
- Modify: `src/mofbb/solvent.py`
- Create: `tests/conftest.py`
- Modify: `tests/test_solvent.py`

**Interfaces:**
- Consumes: complete `ase.Atoms` and `BondNetwork`.
- Produces: `MofidSolventRemover(atoms, bonds).classify() -> SolventRemovalResult` and `SolventRemovalResult.subset(mode)`.

- [ ] **Step 1: Write public classifier and periodic-connection tests**

```python
def make_atoms_and_bonds(symbols, pairs, shifts=None):
    positions = np.array([(float(index), 0.0, 0.0) for index in range(len(symbols))])
    atoms = Atoms(symbols, positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)
    adjacency = [set() for _ in symbols]
    vectors = {}
    shifts = shifts or {}
    for i, j in pairs:
        a, b = sorted((i, j))
        adjacency[a].add(b)
        adjacency[b].add(a)
        shift = np.array(shifts.get((a, b), (0, 0, 0)), dtype=float)
        vectors[(a, b)] = positions[b] + shift @ atoms.cell.array - positions[a]
    return atoms, BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )


def make_two_connection_fixture(second_shift):
    return make_atoms_and_bonds(
        ["Zn", "N", "N", "C"],
        [(0, 1), (0, 2), (1, 3), (2, 3)],
        shifts={(0, 2): second_shift},
    )


def test_classifier_repeats_node_merge_and_solvent_deletion() -> None:
    atoms, bonds = make_atoms_and_bonds(
        ["Zn", "N", "C", "N", "Zn"],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
    )
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.bound_solvent_components == ((0, 1, 2, 3),)
    assert result.free_solvent_components == ((4,),)
    assert result.fsr_indices == (0, 1, 2, 3)
    assert result.asr_indices == ()


def test_distinct_periodic_images_prevent_bound_classification() -> None:
    atoms, bonds = make_two_connection_fixture(second_shift=(1, 0, 0))
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.free_solvent_components == ()
    assert result.bound_solvent_components == ()


def test_redundant_nonperiodic_connections_collapse_to_one() -> None:
    atoms, bonds = make_two_connection_fixture(second_shift=(0, 0, 0))
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.bound_solvent_components == ((1, 2, 3),)
    assert result.free_solvent_components == ((0,),)


def test_subset_maps_clean_indices_to_original_indices(
    framework_with_free_and_bound_solvent,
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    result = MofidSolventRemover(atoms, bonds).classify()
    selected_atoms, selected_bonds, mapping = result.subset("all")
    assert mapping == result.asr_indices
    assert len(selected_atoms) == len(result.asr_indices)
    assert len(selected_bonds.adjacency) == len(selected_atoms)
```

Create `tests/conftest.py` with the cross-module fixture:

```python
import numpy as np
import pytest
from ase import Atoms

from mofbb.bonding import BondNetwork


@pytest.fixture
def framework_with_free_and_bound_solvent() -> tuple[Atoms, BondNetwork]:
    symbols = ["Zn", "Zn", "N", "C", "N", "N", "C", "N", "O", "N", "C"]
    positions = np.array(
        [(float(index), float(index % 2), 0.0) for index in range(len(symbols))]
    )
    pairs = [
        (0, 2), (2, 3), (3, 4), (4, 1),
        (0, 5), (5, 6), (6, 7), (7, 1),
        (0, 9), (9, 10),
    ]
    atoms = Atoms(symbols, positions=positions, cell=[30.0, 30.0, 30.0], pbc=True)
    adjacency = [set() for _ in atoms]
    vectors = {}
    for i, j in pairs:
        a, b = sorted((i, j))
        adjacency[a].add(b)
        adjacency[b].add(a)
        vectors[(a, b)] = positions[b] - positions[a]
    return atoms, BondNetwork(
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency),
        vectors=vectors,
    )
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_solvent.py -k 'classifier or periodic or redundant or subset' -v`

Expected: imports or attributes fail because the public classifier is absent.

- [ ] **Step 3: Implement classification records and component collapse**

Add these public types and properties:

```python
SolventMode: TypeAlias = Literal["none", "free", "all"]
ActiveSolventMode: TypeAlias = Literal["free", "all"]


@dataclass(frozen=True)
class SolventRemovalResult:
    original_atoms: ase.Atoms
    original_bonds: BondNetwork
    free_solvent_components: tuple[tuple[int, ...], ...]
    bound_solvent_components: tuple[tuple[int, ...], ...]

    @property
    def free_solvent_indices(self) -> tuple[int, ...]:
        return tuple(sorted(index for part in self.free_solvent_components for index in part))

    @property
    def bound_solvent_indices(self) -> tuple[int, ...]:
        return tuple(sorted(index for part in self.bound_solvent_components for index in part))

    @property
    def fsr_indices(self) -> tuple[int, ...]:
        removed = set(self.free_solvent_indices)
        return tuple(index for index in range(len(self.original_atoms)) if index not in removed)

    @property
    def asr_indices(self) -> tuple[int, ...]:
        removed = set(self.free_solvent_indices) | set(self.bound_solvent_indices)
        return tuple(index for index in range(len(self.original_atoms)) if index not in removed)

    def validate(self) -> None:
        original = set(range(len(self.original_atoms)))
        free = set(self.free_solvent_indices)
        bound = set(self.bound_solvent_indices)
        if len(free) != sum(map(len, self.free_solvent_components)):
            raise ValueError("Free-solvent components overlap.")
        if len(bound) != sum(map(len, self.bound_solvent_components)):
            raise ValueError("Bound-solvent components overlap.")
        if free & bound:
            raise ValueError("Free- and bound-solvent atom sets overlap.")
        if (set(self.fsr_indices) | free) != original:
            raise ValueError("FSR atom accounting does not reconstruct the input.")
        if (set(self.asr_indices) | free | bound) != original:
            raise ValueError("ASR atom accounting does not reconstruct the input.")

    def subset(
        self, mode: ActiveSolventMode
    ) -> tuple[ase.Atoms, BondNetwork, tuple[int, ...]]:
        if mode not in {"free", "all"}:
            raise ValueError(f"Unsupported active solvent mode: {mode!r}.")
        indices = self.fsr_indices if mode == "free" else self.asr_indices
        if not indices:
            raise ValueError(
                "Solvent removal deleted the entire structure "
                f"({len(self.free_solvent_indices)} free atoms, "
                f"{len(self.bound_solvent_indices)} bound atoms)."
            )
        return subset_bonded_structure(self.original_atoms, self.original_bonds, indices)
```

Implement `MofidSolventRemover.classify()` with these exact stages:

```python
class MofidSolventRemover:
    def __init__(self, atoms: ase.Atoms, bonds: BondNetwork) -> None:
        if len(bonds.adjacency) != len(atoms):
            raise ValueError("Bond network size does not match the atom count.")
        self.atoms = atoms
        self.bonds = bonds
        self.symbols = atoms.get_chemical_symbols()
        self._inverse_cell: np.ndarray | None = None

    def classify(self) -> SolventRemovalResult:
        assignment = SingleNodeAtomAssigner(self.symbols, self.bonds).assign()
        node_indices = set(assignment.node_indices)
        linker_indices = set(range(len(self.atoms))) - node_indices
        components = [
            ("node", component)
            for component in connected_components(node_indices, self.bonds.pairs)
        ] + [
            ("linker", component)
            for component in connected_components(linker_indices, self.bonds.pairs)
        ]
        components.sort(key=lambda item: min(item[1]))
        sites = {
            site_id: _TopologySite(site_id, role, set(component))
            for site_id, (role, component) in enumerate(components)
        }
        atom_to_site = {
            atom_index: site_id
            for site_id, site in sites.items()
            for atom_index in site.atom_indices
        }
        self._validate_initial_ownership(atom_to_site)
        offsets, self_connections = self._component_offsets(sites)
        connections = self_connections | self._crossing_connections(atom_to_site, offsets)
        free, bound = _simplify_topology(sites, connections)
        result = SolventRemovalResult(self.atoms.copy(), self.bonds, free, bound)
        result.validate()
        return result

    def _validate_initial_ownership(self, atom_to_site: dict[int, int]) -> None:
        expected = set(range(len(self.atoms)))
        if set(atom_to_site) != expected or len(atom_to_site) != len(self.atoms):
            raise ValueError("Initial solvent topology does not own every atom exactly once.")

    def _inverse_cell_matrix(self) -> np.ndarray | None:
        if not np.any(self.atoms.pbc):
            return None
        cell = np.asarray(self.atoms.cell.array, dtype=float)
        if np.linalg.matrix_rank(cell) < 3:
            raise ValueError("Cannot resolve periodic translations from a singular periodic cell.")
        return np.linalg.inv(cell)

    def _bond_shift(self, i: int, j: int) -> np.ndarray:
        if self._inverse_cell is None:
            self._inverse_cell = self._inverse_cell_matrix()
        if self._inverse_cell is None:
            return np.zeros(3, dtype=int)
        direct = np.asarray(self.atoms.positions[j] - self.atoms.positions[i])
        shift_cart = np.asarray(self.bonds.vector(i, j)) - direct
        fractional_shift = shift_cart @ self._inverse_cell
        rounded = np.rint(fractional_shift)
        if not np.allclose(fractional_shift, rounded, atol=1.0e-6):
            raise ValueError(
                f"Bond ({i}, {j}) has a non-integral periodic translation: "
                f"{fractional_shift.tolist()}."
            )
        shift = rounded.astype(int)
        if np.any(shift[~np.asarray(self.atoms.pbc, dtype=bool)]):
            raise ValueError(f"Bond ({i}, {j}) shifts along a non-periodic axis.")
        return shift

    @staticmethod
    def _normalize_self_shift(shift: np.ndarray) -> CellShift:
        forward = tuple(int(value) for value in shift)
        reverse = tuple(-value for value in forward)
        return min(forward, reverse)

    def _component_offsets(
        self, sites: dict[int, _TopologySite]
    ) -> tuple[dict[int, np.ndarray], set[_TopologyConnection]]:
        offsets: dict[int, np.ndarray] = {}
        self_connections: set[_TopologyConnection] = set()
        for site_id, site in sorted(sites.items()):
            root = min(site.atom_indices)
            offsets[root] = np.zeros(3, dtype=int)
            queue = deque([root])
            while queue:
                current = queue.popleft()
                for neighbor in sorted(self.bonds.adjacency[current]):
                    if neighbor not in site.atom_indices:
                        continue
                    expected = offsets[current] + self._bond_shift(current, neighbor)
                    if neighbor not in offsets:
                        offsets[neighbor] = expected
                        queue.append(neighbor)
                        continue
                    cycle_shift = expected - offsets[neighbor]
                    if np.any(cycle_shift):
                        self_connections.add(
                            _TopologyConnection(
                                site_id,
                                site_id,
                                self._normalize_self_shift(cycle_shift),
                            )
                        )
        return offsets, self_connections

    def _crossing_connections(
        self,
        atom_to_site: dict[int, int],
        offsets: dict[int, np.ndarray],
    ) -> set[_TopologyConnection]:
        connections: set[_TopologyConnection] = set()
        for i, j in self.bonds.pairs:
            site_i = atom_to_site[i]
            site_j = atom_to_site[j]
            if site_i == site_j:
                continue
            shift = self._bond_shift(i, j) + offsets[i] - offsets[j]
            if site_i < site_j:
                left, right = site_i, site_j
            else:
                left, right = site_j, site_i
                shift = -shift
            connections.add(
                _TopologyConnection(
                    left,
                    right,
                    tuple(int(value) for value in shift),
                )
            )
        return connections
```

Use `fractional_shift = shift_cart @ inverse_cell`, `rounded = np.rint(fractional_shift)`, and `np.allclose(fractional_shift, rounded, atol=1e-6)` to derive integer shifts. Normalize crossing connections as `(lower_site, upper_site, shift)` and negate the shift when endpoint order reverses. During component traversal, record any nonzero offset inconsistency as a normalized periodic self-connection; self-connections contribute degree two.

- [ ] **Step 4: Add conservation and error tests**

```python
def test_solvent_products_conserve_original_atoms(
    framework_with_free_and_bound_solvent,
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    original = set(range(len(result.original_atoms)))
    assert set(result.fsr_indices) | set(result.free_solvent_indices) == original
    assert (
        set(result.asr_indices)
        | set(result.free_solvent_indices)
        | set(result.bound_solvent_indices)
    ) == original
    assert set(result.free_solvent_indices).isdisjoint(result.bound_solvent_indices)


def test_periodic_classifier_rejects_singular_cell() -> None:
    atoms, bonds = make_atoms_and_bonds(["Zn", "N"], [(0, 1)])
    atoms.set_cell(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="periodic cell"):
        MofidSolventRemover(atoms, bonds).classify()
```

Run: `pytest tests/test_solvent.py -v`

Expected: all classifier, periodicity, invariant, and subset tests pass.

- [ ] **Step 5: Commit the public classifier**

```bash
git add src/mofbb/solvent.py tests/conftest.py tests/test_solvent.py
git commit -m "feat: classify free and bound solvent"
```

---

### Task 4: MOFid-Named CIF and JSON Outputs

**Files:**
- Modify: `src/mofbb/solvent.py`
- Modify: `tests/test_solvent.py`

**Interfaces:**
- Consumes: `SolventRemovalResult`, selected mode, and output directory.
- Produces: `SolventRemovalResult.write(output_dir, mode, selected_to_original) -> dict[str, object]`.

- [ ] **Step 1: Write artifact and summary tests**

```python
def test_write_emits_mofid_named_outputs_and_summary(
    tmp_path, framework_with_free_and_bound_solvent
) -> None:
    result = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    _, _, mapping = result.subset("all")
    summary = result.write(tmp_path, "all", mapping)

    assert set(path.name for path in tmp_path.glob("*.cif")) == {
        "free_solvent.cif",
        "bound_solvent.cif",
        "mof_fsr.cif",
        "mof_asr.cif",
    }
    assert summary["mode"] == "all"
    assert summary["free"]["components"] == [list(part) for part in result.free_solvent_components]
    assert summary["bound"]["num_atoms"] == len(result.bound_solvent_indices)
    assert summary["selected_to_original"] == list(mapping)


def test_empty_solvent_files_are_metadata_only_valid_cif(
    tmp_path, framework_with_free_and_bound_solvent
) -> None:
    classified = MofidSolventRemover(*framework_with_free_and_bound_solvent).classify()
    atoms, bonds, mapping = classified.subset("all")
    result = MofidSolventRemover(atoms, bonds).classify()
    assert result.free_solvent_components == ()
    assert result.bound_solvent_components == ()
    result.write(tmp_path, "all", tuple(range(len(result.original_atoms))))
    text = (tmp_path / "free_solvent.cif").read_text()
    assert text.startswith("data_free_solvent")
    assert "_atom_site_fract_z" in text
    assert not any(line.startswith(("C", "N", "O", "Zn")) for line in text.splitlines())
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_solvent.py -k 'write or empty_solvent' -v`

Expected: tests fail because `SolventRemovalResult.write` is absent.

- [ ] **Step 3: Implement non-empty and empty CIF writing**

```python
def _write_cif_selection(path: Path, atoms: ase.Atoms, indices: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if indices:
        ase.io.write(path, atoms[list(indices)], format="cif")
        return
    a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
    block = re.sub(r"[^A-Za-z0-9_]", "_", path.stem)
    lines = [
        f"data_{block}",
        f"_cell_length_a {a:.15g}",
        f"_cell_length_b {b:.15g}",
        f"_cell_length_c {c:.15g}",
        f"_cell_angle_alpha {alpha:.15g}",
        f"_cell_angle_beta {beta:.15g}",
        f"_cell_angle_gamma {gamma:.15g}",
        "_space_group_name_H-M_alt 'P 1'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
```

Implement `write` to call the helper for free, bound, FSR, and ASR and return the fixed schema from the approved design. Read non-empty artifacts with ASE in tests; validate empty CIF text because ASE 3.x raises on zero atoms.

- [ ] **Step 4: Run artifact tests and verify GREEN**

Run: `pytest tests/test_solvent.py -v`

Expected: all solvent tests pass.

- [ ] **Step 5: Commit solvent artifacts**

```bash
git add src/mofbb/solvent.py tests/test_solvent.py
git commit -m "feat: write solvent removal artifacts"
```

---

### Task 5: SingleNode Decomposer and Result Integration

**Files:**
- Modify: `src/mofbb/decomposer.py:29-85`
- Modify: `src/mofbb/results.py:24-97`
- Modify: `tests/test_models_and_results.py`
- Modify: `tests/test_solvent.py`

**Interfaces:**
- Consumes: `solvent_mode`, `MofidSolventRemover`, `SolventRemovalResult`.
- Produces: a decomposition over legacy, FSR, or ASR atoms with optional solvent metadata.

- [ ] **Step 1: Write constructor/mode preparation tests**

```python
@pytest.mark.parametrize("mode,expected", [("free", "fsr_indices"), ("all", "asr_indices")])
def test_single_node_decomposer_selects_solvent_mode(
    monkeypatch, mode, expected, framework_with_free_and_bound_solvent
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    cleanup_values = []

    def fake_read(*args, cleanup, **kwargs):
        cleanup_values.append(cleanup)
        return atoms.copy(), bonds

    monkeypatch.setattr("mofbb.decomposer.read_bonded_structure", fake_read)
    decomposer = SingleNodePorDecomposer("fixture.cif", solvent_mode=mode)
    result = decomposer.decompose()

    assert cleanup_values == [False]
    assert result.solvent_mode == mode
    assert result.solvent is not None
    assert result.selected_to_original == getattr(result.solvent, expected)


def test_single_node_rejects_invalid_solvent_mode() -> None:
    with pytest.raises(ValueError, match="solvent mode"):
        SingleNodePorDecomposer("fixture.cif", solvent_mode="invalid")
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_solvent.py -k single_node -v`

Expected: constructor rejects the new argument or result fields are absent.

- [ ] **Step 3: Implement preparation and result fields**

Add constructor validation and preparation:

```python
if solvent_mode not in {"none", "free", "all"}:
    raise ValueError(f"Unsupported solvent mode: {solvent_mode!r}.")
self.solvent_mode = solvent_mode
self.solvent: SolventRemovalResult | None = None
self.selected_to_original: tuple[int, ...] | None = None
self.atoms, self.bonds = read_bonded_structure(
    self.cif,
    scale=self.bond_scale,
    cleanup=self.cleanup if solvent_mode == "none" else False,
)
if solvent_mode != "none":
    self.solvent = MofidSolventRemover(self.atoms, self.bonds).classify()
    self.atoms, self.bonds, self.selected_to_original = self.solvent.subset(solvent_mode)
```

Add optional fields to `DecompositionResult` after existing required fields:

```python
solvent_mode: SolventMode = "none"
solvent: SolventRemovalResult | None = None
selected_to_original: tuple[int, ...] | None = None
```

In `write`, add `summary["solvent"] = self.solvent.write(output_dir, self.solvent_mode, self.selected_to_original)` only when the solvent result and mapping are present.

- [ ] **Step 4: Verify output integration and default compatibility**

```python
def test_result_write_includes_solvent_block_when_enabled(
    tmp_path, monkeypatch, framework_with_free_and_bound_solvent
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    enabled = SingleNodePorDecomposer("fixture.cif", solvent_mode="all").decompose()
    enabled_summary = enabled.write(tmp_path / "enabled")
    assert enabled_summary["solvent"]["mode"] == "all"
    assert (tmp_path / "enabled" / "mof_asr.cif").exists()
```

Extend the existing legacy result test with `assert "solvent" not in summary`.

Run: `pytest tests/test_models_and_results.py tests/test_solvent.py -v`

Expected: all SingleNode/result tests pass.

- [ ] **Step 5: Commit SingleNode integration**

```bash
git add src/mofbb/decomposer.py src/mofbb/results.py tests/test_models_and_results.py tests/test_solvent.py
git commit -m "feat: add solvent modes to SingleNode decomposition"
```

---

### Task 6: NodeEdge Integration and FSR Accessory Linkers

**Files:**
- Modify: `src/mofbb/edge_decomposer.py:51-146,698-834`
- Modify: `tests/test_edge_decomposer.py`

**Interfaces:**
- Consumes: the same solvent preparation and selected-to-original mapping as SingleNode.
- Produces: NodeEdge results with optional `bound_linker_fragments` in `free` mode.

- [ ] **Step 1: Write NodeEdge mode and atom-coverage tests**

```python
def test_node_edge_free_mode_preserves_bound_solvent_as_accessory_linker(
    monkeypatch, framework_with_free_and_bound_solvent
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )

    result = NodeEdgePorDecomposer("fixture.cif", solvent_mode="free").decompose()

    assert len(result.bound_linker_fragments) == 1
    covered = {
        original_index
        for fragment in result.fragments
        for original_index in (
            result.selected_to_original[index] for index in fragment.source_indices
        )
    }
    assert covered == set(result.solvent.fsr_indices)


def test_node_edge_all_mode_removes_bound_linker(
    monkeypatch, framework_with_free_and_bound_solvent
) -> None:
    atoms, bonds = framework_with_free_and_bound_solvent
    monkeypatch.setattr(
        "mofbb.edge_decomposer.read_bonded_structure",
        lambda *args, **kwargs: (atoms.copy(), bonds),
    )
    result = NodeEdgePorDecomposer("fixture.cif", solvent_mode="all").decompose()
    assert result.bound_linker_fragments == []
    assert result.selected_to_original == result.solvent.asr_indices
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_edge_decomposer.py -k 'free_mode or all_mode' -v`

Expected: constructor or result fields are missing, or the splitter rejects a one-port fragment.

- [ ] **Step 3: Implement shared preparation and accessory filtering**

Apply the same constructor preparation as Task 5. Before `OrganicLinkerSplitter.split`, partition raw organic fragments in `free` mode:

```python
bound_original = set(self.solvent.bound_solvent_indices) if self.solvent else set()
bound_linkers: list[Fragment] = []
splittable_organics: list[Fragment] = []
for fragment in raw_organics:
    original_indices = {
        self.selected_to_original[index]
        for index in fragment.source_indices
    }
    if bound_original and original_indices <= bound_original:
        bound_linkers.append(fragment)
    else:
        splittable_organics.append(fragment)
organic_split = OrganicLinkerSplitter(
    self.atoms,
    self.bonds,
    self.symbols,
    x_symbol=self.x_symbol,
).split(
    splittable_organics,
    connecting_bonds,
)
```

Include accessory linkers in topology construction, `fragments`, instance writing, grouping, reconstruction mapping, and summary fields:

```python
bound_linker_fragments: list[Fragment] = field(default_factory=list)

@property
def fragments(self) -> list[Fragment]:
    return [
        *self.metal_node_fragments,
        *self.organic_node_fragments,
        *self.edge_fragments,
        *self.bound_linker_fragments,
    ]
```

Add `num_bound_linkers` and `bound_linkers` only when the list is non-empty, leaving default node-edge summaries unchanged.

Extend the convenience function without changing its defaults:

```python
def decompose_mof_with_edges(
    cif: str | Path,
    x_symbol: str = "X",
    bond_scale: float = 1.2,
    cleanup: bool = True,
    solvent_mode: SolventMode = "none",
) -> NodeEdgeDecompositionResult:
    return NodeEdgePorDecomposer(
        cif,
        x_symbol=x_symbol,
        bond_scale=bond_scale,
        cleanup=cleanup,
        solvent_mode=solvent_mode,
    ).decompose()
```

Add `solvent_mode`, `solvent`, and `selected_to_original` fields to
`NodeEdgeDecompositionResult`; invoke
`self.solvent.write(output_dir, self.solvent_mode, self.selected_to_original)`
from its writer under the same condition and schema as `DecompositionResult`.

- [ ] **Step 4: Run NodeEdge tests and verify GREEN**

Run: `pytest tests/test_edge_decomposer.py -v`

Expected: all existing and new NodeEdge tests pass.

- [ ] **Step 5: Commit NodeEdge integration**

```bash
git add src/mofbb/edge_decomposer.py tests/test_edge_decomposer.py
git commit -m "feat: add solvent modes to node-edge decomposition"
```

---

### Task 7: Public Exports, CLI, and README

**Files:**
- Modify: `src/mofbb/__init__.py`
- Modify: `src/mofbb/cli.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: public solvent types and decomposer constructor arguments.
- Produces: lazy imports and `mofbb --solvent-mode {none,free,all}`.

- [ ] **Step 1: Write parser/export tests**

```python
from mofbb import MofidSolventRemover, SolventMode, SolventRemovalResult
from mofbb.cli import build_parser, main


def test_cli_parser_accepts_solvent_modes() -> None:
    parser = build_parser()
    args = parser.parse_args(["input.cif", "--solvent-mode", "all"])
    assert args.solvent_mode == "all"


def test_solvent_types_are_public_lazy_exports() -> None:
    assert MofidSolventRemover.__name__ == "MofidSolventRemover"
    assert SolventRemovalResult.__name__ == "SolventRemovalResult"
    assert "all" in SolventMode.__args__


def test_cli_forwards_solvent_mode(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeResult:
        def write(self, output_dir):
            return {"num_nodes": 0, "num_linkers": 0, "num_connecting_bonds": 0}

    class FakeDecomposer:
        def __init__(self, cif, **kwargs):
            captured.update(kwargs)

        def decompose(self):
            return FakeResult()

    monkeypatch.setattr("mofbb.cli.SingleNodePorDecomposer", FakeDecomposer)
    exit_code = main(
        ["input.cif", "--solvent-mode", "free", "--output-dir", str(tmp_path)]
    )
    assert exit_code == 0
    assert captured["solvent_mode"] == "free"
```

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_cli.py -v`

Expected: parser rejects the option or public imports fail.

- [ ] **Step 3: Implement CLI forwarding and lazy exports**

```python
parser.add_argument(
    "--solvent-mode",
    choices=("none", "free", "all"),
    default="none",
    help=(
        "Solvent removal before decomposition: none keeps legacy cleanup, "
        "free uses the FSR structure, and all uses the ASR structure."
    ),
)
```

Pass `solvent_mode=args.solvent_mode` to the selected decomposer. Add
`MofidSolventRemover`, `SolventMode`, and `SolventRemovalResult` to `__all__`
and `_LAZY_EXPORTS`.

- [ ] **Step 4: Document modes and verify docs examples**

Add README examples:

```bash
mofbb structure.cif --solvent-mode free --output-dir outputs/fsr
mofbb structure.cif --solvent-mode all --output-dir outputs/asr
```

Document FSR, ASR, all four files, JSON mapping, accessory bound linkers in
node-edge FSR mode, the broad capping-ligand rule, and the distinction from
legacy largest-component cleanup.

Run: `pytest tests/test_cli.py -v`

Expected: parser/export tests pass.

- [ ] **Step 5: Commit API and documentation**

```bash
git add src/mofbb/__init__.py src/mofbb/cli.py tests/test_cli.py README.md docs/superpowers/specs/2026-07-10-mofid-solvent-removal-design.md
git commit -m "docs: expose solvent removal workflow"
```

---

### Task 8: Reference Audit and Full Verification

**Files:**
- Modify only files required by failures directly attributable to this feature.

**Interfaces:**
- Consumes: completed feature and `/nanolab/libo/lvebmof/mofid/Resources/KnownCIFs` reference data.
- Produces: authoritative verification evidence for every completion criterion.

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_solvent.py tests/test_bonding.py tests/test_models_and_results.py tests/test_edge_decomposer.py tests/test_cli.py -v`

Expected: all focused tests pass with no unexpected warnings.

- [ ] **Step 2: Run static checks**

Run: `ruff check src tests`

Expected: no lint violations.

- [ ] **Step 3: Run the full suite**

Run: `pytest -v`

Expected: all tests pass; environment-dependent benchmark tests may skip only for their pre-existing documented conditions.

- [ ] **Step 4: Audit MOFid reference structures**

Run:

```bash
python - <<'PY'
from pathlib import Path

import ase.io

from mofbb.bonding import read_bonded_structure
from mofbb.solvent import MofidSolventRemover

root = Path("/nanolab/libo/lvebmof/mofid/Resources/KnownCIFs")


def reference_count_formula(path: Path) -> tuple[int, str]:
    try:
        atoms = ase.io.read(path)
    except Exception:
        return 0, ""
    return len(atoms), atoms.get_chemical_formula()


print("structure,python_free,mofid_free,python_bound,mofid_bound,conserved")
for cif in sorted(root.glob("*.cif")):
    reference_dir = root / "Outputs" / cif.stem / "SingleNode"
    if not reference_dir.is_dir():
        continue
    atoms, bonds = read_bonded_structure(cif, cleanup=False)
    result = MofidSolventRemover(atoms, bonds).classify()
    py_free = atoms[list(result.free_solvent_indices)]
    py_bound = atoms[list(result.bound_solvent_indices)]
    ref_free = reference_count_formula(reference_dir / "free_solvent.cif")
    ref_bound = reference_count_formula(reference_dir / "bound_solvent.cif")
    original = set(range(len(atoms)))
    conserved = (
        set(result.asr_indices)
        | set(result.free_solvent_indices)
        | set(result.bound_solvent_indices)
    ) == original
    print(
        cif.stem,
        (len(py_free), py_free.get_chemical_formula()),
        ref_free,
        (len(py_bound), py_bound.get_chemical_formula()),
        ref_bound,
        conserved,
        sep=",",
    )
PY
```

Expected columns:

```text
structure, Python free formula/count, MOFid free formula/count,
Python bound formula/count, MOFid bound formula/count,
Python FSR/ASR conservation
```

Use MOFid's `SingleNode/free_solvent.cif`, `bound_solvent.cif`, `mof_fsr.cif`,
and `mof_asr.cif` as the reference. Do not change the approved MOFBB assignment
rules to force parity; diagnose discrepancies as bond-graph, assignment, or
topology-rule differences.

- [ ] **Step 5: Verify repository state and diff**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors and no unintended files.
