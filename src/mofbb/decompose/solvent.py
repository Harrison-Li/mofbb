"""MOFid-style solvent removal for decomposition workflows."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Literal, TypeAlias

import ase
import ase.io
import numpy as np
from ase.geometry import cell_to_cellpar

from .assignment import SingleNodeAtomAssigner
from .bonding import BondNetwork, subset_bonded_structure
from .fragments import connected_components
from .pbc import periodic_bonds

SolventMode: TypeAlias = Literal["none", "free", "all"]
ActiveSolventMode: TypeAlias = Literal["free", "all"]
SiteRole: TypeAlias = Literal["node", "linker"]
CellShift: TypeAlias = tuple[int, int, int]


def _write_cif_selection(
    path: Path, atoms: ase.Atoms, indices: tuple[int, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if indices:
        ase.io.write(path, atoms[list(indices)], format="cif", wrap=False)
        return

    block = re.sub(r"[^A-Za-z0-9_]", "_", path.stem)
    lines = [f"data_{block}"]
    if atoms.cell.rank == 3:
        a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
        lines.extend(
            [
                f"_cell_length_a {a:.15g}",
                f"_cell_length_b {b:.15g}",
                f"_cell_length_c {c:.15g}",
                f"_cell_angle_alpha {alpha:.15g}",
                f"_cell_angle_beta {beta:.15g}",
                f"_cell_angle_gamma {gamma:.15g}",
                "_space_group_name_H-M_alt 'P 1'",
            ]
        )
    lines.extend(["_mofbb_atom_count 0", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class SolventRemovalResult:
    original_atoms: ase.Atoms
    original_bonds: BondNetwork
    free_solvent_components: tuple[tuple[int, ...], ...]
    bound_solvent_components: tuple[tuple[int, ...], ...]

    @property
    def free_solvent_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(index for part in self.free_solvent_components for index in part)
        )

    @property
    def bound_solvent_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(index for part in self.bound_solvent_components for index in part)
        )

    @property
    def fsr_indices(self) -> tuple[int, ...]:
        removed = set(self.free_solvent_indices)
        return tuple(
            index for index in range(len(self.original_atoms)) if index not in removed
        )

    @property
    def asr_indices(self) -> tuple[int, ...]:
        removed = set(self.free_solvent_indices) | set(self.bound_solvent_indices)
        return tuple(
            index for index in range(len(self.original_atoms)) if index not in removed
        )

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
        return subset_bonded_structure(
            self.original_atoms,
            self.original_bonds,
            indices,
        )

    def validate_write_request(
        self,
        mode: ActiveSolventMode,
        selected_to_original: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Validate a solvent output request without writing any artifacts."""

        if mode not in {"free", "all"}:
            raise ValueError(f"Unsupported active solvent mode: {mode!r}.")
        self.validate()
        mapping = tuple(selected_to_original)
        expected_mapping = self.fsr_indices if mode == "free" else self.asr_indices
        if mapping != expected_mapping:
            raise ValueError(
                "Solvent selected-to-original mapping is inconsistent with "
                f"mode {mode!r}: expected {expected_mapping}, received {mapping}."
            )
        cell_rank = self.original_atoms.cell.rank
        periodic = tuple(bool(value) for value in self.original_atoms.pbc)
        representable = (cell_rank == 3 and all(periodic)) or (
            cell_rank == 0 and not any(periodic)
        )
        if not representable:
            raise ValueError(
                "CIF output requires either a rank-3 cell with full periodicity "
                "or a rank-0 cell with no periodicity; "
                f"got cell rank {cell_rank} with periodic flags {periodic}."
            )
        return mapping

    def write(
        self,
        output_dir: Path,
        mode: ActiveSolventMode,
        selected_to_original: tuple[int, ...],
    ) -> dict[str, object]:
        mapping = self.validate_write_request(mode, selected_to_original)

        output_dir = Path(output_dir)
        selections = {
            "free_solvent.cif": self.free_solvent_indices,
            "bound_solvent.cif": self.bound_solvent_indices,
            "mof_fsr.cif": self.fsr_indices,
            "mof_asr.cif": self.asr_indices,
        }
        for filename, indices in selections.items():
            _write_cif_selection(output_dir / filename, self.original_atoms, indices)

        return {
            "mode": mode,
            "free": {
                "num_components": len(self.free_solvent_components),
                "num_atoms": len(self.free_solvent_indices),
                "components": [list(part) for part in self.free_solvent_components],
                "path": "free_solvent.cif",
            },
            "bound": {
                "num_components": len(self.bound_solvent_components),
                "num_atoms": len(self.bound_solvent_indices),
                "components": [list(part) for part in self.bound_solvent_components],
                "path": "bound_solvent.cif",
            },
            "fsr": {
                "num_atoms": len(self.fsr_indices),
                "path": "mof_fsr.cif",
            },
            "asr": {
                "num_atoms": len(self.asr_indices),
                "path": "mof_asr.cif",
            },
            "selected_to_original": list(mapping),
        }


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
        for site_id in sorted(sites):
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


def _canonical_bond_pairs(
    keys: Iterable[object],
    atom_count: int,
    label: str,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for key in keys:
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(
                not isinstance(index, Integral) or isinstance(index, bool)
                for index in key
            )
        ):
            raise ValueError(
                f"{label} key {key!r} must be a two-integer canonical pair "
                "(i, j) with i < j."
            )
        i, j = (int(index) for index in key)
        if i >= j:
            raise ValueError(
                f"{label} key {key!r} must be a canonical pair (i, j) with i < j."
            )
        if i < 0 or j >= atom_count:
            raise ValueError(
                f"{label} key {key!r} is out of range for {atom_count} atoms."
            )
        pairs.add((i, j))
    return pairs


def _validate_bond_network(atom_count: int, bonds: BondNetwork) -> None:
    if len(bonds.adjacency) != atom_count:
        raise ValueError("Bond network size does not match the atom count.")

    adjacency_pairs: set[tuple[int, int]] = set()
    for atom_index, neighbors in enumerate(bonds.adjacency):
        for neighbor in neighbors:
            if not isinstance(neighbor, Integral) or isinstance(neighbor, bool):
                raise ValueError(  # noqa: TRY004 - preserve public validation API
                    f"Adjacency entry {atom_index} -> {neighbor!r} must use an "
                    "integer atom index."
                )
            neighbor_index = int(neighbor)
            if neighbor_index == atom_index:
                raise ValueError(
                    f"Adjacency entry {atom_index} -> {neighbor_index} contains "
                    "a self-neighbor."
                )
            if neighbor_index < 0 or neighbor_index >= atom_count:
                raise ValueError(
                    f"Adjacency entry {atom_index} -> {neighbor_index} is out of "
                    f"range for {atom_count} atoms."
                )
            if atom_index not in bonds.adjacency[neighbor_index]:
                pair = tuple(sorted((atom_index, neighbor_index)))
                raise ValueError(f"Adjacency is not symmetric for bond {pair}.")
            if atom_index < neighbor_index:
                adjacency_pairs.add((atom_index, neighbor_index))

    vector_pairs = _canonical_bond_pairs(bonds.vectors, atom_count, "Vector")
    missing_vectors = adjacency_pairs - vector_pairs
    if missing_vectors:
        raise ValueError(
            f"Adjacency bonds missing from vectors: {sorted(missing_vectors)}."
        )
    missing_adjacency = vector_pairs - adjacency_pairs
    if missing_adjacency:
        raise ValueError(
            f"Vector bonds missing from adjacency: {sorted(missing_adjacency)}."
        )

    bond_type_pairs = _canonical_bond_pairs(
        bonds.bond_types,
        atom_count,
        "Bond type",
    )
    missing_type_vectors = bond_type_pairs - vector_pairs
    if missing_type_vectors:
        raise ValueError(
            f"Bond type bonds missing from vectors: {sorted(missing_type_vectors)}."
        )


class MofidSolventRemover:
    def __init__(self, atoms: ase.Atoms, bonds: BondNetwork) -> None:
        _validate_bond_network(len(atoms), bonds)
        self.atoms = atoms
        self.bonds = bonds
        self.symbols = atoms.get_chemical_symbols()
        self._inverse_cell: np.ndarray | None = None
        self._periodic_bond_by_pair = {
            record.pair: record for record in periodic_bonds(atoms, bonds)
        }

    def classify(self) -> SolventRemovalResult:
        self._inverse_cell = self._inverse_cell_matrix()
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
        connections = self_connections | self._crossing_connections(
            atom_to_site, offsets
        )
        free, bound = _simplify_topology(sites, connections)
        result = SolventRemovalResult(self.atoms.copy(), self.bonds, free, bound)
        result.validate()
        return result

    def _validate_initial_ownership(self, atom_to_site: dict[int, int]) -> None:
        expected = set(range(len(self.atoms)))
        if set(atom_to_site) != expected or len(atom_to_site) != len(self.atoms):
            raise ValueError(
                "Initial solvent topology does not own every atom exactly once."
            )

    def _inverse_cell_matrix(self) -> np.ndarray | None:
        if not np.any(self.atoms.pbc):
            return None
        cell = np.asarray(self.atoms.cell.array, dtype=float)
        if np.linalg.matrix_rank(cell) < 3:
            raise ValueError(
                "Cannot resolve periodic translations from a singular periodic cell."
            )
        return np.linalg.inv(cell)

    def _bond_shift(self, i: int, j: int) -> np.ndarray:
        pair = tuple(sorted((i, j)))
        return np.asarray(
            self._periodic_bond_by_pair[pair].oriented_shift(i, j),
            dtype=int,
        )

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
