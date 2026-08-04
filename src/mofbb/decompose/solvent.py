"""MOFid-style free and bound solvent removal."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import ase
import ase.io
import numpy as np
from ase.geometry import cell_to_cellpar

from .assignment import NodeAtomAssigner
from .bonding import BondNetwork, subset_bonded_structure
from .fragments import connected_components
from .pbc import CellShift, periodic_bonds

SolventMode: TypeAlias = Literal["none", "free", "all"]
ActiveSolventMode: TypeAlias = Literal["free", "all"]
SiteRole: TypeAlias = Literal["node", "linker"]


def _write_cif(path: Path, atoms: ase.Atoms, indices: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if indices:
        ase.io.write(path, atoms[list(indices)], format="cif", wrap=False)
        return

    name = re.sub(r"[^A-Za-z0-9_]", "_", path.stem)
    lines = [f"data_{name}"]
    if atoms.cell.rank == 3:
        a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
        lines += [
            f"_cell_length_a {a:.15g}",
            f"_cell_length_b {b:.15g}",
            f"_cell_length_c {c:.15g}",
            f"_cell_angle_alpha {alpha:.15g}",
            f"_cell_angle_beta {beta:.15g}",
            f"_cell_angle_gamma {gamma:.15g}",
            "_space_group_name_H-M_alt 'P 1'",
        ]
    path.write_text("\n".join([*lines, "_mofbb_atom_count 0", ""]), encoding="utf-8")


@dataclass(frozen=True)
class SolventRemovalResult:
    """Classified solvent and atom selections for one structure."""

    original_atoms: ase.Atoms
    original_bonds: BondNetwork
    free_solvent_components: tuple[tuple[int, ...], ...]
    bound_solvent_components: tuple[tuple[int, ...], ...]

    @staticmethod
    def _flatten(components: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        return tuple(sorted(atom for component in components for atom in component))

    @property
    def free_solvent_indices(self) -> tuple[int, ...]:
        return self._flatten(self.free_solvent_components)

    @property
    def bound_solvent_indices(self) -> tuple[int, ...]:
        return self._flatten(self.bound_solvent_components)

    def kept_indices(self, mode: ActiveSolventMode) -> tuple[int, ...]:
        if mode not in {"free", "all"}:
            raise ValueError(f"Unsupported active solvent mode: {mode!r}.")
        removed = set(self.free_solvent_indices)
        if mode == "all":
            removed.update(self.bound_solvent_indices)
        return tuple(i for i in range(len(self.original_atoms)) if i not in removed)

    @property
    def fsr_indices(self) -> tuple[int, ...]:
        return self.kept_indices("free")

    @property
    def asr_indices(self) -> tuple[int, ...]:
        return self.kept_indices("all")

    def subset(
        self,
        mode: ActiveSolventMode,
    ) -> tuple[ase.Atoms, BondNetwork, tuple[int, ...]]:
        indices = self.kept_indices(mode)
        if not indices:
            raise ValueError(f"Solvent removal deleted the entire structure ({mode}).")
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
        mapping = tuple(selected_to_original)
        expected = self.kept_indices(mode)
        if mapping != expected:
            raise ValueError(
                "Solvent selected-to-original mapping is inconsistent with "
                f"mode {mode!r}: expected {expected}, received {mapping}."
            )
        rank = self.original_atoms.cell.rank
        periodic = tuple(bool(value) for value in self.original_atoms.pbc)
        if not ((rank == 3 and all(periodic)) or (rank == 0 and not any(periodic))):
            raise ValueError(
                "CIF output requires either a rank-3 cell with full periodicity "
                "or a rank-0 cell with no periodicity."
            )
        return mapping

    def write(
        self,
        output_dir: str | Path,
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
            _write_cif(output_dir / filename, self.original_atoms, indices)

        def solvent_summary(
            components: tuple[tuple[int, ...], ...],
            filename: str,
        ) -> dict[str, object]:
            return {
                "num_components": len(components),
                "num_atoms": sum(map(len, components)),
                "components": [list(component) for component in components],
                "path": filename,
            }

        return {
            "mode": mode,
            "free": solvent_summary(
                self.free_solvent_components,
                "free_solvent.cif",
            ),
            "bound": solvent_summary(
                self.bound_solvent_components,
                "bound_solvent.cif",
            ),
            "fsr": {"num_atoms": len(self.fsr_indices), "path": "mof_fsr.cif"},
            "asr": {"num_atoms": len(self.asr_indices), "path": "mof_asr.cif"},
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


def _simplify_topology(
    source_sites: dict[int, _TopologySite],
    source_connections: set[_TopologyConnection],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    sites = {
        key: _TopologySite(key, site.role, set(site.atom_indices))
        for key, site in source_sites.items()
    }
    connections = set(source_connections)
    free: list[tuple[int, ...]] = []
    bound: list[tuple[int, ...]] = []

    def incident(site_id: int) -> list[_TopologyConnection]:
        return [
            edge
            for edge in connections
            if edge.left == site_id or edge.right == site_id
        ]

    def degree(site_id: int) -> int:
        return sum(2 if edge.left == edge.right else 1 for edge in incident(site_id))

    while True:
        for site_id in sorted(sites):
            site = sites[site_id]
            site_degree = degree(site_id)
            if site_degree == 0:
                free.append(tuple(sorted(site.atom_indices)))
                del sites[site_id]
                break
            if site_degree != 1:
                continue
            edge = incident(site_id)[0]
            neighbor = edge.right if edge.left == site_id else edge.left
            if site.role == "node" and degree(neighbor) == 1:
                continue
            if site.role == "node":
                sites[neighbor].atom_indices.update(site.atom_indices)
            else:
                bound.append(tuple(sorted(site.atom_indices)))
            connections.remove(edge)
            del sites[site_id]
            break
        else:
            break

    order = lambda component: component[0]
    return tuple(sorted(free, key=order)), tuple(sorted(bound, key=order))


class MofidSolventRemover:
    """Classify free and bound solvent from periodic MOF connectivity."""

    def __init__(self, atoms: ase.Atoms, bonds: BondNetwork) -> None:
        if len(atoms) != len(bonds.adjacency):
            raise ValueError("Bond network size does not match the atom count.")
        self.atoms = atoms
        self.bonds = bonds
        self.records = {record.pair: record for record in periodic_bonds(atoms, bonds)}

    def classify(self) -> SolventRemovalResult:
        nodes = set(
            NodeAtomAssigner(
                self.atoms.get_chemical_symbols(),
                self.bonds,
            ).assign()
        )
        roles = (
            ("node", nodes),
            ("linker", set(range(len(self.atoms))) - nodes),
        )
        components = [
            (role, component)
            for role, indices in roles
            for component in connected_components(indices, self.bonds.pairs)
        ]
        components.sort(key=lambda item: min(item[1]))
        sites = {
            site_id: _TopologySite(site_id, role, component)
            for site_id, (role, component) in enumerate(components)
        }
        site_by_atom = {
            atom: site_id
            for site_id, site in sites.items()
            for atom in site.atom_indices
        }
        offsets, connections = self._component_offsets(sites)
        connections.update(self._crossing_connections(site_by_atom, offsets))
        free, bound = _simplify_topology(sites, connections)
        return SolventRemovalResult(self.atoms.copy(), self.bonds, free, bound)

    def _shift(self, i: int, j: int) -> np.ndarray:
        return np.asarray(self.records[tuple(sorted((i, j)))].oriented_shift(i, j))

    def _component_offsets(
        self,
        sites: dict[int, _TopologySite],
    ) -> tuple[dict[int, np.ndarray], set[_TopologyConnection]]:
        offsets: dict[int, np.ndarray] = {}
        connections: set[_TopologyConnection] = set()
        for site_id, site in sites.items():
            root = min(site.atom_indices)
            offsets[root] = np.zeros(3, dtype=int)
            queue = deque([root])
            while queue:
                atom = queue.popleft()
                for neighbor in sorted(self.bonds.adjacency[atom] & site.atom_indices):
                    expected = offsets[atom] + self._shift(atom, neighbor)
                    if neighbor not in offsets:
                        offsets[neighbor] = expected
                        queue.append(neighbor)
                    elif np.any(cycle := expected - offsets[neighbor]):
                        shift = tuple(int(value) for value in cycle)
                        reverse = tuple(-value for value in shift)
                        connections.add(
                            _TopologyConnection(site_id, site_id, min(shift, reverse))
                        )
        return offsets, connections

    def _crossing_connections(
        self,
        site_by_atom: dict[int, int],
        offsets: dict[int, np.ndarray],
    ) -> set[_TopologyConnection]:
        connections = set()
        for i, j in self.bonds.pairs:
            left, right = site_by_atom[i], site_by_atom[j]
            if left == right:
                continue
            shift = self._shift(i, j) + offsets[i] - offsets[j]
            if left > right:
                left, right, shift = right, left, -shift
            connections.add(
                _TopologyConnection(
                    left,
                    right,
                    tuple(int(value) for value in shift),
                )
            )
        return connections
