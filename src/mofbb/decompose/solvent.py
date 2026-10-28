"""MOFid-style free and bound solvent removal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import ase
import numpy as np
from mofbb.io.cif import write_cif

from mofbb.decompose.assignment import NodeAtomAssigner
from mofbb.decompose.bonding import BondNetwork, subset_bonded_structure
from mofbb.decompose.constants import is_metal
from mofbb.decompose.fragments import connected_components
from mofbb.decompose.object import Edge, Site
from mofbb.decompose.pbc import periodic_bonds

SolventMode: TypeAlias = Literal["none", "free", "all"]
ActiveSolventMode: TypeAlias = Literal["free", "all"]

# Two independently occupied oxygen sites closer than a real O--O bond cannot
# coexist.  Treat such sites as a disordered coordinated-solvent fragment when
# they are terminal alternatives on the same metal.
_SPLIT_O_MAX_DISTANCE = 1.05


@dataclass(frozen=True)
class SolventResult:
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
        self, mode: ActiveSolventMode
    ) -> tuple[ase.Atoms, BondNetwork, tuple[int, ...]]:
        indices = self.kept_indices(mode)
        if not indices:
            raise ValueError(f"Solvent removal deleted the entire structure ({mode}).")
        return subset_bonded_structure(
            self.original_atoms, self.original_bonds, indices
        )

    def write(
        self,
        output_dir: str | Path,
        mode: ActiveSolventMode,
        selected_to_original: tuple[int, ...],
    ) -> dict[str, object]:
        output_dir = Path(output_dir)
        selections = {
            "free_solvent.cif": self.free_solvent_indices,
            "bound_solvent.cif": self.bound_solvent_indices,
            "mof_fsr.cif": self.fsr_indices,
            "mof_asr.cif": self.asr_indices,
        }
        for filename, indices in selections.items():
            write_cif(output_dir / filename, self.original_atoms[list(indices)])

        def summary(
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
            "free": summary(self.free_solvent_components, "free_solvent.cif"),
            "bound": summary(self.bound_solvent_components, "bound_solvent.cif"),
            "fsr": {"num_atoms": len(self.fsr_indices), "path": "mof_fsr.cif"},
            "asr": {"num_atoms": len(self.asr_indices), "path": "mof_asr.cif"},
            "selected_to_original": list(selected_to_original),
        }


def _simplify_topology(
    source_sites: dict[int, Site],
    source_connections: set[Edge],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    sites = {
        key: Site(site.role, set(site.atom_indices))
        for key, site in source_sites.items()
    }
    connections = set(source_connections)
    free: list[tuple[int, ...]] = []
    bound: list[tuple[int, ...]] = []

    def incident(site_id: int) -> list[Edge]:
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

    return tuple(sorted(free)), tuple(sorted(bound))


class SolventRemover:
    """Classify free and bound solvent."""

    def __init__(self, atoms: ase.Atoms, bonds: BondNetwork) -> None:
        self.atoms = atoms
        self.bonds = bonds
        self.records = {record.pair: record for record in periodic_bonds(atoms, bonds)}

    def classify(self) -> SolventResult:
        nodes = set(
            NodeAtomAssigner(self.atoms.get_chemical_symbols(), self.bonds).assign()
        )
        # The node assigner deliberately starts from every direct metal
        # neighbor.  Pull recognizable coordinated solvent back out before
        # simplifying the framework topology so that it is classified as
        # bound solvent rather than part of the metal node.
        coordinated_solvent = self._coordinated_atoms()
        nodes.difference_update(coordinated_solvent)
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
            site_id: Site(role, component)
            for site_id, (role, component) in enumerate(components)
        }
        site_by_atom = {
            atom: site_id
            for site_id, site in sites.items()
            for atom in site.atom_indices
        }
        offsets, connections = self._site_offsets(sites)
        connections.update(self._site_edges(site_by_atom, offsets))
        coordinated_site_ids = {site_by_atom[atom] for atom in coordinated_solvent}
        coordinated_bound = tuple(
            tuple(sorted(sites[site_id].atom_indices))
            for site_id in sorted(coordinated_site_ids)
        )
        for site_id in coordinated_site_ids:
            del sites[site_id]
        connections = {
            connection
            for connection in connections
            if connection.left not in coordinated_site_ids
            and connection.right not in coordinated_site_ids
        }
        free, bound = _simplify_topology(sites, connections)
        bound = tuple(sorted((*bound, *coordinated_bound)))
        return SolventResult(self.atoms.copy(), self.bonds, free, bound)

    def _coordinated_atoms(self) -> set[int]:
        symbols = self.atoms.get_chemical_symbols()
        solvent_atoms: set[int] = set()

        # Explicit terminal aqua ligand: M--O(H)2.  Requiring one metal and
        # terminal hydrogens preserves bridging water, hydroxide, and oxo.
        for oxygen, symbol in enumerate(symbols):
            if symbol != "O":
                continue
            neighbors = self.bonds.adjacency[oxygen]
            metals = {atom for atom in neighbors if is_metal(symbols[atom])}
            hydrogens = {atom for atom in neighbors if symbols[atom] == "H"}
            other = neighbors - metals - hydrogens
            if (
                len(metals) == 1
                and len(hydrogens) == 2
                and not other
                and all(
                    self.bonds.adjacency[atom] <= metals | {oxygen}
                    for atom in hydrogens
                )
            ):
                solvent_atoms.update({oxygen, *hydrogens})

        # Crystallographic disorder may encode one aqua oxygen as two
        # mutually exclusive sites.  Distance bonding then creates an
        # impossible O--O contact and makes both sites look like node atoms.
        for left, right in self.bonds.pairs:
            if symbols[left] != "O" or symbols[right] != "O":
                continue
            if np.linalg.norm(self.bonds.vector(left, right)) > (_SPLIT_O_MAX_DISTANCE):
                continue

            left_neighbors = self.bonds.adjacency[left]
            right_neighbors = self.bonds.adjacency[right]
            left_metals = {atom for atom in left_neighbors if is_metal(symbols[atom])}
            right_metals = {atom for atom in right_neighbors if is_metal(symbols[atom])}
            if len(left_metals) != 1 or left_metals != right_metals:
                continue

            allowed = left_metals | {left, right}
            if any(
                symbols[atom] != "H" and atom not in allowed
                for atom in left_neighbors | right_neighbors
            ):
                continue

            hydrogens = {
                atom
                for atom in left_neighbors | right_neighbors
                if symbols[atom] == "H"
            }
            if any(
                self.bonds.adjacency[atom] - left_metals - {left, right}
                for atom in hydrogens
            ):
                continue
            solvent_atoms.update({left, right, *hydrogens})

        return solvent_atoms

    def _shift(self, i: int, j: int) -> np.ndarray:
        return np.asarray(self.records[tuple(sorted((i, j)))].oriented_shift(i, j))

    def _site_offsets(
        self,
        sites: dict[int, Site],
    ) -> tuple[dict[int, np.ndarray], set[Edge]]:
        offsets: dict[int, np.ndarray] = {}
        connections: set[Edge] = set()
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
                        connections.add(Edge(site_id, site_id, min(shift, reverse)))
        return offsets, connections

    def _site_edges(
        self,
        site_by_atom: dict[int, int],
        offsets: dict[int, np.ndarray],
    ) -> set[Edge]:
        connections = set()
        for i, j in self.bonds.pairs:
            left, right = site_by_atom[i], site_by_atom[j]
            if left == right:
                continue
            shift = self._shift(i, j) + offsets[i] - offsets[j]
            if left > right:
                left, right, shift = right, left, -shift
            connections.add(
                Edge(
                    left,
                    right,
                    tuple(int(value) for value in shift),
                )
            )
        return connections
