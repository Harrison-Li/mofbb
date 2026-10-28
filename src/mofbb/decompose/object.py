"""Data object for decomposition fragments and topology records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ase

from mofbb.io.xyz import write_bb_xyz
import numpy as np

from mofbb.common.pbc import CellShift
from mofbb.decompose.constants import is_metal

@dataclass
class Site:
    """A node or linker atom group in a collapsed connectivity graph."""

    role: Literal["node", "linker"]
    atom_indices: set[int]


@dataclass(frozen=True, order=True)
class Edge:
    """A connection between site IDs with an integer cell translation."""

    left: int
    right: int
    shift: CellShift


@dataclass(frozen=True)
class XSite:
    """One dummy connection site on a capped fragment."""

    x_index: int
    anchor_index: int
    source_cut_bond: tuple[int, int]
    internal_source_index: int
    external_source_index: int
    source_cut_shift: tuple[int, int, int] = (0, 0, 0)
    external_image_shift: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class ConnectionPort:
    """One stable group of physical connection sites to a fragment slot."""

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


@dataclass
class Fragment:
    """One capped node or linker building-block fragment."""

    kind: str
    index: int
    source_indices: tuple[int, ...]
    atoms: ase.Atoms
    bonds: tuple[tuple[int, int, str], ...]
    fingerprint: str
    x_sites: tuple[XSite, ...] = ()
    source_image_offsets: tuple[tuple[int, tuple[int, int, int]], ...] = ()
    topology_kind: str | None = None
    connection_ports: tuple[ConnectionPort, ...] = ()
    real_atom_centroid_frac: tuple[float, float, float] | None = None

    @property
    def formula(self) -> str:
        return self.atoms.get_chemical_formula()

    @property
    def num_atoms(self) -> int:
        return len(self.atoms)

    @property
    def num_x(self) -> int:
        if self.x_sites:
            return len(self.x_sites)
        return self.atoms.get_chemical_symbols().count("X")

    def _x_indices(self) -> list[int]:
        if self.x_sites:
            return [site.x_index for site in self.x_sites]
        return [
            i for i, symbol in enumerate(self.atoms.get_chemical_symbols())
            if symbol == "X"
        ]

    def identify_kind(self) -> str:
        """Infer the physical kind from real elements and physical X sites.

        Return ``metal_node`` for metal-containing fragments, otherwise ``edge``
        for two X sites and ``organic_node`` for other counts. This does not
        overwrite the assigned kind or perform connection-port classification.
        """
        x_indices = set(self._x_indices())
        if any(
            is_metal(symbol)
            for i, symbol in enumerate(self.atoms.get_chemical_symbols())
            if i not in x_indices
        ):
            return "metal_node"
        return "edge" if len(x_indices) == 2 else "organic_node"

    def get_bb_centroid(self) -> np.ndarray:
        """Return the real-atom centroid in local Cartesian coordinates (angstroms)."""
        x_indices = set(self._x_indices())
        indices = [i for i in range(len(self.atoms)) if i not in x_indices]
        if not indices:
            raise ValueError("Cannot calculate a centroid without real atoms.")
        return self.atoms.positions[indices].mean(axis=0)

    def get_x_centroid(self) -> np.ndarray:
        """Return the local Cartesian X centroid (angstroms), or real-atom centroid."""
        indices = self._x_indices()
        if not indices:
            return self.get_bb_centroid()
        return self.atoms.positions[indices].mean(axis=0)

    def write_xyz(self, path: Path) -> Path:
        """Write a PORMAKE-readable extended XYZ file and return its path."""
        return write_bb_xyz(path, self.atoms, self.bonds)


@dataclass(frozen=True)
class TopologyVertex:
    """One collapsed node or linker vertex in the CGD topology."""

    kind: str
    index: int
    coordination: int
    frac_position: tuple[float, float, float]


@dataclass(frozen=True)
class TopologyEdge:
    """One edge between collapsed topology vertices."""

    begin_kind: str
    begin_index: int
    end_kind: str
    end_index: int
    begin_frac: tuple[float, float, float]
    end_frac: tuple[float, float, float]
    center_frac: tuple[float, float, float]
    source_cut_bond: tuple[int, int] | None = None
    source_cut_shift: tuple[int, int, int] = (0, 0, 0)
    slot_translation: tuple[int, int, int] = (0, 0, 0)
