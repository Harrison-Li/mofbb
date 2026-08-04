"""Data object for decomposition fragments and topology records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ase


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
class LogicalPort:
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
    logical_kind: str | None = None
    logical_ports: tuple[LogicalPort, ...] = ()
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

    def write_xyz(self, path: Path) -> Path:
        """Write a PORMAKE-readable extended XYZ file and return the path.

        The first comment line contains whitespace-separated local indices of
        dummy ``X`` atoms. Bond records are appended after the coordinate block.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        symbols = self.atoms.get_chemical_symbols()
        x_indices = [str(i) for i, symbol in enumerate(symbols) if symbol == "X"]
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"{len(self.atoms)}\n")
            handle.write(" ".join(x_indices) + "\n")
            for symbol, coord in zip(symbols, self.atoms.positions):
                handle.write(
                    f"{symbol:<2s} {coord[0]: .10f} {coord[1]: .10f} {coord[2]: .10f}\n"
                )
            for i, j, bond_type in self.bonds:
                handle.write(f"{i:4d} {j:4d} {bond_type}\n")
        return path


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
