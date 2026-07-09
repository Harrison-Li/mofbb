"""Data models for capped fragments and collapsed topology records."""

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

    def as_dict(self) -> dict[str, int | list[int]]:
        """Return JSON-serializable site metadata."""

        return {
            "x_index": self.x_index,
            "anchor_index": self.anchor_index,
            "source_cut_bond": list(self.source_cut_bond),
            "internal_source_index": self.internal_source_index,
            "external_source_index": self.external_source_index,
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

    @property
    def formula(self) -> str:
        """Return the fragment chemical formula."""

        return self.atoms.get_chemical_formula()

    @property
    def num_atoms(self) -> int:
        """Return the number of atoms, including dummy ``X`` atoms."""

        return len(self.atoms)

    @property
    def num_x(self) -> int:
        """Return the number of PORMAKE dummy connection atoms."""

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

    def as_dict(self, path: str | Path | None = None) -> dict[str, object]:
        """Return JSON-serializable fragment metadata."""

        return {
            "kind": self.kind,
            "index": self.index,
            "formula": self.formula,
            "num_atoms": self.num_atoms,
            "num_x": self.num_x,
            "source_indices": list(self.source_indices),
            "fingerprint": self.fingerprint,
            "x_sites": [site.as_dict() for site in self.x_sites],
            "path": None if path is None else str(path),
        }


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
