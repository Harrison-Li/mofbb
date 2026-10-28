"""Read atomistic structures using ASE's format support."""

from pathlib import Path

import ase
import ase.io


def read_atoms(input_path: str | Path) -> ase.Atoms:
    """Read the last frame, treating the input as a literal file path."""
    return ase.io.read(input_path, do_not_split_by_at_sign=True)
