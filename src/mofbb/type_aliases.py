"""Shared type aliases used throughout the package."""

from __future__ import annotations

from typing import Literal, TypeAlias

AtomIndex: TypeAlias = int
Bond: TypeAlias = tuple[int, int]
BondRecord: TypeAlias = tuple[int, int, str]
FragmentKind: TypeAlias = Literal["node", "linker"]
FractionalPoint: TypeAlias = tuple[float, float, float]
CellParameters: TypeAlias = tuple[float, float, float, float, float, float]
