"""Chemical classification constants for decomposition assignment rules."""

from __future__ import annotations

NONMETALS: frozenset[str] = frozenset(
    {
        "H",
        "He",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "As",
        "Se",
        "Br",
        "Kr",
        "Te",
        "I",
        "Xe",
        "At",
        "Rn",
    }
)


def is_metal(symbol: str) -> bool:
    """Return whether MOFid classifies an element as a metal."""

    return symbol not in NONMETALS
