"""Chemical classification constants for decomposition assignment rules."""

from __future__ import annotations

MOFID_NONMETALS: frozenset[str] = frozenset(
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


def is_mofid_metal(symbol: str) -> bool:
    """Return MOFid's InChI-style metal classification.

    Parameters
    ----------
    symbol
        Element symbol, for example ``"Zn"`` or ``"O"``.

    Returns
    -------
    bool
        ``True`` when the symbol is not in MOFid's nonmetal set.
    """

    return symbol not in MOFID_NONMETALS
