"""Conservative coordination screening without charge or bond-order inference.

These limits describe ordinary nonmetal coordination, not universal valence
rules. Passing does not establish neutrality; valid ionic, donor-adduct, and
multicenter environments can be excluded by this generation policy.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from ase.data import chemical_symbols
from pymatgen.core import Element

NONMETAL_NEIGHBOR_LIMITS = {"H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1}
_LIMITS = np.array([NONMETAL_NEIGHBOR_LIMITS.get(s, 0) for s in chemical_symbols])
# Match the composer's existing metal/ligand distinction. X is an unknown
# future partner, not a nonmetal atom and not a metal atom.
_NONMETAL = np.array(
    [False] + [not Element.from_Z(z).is_metal for z in range(1, len(chemical_symbols))]
)


def nonmetal_neighbor_counts(numbers, bonds):
    """Count real nonmetal neighbors; each periodic image bond counts separately."""
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    left, right = bonds.T
    return np.bincount(
        np.concatenate(
            (left[_NONMETAL[numbers[right]]], right[_NONMETAL[numbers[left]]])
        ),
        minlength=len(numbers),
    )


def validate_coordination(numbers, counts, *, context, indices=None):
    """Reject screened atoms above their ordinary nonmetal-neighbor limit."""
    if indices is None:
        indices = np.arange(len(numbers))
    else:
        indices = np.asarray(indices, dtype=int)
    limits = _LIMITS[numbers[indices]]
    bad = indices[(limits > 0) & (counts[indices] > limits)]
    if len(bad):
        i = int(bad[0])
        symbol = chemical_symbols[numbers[i]]
        raise ValueError(
            f"Coordination screen rejected {context}: atom {i} ({symbol}) has "
            f"{counts[i]} nonmetal neighbors; ordinary limit {_LIMITS[numbers[i]]}. "
            "Charge-dependent or unusual bonding is outside unrestricted generation."
        )


@dataclass(frozen=True)
class CoordinationProfile:
    """Reusable graph analysis; geometry and bond labels are deliberately absent."""

    nonmetal_counts: np.ndarray
    error: str | None


def coordination_profile(numbers, bonds):
    """Cache by graph content, so edits invalidate the result and copies reuse it."""
    return _coordination_profile(
        np.asarray(numbers, dtype=np.int64).tobytes(),
        np.asarray(bonds, dtype=np.int64).reshape(-1, 2).tobytes(),
    )


@lru_cache(maxsize=4096)
def _coordination_profile(number_bytes, bond_bytes):
    numbers = np.frombuffer(number_bytes, dtype=np.int64)
    bonds = np.frombuffer(bond_bytes, dtype=np.int64).reshape(-1, 2)
    if np.any(bonds < 0) or np.any(bonds >= len(numbers)):
        raise ValueError("Building-block bond references an invalid atom index.")
    if np.any(bonds[:, 0] == bonds[:, 1]):
        raise ValueError("A finite building block cannot contain self-bonds.")
    if len(np.unique(np.sort(bonds, axis=1), axis=0)) != len(bonds):
        raise ValueError("A finite building block cannot contain duplicate bonds.")

    is_x = numbers == 0
    degrees = np.bincount(bonds.ravel(), minlength=len(numbers))
    if np.any(degrees[is_x] != 1) or np.any(is_x[bonds].all(axis=1)):
        raise ValueError("Each X port must have exactly one real-atom anchor.")
    counts = nonmetal_neighbor_counts(numbers, bonds)
    counts.setflags(write=False)
    error = None
    try:
        validate_coordination(numbers, counts, context="building block")
    except ValueError as exc:
        error = str(exc)
    return CoordinationProfile(counts, error)


def add_fused_coordination(numbers, counts, atom_i, atom_j):
    """Replace two X sites by their actual partners, including periodic self-edges."""
    counts[atom_i] += _NONMETAL[numbers[atom_j]]
    counts[atom_j] += _NONMETAL[numbers[atom_i]]
