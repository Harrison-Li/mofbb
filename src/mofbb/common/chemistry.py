'''
Conservative coordination screening without charge or bond-order inference.

These limits describe ordinary nonmetal coordination, not universal valence
rules. Passing does not establish neutrality; valid ionic, donor-adduct, and
multicenter environments can be excluded by this generation policy.
'''

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
    '''
    Count each atom's real nonmetal neighbors.

    Each bond record counts separately, including bonds to periodic images.
    Dummy X atoms and metals do not contribute to their partners' counts.

    Args:
        numbers (numpy.ndarray):
            (n,) atomic numbers; zero denotes an X port.
        bonds (array-like):
            (m, 2) zero-based atom pairs.

    Returns:
        An (n,) integer array of neighbor counts.
    '''
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    left, right = bonds.T
    return np.bincount(
        np.concatenate(
            (left[_NONMETAL[numbers[right]]], right[_NONMETAL[numbers[left]]])
        ),
        minlength=len(numbers),
    )


def validate_coordination(numbers, counts, *, context, indices=None):
    '''
    Reject atoms that exceed the configured nonmetal-neighbor limits.

    Args:
        numbers (numpy.ndarray):
            (n,) atomic numbers.
        counts (numpy.ndarray):
            (n,) nonmetal-neighbor counts.
        context (str):
            Description included in the error message.
        indices (array-like or None):
            Atom indices to check; check every atom when omitted.

    Returns:
        None. Raises ValueError for the first atom that exceeds its limit.
    '''
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
    '''
    Store a building block's nonmetal-neighbor counts and screening result.

    Args:
        nonmetal_counts (numpy.ndarray):
            Read-only (n,) array of nonmetal-neighbor counts.
        error (str or None):
            Coordination-screen error message, or None when the screen passes.
    '''

    nonmetal_counts: np.ndarray
    error: str | None


def coordination_profile(numbers, bonds):
    '''
    Check building-block bonds and cache their coordination counts.

    Malformed bonds or X ports raise ValueError. Coordination-limit failures
    are stored in the result so callers can decide how to handle them.

    Args:
        numbers (array-like):
            (n,) atomic numbers; zero denotes an X port.
        bonds (array-like):
            (m, 2) zero-based atom pairs for a finite building block.

    Returns:
        A cached CoordinationProfile with read-only counts and any screen error.
    '''
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
    paired = np.zeros(len(numbers), dtype=bool)
    for x in np.flatnonzero(is_x & (degrees == 2)):
        neighbors = bonds[np.any(bonds == x, axis=1)]
        conn_atoms = neighbors[neighbors != x]
        paired[x] = len(conn_atoms) == 2 and np.all(numbers[conn_atoms] == 8)
    if np.any((degrees[is_x] != 1) & ~paired[is_x]) or np.any(is_x[bonds].all(axis=1)):
        raise ValueError("Each X port must have exactly one real connection atom.")
    counts = nonmetal_neighbor_counts(numbers, bonds)
    counts.setflags(write=False)
    error = None
    try:
        validate_coordination(numbers, counts, context="building block")
    except ValueError as exc:
        error = str(exc)
    return CoordinationProfile(counts, error)


def add_fused_coordination(numbers, counts, atom_i, atom_j):
    '''
    Update neighbor counts after replacing X ports with a real bond.

    Args:
        numbers (numpy.ndarray):
            (n,) atomic numbers.
        counts (numpy.ndarray):
            (n,) counts to update in place.
        atom_i (int):
            First real atom of the new bond.
        atom_j (int):
            Second real atom; may equal atom_i for a periodic self-edge.

    Returns:
        None. Each endpoint gains one count if its partner is a nonmetal.
    '''
    counts[atom_i] += _NONMETAL[numbers[atom_j]]
    counts[atom_j] += _NONMETAL[numbers[atom_i]]
