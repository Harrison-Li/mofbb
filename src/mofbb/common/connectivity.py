'''
Check 3D connectivity using bonds to explicit periodic atom images.
'''

import networkx as nx
import numpy as np
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from pymatgen.core import Element

from mofbb.common.pbc import PeriodicBond, analyze_component

CONNECTIVITY_POLICY_VERSION = 2


def bond_arrays(num_atoms, bonds, bond_shifts):
    '''
    Normalize bond arrays and check atom indices and lattice shifts.

    Args:
        num_atoms (int):
            Number of atoms referenced by the bonds.
        bonds (array-like):
            (m, 2) zero-based atom pairs.
        bond_shifts (array-like):
            (m, 3) finite integer translations from the first endpoint to the second.

    Returns:
        A pair of integer arrays with shapes (m, 2) and (m, 3). Invalid indices,
        noninteger shifts, or zero-image self-bonds raise ValueError.
    '''
    bonds = np.asarray(bonds, dtype=int).reshape(-1, 2)
    shifts = np.asarray(bond_shifts).reshape(-1, 3)
    if len(bonds) != len(shifts) or not np.isfinite(shifts).all():
        raise ValueError("Every periodic bond must have a finite lattice shift.")
    if not np.array_equal(shifts, np.rint(shifts)):
        raise ValueError("Bond lattice shifts must be integers.")
    if np.any(bonds < 0) or np.any(bonds >= num_atoms):
        raise ValueError("Periodic bond references an invalid atom index.")
    if np.any((bonds[:, 0] == bonds[:, 1]) & np.all(shifts == 0, axis=1)):
        raise ValueError("Zero-image self-bond is not a physical bond.")
    return bonds, shifts.astype(int)


def mic_shifts(atoms, bonds):
    '''
    Find minimum-image cell shifts for bonds between distinct atoms.

    Args:
        atoms (ase.Atoms):
            Structure with Cartesian positions in angstroms and its unit cell.
        bonds (array-like):
            (m, 2) zero-based atom pairs. Self-bonds need explicit shifts instead.

    Returns:
        An (m, 3) integer array of image shifts, or zeros for a nonperiodic structure.
    '''
    bonds = np.asarray(bonds, dtype=int).reshape(-1, 2)
    if np.any(bonds[:, 0] == bonds[:, 1]):
        raise ValueError("Periodic self-bonds require explicit lattice shifts.")
    if not len(bonds):
        return np.empty((0, 3), dtype=int)
    direct = atoms.positions[bonds[:, 1]] - atoms.positions[bonds[:, 0]]
    vectors, _ = find_mic(direct, atoms.cell, atoms.pbc)
    if not np.any(atoms.pbc):
        return np.zeros((len(bonds), 3), dtype=int)
    return np.rint((vectors - direct) @ np.linalg.inv(atoms.cell)).astype(int)


def periodic_components(num_atoms, bonds, bond_shifts):
    '''
    Find connected components and count their independent periodic directions.

    Args:
        num_atoms (int):
            Total number of atoms, including isolated atoms.
        bonds (array-like):
            (m, 2) zero-based atom pairs.
        bond_shifts (array-like):
            (m, 3) integer lattice shifts for the corresponding bonds.

    Returns:
        A list of dictionaries with num_atoms and rank for each component.
        Rank 0, 1, 2, or 3 denotes a finite fragment, chain, layer, or 3D network.
    '''
    bonds, shifts = bond_arrays(num_atoms, bonds, bond_shifts)
    records = tuple(
        PeriodicBond(int(i), int(j), tuple(map(int, shift)))
        for (i, j), shift in zip(bonds, shifts)
    )
    # The simple graph only identifies components. Keep all image edges for
    # rank analysis, including parallel edges and bonds to an atom's own image.
    graph = nx.Graph()
    graph.add_nodes_from(range(num_atoms))
    graph.add_edges_from(bonds)
    return [
        {"num_atoms": len(nodes), "rank": analyze_component(nodes, records).rank}
        for nodes in nx.connected_components(graph)
    ]


def validate_3d_connectivity(
    atoms, bonds=None, bond_shifts=None, *, bond_scale=1.2,
    minimum_distance_cutoff=None,
):
    '''
    Require every bonded component to extend in three periodic directions.

    Overlong bonds are excluded. Coordinate-based inference excludes metal-metal
    bonds; explicitly supplied metal-metal bonds are retained if short enough.

    Args:
        atoms (ase.Atoms):
            Structure with Cartesian positions in angstroms and its unit cell.
        bonds (array-like or None):
            Optional (m, 2) array of zero-based atom pairs; infer bonds when omitted.
        bond_shifts (array-like or None):
            Optional (m, 3) integer cell shifts from the first atom to the second; infer
            minimum-image shifts when omitted.
        bond_scale (float):
            Multiplier for the sum of the two atoms' covalent radii. Defaults to 1.2.
        minimum_distance_cutoff (float or None):
            Optional distance in angstroms for reporting the closest contact below
            this cutoff. Only available when bonds are inferred.

    Returns:
        A dictionary of connectivity policy, bond scale, component sizes and
        ranks, plus the requested minimum-distance result. Invalid geometry or
        any component with rank below three raises ValueError.
    '''
    if not np.isfinite(bond_scale) or bond_scale <= 0:
        raise ValueError("Connectivity bond scale must be positive and finite.")
    if (
        not len(atoms)
        or not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell).all()
        or atoms.cell.volume <= 0
    ):
        raise ValueError("3D connectivity requires nonempty, finite periodic geometry.")
    if not np.all(atoms.pbc):
        raise ValueError("3D connectivity requires three periodic cell directions.")
    if minimum_distance_cutoff is not None and (
        bonds is not None
        or not np.isfinite(minimum_distance_cutoff)
        or minimum_distance_cutoff <= 0
    ):
        raise ValueError(
            "Minimum-distance reporting requires inferred bonds and a positive finite cutoff."
        )
    quality = {}
    radii = covalent_radii[atoms.numbers]
    if bonds is None:
        cutoffs = bond_scale * radii
        search_cutoffs = cutoffs
        if minimum_distance_cutoff is not None:
            search_cutoffs = np.maximum(cutoffs, minimum_distance_cutoff / 2)
        i, j, distances, shifts = neighbor_list("ijdS", atoms, search_cutoffs)
        if minimum_distance_cutoff is not None:
            nearby = distances[distances < minimum_distance_cutoff]
            quality["minimum_atom_distance_angstrom"] = (
                float(nearby.min()) if len(nearby) else None
            )
        metals = np.array([Element(symbol).is_metal for symbol in atoms.symbols])
        # Drop reverse edges; retain every image, including periodic self-bonds.
        keep = (i <= j) & (distances < cutoffs[i] + cutoffs[j])
        keep &= ~(metals[i] & metals[j])
        bonds, bond_shifts = np.column_stack((i[keep], j[keep])), shifts[keep]
    else:
        if bond_shifts is None:
            bond_shifts = mic_shifts(atoms, bonds)
        bonds, bond_shifts = bond_arrays(len(atoms), bonds, bond_shifts)
        vectors = (
            atoms.positions[bonds[:, 1]] - atoms.positions[bonds[:, 0]]
            + bond_shifts @ atoms.cell
        )
        limits = bond_scale * radii[bonds].sum(axis=1)
        distances = np.linalg.norm(vectors, axis=1)
        keep = (distances > 1e-8) & (distances <= limits)
        bonds, bond_shifts = bonds[keep], bond_shifts[keep]
    components = periodic_components(len(atoms), bonds, bond_shifts)
    failures = [c for c in components if c["rank"] != 3]
    if failures:
        sizes = ", ".join(f"{c['rank']}D ({c['num_atoms']} atoms)" for c in failures[:8])
        raise ValueError(
            f"3D connectivity required; found components: {sizes}. "
            "Layers, chains, and isolated fragments are rejected."
        )
    return {
        **quality,
        "connectivity_policy_version": CONNECTIVITY_POLICY_VERSION,
        "connectivity_bond_scale": float(bond_scale),
        "periodic_component_ranks": [c["rank"] for c in components],
        "periodic_component_sizes": [c["num_atoms"] for c in components],
    }
