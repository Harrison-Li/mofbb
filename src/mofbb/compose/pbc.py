'''
Check periodic port gaps, proposed bond lengths, and atomic overlaps.
'''
import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from pymatgen.core import Element

from mofbb.common.connectivity import mic_shifts


class PBC:
    '''
    Utilities for validating connections under periodic boundary conditions.
    '''

    bond_scale = 1.2
    port_tol = 0.3
    overlap_scale = 1.0

    @staticmethod
    def _pair_distances(
        atoms,
        pairs: list[tuple[int, int]],
        shifts=None,
    ) -> np.ndarray:
        '''
        Measure pair distances in selected periodic images.

        Args:
            atoms (ase.Atoms):
                Atomic positions in angstroms, cell, and periodic boundary flags.
            pairs (list[tuple[int, int]]):
                Local atom-index pairs to measure.
            shifts (array-like or None):
                An (n, 3) array of integer cell shifts added to the second site. If
                omitted, minimum images are used.

        Returns:
            An (n,) array of distances in angstroms, or an empty array for no pairs.
        '''
        pair_indices = np.asarray(pairs, dtype=int)
        if pair_indices.size == 0:
            return np.empty(0, dtype=float)

        vectors = (
            atoms.positions[pair_indices[:, 1]]
            - atoms.positions[pair_indices[:, 0]]
        )
        if shifts is not None:
            return np.linalg.norm(vectors + np.asarray(shifts) @ atoms.cell, axis=1)
        _, distances = find_mic(vectors, atoms.cell, atoms.pbc)
        return distances

    @classmethod
    def calc_atomic_overlaps(
        cls,
        atoms,
        scale: float | None = None,
    ) -> list[tuple[int, int, float, float]]:
        '''
        Find atom pairs that are too close, including periodic images.

        Args:
            atoms (ase.Atoms):
                Atomic positions in angstroms, cell, and periodic boundary flags.
            scale (float or None):
                Multiplier for the smaller covalent radius. Zero disables the check;
                None uses the class default.

        Returns:
            A sorted list of (atom_i, atom_j, distance, limit) tuples in angstroms. Only
            the shortest contact for each atom pair is kept.
        '''
        if scale is None:
            scale = cls.overlap_scale
        if not len(atoms) or scale == 0:
            return []

        radii = covalent_radii[atoms.numbers]

        # ASE adds the two per-atom cutoffs. Half-radii therefore produce
        # (r_i + r_j) / 2, which is always a safe candidate cutoff for
        # min(r_i, r_j).
        indices_i, indices_j, distances = neighbor_list(
            "ijd",
            atoms,
            cutoff=0.5 * scale * radii,
            self_interaction=False,
        )

        shortest_contacts = {}
        for atom_i, atom_j, distance in zip(indices_i, indices_j, distances):
            limit = scale * min(radii[atom_i], radii[atom_j])
            if distance >= limit:
                continue

            pair = tuple(sorted((atom_i, atom_j)))
            previous = shortest_contacts.get(pair)
            if previous is None or distance < previous[2]:
                shortest_contacts[pair] = (*pair, distance, limit)

        return [shortest_contacts[pair] for pair in sorted(shortest_contacts)]

    @staticmethod
    def is_carbon_metal_pair(symbol_i: str, symbol_j: str) -> bool:
        '''
        Check whether a pair joins carbon to boron or a metal.

        Args:
            symbol_i (str):
                Element symbol of the first atom.
            symbol_j (str):
                Element symbol of the second atom.

        Returns:
            True for a carbon-boron or carbon-metal pair, in either order.
        '''
        if symbol_i == "C":
            other = symbol_j
        elif symbol_j == "C":
            other = symbol_i
        else:
            return False

        return other == "B" or Element(other).is_metal

    @classmethod
    def is_forbidden_fused_pair(cls, symbol_i: str, symbol_j: str) -> bool:
        '''
        Check whether joining two blocks would create a forbidden element pair.

        Args:
            symbol_i (str):
                Element symbol of the first atom.
            symbol_j (str):
                Element symbol of the second atom.

        Returns:
            True for carbon-boron, carbon-metal, or metal-metal pairs.
        '''
        return cls.is_carbon_metal_pair(symbol_i, symbol_j) or (
            Element(symbol_i).is_metal and Element(symbol_j).is_metal
        )

    @classmethod
    def calc_x_gaps(
        cls,
        atoms,
        x_pairs: list[tuple[int, int]],
        port_tol: float | None = None,
        *,
        shifts=None,
    ) -> list[tuple[int, int, float]]:
        '''
        Find matched dummy sites whose separation exceeds the allowed gap.

        Args:
            atoms (ase.Atoms):
                Atomic positions in angstroms, cell, and periodic boundary flags.
            x_pairs (list[tuple[int, int]]):
                Pairs of dummy X-site atom indices to join.
            port_tol (float or None):
                Maximum X-site separation in angstroms. None uses the class default.
            shifts (array-like or None):
                An (n, 3) array of integer cell shifts added to the second site. If
                omitted, minimum images are used.

        Returns:
            A list of (x_i, x_j, distance) tuples for gaps larger than the tolerance.
        '''
        if port_tol is None:
            port_tol = cls.port_tol

        distances = cls._pair_distances(atoms, x_pairs, shifts)
        return [
            (x_i, x_j, distance)
            for (x_i, x_j), distance in zip(x_pairs, distances)
            if distance > port_tol
        ]

    @classmethod
    def max_bond_len(
        cls,
        symbol_i: str,
        symbol_j: str,
        bond_scale: float | None = None,
    ) -> float:
        '''
        Calculate the distance limit for a bond formed by joining two blocks.

        Args:
            symbol_i (str):
                Element symbol of the first atom.
            symbol_j (str):
                Element symbol of the second atom.
            bond_scale (float or None):
                Multiplier for the sum of covalent radii. Uses the class default when
                omitted.

        Returns:
            The scaled sum of the two covalent radii, in angstroms.
        '''
        if bond_scale is None:
            bond_scale = cls.bond_scale

        radius_i = covalent_radii[atomic_numbers[symbol_i]]
        radius_j = covalent_radii[atomic_numbers[symbol_j]]

        return bond_scale * (radius_i + radius_j)

    @classmethod
    def calc_fail_connection(
        cls,
        atoms,
        x_pairs: list[tuple[int, int]],
        x_neighbors: dict[int, int | list[int]],
        bond_scale: float | None = None,
        *,
        shifts=None,
    ) -> list[tuple[int, int, float, float]]:
        '''
        Find real-atom connections that would be too long after joining X sites.

        Args:
            atoms (ase.Atoms):
                Atomic positions in angstroms, cell, and periodic boundary flags.
            x_pairs (list[tuple[int, int]]):
                Pairs of dummy X-site atom indices to join.
            x_neighbors (dict):
                Map each X-site atom index to one or more attached real-atom indices.
            bond_scale (float or None):
                Multiplier for the sum of covalent radii. Uses the class default when
                omitted.
            shifts (array-like or None):
                An (n, 3) array of integer cell shifts added to the second site. If
                omitted, minimum images are used.

        Returns:
            A list of (atom_i, atom_j, distance, limit) tuples in angstroms. Multi-atom
            ports are checked pair by pair.
        '''
        if bond_scale is None:
            bond_scale = cls.bond_scale

        atom_pairs = []
        expanded_shifts = []
        if shifts is None:
            shifts = mic_shifts(atoms, x_pairs)
        for (x_i, x_j), shift in zip(x_pairs, shifts, strict=True):
            for atom_i in np.atleast_1d(x_neighbors[x_i]):
                for atom_j in np.atleast_1d(x_neighbors[x_j]):
                    atom_pairs.append((atom_i, atom_j))
                    expanded_shifts.append(shift)
        pair_indices = np.asarray(atom_pairs, dtype=int).reshape(-1, 2)
        shifts = np.asarray(expanded_shifts, dtype=int).reshape(-1, 3)
        distances = cls._pair_distances(atoms, atom_pairs, shifts)
        radii = covalent_radii[atoms.numbers]
        limits = bond_scale * (
            radii[pair_indices[:, 0]] + radii[pair_indices[:, 1]]
        )

        return [
            (atom_i, atom_j, distance, limit)
            for (atom_i, atom_j), distance, limit in zip(
                atom_pairs,
                distances,
                limits,
            )
            if distance > limit
        ]

    def find_matched_x_sites(
        self,
        topology,
        bbs,
        perms,
        edge_index: int,
    ) -> tuple[int, ...]:
        '''
        Find the node ports attached to the two ends of a topology edge.

        Args:
            topology (Topology):
                Topology with periodic node-edge connectivity.
            bbs (sequence):
                Building blocks indexed by topology slot.
            perms (sequence):
                Source port indices in neighbor order for each slot.
            edge_index (int):
                Topology edge slot to inspect.

        Returns:
            A tuple of local X-site atom indices, one for each endpoint node in
            edge-neighbor order.
        '''
        edge_neighbors = topology.neighbor_list[edge_index]
        matched = []

        for edge_neighbor in edge_neighbors:
            node_index = edge_neighbor.index
            node_bb = bbs[node_index]

            for offset, node_neighbor in enumerate(
                topology.neighbor_list[node_index]
            ):
                if node_neighbor.index == edge_index and np.linalg.norm(
                    node_neighbor.distance_vector + edge_neighbor.distance_vector
                ) < 0.01:
                    perm = perms[node_index]
                    matched.append(node_bb.port_idx[perm][offset])
                    break
            else:
                raise ValueError(
                    f"Could not match topology edge {edge_index} to node "
                    f"{node_index}'s connection points."
                )

        return tuple(matched)

    @staticmethod
    def port_info(
        bb,  # building block
        x_index: int,
    ) -> tuple[str, int | tuple[int, ...], float]:
        '''
        Look up the real atoms and cap length associated with an X site.

        Args:
            bb (BuildingBlock):
                Block containing the dummy site.
            x_index (int):
                Local atom index of the dummy X site.

        Returns:
            The connection element, a real-atom index or tuple of indices, and the port
            cap length in angstroms.
        '''
        port_index = np.flatnonzero(bb.port_idx == x_index)[0]
        group = bb.conn_groups[port_index]
        conn_idx = group[0] if len(group) == 1 else group

        symbol, cap_length = bb.port_descriptors[port_index]
        return symbol, conn_idx, cap_length
