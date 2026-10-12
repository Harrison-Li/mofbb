import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from pymatgen.core import Element

from mofbb.compose.connectivity import minimum_image_shifts


class PBC:
    """Utilities for validating connections under periodic boundary conditions."""

    scale = 1.2
    tolerance = 0.3
    overlap_scale = 1.0

    @staticmethod
    def _pair_distances(
        atoms,
        pairs: list[tuple[int, int]],
        shifts=None,
    ) -> np.ndarray:
        """Return distances using supplied images, or minimum images if omitted."""
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
        return np.asarray(distances, dtype=float)

    @classmethod
    def calc_atomic_overlaps(
        cls,
        atoms,
        scale: float | None = None,
    ) -> list[tuple[int, int, float, float]]:
        """Return atom pairs closer than their smaller covalent radius."""
        if scale is None:
            scale = cls.overlap_scale
        scale = float(scale)
        if scale < 0.0:
            raise ValueError("Atomic-overlap scale cannot be negative.")
        if not len(atoms):
            return []
        if scale == 0.0:
            return []

        radii = np.asarray(covalent_radii[atoms.numbers], dtype=float)

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
        for atom_i, atom_j, distance in zip(
            indices_i,
            indices_j,
            distances,
            strict=True,
        ):
            atom_i = int(atom_i)
            atom_j = int(atom_j)
            distance = float(distance)
            limit = float(scale * min(radii[atom_i], radii[atom_j]))
            if distance >= limit:
                continue

            pair = tuple(sorted((atom_i, atom_j)))
            previous = shortest_contacts.get(pair)
            if previous is None or distance < previous[2]:
                shortest_contacts[pair] = (
                    pair[0],
                    pair[1],
                    distance,
                    limit,
                )

        return [shortest_contacts[pair] for pair in sorted(shortest_contacts)]

    @staticmethod
    def is_carbon_metal_pair(symbol_i: str, symbol_j: str) -> bool:
        """Return whether a proposed bond joins carbon to boron or a metal."""
        if symbol_i == "C":
            other = symbol_j
        elif symbol_j == "C":
            other = symbol_i
        else:
            return False

        return other == "B" or bool(Element(other).is_metal)

    @classmethod
    def is_forbidden_fused_pair(cls, symbol_i: str, symbol_j: str) -> bool:
        """Reject new carbon-metal or metal-metal bonds between blocks."""
        return cls.is_carbon_metal_pair(symbol_i, symbol_j) or (
            Element(symbol_i).is_metal and Element(symbol_j).is_metal
        )

    @classmethod
    def calc_x_mid(
        cls,
        atoms,
        x_pairs: list[tuple[int, int]],
        tolerance: float | None = None,
        *,
        shifts=None,
    ) -> list[tuple[int, int, float]]:
        """Return X-X pairs farther apart than tolerance in the selected images."""
        if tolerance is None:
            tolerance = cls.tolerance

        distances = cls._pair_distances(atoms, x_pairs, shifts)
        return [
            (x_i, x_j, float(distance))
            for (x_i, x_j), distance in zip(
                x_pairs,
                distances,
                strict=True,
            )
            if distance > tolerance
        ]

    @classmethod
    def max_bond_len(
        cls,
        symbol_i: str,
        symbol_j: str,
        scale: float | None = None,
    ) -> float:
        """Return the maximum allowed bond distance after merging dummy sites."""
        if scale is None:
            scale = cls.scale

        radius_i = covalent_radii[atomic_numbers[symbol_i]]
        radius_j = covalent_radii[atomic_numbers[symbol_j]]

        return float(scale * (radius_i + radius_j))

    @classmethod
    def calc_fail_connection(
        cls,
        atoms,
        x_pairs: list[tuple[int, int]],
        x_neighbors: dict[int, int | list[int]],
        scale: float | None = None,
        *,
        shifts=None,
    ) -> list[tuple[int, int, float, float]]:
        """Return real-atom bonds that are too long after X-site fusion."""
        if scale is None:
            scale = cls.scale

        atom_pairs = []
        expanded_shifts = []
        if shifts is None:
            shifts = minimum_image_shifts(atoms, x_pairs)
        for (x_i, x_j), shift in zip(x_pairs, shifts, strict=True):
            for atom_i in np.atleast_1d(x_neighbors[x_i]):
                for atom_j in np.atleast_1d(x_neighbors[x_j]):
                    atom_pairs.append((int(atom_i), int(atom_j)))
                    expanded_shifts.append(shift)
        pair_indices = np.asarray(atom_pairs, dtype=int).reshape(-1, 2)
        shifts = np.asarray(expanded_shifts, dtype=int).reshape(-1, 3)
        distances = cls._pair_distances(atoms, atom_pairs, shifts)
        radii = covalent_radii[atoms.numbers]
        limits = scale * (
            radii[pair_indices[:, 0]] + radii[pair_indices[:, 1]]
        )

        return [
            (atom_i, atom_j, float(distance), float(limit))
            for (atom_i, atom_j), distance, limit in zip(
                atom_pairs,
                distances,
                limits,
                strict=True,
            )
            if distance > limit
        ]

    def find_matched_x_sites(
        self,
        topology,
        bbs,
        permutations,
        edge_index: int,
    ) -> tuple[int, ...]:
        """Return node connection points matched to a topology edge."""
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
                    permutation = permutations[node_index]
                    connection_index = (
                        node_bb.connection_point_indices[permutation][offset]
                    )
                    matched.append(int(connection_index))
                    break
            else:
                raise ValueError(
                    f"Could not match topology edge {edge_index} to node "
                    f"{node_index}'s connection points."
                )

        return tuple(matched)

    @staticmethod
    def connection_port(
        bb,  # building block
        x_index: int,
    ) -> tuple[str, int | tuple[int, ...], float]:
        """Return anchor element, index or paired indices, and maximum cap length."""
        x_indices = np.asarray(
            bb.connection_point_indices
        )

        matches = np.where(x_indices == x_index)[0]

        if len(matches) != 1:
            raise ValueError(
                f"Connection point {x_index} occurs {len(matches)} times in "
                f"{bb.name!r}; expected exactly one occurrence."
            )

        port_index = int(matches[0])
        group = bb.connection_point_anchor_groups[port_index]
        anchor_index = group[0] if len(group) == 1 else group

        symbol, cap_length = bb.port_descriptors[port_index]
        return symbol, anchor_index, float(cap_length)
