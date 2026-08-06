import numpy as np
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from pymatgen.core import Element


class PBC:
    """
    Utilities for validating connections under periodic boundary conditions.

    Distances are evaluated using ASE's minimum-image convention.
    """

    scale = 1.2
    tolerance = 0.3
    overlap_scale = 1.0

    @staticmethod
    def _pair_distances(
        atoms,
        pairs: list[tuple[int, int]],
    ) -> np.ndarray:
        """Return minimum-image distances for explicit atom pairs."""
        pair_indices = np.asarray(pairs, dtype=int)
        if pair_indices.size == 0:
            return np.empty(0, dtype=float)

        vectors = (
            atoms.positions[pair_indices[:, 1]]
            - atoms.positions[pair_indices[:, 0]]
        )
        _, distances = find_mic(vectors, atoms.cell, atoms.pbc)
        return np.asarray(distances, dtype=float)

    @classmethod
    def calc_atomic_overlaps(
        cls,
        atoms,
        scale: float | None = None,
    ) -> list[tuple[int, int, float, float]]:
        """Return atom pairs closer than their smaller covalent radius.

        Candidate pairs are found with ASE's periodic neighbor list and then
        filtered using the strict overlap criterion
        ``distance < scale * min(radius_i, radius_j)``. For a pair with
        multiple periodic images, only its shortest contact is returned.
        """
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
        """Return whether a proposed bond directly joins carbon to a metal."""
        if symbol_i == "C":
            other = symbol_j
        elif symbol_j == "C":
            other = symbol_i
        else:
            return False

        try:
            return bool(Element(other).is_metal)
        except ValueError as exc:
            raise ValueError(f"Unknown chemical symbol: {other!r}") from exc

    @classmethod
    def calc_x_mid(
        cls,
        atoms,
        x_pairs: list[tuple[int, int]],
        tolerance: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """Return X-X pairs whose minimum-image distance exceeds tolerance."""
        if tolerance is None:
            tolerance = cls.tolerance

        distances = cls._pair_distances(atoms, x_pairs)
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
        """Return the maximum allowed bond distance after merging dummy sites.

        The limit uses scaled ASE covalent radii.
        """
        if scale is None:
            scale = cls.scale

        try:
            radius_i = covalent_radii[atomic_numbers[symbol_i]]
            radius_j = covalent_radii[atomic_numbers[symbol_j]]
        except KeyError as exc:
            raise ValueError(f"Unknown chemical symbol: {exc.args[0]!r}") from exc

        return float(scale * (radius_i + radius_j))

    @classmethod
    def calc_fail_connection(
        cls,
        atoms,
        x_pairs: list[tuple[int, int]],
        x_neighbors: dict[int, int],
        scale: float | None = None,
    ) -> list[tuple[int, int, float, float]]:
        """Return real-atom bonds that are too long after X-site fusion."""
        if scale is None:
            scale = cls.scale

        atom_pairs = []
        for x_i, x_j in x_pairs:
            try:
                atom_i = x_neighbors[x_i]
                atom_j = x_neighbors[x_j]
            except KeyError as exc:
                # No dummy connection sites: the node/edge should have more
                # than one X site. Check the decomposed XYZ file.
                raise ValueError(
                    f"X site {exc.args[0]} does not have a mapped neighbor."
                ) from exc

            atom_pairs.append((atom_i, atom_j))

        pair_indices = np.asarray(atom_pairs, dtype=int).reshape(-1, 2)
        distances = cls._pair_distances(atoms, atom_pairs)
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

    def _find_matched_x_sites(
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
    def _connection_port(
        bb, # building block
        x_index: int,
    ) -> tuple[str, int, float]:
        """Return an X site's anchor symbol, anchor index, and cap length."""
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
        anchor_index = int(
            bb.connection_point_anchor_indices[port_index]
        )

        symbol, cap_length = bb.port_descriptors[port_index]
        return symbol, anchor_index, float(cap_length)
