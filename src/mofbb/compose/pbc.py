import numpy as np
from ase.data import atomic_numbers, covalent_radii


class PBC:
    """
    Utilities for validating connections under periodic boundary conditions.

    Distances are evaluated using ASE's minimum-image convention through
    ``atoms.get_distance(..., mic=True)``.
    """

    scale = 1.2
    tolerance = 0.3

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

        failures = []

        for x_i, x_j in x_pairs:
            distance = float(atoms.get_distance(x_i, x_j, mic=True))

            if distance > tolerance:
                failures.append((x_i, x_j, distance))

        return failures

    @classmethod
    def max_bond_len(
        cls,
        symbol_i: str,
        symbol_j: str,
        scale: float | None = None,
    ) -> float:
        """Return the maximum allowed bond distance after merging the dummy x sites from scaled ASE covalent radii."""
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

        failures = []

        for x_i, x_j in x_pairs:
            try:
                atom_i = x_neighbors[x_i]
                atom_j = x_neighbors[x_j]
            except KeyError as exc:
                # No dummy connection sites, the node/edge should has more than 1 X sites, check the decomposed xyz file.
                raise ValueError(
                    f"X site {exc.args[0]} does not have a mapped neighbor."
                ) from exc

            distance = float(
                atoms.get_distance(atom_i, atom_j, mic=True)
            )

            limit = cls.max_bond_len(
                atoms[atom_i].symbol,
                atoms[atom_j].symbol,
                scale=scale,
            )

            if distance > limit:
                failures.append((atom_i, atom_j, distance, limit))

        return failures

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
                displacement = (
                    node_neighbor.distance_vector
                    + edge_neighbor.distance_vector
                )

                if np.linalg.norm(displacement) < 0.01:
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

        return (
            bb.atoms[anchor_index].symbol,
            anchor_index,
            float(
                bb.connection_point_anchor_lengths[port_index]
            ),
        )