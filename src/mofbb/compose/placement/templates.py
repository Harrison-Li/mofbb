"""Compare building-block templates across topology orbits."""

from __future__ import annotations

from functools import cache

import numpy as np

from mofbb.matching import BuildingBlockGraph, compare_building_blocks


class BlockGroups:
    """Answer whether orbit members use the same rigid template."""

    def __init__(self, blocks, symmetry, fixed):
        self.blocks = blocks
        self.symmetry = symmetry
        self.fixed = fixed

    @cache
    def same(self, left, right):
        if left is right:
            return True
        if left is None or right is None:
            return left is right
        if (
            left.atoms.get_chemical_symbols()
            == right.atoms.get_chemical_symbols()
            and np.array_equal(
                left.connection_point_indices,
                right.connection_point_indices,
            )
            and np.array_equal(left.bonds, right.bonds)
            and np.array_equal(left.bond_types, right.bond_types)
            and np.allclose(
                left.atoms.positions - left.centroid,
                right.atoms.positions - right.centroid,
                atol=1.0e-8,
            )
        ):
            return True

        match = compare_building_blocks(
            BuildingBlockGraph.from_building_block(left),
            BuildingBlockGraph.from_building_block(right),
            require_kind=False,
            require_bond_types=True,
        )
        if not match.matched:
            return False

        atom_map = dict(match.mapping)
        order = np.asarray([atom_map[i] for i in range(len(left.atoms))])
        mapped_x = {atom_map[int(i)] for i in left.connection_point_indices}
        if mapped_x != set(right.connection_point_indices) or not np.allclose(
            left.atoms.get_initial_charges(),
            right.atoms.get_initial_charges()[order],
        ):
            return False

        left_pos = left.atoms.positions - left.centroid
        right_pos = right.atoms.positions[order] - right.centroid
        src, _, dst = np.linalg.svd(right_pos.T @ left_pos)
        rot = src @ dst
        if np.linalg.det(rot) < 0.0:
            src[:, -1] *= -1.0
            rot = src @ dst
        err = np.linalg.norm(right_pos @ rot - left_pos, axis=1)
        return bool(np.max(err, initial=0.0) <= 1.0e-6)

    def one(self, orbit):
        """Return whether every orbit member shares one template."""
        rep = self.blocks[orbit.representative]
        return all(self.same(rep, self.blocks[int(i)]) for i in orbit.members)

    def perm(self, orbit):
        """Move the first supplied member permutation to the representative."""
        src = min(
            (int(i) for i in orbit.members if int(i) in self.fixed),
            default=None,
        )
        if src is None:
            return None
        return self.symmetry.to_representative(orbit, src, self.fixed[src])
