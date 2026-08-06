"""Rigid-pose fitting inside a fixed framework cell."""

from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.spatial.transform import Rotation


class PoseFit:
    """Fit symmetry-linked rotations and translations for occupied slots."""

    def __init__(self, cell):
        self.cell = cell

    def params(self):
        """Map slots to independent rotation variables and symmetry frames."""
        job = self.cell
        ids = {}
        frames = {}
        count = 0
        groups = (
            (job.symmetry.node_orbits, job.covariant_node_orbits),
            (job.symmetry.edge_orbits, job.covariant_edge_orbits),
        )
        for orbits, covariant in groups:
            for orbit in orbits:
                members = [int(i) for i in orbit.members if int(i) in job.slot_offsets]
                if not members:
                    continue
                if orbit.representative not in covariant:
                    for member in members:
                        ids[member] = count
                        frames[member] = np.eye(3)
                        count += 1
                    continue

                var = count
                count += 1
                for member in members:
                    ids[member] = var
                    frames[member] = (
                        np.eye(3)
                        if member == orbit.representative
                        else job.symmetry.transform(
                            job.symmetry.operation(orbit, member),
                            job.original_cell,
                        )
                    )
        return ids, frames, count

    def moves(self):
        """Map movable slots to independent translation variables."""
        job = self.cell
        ids = {}
        count = 0
        expanded = "symmetry_operations" in job.topology.atoms.info
        anchor = job.node_indices[0] if expanded else None
        groups = (
            (
                job.symmetry.node_orbits if expanded else (),
                job.covariant_node_orbits,
            ),
            (job.symmetry.edge_orbits, job.covariant_edge_orbits),
        )
        for orbits, covariant in groups:
            for orbit in orbits:
                members = [int(i) for i in orbit.members if int(i) in job.slot_offsets]
                if not members:
                    continue
                if orbit.representative in covariant:
                    if anchor in members:
                        continue
                    ids.update({member: count for member in members})
                    count += 1
                    continue
                for member in members:
                    if member == anchor:
                        continue
                    ids[member] = count
                    count += 1
        return ids, count

    def fit(self, perms, centers, blocks):
        """Fit occupied block poses and return residual connection gaps."""
        job = self.cell
        atom_pos = [
            np.round(blocks[slot].atoms.positions - blocks[slot].centroid, 10)
            for slot in job.slot_indices
        ]
        points = [
            atom_pos[i][blocks[slot].connection_point_indices]
            for i, slot in enumerate(job.slot_indices)
        ]
        rot_ids, frames, n_rots = self.params()
        move_ids, n_moves = self.moves()
        rot_size = 3 * n_rots
        size = rot_size + 3 * n_moves

        def make_rots(vals):
            raw = vals[:rot_size].reshape(-1, 3)
            return Rotation.from_rotvec(
                [
                    round(np.linalg.det(frames[slot]))
                    * frames[slot].T
                    @ raw[rot_ids[slot]]
                    for slot in job.slot_indices
                ]
            )

        def residual(vals):
            rots = make_rots(vals)
            moves = np.zeros((len(job.slot_indices), 3))
            raw = vals[rot_size:].reshape(-1, 3)
            for slot, var in move_ids.items():
                moves[job.slot_offsets[slot]] = frames[slot].T @ raw[var]
            return np.asarray(
                [
                    centers[job.slot_offsets[right]]
                    + moves[job.slot_offsets[right]]
                    + image
                    + rots[job.slot_offsets[right]].apply(
                        points[job.slot_offsets[right]][perms[right][x_j]]
                    )
                    - centers[job.slot_offsets[left]]
                    - moves[job.slot_offsets[left]]
                    - rots[job.slot_offsets[left]].apply(
                        points[job.slot_offsets[left]][perms[left][x_i]]
                    )
                    for left, x_i, right, x_j, image in job.connections
                ]
            ).ravel()

        result = scipy.optimize.least_squares(
            residual,
            np.zeros(size),
            max_nfev=50,
        )
        rots = make_rots(result.x)
        for i, slot in enumerate(job.slot_indices):
            atom_pos[i] = rots[i].apply(atom_pos[i])
            points[i] = atom_pos[i][blocks[slot].connection_point_indices]
        raw = result.x[rot_size:].reshape(-1, 3)
        for slot, var in move_ids.items():
            centers[job.slot_offsets[slot]] += frames[slot].T @ raw[var]
        gaps = np.linalg.norm(
            residual(np.zeros(size)).reshape(-1, 3),
            axis=1,
        )
        return result, gaps, atom_pos, points
