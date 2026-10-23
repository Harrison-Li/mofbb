"""Align building blocks onto topology slots by rotation and matching.

https://github.com/Sangwon91/PORMAKE/blob/main/src/pormake/locator.py
"""

import itertools
import warnings

import numpy as np
import scipy


_EDGE_FRAME_EPSILON = 1e-12
_MAX_CANONICAL_PORTS = 15


class PortAssignmentError(ValueError):
    """Raised when no feasible connection-point permutation exists."""


def _target_edge_frame(edge_axis, transverse_direction):
    """Build a right-handed target frame from an edge and transverse axis."""

    z = np.asarray(edge_axis, dtype=float).copy()
    z_length = np.linalg.norm(z)
    if z_length < _EDGE_FRAME_EPSILON:
        raise ValueError("Target edge endpoints must be distinct.")
    z /= z_length

    x = np.asarray(transverse_direction, dtype=float).copy()
    x -= np.dot(x, z) * z
    x_length = np.linalg.norm(x)
    if x_length < _EDGE_FRAME_EPSILON:
        raise ValueError(
            "Target transverse direction must not be parallel to the edge."
        )
    x /= x_length

    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    return np.stack([x, y, z])


def find_best_permutation(p, q, allowed=None):
    """Find the assignment of ``q`` points onto ``p`` points."""
    dist = np.linalg.norm(p[:, None] - q[None, :], axis=-1)
    compatible = None
    if allowed is not None:
        compatible = np.asarray(allowed, dtype=bool)
        dist = np.where(compatible, dist, np.max(dist) + 1.0)
    rows, perm = scipy.optimize.linear_sum_assignment(dist)
    if compatible is not None and not np.all(compatible[rows, perm]):
        raise PortAssignmentError(
            "No chemically compatible connection-point permutation exists."
        )
    return perm


def find_best_orientation(p, q):
    """Find the rotation that best overlays ``q`` onto ``p``."""
    # This function gives root "sum" squared distance...
    with warnings.catch_warnings():
        if len(p) == 2:
            warnings.filterwarnings(
                "ignore",
                message="Optimal rotation is not uniquely or poorly defined",
                category=UserWarning,
            )
        U, rmsd = scipy.spatial.transform.Rotation.align_vectors(p, q)
    return U.as_matrix().T, np.sqrt(np.square(rmsd) / len(p))


class Locator:
    """Orient a building block to a target slot's local structure."""

    @staticmethod
    def has_compatible_permutation(allowed):
        """Return whether a port mask admits a complete assignment."""
        allowed = np.asarray(allowed, dtype=bool)
        rows, columns = scipy.optimize.linear_sum_assignment(~allowed)
        return bool(np.all(allowed[rows, columns]))

    def candidate_permutations(
        self,
        target,
        bb,
        n_angles=4,
        allowed=None,
    ):
        """Yield distinct port mappings, sampling ``n_angles`` per Euler axis.

        For at most five ports, enumerate all permutations instead of sampling.
        """
        p_coord = target.atoms.positions
        source_atoms = bb.local_structure().atoms
        n_points = p_coord.shape[0]
        allowed = None if allowed is None else np.asarray(allowed, dtype=bool)
        if allowed is not None and not self.has_compatible_permutation(allowed):
            return

        if n_points <= 5:
            for values in itertools.permutations(range(n_points)):
                perm = np.asarray(values)
                if allowed is None or np.all(
                    allowed[np.arange(n_points), perm]
                ):
                    yield perm
            return

        points = source_atoms.positions
        if n_points <= _MAX_CANONICAL_PORTS:
            weights = np.arange(1, n_points + 1, dtype=float)
            centered = points - np.average(points, axis=0, weights=weights)
            _, axes = np.linalg.eigh((centered * weights[:, None]).T @ centered)
            pivots = np.argmax(np.abs(centered @ axes), axis=0)
            signs = np.sign((centered @ axes)[pivots, np.arange(3)])
            axes *= np.where(signs == 0, 1, signs)
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            points = points @ axes
        angles = (
            np.linspace(0, 360, n_angles),
            np.linspace(0, 180, n_angles),
            np.linspace(0, 360, n_angles),
        )
        eulers = np.asarray(list(itertools.product(*angles)))
        matrices = scipy.spatial.transform.Rotation.from_euler(
            "zxz", -eulers, degrees=True
        ).as_matrix()
        rotated = np.einsum("nij,pj->npi", matrices, points)
        seen = set()
        for positions in rotated:
            perm = find_best_permutation(
                p_coord,
                positions,
                allowed=allowed,
            )
            key = tuple(perm)
            if key in seen:
                continue
            seen.add(key)
            yield perm

    def locate(
        self,
        target,
        bb,
        n_angles=4,
        allowed=None,
    ):
        """Orient ``bb`` onto ``target`` and find the best permutation."""
        p_coord = target.atoms.positions
        source_atoms = bb.local_structure().atoms

        best = (np.inf, None, None)
        for perm in self.candidate_permutations(
            target,
            bb,
            n_angles=n_angles,
            allowed=allowed,
        ):
            rotation, rmsd = find_best_orientation(
                p_coord, source_atoms.positions[perm]
            )
            if rmsd < best[0]:
                best = rmsd, rotation, perm
            if best[0] < 1e-4:
                break

        if best[1] is None:
            return None, None, np.inf
        rmsd, rotation, perm = best
        located = bb.copy()
        located.atoms.set_positions(
            (bb.atoms.positions - bb.centroid) @ rotation + bb.centroid
        )
        return located, perm, rmsd

    def locate_with_permutation(self, target, bb, perm):
        """Orient ``bb`` onto ``target`` using a known permutation."""
        rotation, rmsd = find_best_orientation(
            target.atoms.positions,
            bb.local_structure().atoms.positions[perm],
        )
        located = bb.copy()
        located.atoms.set_positions(
            (bb.atoms.positions - bb.centroid) @ rotation + bb.centroid
        )
        return located, rmsd

    def locate_edge_with_frame(
        self,
        target_points,
        bb,
        perm,
        target_transverse,
    ):
        """Place an edge linker using its complete PCA frame."""

        target_points = np.asarray(target_points, dtype=float)
        target_transverse = np.asarray(target_transverse, dtype=float)
        perm = np.asarray(perm, dtype=int)
        source_frame = bb.edge_frame()

        # perm[target] = source. Inverting it gives target positions
        # ordered by the linker's fixed intrinsic X0-to-X1 direction.
        target_by_source = target_points[np.argsort(perm)]
        target_frame = _target_edge_frame(
            target_by_source[1] - target_by_source[0],
            target_transverse,
        )

        # Frames store basis vectors as rows, so source @ rotation = target.
        rotation = source_frame.axes.T @ target_frame

        located_bb = bb.copy()
        target_center = np.mean(target_points, axis=0)
        positions = (bb.atoms.positions - bb.centroid) @ rotation
        located_bb.atoms.set_positions(positions + target_center)

        source_directions = bb.port_pos - bb.centroid
        source_directions /= np.linalg.norm(
            source_directions, axis=1, keepdims=True
        )
        target_directions = target_points - target_center
        target_directions /= np.linalg.norm(
            target_directions, axis=1, keepdims=True
        )
        residual = source_directions[perm] @ rotation - target_directions
        rmsd = np.sqrt(np.mean(np.sum(residual**2, axis=1)))

        return located_bb, rmsd, source_frame
