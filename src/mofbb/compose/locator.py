'''
Align building blocks onto topology slots by rotation and matching.

https://github.com/Sangwon91/PORMAKE/blob/main/src/pormake/locator.py
'''

import itertools
import warnings

import numpy as np
import scipy



class PortAssignmentError(ValueError):
    '''
    Raised when no feasible connection-point permutation exists.
    '''


def _target_edge_frame(edge_axis, transverse_direction):
    '''
    Build a right-handed frame for an edge.

    Args:
        edge_axis (array-like):
            Three-component vector along the edge.
        transverse_direction (array-like):
            Direction used to fix rotation around the edge axis.

    Returns:
        A (3, 3) array with unit x, y, and z axes as rows. The z axis follows the edge.
    '''

    z = np.asarray(edge_axis, dtype=float).copy()
    z_length = np.linalg.norm(z)
    if z_length < 1e-12: # Edge must has transversial axis
        raise ValueError("Target edge endpoints must be distinct.")
    z /= z_length

    x = np.asarray(transverse_direction, dtype=float).copy()
    x -= np.dot(x, z) * z
    x_length = np.linalg.norm(x)
    if x_length < 1e-12:
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
    '''
    Match source points to target points with the smallest total distance.

    Args:
        p (numpy.ndarray):
            An (n, 3) array of target points.
        q (numpy.ndarray):
            An (n, 3) array of source points.
        allowed (numpy.ndarray or None):
            An (n, n) mask where True allows a target-to-source port match.

    Returns:
        An (n,) array of source indices in target order. Raises PortAssignmentError if
        no allowed assignment exists.
    '''
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
    '''
    Find the rotation that best aligns two sets of vectors.

    Args:
        p (numpy.ndarray):
            An (n, 3) array of target vectors.
        q (numpy.ndarray):
            An (n, 3) array of source vectors in matching order.

    Returns:
        A (3, 3) rotation matrix for q @ rotation and the root mean squared vector
        distance. No centering or translation is applied.
    '''
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
    '''
    Orient a building block to a target slot's connection directions.
    '''

    @staticmethod
    def has_compatible_permutation(allowed):
        '''
        Check whether each target can be matched to a different source port.

        Args:
            allowed (numpy.ndarray):
                A square Boolean mask of allowed target-to-source matches.

        Returns:
            True if a complete allowed assignment exists; otherwise False.
        '''
        allowed = np.asarray(allowed, dtype=bool)
        rows, columns = scipy.optimize.linear_sum_assignment(~allowed)
        return bool(np.all(allowed[rows, columns]))

    def candidate_permutations(
        self,
        target: np.ndarray,
        bb,
        n_angles=4,
        allowed=None,
    ):
        '''
        Yield allowed port mappings for trial orientations.

        All permutations are tried for up to five ports. Larger blocks use sampled
        rotations, and repeated mappings are skipped.

        Args:
            target (numpy.ndarray):
                An (n, 3) array of target connection directions.
            bb (BuildingBlock):
                Building block to orient. The input block is left unchanged.
            n_angles (int):
                Number of sampled angles per Euler axis when there are more than five
                ports.
            allowed (numpy.ndarray or None):
                An (n, n) mask where True allows a target-to-source port match.

        Returns:
            An iterator of (n,) source-index arrays in target order.
        '''
        p_coord = target
        source_directions = bb.connection_directions()
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

        points = source_directions
        if n_points <= 15: # Maxium dummy connection points
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
        target: np.ndarray,
        bb,
        n_angles=4,
        allowed=None,
    ):
        '''
        Rotate a building block to best match the target connection directions.

        Args:
            target (numpy.ndarray):
                An (n, 3) array of target connection directions.
            bb (BuildingBlock):
                Building block to orient. The input block is left unchanged.
            n_angles (int):
                Number of sampled angles per Euler axis when there are more than five
                ports.
            allowed (numpy.ndarray or None):
                An (n, n) mask where True allows a target-to-source port match.

        Returns:
            The rotated copy, its port permutation, and the direction RMSD. Returns
            (None, None, infinity) if no allowed mapping exists.
        '''
        p_coord = target
        source_directions = bb.connection_directions()

        best = (np.inf, None, None)
        for perm in self.candidate_permutations(
            target,
            bb,
            n_angles=n_angles,
            allowed=allowed,
        ):
            rotation, rmsd = find_best_orientation(
                p_coord, source_directions[perm]
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

    def locate_with_permutation(self, target: np.ndarray, bb, perm):
        '''
        Rotate a building block using a fixed port mapping.

        Args:
            target (numpy.ndarray):
                An (n, 3) array of target connection directions.
            bb (BuildingBlock):
                Building block to orient. The input block is left unchanged.
            perm (numpy.ndarray):
                Source port indices in target connection order.

        Returns:
            The rotated copy and direction RMSD. The block centroid stays in place.
        '''
        rotation, rmsd = find_best_orientation(
            target,
            bb.connection_directions()[perm],
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
        '''
        Orient and center a two-port edge using its axis and transverse direction.

        Args:
            target_points (numpy.ndarray):
                Two Cartesian endpoint positions, with shape (2, 3), in angstroms.
            bb (BuildingBlock):
                Building block to orient. The input block is left unchanged.
            perm (numpy.ndarray):
                Source port indices in target connection order.
            target_transverse (numpy.ndarray):
                A three-component direction that fixes rotation about the edge axis.

        Returns:
            The placed copy, direction RMSD, and original edge frame. The edge is
            rotated and translated, without stretching.
        '''

        target_points = np.asarray(target_points, dtype=float)
        target_transverse = np.asarray(target_transverse, dtype=float)
        perm = np.asarray(perm, dtype=int)
        source_frame = bb.edge_frame()

        # perm[target] = source. Inverting it gives target positions
        # ordered by the edge's fixed intrinsic X0-to-X1 direction.
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
