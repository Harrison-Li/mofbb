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
    """Find the assignment of ``q`` points onto ``p`` points.

    Solves the linear assignment problem (Hungarian algorithm) on the
    pairwise distance matrix so that each point in ``q`` is matched to a
    distinct point in ``p`` with minimal total distance. This decides
    which connection point of a building block serves which neighbor
    direction of the target slot before a rotation is fitted.

    Parameters
    ----------
    p : numpy.ndarray
        Target points, shape ``(n, 3)`` (e.g. slot neighbor directions).
    q : numpy.ndarray
        Points to be matched against ``p``, shape ``(n, 3)`` (e.g. the
        building block's connection-point directions).
    allowed : array_like of bool, optional
        Feasibility mask with shape ``(n, n)``. Forbidden target/source pairs
        are excluded from the assignment solver.

    Returns
    -------
    numpy.ndarray
        Permutation index array of length ``n`` such that ``q[perm]`` is
        aligned, element-wise, with ``p``.
    """
    dist = np.linalg.norm(p[:, None] - q[None, :], axis=-1)
    if allowed is not None:
        dist = np.where(np.asarray(allowed, dtype=bool), dist, np.inf)
    try:
        rows, perm = scipy.optimize.linear_sum_assignment(dist)
        if not np.all(np.isfinite(dist[rows, perm])):
            raise ValueError
    except ValueError as error:
        raise PortAssignmentError(
            "No chemically compatible connection-point permutation exists."
        ) from error
    return perm


def find_best_orientation(p, q):
    """Find the rotation that best overlays ``q`` onto ``p``.

    Wraps :func:`scipy.spatial.transform.Rotation.align_vectors`
    (Kabsch algorithm) to obtain the rigid rotation minimizing the
    squared distance between the already-paired point sets. The returned
    error is converted from the root-"sum"-squared distance that SciPy
    reports into a per-point root-mean-square deviation (RMSD), the
    quantity the rest of the pipeline uses to compare fits.

    Parameters
    ----------
    p : numpy.ndarray
        Target points, shape ``(n, 3)``; assumed already permuted to
        correspond element-wise with ``q``.
    q : numpy.ndarray
        Points to be rotated onto ``p``, shape ``(n, 3)``.

    Returns
    -------
    U : numpy.ndarray
        ``(3, 3)`` rotation matrix. It is transposed so that it can be
        applied as ``positions @ U`` (right-multiplication of row
        vectors) by the callers.
    rmsd : float
        Per-point RMSD of the aligned configuration.
    """
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
    """Orient a building block to a target slot's local structure.

    A topology slot only fixes the *directions* toward its neighbors;
    the building block must be rotated so that its connection points
    point along those directions. Because the correct point-to-direction
    pairing is not known in advance, this class tests every permutation
    for common two- to five-connected slots. Larger slots use the bounded
    Euler-angle search to avoid factorial growth. The matched building
    block, its winning permutation, and RMSD are returned to the builder.
    """

    def locate(
        self,
        target,
        bb,
        max_n_slices=4,
        allowed=None,
    ):
        """Orient ``bb`` onto ``target`` and find the best permutation.

        Exhaustively evaluates feasible permutations for up to five
        connection points. For larger slots it scans a bounded Euler grid,
        solves the assignment at each trial, and keeps the lowest-RMSD fit.
        Translation onto the final cell position happens later.

        Parameters
        ----------
        target : LocalStructure
            The slot's local structure (neighbor direction vectors) the
            building block should be aligned to.
        bb : BuildingBlock
            The building block to orient. It is copied before rotation,
            so the input is left unchanged.
        max_n_slices : int, optional
            Euler-angle grid resolution for slots with more than five
            connection points. Larger values trade speed for accuracy.
        allowed : array_like of bool, optional
            Feasibility mask applied to every connection-point assignment.
        Returns
        -------
        bb : BuildingBlock
            A rotated copy of the input building block.
        min_perm : numpy.ndarray
            The connection-point permutation that achieved the best fit.
        rmsd_val : float
            Per-point RMSD of the returned orientation.
        """
        p_coord = target.atoms.positions
        source_atoms = bb.local_structure().atoms
        n_points = p_coord.shape[0]
        allowed = None if allowed is None else np.asarray(allowed, dtype=bool)

        def permutations():
            if n_points <= 5:
                for values in itertools.permutations(range(n_points)):
                    permutation = np.asarray(values)
                    if allowed is None or np.all(
                        allowed[np.arange(n_points), permutation]
                    ):
                        yield permutation
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
            n_slices = max_n_slices
            angles = (
                np.linspace(0, 360, n_slices),
                np.linspace(0, 180, n_slices),
                np.linspace(0, 360, n_slices),
            )
            eulers = np.asarray(list(itertools.product(*angles)))
            matrices = scipy.spatial.transform.Rotation.from_euler(
                "zxz", -eulers, degrees=True
            ).as_matrix()
            rotated = np.einsum("nij,pj->npi", matrices, points)
            for positions in rotated:
                yield find_best_permutation(p_coord, positions, allowed=allowed)

        best = (np.inf, None, None)
        for permutation in permutations():
            rotation, rmsd = find_best_orientation(
                p_coord, source_atoms.positions[permutation]
            )
            if rmsd < best[0]:
                best = rmsd, rotation, permutation
            if best[0] < 1e-4:
                break

        if best[1] is None:
            raise PortAssignmentError("No compatible connection-point permutation exists.")
        rmsd, rotation, permutation = best
        located = bb.copy()
        located.atoms.set_positions(
            (bb.atoms.positions - bb.centroid) @ rotation + bb.centroid
        )
        return located, permutation, rmsd

    def locate_with_permutation(self, target, bb, permutation):
        """Orient ``bb`` onto ``target`` using a known permutation.

        Skips the expensive Euler-angle search of :meth:`locate` by
        reusing a connection-point permutation that was determined
        earlier (e.g. the one returned by :meth:`locate`, or the trivial
        edge permutation). Only the optimal rotation for that fixed
        permutation is fitted. This is used during the relocation step
        after the topology cell has been scaled, where the pairing is
        already settled and only the new geometry needs re-fitting.

        Parameters
        ----------
        target : LocalStructure
            The slot's local structure to align the building block to.
        bb : BuildingBlock
            The building block to orient; copied before rotation.
        permutation : array_like
            Connection-point permutation mapping ``bb`` points onto the
            target directions.

        Returns
        -------
        bb : BuildingBlock
            A rotated copy of the input building block.
        rmsd_val : float
            Per-point RMSD of the returned orientation.
        """
        rotation, rmsd = find_best_orientation(
            target.atoms.positions,
            bb.local_structure().atoms.positions[permutation],
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
        permutation,
        target_transverse,
    ):
        """Place an edge linker using its complete PCA frame.

        Unlike the two-point Kabsch fit in :meth:`locate_with_permutation`,
        this method fixes rotation about the edge axis. The linker supplies
        its intrinsic transverse direction through
        :meth:`BuildingBlock.edge_frame`; the caller supplies the matching
        target direction.

        For symmetry-aware placement, ``target_transverse`` must be defined
        for one representative edge and transformed with the crystallographic
        operation for the other symmetry-equivalent edges. It must not be
        independently replaced by a global Cartesian "up" vector at every edge.

        Parameters
        ----------
        target_points : array_like of shape (2, 3)
            Ordered Cartesian endpoint positions of the target edge.
        bb : BuildingBlock
            Two-connected linker to place; the input is not modified.
        permutation : array_like of shape (2,)
            Source connection-point index assigned to each target endpoint.
        target_transverse : array_like of shape (3,)
            Direction corresponding to the positive intrinsic PCA axis.

        Returns
        -------
        located_bb : BuildingBlock
            Rigidly transformed copy centered on the target edge.
        rmsd : float
            RMSD between normalized source and target endpoint directions.
        source_frame : EdgeFrame
            PCA frame and its degeneracy diagnostics.
        """

        target_points = np.asarray(target_points, dtype=float)
        target_transverse = np.asarray(target_transverse, dtype=float)
        permutation = np.asarray(permutation, dtype=int)
        source_frame = bb.edge_frame()

        # permutation[target] = source. Inverting it gives target positions
        # ordered by the linker's fixed intrinsic X0-to-X1 direction.
        target_by_source = target_points[np.argsort(permutation)]
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

        source_directions = bb.connection_points - bb.centroid
        source_directions /= np.linalg.norm(
            source_directions, axis=1, keepdims=True
        )
        target_directions = target_points - target_center
        target_directions /= np.linalg.norm(
            target_directions, axis=1, keepdims=True
        )
        residual = source_directions[permutation] @ rotation - target_directions
        rmsd = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

        return located_bb, rmsd, source_frame
