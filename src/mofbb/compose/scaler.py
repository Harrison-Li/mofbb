"""Cell rescaling that fits a net to its building blocks.
https://github.com/Sangwon91/PORMAKE/blob/main/src/pormake/scaler.py
This module defines :class:`Scaler`, the geometry-optimization stage of
the build pipeline. An abstract net has arbitrary edge lengths, so before
real molecular fragments can be placed the unit cell and slot positions
must be resized to match the actual sizes of the chosen building blocks.
:class:`Scaler` formulates that as a least-squares problem over node-edge
angles and lengths and solves it with SciPy's L-BFGS-B optimizer.
"""
from collections import defaultdict
from itertools import product

import numpy as np
import scipy as sp
import scipy.optimize

from .log import logger
from .symmetry_scaling import SymmetryReducedParameterization


def _wrap_fractional(values):
    """Return a canonical periodic representative without nudging boundaries."""

    wrapped = np.mod(values, 1.0)
    on_boundary = (np.abs(wrapped) < 1.0e-12) | (
        np.abs(wrapped - 1.0) < 1.0e-12
    )
    return np.where(on_boundary, 0.0, wrapped)


def _objective_value_and_gradient(
    scaled_positions,
    cell,
    ij,
    ik,
    ij_image,
    ik_image,
    target_dots,
    weights,
):
    """Return the scaling objective and its exact NumPy gradients."""
    ij_displacements = (
        scaled_positions[ij[:, 1]]
        - scaled_positions[ij[:, 0]]
        + ij_image
    )
    ik_displacements = (
        scaled_positions[ik[:, 1]]
        - scaled_positions[ik[:, 0]]
        + ik_image
    )
    ij_vectors = ij_displacements @ cell
    ik_vectors = ik_displacements @ cell
    errors = np.sum(ij_vectors * ik_vectors, axis=1) - target_dots
    value = float(np.mean(np.square(errors) * weights))

    dot_gradient = 2.0 * weights * errors / len(errors)
    ij_vector_gradient = dot_gradient[:, np.newaxis] * ik_vectors
    ik_vector_gradient = dot_gradient[:, np.newaxis] * ij_vectors
    ij_displacement_gradient = ij_vector_gradient @ cell.T
    ik_displacement_gradient = ik_vector_gradient @ cell.T

    position_gradient = np.zeros_like(scaled_positions)
    np.add.at(position_gradient, ij[:, 1], ij_displacement_gradient)
    np.add.at(position_gradient, ij[:, 0], -ij_displacement_gradient)
    np.add.at(position_gradient, ik[:, 1], ik_displacement_gradient)
    np.add.at(position_gradient, ik[:, 0], -ik_displacement_gradient)
    cell_gradient = (
        ij_displacements.T @ ij_vector_gradient
        + ik_displacements.T @ ik_vector_gradient
    )
    return value, position_gradient, cell_gradient


class Scaler:
    """Resize an abstract net so chosen building blocks fit without strain.

    A :class:`~pormake.topology.Topology` read from a ``.cgd`` file is a
    purely abstract net: its unit cell and node spacing are arbitrary
    and do not reflect the real sizes of the molecular fragments that
    will be placed on it. ``Scaler`` solves that mismatch. Given the
    building blocks assigned to each node and edge, it rescales the
    topology's unit cell and node positions so that node-to-node
    distances and inter-edge angles agree with the actual building-block
    geometry. Producing a net whose dimensions match the fragments is
    what makes the final framework chemically reasonable rather than
    distorted.

    Internally, :meth:`scale` derives a set of target dot products
    between connection-point vectors (encoding both bond lengths and
    angles) from the building blocks, then runs an L-BFGS-B geometry
    optimization over crystallographically independent node displacements
    and point-group-invariant cell strains to match those targets. It
    finishes by rebuilding the topology's neighbor list with the
    rescaled edge geometry.

    Parameters
    ----------
    length_weight : float, optional
        Relative weight given to length (self dot-product) terms versus
        angle (cross dot-product) terms in the optimization objective.
        Values above 1 emphasize matching edge lengths over angles.

    Attributes
    ----------
    length_weight : float
        The stored length-versus-angle weighting used by :meth:`scale`.
    """

    def __init__(self, length_weight=1.0):
        """Store the length-versus-angle weighting for the objective."""
        self.length_weight = length_weight

    def scale(self, topology, bbs, perms, return_result=False):
        """Rescale a topology to match its assigned building blocks.

        Forms length/angle targets from the building blocks placed on
        each node and edge, then optimizes the scaled atomic positions
        and the unit-cell matrix so the net's geometry matches those
        targets. The returned topology is a rescaled copy whose neighbor
        list has been rebuilt with the new edge centers, ready for the
        builder to assemble the final framework. Both edge lengths and
        node-edge-node angles are optimized simultaneously.

        Parameters
        ----------
        topology : pormake.topology.Topology
            The abstract net to rescale. Not modified in place.
        bbs : list of pormake.building_block.BuildingBlock or None
            Building blocks indexed by slot. Node slots must hold a
            building block; edge slots may be ``None`` when no edge
            building block is present.
        perms : list
            Per-node connection-point permutations, indexed by slot,
            describing how each building block's connection points map
            onto the node's edges.
        return_result : bool, optional
            If ``True``, also return the raw SciPy optimization result.

        Returns
        -------
        scaled_topology : pormake.topology.Topology
            A rescaled copy of ``topology`` with updated cell, positions,
            and neighbor list.
        result : scipy.optimize.OptimizeResult
            The optimizer result, returned only when ``return_result``
            is ``True``.
        """
        logger.debug("Scaler.scale starts.")

        # Get pairs of bond indices and images (periodic boundary) and
        pairs = []
        images = []
        c = topology.atoms.cell
        invc = np.linalg.inv(c)
        for e in topology.edge_indices:
            # ni: neigbor with index i.
            ni, nj = topology.neighbor_list[e]

            i = ni.index
            j = nj.index

            # Save index pair.
            pairs.append([i, j])

            # Calculate image.
            # d = d_{ij}
            d = nj.distance_vector - ni.distance_vector

            ri = topology.atoms.positions[i]
            rj = topology.atoms.positions[j]
            s = (d - (rj - ri)) @ invc

            images.append(s)

        # Type casting to np.array.
        pairs = np.array(pairs)
        images = np.around(images)

        # Calculate target norms and vectors for angles.
        # ij_vectors: list of vectors node i to j with building block size.
        ij_vectors = []
        ji_vectors = []
        for e in topology.edge_indices:
            # ni: neigbor with index i.
            ni, nj = topology.neighbor_list[e]

            i = ni.index
            j = nj.index

            # Find connection point index.
            for ci, n in enumerate(topology.neighbor_list[i]):
                zero_sum = np.abs(n.distance_vector + ni.distance_vector)
                if (zero_sum < 1e-3).all():
                    # ci saved.
                    break

            for cj, n in enumerate(topology.neighbor_list[j]):
                zero_sum = np.abs(n.distance_vector + nj.distance_vector)
                if (zero_sum < 1e-3).all():
                    # cj saved.
                    break

            # Get node bb length to the connection point.
            # cp: connection point.
            bb = bbs[i]
            p = perms[i]
            len_i = bb.lengths[p][ci]
            vec_i = bb.connection_points[p][ci] - bb.centroid

            bb = bbs[j]
            p = perms[j]
            len_j = bb.lengths[p][cj]
            vec_j = bb.connection_points[p][cj] - bb.centroid

            edge_length = len_i + len_j
            if bbs[e] is not None:
                edge_length += 2 * bbs[e].lengths[0]

            # Rescaling.
            vec_i = vec_i / np.linalg.norm(vec_i) * edge_length
            vec_j = vec_j / np.linalg.norm(vec_j) * edge_length

            ij_vectors.append(vec_i)
            ji_vectors.append(vec_j)

        # Cast to numpy array.
        ij_vectors = np.array(ij_vectors)
        ji_vectors = np.array(ji_vectors)

        # Get angle triples.
        # Triples are used for tensor operations during optimization.

        # New data view of pairs and images for estimation of triples.
        data_view = defaultdict(list)
        for (i, j), image in zip(pairs, images):
            data_view[i].append((j, image))
            data_view[j].append((i, -image))

        # Triples for the calculatation of dots between r_{ij} and r_{ik}.
        ij = []
        ik = []

        ij_image = []
        ik_image = []

        # Weights for objective function.
        weights = []

        for i in topology.node_indices:
            neigbors = data_view[i]
            for (j, j_image), (k, k_image) in product(neigbors, repeat=2):
                ij.append([i, j])
                ik.append([i, k])

                ij_image.append(j_image)
                ik_image.append(k_image)

                if (j == k) and np.allclose(j_image, k_image):
                    # 2 for count collection for dot product of same edges.
                    weights.append(2 * self.length_weight)
                else:
                    weights.append(1.0)

        # Type cast.
        ij = np.array(ij)
        ik = np.array(ik)

        ij_image = np.array(ij_image)
        ik_image = np.array(ik_image)

        weights = np.array(weights)

        # Calculate target angles.
        # Similar method to above loops.
        vectors_view = defaultdict(list)
        for (i, j), v_ij, v_ji in zip(pairs, ij_vectors, ji_vectors):
            vectors_view[i].append(v_ij)
            vectors_view[j].append(v_ji)

        # Now, i represents node i (center). j and k are represent indices of
        # connection points from i.
        target_dots = []
        target_ij_vec = []
        target_ik_vec = []
        for i in topology.node_indices:
            # Get all connection point vectors of node i.
            vectors = vectors_view[i]
            # Product includes self dot product (vj == vk).
            for vj, vk in product(vectors, repeat=2):
                target_ij_vec.append(vj)
                target_ik_vec.append(vk)

        target_ij_vec = np.array(target_ij_vec)
        target_ik_vec = np.array(target_ik_vec)

        target_dots = np.sum(target_ij_vec * target_ik_vec, axis=-1)

        # Get max / min ratio of edge length.
        lengths = np.sqrt(target_dots[weights > 1.1])
        # for l in lengths:
        #    logger.info("Length: %.3f", l)
        max_len = np.max(lengths)
        min_len = np.min(lengths)
        ratio = max_len / min_len
        logger.debug("Max min ratio of edge length: %.3f", ratio)

        # Normalize target dots. This enhances the optimization convegences.
        max_dot = np.mean(np.abs(target_dots))
        target_dots /= max_dot

        node_indices = np.asarray(topology.node_indices, dtype=int)
        slot_to_node = np.full(topology.n_slots, -1, dtype=int)
        slot_to_node[node_indices] = np.arange(len(node_indices))
        ij = slot_to_node[ij]
        ik = slot_to_node[ik]

        parameterization = SymmetryReducedParameterization(topology)

        # Edge centers are reconstructed below. The objective still evaluates
        # full node positions and a full cell, while the optimizer sees only
        # independent orbit amplitudes and invariant cell strains.
        def fun_and_jac(x):
            node_parameters, cell_parameters = parameterization.split(x)
            s = parameterization.positions(node_parameters)
            c, cell_derivatives = (
                parameterization.cell.cell_and_derivatives(cell_parameters)
            )
            value, position_gradient, cell_gradient = (
                _objective_value_and_gradient(
                    s,
                    c,
                    ij,
                    ik,
                    ij_image,
                    ik_image,
                    target_dots,
                    weights,
                )
            )
            gradient = np.concatenate(
                [
                    parameterization.pos_grad(position_gradient),
                    parameterization.cell.grad(
                        cell_gradient,
                        cell_derivatives,
                    ),
                ]
            )
            return value, gradient

        x0 = np.zeros(parameterization.n_parameters, dtype=float)
        bounds = list(parameterization.position_bounds)
        bounds.extend([(None, None)] * parameterization.n_cell_parameters)

        logger.info(
            "Topology optimization starts with %d node and %d cell variables.",
            parameterization.n_node_parameters,
            parameterization.n_cell_parameters,
        )
        # Perform optimization.
        result = sp.optimize.minimize(
            x0=x0,
            fun=fun_and_jac,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "disp": False},
        )

        node_parameters, cell_parameters = parameterization.split(result.x)
        c, _ = parameterization.cell.cell_and_derivatives(cell_parameters)
        s = topology.atoms.get_scaled_positions(wrap=False)
        s[node_indices] = parameterization.positions(node_parameters)

        logger.info("MESSAGE: %s", result.message)
        logger.info("SUCCESS: %s", result.success)
        logger.info("ITER: %s", result.nit)
        logger.info("OBJ: %.3f", result.fun)

        # Update neigbors list in topology.
        new_data = [[] for _ in range(topology.n_slots)]
        # Rescaling cell to original scale.
        c *= np.sqrt(max_dot)
        # Transform to Cartesian coordinates.
        r = s @ c
        invc = np.linalg.inv(c)
        inv_old_c = np.linalg.inv(topology.atoms.cell)
        for e in topology.edge_indices:
            ni, nj = topology.neighbor_list[e]

            i = ni.index
            j = nj.index

            ri = topology.atoms.positions[i]
            rj = topology.atoms.positions[j]

            d = nj.distance_vector - ni.distance_vector

            image = (d - (rj - ri)) @ inv_old_c

            # Calculate new edge center.
            ri = r[i]
            rj = r[j]

            d = rj - ri + np.dot(image, c)

            # Select center position wrapped by unit cell.
            rc = ri + 0.5 * d
            sc = np.dot(rc, invc)
            sc = _wrap_fractional(sc)
            rc = np.dot(sc, c)
            r[e] = rc

            # Save in proper order.
            new_data[e] += [(i, -0.5 * d), (j, 0.5 * d)]

        # Should change this stupidly nested loop.
        # The new neigbor list is updated with same order of original neigbor
        # list. Then we can use the permutation information for new location
        # after topology scaling.
        for i in topology.node_indices:
            # Same order loop. Note that topology is the original.
            for n in topology.neighbor_list[i]:
                e = n.index
                d = n.distance_vector
                # Find cross reference.
                for j, en in enumerate(topology.neighbor_list[e]):
                    if np.linalg.norm(d + en.distance_vector) < 1e-4:
                        # j and en saved.
                        break

                _, v = new_data[e][j]
                new_data[i].append((e, -v))

        # Make scaled topology.
        s = np.dot(r, invc)
        s = _wrap_fractional(s)
        r = np.dot(s, c)

        scaled_topology = topology.copy()
        scaled_topology.atoms.set_positions(r)
        scaled_topology.atoms.set_cell(c)
        scaled_topology.neighbor_list.set_data(new_data)

        if return_result:
            return scaled_topology, result
        else:
            return scaled_topology
